import assert from 'node:assert/strict'
import { mkdtemp, mkdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { createWorkspaceScope, contextOutsideWorkspace } from '../src/workspace-scope.js'
import { applyWorkspaceSetup } from '../src/workspace-setup.js'

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'factor-workspace-'))
  t.after(() => rm(root, { recursive: true, force: true }))
  const ordinaryPath = join(root, 'ordinary')
  await mkdir(ordinaryPath)
  const config = { workspaceId: 'research', workspacePath: root, setupSessionId: 'session-planning' }
  const workspace = { path: root, sessionIds: [] }
  const registry = { get: id => id === 'research' ? workspace : undefined }
  return { root, ordinaryPath, config, workspace, registry, scope: createWorkspaceScope(registry, config) }
}

test('scope follows the configured Workspace, including newly registered Sessions', async t => {
  const f = await fixture(t)
  assert.equal(f.scope.ownsCreate({ workspaceId: 'ordinary' }), false)
  assert.equal(f.scope.ownsCreate({ cwd: f.ordinaryPath }), false)
  assert.equal(f.scope.ownsCreate({ cwd: join(f.ordinaryPath, 'new-directory') }), false)
  assert.equal(f.scope.ownsCreate({ workspaceId: 'research' }), true)
  assert.equal(f.scope.ownsCreate({ cwd: f.root }), true)
  assert.equal(f.scope.ownsSession('session-planning'), true)
  assert.equal(f.scope.ownsAgent({ session: { id: 'other', header: { cwd: f.root } } }), true)
  f.workspace.sessionIds.push('new-research-session')
  assert.equal(f.scope.ownsSession('new-research-session'), true)
  assert.throws(() => createWorkspaceScope(f.registry, { ...f.config, workspaceId: 'missing' }), /not registered/)
  assert.throws(() => createWorkspaceScope(f.registry, { ...f.config, workspacePath: f.ordinaryPath }), /differs/)
})

test('native Goal driver retains ordinary events and excludes research roster and events', async t => {
  const { scope } = await fixture(t)
  const ordinary = { session: { id: 'ordinary' } }
  const research = { session: { id: 'session-planning' } }
  const callbacks = new Map()
  const agents = { list: () => [ordinary, research], get: id => id === 'ordinary' ? ordinary : research }
  const ctx = { get: () => agents, on(name, fn) { callbacks.set(name, fn) }, extend(meta) { return Object.assign(Object.create(this), meta) } }
  const filtered = contextOutsideWorkspace(ctx, scope)
  assert.deepEqual(filtered.agents.list(), [ordinary])
  let calls = 0
  filtered.on('goal/changed', () => ++calls)
  callbacks.get('goal/changed')({ agent: research })
  callbacks.get('goal/changed')({ agent: ordinary })
  assert.equal(calls, 1)
  filtered.on('session/event', () => ++calls)
  callbacks.get('session/event')(research.session, {})
  assert.equal(calls, 1)
  callbacks.get('session/event')(ordinary.session, {})
  assert.equal(calls, 2)
  filtered.on('agent/pre-step', () => assert.fail('native Goal driver received research input'))
  assert.equal(callbacks.get('agent/pre-step')({ agent: research }, () => 'original'), 'original')
})

test('new research Workspace admits planning, preserves ordinary operations and denies execution', async t => {
  const f = await fixture(t)
  const ordinaryAgent = { session: { id: 'ordinary', header: { cwd: f.ordinaryPath } } }
  const setupAgent = { session: { id: f.config.setupSessionId, header: { cwd: f.root } } }
  const events = new Map()
  const disposers = []
  const calls = []
  const native = function (...args) { calls.push({ receiver: this, args }); return { native: true } }
  const sessions = { resolveAgent: async () => ({ error: { code: 'session/not-found' } }), create: native, fork: native, rename: native }
  const goals = Object.fromEntries(['create', 'edit', 'pause', 'resume', 'clear', 'complete'].map(name => [name, native]))
  const commands = { execute: native }
  let toolGuard
  const services = { sessionController: sessions, goals, commands, tools: { guard(fn) { toolGuard = fn; return () => {} } } }
  const ctx = {
    get: name => services[name],
    effect(fn) { const dispose = fn(); if (typeof dispose === 'function') disposers.push(dispose) },
    on(name, fn) { events.set(name, fn) },
    logger: { error: message => assert.fail(message) },
  }
  const denied = (_operation, _id, message) => { throw new Error(message) }
  applyWorkspaceSetup(ctx, f.config, f.scope, denied)
  await new Promise(resolve => setImmediate(resolve))
  calls.length = 0
  const request = { workspaceId: 'ordinary', cwd: f.ordinaryPath, agentPreset: 'standard' }
  assert.deepEqual(sessions.create(request), { native: true })
  assert.equal(calls.at(-1).args[0], request)
  assert.equal(calls.at(-1).receiver, sessions)
  const result = await sessions.create({ workspaceId: 'research' })
  assert.deepEqual(result, { native: true })
  assert.deepEqual(calls.at(-1).args[0], { sessionId: 'session-planning', workspaceId: 'research', agentPreset: 'factor-research' })
  assert.throws(() => sessions.create({ workspaceId: 'ordinary', agentPreset: 'factor-research-a' }), /research presets/)
  assert.throws(() => sessions.create({ sessionId: 'session-planning', workspaceId: 'ordinary' }), /registered Workspace/)
  assert.throws(() => sessions.fork({ sessionId: 'session-planning' }), /approved run/)
  assert.deepEqual(sessions.fork({ sessionId: 'ordinary' }), { native: true })
  for (const method of Object.keys(goals)) {
    assert.deepEqual(goals[method](ordinaryAgent, {}), { native: true })
    assert.throws(() => goals[method](setupAgent, {}), /approved research/)
  }
  assert.deepEqual(commands.execute(ordinaryAgent, '/some-command'), { native: true })
  assert.throws(() => commands.execute(setupAgent, '/bash'), /no execution commands/)
  assert.equal(toolGuard({ agent: ordinaryAgent, name: 'bash' }), undefined)
  assert.match(toolGuard({ agent: setupAgent, name: 'bash' }), /no calculation/)
  assert.match(toolGuard({ agent: ordinaryAgent, name: 'mcp__fr_b__read_b_result' }), /bound research Workspace/)
  const ordinaryRequest = { tools: [{ name: 'bash' }] }
  assert.equal(await events.get('agent/request')({ agent: ordinaryAgent }, async () => ordinaryRequest), ordinaryRequest)
  assert.deepEqual(await events.get('agent/request')({ agent: setupAgent }, async () => ({ tools: [] })), { tools: [] })
  await assert.rejects(events.get('agent/request')({ agent: setupAgent }, async () => ordinaryRequest), /execution tools/)
  assert.equal(events.get('agent/request-error')({ agent: ordinaryAgent }, () => 'ordinary-retry'), 'ordinary-retry')
  assert.equal(events.get('agent/request-error')({ agent: setupAgent }, () => assert.fail('research retry')), undefined)
  for (const dispose of disposers.reverse()) dispose()
  assert.equal(sessions.create, native)
  assert.equal(commands.execute, native)
  assert.equal(goals.create, native)
})

test('planning reuses a native blank Session and refuses started or differently located Sessions', async t => {
  const f = await fixture(t)
  const agent = { session: { id: f.config.setupSessionId, header: { cwd: f.root } } }
  let lastTurn = 0
  let cwd = f.root
  let writes = 0
  const errors = []
  const sessions = {
    resolveAgent: async () => { agent.session.header.cwd = cwd; return { agent } },
    async create(request) { writes++; return { sessionId: request.sessionId, agentPreset: request.agentPreset } },
    rename() {}, fork() {},
  }
  const services = {
    sessionController: sessions, goals: Object.fromEntries(['create','edit','pause','resume','clear','complete'].map(name=>[name,()=>{}])),
    commands: { execute() {} }, tools: { guard: () => () => {} },
    sessionProjections: { stateOf: (_session, name) => name === 'agentPreset' ? 'standard' : { lastTurn, openTurnStartSeq: null } },
  }
  const ctx = { get: name=>services[name], effect: fn=>fn(), on() {}, logger: { error: error=>errors.push(error) } }
  applyWorkspaceSetup(ctx, f.config, f.scope, (_operation,_id,message)=>{throw new Error(message)})
  await new Promise(resolve=>setImmediate(resolve))
  assert.equal(errors.length,0)
  assert.equal(writes,1)
  lastTurn=1
  await assert.rejects(sessions.create({workspaceId:'research'}), /must be blank/)
  assert.equal(writes,1)
  lastTurn=0
  cwd=f.ordinaryPath
  await assert.rejects(sessions.create({workspaceId:'research'}), /different Workspace/)
  assert.equal(writes,1)
})
