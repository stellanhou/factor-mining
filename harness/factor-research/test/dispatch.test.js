import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { chmod, mkdtemp, mkdir, readFile, readdir, realpath, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { createCompactionRequestGuard, createGoalEvidenceVerifier, createModelRequestGuard, createTicketDispatcher, openBoundSession } from '../src/dispatch.js'

test('native compaction reserves the same durable total budget before provider dispatch', async t => {
  const budgetRoot = await mkdtemp(join(tmpdir(), 'factor-compaction-budget-'))
  t.after(() => rm(budgetRoot, { recursive: true, force: true }))
  let active = true
  let calls = 0
  const options = { purpose: 'compaction', sessionId: 'session-a',
    provider: 'deepseek-account', model: 'deepseek-flash', maxTokens: 2048 }
  const guard = createCompactionRequestGuard({ budgetRoot, maxTotalRequests: 1,
    maxOutputTokens: 2048, roleForSession: id => id === 'session-a' ? 'A' : undefined,
    requireRunActive: async () => { assert.equal(active, true) } })
  const next = async function* () { calls += 1; yield { type: 'finish' } }
  const run = async value => { for await (const _ of guard(value, next)) {} }
  await assert.rejects(run({ ...options, sessionId: 'other' }), /bound research/)
  await assert.rejects(run({ ...options, maxTokens: 2049 }), /output cap/)
  await assert.rejects(run({ ...options, tools: [{ name: 'mcp__fr_b__get_research_status' }] }), /outside its fixed research role/)
  await assert.rejects(run({ ...options, tools: [{ name: 'bash' }] }), /outside its fixed research role/)
  active = false
  await assert.rejects(run(options))
  assert.equal(calls, 0)
  active = true
  await run({ ...options, tools: [{ name: 'working_directory' }] })
  await assert.rejects(run(options), error => error.code === 'factor-research/run-budget-exhausted')
  assert.equal(calls, 1)
  const reservation = JSON.parse(await readFile(join(budgetRoot,
    '.factor-research-control/model-budget/reservations/000001.json'), 'utf8'))
  assert.equal(reservation.purpose, 'compaction')
  assert.equal(reservation.role, 'A')
})

test('opens a started persisted Session through resume without minting another identity', async () => {
  const request = { sessionId: 'fixed-a', agentPreset: 'factor-research-a', workspaceId: 'fixed-workspace' }
  const options = { request, sessions: { resolveAgent: async () => ({ agent: { session: { id: 'fixed-a' } } }) },
    originalCreate: () => assert.fail('create would mint another ID for a started Session'),
    presetForAgent: () => 'factor-research-a' }
  assert.deepEqual(await openBoundSession(options), { sessionId: 'fixed-a', agentPreset: 'factor-research-a' })
  await assert.rejects(openBoundSession({ ...options, presetForAgent: () => 'factor-research-b' }), /role preset/)
  const missing = { code: 'session/not-found' }
  assert.deepEqual(await openBoundSession({ ...options,
    sessions: { resolveAgent: async () => ({ error: missing }) },
    originalCreate: async actual => { assert.deepEqual(actual, request); return { sessionId: 'fixed-a' } },
  }), { sessionId: 'fixed-a' })
})

function canonicalJson(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
  return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

test('runs the fixed Python evidence checker as a direct child process', async t => {
  const root = await mkdtemp(join(tmpdir(), 'factor-research-evidence-cli-'))
  t.after(() => rm(root, { recursive: true, force: true }))
  const expected = { research_id: 'synthetic-run', evidence: { status: 'incomplete' } }
  const executable = join(root, 'python-shim')
  await writeFile(executable, `#!${process.execPath}\nprocess.stdout.write(${JSON.stringify(JSON.stringify(expected))})\n`)
  await chmod(executable, 0o700)

  const result = createGoalEvidenceVerifier({
    bindingPath: join(root, 'binding.json'),
    pythonExecutable: executable,
    projectRoot: root,
  })()

  assert.deepEqual(result, expected)
})

async function fixture(t, { actionControl } = {}) {
  const runRoot = await mkdtemp(join(tmpdir(), 'factor-research-dispatch-'))
  t.after(() => rm(runRoot, { recursive: true, force: true }))
  const contract = {
    run_id: 'dispatch-fixture',
    objective: 'Deliver a saved A-only engineering task.',
    purpose: 'engineering_check',
    sample_hours: 1,
    b_start: '2025-01-01T00:00:00Z',
    c_start: '2025-01-03T00:00:00Z',
    max_lookback_hours: 0,
    universe_provenance: 'fixed synthetic fixture',
    max_formula_nodes: 40,
  }
  const contractText = `${JSON.stringify(contract, null, 2)}\n`
  await writeFile(join(runRoot, 'contract.json'), contractText)
  const refs = [
    { record_id: 'a-environment', kind: 'a_environment' },
    { record_id: 'goal-spec', kind: 'goal_contract' },
  ]
  await mkdir(join(runRoot, 'a_records'))
  await writeFile(join(runRoot, 'a_records', 'a-environment.json'), JSON.stringify({ id: 'a-environment', kind: 'a_environment', data: {} }))
  const domainGoal = { goal_id: 'domain-goal', objective: contract.objective, target_ideas: 1 }
  await writeFile(join(runRoot, 'a_records', 'goal-spec.json'), JSON.stringify({
    id: 'goal-spec', kind: 'goal_contract', data: domainGoal,
  }))
  await writeFile(join(runRoot, 'binding.json'), JSON.stringify({
    format_version: 1,
    research_id: contract.run_id,
    run_root: '.',
    target: { version: 1, objective: contract.objective },
    goal: domainGoal,
    chats: {
      coordinator: { thread_id: 'session-coordinator', host_id: 'local' },
      A: { thread_id: 'session-a', host_id: 'local' },
      B: { thread_id: 'session-b', host_id: 'local' },
    },
    contract_sha256: sha256(contractText),
    inputs: { include_liquidations: false },
  }))
  const body = {
    format_version: 1,
    research_id: contract.run_id,
    batch_id: contract.run_id,
    task_type: 'initial_a_research',
    target_version: 1,
    contract_version: 'FM-v6',
    contract_sha256: sha256(contractText),
    target: { version: 1, objective: contract.objective },
    goal: domainGoal,
    contract,
    thin_spec: {
      data_environment: { segment: 'A' },
      goal_counts: {
        b_qualified_deliverables: null,
        b_count_availability: 'not available to A before frozen-batch validation',
      },
    },
    hash_probe: 1.0,
    a_snapshot: {
      run_id: contract.run_id,
      a_status: 'open',
      research_contract: contract,
      a_evidence_refs: refs,
    },
  }
  const bodyHashInput = canonicalJson(body).replace('"hash_probe":1', '"hash_probe":1.0')
  const taskId = sha256(bodyHashInput)
  const ticket = {
    ...body,
    task_id: taskId,
    content_sha256: taskId,
    action_id: `prepare-a-task-${taskId}`,
  }
  await mkdir(join(runRoot, 'task-tickets', 'A'), { recursive: true })
  const ticketText = `${JSON.stringify(ticket, null, 2).replace('"hash_probe": 1', '"hash_probe": 1.0')}\n`
  await writeFile(join(runRoot, 'task-tickets', 'A', `${taskId}.json`), ticketText)
  const agent = {
    id: 'agent-coordinator', session: { id: 'session-coordinator' }, status: 'running',
    inbox: { nextTurn: [], nextStep: [] },
    cancel() { this.status = 'idle' },
    whenIdle() { return Promise.resolve() },
  }
  const aAgent = {
    id: 'agent-a', session: { id: 'session-a' }, status: 'idle', cancelCalls: [],
    cancel(cause, options) { this.cancelCalls.push({ cause, options }); this.status = 'idle' },
    whenIdle() { return Promise.resolve() },
  }
  const bAgent = {
    id: 'agent-b', session: { id: 'session-b' }, status: 'idle', cancelCalls: [],
    cancel(cause, options) { this.cancelCalls.push({ cause, options }); this.status = 'idle' },
    whenIdle() { return Promise.resolve() },
  }
  const roles = {
    coordinator: 'session-coordinator',
    A: 'session-a',
    B: 'session-b',
  }
  const events = []
  const requests = []
  const sessions = {
    async inspect() { return { events } },
    async prompt(request) {
      requests.push(request)
      events.push({
        type: 'user/message',
        seq: events.length,
        data: {
          source: { kind: 'user', rpcId: request.requestId },
          content: request.content,
        },
      })
      return { accepted: true }
    },
  }
  const liveAgents = [agent, aAgent, bAgent]
  const agents = {
    get(id) { return liveAgents.find(item => item.id === id || item.session.id === id) },
    list() { return liveAgents },
    currentInitiator() { return agent },
  }
  let currentGoal
  const goals = { get() { return currentGoal } }
  const originalCancel = async request => {
    assert.equal(request.sessionId, agent.session.id)
    agent.cancel({ kind: 'user' })
  }
  const originalGoalPause = async (_target, ref) => {
    if (currentGoal?.id === ref.id && currentGoal.revision === ref.revision) {
      currentGoal = { ...currentGoal, phase: 'paused', revision: currentGoal.revision + 1 }
    }
    return currentGoal
  }
  const dispatcher = createTicketDispatcher({
    config: {
      runRoot,
      bindingPath: join(runRoot, 'binding.json'),
      pythonExecutable: process.execPath,
      projectRoot: runRoot,
      sessionRoles: roles,
      goalMaxRounds: 3,
      stopTimeoutMs: 1000,
    },
    sessions,
    agents,
    goals,
    originalPrompt: sessions.prompt,
    originalCancel,
    originalGoalPause,
    actionControl,
  })
  return {
    runRoot, taskId, ticketText, agent, aAgent, bAgent, agents, sessions, requests, events, dispatcher,
    goals,
    setCurrentGoal(goal) { currentGoal = goal },
  }
}

async function frozenBTicket(value) {
  const contractBytes = await readFile(join(value.runRoot, 'contract.json'))
  const contract = JSON.parse(contractBytes.toString('utf8'))
  const manifestRef = 'a-manifest-001'
  const candidateId = 'synthetic-candidate-001'
  const completion = {
    manifest_ref: manifestRef,
    candidate_ids: [candidateId],
    retained_ids: [candidateId],
  }
  const completionText = `${JSON.stringify(completion, null, 2)}\n`
  await writeFile(join(value.runRoot, 'a-complete.json'), completionText)
  const frozen = {
    contract,
    candidates: { [candidateId]: { definition: { expression: 'perp_close' }, retained_horizons: [24] } },
    a_manifest_ref: manifestRef,
  }
  const frozenText = `${JSON.stringify(frozen, null, 2)}\n`
  await writeFile(join(value.runRoot, 'frozen_batch.json'), frozenText)
  const binding = JSON.parse(await readFile(join(value.runRoot, 'binding.json'), 'utf8'))
  const body = {
    format_version: 1,
    research_id: contract.run_id,
    batch_id: contract.run_id,
    task_type: 'frozen_b_validation',
    target_version: binding.target.version,
    contract_version: 'FM-v6',
    contract_sha256: sha256(contractBytes),
    target: binding.target,
    goal: binding.goal,
    contract,
    a_completion: {
      manifest_ref: manifestRef,
      frozen_candidate_ids: [candidateId],
      content_sha256: sha256(completionText),
    },
    frozen_batch_sha256: sha256(frozenText),
    frozen_batch: frozen,
    thin_spec: {
      data_environment: {
        segment: 'B',
        purpose: contract.purpose,
        include_liquidations: binding.inputs.include_liquidations,
        interval_hours: contract.sample_hours,
        timezone: 'UTC',
        start_inclusive: contract.b_start,
        end_exclusive: contract.c_start,
        warmup_hours: contract.max_lookback_hours,
        universe_provenance: contract.universe_provenance,
        coverage: 'not loaded; the bound B validator reports coverage after its single access',
      },
      validation_policy: {
        candidate_set: 'exact frozen_batch.candidates; no additions, removals, or substitutions',
        definitions: 'frozen formula, direction, and retained horizons are immutable',
        numerical_access: 'one B validation; persist the checkpoint before interpretation',
        incomplete_access: 'if access started without a complete checkpoint, inspect evidence and stop',
        feedback_to_a: 'never return B metrics to A or use them to modify this frozen batch',
      },
    },
    expected_deliverables: {
      numerical_checkpoint: 'single frozen-batch B validation with complete saved checkpoint',
      interpretation: 'evidence-grounded report for every frozen candidate',
      delivery: 'complete B delivery from the saved numerical and interpretation records',
    },
  }
  const taskId = sha256(canonicalJson(body))
  const ticket = {
    ...body,
    task_id: taskId,
    content_sha256: taskId,
    action_id: `prepare-b-task-${taskId}`,
  }
  const ticketText = `${JSON.stringify(ticket, null, 2)}\n`
  await mkdir(join(value.runRoot, 'task-tickets', 'B'), { recursive: true })
  await writeFile(join(value.runRoot, 'task-tickets', 'B', `${taskId}.json`), ticketText)
  return { taskId, ticketText, ticket }
}

function finishATurn(value, seq = 1) {
  value.events.push({ type: 'turn/end', seq,
    data: { turn: 1, reason: { kind: 'completed' } } })
  value.aAgent.status = 'idle'
}

test('A result waits for quiescence and duplicate events notify the Coordinator once', async t => {
  const value = await fixture(t)
  value.setCurrentGoal({ id: 'goal-active', revision: 1, phase: 'active', activation: 'armed' })
  await value.dispatcher.dispatchA(value.taskId, value.agent)
  finishATurn(value)
  let settle
  value.aAgent.status = 'running'
  value.aAgent.whenIdle = () => new Promise(resolve => { settle = resolve })
  const first = value.dispatcher.collectAResult(1)
  const duplicate = value.dispatcher.collectAResult(1)
  await new Promise(resolve => setImmediate(resolve))
  assert.equal(value.requests.length, 1)
  const progress = await value.dispatcher.runtimeStatus(value.agent)
  assert.equal(progress.sessions.A.status, 'running')
  assert.equal(progress.state, 'active')
  value.aAgent.status = 'idle'
  value.aAgent.whenIdle = () => Promise.resolve()
  settle()
  assert.equal((await first).status, 'accepted')
  assert.equal((await duplicate).status, 'accepted')
  assert.equal(value.requests.length, 2)
  assert.equal(value.requests[1].sessionId, value.agent.session.id)
  assert.match(value.requests[1].content[0].text, /not proof the task or A research is complete/)
  assert.match(value.requests[1].content[0].text, /read_a_result/)
})

test('B result notification is receipt-backed, deduplicated and never delivered to A', async t => {
  const value = await fixture(t)
  const frozen = await frozenBTicket(value)
  value.setCurrentGoal({ id: 'goal-active', revision: 1, phase: 'active', activation: 'armed' })
  await value.dispatcher.dispatchB(frozen.taskId, value.agent)
  finishATurn(value)
  const first = await value.dispatcher.collectBResult(1)
  const duplicate = await value.dispatcher.collectBResult(1)
  assert.equal(first.request_id, duplicate.request_id)
  assert.equal(value.requests.length, 2)
  assert.equal(value.requests[0].sessionId, 'session-b')
  assert.equal(value.requests[1].sessionId, 'session-coordinator')
  assert.match(value.requests[1].content[0].text, /read_b_result/)
  assert.match(value.requests[1].content[0].text, /never reload B/)
  assert.equal(value.requests.filter(row => row.sessionId === 'session-a').length, 0)
})

test('known per-turn budget boundary permits only the existing bounded checkpoint continuation', async t => {
  const value = await fixture(t)
  value.setCurrentGoal({ id: 'goal-active', revision: 1, phase: 'active', activation: 'armed' })
  await value.dispatcher.dispatchA(value.taskId, value.agent)
  finishATurn(value)
  value.events.at(-1).data.reason = { kind: 'error', error: {
    message: 'research Agent reached the configured model-request limit for this turn',
  } }
  await value.dispatcher.collectAResult(1)
  assert.match(value.requests[1].content[0].text, /reason per_turn_budget_exhausted/)
  assert.match(value.requests[1].content[0].text, /finite continuation limits/)
})

test('recovers exact A action arguments from the original native tool call and receipt hash', async t => {
  const value = await fixture(t)
  const args = { action_id: 'close-route-original', route_id: 'route-one',
    status: 'stopped', reason: '保持原始中文理由，不从后来的报告猜测。' }
  const { action_id, ...parameters } = args
  await mkdir(join(value.runRoot, 'action_receipts'))
  await writeFile(join(value.runRoot, 'action_receipts', `${action_id}.started.json`), JSON.stringify({
    action_id, name: 'resolve_experiment', parameters_sha256: sha256(canonicalJson(parameters)) }))
  value.events.push({ type: 'tool/call', seq: 4, data: {
    name: 'mcp__fr_a__resolve_experiment', arguments: '{' } })
  value.events.push({ type: 'tool/call', seq: 5, data: {
    name: 'mcp__fr_a__resolve_experiment', arguments: JSON.stringify(args) } })
  value.events.push({ type: 'tool/call', seq: 6, data: {
    name: 'mcp__fr_a__resolve_experiment', arguments: JSON.stringify({ ...args, reason: '后来猜测的理由' }) } })
  value.aAgent.status = 'running'
  value.agents.currentInitiator = () => value.aAgent
  const result = await value.dispatcher.readAActionRequest(action_id, value.aAgent)
  assert.deepEqual(result.arguments, args)
  assert.equal(result.source.seq, 5)
  await assert.rejects(value.dispatcher.readAActionRequest(action_id, value.agent), /active bound A/)
  await assert.rejects(value.dispatcher.readAActionRequest('../B/record', value.aAgent), /identifier/)
  value.events.splice(0, 2)
  await assert.rejects(value.dispatcher.readAActionRequest(action_id, value.aAgent), /cannot be verified/)
})

test('Coordinator resumes only an exact per-turn cap boundary once and never retries provider failures', async t => {
  const value = await fixture(t)
  value.setCurrentGoal({ id: 'goal-active', revision: 1, phase: 'active', activation: 'armed' })
  value.agent.status = 'idle'
  value.events.push({ type: 'turn/end', seq: 81, data: { turn: 2,
    reason: { kind: 'error', error: { message: 'provider disconnected' } } } })
  assert.equal((await value.dispatcher.collectCoordinatorBoundary(81)).status, 'no_current_boundary')
  assert.equal(value.requests.length, 0)
  value.events.push({ type: 'turn/end', seq: 82, data: { turn: 3,
    reason: { kind: 'error', error: { message: 'research Agent reached the configured model-request limit for this turn' } } } })
  assert.equal((await value.dispatcher.collectCoordinatorBoundary(82)).status, 'accepted')
  assert.equal((await value.dispatcher.collectCoordinatorBoundary(82)).status, 'accepted')
  assert.equal(value.requests.length, 1)
  assert.equal(value.requests[0].sessionId, 'session-coordinator')
  assert.match(value.requests[0].content[0].text, /same Goal/)
  value.setCurrentGoal({ id: 'goal-active', revision: 2, phase: 'paused', activation: 'disarmed' })
  assert.equal((await value.dispatcher.collectCoordinatorBoundary(82)).status, 'suppressed')
  assert.equal(value.requests.length, 1)
})

test('recovers numerical A arguments using the domain-bound panel fingerprint', async t => {
  const value = await fixture(t)
  const args = { action_id: 'evaluate-original', candidate_id: 'c1',
    definition: { direction: 1, expression: 'ts_return(perp_close, 3)', reason: '原始定义' } }
  const { action_id, ...parameters } = args
  const panelHash = 'f'.repeat(64)
  await mkdir(join(value.runRoot, 'action_receipts'))
  await writeFile(join(value.runRoot, 'a-action-input-binding.json'), JSON.stringify({
    run_id: 'dispatch-fixture', panel_sha256: panelHash }))
  await writeFile(join(value.runRoot, 'action_receipts', `${action_id}.started.json`), JSON.stringify({
    action_id, name: 'evaluate_candidate', parameters_sha256: sha256(canonicalJson({
      ...parameters, experiment_id: null, a_panel_sha256: panelHash })) }))
  value.events.push({ type: 'tool/call', seq: 8, data: {
    name: 'mcp__fr_a__evaluate_candidate', arguments: JSON.stringify(args) } })
  value.aAgent.status = 'running'
  value.agents.currentInitiator = () => value.aAgent
  const result = await value.dispatcher.readAActionRequest(action_id, value.aAgent)
  assert.deepEqual(result.arguments, args)
  assert.equal(result.source.seq, 8)
  await writeFile(join(value.runRoot, 'a-action-input-binding.json'), JSON.stringify({
    run_id: 'dispatch-fixture', panel_sha256: 'a'.repeat(64) }))
  await assert.rejects(value.dispatcher.readAActionRequest(action_id, value.aAgent), /cannot be verified/)
})

test('collects B continuation one from its actual continuation.json receipt', async t => {
  const value = await fixture(t)
  const frozen = await frozenBTicket(value)
  value.setCurrentGoal({ id: 'goal-active', revision: 1, phase: 'active', activation: 'armed' })
  await value.dispatcher.dispatchB(frozen.taskId, value.agent)
  const id = Object.keys(frozen.ticket.frozen_batch.candidates)[0]
  await writeFile(join(value.runRoot, 'b-access-started.json'), JSON.stringify({
    candidate_ids: [id], use: 'fixed_batch_validation', time: '2026-10-18T00:00:00Z' }))
  await writeFile(join(value.runRoot, 'b-numerical-complete.json'), JSON.stringify({
    candidate_ids: [id], retained_horizons: { [id]: [24] } }))
  await value.dispatcher.continueBDelivery(frozen.taskId, 1, value.agent)
  finishATurn(value, 99)
  assert.equal((await value.dispatcher.collectBResult(99)).status, 'accepted')
  assert.equal(value.requests.at(-1).sessionId, 'session-coordinator')
  assert.match(value.requests.at(-1).content[0].text, /turn 1, end seq 99/)
})

test('explicit human resume authorizes bounded recovery of a prior interrupted turn once per Goal revision', async t => {
  const value = await fixture(t)
  value.setCurrentGoal({ id: 'goal-active', revision: 1, phase: 'active', activation: 'armed' })
  await value.dispatcher.dispatchA(value.taskId, value.agent)
  finishATurn(value)
  value.events.at(-1).data.reason = { kind: 'aborted', reason: { kind: 'user' } }
  await value.dispatcher.collectAResult(1)
  value.setCurrentGoal({ id: 'goal-active', revision: 2, phase: 'active', activation: 'armed' })
  const recovered = await value.dispatcher.collectAResult(1, { explicitResume: true })
  const duplicate = await value.dispatcher.collectAResult(1, { explicitResume: true })
  assert.equal(recovered.request_id, duplicate.request_id)
  assert.equal(value.requests.length, 3)
  assert.match(value.requests.at(-1).content[0].text, /explicitly resumed the same native Goal at revision 2/)
  assert.match(value.requests.at(-1).content[0].text, /original IDs and parameters/)
})

test('A result cannot wake a paused Goal, including a pause during its idle wait', async t => {
  const value = await fixture(t)
  value.setCurrentGoal({ id: 'goal-active', revision: 1, phase: 'active', activation: 'armed' })
  await value.dispatcher.dispatchA(value.taskId, value.agent)
  finishATurn(value)
  let settle
  value.aAgent.whenIdle = () => new Promise(resolve => { settle = resolve })
  const waiting = value.dispatcher.collectAResult(1)
  await new Promise(resolve => setImmediate(resolve))
  value.dispatcher.suspendResults()
  value.setCurrentGoal({ id: 'goal-active', revision: 2, phase: 'paused', activation: 'disarmed' })
  value.aAgent.whenIdle = () => Promise.resolve()
  settle()
  assert.equal((await waiting).status, 'suppressed')
  assert.equal((await value.dispatcher.collectAResult()).status, 'suppressed')
  await assert.rejects(value.dispatcher.dispatchA(value.taskId, value.agent), /explicitly resumed/)
  assert.equal(value.requests.length, 1)
  value.setCurrentGoal({ id: 'goal-active', revision: 3, phase: 'active', activation: 'armed' })
  assert.equal((await value.dispatcher.collectAResult()).status, 'accepted')
  assert.equal(value.requests.length, 2)
})

test('sealed zero-retained A still notifies from the accepted source; old events do not', async t => {
  const value = await fixture(t)
  value.setCurrentGoal({ id: 'goal-active', revision: 1, phase: 'active', activation: 'armed' })
  await value.dispatcher.dispatchA(value.taskId, value.agent)
  finishATurn(value, 2)
  await writeFile(join(value.runRoot, 'a-complete.json'), JSON.stringify({
    status: 'complete', candidate_ids: [], retained_ids: [], manifest_ref: 'a-manifest',
  }))
  assert.equal((await value.dispatcher.collectAResult(1)).status, 'no_current_result')
  assert.equal(value.requests.length, 1)
  assert.equal((await value.dispatcher.collectAResult(2)).status, 'accepted')
  assert.match(value.requests[1].content[0].text, /skip freeze\/B/)
  assert.equal(value.requests.length, 2)
  await assert.rejects(value.dispatcher.dispatchA(value.taskId, value.agent), /after A seal/)
})

test('stopped or active-disarmed research cannot automatically collect or dispatch A', async t => {
  const value = await fixture(t)
  value.setCurrentGoal({ id: 'goal-active', revision: 1, phase: 'active', activation: 'armed' })
  await value.dispatcher.dispatchA(value.taskId, value.agent)
  finishATurn(value)
  value.setCurrentGoal({ id: 'goal-active', revision: 1, phase: 'active', activation: 'disarmed' })
  assert.equal((await value.dispatcher.collectAResult()).status, 'suppressed')
  await assert.rejects(value.dispatcher.continueATask(value.taskId, 1, value.agent), /explicitly resumed/)
  const directory = join(value.runRoot, 'action_control')
  await mkdir(directory)
  await writeFile(join(directory, '00000001-test-stop.json'), JSON.stringify({
    event_id: '00000001-test-stop', event: 'stop', action_id: 'test-stop', reason: 'explicit UI Stop',
  }))
  value.setCurrentGoal({ id: 'goal-active', revision: 1, phase: 'active', activation: 'armed' })
  assert.equal((await value.dispatcher.collectAResult()).status, 'suppressed')
  assert.equal(value.requests.length, 1)
})

test('only explicit resume restores a notice discarded before it entered Coordinator history', async t => {
  const value = await fixture(t)
  value.setCurrentGoal({ id: 'goal-active', revision: 1, phase: 'active', activation: 'armed' })
  await value.dispatcher.dispatchA(value.taskId, value.agent)
  finishATurn(value)
  const first = await value.dispatcher.collectAResult(1)
  assert.equal(value.requests.length, 2)
  assert.equal((await value.dispatcher.collectAResult(undefined, { explicitResume: true })).request_id,
    first.request_id)
  assert.equal(value.requests.length, 2)
  value.events.splice(2) // The accepted notice was cancelled before logging user/message.
  value.dispatcher.suspendResults()
  value.setCurrentGoal({ id: 'goal-active', revision: 2, phase: 'paused', activation: 'disarmed' })
  assert.equal((await value.dispatcher.collectAResult()).status, 'suppressed')
  value.setCurrentGoal({ id: 'goal-active', revision: 3, phase: 'active', activation: 'armed' })
  await value.dispatcher.collectAResult()
  assert.equal(value.requests.length, 2)
  const resumed = await value.dispatcher.collectAResult(undefined, { explicitResume: true })
  assert.equal(resumed.request_id, first.request_id)
  assert.equal(value.requests.length, 3)
  assert.equal(value.requests.filter(request => request.sessionId === 'session-a').length, 1)
})

test('restart can explicitly rearm the same active Coordinator Goal without domain writes', async t => {
  const value = await fixture(t, { actionControl: async () => assert.fail('active run needs no ledger resume') })
  value.agent.status = 'idle'
  value.setCurrentGoal({ id: 'goal-restarted', revision: 2, phase: 'active' })
  assert.deepEqual(await value.dispatcher.resumeRun(value.agent, { id: 'goal-restarted', revision: 2 }),
    { status: 'active', source: 'program_action_control' })
  await assert.rejects(value.dispatcher.resumeRun(value.agent, { id: 'goal-restarted', revision: 1 }),
    /paused or active bound Coordinator Goal/)
})

test('Coordinator Stop cancels and settles all bound Sessions before Goal resume', async t => {
  const value = await fixture(t, {
    actionControl: async (action, { actionId, reason } = {}) => {
      const root = join(value.runRoot, 'action_control')
      await mkdir(root, { recursive: true })
      const events = await readdir(root)
      const prior = events.filter(name => /^\d{8}-.+\.json$/.test(name)).length
      const event = action === 'begin-stop' ? 'stop_requested'
        : action === 'finish-stop' ? 'stop' : 'resume'
      const eventId = `${String(prior + 1).padStart(8, '0')}-${actionId}`
      await writeFile(join(root, `${eventId}.json`), JSON.stringify({
        event_id: eventId, event, action_id: actionId,
        ...(reason === undefined ? {} : { reason }),
      }))
      return { status: action === 'begin-stop' ? 'stopping'
        : action === 'finish-stop' ? 'stopped' : 'active',
      source: 'program_action_control', event_id: eventId,
      ...(reason === undefined ? {} : { reason }) }
    },
  })
  value.agent.status = 'running'
  value.aAgent.status = 'running'
  value.bAgent.status = 'running'
  value.setCurrentGoal({ id: 'goal-001', revision: 1, phase: 'active' })

  const stopped = await value.dispatcher.stopRun('session-coordinator')

  assert.equal(stopped.status, 'stopped')
  assert.equal(value.aAgent.status, 'idle')
  assert.equal(value.bAgent.status, 'idle')
  assert.deepEqual(value.aAgent.cancelCalls[0], { cause: { kind: 'user' }, options: { keepInbox: true } })
  assert.deepEqual(value.bAgent.cancelCalls[0], { cause: { kind: 'user' }, options: { keepInbox: true } })
  assert.equal(value.goals.get().phase, 'paused')
  const status = await value.dispatcher.runtimeStatus(value.agent)
  assert.equal(status.state, 'stopped')
  assert.equal(status.sessions.coordinator.status, 'idle')
  await assert.rejects(value.dispatcher.requireRunActive('A'), /research run is stopped/)

  const resumed = await value.dispatcher.resumeRun(value.agent, {
    id: 'goal-001', revision: value.goals.get().revision,
  })
  assert.equal(resumed.status, 'active')
  await value.dispatcher.requireRunActive('A')
  assert.equal((await value.dispatcher.runtimeStatus(value.agent)).state, 'active')
})

async function attachVersionedChange(value) {
  const contractBytes = await readFile(join(value.runRoot, 'contract.json'))
  const contract = JSON.parse(contractBytes.toString('utf8'))
  const prior = JSON.parse(await readFile(join(value.runRoot, 'task-tickets', 'A', `${value.taskId}.json`), 'utf8'))
  const parentSnapshot = {
    run_id: 'dispatch-parent',
    a_status: 'open',
    research_contract: { run_id: 'dispatch-parent' },
  }
  const parent = {
    research_id: 'dispatch-parent',
    target: { version: 1, objective: 'Prior objective.' },
    contract_sha256: 'a'.repeat(64),
    a_snapshot: parentSnapshot,
    a_snapshot_sha256: sha256(canonicalJson(parentSnapshot)),
    evidence_refs: [{ record_id: 'parent-evidence-1', kind: 'candidate', sha256: 'b'.repeat(64) }],
    task_refs: [{ task_id: 'c'.repeat(64), task_type: 'initial_a_research' }],
    action_refs: [],
  }
  const child = {
    research_id: contract.run_id,
    target: { version: 2, objective: prior.target.objective },
    contract,
    contract_sha256: sha256(contractBytes),
    goal: prior.goal,
  }
  const recordBody = {
    format_version: 1,
    parent,
    child,
    change: {
      original_intent: 'Change the threshold.',
      adopted_change: 'Use version 2.',
      new_objective: child.target.objective,
      contract_changes: {},
      goal_changes: {},
    },
    approval_source: 'local_operator_cli',
    approval_sha256: 'd'.repeat(64),
    recorded_at: '2026-10-17T00:00:00+00:00',
  }
  const record = {
    ...recordBody,
    record_id: `change-${contract.run_id}-v2`,
    content_sha256: sha256(canonicalJson(recordBody)),
  }
  await writeFile(join(value.runRoot, 'change-record.json'), `${JSON.stringify(record)}\n`)
  const changeRef = {
    record_id: record.record_id,
    content_sha256: record.content_sha256,
    parent_research_id: parent.research_id,
    parent_contract_sha256: parent.contract_sha256,
    parent_target_version: parent.target.version,
    parent_a_snapshot_sha256: parent.a_snapshot_sha256,
    affected_task_refs: parent.task_refs,
    affected_evidence_refs: parent.evidence_refs,
  }
  const body = { ...prior }
  delete body.task_id
  delete body.content_sha256
  delete body.action_id
  body.target_version = 2
  body.target = child.target
  body.change_ref = changeRef
  const bodyHashInput = canonicalJson(body).replace('"hash_probe":1', '"hash_probe":1.0')
  const taskId = sha256(bodyHashInput)
  const ticket = {
    ...body,
    task_id: taskId,
    content_sha256: taskId,
    action_id: `prepare-a-task-${taskId}`,
  }
  const ticketText = `${JSON.stringify(ticket, null, 2).replace('"hash_probe": 1', '"hash_probe": 1.0')}\n`
  await writeFile(join(value.runRoot, 'task-tickets', 'A', `${taskId}.json`), ticketText)
  const bindingPath = join(value.runRoot, 'binding.json')
  const binding = JSON.parse(await readFile(bindingPath, 'utf8'))
  binding.target = child.target
  await writeFile(bindingPath, JSON.stringify(binding))
  return { taskId, ticketText }
}

test('validates human change lineage before dispatching the child A ticket', async t => {
  const value = await fixture(t)
  const { taskId, ticketText } = await attachVersionedChange(value)

  assert.deepEqual(await value.dispatcher.dispatchA(taskId, value.agent), {
    task_id: taskId,
    status: 'accepted',
    target_role: 'A',
  })
  assert.deepEqual(value.requests[0].content, [{ type: 'text', text: ticketText }])

  const tampered = await fixture(t)
  const invalid = await attachVersionedChange(tampered)
  const recordPath = join(tampered.runRoot, 'change-record.json')
  const record = JSON.parse(await readFile(recordPath, 'utf8'))
  record.change.adopted_change = 'Changed without a new content digest.'
  await writeFile(recordPath, JSON.stringify(record))
  await assert.rejects(tampered.dispatcher.dispatchA(invalid.taskId, tampered.agent),
    /content hash is invalid/)
  assert.equal(tampered.requests.length, 0)
})

test('creates only the bound Coordinator Goal with the configured finite round cap', async t => {
  const value = await fixture(t)
  await value.dispatcher.initialize()
  const contract = JSON.parse(await readFile(join(value.runRoot, 'contract.json'), 'utf8'))
  const created = []
  const originalCreate = (agent, request) => {
    created.push({ agent, request })
    return { id: 'goal-fixed', revision: 1, phase: 'active', ...request }
  }

  const goal = value.dispatcher.createGoal(value.agent, { objective: contract.objective }, originalCreate)

  assert.equal(goal.id, 'goal-fixed')
  assert.deepEqual(created[0], {
    agent: value.agent,
    request: { objective: contract.objective, maxGoalRounds: 3 },
  })
  assert.throws(() => value.dispatcher.createGoal(value.agent, {
    objective: contract.objective, maxGoalRounds: 4,
  }, originalCreate), error => error.code === 'factor-research/admission-required')
  assert.throws(() => value.dispatcher.createGoal(value.agent, {
    objective: 'model-invented objective',
  }, originalCreate), error => error.code === 'factor-research/admission-required')
  assert.equal(created.length, 1)

  value.setCurrentGoal({ id: 'already-created-goal', phase: 'complete' })
  assert.throws(() => value.dispatcher.createGoal(value.agent, {
    objective: contract.objective,
  }, originalCreate), error => error.code === 'factor-research/admission-required'
    && /already has a native DSH Goal/.test(error.message))
  assert.equal(created.length, 1)
})

test('requires a Coordinator for Goal creation and requires versioned change tickets for edit or clear', async t => {
  const value = await fixture(t)
  await value.dispatcher.initialize()
  const contract = JSON.parse(await readFile(join(value.runRoot, 'contract.json'), 'utf8'))
  const aAgent = { id: 'agent-a', session: { id: 'session-a' }, status: 'running' }
  let creates = 0

  assert.throws(() => value.dispatcher.createGoal(aAgent, { objective: contract.objective }, () => {
    creates += 1
  }), error => error.code === 'factor-research/admission-required')
  assert.throws(() => value.dispatcher.rejectGoalEdit(), error =>
    error.code === 'factor-research/admission-required' && /versioned human change ticket/.test(error.message))
  assert.throws(() => value.dispatcher.rejectGoalClear(), error =>
    error.code === 'factor-research/admission-required' && /disabled/.test(error.message))
  assert.equal(creates, 0)
})

test('caps model requests per bound Agent turn and clamps output tokens', async t => {
  const budgetRoot = await mkdtemp(join(tmpdir(), 'factor-research-model-budget-'))
  t.after(() => rm(budgetRoot, { recursive: true, force: true }))
  const guard = createModelRequestGuard({
    roleForSession: sessionId => sessionId === 'session-a' ? 'A' : undefined,
    budgetRoot,
    maxRequestsPerTurn: 2,
    maxTotalRequests: 5,
    maxOutputTokens: 2048,
  })
  const agent = { session: { id: 'session-a' } }
  let calls = 0
  const next = async () => {
    calls += 1
    return { provider: 'deepseek-account', model: 'deepseek-flash', maxTokens: 8192 }
  }

  assert.equal((await guard({ agent, turn: 1 }, next)).maxTokens, 2048)
  assert.equal((await guard({ agent, turn: 1 }, next)).maxTokens, 2048)
  await assert.rejects(guard({ agent, turn: 1 }, next), error =>
    error.code === 'factor-research/request-budget-exhausted')
  assert.equal(calls, 2)

  assert.equal((await guard({ agent, turn: 2 }, async () => ({ maxTokens: 1024 }))).maxTokens, 1024)
  assert.equal((await guard({ agent: { session: { id: 'unbound-session' } }, turn: 1 }, next)).maxTokens, 2048)
})

test('caps total provider requests across bound Agent turns', async t => {
  const budgetRoot = await mkdtemp(join(tmpdir(), 'factor-research-model-budget-'))
  t.after(() => rm(budgetRoot, { recursive: true, force: true }))
  const guard = createModelRequestGuard({
    roleForSession: sessionId => sessionId === 'session-coordinator' ? 'coordinator' : undefined,
    budgetRoot,
    maxRequestsPerTurn: 8,
    maxTotalRequests: 2,
    maxOutputTokens: 2048,
  })
  const agent = { session: { id: 'session-coordinator' } }
  let admittedRequests = 0
  const next = async () => ({
    provider: 'deepseek-account', model: 'deepseek-flash', maxTokens: 4096,
  })

  await guard({ agent, turn: 1 }, next)
  admittedRequests += 1
  await guard({ agent, turn: 2 }, next)
  admittedRequests += 1
  await assert.rejects(guard({ agent, turn: 3 }, next), error =>
    error.code === 'factor-research/run-budget-exhausted')
  assert.equal(admittedRequests, 2)
})

test('persists total request reservations across Host guard recreation', async t => {
  const budgetRoot = await mkdtemp(join(tmpdir(), 'factor-research-model-budget-'))
  t.after(() => rm(budgetRoot, { recursive: true, force: true }))
  const options = {
    roleForSession: sessionId => sessionId === 'session-coordinator' ? 'coordinator' : undefined,
    budgetRoot,
    maxRequestsPerTurn: 8,
    maxTotalRequests: 2,
    maxOutputTokens: 2048,
  }
  const agent = { session: { id: 'session-coordinator' } }
  const firstHostGuard = createModelRequestGuard(options)
  const next = async () => ({ maxTokens: 2048 })

  await firstHostGuard({ agent, turn: 1 }, next)
  await firstHostGuard({ agent, turn: 2 }, next)

  const restartedHostGuard = createModelRequestGuard(options)
  await assert.rejects(restartedHostGuard({ agent, turn: 3 }, next), error =>
    error.code === 'factor-research/run-budget-exhausted')
})

test('reserves the finite total request budget atomically across bound Agents', async t => {
  const budgetRoot = await mkdtemp(join(tmpdir(), 'factor-research-model-budget-'))
  t.after(() => rm(budgetRoot, { recursive: true, force: true }))
  const guard = createModelRequestGuard({
    roleForSession: sessionId => sessionId.startsWith('session-') ? 'research' : undefined,
    budgetRoot,
    maxRequestsPerTurn: 20,
    maxTotalRequests: 3,
    maxOutputTokens: 2048,
  })
  const agents = Array.from({ length: 10 }, (_, index) => ({
    session: { id: `session-${index}` },
  }))
  let admittedRequests = 0
  const results = await Promise.allSettled(agents.map(async agent => {
    const request = await guard({ agent, turn: 1 }, async () => ({
      provider: 'deepseek-account', model: 'deepseek-flash', maxTokens: 2048,
    }))
    admittedRequests += 1
    return request
  }))

  assert.equal(results.filter(result => result.status === 'fulfilled').length, 3)
  assert.equal(admittedRequests, 3)
})

test('delivers the saved ticket to bound A and treats repeated dispatch as idempotent', async t => {
  const value = await fixture(t)
  assert.deepEqual(await value.dispatcher.dispatchA(value.taskId, value.agent), {
    task_id: value.taskId,
    status: 'accepted',
    target_role: 'A',
  })
  assert.equal(value.requests.length, 1)
  assert.equal(value.requests[0].sessionId, 'session-a')
  assert.equal(value.requests[0].mode, 'queue')
  assert.deepEqual(value.requests[0].content, [{ type: 'text', text: value.ticketText }])
  assert.deepEqual(await value.dispatcher.dispatchA(value.taskId, value.agent), {
    task_id: value.taskId,
    status: 'accepted',
    target_role: 'A',
  })
  assert.equal(value.requests.length, 1)
})

test('continues an accepted A ticket only while its snapshot is current and the A Agent is idle', async t => {
  const value = await fixture(t)
  await value.dispatcher.dispatchA(value.taskId, value.agent)
  value.events.push({ type: 'assistant/message', seq: value.events.length, data: { message: {} } })

  assert.deepEqual(await value.dispatcher.continueATask(value.taskId, 1, value.agent), {
    task_id: value.taskId,
    status: 'accepted',
    target_role: 'A',
  })
  assert.equal(value.requests.length, 2)
  assert.equal(value.requests[1].sessionId, 'session-a')
  assert.equal(value.requests[1].mode, 'queue')
  assert.match(value.requests[1].content[0].text, /Host continuation 1/)
  assert.match(value.requests[1].content[0].text, new RegExp(value.taskId))
  assert.deepEqual(await value.dispatcher.continueATask(value.taskId, 1, value.agent), {
    task_id: value.taskId,
    status: 'accepted',
    target_role: 'A',
  })
  assert.equal(value.requests.length, 2)
  const receipt = JSON.parse(await readFile(join(
    value.runRoot, 'dispatch-receipts', 'A', `continuation-${value.taskId}-1.json`,
  ), 'utf8'))
  assert.equal(receipt.status, 'accepted')
  assert.equal(receipt.target_role, 'A')

  const stale = await fixture(t)
  await stale.dispatcher.dispatchA(stale.taskId, stale.agent)
  await writeFile(join(stale.runRoot, 'a_records', 'new-record.json'), JSON.stringify({
    id: 'new-record', kind: 'candidate', data: {},
  }))
  await assert.rejects(stale.dispatcher.continueATask(stale.taskId, 1, stale.agent),
    /stale relative to the current A evidence/)
  assert.equal(stale.requests.length, 1)

  const busy = await fixture(t)
  await busy.dispatcher.dispatchA(busy.taskId, busy.agent)
  busy.events.push({ type: 'assistant/message', seq: busy.events.length, data: { message: {} } })
  busy.agents.get = id => id === 'session-a' ? { status: 'running' } : id === busy.agent.id ? busy.agent : undefined
  await assert.rejects(busy.dispatcher.continueATask(busy.taskId, 1, busy.agent),
    /A Session is still processing the accepted ticket/)
  assert.equal(busy.requests.length, 1)
})

test('requires an accepted A response, ordered continuation attempts, and an open A stage', async t => {
  const missing = await fixture(t)
  await assert.rejects(missing.dispatcher.continueATask(missing.taskId, 1, missing.agent),
    /accepted A task receipt/)
  assert.equal(missing.requests.length, 0)

  const value = await fixture(t)
  await value.dispatcher.dispatchA(value.taskId, value.agent)
  value.events.push({ type: 'assistant/message', seq: value.events.length, data: { message: {} } })
  await assert.rejects(value.dispatcher.continueATask(value.taskId, 2, value.agent),
    /requires accepted attempt 1/)
  assert.equal(value.requests.length, 1)

  await writeFile(join(value.runRoot, 'a-complete.json'), '{}')
  await assert.rejects(value.dispatcher.continueATask(value.taskId, 1, value.agent),
    /after A seal or B access/)
  assert.equal(value.requests.length, 1)
})

test('recovers an accepted DSH message when the receipt update was interrupted', async t => {
  const value = await fixture(t)
  value.sessions.prompt = async function prompt(request) {
    value.requests.push(request)
    value.events.push({
      type: 'user/message',
      data: {
        source: { kind: 'user', rpcId: request.requestId },
        content: request.content,
      },
    })
    throw new Error('simulated receipt interruption after DSH acceptance')
  }
  const dispatcher = createTicketDispatcher({
    config: {
      runRoot: value.runRoot,
      bindingPath: join(value.runRoot, 'binding.json'),
      pythonExecutable: process.execPath,
      projectRoot: value.runRoot,
      sessionRoles: { coordinator: 'session-coordinator', A: 'session-a', B: 'session-b' },
    },
    sessions: value.sessions,
    agents: { get: () => value.agent, currentInitiator: () => value.agent },
    originalPrompt: value.sessions.prompt,
  })
  await assert.rejects(dispatcher.dispatchA(value.taskId, value.agent), /simulated receipt interruption/)
  assert.deepEqual(await dispatcher.dispatchA(value.taskId, value.agent), {
    task_id: value.taskId,
    status: 'accepted',
    target_role: 'A',
  })
  assert.equal(value.requests.length, 1)
  const receipt = JSON.parse(await readFile(join(value.runRoot, 'dispatch-receipts', 'A', `${value.taskId}.json`), 'utf8'))
  assert.equal(receipt.status, 'accepted')
})

test('refuses unbound callers and stale or B-accessed tickets before prompt admission', async t => {
  const value = await fixture(t)
  const aAgent = { id: 'agent-a', session: { id: 'session-a' }, status: 'running' }
  await assert.rejects(value.dispatcher.dispatchA(value.taskId, aAgent), /active bound Coordinator Agent/)
  await writeFile(join(value.runRoot, 'a_records', 'new-record.json'), JSON.stringify({ id: 'new-record', kind: 'candidate', data: {} }))
  await assert.rejects(value.dispatcher.dispatchA(value.taskId, value.agent), /stale relative to the current A evidence/)
  assert.equal(value.requests.length, 0)
})

test('fails closed when Host runRoot, Python binding, or ticket target version disagree', async t => {
  const mismatchedRoot = await fixture(t)
  const alternateRoot = join(mismatchedRoot.runRoot, 'alternate-run')
  await mkdir(alternateRoot)
  const mismatchedRootBindingPath = join(mismatchedRoot.runRoot, 'binding.json')
  const mismatchedRootBinding = JSON.parse(await readFile(mismatchedRootBindingPath, 'utf8'))
  mismatchedRootBinding.run_root = 'alternate-run'
  await writeFile(mismatchedRootBindingPath, JSON.stringify(mismatchedRootBinding))
  await assert.rejects(mismatchedRoot.dispatcher.dispatchA(mismatchedRoot.taskId, mismatchedRoot.agent),
    /does not match the Host run/)
  assert.equal(mismatchedRoot.requests.length, 0)
  await assert.rejects(readFile(join(
    mismatchedRoot.runRoot, 'dispatch-receipts', 'A', `${mismatchedRoot.taskId}.json`,
  )))

  const staleTarget = await fixture(t)
  const staleBindingPath = join(staleTarget.runRoot, 'binding.json')
  const staleBinding = JSON.parse(await readFile(staleBindingPath, 'utf8'))
  staleBinding.target.version = 2
  await writeFile(staleBindingPath, JSON.stringify(staleBinding))
  await assert.rejects(staleTarget.dispatcher.dispatchA(staleTarget.taskId, staleTarget.agent),
    /does not match the fixed research binding/)
  assert.equal(staleTarget.requests.length, 0)

  const mismatchedSession = await fixture(t)
  const bindingPath = join(mismatchedSession.runRoot, 'binding.json')
  const binding = JSON.parse(await readFile(bindingPath, 'utf8'))
  binding.chats.A.thread_id = 'session-not-the-bound-A'
  await writeFile(bindingPath, JSON.stringify(binding))
  await assert.rejects(mismatchedSession.dispatcher.dispatchA(
    mismatchedSession.taskId, mismatchedSession.agent), /does not match the DSH Web A Session/)
  assert.equal(mismatchedSession.requests.length, 0)
})

test('checks the complete saved ticket hash and rejects A tickets containing B results', async t => {
  const tampered = await fixture(t)
  const ticketPath = join(tampered.runRoot, 'task-tickets', 'A', `${tampered.taskId}.json`)
  const ticket = JSON.parse(await readFile(ticketPath, 'utf8'))
  ticket.goal.objective = 'caller-edited objective'
  await writeFile(ticketPath, JSON.stringify(ticket))
  await assert.rejects(tampered.dispatcher.dispatchA(tampered.taskId, tampered.agent), /content hash is invalid/)
  assert.equal(tampered.requests.length, 0)

  const bData = await fixture(t)
  const bTicketPath = join(bData.runRoot, 'task-tickets', 'A', `${bData.taskId}.json`)
  const bTicket = JSON.parse(await readFile(bTicketPath, 'utf8'))
  bTicket.thin_spec.goal_counts.b_qualified_deliverables = 1
  const body = { ...bTicket }
  delete body.task_id
  delete body.content_sha256
  delete body.action_id
  const nextId = sha256(canonicalJson(body))
  bTicket.task_id = nextId
  bTicket.content_sha256 = nextId
  bTicket.action_id = `prepare-a-task-${nextId}`
  await writeFile(join(bData.runRoot, 'task-tickets', 'A', `${nextId}.json`), JSON.stringify(bTicket))
  await assert.rejects(bData.dispatcher.dispatchA(nextId, bData.agent), /A-only information boundary/)
  assert.equal(bData.requests.length, 0)
})

test('refuses dispatch after A seal or B access', async t => {
  const sealed = await fixture(t)
  await writeFile(join(sealed.runRoot, 'a-complete.json'), '{}')
  await assert.rejects(sealed.dispatcher.dispatchA(sealed.taskId, sealed.agent), /after A seal or B access/)
  assert.equal(sealed.requests.length, 0)

  const bAccessed = await fixture(t)
  await writeFile(join(bAccessed.runRoot, 'b-access-started.json'), '{}')
  await assert.rejects(bAccessed.dispatcher.dispatchA(bAccessed.taskId, bAccessed.agent), /after A seal or B access/)
  assert.equal(bAccessed.requests.length, 0)
})

test('dispatches one frozen B ticket to the bound B Session and reuses its receipt', async t => {
  const value = await fixture(t)
  const prepared = await frozenBTicket(value)

  assert.deepEqual(await value.dispatcher.dispatchB(prepared.taskId, value.agent), {
    task_id: prepared.taskId,
    status: 'accepted',
    target_role: 'B',
  })
  assert.equal(value.requests.length, 1)
  assert.equal(value.requests[0].sessionId, 'session-b')
  assert.deepEqual(value.requests[0].content, [{ type: 'text', text: prepared.ticketText }])
  assert.deepEqual(JSON.parse(await readFile(join(value.runRoot, 'dispatch-receipts', 'B', 'dispatch.json'), 'utf8')), {
    format_version: 1,
    task_id: prepared.taskId,
    ticket_sha256: prepared.taskId,
    target_role: 'B',
    session_id: 'session-b',
    request_id: value.requests[0].requestId,
    status: 'accepted',
  })

  await writeFile(join(value.runRoot, 'b-access-started.json'), '{}')
  assert.deepEqual(await value.dispatcher.dispatchB(prepared.taskId, value.agent), {
    task_id: prepared.taskId,
    status: 'accepted',
    target_role: 'B',
  })
  assert.equal(value.requests.length, 1)
})

test('rejects B dispatch before A freeze and after an unreceipted B access', async t => {
  const unfrozen = await fixture(t)
  await assert.rejects(unfrozen.dispatcher.dispatchB('a'.repeat(64), unfrozen.agent), /ENOENT/)
  assert.equal(unfrozen.requests.length, 0)

  const accessed = await fixture(t)
  const prepared = await frozenBTicket(accessed)
  await writeFile(join(accessed.runRoot, 'b-access-started.json'), '{}')
  await assert.rejects(accessed.dispatcher.dispatchB(prepared.taskId, accessed.agent), /after B access or delivery/)
  assert.equal(accessed.requests.length, 0)
  await assert.rejects(readFile(join(accessed.runRoot, 'dispatch-receipts', 'B', 'dispatch.json')), { code: 'ENOENT' })
})

test('rejects a changed frozen batch before delivering its B ticket', async t => {
  const value = await fixture(t)
  const prepared = await frozenBTicket(value)
  await writeFile(join(value.runRoot, 'frozen_batch.json'), '{"changed":true}\n')

  await assert.rejects(value.dispatcher.dispatchB(prepared.taskId, value.agent), /completed A manifest and frozen batch/)
  assert.equal(value.requests.length, 0)
})

test('continues B delivery only from the accepted ticket and exact completed numerical checkpoint', async t => {
  const value = await fixture(t)
  const prepared = await frozenBTicket(value)
  await assert.rejects(
    value.dispatcher.continueBDelivery(prepared.taskId, 1, value.agent),
    /accepted bound B ticket receipt/,
  )
  assert.equal(value.requests.length, 0)

  await value.dispatcher.dispatchB(prepared.taskId, value.agent)
  await assert.rejects(
    value.dispatcher.continueBDelivery(prepared.taskId, 1, value.agent),
    /completed B numerical checkpoint/,
  )
  assert.equal(value.requests.length, 1)

  const candidateId = Object.keys(prepared.ticket.frozen_batch.candidates)[0]
  await writeFile(join(value.runRoot, 'b-access-started.json'), JSON.stringify({
    candidate_ids: [candidateId],
    time: '2026-10-17T00:00:00Z',
    use: 'fixed_batch_validation',
  }))
  await writeFile(join(value.runRoot, 'b-numerical-complete.json'), JSON.stringify({
    candidate_ids: [candidateId],
    retained_horizons: { [candidateId]: [24] },
  }))

  assert.deepEqual(await value.dispatcher.continueBDelivery(prepared.taskId, 1, value.agent), {
    task_id: prepared.taskId,
    status: 'accepted',
    target_role: 'B',
  })
  assert.equal(value.requests.length, 2)
  assert.equal(value.requests[1].sessionId, 'session-b')
  assert.match(value.requests[1].content[0].text, new RegExp(prepared.taskId))
  assert.match(value.requests[1].content[0].text, /Host continuation 1/)

  assert.deepEqual(await value.dispatcher.continueBDelivery(prepared.taskId, 2, value.agent), {
    task_id: prepared.taskId,
    status: 'accepted',
    target_role: 'B',
  })
  assert.equal(value.requests.length, 3)
  assert.match(value.requests[2].content[0].text, /Host continuation 2/)
  assert.deepEqual(await value.dispatcher.continueBDelivery(prepared.taskId, 2, value.agent), {
    task_id: prepared.taskId,
    status: 'accepted',
    target_role: 'B',
  })
  assert.equal(value.requests.length, 3)

  await writeFile(join(value.runRoot, 'b-delivery.json'), '{}')
  assert.deepEqual(await value.dispatcher.continueBDelivery(prepared.taskId, 2, value.agent), {
    task_id: prepared.taskId,
    status: 'accepted',
    target_role: 'B',
  })
  assert.equal(value.requests.length, 3)
})

test('dispatches a recovery ticket only through its accepted parent A receipt', async t => {
  const value = await fixture(t)
  await value.dispatcher.dispatchA(value.taskId, value.agent)
  const parentPath = join(value.runRoot, 'task-tickets', 'A', `${value.taskId}.json`)
  const parent = JSON.parse(await readFile(parentPath, 'utf8'))
  const receiptPath = join(value.runRoot, 'dispatch-receipts', 'A', `${value.taskId}.json`)
  const receipt = JSON.parse(await readFile(receiptPath, 'utf8'))

  const recoveryBody = { ...parent }
  delete recoveryBody.task_id
  delete recoveryBody.content_sha256
  delete recoveryBody.action_id
  recoveryBody.task_type = 'recovery_a_research'
  recoveryBody.recovery_ref = {
    parent_task_id: value.taskId,
    parent_content_sha256: value.taskId,
    parent_task_type: parent.task_type,
    parent_a_snapshot_sha256: sha256(canonicalJson(parent.a_snapshot)),
    parent_dispatch_receipt_sha256: sha256(canonicalJson(receipt)),
    pending_actions: [{
      action_id: 'evaluate-001',
      name: 'evaluate_candidate',
      parameters_sha256: 'a'.repeat(64),
    }],
  }
  const recoveryTaskId = sha256(canonicalJson(recoveryBody))
  const recoveryTicket = {
    ...recoveryBody,
    task_id: recoveryTaskId,
    content_sha256: recoveryTaskId,
    action_id: `prepare-a-task-${recoveryTaskId}`,
  }
  const recoveryText = `${JSON.stringify(recoveryTicket, null, 2)}\n`
  await writeFile(
    join(value.runRoot, 'task-tickets', 'A', `${recoveryTaskId}.json`), recoveryText,
  )

  assert.deepEqual(await value.dispatcher.dispatchA(recoveryTaskId, value.agent), {
    task_id: recoveryTaskId,
    status: 'accepted',
    target_role: 'A',
  })
  assert.equal(value.requests.length, 2)
  assert.equal(value.requests[1].sessionId, 'session-a')
  assert.deepEqual(value.requests[1].content, [{ type: 'text', text: recoveryText }])
})

test('completes the current native Goal only after direct domain evidence verification', async t => {
  const value = await fixture(t)
  const boundRoot = await realpath(value.runRoot)
  const contractBytes = await readFile(join(value.runRoot, 'contract.json'))
  const contract = JSON.parse(contractBytes.toString('utf8'))
  const domainGoal = JSON.parse(await readFile(join(value.runRoot, 'a_records', 'goal-spec.json'), 'utf8')).data
  const goal = {
    id: 'dsh-goal',
    revision: 4,
    objective: contract.objective,
    phase: 'active',
  }
  let completedRef
  let verificationCalls = 0
  const goals = { get: () => goal }
  const dispatcher = createTicketDispatcher({
    config: {
      runRoot: value.runRoot,
      bindingPath: join(value.runRoot, 'binding.json'),
      pythonExecutable: process.execPath,
      projectRoot: value.runRoot,
      sessionRoles: { coordinator: 'session-coordinator', A: 'session-a', B: 'session-b' },
    },
    sessions: value.sessions,
    agents: { get: () => value.agent, currentInitiator: () => value.agent },
    goals,
    originalPrompt: value.sessions.prompt,
    originalComplete(_agent, ref) {
      completedRef = ref
      goal.phase = 'complete'
      goal.revision = ref.revision + 1
      return { ...goal }
    },
    verifyGoalEvidence() {
      verificationCalls += 1
      return {
        run_root: boundRoot,
        research_id: contract.run_id,
        contract_sha256: sha256(contractBytes),
        goal: domainGoal,
        evidence: {
          status: 'complete', objective_met: true, gaps: [],
          qualified_idea_ids: ['dispatch-fixture--candidate-0001'],
        },
      }
    },
  })
  await dispatcher.initialize()
  const goalRef = { id: goal.id, revision: goal.revision }
  assert.deepEqual(dispatcher.completeGoal(value.agent, { id: goal.id, revision: goal.revision }), {
    ...goal,
    phase: 'complete',
  })
  assert.deepEqual(completedRef, goalRef)
  assert.equal(verificationCalls, 1)
  await assert.rejects(readFile(join(value.runRoot, '.factor-research', 'goal-evidence.key')), { code: 'ENOENT' })
})

test('rejects insufficient evidence and stale Goal refs before native completion', async t => {
  const value = await fixture(t)
  const boundRoot = await realpath(value.runRoot)
  const contractBytes = await readFile(join(value.runRoot, 'contract.json'))
  const contract = JSON.parse(contractBytes.toString('utf8'))
  const domainGoal = JSON.parse(await readFile(join(value.runRoot, 'a_records', 'goal-spec.json'), 'utf8')).data
  const goal = { id: 'dsh-goal', revision: 1, objective: contract.objective, phase: 'active' }
  let completed = 0
  let verificationCalls = 0
  const dispatcher = createTicketDispatcher({
    config: {
      runRoot: value.runRoot,
      bindingPath: join(value.runRoot, 'binding.json'),
      pythonExecutable: process.execPath,
      projectRoot: value.runRoot,
      sessionRoles: { coordinator: 'session-coordinator', A: 'session-a', B: 'session-b' },
    },
    sessions: value.sessions,
    agents: { get: () => value.agent, currentInitiator: () => value.agent },
    goals: { get: () => goal },
    originalPrompt: value.sessions.prompt,
    originalComplete: () => { completed += 1; return goal },
    verifyGoalEvidence() {
      verificationCalls += 1
      return {
        run_root: boundRoot,
        research_id: contract.run_id,
        contract_sha256: sha256(contractBytes),
        goal: domainGoal,
        evidence: { status: 'incomplete', objective_met: false, gaps: ['missing sealed A completion'], qualified_idea_ids: [] },
      }
    },
  })
  await dispatcher.initialize()
  assert.throws(() => dispatcher.completeGoal(value.agent, { id: goal.id, revision: goal.revision }), error =>
    error.code === 'factor-research/evidence-required' && /does not authorize/.test(error.message))
  assert.throws(() => dispatcher.completeGoal(value.agent, { id: goal.id, revision: 2 }), error =>
    error.code === 'factor-research/evidence-required' && /current active Goal id and revision/.test(error.message))
  assert.equal(verificationCalls, 1)
  assert.equal(completed, 0)
})

test('keeps dispatch and Goal completion closed when required Host configuration is missing', async t => {
  const value = await fixture(t)
  const dispatcher = createTicketDispatcher({
    config: { runRoot: value.runRoot },
    sessions: value.sessions,
    agents: { get: () => value.agent, currentInitiator: () => value.agent },
    goals: { get: () => undefined },
    originalPrompt: value.sessions.prompt,
    originalComplete: () => { throw new Error('must not complete') },
  })
  await assert.rejects(dispatcher.initialize(), /sessionRoles must be an object/)
  assert.equal(dispatcher.roleForSession('session-coordinator'), undefined)
  await assert.rejects(dispatcher.dispatchA(value.taskId, value.agent), /active bound Coordinator Agent/)
  await assert.rejects(dispatcher.readyForDomainTools(), /research Host configuration is not ready/)
  assert.throws(() => dispatcher.completeGoal(value.agent, { id: 'dsh-goal', revision: 1 }), /active bound Coordinator Agent/)
})
