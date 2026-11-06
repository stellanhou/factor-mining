import { RemoteError } from '@deepseek-ai/dsh-typert-protocol'
import { lstat, mkdir, readFile, writeFile } from 'node:fs/promises'
import { realpathSync } from 'node:fs'
import { basename, join } from 'node:path'
import { registerHumanChange } from './human-change.js'
import { isResearchToolAllowed, isResearchOnlyTool } from './isolation.js'
import * as NativeGoalDriver from '@deepseek-ai/dsh-goal-round-driver'
import { createWorkspaceScope, contextOutsideWorkspace } from './workspace-scope.js'
import { applyWorkspaceSetup } from './workspace-setup.js'
import {
  createGoalEvidenceVerifier,
  createModelRequestGuard,
  createCompactionRequestGuard,
  createTicketDispatcher,
  openBoundSession,
} from './dispatch.js'

export const name = 'factor-research-admission'
export const inject = ['sessionController', 'goals', 'agents', 'sessions', 'workspaceRegistry', 'sessionProjections', 'commands', 'tools']

function denied(operation, sessionId, reason) {
  const details = { operation }
  if (typeof sessionId === 'string') details.sessionId = sessionId
  throw new RemoteError(
    'factor-research/admission-required',
    `Session ${operation} rejected: ${reason}`,
    details,
  )
}

function goalFailure(error, operation, agent) {
  if (error instanceof RemoteError) throw error
  if (!['factor-research/admission-required', 'factor-research/evidence-required'].includes(error?.code)) {
    throw error
  }
  throw new RemoteError(error.code, error.message, {
    operation,
    ...(typeof agent?.session?.id === 'string' ? { sessionId: agent.session.id } : {}),
  })
}

export function apply(ctx, config) {
  if (config.workspacePolicy === true) {
    ctx.inject(['workspacePluginPolicy'], child => applyResearch(child, config))
    return
  }
  return applyResearch(ctx, config)
}

function applyResearch(ctx, config) {
  const policy = config.workspacePolicy === true ? ctx.get('workspacePluginPolicy') : undefined
  if (policy !== undefined) {
    ctx.effect(() => policy.register({
      id: 'factor-research', label: '因子研究系统',
      description: '固定研究角色、研究工具边界及目标调度。停用后可在此分区新建普通会话，已有研究会话继续保留约束。',
      moduleNames: ['factor-research-dsh-bundle'], workspaceIds: [config.workspaceId], defaultEnabled: true,
      lockedSessionIds: config.setupSessionId === undefined ? Object.values(config.sessionRoles) : [config.setupSessionId],
      createSession: () => ctx.get('sessionController').create({ workspaceId: config.workspaceId }),
    }), 'factor-research.workspace-policy')
  }
  const scope = createWorkspaceScope(ctx.get('workspaceRegistry'), config, policy)
  ctx.plugin({
    name: 'factor-research-other-workspace-goals',
    inject: NativeGoalDriver.inject,
    apply(child) { return NativeGoalDriver.apply(contextOutsideWorkspace(child, scope)) },
  })
  if (config.setupSessionId !== undefined) return applyWorkspaceSetup(ctx, config, scope, denied)
  if (!Number.isSafeInteger(config.stopTimeoutMs) || config.stopTimeoutMs < 1) {
    throw new TypeError('factor-research: stopTimeoutMs must be a finite positive integer')
  }
  const sessions = ctx.get('sessionController')
  const goals = ctx.get('goals')
  const agents = ctx.get('agents')
  const commands = ctx.get('commands')
  const original = {
    create: sessions.create,
    prompt: sessions.prompt,
    selectModel: sessions.selectModel,
    rename: sessions.rename,
    updateQueue: sessions.updateQueue,
    fork: sessions.fork,
    cancel: sessions.cancel,
    goalCreate: goals.create,
    goalEdit: goals.edit,
    goalPause: goals.pause,
    goalResume: goals.resume,
    goalClear: goals.clear,
    complete: goals.complete,
    commandExecute: commands.execute,
  }
  const dispatcher = createTicketDispatcher({
    config,
    sessions,
    agents,
    goals,
    originalPrompt: original.prompt,
    originalCancel: original.cancel,
    originalGoalPause: original.goalPause,
    originalComplete: original.complete,
    verifyGoalEvidence: createGoalEvidenceVerifier(config),
  })
  const modelRequestGuard = createModelRequestGuard({
    roleForSession: dispatcher.roleForSession,
    budgetRoot: config.modelBudget?.budgetRoot ?? config.runRoot,
    maxRequestsPerTurn: config.modelBudget?.maxRequestsPerTurn,
    maxTotalRequests: config.modelBudget?.maxTotalRequests,
    maxOutputTokens: config.modelBudget?.maxOutputTokens,
  })
  const admissionReady = dispatcher.initialize()
  ctx.effect(() => ctx.get('tools').guard(exec => {
    if (!scope.ownsAgent(exec.agent)) return isResearchOnlyTool(exec.name, exec.arguments)
      ? 'research tools belong to the bound research Workspace' : undefined
    const role = dispatcher.roleForSession(exec.agent?.session?.id)
    if (!isResearchToolAllowed(role, exec.name, exec.arguments)) {
      return 'tool is outside the fixed research role boundary'
    }
  }), 'factor-research.tool-boundary')
  function pauseAtBudgetBoundary(error) {
    if (error?.code !== 'factor-research/run-budget-exhausted') return
    dispatcher.suspendResults()
    const coordinator = agents.list().find(item => item.session.id === config.sessionRoles.coordinator)
    const goal = coordinator === undefined ? undefined : goals.get(coordinator)
    if (goal?.phase === 'active') {
      Reflect.apply(original.goalPause, goals, [coordinator, { id: goal.id, revision: goal.revision }])
    }
  }
  const compactionRequestGuard = createCompactionRequestGuard({
    roleForSession: dispatcher.roleForSession,
    requireRunActive: dispatcher.requireRunActive,
    budgetRoot: config.modelBudget?.budgetRoot ?? config.runRoot,
    maxTotalRequests: config.modelBudget.maxTotalRequests,
    maxOutputTokens: config.modelBudget.maxOutputTokens,
  })
  ctx.on('llm/stream', async function* (options, next) {
    if (!scope.ownsSession(options.sessionId)) { yield* next(); return }
    try { yield* compactionRequestGuard(options, next) }
    catch (error) {
      pauseAtBudgetBoundary(error)
      throw error
    }
  })
  ctx.on('webserver/index-inject', table => {
    table.push({ kind: 'global', name: '__FACTOR_RESEARCH_UI__', value: {
      researchId: basename(config.runRoot),
      state: 'bound',
      workspaceId: config.workspaceId,
      sessionRoles: config.sessionRoles,
    } })
  })
  let boundSessionSetup
  const collectResult = (role, seq, options) => {
    void Promise.resolve(boundSessionSetup).then(() => role === 'coordinator'
      ? dispatcher.collectCoordinatorBoundary(seq)
      : role === 'A' ? dispatcher.collectAResult(seq, options) : dispatcher.collectBResult(seq, options)).catch(error => {
      ctx.logger.error(`factor-research: ${role} result collection blocked (${String(error)})`)
    })
  }
  ctx.on('session/event', (session, event) => {
    const role = dispatcher.roleForSession(session.id)
    if (['A', 'B'].includes(role) && event.type === 'turn/end') {
      collectResult(role, event.seq)
    }
    if (role === 'coordinator' && event.type === 'turn/end') {
      collectResult(role, event.seq)
    }
  })
  ctx.on('agent/pre-step', (event, next) => {
    if (dispatcher.roleForSession(event.agent?.session?.id) === 'coordinator'
      && event.messages.some(message => /^factor-(?:[ab]-result|coordinator-boundary)-/.test(message.source?.rpcId ?? ''))
      && !dispatcher.canReceiveResult(event.agent)) return { kind: 'reject' }
    return next()
  })
  ctx.on('agent/request', async (event, next) => {
    if (!scope.ownsAgent(event.agent)) {
      const request = await next()
      if ((request.tools ?? []).some(tool => isResearchOnlyTool(tool.name))) {
        denied('model request', event.agent?.session?.id, 'research tools cannot be used outside the research Workspace')
      }
      return request
    }
    await admissionReady
    if (boundSessionSetup === undefined) throw new Error('factor-research bound Sessions were not initialized')
    await boundSessionSetup
    const role = dispatcher.roleForSession(event?.agent?.session?.id)
    if (role === undefined) denied('model request', event?.agent?.session?.id,
      'only bound research Sessions may issue model requests')
    if (role === 'coordinator') await boundSessionSetup
    if (role !== 'coordinator') await dispatcher.requireRunActive(role)
    try { return await modelRequestGuard(event, async () => {
      const request = await next()
      if ((request.tools ?? []).some(tool => !isResearchToolAllowed(role, tool.name))) {
        denied('model request', event.agent.session.id, 'request includes tools outside its fixed research role')
      }
      return request
    }) } catch (error) {
      pauseAtBudgetBoundary(error)
      throw error
    }
  })
  ctx.on('agent/request-error', (event, next) => scope.ownsAgent(event.agent) ? undefined : next(), { prepend: true })

  sessions.create = function (request) {
    if (!scope.ownsCreate(request)) {
      if (scope.isResearchPreset(request.agentPreset)) denied('create', request.sessionId, 'research presets belong to the research Workspace')
      return Reflect.apply(original.create, this, [request])
    }
    const sessionId = request.sessionId ?? config.sessionRoles.coordinator
    const role = dispatcher.roleForSession(sessionId)
    if (role === undefined) denied('create', sessionId, 'only Sessions bound to this research may be opened')
    const preset = role === 'coordinator' ? 'factor-research-coordinator'
      : role === 'A' ? 'factor-research-a'
        : 'factor-research-b'
    if (typeof config.workspaceId !== 'string' || config.workspaceId.length === 0) {
      denied('create', sessionId, 'the isolated Workspace is not bound in the research profile')
    }
    if (request?.cwd !== undefined || (config.workspacePolicy !== true && request?.workspaceId !== undefined && request.workspaceId !== config.workspaceId)) {
      denied('create', sessionId, 'research Sessions must stay in the bound Workspace')
    }
    if (request?.agentPreset !== undefined && request.agentPreset !== preset) {
      denied('create', sessionId, `the bound Session role requires preset ${preset}`)
    }
    return openBoundSession({ sessions, originalCreate: original.create,
      request: { ...request, sessionId, workspaceId: config.workspaceId, agentPreset: preset },
      presetForAgent: agent => {
        if (realpathSync(agent.session.header.cwd) !== scope.path) {
          denied('create', agent.session.id, 'the saved research Session belongs to a different Workspace folder')
        }
        return ctx.get('sessionProjections').stateOf(agent.session, 'agentPreset')
      },
    })
  }
  boundSessionSetup = admissionReady.then(async () => {
    const opened = []
    const controlDirectory = join(config.runRoot, '.factor-research-control')
    await mkdir(controlDirectory, { recursive: true, mode: 0o700 })
    const bootstrapPath = join(controlDirectory, 'session-bootstrap.json')
    const bootstrap = {
      format_version: 1,
      started_at: new Date().toISOString(),
      status: 'starting',
      requested_sessions: Object.fromEntries(Object.entries(config.sessionRoles).map(([role, sessionId]) => [role, sessionId])),
      opened_sessions: opened,
    }
    const writeBootstrap = () => writeFile(bootstrapPath,
      `${JSON.stringify(bootstrap, null, 2)}\n`, { mode: 0o600 })
    await writeBootstrap()
    try {
      for (const [role, sessionId] of Object.entries(config.sessionRoles)) {
        const result = await sessions.create({ sessionId })
        if (result?.sessionId !== sessionId) {
          denied('create', sessionId, `DSH did not reuse the bound ${role} Session identity`)
        }
        const expectedPreset = role === 'coordinator' ? 'factor-research-coordinator'
          : role === 'A' ? 'factor-research-a' : 'factor-research-b'
        if (result.agentPreset !== expectedPreset) {
          denied('create', sessionId, `DSH did not apply the bound ${role} preset ${expectedPreset}`)
        }
        opened.push({ role, sessionId, agentPreset: result.agentPreset })
        bootstrap.status = 'starting'
        await writeBootstrap()
      }

      const researchLabel = basename(config.runRoot)
      const roleLabels = Object.fromEntries(opened.map(({ role, sessionId }) => [role, {
        thread_id: sessionId,
        title: `${role === 'coordinator' ? '研究协调' : role === 'A' ? 'A 研究' : 'B 验证'} · ${researchLabel}`,
      }]))
      const labelRecord = {
        format_version: 1,
        research_id: researchLabel,
        workspace_id: config.workspaceId,
        sessions: roleLabels,
      }
      const labelPath = join(controlDirectory, 'session-labels.json')
      let existingLabels
      try {
        const info = await lstat(labelPath)
        if (!info.isFile() || info.isSymbolicLink()) {
          throw new Error('research Session label record is not a regular file')
        }
        existingLabels = JSON.parse(await readFile(labelPath, 'utf8'))
      } catch (error) {
        if (error?.code !== 'ENOENT') throw error
      }
      if (existingLabels === undefined) {
        for (const [role, label] of Object.entries(roleLabels)) {
          await Reflect.apply(original.rename, sessions, [{
            sessionId: label.thread_id,
            title: label.title,
          }])
        }
        await writeFile(labelPath, `${JSON.stringify(labelRecord, null, 2)}\n`, {
          flag: 'wx', mode: 0o600,
        })
      } else if (JSON.stringify(existingLabels) !== JSON.stringify(labelRecord)) {
        throw new Error('saved research Session labels differ from the fixed role binding')
      }
      bootstrap.status = 'ready'
      bootstrap.labels = roleLabels
      await writeBootstrap()
      ctx.logger.info(`factor-research: bound Web Sessions initialized (${opened.length})`)
      return opened
    } catch (error) {
      bootstrap.status = 'failed'
      bootstrap.error = { code: error?.code ?? null, message: String(error) }
      await writeBootstrap()
      throw error
    }
  })
  void boundSessionSetup.catch(error => {
    ctx.logger.error(`factor-research: bound Web Session initialization failed (${String(error)})`)
  })

  sessions.prompt = function (request, signal) {
    if (!scope.ownsSession(request.sessionId)) return Reflect.apply(original.prompt, this, [request, signal])
    const role = dispatcher.roleForSession(request.sessionId)
    if (role === undefined) denied('prompt', request.sessionId, 'Session is not bound to this research')
    if (role !== 'coordinator') denied('prompt', request.sessionId, 'A and B accept only validated research dispatches')
    return Promise.resolve(boundSessionSetup).then(
      () => Reflect.apply(original.prompt, this, [request, signal]),
    )
  }
  commands.execute = function (agent, line, ...args) {
    if (!scope.ownsAgent(agent)) return Reflect.apply(original.commandExecute, this, [agent, line, ...args])
    const name = /^\/([a-z0-9_-]+)(?:\s|$)/.exec(line)?.[1]
    if (!dispatcher.isCoordinator(agent?.session?.id)
      || !['compact', 'research-change'].includes(name)) {
      denied('command', agent?.session?.id, 'only bound Coordinator research controls are available')
    }
    return Reflect.apply(original.commandExecute, this, [agent, line, ...args])
  }
  ctx.effect(() => registerHumanChange(ctx, { config, dispatcher, goals,
    originalEdit: original.goalEdit }), 'factor-research.human-change')
  sessions.selectModel = function (request) {
    if (!scope.ownsSession(request.sessionId)) return Reflect.apply(original.selectModel, this, [request])
    const role = dispatcher.roleForSession(request.sessionId)
    if (role !== 'coordinator') denied('selectModel', request.sessionId, 'only the coordinator Session may select a model')
    return Reflect.apply(original.selectModel, this, [request])
  }
  sessions.rename = function (request) {
    if (!scope.ownsSession(request.sessionId)) return Reflect.apply(original.rename, this, [request])
    const role = dispatcher.roleForSession(request.sessionId)
    if (role !== 'coordinator') denied('rename', request.sessionId, 'A and B Sessions are read-only')
    return Reflect.apply(original.rename, this, [request])
  }
  sessions.updateQueue = function (request) {
    if (!scope.ownsSession(request.sessionId)) return Reflect.apply(original.updateQueue, this, [request])
    const role = dispatcher.roleForSession(request.sessionId)
    if (role !== 'coordinator') denied('updateQueue', request.sessionId, 'only the coordinator Session may be edited directly')
    return Reflect.apply(original.updateQueue, this, [request])
  }
  sessions.fork = function (request) {
    if (!scope.ownsSession(request.sessionId)) return Reflect.apply(original.fork, this, [request])
    denied('fork', request.sessionId, 'Session forks are disabled in the controlled research profile')
  }
  sessions.cancel = function (request) {
    if (!scope.ownsSession(request.sessionId)) return Reflect.apply(original.cancel, this, [request])
    const role = dispatcher.roleForSession(request.sessionId)
    if (role !== 'coordinator') denied('cancel', request.sessionId, 'only the coordinator Session may be cancelled directly')
    return dispatcher.stopRun(request.sessionId)
  }
  goals.create = function (agent, request) {
    if (!scope.ownsAgent(agent)) return Reflect.apply(original.goalCreate, this, [agent, request])
    try {
      return dispatcher.createGoal(agent, request, original.goalCreate)
    } catch (error) {
      goalFailure(error, 'create', agent)
    }
  }
  goals.edit = function (agent, ref, request) {
    if (!scope.ownsAgent(agent)) return Reflect.apply(original.goalEdit, this, [agent, ref, request])
    try {
      return dispatcher.rejectGoalEdit(agent, ref, request)
    } catch (error) {
      goalFailure(error, 'edit', agent)
    }
  }
  goals.pause = function (agent, ref) {
    if (!scope.ownsAgent(agent)) return Reflect.apply(original.goalPause, this, [agent, ref])
    try {
      dispatcher.requireCoordinatorGoalAgent(agent, 'Goal pause')
      const result = Reflect.apply(original.goalPause, this, [agent, ref])
      dispatcher.suspendResults()
      return result
    } catch (error) {
      goalFailure(error, 'pause', agent)
    }
  }
  goals.resume = async function (agent, ref) {
    if (!scope.ownsAgent(agent)) return Reflect.apply(original.goalResume, this, [agent, ref])
    try {
      dispatcher.requireCoordinatorGoalAgent(agent, 'Goal resume')
      await dispatcher.resumeRun(agent, ref)
      const result = await Reflect.apply(original.goalResume, this, [agent, ref])
      collectResult('A', undefined, { explicitResume: true })
      collectResult('B', undefined, { explicitResume: true })
      collectResult('coordinator')
      return result
    } catch (error) {
      goalFailure(error, 'resume', agent)
    }
  }
  goals.clear = function (agent, ref) {
    if (!scope.ownsAgent(agent)) return Reflect.apply(original.goalClear, this, [agent, ref])
    try {
      return dispatcher.rejectGoalClear(agent, ref)
    } catch (error) {
      goalFailure(error, 'clear', agent)
    }
  }
  goals.complete = function (agent, ref) {
    if (!scope.ownsAgent(agent)) return Reflect.apply(original.complete, this, [agent, ref])
    try {
      return dispatcher.completeGoal(agent, ref)
    } catch (error) {
      if (error?.code !== 'factor-research/evidence-required') throw error
      throw new RemoteError(error.code, error.message, {
        ...(typeof agent?.session?.id === 'string' ? { sessionId: agent.session.id } : {}),
        ...(typeof ref?.id === 'string' ? { goalId: ref.id } : {}),
        ...(Number.isSafeInteger(ref?.revision) ? { revision: ref.revision } : {}),
      })
    }
  }

  ctx.effect(() => async () => {
    if (boundSessionSetup !== undefined) await boundSessionSetup.catch(() => undefined)
    try { await dispatcher.dispose() }
    finally {
    sessions.create = original.create
    sessions.prompt = original.prompt
    sessions.selectModel = original.selectModel
    sessions.rename = original.rename
    sessions.updateQueue = original.updateQueue
    sessions.fork = original.fork
    sessions.cancel = original.cancel
    goals.create = original.goalCreate
    goals.edit = original.goalEdit
    goals.pause = original.goalPause
    goals.resume = original.goalResume
    goals.clear = original.goalClear
    goals.complete = original.complete
    commands.execute = original.commandExecute
    }
  }, 'factor-research.admission')
  dispatcher.domainBinding = () => ({ bindingPath: config.bindingPath,
    pythonExecutable: config.pythonExecutable, projectRoot: config.projectRoot })
  ctx.effect(() => ctx.provide('factorResearchAdmission', dispatcher), 'factor-research.dispatch')
  ctx.logger.info('factor-research: Session and Goal admission gates active')
  void dispatcher.initialize().then(
    () => ctx.logger.info('factor-research: evidence verifier initialized'),
    error => ctx.logger.error(`factor-research: evidence verifier unavailable; Goal completion remains denied (${String(error)})`),
  )
}
