import { openBoundSession } from './dispatch.js'
import { realpathSync } from 'node:fs'
import { isResearchOnlyTool } from './isolation.js'

// A newly created Workspace has no approved run yet. It admits one planning
// conversation, with no research tools, Goal or A/B calculation authority.
export function applyWorkspaceSetup(ctx, config, scope, denied) {
  if (typeof config.setupSessionId !== 'string' || !config.setupSessionId.startsWith('session-')) {
    throw new TypeError('factor-research: setupSessionId must name the planning Session')
  }
  if (config.bindingPath !== undefined || Object.keys(config.sessionRoles ?? {}).length !== 0) {
    throw new TypeError('factor-research: planning Workspace cannot also bind a research run')
  }
  const sessions = ctx.get('sessionController')
  const goals = ctx.get('goals')
  const originalCreate = sessions.create
  const originals = []
  function wrap(service, method, replacement) {
    const original = service[method]
    originals.push({ method, service, original })
    service[method] = replacement(original)
  }
  const request = { sessionId: config.setupSessionId, workspaceId: config.workspaceId, agentPreset: 'factor-research' }
  const open = async original => {
    const resolved = await sessions.resolveAgent(config.setupSessionId)
    if (resolved.agent !== undefined) {
      if (realpathSync(resolved.agent.session.header.cwd) !== scope.path) {
        denied('create', config.setupSessionId, 'the planning Session belongs to a different Workspace folder')
      }
      const projections = ctx.get('sessionProjections')
      const preset = projections.stateOf(resolved.agent.session, 'agentPreset')
      if (preset !== 'factor-research') {
        const boundary = projections.stateOf(resolved.agent.session, 'turnBoundary')
        if (boundary?.lastTurn !== 0 || boundary.openTurnStartSeq !== null) {
          denied('create', config.setupSessionId, 'the planning Session must be blank before its preset changes')
        }
        const result = await Reflect.apply(original, sessions, [request])
        if (result.sessionId !== config.setupSessionId) denied('create', result.sessionId, 'planning must retain its registered Session identity')
        return result
      }
    }
    return openBoundSession({ sessions, originalCreate: original, request,
      presetForAgent: agent => ctx.get('sessionProjections').stateOf(agent.session, 'agentPreset'),
    })
  }
  wrap(sessions, 'create', original => function (request) {
    if (scope.ownsCreate(request)) {
      if (request.cwd !== undefined || (config.workspacePolicy !== true && request.workspaceId !== undefined && request.workspaceId !== config.workspaceId)) {
        denied('create', request.sessionId, 'research planning must stay in its registered Workspace')
      }
      if (request.sessionId !== undefined && request.sessionId !== config.setupSessionId) {
        denied('create', request.sessionId, 'this Workspace is awaiting an approved research binding')
      }
      if (request.agentPreset !== undefined && request.agentPreset !== 'factor-research') {
        denied('create', request.sessionId, 'this Workspace admits only its research planning preset')
      }
      return open(original)
    }
    if (scope.isResearchPreset(request.agentPreset)) denied('create', request.sessionId, 'research presets belong to the research Workspace')
    return Reflect.apply(original, this, [request])
  })
  wrap(sessions, 'fork', original => function (request) {
    if (scope.ownsSession(request.sessionId)) denied('fork', request.sessionId, 'bind an approved run before creating research branches')
    return Reflect.apply(original, this, [request])
  })
  wrap(ctx.get('commands'), 'execute', original => function (agent, line, ...args) {
    if (scope.ownsAgent(agent) && !/^\/compact(?:\s|$)/.test(line)) {
      denied('command', agent.session.id, 'research planning has no execution commands')
    }
    return Reflect.apply(original, this, [agent, line, ...args])
  })
  for (const method of ['create', 'edit', 'pause', 'resume', 'clear', 'complete']) {
    wrap(goals, method, original => function (agent, ...args) {
      if (scope.ownsAgent(agent)) denied(`Goal ${method}`, agent.session.id, 'bind an approved research objective and inputs first')
      return Reflect.apply(original, this, [agent, ...args])
    })
  }
  ctx.effect(() => ctx.get('tools').guard(exec => {
    if (scope.ownsAgent(exec.agent)) return 'research planning has no calculation or filesystem tools before run binding'
    if (isResearchOnlyTool(exec.name, exec.arguments)) return 'research tools require the bound research Workspace'
  }), 'factor-research.setup-tools')
  ctx.on('agent/request', async (event, next) => {
    if (!scope.ownsAgent(event.agent)) {
      const request = await next()
      if ((request.tools ?? []).some(tool => isResearchOnlyTool(tool.name))) {
        denied('model request', event.agent?.session?.id, 'research tools require an approved run binding')
      }
      return request
    }
    if (event.agent.session.id !== config.setupSessionId) denied('model request', event.agent.session.id, 'only the research planning Session is admitted')
    const request = await next()
    if ((request.tools ?? []).length !== 0) denied('model request', config.setupSessionId, 'research planning must not receive execution tools')
    return request
  })
  ctx.on('agent/request-error', (event, next) => scope.ownsAgent(event.agent) ? undefined : next(), { prepend: true })
  ctx.on('webserver/index-inject', table => table.push({ kind: 'global', name: '__FACTOR_RESEARCH_UI__', value: {
    state: 'setup', workspaceId: config.workspaceId, researchId: '待绑定研究',
    sessionRoles: { coordinator: config.setupSessionId },
  } }))
  ctx.effect(() => () => {
    for (const { method, service, original } of originals) service[method] = original
  }, 'factor-research.setup')
  const ready = open(originalCreate)
  void ready.then(() => sessions.rename({ sessionId: config.setupSessionId, title: '研究协调 · 因子挖掘' }))
    .catch(error => ctx.logger.error(`factor-research: planning Session initialization failed (${String(error)})`))
}
