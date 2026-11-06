import { existsSync, realpathSync } from 'node:fs'

export function createWorkspaceScope(registry, config, policy) {
  const workspace = registry.get(config.workspaceId)
  if (workspace === undefined) throw new Error('factor-research: configured Workspace is not registered')
  const path = realpathSync(workspace.path)
  if (config.workspacePath !== undefined && realpathSync(config.workspacePath) !== path) {
    throw new Error('factor-research: Workspace path differs from its configured folder')
  }
  const fixedIds = new Set(config.setupSessionId === undefined
    ? Object.values(config.sessionRoles) : [config.setupSessionId])
  const ownsPath = candidate => typeof candidate === 'string' && existsSync(candidate)
    && realpathSync(candidate) === path
  const scoped = (id, session) => {
    if (policy === undefined) return registry.get(config.workspaceId)?.sessionIds.includes(id) === true || ownsPath(session?.header?.cwd)
    const workspace = policy.workspaceForSession(id, session)
    if (workspace === undefined && policy.sections.sectionFor(undefined, id) === undefined) return false
    return policy.enabled('factor-research', workspace?.id, id)
  }
  const ownsSession = (id, session) => fixedIds.has(id)
    || scoped(id, session)
  return {
    path,
    ownsSession,
    ownsAgent: agent => ownsSession(agent?.session?.id, agent?.session),
    ownsCreate: request => {
      if (ownsSession(request?.sessionId)) return true
      if (policy === undefined) return request?.workspaceId === config.workspaceId || ownsPath(request?.cwd)
      const workspaceId = request?.workspaceId ?? policy.registry.list().find(w => w.path === request?.cwd)?.id
      return workspaceId !== undefined && policy.enabled('factor-research', workspaceId, request?.sessionId)
    },
    isResearchPreset: preset => typeof preset === 'string' && preset.startsWith('factor-research'),
  }
}

// Keep the upstream native Goal driver intact for all other Workspaces. Its
// listeners and initial Agent roster must both exclude the controlled research.
export function contextOutsideWorkspace(ctx, scope) {
  const agents = new Proxy(ctx.get('agents'), {
    get(target, key) {
      if (key === 'list') return () => target.list().filter(agent => !scope.ownsAgent(agent))
      const value = Reflect.get(target, key, target)
      return typeof value === 'function' ? value.bind(target) : value
    },
  })
  return ctx.extend({
    agents,
    on(name, listener, options) {
      return ctx.on(name, function (...args) {
        const source = args[0]
        if (scope.ownsAgent(source?.agent) || scope.ownsSession(source?.id, source)) {
          const next = args.at(-1)
          return typeof next === 'function' ? next() : undefined
        }
        return listener.apply(this, args)
      }, options)
    },
  })
}
