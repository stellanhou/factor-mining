const resources = new Set(['list_mcp_resources', 'list_mcp_resource_templates', 'read_mcp_resource'])
const coordinatorControls = new Set(['create_goal', 'get_goal', 'update_goal', 'dispatch_a_ticket',
  'resume_a_task', 'dispatch_b_ticket', 'resume_b_delivery', 'get_research_runtime_status'])

export function isResearchOnlyTool(name, args) {
  return typeof name === 'string' && (name.startsWith('mcp__fr_')
    || (resources.has(name) && /^fr_(a|b|coordinator)$/.test(args?.server ?? ''))
    || ['dispatch_a_ticket', 'resume_a_task', 'dispatch_b_ticket', 'resume_b_delivery',
      'get_research_runtime_status', 'read_a_action_request'].includes(name))
}

export function isResearchToolAllowed(role, name, args) {
  if (!['coordinator', 'A', 'B'].includes(role) || typeof name !== 'string') return false
  if (resources.has(name)) {
    const server = args?.server
    return server === undefined || server === `fr_${role.toLowerCase()}`
  }
  if (name.startsWith(`mcp__fr_${role.toLowerCase()}__`)) return true
  if (role === 'A' && name === 'read_a_action_request') return true
  return role === 'coordinator' && coordinatorControls.has(name)
}
