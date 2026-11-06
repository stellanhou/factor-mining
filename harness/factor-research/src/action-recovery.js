import { defineTool } from '@deepseek-ai/dsh-tools'
export const name = 'factor-research-a-action-recovery'
export const inject = ['tools', 'factorResearchAdmission']
export function apply(ctx) {
  const admission = ctx.get('factorResearchAdmission')
  const dispose = ctx.tools.register(defineTool({
    name: 'read_a_action_request',
    description: 'Recover exact original parameters for one saved A action ID from this bound A Session, verified against the domain receipt hash. Use before replaying pending actions; never guess or change their parameters.',
    parameters: { action_id: { type: 'string', required: true } },
    output: { schema: { type: 'object', additionalProperties: true },
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }] },
    execute(args, exec) { return admission.readAActionRequest(args.action_id, exec.agent) },
  }))
  ctx.effect(() => dispose, 'factor-research.a-action-recovery')
}
