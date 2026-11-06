import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'factor-research-dispatch-b'
export const inject = ['tools', 'factorResearchAdmission']

const DISPATCH_OUTPUT = {
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      task_id: { type: 'string', required: true },
      status: { type: 'string', required: true, enum: ['accepted'] },
      target_role: { type: 'string', required: true, enum: ['B'] },
    },
  },
  render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
}

export function apply(ctx) {
  const admission = ctx.get('factorResearchAdmission')

  const dispose = ctx.tools.register(defineTool({
    name: 'dispatch_b_ticket',
    description: 'Deliver one saved B validation ticket after A is sealed and frozen. The B Session and full message are fixed by the research Host.',
    parameters: {
      task_id: {
        type: 'string',
        required: true,
        description: 'The SHA-256 task_id returned by prepare_b_task.',
      },
    },
    output: DISPATCH_OUTPUT,
    execute(args, exec) {
      return admission.dispatchB(args.task_id, exec.agent, exec.signal)
    },
  }))
  const disposeContinuation = ctx.tools.register(defineTool({
    name: 'resume_b_delivery',
    description: 'Resume the bound B Session only after the Host verifies a complete checkpoint for the accepted frozen ticket. Sends a fixed continuation message and never opens B market data again.',
    parameters: {
      task_id: {
        type: 'string',
        required: true,
        description: 'The SHA-256 task_id returned by prepare_b_task.',
      },
      attempt: {
        type: 'number',
        required: true,
        enum: [1, 2, 3],
        description: 'The next numbered Host continuation attempt, starting at 1.',
      },
    },
    output: DISPATCH_OUTPUT,
    execute(args, exec) {
      return admission.continueBDelivery(args.task_id, args.attempt, exec.agent, exec.signal)
    },
  }))
  ctx.effect(() => () => {
    dispose()
    disposeContinuation()
  }, 'factor-research.dispatch-b-ticket')
}
