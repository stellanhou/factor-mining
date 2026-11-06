import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'factor-research-dispatch-a'
export const inject = ['tools', 'factorResearchAdmission']

const DISPATCH_OUTPUT = {
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      task_id: { type: 'string', required: true },
      status: { type: 'string', required: true, enum: ['accepted'] },
      target_role: { type: 'string', required: true, enum: ['A'] },
    },
  },
  render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
}

export function apply(ctx) {
  const admission = ctx.get('factorResearchAdmission')

  const dispose = ctx.tools.register(defineTool({
    name: 'dispatch_a_ticket',
    description: 'Deliver one already-saved A research ticket by task ID. The A Session and full message are fixed by the research Host.',
    parameters: {
      task_id: {
        type: 'string',
        required: true,
        description: 'The SHA-256 task_id returned by prepare_a_task.',
      },
    },
    output: DISPATCH_OUTPUT,
    execute(args, exec) {
      return admission.dispatchA(args.task_id, exec.agent, exec.signal)
    },
  }))
  ctx.effect(() => dispose, 'factor-research.dispatch-a-ticket')

  const disposeResume = ctx.tools.register(defineTool({
    name: 'resume_a_task',
    description: 'Resume an accepted A task only when the Host verifies that the saved A snapshot is unchanged, the A Session is idle, and no unreceipted message exists. Sends a fixed continuation to the bound A Session.',
    parameters: {
      task_id: {
        type: 'string',
        required: true,
        description: 'The SHA-256 task_id of the already accepted A ticket.',
      },
      attempt: {
        type: 'number',
        required: true,
        enum: [1, 2, 3],
        description: 'The next Host continuation attempt, from 1 through 3.',
      },
    },
    output: DISPATCH_OUTPUT,
    execute(args, exec) {
      return admission.continueATask(args.task_id, args.attempt, exec.agent, exec.signal)
    },
  }))
  ctx.effect(() => disposeResume, 'factor-research.resume-a-task')
}
