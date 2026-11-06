import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'factor-research-run-control'
export const inject = ['tools', 'factorResearchAdmission']

const STATUS_OUTPUT = {
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      research_id: { type: 'string', required: true },
      state: { type: 'string', required: true, enum: ['active', 'stopping', 'stopped'] },
      program_actions: {
        type: 'object', required: true, additionalProperties: false,
        properties: {
          status: { type: 'string', required: true, enum: ['active', 'stopping', 'stopped'] },
          source: { type: 'string', required: true },
          event_id: { type: 'string' },
          action_id: { type: 'string' },
          reason: { type: 'string' },
        },
      },
      sessions: {
        type: 'object', required: true, additionalProperties: false,
        properties: {
          coordinator: {
            type: 'object', required: true, additionalProperties: false,
            properties: {
              session_id: { type: 'string', required: true },
              status: { type: 'string', required: true },
            },
          },
          A: {
            type: 'object', required: true, additionalProperties: false,
            properties: {
              session_id: { type: 'string', required: true },
              status: { type: 'string', required: true },
            },
          },
          B: {
            type: 'object', required: true, additionalProperties: false,
            properties: {
              session_id: { type: 'string', required: true },
              status: { type: 'string', required: true },
            },
          },
        },
      },
      in_flight_operations: { type: 'array', required: true, items: { type: 'string' } },
    },
  },
  render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
}

export function apply(ctx) {
  const admission = ctx.get('factorResearchAdmission')
  const dispose = ctx.tools.register(defineTool({
    name: 'get_research_runtime_status',
    description: 'Read live status for the three fixed Sessions and the durable research stop state. Use with get_research_status to explain domain progress.',
    parameters: {},
    output: STATUS_OUTPUT,
    execute(_args, exec) {
      return admission.runtimeStatus(exec.agent)
    },
  }))
  ctx.effect(() => dispose, 'factor-research.runtime-status')
}
