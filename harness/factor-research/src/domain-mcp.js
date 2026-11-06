import { isAbsolute, join } from 'node:path'
import * as McpClient from '@deepseek-ai/dsh-mcp-client'

export const name = 'factor-research-domain-mcp'
export const inject = ['factorResearchAdmission']

export async function apply(ctx, config) {
  const admission = ctx.get('factorResearchAdmission')
  const role = config.role
  if (!['coordinator', 'A', 'B'].includes(role)) {
    throw new TypeError('factor-research: domain MCP role must be fixed by its Agent preset')
  }
  await admission.readyForDomainTools()
  const binding = admission.domainBinding()

  for (const [key, value] of Object.entries({
    bindingPath: binding.bindingPath,
    pythonExecutable: binding.pythonExecutable,
    projectRoot: binding.projectRoot,
  })) {
    if (typeof value !== 'string' || !isAbsolute(value)) {
      throw new TypeError(`factor-research: ${key} must be an absolute profile binding`)
    }
  }

  const serverName = `fr_${role.toLowerCase()}`
  await ctx.plugin(McpClient, {
    transport: 'stdio',
    serverName,
    toolCallTimeoutMs: 300000,
    command: binding.pythonExecutable,
    args: [
      '-m',
      'crypto_quant.research.factor_mining.mcp_server',
      '--role',
      role,
      '--binding',
      binding.bindingPath,
    ],
    env: { PYTHONPATH: join(binding.projectRoot, 'src') },
    cwd: binding.projectRoot,
    failOnStartupError: true,
  })

  ctx.logger.info(`factor-research: ${role} domain MCP tools connected`)
}
