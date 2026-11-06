import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { existsSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

const execute = promisify(execFile)

async function saveExact(path, value) {
  const body = `${JSON.stringify(value, null, 2)}\n`
  try { await writeFile(path, body, { flag: 'wx', mode: 0o600 }) }
  catch (error) {
    if (error.code !== 'EEXIST') throw error
    if (JSON.stringify(JSON.parse(await readFile(path, 'utf8'))) !== JSON.stringify(value)) {
      throw new Error('saved human change differs from this approval')
    }
  }
}

// This handler is registered only in the native human command plane. It is not
// a model tool and accepts no model-carried identity or arbitrary source path.
export function registerHumanChange(ctx, { config, dispatcher, goals, originalEdit }) {
  return ctx.commands.register({
    name: 'research-change',
    description: 'Approve a versioned research change before A is sealed',
    input: { hint: '<approved change JSON>' },
    async handler({ agent, rawInput, signal, commandId }) {
      dispatcher.requireCoordinatorGoalAgent(agent, 'Human research change')
      const approval = JSON.parse(rawInput)
      if (approval === null || Array.isArray(approval) || typeof approval !== 'object') {
        throw new TypeError('research change approval must be a JSON object')
      }
      const hash = createHash('sha256').update(JSON.stringify(approval)).digest('hex')
      const supersededPath = join(config.runRoot, '.factor-research-control', 'superseded-by.json')
      if (existsSync(supersededPath)) {
        const prior = JSON.parse(await readFile(supersededPath, 'utf8'))
        if (prior.approval_sha256 !== hash) throw new Error('the parent research version has already been superseded')
        return { kind: 'success', text: `此变更已批准：研究版本 ${prior.target.version}（${prior.child_research_id}）。旧版本保持停止。` }
      }
      const directory = join(config.runRoot, '.factor-research-control', 'human-changes')
      await mkdir(directory, { recursive: true, mode: 0o700 })
      const approvalPath = join(directory, `${hash}.approval.json`)
      await saveExact(approvalPath, approval)
      signal.throwIfAborted()
      const stopped = await dispatcher.stopRun(agent.session.id)
      if (stopped.status !== 'stopped') throw new Error('research must finish stopping before its contract can change')
      signal.throwIfAborted()
      const { stdout } = await execute(config.pythonExecutable, [
        join(config.projectRoot, 'scripts', 'record_native_research_change.py'),
        '--binding', config.bindingPath, '--approval', approvalPath,
      ], { cwd: config.projectRoot, env: { ...process.env, PYTHONPATH: join(config.projectRoot, 'src') },
        signal, maxBuffer: 1024 * 1024 })
      const change = JSON.parse(stdout)
      const child = JSON.parse(await readFile(change.child_binding_path, 'utf8'))
      if (child.research_id !== change.child_research_id
        || child.target.objective !== approval.new_objective || child.target.version < 2) {
        throw new Error('approved child research binding is inconsistent')
      }
      let goal = goals.get(agent)
      if (goal !== undefined && goal.objective !== child.target.objective) {
        if (goal.phase !== 'paused') throw new Error('the original Goal must be paused before its approved edit')
        goal = Reflect.apply(originalEdit, goals, [agent, { id: goal.id, revision: goal.revision },
          { objective: child.target.objective }])
      }
      const childConfig = { ...config, runRoot: change.child_run_root,
        bindingPath: change.child_binding_path,
        modelBudget: { ...config.modelBudget, budgetRoot: config.modelBudget.budgetRoot ?? config.runRoot } }
      const patchPath = join(directory, `${hash}.child.patch.yml`)
      await saveExact(patchPath, [{ id: 'factor-research-admission',
        name: 'factor-research-dsh-bundle', config: childConfig }])
      const receipt = { format_version: 1, approval_sha256: hash,
        parent_research_id: change.parent_research_id, child_research_id: change.child_research_id,
        child_binding_path: change.child_binding_path, child_run_root: change.child_run_root,
        child_patch_path: patchPath, target: child.target,
        native_goal: goal === undefined ? null : { id: goal.id, revision: goal.revision, phase: goal.phase } }
      await saveExact(join(directory, `${hash}.transition.json`), receipt)
      await saveExact(supersededPath, receipt)
      // The command lifecycle is the durable human authorization. It remains
      // outside A/B model history; child A receives the validated A-only ticket.
      await saveExact(join(directory, `${hash}.command.json`), { command_id: commandId, approval_sha256: hash })
      return { kind: 'success', text: `已批准研究版本 ${child.target.version}：${child.target.objective}\n旧版本已停止并保留，新版本 ${change.child_research_id} 已准备好。重新打开新版本后，显式恢复原 Goal 继续。` }
    },
  })
}
