import { spawnSync } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import { lstat, mkdir, open, readFile, readdir, realpath, rename, writeFile } from 'node:fs/promises'
import { lstatSync, readFileSync, readdirSync, realpathSync } from 'node:fs'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { isResearchToolAllowed } from './isolation.js'

function reject(message, code = 'factor-research/ticket-invalid') {
  const error = new Error(message)
  error.code = code
  throw error
}

export async function openBoundSession({ sessions, originalCreate, request, presetForAgent }) {
  const resolved = await sessions.resolveAgent(request.sessionId)
  if (resolved.error !== undefined) {
    if (resolved.error.code !== 'session/not-found') throw resolved.error
    return Reflect.apply(originalCreate, sessions, [request])
  }
  const agent = resolved.agent
  const agentPreset = presetForAgent(agent)
  if (agent?.session?.id !== request.sessionId || agentPreset !== request.agentPreset) {
    reject('persisted Session identity or role preset differs from its fixed binding',
      'factor-research/admission-required')
  }
  return { sessionId: agent.session.id, agentPreset }
}

function canonicalJson(value, numberSources, parent, key, omittedRootKeys) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') {
    return JSON.stringify(value)
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) reject('ticket contains a non-finite number')
    return numberSources?.get(parent)?.get(key) ?? JSON.stringify(value)
  }
  if (Array.isArray(value)) {
    return `[${value.map((item, index) => canonicalJson(item, numberSources, value, String(index))).join(',')}]`
  }
  if (typeof value !== 'object') reject('ticket contains an unsupported value')
  const keys = Object.keys(value).filter(item => parent !== undefined || !omittedRootKeys?.has(item)).sort()
  return `{${keys.map(item => `${JSON.stringify(item)}:${canonicalJson(value[item], numberSources, value, item)}`).join(',')}}`
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function within(root, target) {
  const path = relative(root, target)
  return path === '' || (path !== '..' && !path.startsWith(`..${sep}`) && !isAbsolute(path))
}

function sameJson(left, right) {
  return canonicalJson(left) === canonicalJson(right)
}

function parseTicket(text) {
  const numberSources = new WeakMap()
  const ticket = JSON.parse(text, function (key, value, context) {
    if (typeof value === 'number' && typeof context?.source === 'string') {
      let entries = numberSources.get(this)
      if (entries === undefined) {
        entries = new Map()
        numberSources.set(this, entries)
      }
      entries.set(key, context.source)
    }
    return value
  })
  return { ticket, numberSources }
}

async function readChangeReference(root, ticket, contract, contractBytes) {
  const recordPath = join(root, 'change-record.json')
  const hasRecord = await exists(recordPath)
  if (!hasRecord) {
    if (ticket.change_ref !== undefined) reject('A change ticket has no saved human change record')
    return
  }
  if (ticket.change_ref === undefined) reject('child A ticket omits its human change record reference')
  const info = await lstat(recordPath)
  if (!info.isFile() || info.isSymbolicLink()) reject('saved human change record is not a regular file')
  const actualPath = await realpath(recordPath)
  if (!within(root, actualPath) || actualPath !== recordPath) {
    reject('saved human change record escaped the bound run root')
  }
  const { ticket: record, numberSources } = parseTicket(await readFile(actualPath, 'utf8'))
  const recordKeys = [
    'format_version', 'record_id', 'content_sha256', 'parent', 'child',
    'change', 'approval_source', 'approval_sha256', 'recorded_at',
  ].sort()
  if (typeof record !== 'object' || record === null || Array.isArray(record)
    || !sameJson(Object.keys(record).sort(), recordKeys)
    || record.format_version !== 1 || record.approval_source !== 'local_operator_cli'
    || !/^[a-f0-9]{64}$/.test(record.approval_sha256 ?? '')
    || typeof record.recorded_at !== 'string') {
    reject('saved human change record is malformed')
  }
  const bodySha = sha256(canonicalJson(
    record, numberSources, undefined, undefined, new Set(['record_id', 'content_sha256']),
  ))
  if (record.content_sha256 !== bodySha) reject('saved human change record content hash is invalid')
  const parent = record.parent
  const child = record.child
  const parentKeys = [
    'research_id', 'target', 'contract_sha256', 'a_snapshot',
    'a_snapshot_sha256', 'evidence_refs', 'task_refs', 'action_refs',
  ].sort()
  const childKeys = ['research_id', 'target', 'contract', 'contract_sha256', 'goal'].sort()
  if (typeof parent !== 'object' || parent === null || Array.isArray(parent)
    || !sameJson(Object.keys(parent).sort(), parentKeys)
    || typeof child !== 'object' || child === null || Array.isArray(child)
    || !sameJson(Object.keys(child).sort(), childKeys)
    || typeof record.change !== 'object' || record.change === null || Array.isArray(record.change)
    || !sameJson(Object.keys(record.change).sort(), [
      'adopted_change', 'contract_changes', 'goal_changes', 'new_objective', 'original_intent',
    ])) {
    reject('saved human change record has an invalid parent, child, or change schema')
  }
  if (typeof parent.research_id !== 'string' || parent.research_id.length === 0
    || parent.research_id === contract.run_id
    || !Number.isSafeInteger(parent.target?.version) || parent.target.version < 1
    || !/^[a-f0-9]{64}$/.test(parent.contract_sha256 ?? '')
    || !/^[a-f0-9]{64}$/.test(parent.a_snapshot_sha256 ?? '')
    || typeof parent.a_snapshot !== 'object' || parent.a_snapshot === null
    || Array.isArray(parent.a_snapshot)
    || !Array.isArray(parent.task_refs) || !Array.isArray(parent.evidence_refs)
    || !Array.isArray(parent.action_refs)
    || child.research_id !== contract.run_id
    || child.contract_sha256 !== sha256(contractBytes)
    || !sameJson(child.contract, contract)
    || !sameJson(child.target, ticket.target)
    || !sameJson(child.goal, ticket.goal)
    || !Number.isSafeInteger(child.target?.version)
    || child.target.version !== parent.target.version + 1
    || record.record_id !== `change-${contract.run_id}-v${child.target.version}`) {
    reject('saved human change record does not match the child A ticket')
  }
  if (typeof record.change.original_intent !== 'string' || record.change.original_intent.length === 0
    || typeof record.change.adopted_change !== 'string' || record.change.adopted_change.length === 0
    || typeof record.change.new_objective !== 'string'
    || record.change.new_objective !== child.target.objective
    || typeof record.change.contract_changes !== 'object' || record.change.contract_changes === null
    || Array.isArray(record.change.contract_changes)
    || typeof record.change.goal_changes !== 'object' || record.change.goal_changes === null
    || Array.isArray(record.change.goal_changes)
    || sha256(canonicalJson(parent.a_snapshot, numberSources)) !== parent.a_snapshot_sha256) {
    reject('saved human change record parent A snapshot hash is invalid')
  }
  const expectedRef = {
    record_id: record.record_id,
    content_sha256: record.content_sha256,
    parent_research_id: parent.research_id,
    parent_contract_sha256: parent.contract_sha256,
    parent_target_version: parent.target.version,
    parent_a_snapshot_sha256: parent.a_snapshot_sha256,
    affected_task_refs: parent.task_refs,
    affected_evidence_refs: parent.evidence_refs,
  }
  if (!sameJson(ticket.change_ref, expectedRef)) {
    reject('child A ticket change reference differs from the saved human change record')
  }
}

function validateRoleMap(sessionRoles) {
  if (typeof sessionRoles !== 'object' || sessionRoles === null || Array.isArray(sessionRoles)) {
    throw new TypeError('factor-research: sessionRoles must be an object')
  }
  const roles = new Map()
  for (const [role, sessionId] of Object.entries(sessionRoles)) {
    if (!['coordinator', 'A', 'B'].includes(role) || typeof sessionId !== 'string' || sessionId.length === 0) {
      throw new TypeError(`factor-research: invalid sessionRoles entry ${JSON.stringify(role)}`)
    }
    if (roles.has(sessionId)) throw new TypeError('factor-research: coordinator, A and B must use distinct Sessions')
    roles.set(sessionId, role)
  }
  return roles
}

export function createModelRequestGuard({ roleForSession, budgetRoot, maxRequestsPerTurn,
  maxTotalRequests, maxOutputTokens }) {
  if (typeof roleForSession !== 'function') throw new TypeError('factor-research: roleForSession must be a function')
  if (!Number.isSafeInteger(maxRequestsPerTurn) || maxRequestsPerTurn < 1) {
    throw new TypeError('factor-research: maxRequestsPerTurn must be a positive safe integer')
  }
  if (!Number.isSafeInteger(maxOutputTokens) || maxOutputTokens < 1) {
    throw new TypeError('factor-research: maxOutputTokens must be a positive safe integer')
  }
  if (!Number.isSafeInteger(maxTotalRequests) || maxTotalRequests < 1) {
    throw new TypeError('factor-research: maxTotalRequests must be a positive safe integer')
  }
  if (typeof budgetRoot !== 'string' || !isAbsolute(budgetRoot)) {
    throw new TypeError('factor-research: budgetRoot must be an absolute research run root')
  }
  const turnRequests = new Map()
  return async function guardModelRequest({ agent, turn }, next) {
    const sessionId = agent?.session?.id
    if (typeof sessionId !== 'string' || sessionId.length === 0) {
      throw new TypeError('factor-research: agent/request is missing its Session identity')
    }
    const role = roleForSession(sessionId) ?? 'unbound'
    if (!Number.isSafeInteger(turn) || turn < 1) {
      throw new TypeError('factor-research: agent/request has an invalid turn number')
    }
    const previous = turnRequests.get(sessionId)
    const count = previous?.turn === turn ? previous.count : 0
    if (count >= maxRequestsPerTurn) {
      reject('research Agent reached the configured model-request limit for this turn',
        'factor-research/request-budget-exhausted')
    }
    turnRequests.set(sessionId, { turn, count: count + 1 })
    const request = await next()
    const boundedRequest = {
      ...request,
      maxTokens: Math.min(request.maxTokens ?? maxOutputTokens, maxOutputTokens),
    }
    await reserveModelRequest({
      budgetRoot,
      maxTotalRequests,
      role,
      sessionId,
      turn,
      provider: boundedRequest.provider,
      model: boundedRequest.model,
      maxTokens: boundedRequest.maxTokens,
    })
    return boundedRequest
  }
}

async function reserveModelRequest({
  budgetRoot, maxTotalRequests, role, sessionId, turn, provider, model, maxTokens, purpose = 'research',
}) {
  const root = await realpath(budgetRoot)
  const controlRoot = join(root, '.factor-research-control')
  const budgetDirectory = join(controlRoot, 'model-budget')
  const reservationDirectory = join(budgetDirectory, 'reservations')
  await mkdir(reservationDirectory, { recursive: true, mode: 0o700 })
  const actualDirectory = await realpath(reservationDirectory)
  if (!within(root, actualDirectory) || actualDirectory !== reservationDirectory) {
    reject('model budget reservation directory escaped the bound run root',
      'factor-research/budget-ledger-invalid')
  }

  for (let slot = 1; slot <= maxTotalRequests; slot += 1) {
    const path = join(reservationDirectory, `${String(slot).padStart(6, '0')}.json`)
    try {
      const reservation = await open(path, 'wx', 0o600)
      try {
        await reservation.writeFile(`${JSON.stringify({
          format_version: 1,
          slot,
          role,
          session_id: sessionId,
          turn,
          provider,
          model,
          max_tokens: maxTokens,
          purpose,
          request_id: randomUUID(),
          reserved_at: new Date().toISOString(),
        })}\n`)
        await reservation.sync()
      } finally {
        await reservation.close()
      }
      return
    } catch (error) {
      if (error?.code === 'EEXIST') continue
      throw error
    }
  }
  reject('research run reached its configured total model-request limit',
    'factor-research/run-budget-exhausted')
}

// Native compaction calls the LLM service outside agent/request. Its exact
// recorded request must already fit the cap; never rewrite that durable input.
export function createCompactionRequestGuard({ roleForSession, requireRunActive,
  budgetRoot, maxTotalRequests, maxOutputTokens }) {
  return async function* guard(options, next) {
    if (options.purpose !== 'compaction') {
      yield* next()
      return
    }
    const role = roleForSession(options.sessionId)
    if (role === undefined) reject('compaction requires a bound research Session')
    // Persisted pre-isolation Sessions declared working_directory. Native
    // summaries replay that exact historical header, but execute no tool calls;
    // the live execution guard still rejects working_directory for every role.
    if ((options.tools ?? []).some(tool => tool.name !== 'working_directory'
      && !isResearchToolAllowed(role, tool.name))) {
      reject('compaction includes tools outside its fixed research role')
    }
    if (!Number.isSafeInteger(options.maxTokens) || options.maxTokens < 1
      || options.maxTokens > maxOutputTokens) reject('compaction output cap exceeds the research budget')
    await requireRunActive(role)
    await reserveModelRequest({ budgetRoot, maxTotalRequests, role,
      sessionId: options.sessionId, turn: null, provider: options.provider,
      model: options.model, maxTokens: options.maxTokens, purpose: 'compaction' })
    yield* next()
  }
}

/** Recompute the bound native evidence with the same Python domain checker used by research tools. */
export function createGoalEvidenceVerifier(config) {
  return function verifyGoalEvidence() {
    for (const [key, value] of Object.entries({
      bindingPath: config?.bindingPath,
      pythonExecutable: config?.pythonExecutable,
      projectRoot: config?.projectRoot,
    })) {
      if (typeof value !== 'string' || !isAbsolute(value)) {
        reject(`Goal evidence verifier requires an absolute ${key}`, 'factor-research/evidence-required')
      }
    }
    const result = spawnSync(config.pythonExecutable, [
      '-m',
      'crypto_quant.research.factor_mining.mcp_server',
      '--role',
      'coordinator',
      '--binding',
      config.bindingPath,
      '--check-goal-evidence',
    ], {
      cwd: config.projectRoot,
      env: { ...process.env, PYTHONPATH: join(config.projectRoot, 'src') },
      encoding: 'utf8',
      maxBuffer: 1024 * 1024,
      timeout: 30_000,
    })
    if (result.error !== undefined || result.status !== 0 || typeof result.stdout !== 'string') {
      reject('domain Goal evidence could not be verified', 'factor-research/evidence-required')
    }
    try {
      return JSON.parse(result.stdout)
    } catch {
      reject('domain Goal evidence verifier returned invalid output', 'factor-research/evidence-required')
    }
  }
}

async function readTicket(runRoot, taskId, { checkCurrentAState = true, forDelivery = true, ancestors = [] } = {}) {
  if (!/^[a-f0-9]{64}$/.test(taskId)) reject('task_id must be a lowercase SHA-256 digest')
  if (ancestors.includes(taskId)) reject('A recovery ticket references a cyclic task chain')
  const nextAncestors = [...ancestors, taskId]
  if (typeof runRoot !== 'string' || !isAbsolute(runRoot)) {
    reject('an absolute research runRoot must be bound in the Host profile', 'factor-research/admission-required')
  }
  const root = await realpath(runRoot)
  const ticketPath = join(root, 'task-tickets', 'A', `${taskId}.json`)
  const ticketStat = await lstat(ticketPath)
  if (!ticketStat.isFile() || ticketStat.isSymbolicLink()) reject('saved A task ticket is not a regular file')
  const actualTicketPath = await realpath(ticketPath)
  if (!within(root, actualTicketPath) || actualTicketPath !== ticketPath) reject('saved A task ticket escaped the bound run root')

  const text = await readFile(actualTicketPath, 'utf8')
  const { ticket, numberSources } = parseTicket(text)
  if (typeof ticket !== 'object' || ticket === null || Array.isArray(ticket)) reject('saved A task ticket must be an object')
  if (ticket.format_version !== 1 || ticket.task_id !== taskId || ticket.content_sha256 !== taskId
    || ticket.action_id !== `prepare-a-task-${taskId}`) {
    reject('saved A task ticket identity does not match its reference')
  }
  const metadata = new Set(['task_id', 'content_sha256', 'action_id'])
  if (sha256(canonicalJson(ticket, numberSources, undefined, undefined, metadata)) !== taskId) {
    reject('saved A task ticket content hash is invalid')
  }

  const contractPath = join(root, 'contract.json')
  const contractStat = await lstat(contractPath)
  if (!contractStat.isFile() || contractStat.isSymbolicLink()) reject('fixed research contract is not a regular file')
  const contractBytes = await readFile(contractPath)
  if (sha256(contractBytes) !== ticket.contract_sha256) reject('saved A task ticket contract hash is invalid')
  const contract = JSON.parse(contractBytes.toString('utf8'))
  if (ticket.contract_version !== 'FM-v6' || !sameJson(ticket.contract, contract)) {
    reject('saved A task ticket does not match the fixed FM-v6 contract')
  }
  if (ticket.research_id !== contract.run_id || ticket.batch_id !== contract.run_id
    || ticket.a_snapshot?.run_id !== contract.run_id || ticket.a_snapshot?.a_status !== 'open') {
    reject('saved A task ticket is not bound to the open research run')
  }
  if (!['initial_a_research', 'continue_a_research', 'recovery_a_research'].includes(ticket.task_type)
    || !Number.isSafeInteger(ticket.target_version) || ticket.target_version < 1
    || ticket.target_version !== ticket.target?.version
    || ticket.goal?.objective !== ticket.target?.objective
    || ticket.target?.objective !== contract.objective
    || ticket.a_snapshot?.research_contract === undefined
    || !sameJson(ticket.a_snapshot.research_contract, contract)) {
    reject('saved A task ticket target, Goal, or stage is inconsistent')
  }
  await readChangeReference(root, ticket, contract, contractBytes)
  if (ticket.thin_spec?.data_environment?.segment !== 'A'
    || ticket.thin_spec?.goal_counts?.b_qualified_deliverables !== null
    || ticket.thin_spec?.goal_counts?.b_count_availability !== 'not available to A before frozen-batch validation') {
    reject('saved A task ticket violates the A-only information boundary')
  }
  if (forDelivery && (await exists(join(root, 'a-complete.json')) || await exists(join(root, 'b-access-started.json')))) {
    reject('A task ticket cannot be delivered after A seal or B access', 'factor-research/admission-required')
  }

  let recoveryReceipt
  if (ticket.task_type === 'recovery_a_research') {
    const recovery = ticket.recovery_ref
    const recoveryKeys = [
      'parent_task_id', 'parent_content_sha256', 'parent_task_type',
      'parent_a_snapshot_sha256', 'parent_dispatch_receipt_sha256', 'pending_actions',
    ].sort()
    if (typeof recovery !== 'object' || recovery === null || Array.isArray(recovery)
      || !sameJson(Object.keys(recovery).sort(), recoveryKeys)
      || recovery.parent_task_id === taskId
      || recovery.parent_content_sha256 !== recovery.parent_task_id
      || !Array.isArray(recovery.pending_actions)) {
      reject('A recovery ticket reference is malformed')
    }
    const pendingActionIds = new Set()
    for (const pending of recovery.pending_actions) {
      if (typeof pending !== 'object' || pending === null || Array.isArray(pending)
        || !sameJson(Object.keys(pending).sort(), ['action_id', 'name', 'parameters_sha256'])
        || typeof pending.action_id !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$/.test(pending.action_id)
        || typeof pending.name !== 'string' || pending.name.length === 0
        || typeof pending.parameters_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(pending.parameters_sha256)
        || pendingActionIds.has(pending.action_id)) {
        reject('A recovery ticket pending-action reference is malformed')
      }
      pendingActionIds.add(pending.action_id)
    }
    const parent = await readTicket(root, recovery.parent_task_id, {
      checkCurrentAState: false,
      forDelivery,
      ancestors: nextAncestors,
    })
    if (recovery.parent_task_type !== parent.ticket.task_type
      || parent.ticket.research_id !== ticket.research_id
      || parent.ticket.contract_sha256 !== ticket.contract_sha256
      || parent.ticket.target_version !== ticket.target_version
      || recovery.parent_a_snapshot_sha256 !== sha256(canonicalJson(parent.ticket.a_snapshot))) {
      reject('A recovery ticket does not match its parent task')
    }
    const receiptPath = join(root, 'dispatch-receipts', 'A', `${recovery.parent_task_id}.json`)
    const receiptStat = await lstat(receiptPath)
    if (!receiptStat.isFile() || receiptStat.isSymbolicLink()) reject('parent A dispatch receipt is not a regular file')
    const actualReceiptPath = await realpath(receiptPath)
    if (!within(root, actualReceiptPath) || actualReceiptPath !== receiptPath) {
      reject('parent A dispatch receipt escaped the bound run root')
    }
    recoveryReceipt = JSON.parse(await readFile(actualReceiptPath, 'utf8'))
    if (typeof recoveryReceipt !== 'object' || recoveryReceipt === null || Array.isArray(recoveryReceipt)
      || !sameJson(Object.keys(recoveryReceipt).sort(), [
        'format_version', 'task_id', 'ticket_sha256', 'target_role', 'session_id', 'request_id', 'status',
      ].sort())
      || recoveryReceipt.format_version !== 1
      || recoveryReceipt.task_id !== recovery.parent_task_id
      || recoveryReceipt.ticket_sha256 !== recovery.parent_task_id
      || recoveryReceipt.target_role !== 'A'
      || typeof recoveryReceipt.session_id !== 'string'
      || typeof recoveryReceipt.request_id !== 'string'
      || recoveryReceipt.status !== 'accepted'
      || recovery.parent_dispatch_receipt_sha256 !== sha256(canonicalJson(recoveryReceipt))) {
      reject('A recovery ticket requires its matching accepted parent dispatch receipt')
    }
  } else if (Object.hasOwn(ticket, 'recovery_ref')) {
    reject('non-recovery A task ticket cannot declare a recovery reference')
  }

  const recordsDirectory = join(root, 'a_records')
  const recordsPath = await realpath(recordsDirectory)
  if (recordsPath !== recordsDirectory || !within(root, recordsPath)) reject('A evidence records escaped the bound run root')
  const recordNames = (await readdir(recordsDirectory)).filter(name => name.endsWith('.json')).sort()
  const currentRefs = []
  for (const name of recordNames) {
    const recordPath = join(recordsDirectory, name)
    const stat = await lstat(recordPath)
    if (!stat.isFile() || stat.isSymbolicLink()) reject('A evidence record is not a regular file')
    const record = JSON.parse(await readFile(recordPath, 'utf8'))
    if (`${record.id}.json` !== name || typeof record.kind !== 'string') reject('A evidence record identity is invalid')
    currentRefs.push({ record_id: record.id, kind: record.kind })
  }
  if (checkCurrentAState && !sameJson(currentRefs, ticket.a_snapshot.a_evidence_refs)) {
    reject('saved A task ticket is stale relative to the current A evidence records', 'factor-research/admission-required')
  }
  return { root, text, ticket, recoveryReceipt }
}

async function readBTicket(runRoot, taskId, boundRun) {
  if (!/^[a-f0-9]{64}$/.test(taskId)) reject('task_id must be a lowercase SHA-256 digest')
  const root = await realpath(runRoot)
  const ticketPath = join(root, 'task-tickets', 'B', `${taskId}.json`)
  const ticketStat = await lstat(ticketPath)
  if (!ticketStat.isFile() || ticketStat.isSymbolicLink()) reject('saved B task ticket is not a regular file')
  const actualTicketPath = await realpath(ticketPath)
  if (!within(root, actualTicketPath) || actualTicketPath !== ticketPath) reject('saved B task ticket escaped the bound run root')

  const text = await readFile(actualTicketPath, 'utf8')
  const { ticket, numberSources } = parseTicket(text)
  if (typeof ticket !== 'object' || ticket === null || Array.isArray(ticket)) reject('saved B task ticket must be an object')
  if (ticket.format_version !== 1 || ticket.task_id !== taskId || ticket.content_sha256 !== taskId
    || ticket.action_id !== `prepare-b-task-${taskId}`) {
    reject('saved B task ticket identity does not match its reference')
  }
  const metadata = new Set(['task_id', 'content_sha256', 'action_id'])
  if (sha256(canonicalJson(ticket, numberSources, undefined, undefined, metadata)) !== taskId) {
    reject('saved B task ticket content hash is invalid')
  }

  const { binding, contract, contractBytes } = boundRun
  const expectedKeys = [
    'format_version', 'research_id', 'batch_id', 'task_type', 'target_version',
    'contract_version', 'contract_sha256', 'target', 'goal', 'contract',
    'a_completion', 'frozen_batch_sha256', 'frozen_batch', 'thin_spec',
    'expected_deliverables', 'task_id', 'content_sha256', 'action_id',
  ].sort()
  if (!sameJson(Object.keys(ticket).sort(), expectedKeys)
    || ticket.research_id !== contract.run_id || ticket.batch_id !== contract.run_id
    || ticket.task_type !== 'frozen_b_validation'
    || ticket.target_version !== binding.target.version
    || ticket.contract_version !== 'FM-v6'
    || ticket.contract_sha256 !== sha256(contractBytes)
    || !sameJson(ticket.contract, contract)
    || !sameJson(ticket.target, binding.target)
    || !sameJson(ticket.goal, binding.goal)) {
    reject('saved B task ticket does not match the fixed research binding')
  }

  const completionPath = join(root, 'a-complete.json')
  const completionStat = await lstat(completionPath)
  if (!completionStat.isFile() || completionStat.isSymbolicLink()) reject('A completion checkpoint is not a regular file')
  const completionBytes = await readFile(completionPath)
  let completion
  try {
    completion = JSON.parse(completionBytes.toString('utf8'))
  } catch {
    reject('A completion checkpoint is invalid')
  }
  const frozenPath = join(root, 'frozen_batch.json')
  const frozenStat = await lstat(frozenPath)
  if (!frozenStat.isFile() || frozenStat.isSymbolicLink()) reject('frozen B batch is not a regular file')
  const frozenBytes = await readFile(frozenPath)
  let frozen
  try {
    frozen = JSON.parse(frozenBytes.toString('utf8'))
  } catch {
    reject('frozen B batch is invalid')
  }
  const candidateIds = Object.keys(frozen?.candidates ?? {})
  if (candidateIds.length === 0
    || completion?.manifest_ref !== frozen?.a_manifest_ref
    || !Array.isArray(completion?.retained_ids)
    || !sameJson(completion.retained_ids, candidateIds)
    || !sameJson(ticket.a_completion, {
      manifest_ref: completion.manifest_ref,
      frozen_candidate_ids: candidateIds,
      content_sha256: sha256(completionBytes),
    })
    || ticket.frozen_batch_sha256 !== sha256(frozenBytes)
    || !sameJson(ticket.frozen_batch, frozen)
    || !sameJson(frozen.contract, contract)) {
    reject('saved B task ticket does not match the completed A manifest and frozen batch')
  }

  const thinSpec = ticket.thin_spec
  const expectedEnvironment = {
    segment: 'B',
    purpose: contract.purpose,
    include_liquidations: binding.inputs.include_liquidations,
    interval_hours: contract.sample_hours,
    timezone: 'UTC',
    start_inclusive: contract.b_start,
    end_exclusive: contract.c_start,
    warmup_hours: contract.max_lookback_hours,
    universe_provenance: contract.universe_provenance,
    coverage: 'not loaded; the bound B validator reports coverage after its single access',
  }
  const expectedPolicy = {
    candidate_set: 'exact frozen_batch.candidates; no additions, removals, or substitutions',
    definitions: 'frozen formula, direction, and retained horizons are immutable',
    numerical_access: 'one B validation; persist the checkpoint before interpretation',
    incomplete_access: 'if access started without a complete checkpoint, inspect evidence and stop',
    feedback_to_a: 'never return B metrics to A or use them to modify this frozen batch',
  }
  const expectedDeliverables = {
    numerical_checkpoint: 'single frozen-batch B validation with complete saved checkpoint',
    interpretation: 'evidence-grounded report for every frozen candidate',
    delivery: 'complete B delivery from the saved numerical and interpretation records',
  }
  if (!sameJson(thinSpec, { data_environment: expectedEnvironment, validation_policy: expectedPolicy })
    || !sameJson(ticket.expected_deliverables, expectedDeliverables)) {
    reject('saved B task ticket violates the frozen validation policy')
  }

  return { root, text, ticket }
}

async function exists(path) {
  try {
    await lstat(path)
    return true
  } catch (error) {
    if (error.code === 'ENOENT') return false
    throw error
  }
}

function acceptedEvent(events, requestId, ticketText) {
  return events.find(event => event.type === 'user/message'
    && event.data?.source?.kind === 'user'
    && event.data.source.rpcId === requestId
    && Array.isArray(event.data.content)
    && event.data.content.length === 1
    && event.data.content[0]?.type === 'text'
    && event.data.content[0].text === ticketText)
}

function bContinuationReceiptPath(receiptDirectory, attempt) {
  return join(receiptDirectory, attempt === 1 ? 'continuation.json' : `continuation-${attempt}.json`)
}

function aContinuationReceiptPath(receiptDirectory, taskId, attempt) {
  return join(receiptDirectory, `continuation-${taskId}-${attempt}.json`)
}

function aContinuationText(taskId, attempt) {
  return `Resume the already accepted A task ${taskId} (Host continuation ${attempt}). The exact saved A ticket and current A evidence are unchanged. Continue from that ticket and existing A-only evidence; finish every pending interpretation and decision, close existing experiment routes using their saved evidence or stop them with a reason, cite complete exact A evidence IDs, and seal a complete manifest. Do not add candidates, open new experiments, or repeat completed calculations. Never access B results.`
}

async function writeReceipt(path, receipt, create = false) {
  const parent = dirname(path)
  await mkdir(parent, { recursive: true })
  const serialized = `${JSON.stringify(receipt, null, 2)}\n`
  if (create) {
    await writeFile(path, serialized, { flag: 'wx', mode: 0o600 })
    return
  }
  const tempPath = `${path}.${randomUUID()}.tmp`
  await writeFile(tempPath, serialized, { flag: 'wx', mode: 0o600 })
  await rename(tempPath, path)
}

export function createTicketDispatcher({
  config, sessions, agents, goals, originalPrompt, originalCancel, originalGoalPause,
  originalComplete, verifyGoalEvidence, actionControl,
}) {
  const controlAction = actionControl ?? ((action, request) => runActionControl(config, action, request))
  let roleBySession = new Map()
  let roleError
  try {
    roleBySession = validateRoleMap(config?.sessionRoles)
  } catch (error) {
    roleError = error
  }
  const inFlight = new Map()
  let inFlightAOperation
  let inFlightBTaskId
  let inFlightBContinuationTaskId
  const operations = new Set()
  const shutdownController = new AbortController()
  let runController = new AbortController()
  let localStopRequest
  let disposed = false
  let boundRoot
  let initialization
  let resultEpoch = 0
  let resultDelivery = Promise.resolve()

  async function initializeInner() {
    if (roleError !== undefined) throw roleError
    if (typeof config.runRoot !== 'string' || !isAbsolute(config.runRoot)) {
      reject('an absolute research runRoot must be bound in the Host profile', 'factor-research/admission-required')
    }
    boundRoot = await realpath(config.runRoot)
    if (typeof config.bindingPath !== 'string' || !isAbsolute(config.bindingPath)
      || typeof config.pythonExecutable !== 'string' || !isAbsolute(config.pythonExecutable)
      || typeof config.projectRoot !== 'string' || !isAbsolute(config.projectRoot)) {
      reject('Goal evidence verifier configuration is incomplete', 'factor-research/admission-required')
    }
    readBoundResearchRun()
  }

  function initialize() {
    initialization ??= initializeInner()
    return initialization
  }

  function readBoundResearchRun() {
    if (boundRoot === undefined) {
      reject('research Host configuration is not ready', 'factor-research/admission-required')
    }
    const contractBytes = readRegularBytes(join(boundRoot, 'contract.json'), 'fixed research contract')
    let contract
    try {
      contract = JSON.parse(contractBytes.toString('utf8'))
    } catch {
      reject('fixed research contract is invalid', 'factor-research/admission-required')
    }
    const goalRecord = readRegularJson(join(boundRoot, 'a_records', 'goal-spec.json'), 'fixed domain Goal record')
    const binding = readRegularJson(config.bindingPath, 'fixed research binding')
    if (typeof binding !== 'object' || binding === null || Array.isArray(binding)
      || typeof binding.target !== 'object' || binding.target === null || Array.isArray(binding.target)
      || typeof binding.goal !== 'object' || binding.goal === null || Array.isArray(binding.goal)
      || typeof binding.chats !== 'object' || binding.chats === null || Array.isArray(binding.chats)
      || typeof binding.run_root !== 'string' || binding.run_root.length === 0
      || typeof goalRecord !== 'object' || goalRecord === null || Array.isArray(goalRecord)
      || typeof goalRecord.data !== 'object' || goalRecord.data === null || Array.isArray(goalRecord.data)) {
      reject('fixed research binding or domain Goal record is malformed', 'factor-research/admission-required')
    }
    let bindingRunRoot
    try {
      bindingRunRoot = realpathSync(resolve(dirname(config.bindingPath), binding.run_root))
    } catch {
      reject('fixed research binding does not resolve to the Host run', 'factor-research/admission-required')
    }
    if (bindingRunRoot !== boundRoot
      || binding.format_version !== 1
      || binding.research_id !== contract.run_id
      || binding.contract_sha256 !== sha256(contractBytes)
      || !Number.isSafeInteger(binding.target?.version) || binding.target.version < 1
      || binding.target?.objective !== contract.objective
      || goalRecord.id !== 'goal-spec' || goalRecord.kind !== 'goal_contract'
      || !sameJson(binding.goal, goalRecord.data)
      || binding.goal.objective !== contract.objective) {
      reject('fixed research binding does not match the Host run, contract and domain Goal',
        'factor-research/admission-required')
    }
    for (const [sessionId, role] of roleBySession) {
      const chat = binding.chats[role]
      if (typeof chat !== 'object' || chat === null || Array.isArray(chat)
        || chat.thread_id !== sessionId || chat.host_id !== 'local') {
        reject(`fixed research binding does not match the DSH Web ${role} Session`,
          'factor-research/admission-required')
      }
    }
    return { contractBytes, contract, goalRecord, binding }
  }

  function track(operation) {
    const task = operation.finally(() => operations.delete(task))
    operations.add(task)
    return task
  }

  function combinedSignal(signal) {
    const signals = [runController.signal, shutdownController.signal]
    if (signal !== undefined) signals.unshift(signal)
    return AbortSignal.any(signals)
  }

  function runControlState() {
    if (boundRoot === undefined) {
      reject('research Host configuration is not ready', 'factor-research/admission-required')
    }
    const persisted = readActionControlState(boundRoot)
    if (persisted.status === 'active' && localStopRequest !== undefined) {
      return { status: 'stopping', source: 'program_action_control',
        action_id: localStopRequest.actionId, reason: localStopRequest.reason }
    }
    return persisted
  }

  async function requireRunActive(role) {
    await initialize()
    if (await exists(join(boundRoot, '.factor-research-control', 'superseded-by.json'))) {
      reject('this research version was superseded by an approved human change', 'factor-research/run-not-active')
    }
    const state = runControlState()
    if (role === 'coordinator' && state.status !== 'stopping') return
    if (state.status === 'stopping') {
      reject('research run is stopping; wait for all bound Sessions and actions to settle',
        'factor-research/run-not-active')
    }
    if (state.status === 'stopped') {
      reject(`research run is ${state.status}; explicitly resume through the bound Coordinator Goal before continuing`,
        'factor-research/run-not-active')
    }
  }

  function boundLiveAgents() {
    if (typeof agents.list !== 'function') {
      reject('DSH Agent registry does not expose live Agent enumeration',
        'factor-research/agent-control-unavailable')
    }
    const liveAgents = agents.list()
    if (!Array.isArray(liveAgents)) reject('DSH Agent registry returned an invalid live Agent list',
      'factor-research/agent-control-unavailable')
    const boundIds = new Map([...roleBySession].map(([sessionId, role]) => [sessionId, role]))
    const matches = liveAgents.filter(agent => boundIds.has(agent?.session?.id))
    const byRole = new Map()
    for (const agent of matches) {
      const role = boundIds.get(agent.session.id)
      if (byRole.has(role)) reject(`multiple live DSH Agents are bound to role ${role}`,
        'factor-research/agent-control-ambiguous')
      byRole.set(role, agent)
    }
    return byRole
  }

  function stopRequestFor(state = runControlState()) {
    if (state.status !== 'stopping' || typeof state.action_id !== 'string'
      || typeof state.reason !== 'string') {
      reject('research run has no recoverable stop request', 'factor-research/action-control-invalid')
    }
    return { actionId: state.action_id, reason: state.reason }
  }

  async function stopRun(sessionId) {
    await initialize()
    if (!isCoordinator(sessionId)) {
      reject('only the bound Coordinator Session can stop the research run',
        'factor-research/admission-required')
    }
    if (disposed) reject('research Host is disposing', 'factor-research/admission-required')
    resultEpoch += 1
    const coordinatorAgent = boundLiveAgents().get('coordinator')
    if (coordinatorAgent !== undefined && coordinatorAgent.session.id !== sessionId) {
      reject('active Coordinator Agent does not match the bound Session',
        'factor-research/agent-control-ambiguous')
    }
    let state = runControlState()
    let request
    if (state.status === 'active') {
      request = { actionId: `host-stop-${randomUUID()}`,
        reason: 'DSH Web Coordinator Session cancellation' }
      localStopRequest = request
      await controlAction('begin-stop', request)
      state = runControlState()
    } else if (state.status === 'stopping') {
      request = stopRequestFor(state)
      localStopRequest = request
      if (readActionControlState(boundRoot).status === 'active') {
        await controlAction('begin-stop', request)
      }
    } else {
      request = { actionId: state.action_id, reason: state.reason }
    }

    if (!runController.signal.aborted) {
      runController.abort(new Error('factor-research run stop requested'))
    }

    if (coordinatorAgent !== undefined && originalGoalPause !== undefined) {
      const goal = goals.get(coordinatorAgent)
      if (goal?.phase === 'active') {
        await Reflect.apply(originalGoalPause, goals, [coordinatorAgent,
          { id: goal.id, revision: goal.revision }])
      }
    }

    const liveAgents = boundLiveAgents()
    const cancelPromises = []
    for (const [role, agent] of liveAgents) {
      if (role === 'coordinator') {
        if (originalCancel !== undefined) {
          cancelPromises.push(Promise.resolve(Reflect.apply(originalCancel, sessions,
            [{ sessionId: agent.session.id }])))
        } else {
          agent.cancel({ kind: 'user' })
        }
      } else {
        agent.cancel({ kind: 'user' }, { keepInbox: true })
      }
    }
    const idlePromise = Promise.all([...liveAgents.values()].map(agent => agent.whenIdle()))
    const cancelPromise = Promise.all(cancelPromises)
    let timeoutHandle
    const quiesced = await Promise.race([
      Promise.all([idlePromise, cancelPromise]).then(() => true),
      new Promise(resolve => {
        timeoutHandle = setTimeout(() => resolve(false), config.stopTimeoutMs)
      }),
    ])
    clearTimeout(timeoutHandle)
    if (!quiesced) {
      return { status: 'stopping', source: 'program_action_control',
        action_id: request?.actionId ?? state.action_id,
        reason: request?.reason ?? state.reason,
        sessions: Object.fromEntries([...boundLiveAgents()].map(([role, agent]) => [role, agent.status])) }
    }

    const pending = [...operations]
    if (pending.length > 0) {
      let operationsHandle
      const operationsSettled = await Promise.race([
        Promise.allSettled(pending).then(() => true),
        new Promise(resolve => {
          operationsHandle = setTimeout(() => resolve(false), config.stopTimeoutMs)
        }),
      ])
      clearTimeout(operationsHandle)
      if (!operationsSettled) {
        return { status: 'stopping', source: 'program_action_control',
          action_id: request?.actionId ?? state.action_id,
          reason: request?.reason ?? state.reason,
          sessions: Object.fromEntries([...boundLiveAgents()].map(([role, agent]) => [role, agent.status])) }
      }
    }

    state = runControlState()
    if (state.status === 'stopped') {
      localStopRequest = undefined
      return { ...state,
        sessions: Object.fromEntries([...boundLiveAgents()].map(([role, agent]) => [role, agent.status])) }
    }
    request = stopRequestFor(state)
    const result = await controlAction('finish-stop', request)
    if (result.status === 'stopped') localStopRequest = undefined
    return { ...result,
      sessions: Object.fromEntries([...boundLiveAgents()].map(([role, agent]) => [role, agent.status])) }
  }

  async function resumeRun(agent, ref) {
    await initialize()
    if (await exists(join(boundRoot, '.factor-research-control', 'superseded-by.json'))) {
      reject('resume the approved child research version; the parent remains stopped', 'factor-research/run-not-active')
    }
    requireCoordinatorGoalAgent(agent, 'Research run resume')
    const goal = goals.get(agent)
    if (goal === undefined || goal.id !== ref?.id || goal.revision !== ref?.revision
      || !['paused', 'active'].includes(goal.phase)) {
      // DSH restores an active Goal disarmed after a Host restart. Its native
      // resume verb rearms that same revision without changing the phase.
      reject('research run resume requires the paused or active bound Coordinator Goal',
        'factor-research/admission-required')
    }
    let state = runControlState()
    if (state.status === 'stopping') {
      const request = stopRequestFor(state)
      const live = boundLiveAgents()
      if ([...live.values()].some(item => item.status !== 'idle')) {
        reject('research run is still stopping; all bound Sessions must be idle before resume',
          'factor-research/run-not-active')
      }
      state = await controlAction('finish-stop', request)
      if (state.status !== 'stopped') {
        reject('research run stop has not settled; resume is refused', 'factor-research/run-not-active')
      }
    }
    if (state.status === 'stopped') {
      const live = boundLiveAgents()
      if ([...live.values()].some(item => item.status !== 'idle')) {
        reject('research run is stopped but a bound Session is still active',
          'factor-research/run-not-active')
      }
      const result = await controlAction('resume', { actionId: `host-resume-${randomUUID()}` })
      if (result.status !== 'active') {
        reject('research action ledger did not confirm resume', 'factor-research/run-not-active')
      }
      runController = new AbortController()
      localStopRequest = undefined
      return result
    }
    return { status: state.status, source: state.source }
  }

  async function runtimeStatus(agent) {
    await initialize()
    requireCoordinatorGoalAgent(agent, 'Research runtime status')
    const boundIds = Object.fromEntries([...roleBySession].map(([sessionId, role]) => [role, sessionId]))
    const live = boundLiveAgents()
    const state = runControlState()
    const inFlightOperations = []
    if (inFlight.has('A')) inFlightOperations.push('A')
    if (inFlight.has('B')) inFlightOperations.push('B')
    if (inFlight.has('B_CONTINUE')) inFlightOperations.push('B delivery')
    return {
      research_id: readBoundResearchRun().binding.research_id,
      state: state.status,
      program_actions: state,
      sessions: Object.fromEntries(['coordinator', 'A', 'B'].map(role => [role, {
        session_id: boundIds[role],
        status: live.get(role)?.status ?? 'not_open',
      }])),
      in_flight_operations: inFlightOperations,
    }
  }

  async function dispose() {
    if (disposed) return
    resultEpoch += 1
    if (boundRoot !== undefined && runControlState().status === 'active') {
      const live = boundLiveAgents()
      if (operations.size > 0 || [...live.values()].some(agent => agent.status === 'running')) {
        const coordinatorSessionId = [...roleBySession.entries()]
          .find(([, role]) => role === 'coordinator')?.[0]
        if (coordinatorSessionId !== undefined) {
          await stopRun(coordinatorSessionId)
        }
      }
    }
    disposed = true
    runController.abort(new Error('factor-research Host is disposing'))
    shutdownController.abort(new Error('factor-research Host is disposing'))
    const pending = [...operations]
    if (initialization !== undefined) pending.push(initialization.catch(() => undefined))
    await Promise.allSettled(pending)
  }

  async function readyForDomainTools() {
    try {
      await initialize()
    } catch {
      reject('research Host configuration is not ready', 'factor-research/admission-required')
    }
    if (disposed) reject('research Host is disposing', 'factor-research/admission-required')
  }

  function isCoordinator(sessionId) {
    return roleBySession.get(sessionId) === 'coordinator'
  }

  function roleForSession(sessionId) {
    return roleBySession.get(sessionId)
  }

  function requireCoordinatorGoalAgent(agent, operation) {
    if (agent === undefined || agents.get(agent.id) !== agent
      || !isCoordinator(agent.session?.id)) {
      reject(`${operation} requires the bound Coordinator Session`, 'factor-research/admission-required')
    }
  }

  function requireGoalDispatchActive(agent) {
    // Existing receipt-only adapters omit Goals; the Web Host always injects it.
    const goal = goals === undefined ? undefined : goals.get(agent)
    if (goal !== undefined && (goal.phase !== 'active' || goal.activation === 'disarmed')) {
      reject('research dispatch requires the explicitly resumed Coordinator Goal',
        'factor-research/run-not-active')
    }
  }

  function canReceiveResult(agent) {
    if (disposed || boundRoot === undefined || runControlState().status !== 'active') return false
    const goal = goals.get(agent)
    return goal?.phase === 'active' && goal.activation === 'armed'
  }

  async function deliverResult(role, expectedSeq, epoch, explicitResume) {
    await initialize()
    const live = boundLiveAgents()
    const worker = live.get(role)
    const coordinator = live.get('coordinator')
    if (disposed || epoch !== resultEpoch) return { status: 'suppressed' }
    if (worker === undefined || coordinator === undefined) reject(`${role} result collection requires both bound live Agents`)
    if (!canReceiveResult(coordinator)) return { status: 'suppressed' }
    const dispatch = role === 'A' ? inFlight.get('A') : inFlight.get('B') ?? inFlight.get('B_CONTINUE')
    if (dispatch !== undefined) await dispatch
    await worker.whenIdle()
    if (epoch !== resultEpoch || !canReceiveResult(coordinator)) return { status: 'suppressed' }
    const inspection = await sessions.inspect(worker.session.id)
    const end = inspection.events.filter(event => event.type === 'turn/end').at(-1)
    if (end === undefined || (expectedSeq !== undefined && end.seq !== expectedSeq)) {
      return { status: 'no_current_result' }
    }
    if (!Number.isSafeInteger(end.seq) || end.seq < 0) reject(`${role} result has an invalid Session sequence`)
    const input = inspection.events.filter(event => event.type === 'user/message'
      && event.seq < end.seq && event.data?.source?.kind === 'user').at(-1)
    if (input === undefined) reject(`${role} result has no accepted task input`)
    const directory = join(boundRoot, 'dispatch-receipts', role)
    if (!await exists(directory)) return { status: 'no_current_result' }
    if (await realpath(directory) !== directory) reject(`${role} result source receipts escaped the bound run`)
    const names = await readdir(directory)
    let source
    for (const name of names) {
      const match = role === 'A'
        ? /^(?:continuation-)?([a-f0-9]{64})(?:-([1-3]))?\.json$/.exec(name)
        : name === 'dispatch.json' ? [name] : name === 'continuation.json'
          ? [name, '1'] : /^continuation-([2-3])\.json$/.exec(name)
      if (match === null) reject(`${role} result receipt directory contains an unknown entry`)
      const receipt = readRegularJson(join(directory, name), `accepted ${role} receipt`)
      if (receipt.request_id !== input.data.source.rpcId) continue
      if (receipt.status !== 'accepted' || (role === 'A' && receipt.task_id !== match[1])
        || receipt.ticket_sha256 !== receipt.task_id || receipt.target_role !== role
        || receipt.session_id !== worker.session.id
        || !/^[A-Za-z0-9_-]{1,100}$/.test(receipt.request_id)) reject(`${role} result receipt conflicts with the bound task`)
      const saved = role === 'A'
        ? await readTicket(boundRoot, receipt.task_id, { checkCurrentAState: false, forDelivery: false })
        : await readBTicket(boundRoot, receipt.task_id, readBoundResearchRun())
      const ids = role === 'B' ? Object.keys(saved.ticket.frozen_batch.candidates) : []
      const horizons = role === 'B' ? Object.fromEntries(ids.map(id => [id, saved.ticket.frozen_batch.candidates[id].retained_horizons])) : {}
      const attempt = role === 'A' ? match[2] : name === 'dispatch.json' ? undefined : match[1]
      const expectedText = attempt === undefined ? saved.text : role === 'A'
        ? aContinuationText(receipt.task_id, Number(attempt))
        : continuationText(receipt.task_id, Number(attempt), ids, horizons)
      if (acceptedEvent([input], receipt.request_id, expectedText) === undefined) {
        reject(`${role} result input differs from its accepted task`)
      }
      source = receipt
      break
    }
    if (source === undefined) reject(`${role} result input has no accepted dispatch receipt`)
    if (epoch !== resultEpoch || !canReceiveResult(coordinator)) return { status: 'suppressed' }
    const resultDirectory = join(boundRoot, '.factor-research-control', `${role.toLowerCase()}-results`)
    await mkdir(resultDirectory, { recursive: true })
    if (await realpath(resultDirectory) !== resultDirectory) reject(`${role} result receipt directory escaped its run`)
    const resumedAbort = explicitResume && end.data.reason.kind === 'aborted'
    const resumeKey = resumedAbort ? `-resume-${goals.get(coordinator).revision}` : ''
    const path = join(resultDirectory, `${end.seq}-${source.request_id}${resumeKey}.json`)
    const turnBudget = end.data.reason.kind === 'error'
      && end.data.reason.error?.message === 'research Agent reached the configured model-request limit for this turn'
    const boundary = turnBudget ? 'per_turn_budget_exhausted' : end.data.reason.kind
    const intro = `${role} turn result is available for accepted task ${source.task_id}, source request ${source.request_id}, ${role} Session ${worker.session.id}, turn ${end.data.turn}, end seq ${end.seq}, reason ${boundary}. This is a turn-end notice, not proof the task or ${role} research is complete. `
    const recovery = resumedAbort
      ? `The human has explicitly resumed the same native Goal at revision ${goals.get(coordinator).revision} after this user interruption. The run is now active and armed. Reconcile saved actions with their original IDs and parameters through the existing A recovery-ticket mechanism, or continue B delivery only from a complete saved numerical checkpoint. This specific interrupted turn is authorized for bounded recovery; do not pause merely because its old end reason was aborted. Never repeat committed calculations, reload B, or retry other infrastructure errors. `
      : ''
    const text = intro + recovery + (role === 'A'
      ? 'Read get_research_status and read_a_result for each current candidate; cite the complete A decision, report and exact evidence references. If A is sealed with no retained candidates, skip freeze/B, check_goal_evidence, report incomplete and pause the same native Goal. If A is sealed with retained candidates, freeze its complete manifest, prepare and dispatch the single B ticket, then end your turn and await the B result notice. If A is unsealed after a completed turn, max-tokens or per_turn_budget_exhausted, continue only through the existing accepted-ticket/recovery mechanism and finite continuation limits; do not recompute committed actions. Other errors are blockers: report and pause without retry. Never repeatedly poll unchanged status or resume a paused/stopped Goal.'
      : 'Read get_research_status and read_b_result for every frozen candidate, with exact narrow B evidence pointers when necessary. If B delivery is complete, check_goal_evidence and complete the same native Goal only when objective_met is true; otherwise report gaps and pause it. If the numerical checkpoint is complete but delivery is unfinished after a normal or per_turn_budget_exhausted turn, use the next finite Host-verified B delivery continuation, then await the next notice. If B access started without a complete checkpoint or another error occurred, report the blocker and pause; never reload B. Never send B results to A or reopen A, and never resume a paused/stopped Goal.')
    let receipt
    if (await exists(path)) {
      receipt = readRegularJson(path, `${role} result notification receipt`)
      if (receipt.task_id !== source.task_id || receipt.source_request_id !== source.request_id
        || receipt.end_seq !== end.seq || receipt.content_sha256 !== sha256(text)
        || receipt.session_id !== coordinator.session.id) reject(`${role} result notification receipt conflicts with its source`)
      if (receipt.status === 'accepted') {
        if (!explicitResume) return { status: 'accepted', request_id: receipt.request_id }
        const observed = await sessions.inspect(coordinator.session.id)
        const pending = [...coordinator.inbox.nextTurn, ...coordinator.inbox.nextStep]
        if (acceptedEvent(observed.events, receipt.request_id, text) !== undefined
          || pending.some(message => message.source?.rpcId === receipt.request_id)) {
          return { status: 'accepted', request_id: receipt.request_id }
        }
        // UI Stop can discard an accepted but unclaimed Coordinator notice.
        // Only explicit Goal resume may restore it, with the same native ID.
      } else if (receipt.status !== 'pending') reject(`${role} result notification receipt status is invalid`)
    } else {
      receipt = { format_version: 1, task_id: source.task_id, source_request_id: source.request_id,
        end_seq: end.seq, session_id: coordinator.session.id,
        request_id: `factor-${role.toLowerCase()}-result-${sha256(`${worker.session.id}:${end.seq}:${source.request_id}${resumeKey}`)}`,
        content_sha256: sha256(text), status: 'pending' }
      await writeReceipt(path, receipt, true)
    }
    if (epoch !== resultEpoch || !canReceiveResult(coordinator)) return { status: 'suppressed' }
    // Native Session.prompt deduplicates requestId against both inbox and history.
    const result = await Reflect.apply(originalPrompt, sessions, [{ sessionId: coordinator.session.id,
      requestId: receipt.request_id, mode: 'queue', content: [{ type: 'text', text }] }, combinedSignal()])
    if (result?.accepted !== true) reject('DSH did not accept the A result notification')
    receipt.status = 'accepted'
    await writeReceipt(path, receipt)
    return { status: 'accepted', request_id: receipt.request_id }
  }

  function collectResult(role, expectedSeq, { explicitResume = false } = {}) {
    const epoch = resultEpoch
    const operation = resultDelivery.then(() => deliverResult(role, expectedSeq, epoch, explicitResume))
    resultDelivery = operation
    return track(operation)
  }

  // A Coordinator can reach the same declared per-turn limit while reading
  // results. Wake it once from that exact boundary, without retrying provider
  // errors or polling an idle worker. The total provider budget still applies.
  async function deliverCoordinatorBoundary(expectedSeq, epoch) {
    await initialize()
    const coordinator = boundLiveAgents().get('coordinator')
    if (coordinator === undefined || epoch !== resultEpoch || !canReceiveResult(coordinator)) return { status: 'suppressed' }
    await coordinator.whenIdle()
    if (epoch !== resultEpoch || !canReceiveResult(coordinator)) return { status: 'suppressed' }
    const inspection = await sessions.inspect(coordinator.session.id)
    const end = inspection.events.filter(event => event.type === 'turn/end').at(-1)
    if (end === undefined || (expectedSeq !== undefined && end.seq !== expectedSeq)
      || end.data.reason.kind !== 'error'
      || end.data.reason.error?.message !== 'research Agent reached the configured model-request limit for this turn') {
      return { status: 'no_current_boundary' }
    }
    const goal = goals.get(coordinator)
    const directory = join(boundRoot, '.factor-research-control', 'coordinator-boundaries')
    await mkdir(directory, { recursive: true })
    if (await realpath(directory) !== directory) reject('Coordinator boundary receipts escaped the run')
    const requestId = `factor-coordinator-boundary-${sha256(`${goal.id}:${goal.revision}:${end.seq}`)}`
    const path = join(directory, `${requestId}.json`)
    if (await exists(path)) {
      const prior = readRegularJson(path, 'Coordinator boundary receipt')
      if (prior.status === 'accepted') return { status: 'accepted', request_id: requestId }
      if (prior.status !== 'pending' || prior.end_seq !== end.seq || prior.goal_id !== goal.id
        || prior.goal_revision !== goal.revision) reject('Coordinator boundary receipt conflicts with its source')
    } else {
      if ((await readdir(directory)).length >= config.goalMaxRounds) {
        await Reflect.apply(originalGoalPause, goals, [coordinator, { id: goal.id, revision: goal.revision }])
        return { status: 'boundary_limit' }
      }
      await writeReceipt(path, { goal_id: goal.id, goal_revision: goal.revision,
        end_seq: end.seq, request_id: requestId, status: 'pending' }, true)
    }
    if (epoch !== resultEpoch || !canReceiveResult(coordinator)) return { status: 'suppressed' }
    const text = `Coordinator turn ${end.data.turn} ended at seq ${end.seq} only because it reached the declared per-turn request cap. Continue the unfinished bookkeeping for the same Goal using saved tool results and exact references; do not repeat completed actions, poll workers, reset the budget, or retry infrastructure/provider errors. If an A recovery ticket is already prepared, dispatch that exact ticket; otherwise use the existing finite A/B continuation mechanisms when their contracts permit. End the turn after dispatch and await the receipt-backed worker notice. Complete only after check_goal_evidence.objective_met=true; otherwise report gaps and pause.`
    await Reflect.apply(originalPrompt, sessions, [{ sessionId: coordinator.session.id,
      requestId, mode: 'queue', content: [{ type: 'text', text }] }, combinedSignal()])
    await writeReceipt(path, { goal_id: goal.id, goal_revision: goal.revision,
      end_seq: end.seq, request_id: requestId, status: 'accepted' })
    return { status: 'accepted', request_id: requestId }
  }

  function collectCoordinatorBoundary(seq) {
    const epoch = resultEpoch
    const operation = resultDelivery.then(() => deliverCoordinatorBoundary(seq, epoch))
    resultDelivery = operation
    return track(operation)
  }

  function createGoal(agent, request, originalCreate) {
    if (disposed) reject('research Host is disposing', 'factor-research/admission-required')
    requireCoordinatorGoalAgent(agent, 'Goal creation')
    if (boundRoot === undefined) {
      reject('research Host configuration is not ready', 'factor-research/admission-required')
    }
    if (!Number.isSafeInteger(config.goalMaxRounds) || config.goalMaxRounds < 1) {
      reject('a finite DSH Goal round cap must be bound in the research profile',
        'factor-research/admission-required')
    }
    const { contract } = readBoundResearchRun()
    if (request?.objective !== contract.objective) {
      reject('DSH Goal objective must match the fixed research objective',
        'factor-research/admission-required')
    }
    if (typeof goals?.get !== 'function') {
      reject('native DSH Goal service is not ready', 'factor-research/admission-required')
    }
    if (goals.get(agent) !== undefined) {
      reject('this fixed research run already has a native DSH Goal',
        'factor-research/admission-required')
    }
    const maxGoalRounds = request.maxGoalRounds ?? config.goalMaxRounds
    if (!Number.isSafeInteger(maxGoalRounds) || maxGoalRounds < 1
      || maxGoalRounds > config.goalMaxRounds) {
      reject('DSH Goal maxGoalRounds exceeds the configured finite run cap',
        'factor-research/admission-required')
    }
    return Reflect.apply(originalCreate, goals, [agent, { ...request, maxGoalRounds }])
  }

  function rejectGoalEdit() {
    reject('research Goal edits require a versioned human change ticket',
      'factor-research/admission-required')
  }

  function rejectGoalClear() {
    reject('clearing the fixed research Goal is disabled', 'factor-research/admission-required')
  }

  async function dispatchOnce(taskId, agent, signal) {
    if (disposed) reject('research Host is disposing', 'factor-research/admission-required')
    if (agent === undefined || agents.get(agent.id) !== agent || agent.status !== 'running'
      || agents.currentInitiator() !== agent || !isCoordinator(agent.session.id)) {
      reject('A task dispatch requires the active bound Coordinator Agent', 'factor-research/admission-required')
    }
    await initialize()
    await requireRunActive('A')
    requireGoalDispatchActive(agent)
    const actionSignal = combinedSignal(signal)
    actionSignal.throwIfAborted()
    const targetSessionId = [...roleBySession.entries()].find(([, role]) => role === 'A')?.[0]
    if (targetSessionId === undefined) reject('the A Session is not bound to this research', 'factor-research/admission-required')
    const boundRun = readBoundResearchRun()
    const { root, text, ticket, recoveryReceipt } = await readTicket(config.runRoot, taskId)
    if (ticket.research_id !== boundRun.binding.research_id
      || ticket.contract_sha256 !== boundRun.binding.contract_sha256
      || !sameJson(ticket.target, boundRun.binding.target)
      || !sameJson(ticket.goal, boundRun.binding.goal)) {
      reject('A task ticket does not match the fixed research binding', 'factor-research/admission-required')
    }
    if (recoveryReceipt !== undefined && recoveryReceipt.session_id !== targetSessionId) {
      reject('parent A dispatch receipt conflicts with the current bound Session')
    }
    actionSignal.throwIfAborted()

    const receiptDirectory = join(root, 'dispatch-receipts', 'A')
    await mkdir(receiptDirectory, { recursive: true })
    const actualReceiptDirectory = await realpath(receiptDirectory)
    if (!within(root, actualReceiptDirectory) || actualReceiptDirectory !== receiptDirectory) {
      reject('A dispatch receipt directory escaped the bound run root')
    }
    const receiptPath = join(actualReceiptDirectory, `${taskId}.json`)
    let receipt
    if (await exists(receiptPath)) {
      const stat = await lstat(receiptPath)
      if (!stat.isFile() || stat.isSymbolicLink()) reject('A dispatch receipt is not a regular file')
      receipt = JSON.parse(await readFile(receiptPath, 'utf8'))
      if (receipt.format_version !== 1 || receipt.task_id !== taskId || receipt.ticket_sha256 !== taskId
        || receipt.target_role !== 'A' || receipt.session_id !== targetSessionId
        || typeof receipt.request_id !== 'string' || !['pending', 'accepted'].includes(receipt.status)) {
        reject('saved A dispatch receipt conflicts with the current binding')
      }
      if (receipt.status === 'accepted') return { task_id: taskId, status: 'accepted', target_role: 'A' }
    } else {
      receipt = {
        format_version: 1,
        task_id: taskId,
        ticket_sha256: taskId,
        target_role: 'A',
        session_id: targetSessionId,
        request_id: randomUUID(),
        status: 'pending',
      }
      await writeReceipt(receiptPath, receipt, true)
    }

    actionSignal.throwIfAborted()
    const inspection = await sessions.inspect(targetSessionId, actionSignal)
    const matching = acceptedEvent(inspection.events, receipt.request_id, text)
    if (matching !== undefined) {
      receipt.status = 'accepted'
      await writeReceipt(receiptPath, receipt)
      return { task_id: taskId, status: 'accepted', target_role: 'A' }
    }
    const conflicting = inspection.events.find(event => event.type === 'user/message'
      && event.data?.source?.kind === 'user' && event.data.source.rpcId === receipt.request_id)
    if (conflicting !== undefined) reject('A dispatch request ID is already bound to different content')

    requireGoalDispatchActive(agent)
    const result = await Reflect.apply(originalPrompt, sessions, [{
      requestId: receipt.request_id,
      sessionId: targetSessionId,
      mode: 'queue',
      content: [{ type: 'text', text }],
    }, actionSignal])
    if (result?.accepted !== true) reject('DSH did not confirm acceptance of the A task ticket', 'factor-research/dispatch-rejected')
    receipt.status = 'accepted'
    await writeReceipt(receiptPath, receipt)
    return { task_id: taskId, status: 'accepted', target_role: 'A' }
  }

  async function continueAOnce(taskId, attempt, agent, signal) {
    if (disposed) reject('research Host is disposing', 'factor-research/admission-required')
    if (!Number.isSafeInteger(attempt) || attempt < 1 || attempt > 3) {
      reject('resume_a_task attempt must be 1, 2, or 3', 'factor-research/admission-required')
    }
    if (agent === undefined || agents.get(agent.id) !== agent || agent.status !== 'running'
      || agents.currentInitiator() !== agent || !isCoordinator(agent.session.id)) {
      reject('A continuation requires the active bound Coordinator Agent', 'factor-research/admission-required')
    }
    await initialize()
    await requireRunActive('A')
    requireGoalDispatchActive(agent)
    const actionSignal = combinedSignal(signal)
    actionSignal.throwIfAborted()
    const targetSessionId = [...roleBySession.entries()].find(([, role]) => role === 'A')?.[0]
    if (targetSessionId === undefined) reject('the A Session is not bound to this research', 'factor-research/admission-required')
    const boundRun = readBoundResearchRun()
    const current = await readTicket(config.runRoot, taskId)
    if (current.root !== boundRoot
      || current.ticket.research_id !== boundRun.binding.research_id
      || current.ticket.contract_sha256 !== boundRun.binding.contract_sha256
      || !sameJson(current.ticket.target, boundRun.binding.target)
      || !sameJson(current.ticket.goal, boundRun.binding.goal)) {
      reject('A task ticket does not match the fixed research binding', 'factor-research/admission-required')
    }

    const receiptDirectory = join(current.root, 'dispatch-receipts', 'A')
    if (!await exists(receiptDirectory)) {
      reject('A continuation requires an accepted A task receipt', 'factor-research/admission-required')
    }
    const actualReceiptDirectory = await realpath(receiptDirectory)
    if (!within(current.root, actualReceiptDirectory) || actualReceiptDirectory !== receiptDirectory) {
      reject('A dispatch receipt directory escaped the bound run root')
    }
    const inspection = await sessions.inspect(targetSessionId, actionSignal)
    const allowedRequestIds = new Set()
    let currentDispatchReceipt
    let currentDispatchEvent
    const names = await readdir(actualReceiptDirectory)
    for (const name of names) {
      const dispatchMatch = /^([a-f0-9]{64})\.json$/.exec(name)
      if (dispatchMatch === null) continue
      const dispatchTaskId = dispatchMatch[1]
      const receiptPath = join(actualReceiptDirectory, name)
      const receiptStat = await lstat(receiptPath)
      if (!receiptStat.isFile() || receiptStat.isSymbolicLink()) reject('A dispatch receipt is not a regular file')
      const receipt = JSON.parse(await readFile(receiptPath, 'utf8'))
      if (!sameJson(Object.keys(receipt).sort(), [
        'format_version', 'task_id', 'ticket_sha256', 'target_role', 'session_id', 'request_id', 'status',
      ].sort())
        || receipt.format_version !== 1 || receipt.task_id !== dispatchTaskId
        || receipt.ticket_sha256 !== dispatchTaskId || receipt.target_role !== 'A'
        || receipt.session_id !== targetSessionId || typeof receipt.request_id !== 'string'
        || receipt.status !== 'accepted') {
        reject('A continuation requires every saved A dispatch receipt to be accepted', 'factor-research/admission-required')
      }
      const saved = await readTicket(current.root, dispatchTaskId, { checkCurrentAState: false })
      const accepted = acceptedEvent(inspection.events, receipt.request_id, saved.text)
      if (accepted === undefined) {
        reject('an accepted A task receipt is missing from the bound Session history', 'factor-research/admission-required')
      }
      allowedRequestIds.add(receipt.request_id)
      if (dispatchTaskId === taskId) {
        currentDispatchReceipt = receipt
        currentDispatchEvent = accepted
      }
    }
    if (currentDispatchReceipt === undefined || currentDispatchEvent === undefined) {
      reject('A continuation requires the accepted receipt for this saved task', 'factor-research/admission-required')
    }
    const dispatchIndex = inspection.events.indexOf(currentDispatchEvent)
    if (!inspection.events.slice(dispatchIndex + 1).some(event => event.type === 'assistant/message')) {
      reject('A Session has not finished a response to the accepted task', 'factor-research/admission-required')
    }

    const continuationReceipts = new Map()
    let currentContinuationReceipt
    let currentContinuationEvent
    for (const name of names) {
      const match = /^continuation-([a-f0-9]{64})-([1-3])\.json$/.exec(name)
      if (match === null) {
        if (!/^[a-f0-9]{64}\.json$/.test(name)) reject('A dispatch receipt directory contains an unknown entry')
        continue
      }
      const receiptTaskId = match[1]
      const receiptAttempt = Number(match[2])
      const receiptPath = join(actualReceiptDirectory, name)
      const stat = await lstat(receiptPath)
      if (!stat.isFile() || stat.isSymbolicLink()) reject('A continuation receipt is not a regular file')
      const receipt = JSON.parse(await readFile(receiptPath, 'utf8'))
      if (!sameJson(Object.keys(receipt).sort(), [
        'format_version', 'task_id', 'ticket_sha256', 'target_role', 'session_id', 'request_id', 'status',
      ].sort())
        || receipt.format_version !== 1 || receipt.task_id !== receiptTaskId
        || receipt.ticket_sha256 !== receiptTaskId || receipt.target_role !== 'A'
        || receipt.session_id !== targetSessionId || typeof receipt.request_id !== 'string'
        || !['pending', 'accepted'].includes(receipt.status)) {
        reject('saved A continuation receipt conflicts with the bound Session')
      }
      const event = acceptedEvent(inspection.events, receipt.request_id,
        aContinuationText(receiptTaskId, receiptAttempt))
      if (receipt.status === 'accepted' && event === undefined) {
        reject('an accepted A continuation receipt is missing from the bound Session history',
          'factor-research/admission-required')
      }
      if (receipt.status === 'pending') {
        if (event !== undefined) {
          receipt.status = 'accepted'
          await writeReceipt(receiptPath, receipt)
        } else if (receiptTaskId !== taskId || receiptAttempt !== attempt) {
          reject('a prior A continuation receipt is still pending', 'factor-research/admission-required')
        }
      }
      continuationReceipts.set(`${receiptTaskId}:${receiptAttempt}`, { receipt, event, receiptPath })
      if (event !== undefined) allowedRequestIds.add(receipt.request_id)
      if (receiptTaskId === taskId && receiptAttempt === attempt) {
        currentContinuationReceipt = receipt
        currentContinuationEvent = event
      }
    }
    for (let priorAttempt = 1; priorAttempt < attempt; priorAttempt += 1) {
      const prior = continuationReceipts.get(`${taskId}:${priorAttempt}`)
      if (prior === undefined || prior.receipt.status !== 'accepted' || prior.event === undefined) {
        reject(`A continuation attempt ${attempt} requires accepted attempt ${priorAttempt}`,
          'factor-research/admission-required')
      }
    }

    const additional = inspection.events.filter(event => event.type === 'user/message'
      && event.data?.source?.kind === 'user'
      && !allowedRequestIds.has(event.data.source.rpcId))
    if (additional.length > 0) reject('A Session has an unreceipted user message after its saved tickets', 'factor-research/admission-required')
    const currentAgent = agents.get(targetSessionId)
    if (currentAgent?.status === 'running') reject('A Session is still processing the accepted ticket', 'factor-research/admission-required')

    const continuationPath = aContinuationReceiptPath(actualReceiptDirectory, taskId, attempt)
    let receipt = currentContinuationReceipt
    if (receipt === undefined) {
      receipt = {
        format_version: 1,
        task_id: taskId,
        ticket_sha256: taskId,
        target_role: 'A',
        session_id: targetSessionId,
        request_id: randomUUID(),
        status: 'pending',
      }
      await writeReceipt(continuationPath, receipt, true)
    } else if (receipt.status === 'accepted') {
      return { task_id: taskId, status: 'accepted', target_role: 'A' }
    } else if (currentContinuationEvent !== undefined) {
      receipt.status = 'accepted'
      await writeReceipt(continuationPath, receipt)
      return { task_id: taskId, status: 'accepted', target_role: 'A' }
    } else {
      const conflicting = inspection.events.find(event => event.type === 'user/message'
        && event.data?.source?.kind === 'user' && event.data.source.rpcId === receipt.request_id)
      if (conflicting !== undefined) reject('A continuation request ID is already bound to different content')
    }

    actionSignal.throwIfAborted()
    requireGoalDispatchActive(agent)
    const result = await Reflect.apply(originalPrompt, sessions, [{
      requestId: receipt.request_id,
      sessionId: targetSessionId,
      mode: 'queue',
      content: [{ type: 'text', text: aContinuationText(taskId, attempt) }],
    }, actionSignal])
    if (result?.accepted !== true) reject('DSH did not confirm acceptance of the A continuation', 'factor-research/dispatch-rejected')
    receipt.status = 'accepted'
    await writeReceipt(continuationPath, receipt)
    return { task_id: taskId, status: 'accepted', target_role: 'A' }
  }

  async function dispatchBOnce(taskId, agent, signal) {
    if (disposed) reject('research Host is disposing', 'factor-research/admission-required')
    if (agent === undefined || agents.get(agent.id) !== agent || agent.status !== 'running'
      || agents.currentInitiator() !== agent || !isCoordinator(agent.session.id)) {
      reject('B task dispatch requires the active bound Coordinator Agent', 'factor-research/admission-required')
    }
    await initialize()
    await requireRunActive('B')
    requireGoalDispatchActive(agent)
    const actionSignal = combinedSignal(signal)
    actionSignal.throwIfAborted()
    const targetSessionId = [...roleBySession.entries()].find(([, role]) => role === 'B')?.[0]
    if (targetSessionId === undefined) reject('the B Session is not bound to this research', 'factor-research/admission-required')
    const boundRun = readBoundResearchRun()
    const { root, text } = await readBTicket(config.runRoot, taskId, boundRun)
    if (root !== boundRoot) reject('B task ticket does not belong to the fixed research run', 'factor-research/admission-required')
    actionSignal.throwIfAborted()

    const receiptBase = join(root, 'dispatch-receipts')
    if (await exists(receiptBase)) {
      const info = await lstat(receiptBase)
      if (!info.isDirectory() || info.isSymbolicLink()) reject('B dispatch receipt parent is not a regular directory')
    } else {
      await mkdir(receiptBase)
    }
    const receiptDirectory = join(receiptBase, 'B')
    if (await exists(receiptDirectory)) {
      const info = await lstat(receiptDirectory)
      if (!info.isDirectory() || info.isSymbolicLink()) reject('B dispatch receipt directory is not a regular directory')
    } else {
      await mkdir(receiptDirectory)
    }
    const actualReceiptDirectory = await realpath(receiptDirectory)
    if (!within(root, actualReceiptDirectory) || actualReceiptDirectory !== receiptDirectory) {
      reject('B dispatch receipt directory escaped the bound run root')
    }
    const receiptPath = join(actualReceiptDirectory, 'dispatch.json')
    let receipt
    if (await exists(receiptPath)) {
      const stat = await lstat(receiptPath)
      if (!stat.isFile() || stat.isSymbolicLink()) reject('B dispatch receipt is not a regular file')
      receipt = JSON.parse(await readFile(receiptPath, 'utf8'))
      if (!sameJson(Object.keys(receipt).sort(), [
        'format_version', 'task_id', 'ticket_sha256', 'target_role', 'session_id', 'request_id', 'status',
      ].sort())
        || receipt.format_version !== 1 || receipt.task_id !== taskId
        || receipt.ticket_sha256 !== taskId || receipt.target_role !== 'B'
        || receipt.session_id !== targetSessionId
        || typeof receipt.request_id !== 'string' || !['pending', 'accepted'].includes(receipt.status)) {
        reject('saved B dispatch receipt conflicts with the current frozen ticket')
      }
      if (receipt.status === 'accepted') return { task_id: taskId, status: 'accepted', target_role: 'B' }
    } else {
      if (await exists(join(root, 'b-access-started.json'))
        || await exists(join(root, 'b-numerical-complete.json'))
        || await exists(join(root, 'validation.json'))) {
        reject('B task ticket cannot be delivered after B access or delivery', 'factor-research/admission-required')
      }
      receipt = {
        format_version: 1,
        task_id: taskId,
        ticket_sha256: taskId,
        target_role: 'B',
        session_id: targetSessionId,
        request_id: randomUUID(),
        status: 'pending',
      }
      await writeReceipt(receiptPath, receipt, true)
    }

    actionSignal.throwIfAborted()
    const inspection = await sessions.inspect(targetSessionId, actionSignal)
    const matching = acceptedEvent(inspection.events, receipt.request_id, text)
    if (matching !== undefined) {
      receipt.status = 'accepted'
      await writeReceipt(receiptPath, receipt)
      return { task_id: taskId, status: 'accepted', target_role: 'B' }
    }
    const conflicting = inspection.events.find(event => event.type === 'user/message'
      && event.data?.source?.kind === 'user' && event.data.source.rpcId === receipt.request_id)
    if (conflicting !== undefined) reject('B dispatch request ID is already bound to different content')
    if (await exists(join(root, 'b-access-started.json'))
      || await exists(join(root, 'b-numerical-complete.json'))
      || await exists(join(root, 'validation.json'))) {
      reject('B access started without a matching accepted dispatch receipt', 'factor-research/admission-required')
    }

    actionSignal.throwIfAborted()
    requireGoalDispatchActive(agent)
    const result = await Reflect.apply(originalPrompt, sessions, [{
      requestId: receipt.request_id,
      sessionId: targetSessionId,
      mode: 'queue',
      content: [{ type: 'text', text }],
    }, actionSignal])
    if (result?.accepted !== true) reject('DSH did not confirm acceptance of the B task ticket', 'factor-research/dispatch-rejected')
    receipt.status = 'accepted'
    await writeReceipt(receiptPath, receipt)
    return { task_id: taskId, status: 'accepted', target_role: 'B' }
  }

  async function continueBOnce(taskId, attempt, agent, signal) {
    if (disposed) reject('research Host is disposing', 'factor-research/admission-required')
    if (!Number.isSafeInteger(attempt) || attempt < 1 || attempt > 3) {
      reject('resume_b_delivery attempt must be 1, 2, or 3', 'factor-research/admission-required')
    }
    if (agent === undefined || agents.get(agent.id) !== agent || agent.status !== 'running'
      || agents.currentInitiator() !== agent || !isCoordinator(agent.session.id)) {
      reject('B delivery continuation requires the active bound Coordinator Agent', 'factor-research/admission-required')
    }
    await initialize()
    await requireRunActive('B')
    requireGoalDispatchActive(agent)
    const actionSignal = combinedSignal(signal)
    actionSignal.throwIfAborted()
    const targetSessionId = [...roleBySession.entries()].find(([, role]) => role === 'B')?.[0]
    if (targetSessionId === undefined) reject('the B Session is not bound to this research', 'factor-research/admission-required')
    const boundRun = readBoundResearchRun()
    const { root, text, ticket } = await readBTicket(config.runRoot, taskId, boundRun)
    if (root !== boundRoot) reject('B task ticket does not belong to the fixed research run', 'factor-research/admission-required')

    const receiptDirectory = join(root, 'dispatch-receipts', 'B')
    if (!await exists(receiptDirectory)) {
      reject('B delivery continuation requires the accepted bound B ticket receipt', 'factor-research/admission-required')
    }
    const actualReceiptDirectory = await realpath(receiptDirectory)
    if (!within(root, actualReceiptDirectory) || actualReceiptDirectory !== receiptDirectory) {
      reject('B dispatch receipt directory escaped the bound run root')
    }
    const dispatchReceiptPath = join(actualReceiptDirectory, 'dispatch.json')
    if (!await exists(dispatchReceiptPath)) {
      reject('B delivery continuation requires the accepted bound B ticket receipt', 'factor-research/admission-required')
    }
    const dispatchInfo = await lstat(dispatchReceiptPath)
    if (!dispatchInfo.isFile() || dispatchInfo.isSymbolicLink()) reject('B dispatch receipt is not a regular file')
    const dispatchReceipt = JSON.parse(await readFile(dispatchReceiptPath, 'utf8'))
    if (!sameJson(Object.keys(dispatchReceipt).sort(), [
      'format_version', 'task_id', 'ticket_sha256', 'target_role', 'session_id', 'request_id', 'status',
    ].sort())
      || dispatchReceipt.format_version !== 1 || dispatchReceipt.task_id !== taskId
      || dispatchReceipt.ticket_sha256 !== taskId || dispatchReceipt.target_role !== 'B'
      || dispatchReceipt.session_id !== targetSessionId || dispatchReceipt.status !== 'accepted'
      || typeof dispatchReceipt.request_id !== 'string') {
      reject('B delivery continuation requires the accepted bound B ticket receipt', 'factor-research/admission-required')
    }

    const checkpoint = readRegularJson(join(root, 'b-numerical-complete.json'), 'completed B numerical checkpoint')
    const access = readRegularJson(join(root, 'b-access-started.json'), 'B access marker')
    const candidateIds = Object.keys(ticket.frozen_batch.candidates)
    const retainedHorizons = Object.fromEntries(candidateIds.map(candidateId => [
      candidateId, ticket.frozen_batch.candidates[candidateId].retained_horizons,
    ]))
    if (!sameJson(checkpoint.candidate_ids, candidateIds)
      || !sameJson(checkpoint.retained_horizons, retainedHorizons)
      || access === null || typeof access !== 'object' || Array.isArray(access)
      || !sameJson(access.candidate_ids, candidateIds)
      || access.use !== 'fixed_batch_validation' || typeof access.time !== 'string') {
      reject('B continuation requires the complete saved checkpoint for the exact frozen batch', 'factor-research/admission-required')
    }

    const priorReceipts = []
    for (let priorAttempt = 1; priorAttempt < attempt; priorAttempt += 1) {
      const priorPath = bContinuationReceiptPath(actualReceiptDirectory, priorAttempt)
      if (!await exists(priorPath)) {
        reject(`B continuation attempt ${attempt} requires accepted attempt ${priorAttempt}`, 'factor-research/admission-required')
      }
      const priorInfo = await lstat(priorPath)
      if (!priorInfo.isFile() || priorInfo.isSymbolicLink()) reject('prior B continuation receipt is not a regular file')
      const prior = JSON.parse(await readFile(priorPath, 'utf8'))
      if (!sameJson(Object.keys(prior).sort(), [
        'format_version', 'task_id', 'ticket_sha256', 'target_role', 'session_id', 'request_id', 'status',
      ].sort())
        || prior.format_version !== 1 || prior.task_id !== taskId
        || prior.ticket_sha256 !== taskId || prior.target_role !== 'B'
        || prior.session_id !== targetSessionId
        || typeof prior.request_id !== 'string' || prior.status !== 'accepted') {
        reject(`B continuation attempt ${attempt} has a mismatched prior receipt`, 'factor-research/admission-required')
      }
      priorReceipts.push(prior)
    }

    const continuationPath = bContinuationReceiptPath(actualReceiptDirectory, attempt)
    let receipt
    if (await exists(continuationPath)) {
      const stat = await lstat(continuationPath)
      if (!stat.isFile() || stat.isSymbolicLink()) reject('B continuation receipt is not a regular file')
      receipt = JSON.parse(await readFile(continuationPath, 'utf8'))
      if (!sameJson(Object.keys(receipt).sort(), [
        'format_version', 'task_id', 'ticket_sha256', 'target_role', 'session_id', 'request_id', 'status',
      ].sort())
        || receipt.format_version !== 1 || receipt.task_id !== taskId
        || receipt.ticket_sha256 !== taskId || receipt.target_role !== 'B'
        || receipt.session_id !== targetSessionId
        || typeof receipt.request_id !== 'string' || !['pending', 'accepted'].includes(receipt.status)) {
        reject('saved B continuation receipt conflicts with the frozen ticket')
      }
      if (receipt.status === 'accepted') return { task_id: taskId, status: 'accepted', target_role: 'B' }
    } else {
      if (await exists(join(root, 'b-delivery.json')) || await exists(join(root, 'validation.json'))) {
        reject('B delivery is already complete', 'factor-research/admission-required')
      }
      receipt = {
        format_version: 1,
        task_id: taskId,
        ticket_sha256: taskId,
        target_role: 'B',
        session_id: targetSessionId,
        request_id: randomUUID(),
        status: 'pending',
      }
      await writeReceipt(continuationPath, receipt, true)
    }

    actionSignal.throwIfAborted()
    const inspection = await sessions.inspect(targetSessionId, actionSignal)
    const initialTicket = acceptedEvent(inspection.events, dispatchReceipt.request_id, text)
    if (initialTicket === undefined) reject('accepted B ticket is missing from the bound Session history', 'factor-research/admission-required')
    const matching = acceptedEvent(inspection.events, receipt.request_id,
      continuationText(taskId, attempt, candidateIds, retainedHorizons))
    if (matching !== undefined) {
      receipt.status = 'accepted'
      await writeReceipt(continuationPath, receipt)
      return { task_id: taskId, status: 'accepted', target_role: 'B' }
    }
    const conflicting = inspection.events.find(event => event.type === 'user/message'
      && event.data?.source?.kind === 'user' && event.data.source.rpcId === receipt.request_id)
    if (conflicting !== undefined) reject('B continuation request ID is already bound to different content')
    for (const prior of priorReceipts) {
      if (!inspection.events.some(event => event.type === 'user/message'
        && event.data?.source?.kind === 'user'
        && event.data.source.rpcId === prior.request_id)) {
        reject('accepted B continuation is missing from the bound Session history', 'factor-research/admission-required')
      }
    }
    if (await exists(join(root, 'b-delivery.json')) || await exists(join(root, 'validation.json'))) {
      reject('B delivery is already complete', 'factor-research/admission-required')
    }
    const allowedRequestIds = new Set([
      dispatchReceipt.request_id, receipt.request_id,
      ...priorReceipts.map(prior => prior.request_id),
    ])
    const additional = inspection.events.filter(event => event.type === 'user/message'
      && event.data?.source?.kind === 'user'
      && !allowedRequestIds.has(event.data.source.rpcId))
    if (additional.length > 0) reject('B Session has an unreceipted user message after the saved ticket', 'factor-research/admission-required')
    const currentAgent = agents.get(targetSessionId)
    if (currentAgent?.status === 'running') reject('B Session is still processing the accepted ticket', 'factor-research/admission-required')

    actionSignal.throwIfAborted()
    requireGoalDispatchActive(agent)
    const result = await Reflect.apply(originalPrompt, sessions, [{
      requestId: receipt.request_id,
      sessionId: targetSessionId,
      mode: 'queue',
      content: [{ type: 'text', text: continuationText(taskId, attempt, candidateIds, retainedHorizons) }],
    }, actionSignal])
    if (result?.accepted !== true) reject('DSH did not confirm acceptance of the B continuation', 'factor-research/dispatch-rejected')
    receipt.status = 'accepted'
    await writeReceipt(continuationPath, receipt)
    return { task_id: taskId, status: 'accepted', target_role: 'B' }
  }

  function continuationText(taskId, attempt, candidateIds, retainedHorizons) {
    return `Resume the already accepted frozen B task ${taskId} (Host continuation ${attempt}). The complete numerical checkpoint is saved for candidates ${candidateIds.join(', ')} with retained horizons ${JSON.stringify(retainedHorizons)}. Use the existing checkpoint and B evidence to save the required interpretation for every frozen candidate, then finalize B delivery. Use only the bound B domain tools; MCP resource listings are not research evidence. Do not add, remove, or alter candidates, formulas, directions, or horizons; do not start another B market-data access. When evidence is insufficient, state that limitation in the report and finalize the delivery.`
  }

  function readRegularJson(path, label) {
    let info
    try {
      info = lstatSync(path)
    } catch {
      reject(`${label} is unavailable`, 'factor-research/evidence-required')
    }
    if (!info.isFile() || info.isSymbolicLink()) {
      reject(`${label} is not a regular file`, 'factor-research/evidence-required')
    }
    try {
      return JSON.parse(readFileSync(path, 'utf8'))
    } catch {
      reject(`${label} is invalid`, 'factor-research/evidence-required')
    }
  }

  function readRegularBytes(path, label) {
    let info
    try {
      info = lstatSync(path)
    } catch {
      reject(`${label} is unavailable`, 'factor-research/evidence-required')
    }
    if (!info.isFile() || info.isSymbolicLink()) {
      reject(`${label} is not a regular file`, 'factor-research/evidence-required')
    }
    try {
      return readFileSync(path)
    } catch {
      reject(`${label} is invalid`, 'factor-research/evidence-required')
    }
  }

  function completeGoal(agent, ref) {
    if (disposed) reject('research Host is disposing', 'factor-research/evidence-required')
    if (agent === undefined || agents.get(agent.id) !== agent || agent.status !== 'running'
      || agents.currentInitiator() !== agent || !isCoordinator(agent.session.id)) {
      reject('Goal completion requires the active bound Coordinator Agent', 'factor-research/evidence-required')
    }
    if (boundRoot === undefined || typeof verifyGoalEvidence !== 'function') {
      reject('Goal evidence verifier is not initialized', 'factor-research/evidence-required')
    }
    const goal = goals.get(agent)
    if (goal === undefined || goal.phase !== 'active'
      || ref?.id !== goal.id || ref?.revision !== goal.revision) {
      reject('Goal completion requires the current active Goal id and revision', 'factor-research/evidence-required')
    }
    const contractBytes = readRegularBytes(join(boundRoot, 'contract.json'), 'fixed research contract')
    let contract
    try {
      contract = JSON.parse(contractBytes.toString('utf8'))
    } catch {
      reject('fixed research contract is invalid', 'factor-research/evidence-required')
    }
    const goalRecord = readRegularJson(join(boundRoot, 'a_records', 'goal-spec.json'), 'fixed domain Goal record')
    const verification = verifyGoalEvidence()
    const evidence = verification?.evidence
    if (verification?.run_root !== boundRoot
      || verification?.research_id !== contract.run_id
      || verification?.contract_sha256 !== sha256(contractBytes)
      || verification?.goal?.objective !== contract.objective
      || verification?.goal?.goal_id !== goalRecord?.data?.goal_id
      || verification?.goal?.objective !== goalRecord?.data?.objective
      || verification?.goal?.objective !== goal.objective
      || goalRecord?.id !== 'goal-spec' || goalRecord?.kind !== 'goal_contract'
      || evidence?.status !== 'complete' || evidence?.objective_met !== true
      || !Array.isArray(evidence?.gaps) || evidence.gaps.length !== 0
      || !Array.isArray(evidence?.qualified_idea_ids) || evidence.qualified_idea_ids.length === 0) {
      reject('saved research evidence does not authorize Goal completion', 'factor-research/evidence-required')
    }
    const current = goals.get(agent)
    if (current === undefined || current.id !== goal.id || current.revision !== ref.revision
      || current.objective !== goal.objective || current.phase !== 'active') {
      reject('Goal changed while its domain evidence was being verified', 'factor-research/evidence-required')
    }
    return Reflect.apply(originalComplete, goals, [agent, { id: goal.id, revision: goal.revision }])
  }

  return {
    initialize,
    dispose,
    readyForDomainTools,
    isCoordinator,
    roleForSession,
    requireRunActive,
    stopRun,
    resumeRun,
    runtimeStatus,
    async readAActionRequest(actionId, agent) {
      if (agent === undefined || agents.get(agent.id) !== agent || agent.status !== 'running'
        || agents.currentInitiator() !== agent || roleForSession(agent.session.id) !== 'A') {
        reject('original A action requests require the active bound A Agent')
      }
      if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$/.test(actionId)) reject('invalid A action identifier')
      await initialize()
      const receipt = readRegularJson(join(boundRoot, 'action_receipts', `${actionId}.started.json`),
        'original A action receipt')
      if (receipt.action_id !== actionId || !/^[a-f0-9]{64}$/.test(receipt.parameters_sha256 ?? '')) {
        reject('original A action receipt is malformed')
      }
      const inspection = await sessions.inspect(agent.session.id)
      for (const event of inspection.events) {
        if (event.type !== 'tool/call' || event.data.name !== `mcp__fr_a__${receipt.name}`) continue
        let parsed
        try { parsed = parseTicket(event.data.arguments) }
        catch (error) {
          if (error instanceof SyntaxError) continue // An invalid model call never entered the domain ledger.
          throw error
        }
        const { ticket: args, numberSources } = parsed
        if (args.action_id !== actionId) continue
        let parameters = args
        if (['evaluate_candidate', 'repair_candidate'].includes(receipt.name)) {
          // The domain binds numerical actions to the fixed A panel as well as
          // model arguments. This field is never supplied by the model.
          const input = readRegularJson(join(boundRoot, 'a-action-input-binding.json'), 'fixed A action input')
          if (input.run_id !== readBoundResearchRun().contract.run_id || !/^[a-f0-9]{64}$/.test(input.panel_sha256 ?? '')) {
            reject('fixed A action input differs from this research run')
          }
          parameters = { ...args, a_panel_sha256: input.panel_sha256 }
          if (receipt.name === 'evaluate_candidate' && parameters.experiment_id === undefined) parameters.experiment_id = null
        }
        const parametersHash = sha256(canonicalJson(parameters, numberSources, undefined, undefined, new Set(['action_id'])))
        if (parametersHash !== receipt.parameters_sha256) continue
        const result = { action_id: actionId, tool_name: event.data.name, arguments: args,
          source: { session_id: agent.session.id, seq: event.seq, parameters_sha256: parametersHash } }
        if (Buffer.byteLength(JSON.stringify(result)) > 16 * 1024) reject('original A action request exceeds 16 KiB')
        return result
      }
      reject('the original A action parameters cannot be verified from the bound A Session')
    },
    collectAResult(seq, options) { return collectResult('A', seq, options) },
    collectBResult(seq, options) { return collectResult('B', seq, options) },
    collectCoordinatorBoundary,
    canReceiveResult,
    suspendResults() { resultEpoch += 1 },
    dispatchA(taskId, agent, signal) {
      if (disposed) reject('research Host is disposing', 'factor-research/admission-required')
      if (typeof taskId !== 'string') reject('dispatch_a_ticket requires a task_id reference')
      const operationId = `dispatch:${taskId}`
      if (inFlight.has('A')) {
        if (inFlightAOperation !== operationId) reject('another A task operation is already in progress')
        return inFlight.get('A')
      }
      inFlightAOperation = operationId
      const task = track(dispatchOnce(taskId, agent, signal)).finally(() => {
        inFlight.delete('A')
        inFlightAOperation = undefined
      })
      inFlight.set('A', task)
      return task
    },
    continueATask(taskId, attempt, agent, signal) {
      if (disposed) reject('research Host is disposing', 'factor-research/admission-required')
      if (typeof taskId !== 'string') reject('resume_a_task requires a task_id reference')
      if (!Number.isSafeInteger(attempt) || attempt < 1 || attempt > 3) {
        reject('resume_a_task attempt must be 1, 2, or 3', 'factor-research/admission-required')
      }
      const operationId = `continue:${taskId}:${attempt}`
      if (inFlight.has('A')) {
        if (inFlightAOperation !== operationId) reject('another A task operation is already in progress')
        return inFlight.get('A')
      }
      inFlightAOperation = operationId
      const task = track(continueAOnce(taskId, attempt, agent, signal)).finally(() => {
        inFlight.delete('A')
        inFlightAOperation = undefined
      })
      inFlight.set('A', task)
      return task
    },
    dispatchB(taskId, agent, signal) {
      if (disposed) reject('research Host is disposing', 'factor-research/admission-required')
      if (typeof taskId !== 'string') reject('dispatch_b_ticket requires a task_id reference')
      if (inFlight.has('B')) {
        if (inFlightBTaskId !== taskId) reject('a different B task ticket is already being dispatched')
        return inFlight.get('B')
      }
      inFlightBTaskId = taskId
      const task = track(dispatchBOnce(taskId, agent, signal)).finally(() => {
        inFlight.delete('B')
        inFlightBTaskId = undefined
      })
      inFlight.set('B', task)
      return task
    },
    continueBDelivery(taskId, attempt, agent, signal) {
      if (disposed) reject('research Host is disposing', 'factor-research/admission-required')
      if (typeof taskId !== 'string') reject('resume_b_delivery requires a task_id reference')
      if (!Number.isSafeInteger(attempt) || attempt < 1 || attempt > 3) {
        reject('resume_b_delivery attempt must be 1, 2, or 3', 'factor-research/admission-required')
      }
      if (inFlight.has('B_CONTINUE')) {
        if (inFlightBContinuationTaskId !== `${taskId}:${attempt}`) reject('a different B continuation is already in progress')
        return inFlight.get('B_CONTINUE')
      }
      inFlightBContinuationTaskId = `${taskId}:${attempt}`
      const task = track(continueBOnce(taskId, attempt, agent, signal)).finally(() => {
        inFlight.delete('B_CONTINUE')
        inFlightBContinuationTaskId = undefined
      })
      inFlight.set('B_CONTINUE', task)
      return task
    },
    createGoal(agent, request, originalCreate) {
      return createGoal(agent, request, originalCreate)
    },
    requireCoordinatorGoalAgent(agent, operation) {
      return requireCoordinatorGoalAgent(agent, operation)
    },
    rejectGoalEdit() {
      rejectGoalEdit()
    },
    rejectGoalClear() {
      rejectGoalClear()
    },
    completeGoal(agent, ref) {
      if (disposed) reject('research Host is disposing', 'factor-research/evidence-required')
      return completeGoal(agent, ref)
    },
  }
}

function readActionControlState(runRoot) {
  const controlRoot = join(runRoot, 'action_control')
  let info
  try {
    info = lstatSync(controlRoot)
  } catch (error) {
    if (error?.code === 'ENOENT') return { status: 'active', source: 'program_action_control' }
    throw error
  }
  if (!info.isDirectory() || info.isSymbolicLink() || realpathSync(controlRoot) !== controlRoot) {
    reject('research action-control directory is not a regular run-local directory',
      'factor-research/action-control-invalid')
  }
  const paths = readdirSync(controlRoot)
    .filter(name => /^\d{8}-.+\.json$/.test(name))
    .sort()
    .map(name => join(controlRoot, name))
  if (paths.length === 0) return { status: 'active', source: 'program_action_control' }
  const events = paths.map(path => {
    const eventInfo = lstatSync(path)
    if (!eventInfo.isFile() || eventInfo.isSymbolicLink() || realpathSync(path) !== path
      || !within(runRoot, path)) {
      reject('research action-control event is not a regular run-local file',
        'factor-research/action-control-invalid')
    }
    let event
    try {
      event = JSON.parse(readFileSync(path, 'utf8'))
    } catch (error) {
      reject(`research action-control event is invalid (${String(error)})`,
        'factor-research/action-control-invalid')
    }
    if (typeof event?.event_id !== 'string' || typeof event?.action_id !== 'string'
      || !['stop_requested', 'stop', 'resume'].includes(event?.event)
      || !path.endsWith(`/${event.event_id}.json`)) {
      reject('research action-control event identity is invalid',
        'factor-research/action-control-invalid')
    }
    return event
  })
  const latest = events.at(-1)
  const status = { stop_requested: 'stopping', stop: 'stopped', resume: 'active' }[latest.event]
  return { status, source: 'program_action_control', event_id: latest.event_id,
    action_id: latest.action_id, ...(typeof latest.reason === 'string' ? { reason: latest.reason } : {}) }
}

function runActionControl(config, action, { actionId, reason } = {}) {
  if (!Number.isSafeInteger(config.stopTimeoutMs) || config.stopTimeoutMs < 1) {
    reject('research Host requires a finite positive stopTimeoutMs',
      'factor-research/action-control-invalid')
  }
  const args = ['-m', 'crypto_quant.research.factor_mining.mcp_server',
    '--role', 'coordinator', '--binding', config.bindingPath,
    '--action-control', action]
  if (action !== 'status') {
    if (typeof actionId !== 'string' || actionId.length === 0) {
      reject('research action control requires a fixed action ID',
        'factor-research/action-control-invalid')
    }
    args.push('--action-id', actionId)
  }
  if (action === 'begin-stop' || action === 'finish-stop') {
    if (typeof reason !== 'string' || reason.trim().length === 0) {
      reject('research stop requires a nonempty reason',
        'factor-research/action-control-invalid')
    }
    args.push('--reason', reason)
  }
  const result = spawnSync(config.pythonExecutable, args, {
    cwd: config.projectRoot,
    env: { ...process.env, PYTHONPATH: join(config.projectRoot, 'src') },
    encoding: 'utf8',
    maxBuffer: 1024 * 1024,
    timeout: config.stopTimeoutMs,
  })
  if (result.error !== undefined) {
    if (result.error.code === 'ETIMEDOUT' && action === 'finish-stop') {
      return { status: 'stopping', source: 'program_action_control',
        action_id: actionId, reason, timed_out: true }
    }
    reject(`research action-control ${action} failed (${result.error.code ?? String(result.error)})`,
      'factor-research/action-control-failed')
  }
  if (result.status !== 0 || typeof result.stdout !== 'string') {
    reject(`research action-control ${action} failed with exit status ${String(result.status)}`,
      'factor-research/action-control-failed')
  }
  try {
    return JSON.parse(result.stdout)
  } catch {
    reject(`research action-control ${action} returned invalid JSON`,
      'factor-research/action-control-failed')
  }
}
