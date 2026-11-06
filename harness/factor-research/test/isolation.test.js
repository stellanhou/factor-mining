import assert from 'node:assert/strict'
import test from 'node:test'
import { isResearchToolAllowed } from '../src/isolation.js'

test('fixed role boundaries cover schemas, direct execution and cross-server resources', () => {
  for (const role of ['coordinator', 'A', 'B']) {
    assert.equal(isResearchToolAllowed(role, `mcp__fr_${role.toLowerCase()}__get_research_status`), true)
    for (const other of ['coordinator', 'A', 'B'].filter(other => other !== role)) {
      assert.equal(isResearchToolAllowed(role, `mcp__fr_${other.toLowerCase()}__get_research_status`), false)
      assert.equal(isResearchToolAllowed(role, 'read_mcp_resource', { server: `fr_${other.toLowerCase()}` }), false)
    }
    for (const name of ['bash', 'run_code', 'read_file', 'working_directory', 'list_sessions', 'plugin_manager']) {
      assert.equal(isResearchToolAllowed(role, name), false)
    }
  }
  assert.equal(isResearchToolAllowed('A', 'dispatch_a_ticket'), false)
  assert.equal(isResearchToolAllowed('B', 'update_goal'), false)
  assert.equal(isResearchToolAllowed('coordinator', 'update_goal'), true)
  assert.equal(isResearchToolAllowed(undefined, 'list_mcp_resources'), false)
})
