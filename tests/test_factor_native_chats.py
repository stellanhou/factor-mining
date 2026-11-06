"""Synthetic MCP binding and real SDK transport checks for native research tools."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mcp import Client, StdioServerParameters

from crypto_quant.research.factor_mining.goal import GoalSpec
from crypto_quant.research.factor_mining.native_chats import (
    NativeChatBinding,
    load_native_adapter,
)
from crypto_quant.research.factor_mining.mcp_server import create_server
from crypto_quant.research.factor_mining.native_changes import record_human_change
from crypto_quant.research.factor_mining.records import RecordStore
from crypto_quant.research.factor_mining.workflow import FactorMiner


ROOT = Path(__file__).resolve().parents[1]


class FailingModel:
    def complete(self, *args, **kwargs):
        raise AssertionError("native MCP tools must not call role models")


def _fixture_builder():
    path = ROOT / "scripts/prepare_native_chats_synthetic_acceptance.py"
    module_spec = importlib.util.spec_from_file_location("native_acceptance_fixture", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _binding(tmp_path: Path) -> Path:
    fixture = _fixture_builder()
    spec = replace(fixture._spec(), run_id="native-mcp-protocol-test",
                   objective="用固定六币合成行情验证MCP工具接线。", purpose="engineering_check")
    database = tmp_path / "market_data.sqlite"
    universe = tmp_path / "universe.csv"
    runs = tmp_path / "runs"
    runs.mkdir()
    idea_pool = tmp_path / "idea_pool"
    idea_pool.mkdir()
    fixture._create_market_database(database, spec)
    fixture._create_universe(universe, spec)

    miner = FactorMiner(spec, FailingModel(), runs)
    goal = GoalSpec(goal_id="native-mcp-protocol-test", objective=spec.objective,
                    target_ideas=1)
    miner.store.append("goal-spec", "goal_contract", goal.as_dict())
    contract_sha256 = hashlib.sha256((miner.root / "contract.json").read_bytes()).hexdigest()
    value = {
        "format_version": 1,
        "research_id": spec.run_id,
        "target": {"version": 1, "objective": spec.objective},
        "goal": goal.as_dict(),
        "run_root": f"runs/{spec.run_id}",
        "contract_sha256": contract_sha256,
        "inputs": {
            "market_db": database.name,
            "universe_csv": universe.name,
            "include_liquidations": False,
            "idea_pool": idea_pool.name,
        },
        "chats": {
            "coordinator": {"thread_id": "synthetic-coordinator", "host_id": "local"},
            "A": {"thread_id": "synthetic-a", "host_id": "local"},
            "B": {"thread_id": "synthetic-b", "host_id": "local"},
        },
    }
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return binding


def test_fixed_binding_and_real_mcp_tool_call(tmp_path):
    binding_path = _binding(tmp_path)
    binding = NativeChatBinding.load(binding_path)
    assert binding.research_id == "native-mcp-protocol-test"
    assert binding.spec.purpose == "engineering_check"
    assert binding.chats["A"].thread_id == "synthetic-a"

    adapter = load_native_adapter(binding_path, "A")
    server = create_server(adapter)

    async def exercise():
        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            assert {"search_a_evidence", "read_a_evidence", "finish_a_research"} <= names
            assert "validate_frozen_batch" not in names
            for tool in listed.tools:
                properties = tool.input_schema.get("properties", {})
                assert not {"role", "path", "thread_id", "host_id", "run_root"} & set(properties)

            result = await client.call_tool(
                "search_a_evidence", {"query": "固定六币合成行情", "limit": 5}
            )
            assert not result.is_error
            assert result.structured_content["total_matches"] >= 1
            assert result.structured_content["matches"][0]["record_id"] == "goal-spec"

            invalid = await client.call_tool(
                "search_a_evidence",
                {"query": "固定六币合成行情", "limit": 5, "path": str(tmp_path)},
            )
            assert invalid.is_error
            assert "path" in invalid.content[0].text

    asyncio.run(exercise())


def test_official_sdk_stdio_process_calls_real_domain_tool(tmp_path):
    binding_path = _binding(tmp_path)

    async def exercise():
        async with Client(_stdio_server("A", binding_path), raise_exceptions=True) as client:
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            assert {"search_a_evidence", "read_a_evidence", "finish_a_research"} <= names
            assert "freeze_a_manifest" not in names
            result = await client.call_tool(
                "search_a_evidence", {"query": "固定六币合成行情", "limit": 5}
            )
            assert not result.is_error
            assert result.structured_content["matches"][0]["record_id"] == "goal-spec"

        async with Client(_stdio_server("coordinator", binding_path),
                          raise_exceptions=True) as client:
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            assert {"prepare_a_task", "freeze_a_manifest", "check_goal_evidence"} <= names
            assert "validate_frozen_batch" not in names
            task = await client.call_tool("prepare_a_task", {})
            assert not task.is_error
            assert task.structured_content["ticket_ref"]["task_id"] == task.structured_content["task_id"]
            assert task.structured_content["task_ticket"]["thin_spec"]["data_environment"]["segment"] == "A"

        async with Client(_stdio_server("B", binding_path), raise_exceptions=True) as client:
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            assert {"validate_frozen_batch", "read_b_evidence", "finalize_b_delivery"} <= names
            assert "freeze_a_manifest" not in names
            status = await client.call_tool("get_research_status", {})
            assert status.is_error

    asyncio.run(exercise())


def test_a_evidence_stdio_errors_are_recoverable_and_remain_a_only(tmp_path):
    binding_path = _binding(tmp_path)
    a_agent = load_native_adapter(binding_path, "A")
    run_root = a_agent.binding.run_root
    a_store = a_agent.actions.miner.store
    a_store.append("c4-evaluation", "evaluation", {
        "candidate_id": "c4-synthetic-0001",
        "segment": "A",
        "summary": {"label": "compact evaluation"},
        "coverage": {"rows": 6},
        "horizon_comparison": {
            "horizons": {"1": {"summary": {"label": "one hour"},
                                 "coverage": {"rows": 6}}},
        },
    })
    a_store.append("c5-comparison", "experiment_result", {
        "candidate_id": "c5-synthetic-0001",
        "paired_improvement": {"mean": 0.25},
    })
    a_store.append("large-a-evidence", "candidate", {
        "segment": "A",
        "summary": {"label": "small field remains readable"},
        "payload": "x" * 300_000,
    })
    RecordStore(run_root / "b_records", run_root=run_root).append(
        "private-b-evidence", "b_delivery", {"segment": "B", "secret": "B-only-value"})
    RecordStore(run_root / "c_records", run_root=run_root).append(
        "private-c-evidence", "c_record", {"segment": "C", "secret": "C-only-value"})

    async def exercise():
        async with Client(_stdio_server("A", binding_path), raise_exceptions=True) as client:
            bad_comparison_path = await client.call_tool("read_a_evidence", {
                "record_id": "c5-comparison", "pointer": "/summary", "offset": 0, "limit": 1,
            })
            assert bad_comparison_path.is_error
            assert "component 'summary' is missing" in bad_comparison_path.content[0].text

            comparison = await client.call_tool("read_a_evidence", {
                "record_id": "c5-comparison", "pointer": "/paired_improvement/mean",
                "offset": 0, "limit": 1,
            })
            assert not comparison.is_error
            assert comparison.structured_content["data"] == 0.25

            bad_evaluation_path = await client.call_tool("read_a_evidence", {
                "record_id": "c4-evaluation", "pointer": "/horizons/1/summary",
                "offset": 0, "limit": 1,
            })
            assert bad_evaluation_path.is_error
            assert "component 'horizons' is missing" in bad_evaluation_path.content[0].text

            evaluation = await client.call_tool("read_a_evidence", {
                "record_id": "c4-evaluation",
                "pointer": "/horizon_comparison/horizons/1/summary",
                "offset": 0, "limit": 1,
            })
            assert not evaluation.is_error
            assert evaluation.structured_content["data"] == {"label": "one hour"}

            oversized = await client.call_tool("read_a_evidence", {
                "record_id": "large-a-evidence", "pointer": "", "offset": 0, "limit": 1,
            })
            assert oversized.is_error
            assert "no content was returned" in oversized.content[0].text
            assert "x" * 100 not in oversized.content[0].text

            small_field = await client.call_tool("read_a_evidence", {
                "record_id": "large-a-evidence", "pointer": "/summary", "offset": 0, "limit": 1,
            })
            assert not small_field.is_error
            assert small_field.structured_content["data"] == {"label": "small field remains readable"}

            for record_id, secret in (("private-b-evidence", "B-only-value"),
                                      ("private-c-evidence", "C-only-value")):
                denied = await client.call_tool("read_a_evidence", {
                    "record_id": record_id, "pointer": "", "offset": 0, "limit": 1,
                })
                assert denied.is_error
                assert secret not in denied.content[0].text

    asyncio.run(exercise())


def _stdio_server(role: str, binding_path: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "crypto_quant.research.factor_mining.mcp_server",
              "--role", role, "--binding", str(binding_path)],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
    )


def _dispatch_b_ticket_with_host(binding_path: Path, task_id: str) -> dict:
    script = r"""
import { createTicketDispatcher } from './harness/factor-research/src/dispatch.js'
const agent = { id: 'agent-coordinator', session: { id: process.env.TEST_COORDINATOR_SESSION }, status: 'running' }
const requests = []
const sessions = {
  async inspect() { return { events: [] } },
  async prompt(request) { requests.push(request); return { accepted: true } },
}
const agents = {
  get(id) { return id === agent.id ? agent : undefined },
  list() { return [agent] },
  currentInitiator() { return agent },
}
const dispatcher = createTicketDispatcher({
  config: {
    runRoot: process.env.TEST_RUN_ROOT,
    bindingPath: process.env.TEST_BINDING_PATH,
    pythonExecutable: process.env.TEST_PYTHON,
    projectRoot: process.env.TEST_PROJECT_ROOT,
    sessionRoles: {
      coordinator: process.env.TEST_COORDINATOR_SESSION,
      A: process.env.TEST_A_SESSION,
      B: process.env.TEST_B_SESSION,
    },
  },
  sessions,
  agents,
  goals: { get() { return undefined } },
  originalPrompt: sessions.prompt,
})
try {
  const result = await dispatcher.dispatchB(process.env.TEST_TASK_ID, agent)
  process.stdout.write(JSON.stringify({ result, request: requests[0] }))
} finally {
  // The receipt-only caller has finished; no live driver remains to cancel.
  agent.status = 'idle'
  await dispatcher.dispose()
}
"""
    binding = NativeChatBinding.load(binding_path)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        env={
            **os.environ,
            "TEST_RUN_ROOT": str(binding.run_root),
            "TEST_BINDING_PATH": str(binding_path),
            "TEST_PYTHON": sys.executable,
            "TEST_PROJECT_ROOT": str(ROOT),
            "TEST_COORDINATOR_SESSION": binding.chats["coordinator"].thread_id,
            "TEST_A_SESSION": binding.chats["A"].thread_id,
            "TEST_B_SESSION": binding.chats["B"].thread_id,
            "TEST_TASK_ID": task_id,
        },
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_roles_have_separate_tools_and_b_prices_stay_unloaded_before_freeze(tmp_path):
    binding_path = _binding(tmp_path)
    import crypto_quant.research.factor_mining.cli as factor_cli

    loaded_stages: list[str] = []
    load_stage = factor_cli.load_stage

    def counted_load_stage(*args, **kwargs):
        stage = args[3] if len(args) > 3 else kwargs["stage"]
        loaded_stages.append(stage)
        return load_stage(*args, **kwargs)

    with patch.object(factor_cli, "load_stage", side_effect=counted_load_stage):
        coordinator = load_native_adapter(binding_path, "coordinator")
        coordinator_tools = {tool["name"] for tool in coordinator.tool_definitions}
        assert "freeze_a_manifest" in coordinator_tools
        assert "prepare_b_task" in coordinator_tools
        assert "validate_frozen_batch" not in coordinator_tools
        assert loaded_stages == []
        try:
            coordinator.prepare_b_task()
        except ValueError as exc:
            assert "complete and seal A" in str(exc)
        else:
            raise AssertionError("B ticket preparation must require a completed A manifest")
        assert loaded_stages == []

        a_agent = load_native_adapter(binding_path, "A")
        a_tools = {tool["name"] for tool in a_agent.tool_definitions}
        assert {"search_a_evidence", "read_a_evidence", "finish_a_research"} <= a_tools
        assert "read_b_evidence" not in a_tools
        assert loaded_stages == ["A"]

        b_agent = load_native_adapter(binding_path, "B")
        b_tools = {tool["name"] for tool in b_agent.tool_definitions}
        assert {"validate_frozen_batch", "read_b_evidence", "finalize_b_delivery"} <= b_tools
        assert "freeze_a_manifest" not in b_tools
        assert "prepare_b_task" not in b_tools
        assert loaded_stages == ["A"]

        async def exercise_b_status():
            async with Client(create_server(b_agent), raise_exceptions=True) as client:
                result = await client.call_tool("get_research_status", {})
                assert result.is_error
                validation = await client.call_tool(
                    "validate_frozen_batch", {"action_id": "before-freeze"}
                )
                assert validation.is_error

        asyncio.run(exercise_b_status())
        assert loaded_stages == ["A"]


def test_prepare_a_task_saves_complete_thin_spec_with_content_bound_reference(tmp_path):
    binding_path = _binding(tmp_path)
    coordinator = load_native_adapter(binding_path, "coordinator")

    prepared = coordinator.prepare_a_task()
    ticket = prepared["task_ticket"]
    reference = prepared["ticket_ref"]
    body = {key: value for key, value in ticket.items()
            if key not in {"task_id", "content_sha256", "action_id"}}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert prepared["task_id"] == ticket["task_id"] == reference["task_id"] == expected_hash
    assert ticket["content_sha256"] == reference["content_sha256"] == expected_hash
    assert ticket["research_id"] == coordinator.binding.research_id
    assert ticket["task_type"] == "initial_a_research"
    assert ticket["target"] == coordinator.binding.target
    assert ticket["goal"] == coordinator.binding.goal
    assert ticket["contract"] == coordinator.binding.spec.as_dict()
    thin_spec = ticket["thin_spec"]
    assert thin_spec["data_environment"]["segment"] == "A"
    assert thin_spec["data_environment"]["eligible_rows"] > 0
    assert len(thin_spec["data_environment"]["panel_sha256"]) == 64
    assert thin_spec["data_environment"]["include_liquidations"] is False
    assert thin_spec["data_environment"]["coverage_summary"]["fields"] > 0
    assert all(field["source"] != "liquidations"
               for field in thin_spec["expression_capabilities"]["fields"])
    assert thin_spec["decision_policy"]["improvement_lower_bound"] == "> 0"
    assert thin_spec["decision_policy"]["mean_improvement"] == ">= min_improvement"
    assert thin_spec["goal_counts"]["a_retained_candidates"] == 0
    assert thin_spec["goal_counts"]["b_qualified_deliverables"] is None
    assert "source_database" not in json.dumps(ticket, ensure_ascii=False)
    assert coordinator.binding.market_db.name not in json.dumps(ticket, ensure_ascii=False)

    ticket_path = (coordinator.binding.run_root / "task-tickets" / "A" / f"{expected_hash}.json")
    assert json.loads(ticket_path.read_text(encoding="utf-8")) == ticket
    dictionary_ref = thin_spec["expression_capabilities"]["dictionary_ref"]
    dictionary_page = coordinator.actions._a_evidence_tools().read(
        record_id=dictionary_ref["record_id"], pointer="/operators", offset=0, limit=100)
    assert dictionary_page["data"]["total"] == len(dictionary_page["data"]["items"])
    assert len(dictionary_page["data"]["items"]) >= 20
    repeated = coordinator.prepare_a_task()
    assert repeated["task_id"] == expected_hash
    assert repeated["task_ticket"] == ticket


def test_a_action_initializes_after_coordinator_saves_environment_record(tmp_path):
    binding_path = _binding(tmp_path)
    coordinator = load_native_adapter(binding_path, "coordinator")
    coordinator.prepare_a_task()

    a_adapter = load_native_adapter(binding_path, "A")
    evaluated = a_adapter.call("evaluate_candidate", {
        "action_id": "first-a-after-ticket",
        "candidate_id": "synthetic-after-ticket-0001",
        "definition": {
            "name": "合成永续价格输入",
            "expression": "perp_close",
            "meaning": "固定六币合成样本的永续收盘价。",
            "hypothesis": "只验证首张任务工单后的A动作初始化。",
            "direction": 1,
            "parent_id": None,
            "proposal_id": None,
            "change_reason": "使用固定字段验证工单环境记录与A动作初始化兼容。",
        },
        "experiment_id": None,
    })

    assert evaluated["status"] == "computed"
    assert (coordinator.binding.run_root / "a-action-input-binding.json").is_file()
    records = {record["id"]: record for record in coordinator.actions.miner.store.all()}
    assert len([record for record in records.values()
                if record["kind"] == "a_environment"]) == 1
    assert records["synthetic-after-ticket-0001-calculation"]["kind"] == "calculation"


def test_prepare_a_task_is_rejected_after_a_seal_or_b_access(tmp_path):
    for marker_name, reason in (("a-complete.json", "sealed"),
                                ("b-access-started.json", "B access")):
        case_dir = tmp_path / reason.replace(" ", "-")
        case_dir.mkdir()
        adapter = load_native_adapter(_binding(case_dir), "coordinator")
        (adapter.binding.run_root / marker_name).write_text("{}", encoding="utf-8")

        try:
            adapter.prepare_a_task()
        except ValueError as exc:
            assert reason in str(exc)
        else:
            raise AssertionError(f"A task preparation must fail after {reason}")
        assert not (adapter.binding.run_root / "task-tickets").exists()


def test_human_change_creates_immutable_linked_child_run_and_a_ticket(tmp_path):
    binding_path = _binding(tmp_path)
    coordinator = load_native_adapter(binding_path, "coordinator")
    assert "record_human_change" not in {
        definition["name"] for definition in coordinator.tool_definitions
    }
    approval_path = tmp_path / "approved-change.json"
    approval_path.write_text(json.dumps({
        "format_version": 1,
        "child_research_id": "native-mcp-change-child",
        "original_intent": "增加最低 IC 门槛并验证研究方向。",
        "adopted_change": "将最低绝对 IC 门槛提高到 0.02，并保持其他数据边界。",
        "new_objective": "在固定合成行情中检验更高最低 IC 门槛。",
        "contract_changes": {"min_abs_ic": 0.02},
        "goal_changes": {},
    }, ensure_ascii=False), encoding="utf-8")

    result = record_human_change(binding_path, approval_path)
    parent = NativeChatBinding.load(binding_path)
    child_root = Path(result["child_run_root"])
    restored_binding = NativeChatBinding.load(result["child_binding_path"])
    assert restored_binding.run_root == child_root
    assert restored_binding.target["version"] == 2
    assert restored_binding.chats == parent.chats
    child_contract = json.loads((child_root / "contract.json").read_text(encoding="utf-8"))
    change_record = json.loads((child_root / "change-record.json").read_text(encoding="utf-8"))
    ticket = json.loads(Path(result["task_ticket_path"]).read_text(encoding="utf-8"))

    assert result["parent_research_id"] == parent.research_id
    assert result["child_research_id"] == "native-mcp-change-child"
    assert child_contract["run_id"] == result["child_research_id"]
    assert child_contract["min_abs_ic"] == 0.02
    assert change_record["parent"]["contract_sha256"] == parent.contract_sha256
    assert change_record["parent"]["evidence_refs"]
    assert change_record["parent"]["target"]["version"] == 1
    assert change_record["child"]["target"]["version"] == 2
    assert change_record["change"]["original_intent"] == "增加最低 IC 门槛并验证研究方向。"
    assert change_record["change"]["contract_changes"] == {"min_abs_ic": 0.02}
    assert ticket["task_type"] == "initial_a_research"
    assert ticket["research_id"] == result["child_research_id"]
    assert ticket["change_ref"]["record_id"] == result["change_record_id"]
    assert ticket["change_ref"]["content_sha256"] == result["change_record_sha256"]
    assert ticket["change_ref"]["affected_evidence_refs"] == change_record["parent"]["evidence_refs"]

    recovered = record_human_change(binding_path, approval_path)
    assert recovered["reused_existing_child"] is True
    assert recovered["task_id"] == result["task_id"]


def test_human_change_rejects_parent_after_a_seal_b_access_or_pending_action(tmp_path):
    cases = ("sealed", "b_access", "b_records", "pending")
    for case in cases:
        case_dir = tmp_path / case
        case_dir.mkdir()
        binding_path = _binding(case_dir)
        binding = NativeChatBinding.load(binding_path)
        approval_path = case_dir / "approved-change.json"
        approval_path.write_text(json.dumps({
            "format_version": 1,
            "child_research_id": f"native-change-{case}",
            "original_intent": "调整研究目标。",
            "adopted_change": "提高最低 IC 门槛。",
            "new_objective": "在固定合成行情中检验变更后的目标。",
            "contract_changes": {"min_abs_ic": 0.02},
            "goal_changes": {},
        }, ensure_ascii=False), encoding="utf-8")
        if case == "sealed":
            (binding.run_root / "a-complete.json").write_text(
                json.dumps({"manifest_ref": "sealed-manifest"}), encoding="utf-8")
        elif case == "b_access":
            (binding.run_root / "b-access-started.json").write_text("{}", encoding="utf-8")
        elif case == "b_records":
            b_records = binding.run_root / "b_records"
            b_records.mkdir()
            (b_records / "saved-b-evidence.json").write_text("{}", encoding="utf-8")
        else:
            receipts = binding.run_root / "action_receipts"
            receipts.mkdir(exist_ok=True)
            (receipts / "pending-action.started.json").write_text(json.dumps({
                "action_id": "pending-action", "name": "evaluate_candidate",
                "parameters_sha256": "a" * 64,
            }), encoding="utf-8")
        try:
            record_human_change(binding_path, approval_path)
        except ValueError as exc:
            assert any(phrase in str(exc) for phrase in (
                "parent A run to remain open", "after B access", "after B evidence",
                "pending domain action",
            ))
        else:
            raise AssertionError(f"expected {case} parent state to reject a human change")


def test_task_ticket_recovery_reuses_content_id_and_a_state_change_mints_a_new_one(tmp_path):
    binding_path = _binding(tmp_path)
    coordinator = load_native_adapter(binding_path, "coordinator")
    first = coordinator.prepare_a_task()
    recovered = load_native_adapter(binding_path, "coordinator").prepare_a_task()
    assert recovered == first

    coordinator.actions.miner.store.append(
        "candidate-0001", "candidate", {"id": "candidate-0001", "source_segment": "A"})
    changed = load_native_adapter(binding_path, "coordinator").prepare_a_task()
    assert changed["task_id"] != first["task_id"]
    assert changed["task_ticket"]["task_type"] == "continue_a_research"
    assert changed["task_ticket"]["a_snapshot"]["candidate_ids"] == ["candidate-0001"]


def test_prepare_a_recovery_task_requires_accepted_dispatch_and_binds_pending_actions(tmp_path):
    binding_path = _binding(tmp_path)
    coordinator = load_native_adapter(binding_path, "coordinator")
    initial = coordinator.prepare_a_task()
    parent_task_id = initial["task_id"]
    run_root = coordinator.binding.run_root
    receipt_path = run_root / "dispatch-receipts" / "A" / f"{parent_task_id}.json"

    try:
        coordinator.call("prepare_a_recovery_task", {"task_id": parent_task_id})
    except ValueError as exc:
        assert "accepted A dispatch receipt does not exist" in str(exc)
    else:
        raise AssertionError("recovery must require an accepted Host dispatch receipt")

    receipt_path.parent.mkdir(parents=True)
    receipt = {
        "format_version": 1,
        "task_id": parent_task_id,
        "ticket_sha256": parent_task_id,
        "target_role": "A",
        "session_id": "session-a",
        "request_id": "dsh-request-001",
        "status": "accepted",
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    pending_path = coordinator.actions.ledger.root / "evaluate-001.started.json"
    pending_path.write_text(json.dumps({
        "action_id": "evaluate-001",
        "name": "evaluate_candidate",
        "parameters_sha256": "a" * 64,
    }), encoding="utf-8")

    recovered = coordinator.call("prepare_a_recovery_task", {"task_id": parent_task_id})
    ticket = recovered["task_ticket"]
    assert ticket["task_type"] == "recovery_a_research"
    assert ticket["recovery_ref"]["parent_task_id"] == parent_task_id
    assert ticket["recovery_ref"]["parent_content_sha256"] == parent_task_id
    assert ticket["recovery_ref"]["pending_actions"] == [{
        "action_id": "evaluate-001",
        "name": "evaluate_candidate",
        "parameters_sha256": "a" * 64,
    }]
    assert ticket["task_id"] != parent_task_id
    assert recovered == coordinator.call(
        "prepare_a_recovery_task", {"task_id": parent_task_id})

    (coordinator.actions.ledger.root / "register-002.started.json").write_text(
        json.dumps({"action_id": "register-002", "name": "register_experiment",
                    "parameters_sha256": "b" * 64}), encoding="utf-8")
    changed = coordinator.call("prepare_a_recovery_task", {"task_id": parent_task_id})
    assert changed["task_id"] != recovered["task_id"]
    assert len(changed["task_ticket"]["recovery_ref"]["pending_actions"]) == 2

    a_tools = {tool["name"] for tool in load_native_adapter(binding_path, "A").tool_definitions}
    b_tools = {tool["name"] for tool in load_native_adapter(binding_path, "B").tool_definitions}
    assert "prepare_a_recovery_task" not in a_tools | b_tools


def test_native_mcp_synthetic_a_freeze_b_checkpoint_recovery(tmp_path):
    binding_path = _binding(tmp_path)
    coordinator = load_native_adapter(binding_path, "coordinator")
    a_agent = load_native_adapter(binding_path, "A")
    b_agent = load_native_adapter(binding_path, "B")
    run_root = coordinator.binding.run_root

    original_b_loader = b_agent.actions.b_panel_loader
    b_load_count = 0

    def counted_b_loader():
        nonlocal b_load_count
        assert (run_root / "b-access-started.json").is_file()
        b_load_count += 1
        return original_b_loader()

    b_agent.actions.b_panel_loader = counted_b_loader

    def report():
        return {
            "analysis": "固定合成样本只用于验证A/B工具闭环。",
            "mechanism": "本报告不声明真实市场机制。",
            "conditions": ["仅覆盖固定合成样本。"],
            "falsifiers": ["该样本不支持真实行情结论。"],
            "limitations": ["未使用真实行情数据。"],
            "next_steps": ["继续检查工具动作与检查点恢复。"],
        }

    async def call(client, name, arguments):
        result = await client.call_tool(name, arguments)
        assert not result.is_error, f"{name} failed: {result.content}"
        return result.structured_content

    async def exercise():
        async with Client(create_server(a_agent), raise_exceptions=True) as a_client, \
                Client(create_server(coordinator), raise_exceptions=True) as coordinator_client, \
                Client(create_server(b_agent), raise_exceptions=True) as b_client:
            candidate_id = "synthetic-candidate-0001"
            definition = {
                "name": "合成永续价格输入",
                "expression": "perp_close",
                "meaning": "固定六币合成样本的永续收盘价。",
                "hypothesis": "只验证研究工具调用路径，不声明预测能力。",
                "direction": 1,
                "parent_id": None,
                "proposal_id": None,
                "change_reason": "用一个固定表达式跑通合成工程流程。",
            }
            evaluated = await call(a_client, "evaluate_candidate", {
                "action_id": "a-evaluate-001",
                "candidate_id": candidate_id,
                "definition": definition,
                "experiment_id": None,
            })
            assert evaluated["status"] == "computed"
            evaluation_ref = evaluated["evidence_pointers"]["evaluation"]
            assert evaluation_ref["record_id"] == f"{candidate_id}-evaluation"
            for field in ("summary", "coverage"):
                source = await call(a_client, "read_a_evidence", {
                    "record_id": evaluation_ref["record_id"],
                    "pointer": evaluation_ref["field_pointers"][field],
                    "offset": 0, "limit": 1,
                })
                assert source["data"] == evaluated["evaluation"][field]
            for horizon, field_pointers in evaluation_ref["field_pointers"]["horizons"].items():
                for field, pointer in field_pointers.items():
                    source = await call(a_client, "read_a_evidence", {
                        "record_id": evaluation_ref["record_id"], "pointer": pointer,
                        "offset": 0, "limit": 1,
                    })
                    assert source["data"] == evaluated["evaluation"]["horizons"][horizon][field]

            decision = {
                "candidate_id": candidate_id,
                "disposition": "retain",
                "retained_horizons": [24],
                "continue_optimization": False,
                "answers": {
                    key: {"supported": False, "reason": "合成闭环不构成真实研究证据。"}
                    for key in ("research_basis", "modification_hypothesis",
                                "verifiable_improvement", "attempt_value")
                },
                "reason": "仅为验证A封存和B检查链路而保留该合成候选。",
                "resume_condition": None,
                "evidence_refs": [f"{candidate_id}-evaluation"],
            }
            await call(a_client, "save_research_decision", {
                "action_id": "a-decision-001",
                "candidate_id": candidate_id,
                "decision": decision,
                "narrative": report(),
            })

            a_record_ids = [record["id"] for record in a_agent.actions.miner.store.all()]
            completed = await call(a_client, "finish_a_research", {
                "action_id": "a-finish-001",
                "semantic_match_report": {
                    "objective": coordinator.binding.goal["objective"],
                    "matches": [{
                        "candidate_id": candidate_id,
                        "matches_goal": False,
                        "reason": "候选只服务于合成工程验收，未证明达到因子研究目标。",
                        "evidence_refs": a_record_ids,
                    }],
                },
            })
            assert completed["status"] == "complete"
            assert completed["objective_met"] is False

            # B is already initialized and exposed before the coordinator opens it.
            pre_freeze = await b_client.call_tool(
                "validate_frozen_batch", {"action_id": "b-validate-001"})
            assert pre_freeze.is_error
            assert b_load_count == 0

            frozen = await call(coordinator_client, "freeze_a_manifest", {
                "action_id": "coordinator-freeze-001",
                "manifest_ref": completed["manifest_ref"],
            })
            assert frozen["frozen"] is True
            assert frozen["candidate_ids"] == [candidate_id]

            prepared_b = await call(coordinator_client, "prepare_b_task", {})
            b_ticket = prepared_b["task_ticket"]
            b_ticket_path = (run_root / "task-tickets" / "B"
                             / f"{prepared_b['task_id']}.json")
            assert b_ticket_path.is_file()
            assert b_ticket["task_id"] == b_ticket["content_sha256"]
            assert b_ticket["task_type"] == "frozen_b_validation"
            assert b_ticket["a_completion"]["manifest_ref"] == completed["manifest_ref"]
            assert b_ticket["a_completion"]["frozen_candidate_ids"] == [candidate_id]
            assert list(b_ticket["frozen_batch"]["candidates"]) == [candidate_id]
            assert b_ticket["thin_spec"]["data_environment"]["segment"] == "B"
            assert b_load_count == 0

            host_result = _dispatch_b_ticket_with_host(binding_path, prepared_b["task_id"])
            assert host_result["result"]["target_role"] == "B"
            assert host_result["request"]["sessionId"] == coordinator.binding.chats["B"].thread_id
            assert host_result["request"]["content"] == [{
                "type": "text", "text": b_ticket_path.read_text(encoding="utf-8"),
            }]
            assert b_load_count == 0

            repeated_b = await call(coordinator_client, "prepare_b_task", {})
            assert repeated_b == prepared_b
            assert b_load_count == 0

            validated = await call(b_client, "validate_frozen_batch", {
                "action_id": "b-validate-001"})
            assert validated["status"] == "numerical_complete"
            assert validated["candidate_ids"] == [candidate_id]
            assert b_load_count == 1
            assert (run_root / "b-numerical-complete.json").is_file()
            co_result = await call(coordinator_client, "read_b_result", {"candidate_id": candidate_id})
            assert co_result["numerical_complete"] is True
            assert co_result["report"] is None
            assert co_result["decision"]["record_id"] == f"{candidate_id}-validation"
            assert b_load_count == 1
            b_result = await call(b_client, "read_b_result", {"candidate_id": candidate_id})
            assert b_result == co_result
            invalid_pointer = await b_client.call_tool("read_b_evidence", {
                "record_id": f"{candidate_id}-validation", "pointer": "/status",
                "offset": 0, "limit": 1})
            assert invalid_pointer.is_error
            assert "root keys:" in str(invalid_pointer.content)
            assert b_load_count == 1

            # Simulate a crash after the numerical checkpoint but before the action receipt.
            # The started receipt remains, so the original action runs its domain recovery path.
            receipt = run_root / "action_receipts/b-validate-001.committed.json"
            assert receipt.is_file()
            receipt.unlink()
            recovered = await call(b_client, "validate_frozen_batch", {
                "action_id": "b-validate-001"})
            assert recovered["status"] == "numerical_complete"
            assert recovered["candidate_ids"] == [candidate_id]
            assert b_load_count == 1

            await call(b_client, "save_b_interpretation", {
                "action_id": "b-report-001",
                "candidate_id": candidate_id,
                "narrative": report(),
            })
            delivered = await call(b_client, "finalize_b_delivery", {
                "action_id": "b-delivery-001"})
            assert delivered["status"] == "complete"

            evidence = await call(coordinator_client, "check_goal_evidence", {})
            assert evidence["status"] == "incomplete"
            assert evidence["objective_met"] is False
            assert evidence["target"] == {"target_ideas": 1}
            assert any("engineering_check" in gap for gap in evidence["gaps"])

    asyncio.run(exercise())


def test_native_mcp_stdio_checkpoint_recovery_in_fresh_b_process(tmp_path):
    binding_path = _binding(tmp_path)
    binding_bytes = binding_path.read_bytes()
    binding = NativeChatBinding.load(binding_path)
    run_root = binding.run_root
    coordinator = load_native_adapter(binding_path, "coordinator")
    a_agent = load_native_adapter(binding_path, "A")

    def report():
        return {
            "analysis": "固定合成样本只用于跨进程检查点验收。",
            "mechanism": "本报告不声明真实市场机制。",
            "conditions": ["仅覆盖固定合成样本。"],
            "falsifiers": ["该样本不支持真实行情结论。"],
            "limitations": ["未使用真实行情数据。"],
            "next_steps": ["检查重启后的检查点恢复。"],
        }

    async def call(client, name, arguments):
        result = await client.call_tool(name, arguments)
        assert not result.is_error, f"{name} failed: {result.content}"
        return result.structured_content

    async def prepare_sealed_a_and_freeze_b():
        async with Client(create_server(a_agent), raise_exceptions=True) as a_client, \
                Client(create_server(coordinator), raise_exceptions=True) as coordinator_client:
            candidate_id = "synthetic-candidate-0001"
            definition = {
                "name": "合成永续价格输入",
                "expression": "perp_close",
                "meaning": "固定六币合成样本的永续收盘价。",
                "hypothesis": "只验证研究工具调用路径，不声明预测能力。",
                "direction": 1,
                "parent_id": None,
                "proposal_id": None,
                "change_reason": "用一个固定表达式跑通合成工程流程。",
            }
            evaluated = await call(a_client, "evaluate_candidate", {
                "action_id": "a-evaluate-001",
                "candidate_id": candidate_id,
                "definition": definition,
                "experiment_id": None,
            })
            assert evaluated["status"] == "computed"
            decision = {
                "candidate_id": candidate_id,
                "disposition": "retain",
                "retained_horizons": [24],
                "continue_optimization": False,
                "answers": {
                    key: {"supported": False, "reason": "合成闭环不构成真实研究证据。"}
                    for key in ("research_basis", "modification_hypothesis",
                                "verifiable_improvement", "attempt_value")
                },
                "reason": "仅为验证跨进程B检查点恢复而保留该合成候选。",
                "resume_condition": None,
                "evidence_refs": [f"{candidate_id}-evaluation"],
            }
            await call(a_client, "save_research_decision", {
                "action_id": "a-decision-001",
                "candidate_id": candidate_id,
                "decision": decision,
                "narrative": report(),
            })
            a_record_ids = [record["id"] for record in a_agent.actions.miner.store.all()]
            completed = await call(a_client, "finish_a_research", {
                "action_id": "a-finish-001",
                "semantic_match_report": {
                    "objective": binding.goal["objective"],
                    "matches": [{
                        "candidate_id": candidate_id,
                        "matches_goal": False,
                        "reason": "候选只服务于跨进程合成验收，未证明达到因子研究目标。",
                        "evidence_refs": a_record_ids,
                    }],
                },
            })
            assert completed["status"] == "complete"
            frozen = await call(coordinator_client, "freeze_a_manifest", {
                "action_id": "coordinator-freeze-001",
                "manifest_ref": completed["manifest_ref"],
            })
            assert frozen["frozen"] is True
            assert frozen["candidate_ids"] == [candidate_id]

    asyncio.run(prepare_sealed_a_and_freeze_b())
    assert binding_path.read_bytes() == binding_bytes
    assert NativeChatBinding.load(binding_path).run_root == run_root

    def child_pid() -> int:
        process_list = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,command="],
            check=True, capture_output=True, text=True,
        ).stdout
        matches = []
        for row in process_list.splitlines():
            fields = row.strip().split(maxsplit=2)
            if len(fields) != 3:
                continue
            pid, ppid, command = fields
            if (int(ppid) == os.getpid() and str(binding_path) in command
                    and "--role B" in command
                    and "crypto_quant.research.factor_mining.mcp_server" in command):
                matches.append(int(pid))
        assert len(matches) == 1, f"expected one child B MCP process, found {matches}"
        return matches[0]

    async def validate_in_first_process() -> tuple[int, dict]:
        async with Client(_stdio_server("B", binding_path), raise_exceptions=True) as client:
            pid = child_pid()
            listed = await client.list_tools()
            assert "validate_frozen_batch" in {tool.name for tool in listed.tools}
            result = await call(client, "validate_frozen_batch", {
                "action_id": "b-validate-001"})
            assert result["status"] == "numerical_complete"
            assert result["candidate_ids"] == ["synthetic-candidate-0001"]
            return pid, result

    first_pid, first_result = asyncio.run(validate_in_first_process())
    assert first_result["status"] == "numerical_complete"
    assert (run_root / "b-access-started.json").is_file()
    assert (run_root / "b-numerical-complete.json").is_file()

    started_receipt = run_root / "action_receipts/b-validate-001.started.json"
    committed_receipt = run_root / "action_receipts/b-validate-001.committed.json"
    assert started_receipt.is_file() and committed_receipt.is_file()
    # Model a process exit after the numerical checkpoint but before its receipt commit.
    committed_receipt.unlink()
    assert started_receipt.is_file()

    patch_root = tmp_path / "forbid_b_reload"
    patch_root.mkdir()
    patched_loader_started = tmp_path / "second_b_sitecustomize_loaded"
    (patch_root / "sitecustomize.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import crypto_quant.research.factor_mining.cli as factor_cli\n"
        "def _forbid_b_market_reload(*args, **kwargs):\n"
        "    raise AssertionError('checkpoint recovery attempted to reload B market data')\n"
        "factor_cli.load_stage = _forbid_b_market_reload\n"
        "Path(os.environ['NATIVE_MCP_PATCH_ACTIVE']).write_text('active', encoding='utf-8')\n",
        encoding="utf-8",
    )
    second_server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "crypto_quant.research.factor_mining.mcp_server",
              "--role", "B", "--binding", str(binding_path)],
        cwd=ROOT,
        env={
            "PYTHONPATH": os.pathsep.join((str(patch_root), str(ROOT / "src"))),
            "NATIVE_MCP_PATCH_ACTIVE": str(patched_loader_started),
        },
    )

    async def recover_and_deliver_in_second_process() -> tuple[int, dict]:
        async with Client(second_server, raise_exceptions=True) as client:
            pid = child_pid()
            listed = await client.list_tools()
            assert {"validate_frozen_batch", "save_b_interpretation",
                    "finalize_b_delivery"} <= {tool.name for tool in listed.tools}
            assert patched_loader_started.is_file()
            recovered = await call(client, "validate_frozen_batch", {
                "action_id": "b-validate-001"})
            assert recovered["status"] == "numerical_complete"
            assert recovered["candidate_ids"] == ["synthetic-candidate-0001"]
            await call(client, "save_b_interpretation", {
                "action_id": "b-report-001",
                "candidate_id": "synthetic-candidate-0001",
                "narrative": report(),
            })
            delivered = await call(client, "finalize_b_delivery", {
                "action_id": "b-delivery-001"})
            assert delivered["status"] == "complete"
            return pid, delivered

    second_pid, delivered = asyncio.run(recover_and_deliver_in_second_process())
    assert second_pid != first_pid
    assert delivered["status"] == "complete"
    assert binding_path.read_bytes() == binding_bytes
    assert NativeChatBinding.load(binding_path).run_root == run_root
    assert json.loads((run_root / "b-numerical-complete.json").read_text()) == {
        "candidate_ids": ["synthetic-candidate-0001"],
        "retained_horizons": {"synthetic-candidate-0001": [24]},
    }
    assert json.loads((run_root / "b-delivery.json").read_text())["status"] == "complete"
    assert (run_root / "B-report.md").is_file()
