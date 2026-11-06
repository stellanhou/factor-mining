"""Small synthetic acceptance tests for the native P2/P5 action contracts."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from crypto_quant.research.factor_mining.goal import GoalSpec
from crypto_quant.research.factor_mining.contracts import digest
from crypto_quant.research.factor_mining.records import write_json
from crypto_quant.research.factor_mining.model import RecoverableToolError
from crypto_quant.research.factor_mining.research_actions import ResearchActions
from crypto_quant.research.factor_mining.workflow import FactorMiner
from test_factor_mining import definition, input_panel, specification


class NoModelCalls:
    def complete(self, *args, **kwargs):
        raise AssertionError("native program actions must not call a role model")


def _native_actions(tmp_path: Path, *, run_id: str,
                    max_repairs: int = 2) -> tuple[ResearchActions, object, dict]:
    spec = specification(run_id=run_id, max_repairs=max_repairs)
    panel_a = input_panel(spec, "A")
    miner = FactorMiner(spec, NoModelCalls(), tmp_path / "runs")
    goal = GoalSpec(goal_id=run_id, objective=spec.objective, target_ideas=1).as_dict()
    miner.store.append("goal-spec", "goal_contract", goal)
    return ResearchActions(miner, a_panel=panel_a, goal_spec=goal), panel_a, goal


def _narrative() -> dict:
    return {"analysis": "合成程序验收解释。", "mechanism": "待检验机制。",
            "conditions": ["固定合成条件。"], "falsifiers": ["反例将推翻该解释。"],
            "limitations": ["不代表真实预测能力。"], "next_steps": ["另行研究。"]}


def _decision(candidate_id: str, disposition: str, retained_horizons: list[int],
              evidence_ref: str) -> dict:
    return {"candidate_id": candidate_id, "disposition": disposition,
            "retained_horizons": retained_horizons, "continue_optimization": False,
            "answers": {key: {"supported": False, "reason": "仅用于合成合同验收。"}
                        for key in ("research_basis", "modification_hypothesis",
                                    "verifiable_improvement", "attempt_value")},
            "reason": "按固定合成目标记录候选去向。", "resume_condition": None,
            "evidence_refs": [evidence_ref]}


def test_native_a_evidence_read_rejects_oversized_root_with_pointer_hint(tmp_path):
    actions, _, _ = _native_actions(tmp_path, run_id="bounded-native-a-evidence")
    actions.miner.store.append("large-a-comparison", "experiment_result", {
        "summary": {"rank_ic": 0.1},
        "periods": [{"payload": "x" * 1_100_000}],
    })

    with pytest.raises(RecoverableToolError, match="no content was returned.*narrow the JSON pointer"):
        actions.handle("A", "read_a_evidence", {
            "record_id": "large-a-comparison", "pointer": "", "offset": 0, "limit": 1,
        })


def test_coordinator_reads_complete_current_a_decision_and_exact_source(tmp_path):
    actions, _, _ = _native_actions(tmp_path, run_id="coordinator-a-result")
    actions.evaluate_candidate(action_id="eval-readable", candidate_id="readable",
                               definition=definition("perp_close"))
    pending = actions.handle("coordinator", "read_a_result", {"candidate_id": "readable"})
    assert pending["decision"] is None and pending["report"] is None
    assert pending["a_status"] == "open"
    decision = _decision("readable", "discard", [], "readable-evaluation")
    actions.save_research_decision(action_id="decide-readable", candidate_id="readable",
                                   decision=decision, narrative=_narrative())
    result = actions.handle("coordinator", "read_a_result", {"candidate_id": "readable"})
    assert result["decision"]["data"] == decision
    assert result["report"]["data"]["analysis"] == _narrative()["analysis"]
    ref = result["decision"]
    exact = actions.handle("coordinator", "read_a_evidence", {
        "record_id": ref["record_id"], "pointer": ref["pointer"], "offset": 0, "limit": 1,
    })
    assert exact["data"] == decision
    assert len(ref["record_sha256"]) == 64
    with pytest.raises(ValueError, match="not available to fixed role B"):
        actions.handle("B", "read_a_result", {"candidate_id": "readable"})
    with pytest.raises(ValueError, match="outside the current A research"):
        actions.handle("coordinator", "read_a_result", {"candidate_id": "other"})


def test_coordinator_a_evidence_is_bounded_and_cannot_read_b_or_paths(tmp_path):
    actions, _, _ = _native_actions(tmp_path, run_id="coordinator-bounded-evidence")
    actions.miner.store.append("large-comparison", "experiment_result", {
        "summary": {"value": 1}, "periods": ["x" * 30_000],
    })
    arguments = {"record_id": "large-comparison", "pointer": "", "offset": 0, "limit": 1}
    with pytest.raises(RecoverableToolError, match="no content was returned"):
        actions.handle("coordinator", "read_a_evidence", arguments)
    assert actions.handle("coordinator", "read_a_evidence", {
        **arguments, "pointer": "/summary"})["data"] == {"value": 1}
    for record_id in ("b-private", "../contract.json"):
        with pytest.raises(ValueError):
            actions.handle("coordinator", "read_a_evidence", {**arguments, "record_id": record_id})


def test_native_syntax_failure_can_be_repaired_with_original_error_preserved(tmp_path):
    actions, _, _ = _native_actions(tmp_path, run_id="syntax-repair-contract")
    candidate_id = "candidate-syntax-repair"
    original_expression = "perp_close +"
    candidate_definition = definition(original_expression, meaning="合成语法错误候选。")

    failed = actions.evaluate_candidate(
        action_id="a-evaluate-syntax", candidate_id=candidate_id,
        definition=candidate_definition)

    assert failed["status"] == "repair_pending"
    attempt = {record["id"]: record for record in actions.miner.store.all()}[
        f"{candidate_id}-calculation-attempt-0000"]["data"]
    assert attempt["expression"] == original_expression
    assert "invalid expression syntax" in attempt["program_error"]
    assert attempt["action_id"] == "a-evaluate-syntax"

    repaired = actions.repair_candidate(
        action_id="a-repair-syntax", candidate_id=candidate_id,
        program_error=attempt["program_error"], repair_expression="perp_close",
        reason="将缺少操作数的表达式修正为有效的价格字段。")

    records = {record["id"]: record for record in actions.miner.store.all()}
    calculation = records[f"{candidate_id}-calculation"]["data"]
    repair = records[f"{candidate_id}-repair-0001"]["data"]
    assert repaired["status"] == "computed"
    assert calculation["status"] == "computed"
    assert calculation["original_expression"] == original_expression
    assert calculation["executed_expression"]["expression"] == "perp_close"
    assert calculation["checks"][0]["expression"] == original_expression
    assert calculation["checks"][0]["program_error"] == attempt["program_error"]
    assert calculation["checks"][1]["expression"] == "perp_close"
    assert calculation["checks"][1]["program_error"] is None
    assert repair["original_expression"] == original_expression
    assert repair["program_error"] == attempt["program_error"]
    assert repair["repair_expression"] == "perp_close"


def test_native_repair_budget_and_formula_budget_fail_closed(tmp_path):
    actions, _, _ = _native_actions(
        tmp_path, run_id="repair-budget-contract", max_repairs=1)
    candidate_id = "candidate-repair-budget"
    original_expression = "perp_close +"
    candidate_definition = definition(original_expression, meaning="合成修复预算候选。")

    first_failure = actions.evaluate_candidate(
        action_id="a-evaluate-budget", candidate_id=candidate_id,
        definition=candidate_definition)
    first_attempt = {record["id"]: record for record in actions.miner.store.all()}[
        f"{candidate_id}-calculation-attempt-0000"]["data"]
    assert first_failure["status"] == "repair_pending"

    exhausted = actions.repair_candidate(
        action_id="a-repair-budget", candidate_id=candidate_id,
        program_error=first_attempt["program_error"], repair_expression="perp_close +",
        reason="验证一次修复失败后遵守 max_repairs。")
    assert exhausted["status"] == "calculation_failed"
    records = {record["id"]: record for record in actions.miner.store.all()}
    assert records[f"{candidate_id}-calculation"]["data"]["original_expression"] == original_expression
    assert records[f"{candidate_id}-calculation-attempt-0001"]["data"]["expression"] == "perp_close +"
    assert f"{candidate_id}-repair-0001" in records
    assert not any(record["kind"] == "formula_repair"
                   and record["data"].get("attempt", 0) > 1 for record in records.values())
    with pytest.raises(ValueError, match="calculation is complete"):
        actions.repair_candidate(
            action_id="a-repair-over-budget", candidate_id=candidate_id,
            program_error=first_attempt["program_error"], repair_expression="perp_close",
            reason="不应超过 max_repairs。")

    budget_candidate = "candidate-formula-budget"
    budget_expression = "ts_mean(perp_close,24)"
    budget_result = actions.evaluate_candidate(
        action_id="a-evaluate-formula-budget", candidate_id=budget_candidate,
        definition=definition(budget_expression, meaning="超出历史窗口预算的合成候选。"))
    budget_records = {record["id"]: record for record in actions.miner.store.all()}
    budget_attempt = budget_records[f"{budget_candidate}-calculation-attempt-0000"]["data"]
    assert budget_result["status"] == "calculation_failed"
    assert budget_attempt["error_type"] == "FormulaBudgetError"
    assert "declared history budget" in budget_attempt["program_error"]
    assert budget_records[f"{budget_candidate}-calculation"]["data"]["original_expression"] == budget_expression
    assert not any(record["kind"] == "formula_repair"
                   and record["data"].get("candidate_id") == budget_candidate
                   for record in budget_records.values())
    with pytest.raises(ValueError):
        actions.repair_candidate(
            action_id="a-repair-formula-budget", candidate_id=budget_candidate,
            program_error=budget_attempt["program_error"], repair_expression="perp_close",
            reason="超预算公式必须作为新候选研究。")


def test_native_modification_requires_predeclared_experiment_and_closes_pending_route(tmp_path):
    actions, _, goal = _native_actions(tmp_path, run_id="predeclared-experiment-contract")
    control_id = "candidate-control"
    child_id = "candidate-modified"
    control_definition = definition("perp_close", meaning="固定合成对照。")
    child_definition = {
        **definition("perp_high", meaning="预声明路线后的合成候选。"),
        "parent_id": control_id,
        "proposal_id": "proposal-predeclared-route",
    }
    control = actions.evaluate_candidate(
        action_id="a-evaluate-control", candidate_id=control_id,
        definition=control_definition)
    assert control["status"] == "computed"

    route_id = "route-predeclared"
    plan = {
        "route_id": route_id,
        "control_id": control_id,
        "evidence_refs": [f"{control_id}-evaluation"],
        "modification_task": {
            "core_hypothesis": "固定合成对照假设。",
            "observed_problem": "只验证必须先登记实验的程序门。",
            "modification_hypothesis": "改变输入字段可形成独立候选。",
            "change_target": "用高价字段替代收盘价字段。",
            "fixed_components": "运行、方向和标签合同保持不变。",
        },
        "experiment_design": {
            "question": "合成修改是否改善 A 配对 Rank IC？",
            "metric": "rank_ic",
            "min_improvement": 0.01,
            "max_ic_loss": 0.2,
            "expected_outcome": "配对改善均值达标。",
            "stop_condition": "改善条件不满足时停止路线。",
            "pause_condition": "样本不足时暂停路线。",
        },
        "restart_of": None,
        "new_evidence": None,
    }
    child_action = {
        "action_id": "a-evaluate-modified",
        "candidate_id": child_id,
        "definition": child_definition,
        "experiment_id": route_id,
    }
    with pytest.raises(ValueError, match="predeclared and registered before evaluation"):
        actions.evaluate_candidate(**child_action)
    assert f"{child_id}-definition" not in {
        record["id"] for record in actions.miner.store.all()
    }
    assert [row["action_id"] for row in actions.ledger.pending_actions()] == [
        "a-evaluate-modified"
    ]

    registered = actions.register_experiment(action_id="a-register-route", plan=plan)
    assert registered["status"] == "continue"
    modified = actions.evaluate_candidate(**child_action)
    assert modified["status"] == "computed"
    assert "comparison" in modified
    assert not actions.ledger.pending_actions()
    close_parameters = {"route_id": route_id, "status": "stopped", "reason": "关闭已完成路线。"}
    with pytest.raises(ValueError, match="through its candidate decision"):
        actions.resolve_experiment(action_id="close-child-route", **close_parameters)
    assert not actions.ledger.pending_actions()
    # Reproduce the real pre-fix failure: a rejected child-route close left a
    # started receipt, then the candidate's final decision closed that route.
    write_json(actions.miner.root / "action_receipts/close-child-route.started.json", {
        "action_id": "close-child-route", "name": "resolve_experiment",
        "parameters_sha256": digest(close_parameters)})

    actions.save_research_decision(
        action_id="a-decision-control", candidate_id=control_id,
        decision=_decision(control_id, "discard", [], f"{control_id}-evaluation"),
        narrative=_narrative())
    actions.save_research_decision(
        action_id="a-decision-modified", candidate_id=child_id,
        decision=_decision(child_id, "retain", [24], f"{child_id}-evaluation"),
        narrative=_narrative())
    actions.save_research_decision(
        action_id="a-decision-modified-later", candidate_id=child_id,
        decision=_decision(child_id, "retain", [24], f"{child_id}-evaluation"),
        narrative=_narrative())
    closed = actions.resolve_experiment(action_id="close-child-route", **close_parameters)
    assert closed["already_resolved"] is True
    assert closed["decision_ref"] == "a-review-000003"
    assert closed["resolution_decision_ref"] == "a-review-000002"
    assert not actions.ledger.pending_actions()
    a_record_ids = [record["id"] for record in actions.miner.store.all()]
    completed = actions.finish_a_research(
        action_id="a-finish-predeclared-route",
        semantic_match_report={
            "objective": goal["objective"],
            "matches": [
                {"candidate_id": control_id, "matches_goal": False,
                 "reason": "合成路线控制只用于配对，不满足交付目标。",
                 "evidence_refs": a_record_ids},
                {"candidate_id": child_id, "matches_goal": True,
                 "reason": "合成验收只检查变更路线的证据闭环。",
                 "evidence_refs": a_record_ids},
            ],
        })

    manifest = {record["id"]: record for record in actions.miner.store.all()}[
        completed["manifest_ref"]
    ]["data"]
    assert manifest["candidate_ids"] == [control_id, child_id]
    assert manifest["retained_ids"] == [child_id]
    assert manifest["experiments"] == [{
        "record_id": f"experiment-{route_id}",
        "plan": plan,
        "status": "stopped",
    }]


def test_empty_retained_a_seals_but_cannot_freeze_or_meet_goal(tmp_path):
    actions, _, goal = _native_actions(tmp_path, run_id="empty-retained-contract")
    spec = actions.miner.spec
    panel_b = input_panel(spec, "B")
    candidate_id = "candidate-discarded"
    candidate_definition = definition("perp_close", meaning="无保留候选合成验收。")
    evaluated = actions.evaluate_candidate(
        action_id="a-evaluate-discarded", candidate_id=candidate_id,
        definition=candidate_definition)
    assert evaluated["status"] == "computed"
    actions.save_research_decision(
        action_id="a-decision-discarded", candidate_id=candidate_id,
        decision=_decision(candidate_id, "discard", [], f"{candidate_id}-evaluation"),
        narrative=_narrative())

    a_record_ids = [record["id"] for record in actions.miner.store.all()]
    completed = actions.finish_a_research(
        action_id="a-finish-empty-retained",
        semantic_match_report={"objective": goal["objective"], "matches": [{
            "candidate_id": candidate_id, "matches_goal": False,
            "reason": "候选被丢弃，目标未达成。", "evidence_refs": a_record_ids}]})

    assert completed["status"] == "complete"
    assert completed["retained_ids"] == []
    manifest = {record["id"]: record for record in actions.miner.store.all()}["a-manifest"]["data"]
    assert manifest["retained_ids"] == []
    evidence = actions.check_goal_evidence()
    assert evidence["status"] == "incomplete"
    assert evidence["objective_met"] is False
    assert evidence["qualified_idea_ids"] == []

    coordinator = ResearchActions(
        actions.miner, validation_universe=panel_b.universe, goal_spec=goal)
    result = coordinator.handle("coordinator", "read_a_result", {"candidate_id": candidate_id})
    assert result["a_status"] == "sealed"
    assert result["manifest_ref"] == "a-manifest"
    assert result["decision"]["data"]["disposition"] == "discard"
    assert coordinator.get_research_status_for("coordinator")["manifest_ref"] == "a-manifest"
    with pytest.raises(ValueError, match="without retained candidates"):
        coordinator.freeze_a_manifest(
            action_id="coordinator-freeze-empty", manifest_ref=completed["manifest_ref"])
    assert not (actions.miner.root / "frozen_batch.json").exists()

    b_loader = Mock(side_effect=AssertionError("B loader must stay closed for an empty batch"))
    b_actions = ResearchActions(actions.miner, b_panel_loader=b_loader, goal_spec=goal)
    with pytest.raises(ValueError, match="before B validation"):
        b_actions.validate_frozen_batch(action_id="b-validate-empty")
    b_loader.assert_not_called()


def test_a_seal_waits_for_original_candidate_action_receipt_recovery(tmp_path):
    actions, _, goal = _native_actions(tmp_path, run_id="seal-pending-a-action")
    candidate_id = "candidate-seal-pending"
    candidate_definition = definition("perp_close", meaning="验证待恢复动作的封存门。")
    evaluated = actions.evaluate_candidate(
        action_id="a-evaluate-pending", candidate_id=candidate_id,
        definition=candidate_definition)
    assert evaluated["status"] == "computed"
    (actions.ledger.root / "a-evaluate-pending.committed.json").unlink()
    status = actions.get_research_status_for("coordinator")
    assert status["calculation_status"] == "pending"
    assert status["pending_actions"][0]["action_id"] == "a-evaluate-pending"
    actions.save_research_decision(
        action_id="a-decision-pending", candidate_id=candidate_id,
        decision=_decision(candidate_id, "retain", [24], f"{candidate_id}-evaluation"),
        narrative=_narrative())
    evidence_refs = [record["id"] for record in actions.miner.store.all()]
    semantic_match_report = {"objective": goal["objective"], "matches": [{
        "candidate_id": candidate_id, "matches_goal": True,
        "reason": "只验证封存前需要恢复原动作回执。", "evidence_refs": evidence_refs}]}

    with pytest.raises(ValueError, match="earlier A actions lack committed receipts"):
        actions.finish_a_research(action_id="a-finish-pending",
                                  semantic_match_report=semantic_match_report)

    assert not (actions.ledger.root / "a-finish-pending.started.json").exists()
    assert not (actions.miner.root / "a-manifest.json").exists()
    assert not (actions.miner.root / "a-complete.json").exists()
    assert [row["action_id"] for row in actions.ledger.pending_actions()] == ["a-evaluate-pending"]

    recovered = actions.evaluate_candidate(
        action_id="a-evaluate-pending", candidate_id=candidate_id,
        definition=candidate_definition)
    assert recovered == evaluated
    assert not actions.ledger.pending_actions()

    completed = actions.finish_a_research(
        action_id="a-finish-pending", semantic_match_report=semantic_match_report)
    assert completed["status"] == "complete"
    assert not actions.ledger.pending_actions()


def test_invalid_semantic_match_fails_before_creating_a_seal_receipt(tmp_path):
    actions, _, goal = _native_actions(tmp_path, run_id="invalid-semantic-match-preflight")
    candidate_id = "candidate-invalid-semantic-match"
    actions.evaluate_candidate(
        action_id="a-evaluate-semantic-preflight", candidate_id=candidate_id,
        definition=definition("perp_close", meaning="验证语义封存前置校验。"))
    actions.save_research_decision(
        action_id="a-decision-semantic-preflight", candidate_id=candidate_id,
        decision=_decision(candidate_id, "retain", [24], f"{candidate_id}-evaluation"),
        narrative=_narrative())
    invalid_report = {"objective": goal["objective"], "matches": [{
        "candidate_id": candidate_id,
        "matches_goal": False,
        "reason": "该引用集合故意不完整。",
        "evidence_refs": ["goal-spec"],
    }]}

    with pytest.raises(ValueError, match="semantic-match evidence must cite"):
        actions.finish_a_research(action_id="a-finish-invalid-semantic", 
                                  semantic_match_report=invalid_report)
    assert not (actions.ledger.root / "a-finish-invalid-semantic.started.json").exists()
    assert not actions.ledger.pending_actions()

    evidence_refs = [record["id"] for record in actions.miner.store.all()]
    valid_report = {"objective": goal["objective"], "matches": [{
        **invalid_report["matches"][0],
        "reason": "完整证据已经由本地领域动作保存。",
        "evidence_refs": evidence_refs,
    }]}
    completed = actions.finish_a_research(
        action_id="a-finish-valid-semantic", semantic_match_report=valid_report)
    assert completed["status"] == "complete"
    assert not actions.ledger.pending_actions()


def test_freeze_waits_for_read_only_receipt_recovery_after_seal(tmp_path):
    actions, _, goal = _native_actions(tmp_path, run_id="freeze-pending-a-action")
    candidate_id = "candidate-freeze-pending"
    candidate_definition = definition("perp_close", meaning="验证封存后只读回执恢复。")
    evaluated = actions.evaluate_candidate(
        action_id="a-evaluate-sealed", candidate_id=candidate_id,
        definition=candidate_definition)
    actions.save_research_decision(
        action_id="a-decision-sealed", candidate_id=candidate_id,
        decision=_decision(candidate_id, "retain", [24], f"{candidate_id}-evaluation"),
        narrative=_narrative())
    evidence_refs = [record["id"] for record in actions.miner.store.all()]
    semantic_match_report = {"objective": goal["objective"], "matches": [{
        "candidate_id": candidate_id, "matches_goal": True,
        "reason": "只验证封存后的程序回执恢复。", "evidence_refs": evidence_refs}]}
    completed = actions.finish_a_research(
        action_id="a-finish-sealed", semantic_match_report=semantic_match_report)
    a_records_before = actions.miner.store.all()
    completion_before = (actions.miner.root / "a-complete.json").read_bytes()

    (actions.ledger.root / "a-evaluate-sealed.committed.json").unlink()
    coordinator = ResearchActions(
        actions.miner, validation_universe=input_panel(actions.miner.spec, "B").universe,
        goal_spec=goal)
    with pytest.raises(ValueError, match="earlier A actions lack committed receipts"):
        coordinator.freeze_a_manifest(
            action_id="coordinator-freeze-after-replay", manifest_ref=completed["manifest_ref"])

    assert not (actions.ledger.root / "coordinator-freeze-after-replay.started.json").exists()
    assert not (actions.miner.root / "frozen_batch.json").is_file()
    assert [row["action_id"] for row in actions.ledger.pending_actions()] == ["a-evaluate-sealed"]

    recovered = actions.evaluate_candidate(
        action_id="a-evaluate-sealed", candidate_id=candidate_id,
        definition=candidate_definition)
    assert recovered == evaluated
    assert actions.miner.store.all() == a_records_before
    assert (actions.miner.root / "a-complete.json").read_bytes() == completion_before
    assert not actions.ledger.pending_actions()

    frozen = coordinator.freeze_a_manifest(
        action_id="coordinator-freeze-after-replay", manifest_ref=completed["manifest_ref"])
    assert frozen["candidate_ids"] == [candidate_id]
    assert not actions.ledger.pending_actions()


def test_candidate_action_cannot_be_half_frozen_and_seal_blocks_later_writes(tmp_path):
    actions, panel_a, goal = _native_actions(tmp_path, run_id="seal-freeze-race-contract")
    spec = actions.miner.spec
    panel_b = input_panel(spec, "B")
    candidate_id = "candidate-freeze-race"
    candidate_definition = definition("perp_close", meaning="并发封存合成候选。")
    original_evaluate = __import__(
        "crypto_quant.research.factor_mining.research_actions", fromlist=["evaluate_expression"]
    ).evaluate_expression
    calculation_started = threading.Event()
    release_calculation = threading.Event()
    manifest_check_started = threading.Event()
    freeze_call_started = threading.Event()
    evaluate_result: list[dict] = []
    freeze_errors: list[BaseException] = []
    coordinator = ResearchActions(
        actions.miner, validation_universe=panel_b.universe, goal_spec=goal)
    original_manifest_check = coordinator._completed_a_manifest

    def blocked_evaluate(*args, **kwargs):
        calculation_started.set()
        assert release_calculation.wait(timeout=10)
        return original_evaluate(*args, **kwargs)

    def observed_manifest_check(manifest_ref: str):
        manifest_check_started.set()
        return original_manifest_check(manifest_ref)

    def evaluate_candidate_action():
        evaluate_result.append(actions.evaluate_candidate(
            action_id="a-evaluate-freeze-race", candidate_id=candidate_id,
            definition=candidate_definition))

    def freeze_ahead_of_seal():
        freeze_call_started.set()
        try:
            coordinator.freeze_a_manifest(action_id="coordinator-freeze-race",
                                          manifest_ref="a-manifest")
        except BaseException as exc:
            freeze_errors.append(exc)

    with patch("crypto_quant.research.factor_mining.research_actions.evaluate_expression",
               side_effect=blocked_evaluate):
        coordinator._completed_a_manifest = observed_manifest_check
        evaluating = threading.Thread(target=evaluate_candidate_action)
        evaluating.start()
        assert calculation_started.wait(timeout=10)
        freezing = threading.Thread(target=freeze_ahead_of_seal)
        freezing.start()
        assert freeze_call_started.wait(timeout=10)
        time.sleep(0.05)
        assert not manifest_check_started.is_set()
        assert not (actions.miner.root / "frozen_batch.json").exists()
        release_calculation.set()
        evaluating.join(timeout=10)
        freezing.join(timeout=10)

    assert not evaluating.is_alive() and not freezing.is_alive()
    assert evaluate_result and evaluate_result[0]["status"] == "computed"
    assert freeze_errors and "complete and seal A" in str(freeze_errors[0])
    assert manifest_check_started.is_set()
    assert not (actions.miner.root / "a-manifest.json").exists()
    assert not (actions.miner.root / "a-complete.json").exists()
    assert not (actions.miner.root / "frozen_batch.json").exists()

    decision = _decision(candidate_id, "retain", [24], f"{candidate_id}-evaluation")
    actions.save_research_decision(
        action_id="a-decision-freeze-race", candidate_id=candidate_id,
        decision=decision, narrative=_narrative())
    a_record_ids = [record["id"] for record in actions.miner.store.all()]
    completed = actions.finish_a_research(
        action_id="a-finish-freeze-race",
        semantic_match_report={"objective": goal["objective"], "matches": [{
            "candidate_id": candidate_id, "matches_goal": True,
            "reason": "只验证完整manifest和阶段门。", "evidence_refs": a_record_ids}]})
    frozen = coordinator.freeze_a_manifest(
        action_id="coordinator-freeze-race", manifest_ref=completed["manifest_ref"])
    assert frozen["candidate_ids"] == [candidate_id]
    assert actions.miner._checked_frozen()["a_manifest_ref"] == completed["manifest_ref"]

    before = [record["id"] for record in actions.miner.store.all()]
    with pytest.raises(ValueError, match="A is sealed"):
        actions.evaluate_candidate(
            action_id="a-after-seal", candidate_id="candidate-after-seal",
            definition=definition("perp_open", meaning="封存后不得添加。"))
    assert [record["id"] for record in actions.miner.store.all()] == before
