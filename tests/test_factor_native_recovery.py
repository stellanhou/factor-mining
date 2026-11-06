"""Recovery and stop-boundary tests for native factor research actions."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from crypto_quant.research.factor_mining.goal import GoalSpec
from crypto_quant.research.factor_mining.research_actions import ResearchActions
from crypto_quant.research.factor_mining.workflow import FactorMiner
from crypto_quant.research.factor_mining.action_ledger import ResearchActionLedger
from crypto_quant.features.factor_expressions import evaluate_expression
from test_factor_mining import definition, input_panel, specification


class FailingModel:
    def complete(self, *args, **kwargs):
        raise AssertionError("native program recovery must not call role models")


def _actions(tmp_path: Path):
    spec = specification(run_id="native-action-recovery")
    panel = input_panel(spec)
    miner = FactorMiner(spec, FailingModel(), tmp_path / "runs")
    goal = GoalSpec(goal_id=spec.run_id, objective=spec.objective, target_ideas=1)
    miner.store.append("goal-spec", "goal_contract", goal.as_dict())
    actions = ResearchActions(miner, a_panel=panel, goal_spec=goal.as_dict())
    candidate_definition = definition("perp_close", meaning="合成恢复测试公式。")
    return actions, panel, candidate_definition


def _evaluate(actions: ResearchActions, candidate_definition: dict, action_id="evaluate-001"):
    return actions.evaluate_candidate(action_id=action_id, candidate_id="candidate-recovery-0001",
                                      definition=candidate_definition)


def test_factor_expression_observes_a_stop_between_formula_nodes(tmp_path):
    spec = specification(run_id="native-expression-stop")
    panel = input_panel(spec)
    checks = 0

    def stop_check():
        nonlocal checks
        checks += 1
        if checks == 6:
            raise ValueError("program actions are stopped")

    with pytest.raises(ValueError, match="actions are stopped"):
        evaluate_expression("mul(ts_mean(perp_close,3),ts_return(spot_close,2))",
                            panel, stop_check=stop_check)
    assert checks == 6


def test_pending_candidate_before_archive_can_resume_only_with_original_action(tmp_path):
    actions, _, candidate_definition = _actions(tmp_path)
    with patch("crypto_quant.research.factor_mining.research_actions.archive_factor_values",
               side_effect=ValueError("synthetic interruption before value-set commit")):
        with pytest.raises(ValueError, match="before value-set commit"):
            _evaluate(actions, candidate_definition)

    candidate_records = {record["id"] for record in actions.miner.store.all()}
    assert "candidate-recovery-0001-definition" in candidate_records
    assert "candidate-recovery-0001-calculation" not in candidate_records
    assert not list(actions.miner.archive_root.rglob("*.sqlite3"))

    result = _evaluate(actions, candidate_definition)
    assert result["status"] == "computed"
    assert "candidate-recovery-0001-calculation" in {
        record["id"] for record in actions.miner.store.all()}
    with pytest.raises(ValueError, match="action_id was already committed with different parameters"):
        _evaluate(actions, {**candidate_definition, "meaning": "different request"})


def test_pending_candidate_after_archive_recovers_without_formula_recomputation(tmp_path):
    actions, _, candidate_definition = _actions(tmp_path)
    original_evaluate = __import__(
        "crypto_quant.research.factor_mining.research_actions", fromlist=["evaluate_expression"]
    ).evaluate_expression
    evaluate = Mock(wraps=original_evaluate)

    with patch("crypto_quant.research.factor_mining.research_actions.evaluate_expression", evaluate):
        with patch.object(actions, "_persist_candidate_evaluation",
                          side_effect=RuntimeError("synthetic interruption after archive commit")):
            with pytest.raises(RuntimeError, match="after archive commit"):
                _evaluate(actions, candidate_definition)
        assert evaluate.call_count == 1
        result = _evaluate(actions, candidate_definition)

    assert result["status"] == "computed"
    assert evaluate.call_count == 1


def test_stop_waits_for_computation_and_replay_uses_uncommitted_action(tmp_path):
    actions, _, candidate_definition = _actions(tmp_path)
    module = __import__("crypto_quant.research.factor_mining.research_actions",
                        fromlist=["evaluate_expression"])
    original_evaluate = module.evaluate_expression
    calculation_started = threading.Event()
    release_calculation = threading.Event()
    action_errors: list[BaseException] = []
    stop_results: list[dict] = []

    def blocked_evaluate(*args, **kwargs):
        calculation_started.set()
        assert release_calculation.wait(timeout=10)
        return original_evaluate(*args, **kwargs)

    def run_action():
        try:
            _evaluate(actions, candidate_definition)
        except BaseException as exc:  # The stop is expected to reject the next commit.
            action_errors.append(exc)

    def request_stop():
        stop_results.append(actions.request_stop(action_id="stop-001", reason="synthetic stop test"))

    with patch.object(module, "evaluate_expression", side_effect=blocked_evaluate):
        calculation_thread = threading.Thread(target=run_action)
        calculation_thread.start()
        assert calculation_started.wait(timeout=10)
        stop_thread = threading.Thread(target=request_stop)
        stop_thread.start()
        deadline = time.monotonic() + 10
        while actions.ledger.status()["status"] != "stopping" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert actions.ledger.status()["status"] == "stopping"
        release_calculation.set()
        calculation_thread.join(timeout=10)
        stop_thread.join(timeout=10)

    assert not calculation_thread.is_alive() and not stop_thread.is_alive()
    assert stop_results and stop_results[0]["status"] == "stopped"
    assert action_errors and "stopped" in str(action_errors[0])
    assert not (actions.miner.root / "action_receipts/evaluate-001.committed.json").exists()
    assert "candidate-recovery-0001-calculation" not in {
        record["id"] for record in actions.miner.store.all()}

    actions.resume_actions(action_id="resume-001")
    assert _evaluate(actions, candidate_definition)["status"] == "computed"


def test_stop_request_can_be_finished_after_action_ledger_reopens(tmp_path):
    actions, _, _ = _actions(tmp_path)
    action_id = "host-stop-restart-001"
    reason = "Coordinator Session cancellation"

    pending = actions.ledger.begin_stop(action_id, reason=reason)
    assert pending["status"] == "stopping"
    reopened = ResearchActionLedger(actions.miner.root)
    assert reopened.status()["status"] == "stopping"

    result = reopened.finish_stop(action_id, reason=reason)
    assert result["status"] == "stopped"
    assert reopened.status()["status"] == "stopped"
    assert reopened.finish_stop(action_id, reason=reason) == result

    resumed = actions.resume_actions(action_id="host-resume-restart-001")
    assert resumed["status"] == "active"
    assert reopened.status()["status"] == "active"


def _concurrent_evaluations(actions, candidate_definition, action_ids):
    barrier = threading.Barrier(len(action_ids) + 1)
    results: list[dict] = []
    errors: list[BaseException] = []

    def run(action_id):
        barrier.wait(timeout=10)
        try:
            results.append(_evaluate(actions, candidate_definition, action_id=action_id))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(action_id,)) for action_id in action_ids]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=10)
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    return results, errors


def test_concurrent_replay_of_same_action_id_executes_once(tmp_path):
    actions, _, candidate_definition = _actions(tmp_path)
    module = __import__("crypto_quant.research.factor_mining.research_actions",
                        fromlist=["evaluate_expression"])
    original_evaluate = module.evaluate_expression
    evaluate = Mock(wraps=original_evaluate)
    with patch.object(module, "evaluate_expression", evaluate):
        results, errors = _concurrent_evaluations(
            actions, candidate_definition, ["same-action", "same-action"])

    assert not errors
    assert len(results) == 2 and results[0] == results[1]
    assert evaluate.call_count == 1


def test_concurrent_different_action_ids_cannot_take_over_one_candidate(tmp_path):
    actions, _, candidate_definition = _actions(tmp_path)
    module = __import__("crypto_quant.research.factor_mining.research_actions",
                        fromlist=["evaluate_expression"])
    original_evaluate = module.evaluate_expression
    evaluate = Mock(wraps=original_evaluate)
    with patch.object(module, "evaluate_expression", evaluate):
        results, errors = _concurrent_evaluations(
            actions, candidate_definition, ["candidate-action-a", "candidate-action-b"])

    assert len(results) == len(errors) == 1
    assert results[0]["status"] == "computed"
    assert "candidate ID is already bound" in str(errors[0])
    assert evaluate.call_count == 1


def test_native_formula_repair_uses_the_saved_error_and_records_provenance(tmp_path):
    actions, _, candidate_definition = _actions(tmp_path)
    candidate_definition = {**candidate_definition, "expression": "unknown_field"}
    failed = _evaluate(actions, candidate_definition)
    assert failed["status"] == "repair_pending"
    program_error = failed["program_error"]

    repaired = actions.repair_candidate(
        action_id="repair-001", candidate_id="candidate-recovery-0001",
        program_error=program_error, repair_expression="perp_close",
        reason="用允许的永续收盘价字段替换不存在的字段。")

    assert repaired["status"] == "computed"
    assert repaired["calculation"]["repair_refs"] == [
        "candidate-recovery-0001-repair-0001"]
    repair = actions.miner.store._load(
        actions.miner.store.root / "candidate-recovery-0001-repair-0001.json")
    assert repair["kind"] == "formula_repair"
    assert repair["data"]["action_id"] == "repair-001"
    assert repair["data"]["program_error"] == program_error


def test_native_formula_repair_recovers_archived_values_without_recalculation(tmp_path):
    actions, _, candidate_definition = _actions(tmp_path)
    candidate_definition = {**candidate_definition, "expression": "unknown_field"}
    failed = _evaluate(actions, candidate_definition)
    module = __import__("crypto_quant.research.factor_mining.research_actions",
                        fromlist=["evaluate_expression"])
    original_evaluate = module.evaluate_expression
    evaluate = Mock(wraps=original_evaluate)
    with patch.object(module, "evaluate_expression", evaluate):
        with patch.object(actions, "_persist_candidate_evaluation",
                          side_effect=RuntimeError("synthetic repair interruption after archive")):
            with pytest.raises(RuntimeError, match="after archive"):
                actions.repair_candidate(
                    action_id="repair-archive-001", candidate_id="candidate-recovery-0001",
                    program_error=failed["program_error"], repair_expression="perp_close",
                    reason="验证归档后的修复恢复。")
        assert evaluate.call_count == 1
        result = actions.repair_candidate(
            action_id="repair-archive-001", candidate_id="candidate-recovery-0001",
            program_error=failed["program_error"], repair_expression="perp_close",
            reason="验证归档后的修复恢复。")

    assert result["status"] == "computed"
    assert evaluate.call_count == 1
