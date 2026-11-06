"""Native goal checks report missing evidence without synthesizing completion."""

from __future__ import annotations

from dataclasses import replace
import json
import tempfile
from pathlib import Path

from crypto_quant.research.factor_mining.goal import GoalSpec
from crypto_quant.research.factor_mining.goal_evidence import check_native_goal_evidence
from crypto_quant.research.factor_mining.research_actions import ResearchActions
from crypto_quant.research.factor_mining.workflow import FactorMiner
from test_factor_rank_displacement_workflow import _panel, _spec


class NoModelCalls:
    def complete(self, *args, **kwargs):
        raise AssertionError("native goal evidence checks must not call a model")


def _native_run(root: Path, *, purpose: str = "research"):
    spec = replace(_spec(), run_id="native-goal-evidence", purpose=purpose)
    miner = FactorMiner(spec, NoModelCalls(), root / "runs")
    goal = GoalSpec(goal_id="native-goal-evidence", objective=spec.objective,
                    target_ideas=1).as_dict()
    miner.store.append("goal-spec", "goal_contract", goal)
    return miner, goal


def _narrative():
    return {"analysis": "固定合成样本的程序结果解释。", "mechanism": "待检验机制。",
            "conditions": ["合成样本适用条件。"], "falsifiers": ["推翻条件。"],
            "limitations": ["合成数据不能支持实盘结论。"], "next_steps": ["后续独立检验。"]}


def test_native_action_checker_reports_missing_seal_and_checkpoint_without_writes():
    with tempfile.TemporaryDirectory() as directory:
        miner, goal = _native_run(Path(directory))
        actions = ResearchActions(miner, goal_spec=goal)
        before = {path.relative_to(miner.root): path.read_bytes()
                  for path in miner.root.rglob("*") if path.is_file()}

        result = actions.check_goal_evidence()

        after = {path.relative_to(miner.root): path.read_bytes()
                 for path in miner.root.rglob("*") if path.is_file()}
        assert result["status"] == "incomplete"
        assert result["objective_met"] is False
        assert result["qualified_idea_ids"] == []
        assert any("sealed A completion" in gap for gap in result["gaps"])
        assert before == after


def test_native_checker_rejects_missing_or_inconsistent_fixed_goal_spec():
    with tempfile.TemporaryDirectory() as directory:
        miner, goal = _native_run(Path(directory))

        missing = check_native_goal_evidence(miner, None)
        changed = {**goal, "target_ideas": 2}
        inconsistent = check_native_goal_evidence(miner, changed)

        assert missing["objective_met"] is False
        assert any("fixed GoalSpec" in gap for gap in missing["gaps"])
        assert inconsistent["objective_met"] is False
        assert any("fixed GoalSpec differs from the saved A goal contract" in gap
                   for gap in inconsistent["gaps"])


def test_engineering_check_cannot_satisfy_native_research_goal():
    with tempfile.TemporaryDirectory() as directory:
        miner, goal = _native_run(Path(directory), purpose="engineering_check")
        result = ResearchActions(miner, goal_spec=goal).check_goal_evidence()

        assert result["status"] == "incomplete"
        assert result["objective_met"] is False
        assert result["qualified_idea_ids"] == []
        assert any("engineering_check runs cannot satisfy a research goal" in gap
                   for gap in result["gaps"])


def test_native_quantity_goal_counts_only_a_matched_and_program_delivered_card():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec = replace(_spec(), run_id="native-goal-positive", purpose="research")
        panel_a, panel_b = _panel(spec, "A"), _panel(spec, "B")
        miner = FactorMiner(spec, NoModelCalls(), root / "runs")
        goal = GoalSpec("native-goal-positive", spec.objective, target_ideas=1).as_dict()
        miner.store.append("goal-spec", "goal_contract", goal)
        idea_pool = root / "idea_pool"
        idea_pool.mkdir()
        actions = ResearchActions(
            miner, a_panel=panel_a, b_panel_loader=lambda: panel_b,
            validation_universe=panel_b.universe, idea_pool=idea_pool, goal_spec=goal)
        candidate_id = "candidate-0001"
        definition = {"name": "合成永续价格输入", "expression": "perp_close",
                      "meaning": "固定六币合成样本的永续收盘价。",
                      "hypothesis": "只验证程序证据闭环。", "direction": 1,
                      "parent_id": None, "proposal_id": None,
                      "change_reason": "使用固定合成表达式。"}

        evaluated = actions.evaluate_candidate(
            action_id="a-evaluate", candidate_id=candidate_id, definition=definition)
        assert evaluated["status"] == "computed"
        decision = {"candidate_id": candidate_id, "disposition": "retain",
                    "retained_horizons": [24], "continue_optimization": False,
                    "answers": {key: {"supported": False, "reason": "合成流程验收。"}
                                for key in ("research_basis", "modification_hypothesis",
                                            "verifiable_improvement", "attempt_value")},
                    "reason": "保留候选以检验合成A/B闭环。", "resume_condition": None,
                    "evidence_refs": [f"{candidate_id}-evaluation"]}
        actions.save_research_decision(
            action_id="a-decision", candidate_id=candidate_id,
            decision=decision, narrative=_narrative())
        a_refs = [record["id"] for record in miner.store.all()]
        completed = actions.finish_a_research(
            action_id="a-finish",
            semantic_match_report={"objective": spec.objective, "matches": [{
                "candidate_id": candidate_id, "matches_goal": True,
                "reason": "此测试仅验证原生目标核验链路。", "evidence_refs": a_refs}]})
        actions.freeze_a_manifest(action_id="coordinator-freeze",
                                  manifest_ref=completed["manifest_ref"])
        numerical = actions.validate_frozen_batch(action_id="b-validate")
        assert numerical["status"] == "numerical_complete"
        assert numerical["decisions"][candidate_id]["eligible_for_idea_pool"] is True
        actions.save_b_interpretation(action_id="b-report", candidate_id=candidate_id,
                                      narrative=_narrative())
        delivered = actions.finalize_b_delivery(action_id="b-delivery")
        assert delivered["status"] == "complete"

        result = actions.check_goal_evidence()

        assert result["status"] == "complete"
        assert result["objective_met"] is True
        assert result["qualified_idea_ids"] == [
            f"{spec.run_id}--{candidate_id}"]
        assert result["gaps"] == []

        delivery = json.loads((miner.root / "b-delivery.json").read_text(encoding="utf-8"))
        card_path = Path(delivery["candidates"][0]["card_path"])
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card["id"] = "wrong-card-identity"
        card_path.write_text(json.dumps(card), encoding="utf-8")
        corrupted = actions.check_goal_evidence()
        assert corrupted["status"] == "incomplete"
        assert corrupted["objective_met"] is False
        assert corrupted["qualified_idea_ids"] == []
        assert any("card hash differs" in gap for gap in corrupted["gaps"])


def test_native_quality_goal_recomputes_a_evidence_and_requires_the_same_b_horizon():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec = replace(_spec(), run_id="native-quality-positive", purpose="research")
        panel_a, panel_b = _panel(spec, "A"), _panel(spec, "B")
        target = {"metric": "rank_displacement", "horizon_hours": 24,
                  "displacement_hours": 1, "min_improvement": 0.01, "max_ic_loss": 0.2}
        miner = FactorMiner(spec, NoModelCalls(), root / "runs")
        goal = GoalSpec("native-quality-positive", spec.objective,
                        quality_target=target).as_dict()
        miner.store.append("goal-spec", "goal_contract", goal)
        idea_pool = root / "idea_pool"
        idea_pool.mkdir()
        actions = ResearchActions(
            miner, a_panel=panel_a, b_panel_loader=lambda: panel_b,
            validation_universe=panel_b.universe, idea_pool=idea_pool, goal_spec=goal)
        control_id, candidate_id = "candidate-0001", "candidate-0002"
        control = {"name": "合成噪声对照", "expression": "premium_index",
                   "meaning": "合成噪声序列。", "hypothesis": "作为配对对照。",
                   "direction": 1, "parent_id": None, "proposal_id": None,
                   "change_reason": "固定合成对照。"}
        trial = {"name": "合成永续价格候选", "expression": "perp_close",
                 "meaning": "合成永续价格序列。", "hypothesis": "检验程序质量链。",
                 "direction": 1, "parent_id": control_id, "proposal_id": "proposal-1",
                 "change_reason": "固定合成候选。"}
        actions.evaluate_candidate(action_id="a-control", candidate_id=control_id,
                                   definition=control)
        plan = {"route_id": "quality-route", "control_id": control_id,
                "evidence_refs": [f"{control_id}-evaluation"],
                "modification_task": {"core_hypothesis": "合成检验。",
                                      "observed_problem": "对照序列排名变化较大。",
                                      "modification_hypothesis": "候选排序较稳定。",
                                      "change_target": "替换为固定价格输入。",
                                      "fixed_components": "目标期限、方向和数据保持不变。"},
                "experiment_design": {"question": "候选是否降低排名变化？",
                                      "metric": "rank_displacement", "horizon_hours": 24,
                                      "displacement_hours": 1, "min_improvement": 0.01,
                                      "max_ic_loss": 0.2, "expected_outcome": "D下降。",
                                      "stop_condition": "改善不足则停止。",
                                      "pause_condition": "样本不足时暂停。"},
                "restart_of": None, "new_evidence": None}
        actions.register_experiment(action_id="a-register-route", plan=plan)
        actions.evaluate_candidate(action_id="a-trial", candidate_id=candidate_id,
                                   definition=trial, experiment_id="quality-route")
        for item_id, disposition, retained in (
                (control_id, "discard", []), (candidate_id, "retain", [24])):
            decision = {"candidate_id": item_id, "disposition": disposition,
                        "retained_horizons": retained, "continue_optimization": False,
                        "answers": {key: {"supported": False, "reason": "合成流程验收。"}
                                    for key in ("research_basis", "modification_hypothesis",
                                                "verifiable_improvement", "attempt_value")},
                        "reason": "固定合成流程决策。", "resume_condition": None,
                        "evidence_refs": [f"{item_id}-evaluation"]}
            actions.save_research_decision(
                action_id=f"a-decision-{item_id}", candidate_id=item_id,
                decision=decision, narrative=_narrative())
        a_refs = [record["id"] for record in miner.store.all()]
        matches = [
            {"candidate_id": control_id, "matches_goal": False,
             "reason": "对照不满足目标。", "evidence_refs": a_refs},
            {"candidate_id": candidate_id, "matches_goal": True,
             "reason": "候选对应目标。", "evidence_refs": a_refs},
        ]
        completed = actions.finish_a_research(
            action_id="a-finish-quality",
            semantic_match_report={"objective": spec.objective, "matches": matches})
        actions.freeze_a_manifest(action_id="coordinator-freeze-quality",
                                  manifest_ref=completed["manifest_ref"])
        numerical = actions.validate_frozen_batch(action_id="b-validate-quality")
        assert numerical["decisions"][candidate_id]["passed_horizons"] == [24]
        actions.save_b_interpretation(
            action_id="b-report-quality", candidate_id=candidate_id,
            narrative=_narrative())
        actions.finalize_b_delivery(action_id="b-delivery-quality")

        result = actions.check_goal_evidence()

        assert result["status"] == "complete"
        assert result["objective_met"] is True
        assert result["target"] == target
        assert result["qualified_idea_ids"] == [f"{spec.run_id}--{candidate_id}"]
        assert result["gaps"] == []

