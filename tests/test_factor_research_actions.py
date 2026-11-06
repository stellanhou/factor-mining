"""Program factor actions preserve the legacy evidence and do not call role models."""
from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from test_factor_rank_displacement_workflow import (
    DeterministicModel, _panel, _spec,
)
from crypto_quant.research.factor_mining.evaluation import build_labels
from crypto_quant.features.factor_expressions import evaluate_expression
from crypto_quant.research.factor_mining.goal import GoalRunner, GoalSpec
from crypto_quant.research.factor_mining.research_actions import ResearchActions
from crypto_quant.research.factor_mining.goal_evidence import check_goal_evidence
from crypto_quant.research.factor_mining.records import RecordStore
from crypto_quant.research.factor_mining.workflow import FactorMiner


class FailingModel:
    def complete(self, *args, **kwargs):
        raise AssertionError("program action called an old role model")


class ResearchActionTests(unittest.TestCase):
    def test_program_formula_evaluation_pair_and_archives_match_legacy_evidence(self):
        spec = _spec()
        panel = _panel(spec, "A")
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, DeterministicModel(), Path(directory) / "runs")
            miner.explore(panel)
            miner.model = FailingModel()
            miner.gateway.model = miner.model
            actions = ResearchActions(miner)

            for candidate_id, candidate in miner.candidates.items():
                expression = candidate["definition"]["expression"]
                calculated_values = actions.calculate_candidate(candidate_id, panel)
                record = miner.store._load(miner.store.root / f"{candidate_id}-calculation.json")
                self.assertEqual(record["data"], {"candidate_id": candidate_id, **candidate["calculation"]})
                compiled, result = actions.calculate_formula(expression, panel)
                legacy = evaluate_expression(expression, panel)
                self.assertEqual(compiled.expanded_expression, legacy.definition["expanded_expression"])
                pd.testing.assert_series_equal(result.values, legacy.values)
                pd.testing.assert_series_equal(calculated_values, legacy.values)
                report = actions.evaluate_a(result.values, panel, candidate["definition"]["direction"])
                self.assertEqual(report, candidate["evaluation"])
                self.assertEqual(actions.evaluate_d(result.values, panel), candidate["rank_displacement"])

                calculation = actions.archive_factor_values(candidate_id, candidate["definition"], result)
                self.assertEqual(calculation["values_artifact"], candidate["calculation"]["values_artifact"])

            paired = next((cid, candidate) for cid, candidate in miner.candidates.items()
                          if candidate["experiment"] is not None)
            candidate_id, candidate = paired
            control = miner.candidates[candidate["experiment"]["control_id"]]
            trial_values = evaluate_expression(candidate["definition"]["expression"], panel).values
            control_values = evaluate_expression(control["definition"]["expression"], panel).values
            comparison = actions.compare_candidates(
                trial_values, control_values, panel, build_labels(panel, spec, "A"),
                candidate["definition"]["direction"], control["definition"]["direction"],
                candidate["experiment"])
            saved_comparison = {key: value for key, value in candidate["comparison"].items()
                                if key != "route_decision"}
            self.assertEqual(comparison, saved_comparison)

            displayed = actions._candidate_result(candidate_id)
            evaluation_ref = displayed["evidence_pointers"]["evaluation"]
            for field in ("summary", "coverage"):
                source = actions.read_a_evidence(
                    record_id=evaluation_ref["record_id"],
                    pointer=evaluation_ref["field_pointers"][field], offset=0, limit=1)
                self.assertEqual(source["data"], displayed["evaluation"][field])
            for horizon, field_pointers in evaluation_ref["field_pointers"]["horizons"].items():
                for field, pointer in field_pointers.items():
                    source = actions.read_a_evidence(
                        record_id=evaluation_ref["record_id"], pointer=pointer, offset=0, limit=1)
                    self.assertEqual(source["data"], displayed["evaluation"]["horizons"][horizon][field])

            comparison_ref = displayed["evidence_pointers"]["comparison"]
            for field, pointer in comparison_ref["field_pointers"].items():
                source = actions.read_a_evidence(
                    record_id=comparison_ref["record_id"], pointer=pointer, offset=0, limit=1)
                self.assertEqual(source["data"], displayed["comparison"][field])

    def test_program_freeze_b_completion_and_admission_do_not_call_role_models(self):
        spec = _spec()
        panel_a, panel_b = _panel(spec, "A"), _panel(spec, "B")
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, DeterministicModel(), Path(directory) / "runs")
            completion = miner.explore(panel_a)
            retained = completion["retained_ids"]
            narrative = {"analysis": "固定样本的程序结果解释。", "mechanism": "待验证机制。",
                         "conditions": ["适用条件。"], "falsifiers": ["推翻条件。"],
                         "limitations": ["数据限制。"], "next_steps": ["下一步检验。"]}
            for cid in retained:
                (miner.store.root / f"{cid}-report.json").unlink()
            program = FactorMiner.open(miner.root, FailingModel())
            actions = ResearchActions(program)
            for cid in retained:
                actions.submit_report("A", cid, narrative)
            actions.freeze(retained, panel_b.universe)

            legacy = FactorMiner.open(miner.root, DeterministicModel())
            legacy._try_report = lambda _cid, _gateway: None
            pending = legacy.validate(lambda: panel_b, Path(directory) / "ideas")
            self.assertEqual(pending["status"], "reports_pending")
            self.assertTrue(pending["pending_report_ids"])

            program = FactorMiner.open(miner.root, FailingModel())
            actions = ResearchActions(program)
            pending_program = actions.complete_b_reports(Path(directory) / "ideas")
            self.assertEqual(pending_program, pending)
            for cid in pending["pending_report_ids"]:
                actions.submit_report("B", cid, narrative)
            completed = actions.complete_b_reports(Path(directory) / "ideas")
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(actions.correct_and_admit(
                {record["data"]["candidate_id"]: {key: value for key, value in record["data"].items()
                                                  if key != "candidate_id"}
                 for record in RecordStore(program.root / "b_records", run_root=program.root,
                                           archive_root=program.archive_root).all()
                 if record["kind"] == "evaluation"}),
                {"batch_correction": completed["batch_correction"],
                 "decisions": {cid: {key: value for key, value in decision.items()
                                     if key != "idea_card"}
                               for cid, decision in completed["decisions"].items()}})

    def test_goal_evidence_reports_missing_semantic_match_and_delivery(self):
        spec = _spec()
        with tempfile.TemporaryDirectory() as directory:
            runner = GoalRunner.create(
                GoalSpec("evidence-gaps", "检查目标证据", target_ideas=1), spec,
                FailingModel(), Path(directory), inputs={}, model_settings={})
            result = check_goal_evidence(runner.root)
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("semantic-match report", " ".join(result["gaps"]))
            self.assertIn("delivery evidence", " ".join(result["gaps"]))

    def test_program_goal_review_preparation_and_match_submission_avoid_model_calls(self):
        from test_factor_goal import GoalTests

        with tempfile.TemporaryDirectory() as directory:
            fixtures = GoalTests()
            runner, panel_a, panel_b = fixtures.create(directory)
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor",
                       side_effect=fixtures.passing_B):
                result = fixtures.run_goal(runner, panel_a, panel_b)
            self.assertEqual(result["status"], "complete")
            runner.model = FailingModel()
            run_root = runner.root / "runs" / runner.state["current_run"]
            miner = FactorMiner.open(run_root, FailingModel())
            validation = json.loads((run_root / "validation.json").read_text())
            actions = ResearchActions(miner)

            prepared = actions.prepare_goal_review(runner, validation)
            self.assertTrue(prepared["ids"])
            match_record = next(record for record in runner.receipts.all()
                                if record["id"] == prepared["match_id"])
            submitted = actions.submit_goal_match(
                runner, prepared["match_id"], prepared["ids"], match_record["data"])
            self.assertEqual(submitted, match_record["data"])
            self.assertEqual(actions.check_goal_evidence(runner)["status"], "complete")

    def _completed_quantity_goal(self, directory):
        from test_factor_goal import GoalTests

        fixtures = GoalTests()
        runner, panel_a, panel_b = fixtures.create(directory)
        with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor",
                   side_effect=fixtures.passing_B):
            result = fixtures.run_goal(runner, panel_a, panel_b)
        self.assertEqual(result["status"], "complete")
        return runner

    def test_unverified_delivery_card_does_not_count_toward_goal(self):
        for corruption in ("missing_card", "changed_card_id"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                runner = self._completed_quantity_goal(directory)
                before = check_goal_evidence(runner.root)
                self.assertEqual(before["status"], "complete")
                idea_id = runner.state["qualified_ideas"][0]
                delivery_file = next((runner.root / "delivery_evidence").glob("admission-*.json"))
                delivery = json.loads(delivery_file.read_text())
                card = next(item for item in delivery["ideas"] if item["idea_id"] == idea_id)
                card_path = Path(card["card_path"])
                if corruption == "missing_card":
                    card_path.unlink()
                else:
                    saved_card = json.loads(card_path.read_text())
                    saved_card["id"] = idea_id + "-changed"
                    card_path.write_text(json.dumps(saved_card))

                result = check_goal_evidence(runner.root)
                self.assertEqual(result["status"], "incomplete")
                self.assertFalse(result["objective_met"])
                self.assertNotIn(idea_id, result["qualified_idea_ids"])

    def test_goal_evidence_requires_quality_and_same_horizon_b_admission(self):
        spec = _spec()
        target = {"metric": "rank_displacement", "horizon_hours": 24,
                  "displacement_hours": 1, "min_improvement": 0.01, "max_ic_loss": 0.2}
        with tempfile.TemporaryDirectory() as directory:
            runner = GoalRunner.create(
                GoalSpec("quality-evidence", "检查质量证据", quality_target=target), spec,
                FailingModel(), Path(directory), inputs={}, model_settings={})
            idea_id = "quality-evidence--candidate-0001"
            card_path = runner.root / "ideas" / f"{idea_id}.json"
            card_path.parent.mkdir(parents=True)
            card_path.write_text(json.dumps({"id": idea_id}))
            quality = {"passed": True, "target": target, "segment": "A"}
            admission = {"eligible_for_idea_pool": True}
            runner.receipts.append("admission-00000001", "program_admission",
                                   {"ideas": [{"idea_id": idea_id, "quality_evidence": quality}]})
            runner.receipts.append("match-00000001", "goal_match",
                                   {"matches": [{"idea_id": idea_id, "matches_goal": True,
                                                  "reason": "逐项匹配。"}]})
            delivery_path = runner.root / "delivery_evidence" / "admission-00000001.json"
            delivery_path.parent.mkdir(parents=True)
            delivery_path.write_text(json.dumps({"ideas": [{
                "idea_id": idea_id, "card_path": str(card_path), "quality_evidence": quality,
                "passed_horizons": [24], "admission_evidence": admission}]}))
            runner._save(qualified_ideas=[idea_id])

            result = check_goal_evidence(runner.root)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["qualified_idea_ids"], [idea_id])


if __name__ == "__main__":
    unittest.main()
