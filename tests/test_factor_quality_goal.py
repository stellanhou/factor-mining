"""Quality completion uses paired A evidence and matching-horizon B admission."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from test_factor_goal import GoalModel, SETTINGS
from test_factor_rank_displacement_workflow import DeterministicModel, _panel, _spec
from crypto_quant.research.factor_mining.contracts import dumps
from crypto_quant.research.factor_mining.evaluation import evaluate_factor
from crypto_quant.research.factor_mining.goal import GoalRunner, GoalSpec
from crypto_quant.research.factor_mining.model import ModelReply


TARGET = {"metric": "rank_displacement", "horizon_hours": 24, "displacement_hours": 1,
          "min_improvement": 0.01, "max_ic_loss": 0.2}


class QualityModel(GoalModel):
    def __init__(self, *, horizons=(24,), weaken=False):
        super().__init__()
        self.factor_model = DeterministicModel()
        self.horizons = list(horizons)
        self.weaken = weaken

    def complete(self, messages, *, max_output_tokens, session_id):
        request = json.loads(messages[1]["content"])
        if "goal_phase" in request["payload"]:
            return super().complete(messages, max_output_tokens=max_output_tokens, session_id=session_id)
        reply = self.factor_model.complete(messages, max_output_tokens=max_output_tokens, session_id=session_id)
        if request["role"] == "optimizer":
            result = json.loads(reply.text)
            for decision in result["result"]["decisions"]:
                decision["retained_horizons"] = self.horizons
            if self.weaken:
                for proposal in result["result"]["proposals"]:
                    proposal["experiment_design"]["min_improvement"] = 0.00001
            return ModelReply(dumps(result), reply.usage, reply.model)
        return reply


class QualityGoalTests(unittest.TestCase):
    def create(self, directory, *, stable=False, model=None, cycles=1):
        spec = _spec()
        a, b = _panel(spec, "A"), _panel(spec, "B")
        if not stable:
            a.values["perp_close"] = 100 + np.random.default_rng(14).normal(0, 1, len(a.values))
        goal = GoalSpec("quality-fixture", "研究有依据的差异与配对改善", quality_target=copy.deepcopy(TARGET),
                        max_cycles=cycles)
        runner = GoalRunner.create(goal, spec, model or QualityModel(), Path(directory),
                                   inputs={}, model_settings=SETTINGS)
        return runner, a, b

    @staticmethod
    def passing_B(values, labels, spec, stage, direction, *, horizon_hours=24):
        report = evaluate_factor(values, labels, spec, stage, direction, horizon_hours=horizon_hours)
        if stage == "B":
            report["summary"]["rank_ic"].update(mean=direction * 0.2, p_value=0.000001)
            report["summary"]["directional_spread"].update(mean=0.003, p_value=0.000001)
        return report

    def run_goal(self, runner, a, b, evaluation=None):
        with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor",
                   side_effect=evaluation or self.passing_B):
            return runner.run(lambda: a, lambda: b, lambda: b.universe, runner.root / "ideas",
                              poll_seconds=1)

    def test_quality_contract_is_explicit_and_exclusive(self):
        goal = GoalSpec("quality", "配对质量", quality_target=TARGET, max_cycles=12)
        self.assertNotIn("target_ideas", goal.as_dict())
        unlimited = GoalSpec("quality", "配对质量", quality_target=TARGET)
        self.assertIsNone(unlimited.max_cycles)
        self.assertNotIn("max_cycles", unlimited.as_dict())
        for changes in ({"target_ideas": 1}, {"max_cycles": 0}, {"max_cycles": -1},
                        {"max_cycles": True}, {"max_cycles": 1.5},
                        {"quality_target": {**TARGET, "max_ic_loss": -0.1}},
                        {"quality_target": {**TARGET, "min_improvement": float("nan")}},
                        {"quality_target": {**TARGET, "horizon_hours": 3}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                GoalSpec("quality", "配对质量", **{"quality_target": TARGET, "max_cycles": 12, **changes})

    def test_unlimited_quality_goal_completes_on_program_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, b = self.create(directory, cycles=None)
            result = self.run_goal(runner, a, b)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["qualified_ideas"]), 1)
            forbidden = lambda: self.fail("completed unlimited Goal must not read data")
            self.assertEqual(GoalRunner(runner.root, runner.model).run(
                forbidden, forbidden, forbidden, runner.root / "ideas", poll_seconds=1), result)

    def test_cancelled_limit_resumes_next_cycle_without_replaying_finished_run(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, b = self.create(directory, stable=True)
            result = self.run_goal(runner, a, b)
            self.assertEqual(result["status"], "budget_exhausted")
            original_goal = (runner.root / "goal.json").read_bytes()
            old_run = runner.root / "runs" / result["current_run"]
            b_checkpoint = (old_run / "b-numerical-complete.json").read_bytes()
            runner._save(max_cycles=None)
            resumed = GoalRunner(runner.root, runner.model)
            self.assertIsNone(resumed.goal.max_cycles)

            def load_A():
                if resumed.state["phase"] == "explore":
                    self.assertEqual(resumed.state["cycle"], 2)
                    self.assertNotEqual(resumed.state["current_run"], old_run.name)
                    raise KeyboardInterrupt()
                self.assertEqual(resumed.state["phase"], "select_task")
                self.assertIsNone(resumed.state["current_run"])
                return a

            forbidden_B = lambda: self.fail("continuation must not replay the finished B cycle")
            with self.assertRaises(KeyboardInterrupt):
                resumed.run(load_A, forbidden_B, forbidden_B, runner.root / "ideas", poll_seconds=1)
            self.assertEqual(resumed.state["status"], "paused")
            self.assertEqual(resumed.state["cycle"], 2)
            self.assertEqual((old_run / "b-numerical-complete.json").read_bytes(), b_checkpoint)
            self.assertEqual((runner.root / "goal.json").read_bytes(), original_goal)

    def test_actual_pairing_achieves_quality_and_resumes_without_new_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, b = self.create(directory)
            result = self.run_goal(runner, a, b)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["qualified_ideas"]), 1)
            receipt = runner.receipts.all()[0]["data"]["ideas"]
            parent, trial = receipt
            self.assertFalse(parent["quality_evidence"]["passed"])
            self.assertTrue(trial["quality_evidence"]["passed"])
            self.assertGreaterEqual(trial["quality_evidence"]["paired_improvement"]["ci"][0], 0.01)
            self.assertGreaterEqual(trial["quality_evidence"]["paired_ic_change"]["ci"][0], -0.2)
            review = next(r for r in runner.model.requests if r["payload"].get("goal_phase") == "verify_completion")
            encoded = dumps(review)
            self.assertNotIn('"segment":"B"', encoded)
            self.assertNotIn('"passed_horizons"', encoded)
            resumed = GoalRunner(runner.root, runner.model)
            forbidden = lambda: self.fail("completed quality Goal must not read data")
            self.assertEqual(resumed.run(forbidden, forbidden, forbidden, runner.root / "ideas", poll_seconds=1), result)

    def test_B_cards_and_positive_model_review_cannot_override_failed_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, b = self.create(directory, stable=True)
            result = self.run_goal(runner, a, b)
            self.assertEqual(result["status"], "budget_exhausted")
            self.assertEqual(result["qualified_ideas"], [])
            self.assertTrue(list((runner.root / "ideas").glob("*.json")))
            receipt = runner.receipts.all()[0]["data"]["ideas"]
            self.assertTrue(all(not item["quality_evidence"]["passed"] for item in receipt))
            forbidden = lambda: self.fail("exhausted quality Goal must not restart")
            self.assertEqual(GoalRunner(runner.root, runner.model).run(
                forbidden, forbidden, forbidden, runner.root / "ideas", poll_seconds=1), result)

    def test_quality_horizon_must_itself_pass_B(self):
        def only_one_hour(values, labels, spec, stage, direction, *, horizon_hours=24):
            report = self.passing_B(values, labels, spec, stage, direction, horizon_hours=horizon_hours)
            if stage == "B" and horizon_hours == 24:
                report["summary"]["rank_ic"].update(mean=0.0, p_value=1.0)
            return report
        with tempfile.TemporaryDirectory() as directory:
            runner, a, b = self.create(directory, model=QualityModel(horizons=(1, 24)))
            result = self.run_goal(runner, a, b, only_one_hour)
            self.assertEqual(result["status"], "budget_exhausted")
            self.assertEqual(result["qualified_ideas"], [])
            receipt = runner.receipts.all()[0]["data"]["ideas"]
            self.assertTrue(any(item["quality_evidence"]["passed"] for item in receipt))

    def test_optimizer_cannot_weaken_frozen_quality_target(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, b = self.create(directory, model=QualityModel(weaken=True))
            result = self.run_goal(runner, a, b)
            self.assertEqual(result["status"], "budget_exhausted")
            comparisons = list((runner.root / "runs").glob("*/a_records/*-comparison.json"))
            self.assertFalse(comparisons)
            calls = list((runner.root / "runs").glob("*/model_calls/*-response.json"))
            self.assertTrue(calls)
            records = list((runner.root / "runs").glob("*/a_records/*.json"))
            self.assertTrue(any("frozen Goal quality target" in path.read_text() for path in records))
            self.assertEqual(runner.state["qualified_ideas"], [])
