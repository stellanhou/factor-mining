"""Frozen multi-horizon hypotheses, correction family, and single-card delivery."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from test_factor_mining import ScenarioModel, definition, input_panel, research_decision, specification
from test_factor_goal import GoalModel, SETTINGS
from crypto_quant.features.factor_expressions import evaluate_expression
from crypto_quant.research.factor_mining.contracts import dumps
from crypto_quant.research.factor_mining.evaluation import build_labels, correct_batch, evaluate_factor
from crypto_quant.research.factor_mining.goal import GoalRunner, GoalSpec
from crypto_quant.research.factor_mining.records import RecordStore
from crypto_quant.research.factor_mining.workflow import FactorMiner


class HorizonTests(unittest.TestCase):
    def test_mixed_frozen_scopes_include_unavailable_tests_in_one_bh_family(self):
        def report(horizon, p):
            return {"horizon_hours": horizon, "summary": {"rank_ic": {"p_value": p}}}
        reports = {
            "a": {"horizons": {"1": report(1, .001), "4": report(4, .04)}},
            "b": {"horizons": {"24": report(24, None)}},
        }
        result = correct_batch(reports, specification())
        self.assertEqual(result["family_size"], 3)
        self.assertEqual([(t["candidate_id"], t["horizon_hours"]) for t in result["tests"]],
                         [("a", 1), ("a", 4), ("b", 24)])
        self.assertEqual([t["adjusted_p"] for t in result["tests"]], [.003, .06, 1.0])
        self.assertEqual([t["rejected"] for t in result["tests"]], [True, False, False])
        reordered = {cid: {"horizons": dict(reversed(list(reports[cid]["horizons"].items())))}
                     for cid in reversed(reports)}
        self.assertEqual(correct_batch(reordered, specification()), result)
        reports["a"]["horizons"]["1"]["horizon_hours"] = 4
        with self.assertRaisesRegex(ValueError, "correction horizon"):
            correct_batch(reports, specification())

    def test_b_labels_use_each_horizons_next_open_and_boundary(self):
        spec = specification()
        panel = input_panel(spec, "B")
        start, end = spec.bounds("B")
        for horizon in (1, 4, 24):
            with self.subTest(horizon=horizon):
                labels = build_labels(panel, spec, "B", horizon_hours=horizon)
                prices = panel.values["perp_open"]
                expected = prices.loc[(start + pd.Timedelta(hours=horizon + 1), "BTCUSDT")] / prices.loc[
                    (start + pd.Timedelta(hours=1), "BTCUSDT")] - 1
                self.assertAlmostEqual(labels.loc[(start, "BTCUSDT"), "forward_return"], expected)
                self.assertTrue(labels.loc[(end - pd.Timedelta(hours=horizon + 1), "BTCUSDT"), "purged"])
                self.assertEqual(int(labels.groupby(level="symbol")["purged"].sum().iloc[0]), horizon + 1)

    def test_unavailable_horizons_remain_in_family_and_render_without_a_card(self):
        def transform(request, result):
            if request["role"] == "ideator":
                result["candidates"] = [definition("ts_mean(premium_index,3)")]
            if request["role"] == "optimizer":
                for decision in result["decisions"]:
                    if decision["disposition"] == "retain":
                        decision["retained_horizons"] = [1, 4, 24]
        with tempfile.TemporaryDirectory() as directory:
            model = ScenarioModel(transform, propose_once=False)
            miner, panel, _ = self._miner(directory, [1, 4, 24], model=model)
            panel.values["premium_index"] = float("nan")
            result = miner.validate(lambda: panel, Path(directory) / "ideas")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["batch_correction"]["family_size"], 3)
            self.assertTrue(all(test["raw_p"] is None and test["adjusted_p"] == 1.0
                                for test in result["batch_correction"]["tests"]))
            self.assertFalse(result["decisions"]["candidate-0001"]["eligible_for_idea_pool"])
            self.assertIn("N/A", (miner.root / "B-report.md").read_text())
            self.assertFalse((Path(directory) / "ideas").exists())

    def _miner(self, directory, horizons, *, model=None):
        def transform(request, result):
            if request["role"] == "optimizer":
                for decision in result["decisions"]:
                    if decision["disposition"] == "retain":
                        decision["retained_horizons"] = list(horizons)
        model = model or ScenarioModel(transform, propose_once=False)
        spec = specification(purpose="research")
        miner = FactorMiner(spec, model, Path(directory))
        miner.explore(input_panel(spec))
        panel = input_panel(spec, "B")
        miner.freeze(["candidate-0001"], panel.universe)
        return miner, panel, model

    @staticmethod
    def _controlled_evaluation(values, labels, spec, stage, direction, *, horizon_hours=24):
        # Actual numerical payload; controlled summary checks the admission branch,
        # not predictive ability of synthetic prices.
        report = evaluate_factor(values, labels, spec, stage, direction, horizon_hours=horizon_hours)
        report["summary"]["rank_ic"].update(mean=direction * .2, p_value=.000001)
        report["summary"]["directional_spread"].update(mean=.003)
        return report

    def test_multiple_passed_horizons_write_one_card_and_recovery_reads_no_b(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, panel, model = self._miner(directory, [1, 4, 24])
            loader = Mock(return_value=panel)
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor",
                       side_effect=self._controlled_evaluation), patch(
                           "crypto_quant.research.factor_mining.workflow.evaluate_expression",
                           wraps=evaluate_expression) as calculation:
                result = miner.validate(loader, Path(directory) / "ideas")
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(calculation.call_count, 1)
            verdict = result["decisions"]["candidate-0001"]
            self.assertEqual(verdict["retained_horizons"], [1, 4, 24])
            self.assertEqual(verdict["passed_horizons"], [1, 4, 24])
            self.assertEqual(result["batch_correction"]["family_size"], 3)
            cards = list((Path(directory) / "ideas").glob("*.json"))
            self.assertEqual(len(cards), 1)
            card = json.loads(cards[0].read_text())
            self.assertEqual(card["market_and_horizon"]["passed_horizons"], [1, 4, 24])
            self.assertEqual(card["admission_evidence"]["passed_horizons"], [1, 4, 24])
            rendered = (miner.root / "B-report.md").read_text()
            for horizon in (1, 4, 24):
                self.assertIn(f"plots/candidate-0001-B-{horizon}h.svg", rendered)
                self.assertIn(f"· {horizon}h", rendered)
            calls = len(model.requests)
            self.assertEqual(miner.complete_reports("B", Path(directory) / "ideas"), result)
            self.assertEqual(len(model.requests), calls)
            self.assertEqual(len(list((Path(directory) / "ideas").glob("*.json"))), 1)
            store = RecordStore(miner.root / "b_records")
            for horizon in (1, 4, 24):
                report = store.read("candidate-0001-evaluation", f"/horizons/{horizon}", 0, 1)
                self.assertEqual(report["horizon_hours"], horizon)
                self.assertEqual(report["coverage"]["purged_hours"], horizon + 1)
                rows = store.read("candidate-0001-evaluation", f"/horizons/{horizon}/periods", 0, 2)
                self.assertEqual(len(rows["items"]), 2)

    def test_a_retained_horizon_cannot_be_replaced_by_another_b_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, panel, _ = self._miner(directory, [1])
            def no_support(*args, **kwargs):
                report = self._controlled_evaluation(*args, **kwargs)
                report["summary"]["rank_ic"]["p_value"] = .8
                return report
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=no_support):
                result = miner.validate(lambda: panel, Path(directory) / "ideas")
            verdict = result["decisions"]["candidate-0001"]
            self.assertEqual(verdict["passed_horizons"], [])
            self.assertFalse(verdict["eligible_for_idea_pool"])
            self.assertEqual([test["horizon_hours"] for test in verdict["tests"]], [1])
            self.assertFalse((Path(directory) / "ideas").exists())

    def test_changed_frozen_horizons_fail_before_the_b_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, panel, _ = self._miner(directory, [1, 4])
            path = miner.root / "frozen_batch.json"
            frozen = json.loads(path.read_text())
            frozen["candidates"]["candidate-0001"]["a_decision"]["retained_horizons"] = [1, 4, 24]
            path.write_text(json.dumps(frozen))
            loader = Mock(return_value=panel)
            with self.assertRaisesRegex(ValueError, "frozen.*decision|retained.*horizon"):
                miner.validate(loader, Path(directory) / "ideas")
            loader.assert_not_called()

    def test_a_retention_needs_complete_evidence_but_not_significance(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, _, _ = self._miner(directory, [24])
            records = copy.deepcopy(miner.store.all())
            evidence = next(r["data"] for r in records if r["id"] == "candidate-0001-evaluation")
            for horizon in (1, 4, 24):
                report = evidence["horizon_comparison"]["horizons"][str(horizon)]
                report["summary"]["rank_ic"].update(mean=0.0, p_value=1.0)
            decision = research_decision("candidate-0001", "retain")
            decision["retained_horizons"] = [1, 4, 24]
            with patch.object(miner.store, "all", return_value=records):
                self.assertEqual(miner._check_decisions([decision], ["candidate-0001"]),
                                 {"candidate-0001": decision})
                evidence["horizon_comparison"]["horizons"].pop("4")
                with self.assertRaisesRegex(ValueError, "4h|horizon"):
                    miner._check_decisions([decision], ["candidate-0001"])

    def test_a_retained_horizons_reject_duplicate_unknown_and_empty_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, _, _ = self._miner(directory, [24])
            decision = research_decision("candidate-0001", "retain")
            for horizons in ([], [1, 1], [48], [True], [1.0]):
                with self.subTest(horizons=horizons):
                    decision["retained_horizons"] = horizons
                    with self.assertRaisesRegex(ValueError, "horizon"):
                        miner._check_decisions([decision], ["candidate-0001"])

    def test_goal_counts_one_card_and_keeps_b_evidence_out_of_model_context(self):
        class MultiGoalModel(GoalModel):
            def complete(self, messages, *, max_output_tokens, session_id):
                reply = super().complete(messages, max_output_tokens=max_output_tokens, session_id=session_id)
                result = json.loads(reply.text)
                request = json.loads(messages[1]["content"])
                if request["role"] == "optimizer" and "decisions" in result["result"]:
                    for decision in result["result"]["decisions"]:
                        if decision["disposition"] == "retain":
                            # Arbitrary unique ordering is allowed by the A contract.
                            decision["retained_horizons"] = [24, 1, 4]
                reply.text = dumps(result)
                return reply
        with tempfile.TemporaryDirectory() as directory:
            spec, model = specification(purpose="research"), MultiGoalModel()
            runner = GoalRunner.create(GoalSpec("horizons-goal", "研究价格偏离关系的合格成果", 1),
                                       spec, model, Path(directory), inputs={}, model_settings=SETTINGS)
            a, b = input_panel(spec), input_panel(spec, "B")
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor",
                       side_effect=self._controlled_evaluation):
                state = runner.run(lambda: a, lambda: b, lambda: b.universe, runner.root / "ideas",
                                   poll_seconds=1, sleep=lambda _: None)
            self.assertEqual(state["status"], "complete")
            self.assertEqual(len(state["qualified_ideas"]), 1)
            self.assertEqual(len(list((runner.root / "ideas").glob("*.json"))), 1)
            deliveries = list((runner.root / "delivery_evidence").glob("admission-*.json"))
            self.assertEqual(len(deliveries), 1)
            delivery = json.loads(deliveries[0].read_text())["ideas"][0]
            self.assertEqual(delivery["passed_horizons"], [24, 1, 4])
            self.assertEqual(set(delivery["horizon_evidence"]), {"1", "4", "24"})
            for request in model.requests:
                if request["role"] == "optimizer":
                    encoded = dumps(request)
                    self.assertNotIn('"passed_horizons"', encoded)
                    self.assertNotIn('"horizon_results"', encoded)
                    self.assertNotIn('"segment":"B"', encoded)
                    self.assertNotIn('"batch_correction"', encoded)
            miner = FactorMiner.open(runner.root / "runs" / state["current_run"], model)
            validation = json.loads((miner.root / "validation.json").read_text())
            calls = len(model.requests)
            runner._review_completion(miner, validation)
            self.assertEqual(len(model.requests), calls)
            self.assertEqual(runner.state["qualified_ideas"], state["qualified_ideas"])
            card_path = Path(validation["decisions"]["candidate-0001"]["idea_card"])
            card = json.loads(card_path.read_text())
            card["source"]["candidate_id"] = "candidate-9999"
            card_path.write_text(dumps(card))
            with self.assertRaisesRegex(ValueError, "source.*formula|candidate"):
                runner._review_completion(miner, validation)


if __name__ == "__main__":
    unittest.main()
