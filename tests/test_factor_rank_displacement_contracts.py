"""Strict contracts for the controlled rank-displacement experiment route."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from crypto_quant.research.factor_mining.agent_optimizer import EXPERIMENT_DESIGN_SCHEMA, OptimizerRole
from crypto_quant.research.factor_mining.contracts import ResearchSpec, model_fields, modification_plan
from crypto_quant.research.factor_mining.goal import GoalResearchStore, GoalRunner
from crypto_quant.research.factor_mining.workflow import FactorMiner


BASE_DESIGN = {
    "question": "Does the controlled change reduce ranking displacement while preserving prediction?",
    "min_improvement": 0.02,
    "max_ic_loss": 0.01,
    "expected_outcome": "Lower D with acceptable directed IC loss",
    "stop_condition": "Stop if either declared bound cannot be met",
    "pause_condition": "Pause if paired evidence is insufficient",
}


def plan_for(design):
    return {
        "route_id": "displacement-route",
        "control_id": "candidate-0001",
        "evidence_refs": ["candidate-0001-evaluation"],
        "modification_task": {
            "core_hypothesis": "Relative price deviation predicts reversal",
            "observed_problem": "A evidence shows ranking changes across adjacent observations",
            "modification_hypothesis": "A bounded temporal transform may stabilize ranks",
            "change_target": "Apply one evidence-backed temporal transform to the relative deviation",
            "fixed_components": "Keep input fields, direction and prediction horizon fixed",
        },
        "experiment_design": design,
        "restart_of": None,
        "new_evidence": None,
    }


def goal_research_spec():
    return ResearchSpec(
        run_id="goal-run-1", objective="A-only context fixture", purpose="engineering_check",
        a_start="2026-07-01T00:00:00Z", b_start="2026-07-05T00:00:00Z",
        c_start="2026-07-09T00:00:00Z", c_end="2026-07-13T00:00:00Z",
        universe_provenance="synthetic context fixture", data_usage_review="synthetic only",
        label="perp_next_open_24h", sample_hours=1, groups=3, min_symbols=6,
        min_periods=30, hac_lags=23, confidence=0.95, stage_hours=48, rolling_periods=26,
        fdr_method="BH", fdr_alpha=0.05, min_abs_ic=0.02, min_directional_spread=0.0001,
        min_stage_share=0.5, max_repairs=2, max_formula_nodes=80, max_lookback_hours=7,
        context_tokens=2000000, output_tokens=4000, b_horizons=(1, 4, 24),
        admission_scheme="plan3", plan3_tracks_gate=False)


def horizon_evidence(horizon):
    inference = {"mean": 0.03, "std": 0.01, "mean_std_ratio": 3.0, "n": 12,
                "grid_periods": 12, "method": "HAC", "lags": 23, "confidence": 0.95,
                "alternative": "two-sided", "se": 0.01, "ci": [0.01, 0.05],
                "p_value": 0.01, "status": "computed"}
    return {"segment": "A", "horizon_hours": horizon, "direction": 1,
            "summary": {"rank_ic": inference, "directional_spread": inference,
                        "raw_high_low_spread_mean": 0.001, "ic_direction_share": 0.8,
                        "spread_direction_share": 0.7, "group_means": [0, 1, 2],
                        "positive_stage_share": 0.6, "valid_stages": 12},
            "coverage": {"eligible_observations": 12, "purged_observations": 0,
                         "purged_hours": 0, "status_counts": {}},
            "periods": [], "stages": [], "per_symbol": []}


def rank_displacement_evidence():
    delta = {"summary": {"mean": 0.1, "median": 0.1, "p90": 0.2,
                          "valid_periods": 12, "expected_periods": 12,
                          "valid_period_share": 1.0},
             "coverage": {"common_symbols": 72}, "periods": [], "stages": [{"mean": 0.1}]}
    return {"definition_version": "factor-rank-displacement-v1", "segment": "A",
            "deltas": {key: copy.deepcopy(delta) for key in ("1", "4", "24")}}


class RankDisplacementContractTests(unittest.TestCase):
    def test_existing_experiment_contracts_keep_their_original_fields(self):
        for metric in ("rank_ic", "directional_spread"):
            design = {**BASE_DESIGN, "metric": metric}
            self.assertEqual(modification_plan(plan_for(design))["experiment_design"], design)

        with self.assertRaisesRegex(ValueError, "fields do not match"):
            modification_plan(plan_for({**BASE_DESIGN, "metric": "rank_ic", "horizon_hours": 24,
                                        "displacement_hours": 4}))

    def test_rank_displacement_requires_explicit_supported_horizon_and_interval(self):
        for horizon in (1, 4, 24):
            for displacement in (1, 4, 24):
                design = {**BASE_DESIGN, "metric": "rank_displacement",
                          "horizon_hours": horizon, "displacement_hours": displacement}
                self.assertEqual(modification_plan(plan_for(design))["experiment_design"], design)

        base = {**BASE_DESIGN, "metric": "rank_displacement", "displacement_hours": 4}
        with self.assertRaisesRegex(ValueError, "fields do not match"):
            modification_plan(plan_for(base))
        base = {**BASE_DESIGN, "metric": "rank_displacement", "horizon_hours": 24}
        with self.assertRaisesRegex(ValueError, "fields do not match"):
            modification_plan(plan_for(base))

        for field in ("horizon_hours", "displacement_hours"):
            for value in (0, 2, 6, "24", True):
                design = {**BASE_DESIGN, "metric": "rank_displacement",
                          "horizon_hours": 24, "displacement_hours": 4}
                design[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    modification_plan(plan_for(design))

    def test_model_schema_selects_rank_displacement_branch_and_requires_both_fields(self):
        legacy = {**BASE_DESIGN, "metric": "rank_ic"}
        self.assertEqual(model_fields(copy.deepcopy(legacy), EXPERIMENT_DESIGN_SCHEMA, []), legacy)

        design = {**BASE_DESIGN, "metric": "rank_displacement",
                  "horizon_hours": 4, "displacement_hours": 24}
        extras = []
        parsed = model_fields(copy.deepcopy(design), EXPERIMENT_DESIGN_SCHEMA, extras)
        self.assertEqual(parsed, design)
        self.assertEqual(extras, [])

        for missing in ("horizon_hours", "displacement_hours"):
            incomplete = dict(design)
            del incomplete[missing]
            extras = []
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                model_fields(incomplete, EXPERIMENT_DESIGN_SCHEMA, extras)
            self.assertEqual(extras, [])

        invalid = {**design, "displacement_hours": 12}
        with self.assertRaises(ValueError):
            model_fields(invalid, EXPERIMENT_DESIGN_SCHEMA, [])

    def test_unrecognized_model_fields_are_still_recorded_for_the_matching_branch(self):
        design = {**BASE_DESIGN, "metric": "rank_displacement",
                  "horizon_hours": 1, "displacement_hours": 4, "unrequested": "value"}
        extras = []
        parsed = model_fields(design, EXPERIMENT_DESIGN_SCHEMA, extras)
        self.assertNotIn("unrequested", parsed)
        self.assertEqual(extras, ["/unrequested"])

    def test_invalid_union_payload_still_logs_unrecognized_fields(self):
        schema = {"anyOf": [
            {"type": "object", "properties": {"analysis": {"type": "string"}},
             "required": ["analysis"], "additionalProperties": False},
            {"type": "null"},
        ]}
        extras = []
        with self.assertRaises(ValueError):
            model_fields({"analysis": None, "unknown_field": True}, schema, extras)
        self.assertEqual(extras, ["/unknown_field"])

    def test_optimizer_response_cannot_supply_or_override_program_route_state(self):
        role = object.__new__(OptimizerRole)
        role._check_decisions = lambda decisions, review_ids: {}
        result = {"analysis": "review", "decisions": [], "diagnostics": [], "proposals": [],
                  "route_states": {"displacement-route": {"decision": "continue"}}}
        with self.assertRaisesRegex(ValueError, "invalid optimizer schema"):
            role._check_optimization(result, [])

    def test_rank_displacement_route_requires_selected_a_horizon_and_d_evidence(self):
        design = {**BASE_DESIGN, "metric": "rank_displacement",
                  "horizon_hours": 4, "displacement_hours": 24}
        role = object.__new__(OptimizerRole)
        role.candidates = {"candidate-0001": {
            "definition": {"direction": 1},
            "evaluation": {"segment": "A", "horizon_comparison": {
                "horizons": {"4": horizon_evidence(4)}}},
            "rank_displacement": rank_displacement_evidence(),
        }}
        role.store = Mock()
        role.store.all.return_value = [{"id": "candidate-0001-evaluation", "kind": "evaluation",
                                       "data": role.candidates["candidate-0001"]["evaluation"]}]
        role.routes = {}
        role._register_route(plan_for(design), role.routes)
        self.assertEqual(role.routes["displacement-route"]["decision"], "continue")

        role.candidates["candidate-0001"].pop("rank_displacement")
        with self.assertRaisesRegex(ValueError, "complete A displacement diagnostics"):
            role._check_rank_displacement_evidence(role.candidates["candidate-0001"], design)

        role.candidates["candidate-0001"]["rank_displacement"] = rank_displacement_evidence()
        role.candidates["candidate-0001"]["evaluation"]["horizon_comparison"]["horizons"].clear()
        with self.assertRaisesRegex(ValueError, "matching A horizon evidence"):
            role._check_rank_displacement_evidence(role.candidates["candidate-0001"], design)

    def test_goal_cycle_keeps_only_a_displacement_summary_and_marks_missing_evidence(self):
        base = {"run_id": "run-1", "records": [
            {"id": "candidate-0001", "kind": "candidate", "data": {
                "id": "candidate-0001", "definition": {"expression": "perp_close", "direction": 1}}},
            {"id": "candidate-0001-evaluation", "kind": "evaluation", "data": {
                "segment": "A", "summary": {"rank_ic": {"mean": 0.03}}, "coverage": {"valid": 10},
                "rank_displacement": {
                    "definition_version": "factor-rank-displacement-v1", "segment": "A",
                    "deltas": {delta: {"summary": {"mean": 0.1}, "coverage": {"valid": 8},
                                      "periods": [{"large": "table"}]}
                               for delta in ("1", "4", "24")},
                },
            }},
        ]}
        context = GoalRunner._cycle_context("cycle-00000001", base)
        displacement = context["candidates"][0]["A_evaluation"]["rank_displacement"]
        self.assertEqual(displacement["status"], "available")
        self.assertEqual(displacement["definition_version"], "factor-rank-displacement-v1")
        self.assertEqual(set(displacement["deltas"]), {"1", "4", "24"})
        self.assertNotIn("periods", displacement["deltas"]["1"])

        del base["records"][1]["data"]["rank_displacement"]
        context = GoalRunner._cycle_context("cycle-00000002", base)
        self.assertEqual(context["candidates"][0]["A_evaluation"]["rank_displacement"],
                         {"status": "not_calculated"})

    def test_goal_cycle_refuses_non_a_evaluation_evidence(self):
        data = {"run_id": "run-1", "records": [
            {"id": "candidate-0001", "kind": "candidate", "data": {
                "id": "candidate-0001", "definition": {"expression": "perp_close", "direction": 1}}},
            {"id": "candidate-0001-evaluation", "kind": "evaluation", "data": {
                "segment": "B", "summary": {}, "coverage": {},
                "rank_displacement": {"segment": "B"}}},
        ]}
        with self.assertRaisesRegex(ValueError, "A evaluation evidence only"):
            GoalRunner._cycle_context("cycle-00000001", data)

    def test_goal_archive_context_keeps_compact_summary_and_retrieves_archived_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            goal_root = output / "goal-1"
            goal_root.mkdir()
            miner = FactorMiner(goal_research_spec(), Mock(), goal_root / "runs")
            identity = miner._factor_identity("perp_close", 1)
            periods = [{"timestamp": f"2026-07-01T0{hour}:00:00Z", "displacement": 0.1,
                        "status": "valid"} for hour in range(3)]
            displacement = {
                "definition_version": "factor-rank-displacement-v1", "segment": "A",
                "deltas": {delta: {"summary": {"mean": 0.1, "valid_periods": 3},
                                   "coverage": {"eligible": 6}, "periods": periods,
                                   "stages": [{"mean": 0.1}]}
                           for delta in ("1", "4", "24")},
            }
            locator = miner._write_rank_displacement(
                "candidate-0001", "A", displacement, identity, source={"fixture": True})
            miner.store.append("candidate-0001", "candidate", {
                "id": "candidate-0001", "definition": {"expression": "perp_close", "direction": 1}})
            report = {"candidate_id": "candidate-0001", "segment": "A", "direction": 1,
                      "horizon_hours": 24, "summary": {"rank_ic": {"mean": 0.03}},
                      "coverage": {"valid": 10}}
            prediction_locator = miner._write_evaluation(
                "candidate-0001", "A", report, identity)
            miner.store.append("candidate-0001-evaluation", "evaluation",
                               FactorMiner._evaluation_record_data(
                                   "candidate-0001", report, prediction_locator,
                                   displacement, locator))

            runner = object.__new__(GoalRunner)
            runner.root = goal_root
            runner.state = {"cycle": 1}
            runner.research = GoalResearchStore(goal_root / "research_records")
            runner._archive_A(miner)

            cycle = runner.research.all()[0]
            candidate_context = cycle["data"]["context"]["candidates"][0]
            self.assertEqual(candidate_context["A_evaluation"]["rank_displacement"]["status"], "available")
            self.assertNotIn("periods", candidate_context["A_evaluation"]["rank_displacement"]["deltas"]["1"])
            archived = runner.research.source_records("cycle-00000001")
            evaluation = next(record for record in archived if record["kind"] == "evaluation")
            self.assertEqual(evaluation["data"]["rank_displacement"]["deltas"]["1"]["periods"], periods)


if __name__ == "__main__":
    unittest.main()
