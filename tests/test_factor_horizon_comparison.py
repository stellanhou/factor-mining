"""Synthetic integration checks for the isolated fixed-sample comparison tool."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from test_factor_goal import GoalModel, SETTINGS
from test_factor_mining import ScenarioModel, definition, input_panel, research_decision, specification
from crypto_quant.research.factor_mining.contracts import dumps
from crypto_quant.research.factor_mining.model import ModelReply
from crypto_quant.research.factor_mining.workflow import FactorMiner


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "compare_fm_v6_horizons.py"
sys.path.insert(0, str(ROOT / "scripts"))
import compare_fm_v6_horizons as comparison


def _source_a(root: Path) -> dict:
    source_goal = root / "source_goal"
    source_goal.mkdir()
    source_runs = source_goal / "runs"
    db_path = root / "market.sqlite3"
    db_path.write_bytes(b"synthetic source identity only")
    universe_path = root / "universe.csv"
    universe_path.write_text("timestamp,symbol,eligible\n", encoding="utf-8")
    original_pool = root / "source_idea_pool"
    original_pool.mkdir()
    spec = specification(
        purpose="research", run_id="source-run-1", context_tokens=1_000_000,
        output_tokens=None,
    )

    def transform(request, result):
        if request["role"] == "ideator":
            result["candidates"] = [definition("div(perp_close,ts_mean(perp_close,3))")]
            result["dispositions"] = []
        elif request["role"] == "optimizer":
            result["decisions"] = [research_decision("candidate-0001", "retain")]

    miner = FactorMiner(spec, ScenarioModel(transform=transform, propose_once=False), source_runs)
    miner.explore(input_panel(spec))
    b_panel = input_panel(spec, "B")
    miner.freeze(["candidate-0001"], b_panel.universe)
    run_dir = miner.root
    frozen = json.loads((run_dir / "frozen_batch.json").read_text(encoding="utf-8"))
    item = frozen["candidates"]["candidate-0001"]
    decision = json.loads((run_dir / "a-complete.json").read_text(encoding="utf-8"))["candidate_decisions"]["candidate-0001"]
    identity = item["a_evaluation_archive"]["identity"]
    key = comparison.factor_key(identity)
    archive_root = (run_dir / json.loads((run_dir / "factor_archive.json").read_text())["archive_root"]).resolve()
    archive_path = comparison.discover_factor_archives(archive_root, {key: identity})[key]
    records = {}
    for suffix in ("definition", "calculation", "evaluation", "report"):
        record = json.loads((run_dir / "a_records" / f"candidate-0001-{suffix}.json").read_text())
        records[suffix] = record["id"]

    candidate = {
        "factor_key": key,
        "candidate_id": "candidate-0001",
        "source_run_id": spec.run_id,
        "batch_index": 1,
        "definition": item["definition"],
        "executed": item["executed"],
        "identity": identity,
        "source_decision": decision,
        "a_evaluation_ref": item["a_evaluation_ref"],
        "a_evaluation_archive": item["a_evaluation_archive"],
        "a_model_report": item["a_model_report"],
        "source_record_ids": records,
        "source_archive": str(archive_path),
        "source_archive_signature": comparison.file_signature(archive_path),
        "source_evaluation_id": str(item["a_evaluation_archive"]["evaluation_id"]),
    }
    scope = {
        "schema": "fm-v6-fixed-horizon-comparison/v1",
        "source_goal": str(source_goal.resolve()),
        "source_goal_id": "synthetic-source-goal",
        "source_goal_data": {
            "goal": {"goal_id": "synthetic-source-goal", "objective": "fixture", "target_ideas": 1},
            "research": spec.as_dict(),
            "model_settings": SETTINGS,
            "inputs": {
                "db": str(db_path.resolve()), "universe": str(universe_path.resolve()),
                "idea_pool": str(original_pool.resolve()), "include_liquidations": False,
            },
        },
        "research_spec": spec.as_dict(),
        "input_signatures": {
            "db": comparison.file_signature(db_path),
            "universe": comparison.file_signature(universe_path, hash_contents=True),
        },
        "batch_count": 1,
        "candidate_count": 1,
        "original_card_count": 0,
        "original_cards": [],
        "batches": [{
            "batch_index": 1, "source_run_id": spec.run_id,
            "candidate_count": 1, "factor_keys": [key], "candidates": [candidate],
        }],
        "candidates": [candidate],
    }
    return {"scope": scope, "candidate": candidate, "source_miner": miner,
            "b_panel": b_panel, "db_path": db_path, "universe_path": universe_path}


def _write_prepared_inputs(root: Path, fixture: dict) -> tuple[Path, dict, dict]:
    experiment = root / "experiment"
    experiment.mkdir()
    scope = fixture["scope"]
    comparison.write_new_json(experiment / "fixed_scope.json", scope)
    candidate = fixture["candidate"]
    review = {
        "schema": "fm-v6-a-horizon-review/v1",
        "source_goal_id": scope["source_goal_id"],
        "fixed_scope_sha256": hashlib.sha256(comparison.canonical_json(scope)).hexdigest(),
        "model": {"settings": SETTINGS, "output_tokens": None},
        "candidate_count": 1,
        "allowed_horizons": [1, 4, 24],
        "decisions": [{
            "factor_key": candidate["factor_key"],
            "source_run_id": candidate["source_run_id"],
            "candidate_id": candidate["candidate_id"],
            "retained_horizons": [24],
            "reason": "A-only fixture review keeps the original 24h horizon.",
            "preserved_original_retain_reason": candidate["source_decision"]["reason"],
        }],
    }
    comparison.write_new_json(experiment / "a_review" / "a_review.json", review)
    frozen_root = experiment / "frozen_inputs"
    frozen_root.mkdir()
    panel = fixture["b_panel"]
    values_path = frozen_root / "values.parquet"
    universe_path = frozen_root / "universe.parquet"
    panel.values.to_parquet(values_path)
    panel.universe.rename("eligible").to_frame().to_parquet(universe_path)
    db_sig = scope["input_signatures"]["db"]
    universe_sig = scope["input_signatures"]["universe"]
    manifest = {
        "schema": "fm-v6-frozen-b-inputs/v1",
        "source_db_before": db_sig, "source_db_after": db_sig,
        "source_universe_before": universe_sig, "source_universe_after": universe_sig,
        "data_usage": {"fixture": "synthetic"},
        "rows": len(panel.values), "columns": list(panel.values.columns),
        "artifacts": {
            "values": comparison.file_signature(values_path, hash_contents=True),
            "universe": comparison.file_signature(universe_path, hash_contents=True),
        },
    }
    comparison.write_new_json(frozen_root / "provenance.json", manifest)
    return experiment, review, manifest


class HorizonComparisonToolTests(unittest.TestCase):
    def test_fixed_a_csv_round_trip_preserves_real_ulp_pair_and_rank_displacement(self):
        # Read-only rows from preregistration.json E2.parents[0],
        # experiments/factor_revalidation/historical_fm_v6_rerun_20260930/runs/
        # factor_archive_v2/factor-000153.sqlite3, evaluation_id=1, value_set_id=1.
        source_pair_csv = (
            b"timestamp,symbol,factor_value\n"
            b"2024-06-11 10:00:00+00:00,BNBUSDT,0.17857142857142858\n"
            b"2024-06-17 15:00:00+00:00,DOGEUSDT,0.17857142857142855\n"
        )
        exact_pair = comparison.read_fixed_a_factor_values_csv(source_pair_csv, 2)
        high = float(exact_pair.iloc[0])
        low = float(exact_pair.iloc[1])
        self.assertEqual(high.hex(), "0x1.6db6db6db6db7p-3")
        self.assertEqual(low.hex(), "0x1.6db6db6db6db6p-3")
        self.assertNotEqual(high, low)

        default_pair = pd.read_csv(io.BytesIO(source_pair_csv))
        self.assertEqual(default_pair["factor_value"].iloc[0], default_pair["factor_value"].iloc[1])

        spec = specification()
        start, end = spec.bounds("A")
        times = pd.date_range(start, end, freq="h", inclusive="left")
        symbols = ["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT"]
        base = dict(zip(symbols, (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)))
        rows = []
        for timestamp in times:
            current = dict(base)
            if timestamp == times[1]:
                current["BNBUSDT"] = high
                current["DOGEUSDT"] = low
            rows.extend(current[symbol] for symbol in symbols)
        index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
        original_values = pd.Series(rows, index=index, name="factor_value", dtype=float)
        values_csv = original_values.to_csv().encode("utf-8")

        imported_values = comparison.read_fixed_a_factor_values_csv(values_csv, len(original_values))
        default_rows = pd.read_csv(io.BytesIO(values_csv))
        default_values = pd.Series(default_rows["factor_value"].to_numpy(dtype=float),
                                   index=index, name="factor_value")
        self.assertTrue(imported_values.equals(original_values))
        self.assertEqual(default_values.loc[(times[1], "BNBUSDT")],
                         default_values.loc[(times[1], "DOGEUSDT")])
        self.assertNotEqual(imported_values.loc[(times[1], "BNBUSDT")],
                            imported_values.loc[(times[1], "DOGEUSDT")])

        universe = pd.Series(True, index=index, name="eligible")
        from crypto_quant.research.factor_mining.evaluation import evaluate_rank_displacement
        original_d = evaluate_rank_displacement(original_values, universe, spec, "A")
        imported_d = evaluate_rank_displacement(imported_values, universe, spec, "A")
        default_d = evaluate_rank_displacement(default_values, universe, spec, "A")
        self.assertEqual(imported_d, original_d)
        exact_period = imported_d["deltas"]["1"]["periods"][1]
        default_period = default_d["deltas"]["1"]["periods"][1]
        self.assertFalse(exact_period["has_ties_t"])
        self.assertEqual(exact_period["displacement"], 4 / 36)
        self.assertTrue(default_period["has_ties_t"])
        self.assertEqual(default_period["displacement"], 3 / 36)

    def test_run_worker_imports_fixed_a_then_writes_goal_scoped_card_and_receipts(self):
        class MultiGoalModel(GoalModel):
            def complete(self, messages, *, max_output_tokens, session_id):
                reply = super().complete(messages, max_output_tokens=max_output_tokens, session_id=session_id)
                request = json.loads(messages[1]["content"])
                response = json.loads(reply.text)
                if request["role"] == "optimizer" and "decisions" in response["result"]:
                    for decision in response["result"]["decisions"]:
                        if decision["disposition"] == "retain":
                            decision["retained_horizons"] = [24]
                    reply.text = dumps(response)
                return reply

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _source_a(root)
            experiment, _, frozen_manifest = _write_prepared_inputs(root, fixture)
            group_dir = experiment / "new_multi"
            group_dir.mkdir()
            model = MultiGoalModel()
            args = Namespace(
                group="new_multi", engine_root=ROOT / "src", experiment_root=experiment,
                group_dir=group_dir, frozen_panel_manifest=experiment / "frozen_inputs" / "provenance.json",
                resume=False,
            )
            with patch.object(comparison, "EXPECTED_FACTORS", 1), \
                    patch.object(comparison, "EXPECTED_BATCHES", 1), \
                    patch.object(comparison, "EXPECTED_OLD_CARDS", 1), \
                    patch("crypto_quant.research.factor_mining.model_config.model_from_settings",
                          return_value=model), \
                    patch("crypto_quant.research.factor_mining.workflow.evaluate_factor",
                          side_effect=self._passing_b_evaluation):
                comparison.run_group_worker(args)

            group_result = json.loads((group_dir / "fixed_group_result.json").read_text())
            goal_root = Path(group_result["goal_root"])
            run_id = group_result["runs"][0]
            run_dir = goal_root / "runs" / run_id
            self.assertTrue(run_dir.is_relative_to(goal_root / "runs"))
            self.assertFalse((group_dir / "goal" / "runs").exists())
            frozen = json.loads((run_dir / "frozen_batch.json").read_text())
            frozen_candidate = frozen["candidates"]["candidate-0001"]
            self.assertEqual(frozen_candidate["rank_displacement_version"],
                             "factor-rank-displacement-v1")
            self.assertEqual(frozen_candidate["a_rank_displacement"]["definition_version"],
                             "factor-rank-displacement-v1")
            imported_evaluation = json.loads(
                (run_dir / "a_records" / "candidate-0001-evaluation.json").read_text())["data"]
            self.assertEqual(imported_evaluation["rank_displacement"]["definition_version"],
                             "factor-rank-displacement-v1")
            self.assertNotIn("periods", imported_evaluation["rank_displacement"]["deltas"]["1"])
            source_evaluation = fixture["source_miner"].store._load(
                fixture["source_miner"].store.root / "candidate-0001-evaluation.json")
            source_d_locator = source_evaluation["data"]["rank_displacement_archive"]
            self.assertEqual(imported_evaluation["rank_displacement_archive"]["evaluation_key"]["data_version"],
                             f"{run_id}/A")
            self.assertEqual(source_d_locator["evaluation_key"]["data_version"],
                             f"{fixture['candidate']['source_run_id']}/A")
            from crypto_quant.research.factor_mining.factor_archive import EvaluationKey, FactorArchive, FactorIdentity
            marker = json.loads((run_dir / "factor_archive.json").read_text())
            new_archive_root = (run_dir / marker["archive_root"]).resolve()
            d_locator = imported_evaluation["rank_displacement_archive"]
            d_archive = FactorArchive.open_existing(new_archive_root, FactorIdentity(**d_locator["identity"]))
            d_evidence = d_archive.get_evaluation(EvaluationKey(**d_locator["evaluation_key"]))
            source = d_evidence["provenance"][0]["source"]
            self.assertEqual(source["source_run_id"], fixture["candidate"]["source_run_id"])
            self.assertEqual(source["source_evaluation_id"],
                             fixture["candidate"]["source_evaluation_id"])
            self.assertEqual(d_evidence["payload"]["deltas"],
                             source_evaluation["data"]["rank_displacement"]["deltas"])
            goal_data = json.loads((goal_root / "goal.json").read_text())
            self.assertEqual(goal_data["goal"]["objective"],
                             fixture["scope"]["source_goal_data"]["goal"]["objective"])
            self.assertEqual(goal_data["model_settings"], SETTINGS)
            self.assertEqual(goal_data["inputs"]["frozen_b_values_sha256"],
                             frozen_manifest["artifacts"]["values"]["sha256"])
            self.assertTrue((run_dir / "b-numerical-complete.json").is_file())
            self.assertTrue((run_dir / "B-report.md").is_file())
            cards = list((group_dir / "idea_pool").glob("*.json"))
            self.assertEqual(len(cards), 1)
            self.assertEqual(cards[0].stem, f"{run_id}--candidate-0001")
            admissions = [json.loads(path.read_text()) for path in
                          (goal_root / "completion_records").glob("admission-*.json")]
            matches = [json.loads(path.read_text()) for path in
                       (goal_root / "completion_records").glob("match-*.json")]
            self.assertEqual(len(admissions), 1)
            self.assertEqual(admissions[0]["kind"], "program_admission")
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["kind"], "goal_match")
            delivery = list((goal_root / "delivery_evidence").glob("admission-*.json"))
            self.assertEqual(len(delivery), 1)

            goal_path = goal_root / "goal.json"
            saved_goal = json.loads(goal_path.read_text())
            broken_objective = "只核验固定批次运行是否达到程序卡数"
            saved_goal["goal"]["objective"] = broken_objective
            goal_path.write_text(dumps(saved_goal))
            old_match_path = next((goal_root / "completion_records").glob("match-*.json"))
            old_match = json.loads(old_match_path.read_text())
            old_match["data"]["matches"][0]["matches_goal"] = False
            old_match_path.write_text(dumps(old_match))
            old_match_bytes = old_match_path.read_bytes()
            recheck_root = goal_root / "completion_rechecks" / "source-objective-recheck-v1"
            recheck_args = Namespace(
                group="new_multi", experiment_root=experiment,
                engine_root=ROOT / "src", goal_root=goal_root, recheck_root=recheck_root,
            )
            with patch.object(comparison, "EXPECTED_FACTORS", 1), \
                    patch.object(comparison, "EXPECTED_BATCHES", 1), \
                    patch.object(comparison, "EXPECTED_OLD_CARDS", 1), \
                    patch("crypto_quant.research.factor_mining.model_config.model_from_settings",
                       return_value=model), \
                    patch("crypto_quant.research.factor_mining.workflow.FactorMiner.validate",
                          side_effect=AssertionError("B must not be validated again")), \
                    patch("crypto_quant.research.factor_mining.workflow.evaluate_factor",
                          side_effect=AssertionError("B evaluation must not run again")):
                comparison.recheck_completion_worker(recheck_args)
            corrected = json.loads((recheck_root / "recheck_manifest.json").read_text())
            result = json.loads((recheck_root / "recheck_result.json").read_text())
            self.assertEqual(corrected["previous_goal_objective"], broken_objective)
            self.assertEqual(corrected["corrected_goal"]["objective"],
                             fixture["scope"]["source_goal_data"]["goal"]["objective"])
            self.assertEqual(goal_path.read_bytes(), dumps(saved_goal).encode("utf-8"))
            self.assertEqual(old_match_path.read_bytes(), old_match_bytes)
            self.assertEqual(result["qualified_count"], 1)
            recheck_match = next((recheck_root / "completion_records").glob("match-*.json"))
            self.assertTrue(json.loads(recheck_match.read_text())["data"]["matches"][0]["matches_goal"])
            match_requests = [request for request in model.requests
                              if request.get("payload", {}).get("goal_phase") == "verify_completion"]
            self.assertTrue(match_requests)
            self.assertEqual(match_requests[-1]["payload"]["goal"]["objective"],
                             fixture["scope"]["source_goal_data"]["goal"]["objective"])

    def test_fixed_a_import_keeps_legacy_engine_without_rank_displacement(self):
        class LegacyStore:
            def __init__(self):
                self.records = {}

            def append(self, record_id, kind, data):
                self.records[record_id] = {"id": record_id, "kind": kind, "data": data}

        class LegacyMiner:
            def __init__(self, root):
                self.root = root
                self.root.mkdir()
                self.store = LegacyStore()
                self.frozen = None

            @staticmethod
            def _factor_identity(expression, direction):
                from crypto_quant.research.factor_mining.factor_archive import FactorIdentity
                return FactorIdentity(expression, str(direction), "expr-v1")

            def _write_evaluation(self, candidate_id, stage, report, identity, *, definition):
                return {"legacy_archive_locator": candidate_id, "stage": stage}

            @staticmethod
            def _evaluation_record_data(candidate_id, report, locator):
                return {"candidate_id": candidate_id, "segment": report["segment"],
                        "summary": report["summary"], "factor_archive": locator}

            def freeze(self, candidate_ids, b_membership):
                self.frozen = (candidate_ids, b_membership)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _source_a(root)
            miner = LegacyMiner(root / "legacy_run")

            def write_json(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(dumps(value), encoding="utf-8")

            membership = object()
            imported = comparison.import_fixed_a_batch(
                miner, fixture["scope"], fixture["scope"]["batches"][0], {}, True,
                {"provider": "legacy-fixture"}, membership, write_json,
            )
            record = miner.store.records["candidate-0001-evaluation"]
            self.assertEqual(imported, ["candidate-0001"])
            self.assertNotIn("rank_displacement", record["data"])
            self.assertNotIn("rank_displacement_archive", record["data"])
            self.assertEqual((miner.root / "A-universe.csv").read_bytes(),
                             (Path(fixture["scope"]["source_goal"]) / "runs" /
                              fixture["candidate"]["source_run_id"] / "A-universe.csv").read_bytes())
            self.assertEqual(miner.frozen, (imported, membership))

    def test_a_review_pages_only_a_evidence_with_budget_below_one_mib(self):
        class PagedReviewModel:
            def __init__(self):
                self.requests = []
                self.pages = []
                self.page_evidence = None

            def complete(self, messages, *, max_output_tokens, session_id):
                initial = json.loads(messages[1]["content"])
                latest = json.loads(messages[-1]["content"])
                self.requests.append(initial)
                if "requested_original_evidence" in latest:
                    self.page_evidence = latest["requested_original_evidence"]
                    candidate = initial["payload"]["candidate_evidence_order"][0]
                    envelope = {"result": {"decisions": [{
                        "factor_key": candidate["factor_key"],
                        "candidate_id": candidate["candidate_id"],
                        "retained_horizons": [24],
                        "reason": "A共同样本提供原24h以外期限的追加依据；保留24h并说明不确定性。",
                    }]}, "read_records": []}
                else:
                    record_id = initial["records"][0]["id"]
                    self.pages.append({
                        "record_id": record_id,
                        "pointer": "/a_evidence/horizon_comparison/horizons/1/periods",
                        "offset": 0, "limit": 2,
                    })
                    envelope = {"result": None, "read_records": [self.pages[-1]]}
                return ModelReply(dumps(envelope), {"fixture": True}, "scripted-a-review")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _source_a(root)
            output_root = root / "a_review_output"
            output_root.mkdir()
            model = PagedReviewModel()
            args = Namespace(provider=None, model=None, reasoning_effort=None,
                             thinking=None, timeout_seconds=None)
            with patch.object(comparison, "EXPECTED_FACTORS", 1), \
                    patch.object(comparison, "choose_model", return_value=(model, SETTINGS)):
                result = comparison.run_a_review(
                    fixture["scope"], output_root, ROOT / "src", args
                )
            self.assertEqual(result["candidate_count"], 1)
            self.assertLess(result["context_measure"]["input_bound_after_paging"], 1_000_000)
            self.assertEqual(len(model.pages), 1)
            self.assertEqual(model.pages[0]["offset"], 0)
            self.assertEqual(model.pages[0]["limit"], 2)
            self.assertEqual(len(model.page_evidence[0]["data"]["items"]), 2)
            request = model.requests[0]
            self.assertEqual(len(request["records"]), 1)
            record = request["records"][0]
            self.assertEqual(record["kind"], "fixed_A_horizon_evidence")
            horizon_reports = record["data"]["a_evidence"]["horizon_comparison"]["horizons"]
            self.assertEqual({item["segment"] for item in horizon_reports.values()}, {"A"})
            prohibited = {"passed_horizons", "horizon_results", "batch_correction", "validation_result"}
            def assert_a_only(value):
                if isinstance(value, dict):
                    self.assertTrue(prohibited.isdisjoint(value))
                    if "segment" in value:
                        self.assertEqual(value["segment"], "A")
                    for nested in value.values():
                        assert_a_only(nested)
                elif isinstance(value, list):
                    for nested in value:
                        assert_a_only(nested)
            assert_a_only(request)
            assert_a_only(model.page_evidence)

    def test_compare_rejects_old_count_mismatch_and_new_count_shortfall(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary)
            key = "factor-fixture"
            candidate = {
                "factor_key": key, "batch_index": 1, "candidate_id": "candidate-0001",
                "source_run_id": "source-run", "identity": {
                    "expanded_expression": "div(perp_close,perp_open)",
                    "direction": "-1", "semantics_version": "expr-v1",
                },
            }
            scope = {
                "source_goal": "synthetic", "source_goal_id": "synthetic",
                "original_card_count": 1, "original_cards": [{"factor_key": key}],
                "batch_count": 1, "candidate_count": 1,
                "batches": [{"batch_index": 1, "source_run_id": "source-run",
                             "candidate_count": 1, "factor_keys": [key]}],
                "candidates": [candidate],
            }
            review = {"decisions": [{"factor_key": key, "retained_horizons": [24]}]}

            def group(name, qualified):
                state = {
                    "validation_status": "passed" if qualified else "not_passed",
                    "eligible_for_idea_pool": bool(qualified), "reasons": [],
                    "passed_horizons": [24] if qualified else [],
                    "retained_horizons": [24], "horizon_results": {}, "horizons": {},
                    "run_path": "/synthetic/run", "evaluation_record_path": "/synthetic/eval.json",
                    "validation_record_path": "/synthetic/validation.json",
                    "model_report_path": "/synthetic/report.json",
                }
                if qualified:
                    state.update({"idea_id": f"{name}--candidate-0001",
                                  "card_path": "/synthetic/card.json",
                                  "admission_receipt": "/synthetic/admission.json",
                                  "match_receipt": "/synthetic/match.json", "matches_goal": True})
                return {
                    "group": name, "policy": name, "engine_root": "/synthetic/src",
                    "goal_root": f"/synthetic/{name}/goal", "idea_pool": f"/synthetic/{name}/pool",
                    "corrected_goal": {"goal_id": name, "objective": "source goal objective", "target_ideas": 1},
                    "previous_goal_objective": "wrong process objective",
                    "recheck_manifest": f"/synthetic/{name}/recheck_manifest.json",
                    "recheck_result": f"/synthetic/{name}/recheck_result.json",
                    "prior_completion_records": f"/synthetic/{name}/completion_records",
                    "rechecked_completion_records": f"/synthetic/{name}/completion_rechecks/v1/completion_records",
                    "target_ideas": 1, "qualified_count": qualified,
                    "qualified_identity_ids": [key] if qualified else [],
                    "qualified_identities": [key] if qualified else [],
                    "qualified": {key: state} if qualified else {}, "cards": [], "batches": [],
                    "run_dirs": [], "factor_archive_root": "/synthetic/archive",
                    "fixed_b_panel": "/synthetic/panel/provenance.json",
                    "frozen_b_values_sha256": "values", "frozen_b_universe_sha256": "universe",
                    "all_candidates": {key: state},
                }

            with patch.object(comparison, "EXPECTED_OLD_CARDS", 1), \
                    patch.object(comparison, "load_prepared_scope", return_value=(scope, review)), \
                    patch.object(comparison, "verify_frozen_panel", return_value={
                        "artifacts": {"values": {"sha256": "values"}, "universe": {"sha256": "universe"}}
                    }), \
                    patch.object(comparison, "collect_group_results", side_effect=[group("old24", 0),
                                                                                   group("new_multi", 1)]):
                with self.assertRaises(comparison.ComparisonError):
                    comparison.compare(Namespace(output_root=experiment))
            result = json.loads((experiment / "comparison.json").read_text())
            self.assertEqual(result["status"], "not_met")
            self.assertFalse(result["acceptance"]["old24_reproduced_five"])

        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary)
            key = "factor-fixture"
            candidate = {"factor_key": key, "batch_index": 1, "candidate_id": "candidate-0001",
                         "source_run_id": "source-run", "identity": {
                             "expanded_expression": "div(perp_close,perp_open)",
                             "direction": "-1", "semantics_version": "expr-v1"}}
            scope = {"source_goal": "synthetic", "source_goal_id": "synthetic",
                     "original_card_count": 1, "original_cards": [{"factor_key": key}], "batch_count": 1,
                     "candidate_count": 1, "batches": [], "candidates": [candidate]}
            review = {"decisions": [{"factor_key": key, "retained_horizons": [24]}]}
            state = {
                "validation_status": "passed", "eligible_for_idea_pool": True,
                "reasons": [], "passed_horizons": [24], "retained_horizons": [24],
                "horizon_results": {}, "horizons": {}, "run_path": "/synthetic/run",
                "evaluation_record_path": "/synthetic/evaluation.json",
                "validation_record_path": "/synthetic/validation.json",
                "model_report_path": "/synthetic/report.json",
            }
            def fake_group(name, count):
                return {
                    "group": name, "policy": name, "engine_root": "/synthetic/src",
                    "goal_root": f"/synthetic/{name}/goal", "idea_pool": f"/synthetic/{name}/pool",
                    "corrected_goal": {"goal_id": name, "objective": "source goal objective", "target_ideas": 1},
                    "previous_goal_objective": "wrong process objective",
                    "recheck_manifest": f"/synthetic/{name}/recheck_manifest.json",
                    "recheck_result": f"/synthetic/{name}/recheck_result.json",
                    "prior_completion_records": f"/synthetic/{name}/completion_records",
                    "rechecked_completion_records": f"/synthetic/{name}/completion_rechecks/v1/completion_records",
                    "target_ideas": 1, "qualified_count": count,
                    "qualified_identities": [key] if count else [],
                    "qualified": {key: state} if count else {}, "cards": [], "batches": [],
                    "run_dirs": [], "factor_archive_root": "/synthetic/archive",
                    "fixed_b_panel": "/synthetic/panel/provenance.json",
                    "frozen_b_values_sha256": "values", "frozen_b_universe_sha256": "universe",
                    "all_candidates": {key: state},
                }
            with patch.object(comparison, "EXPECTED_OLD_CARDS", 1), \
                    patch.object(comparison, "load_prepared_scope", return_value=(scope, review)), \
                    patch.object(comparison, "verify_frozen_panel", return_value={
                        "artifacts": {"values": {"sha256": "values"}, "universe": {"sha256": "universe"}}
                    }), \
                    patch.object(comparison, "collect_group_results", side_effect=[
                        fake_group("old24", 1), fake_group("new_multi", 1),
                    ]):
                with self.assertRaises(comparison.ComparisonError):
                    comparison.compare(Namespace(output_root=experiment))
            result = json.loads((experiment / "comparison.json").read_text())
            self.assertTrue(result["acceptance"]["old24_reproduced_five"])
            self.assertFalse(result["acceptance"]["new_multi_at_least_six"])

    @staticmethod
    def _passing_b_evaluation(values, labels, spec, stage, direction, *, horizon_hours=24):
        from crypto_quant.research.factor_mining.evaluation import evaluate_factor
        report = evaluate_factor(values, labels, spec, stage, direction, horizon_hours=horizon_hours)
        if stage == "B":
            report["summary"]["rank_ic"].update(mean=direction * 0.2, p_value=0.000001)
            report["summary"]["directional_spread"].update(mean=0.003, p_value=0.000001)
            report["summary"]["positive_stage_share"] = 1.0
        return report


if __name__ == "__main__":
    unittest.main()
