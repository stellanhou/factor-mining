"""Synthetic tests for the standalone B numerical checkpoint boundary."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from crypto_quant.research.factor_mining.contracts import dumps
from crypto_quant.research.factor_mining.factor_archive import FactorArchive
from crypto_quant.research.factor_mining.records import RecordStore
import crypto_quant.research.factor_mining.research_actions as research_actions
import crypto_quant.research.factor_mining.workflow as workflow
from crypto_quant.research.factor_mining.workflow import FactorMiner
from test_factor_mining import ScenarioModel, definition, input_panel, specification


class NumericalCheckpointTests(unittest.TestCase):
    def _prepare_B(self, directory: str):
        def two_candidates(request, result):
            if request["role"] == "ideator":
                result["candidates"] = [
                    definition(f"div(perp_close,ts_mean(perp_close,{window}))")
                    for window in (3, 2)]

        model = ScenarioModel(two_candidates)
        model.propose_once = False
        miner = FactorMiner(specification(), model, Path(directory))
        completion = miner.explore(input_panel(miner.spec))
        panel = input_panel(miner.spec, "B")
        miner.freeze(completion["retained_ids"], panel.universe)
        return miner, model, panel

    def _prepare_A(self, directory: str):
        def two_candidates(request, result):
            if request["role"] == "ideator":
                result["candidates"] = [
                    definition(f"div(perp_close,ts_mean(perp_close,{window}))")
                    for window in (3, 2)]

        model = ScenarioModel(two_candidates)
        model.propose_once = False
        miner = FactorMiner(specification(), model, Path(directory))
        completion = miner.explore(input_panel(miner.spec))
        return miner, model, completion, input_panel(miner.spec, "B")

    def _assert_rejected_after_tampering(self, miner: FactorMiner, path: Path, mutate) -> None:
        original = path.read_bytes()
        value = copy.deepcopy(json.loads(original))
        mutate(value)
        path.write_text(dumps(value), encoding="utf-8")
        loader = Mock(side_effect=AssertionError("a saved checkpoint must not reload B"))
        try:
            with self.assertRaises(ValueError):
                miner.validate_numerical(loader)
            loader.assert_not_called()
        finally:
            path.write_bytes(original)

    def test_numerical_checkpoint_finishes_before_reports_and_recovers_without_B_access(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, model, panel = self._prepare_B(directory)
            access_marker = miner.root / "b-access-started.json"

            def load_after_marker():
                self.assertTrue(access_marker.is_file())
                return panel

            loader = Mock(side_effect=load_after_marker)
            model_calls = len(model.requests)
            result = miner.validate_numerical(loader)

            self.assertEqual(result["status"], "numerical_complete")
            self.assertEqual(result["candidate_ids"], ["candidate-0001", "candidate-0002"])
            self.assertEqual(result["retained_horizons"], {
                "candidate-0001": [24], "candidate-0002": [24]})
            self.assertEqual(result["pending_report_ids"], ["candidate-0001", "candidate-0002"])
            self.assertIn("batch_correction", result)
            self.assertEqual(set(result["decisions"]), set(result["candidate_ids"]))
            self.assertEqual(len(model.requests), model_calls)
            self.assertTrue((miner.root / "b-numerical-complete.json").is_file())
            self.assertFalse((miner.root / "validation.json").exists())
            self.assertFalse((miner.root / "B-report.md").exists())
            loader.assert_called_once_with()

            restored = FactorMiner.open(miner.root, model)
            resume_loader = Mock(side_effect=AssertionError("checkpoint recovery must not reload B"))
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_expression") as evaluate_expression, \
                    patch("crypto_quant.research.factor_mining.workflow.evaluate_factor") as evaluate_factor:
                recovered = restored.validate_numerical(resume_loader)
            self.assertEqual(recovered, result)
            resume_loader.assert_not_called()
            evaluate_expression.assert_not_called()
            evaluate_factor.assert_not_called()

    def test_loader_failure_records_stop_and_prevents_a_second_B_read(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, _, _ = self._prepare_B(directory)
            loader = Mock(side_effect=RuntimeError("synthetic loader failure"))
            with self.assertRaisesRegex(RuntimeError, "synthetic loader failure"):
                miner.validate_numerical(loader)

            marker = json.loads((miner.root / "b-access-started.json").read_text())
            stopped = json.loads((miner.root / "validation-stopped.json").read_text())
            self.assertEqual(marker["candidate_ids"], ["candidate-0001", "candidate-0002"])
            self.assertIn("inspect existing evidence", stopped["B_status"])
            self.assertFalse((miner.root / "b-numerical-complete.json").exists())
            loader.assert_called_once_with()

            blocked_loader = Mock(side_effect=AssertionError("B access must stay stopped"))
            with self.assertRaisesRegex(ValueError, "B access was attempted"):
                miner.validate_numerical(blocked_loader)
            loader.assert_called_once_with()
            blocked_loader.assert_not_called()

    def test_stop_during_loader_leaves_access_marker_without_recovery_or_reread(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, _, panel = self._prepare_B(directory)
            active = True
            inside_commit = False

            def stop_check():
                if not active:
                    raise ValueError("program actions are stopped")

            def commit(operation):
                nonlocal inside_commit
                stop_check()
                inside_commit = True
                try:
                    return operation()
                finally:
                    inside_commit = False

            def stop_after_load():
                nonlocal active
                self.assertFalse(inside_commit)
                self.assertTrue((miner.root / "b-access-started.json").is_file())
                active = False
                return panel

            loader = Mock(side_effect=stop_after_load)
            with self.assertRaisesRegex(ValueError, "program actions are stopped"):
                miner.validate_numerical(loader, commit=commit, stop_check=stop_check)
            loader.assert_called_once_with()
            self.assertTrue((miner.root / "b-access-started.json").is_file())
            self.assertFalse((miner.root / "B-universe.csv").exists())
            self.assertFalse((miner.root / "b-numerical-complete.json").exists())

            blocked_loader = Mock(side_effect=AssertionError("a stopped B load cannot be retried"))
            with self.assertRaisesRegex(ValueError, "program actions are stopped"):
                miner.validate_numerical(blocked_loader, commit=commit, stop_check=stop_check)
            blocked_loader.assert_not_called()

    def test_formal_B_writes_use_commit_and_computation_stays_outside_it(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, _, panel = self._prepare_B(directory)
            inside_commit = False
            committed_writes = []
            original_write_json = workflow.write_json
            original_group_plot = workflow.write_group_plot
            original_evaluate_expression = workflow.evaluate_expression
            original_evaluate_factor = workflow.evaluate_factor
            original_record_append = RecordStore.append
            original_archive_append = FactorArchive.append_evaluation
            original_path_open = Path.open

            def commit(operation):
                nonlocal inside_commit
                self.assertFalse(inside_commit)
                inside_commit = True
                try:
                    return operation()
                finally:
                    inside_commit = False

            def check_stop():
                self.assertFalse(inside_commit)

            def checked_write_json(path, value):
                self.assertTrue(inside_commit)
                committed_writes.append(path.name)
                return original_write_json(path, value)

            def checked_record_append(store, *args, **kwargs):
                self.assertTrue(inside_commit)
                committed_writes.append("record:" + str(args[0]))
                return original_record_append(store, *args, **kwargs)

            def checked_archive_append(archive, *args, **kwargs):
                self.assertTrue(inside_commit)
                committed_writes.append("archive")
                return original_archive_append(archive, *args, **kwargs)

            def checked_group_plot(path, report):
                self.assertTrue(inside_commit)
                committed_writes.append(path.name)
                return original_group_plot(path, report)

            def checked_evaluate_expression(*args, **kwargs):
                self.assertFalse(inside_commit)
                return original_evaluate_expression(*args, **kwargs)

            def checked_evaluate_factor(*args, **kwargs):
                self.assertFalse(inside_commit)
                return original_evaluate_factor(*args, **kwargs)

            def checked_path_open(path, *args, **kwargs):
                if path.name == "B-universe.csv":
                    self.assertTrue(inside_commit)
                    committed_writes.append(path.name)
                return original_path_open(path, *args, **kwargs)

            def load_outside_commit():
                self.assertFalse(inside_commit)
                return panel

            with patch.object(research_actions, "read_b_checkpoint",
                              return_value={"status": "numerical_complete"}, create=True), \
                    patch.object(workflow, "write_json", side_effect=checked_write_json), \
                    patch.object(RecordStore, "append", checked_record_append), \
                    patch.object(FactorArchive, "append_evaluation", checked_archive_append), \
                    patch.object(Path, "open", checked_path_open), \
                    patch.object(workflow, "write_group_plot", checked_group_plot), \
                    patch.object(workflow, "evaluate_expression", side_effect=checked_evaluate_expression) as evaluate_expression, \
                    patch.object(workflow, "evaluate_factor", side_effect=checked_evaluate_factor) as evaluate_factor:
                result = miner.validate_numerical(load_outside_commit, commit=commit,
                                                  stop_check=check_stop)

            self.assertEqual(result["status"], "numerical_complete")
            self.assertIn("b-access-started.json", committed_writes)
            self.assertIn("B-universe.csv", committed_writes)
            self.assertIn("b-numerical-complete.json", committed_writes)
            self.assertTrue(any(value.startswith("record:") for value in committed_writes))
            self.assertIn("archive", committed_writes)
            self.assertTrue(any(value.endswith("-B-24h.svg") for value in committed_writes))
            evaluate_expression.assert_called()
            evaluate_factor.assert_called()

    def test_freeze_can_bind_a_complete_A_manifest_record(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, _, completion, panel = self._prepare_A(directory)
            manifest_ref = "a-manifest-fixture"
            manifest_data = {key: completion[key]
                             for key in ("candidate_ids", "evaluated_ids", "retained_ids")}
            miner.store.append(manifest_ref, "a_manifest", manifest_data)
            completion_path = miner.root / "a-complete.json"
            saved_completion = json.loads(completion_path.read_text())
            completion_with_manifest = {**saved_completion, "status": "complete",
                                        "manifest_ref": manifest_ref}
            completion_path.write_text(dumps(completion_with_manifest), encoding="utf-8")

            self.assertGreater(len(completion["retained_ids"]), 1)
            with self.assertRaisesRegex(ValueError, "manifest_ref"):
                miner.freeze(completion["retained_ids"], panel.universe)
            with self.assertRaisesRegex(ValueError, "every retained candidate"):
                miner.freeze(completion["retained_ids"][:-1], panel.universe,
                             a_manifest_ref=manifest_ref)
            with self.assertRaisesRegex(ValueError, "A manifest is sealed"):
                miner.resume_explore(input_panel(miner.spec))
            with self.assertRaisesRegex(ValueError, "A manifest is sealed"):
                miner.complete_reports("A", Path(directory) / "ideas")

            miner.freeze(completion["retained_ids"], panel.universe,
                         a_manifest_ref=manifest_ref)
            frozen = json.loads((miner.root / "frozen_batch.json").read_text())
            self.assertEqual(frozen["a_manifest_ref"], manifest_ref)
            miner._checked_frozen()

            loader = Mock(side_effect=AssertionError("a native manifest run must use the program action"))
            with self.assertRaisesRegex(ValueError, "program action"):
                miner.validate(loader, Path(directory) / "ideas")
            with self.assertRaisesRegex(ValueError, "program action"):
                miner.complete_reports("B", Path(directory) / "ideas")
            with self.assertRaisesRegex(ValueError, "action-bound commit and stop checks"):
                miner.validate_numerical(loader)
            loader.assert_not_called()

            manifest_path = miner.store.root / f"{manifest_ref}.json"
            saved_manifest = manifest_path.read_bytes()
            manifest_path.unlink()
            try:
                with self.assertRaisesRegex(ValueError, "missing or ambiguous manifest record"):
                    miner._checked_frozen()
            finally:
                manifest_path.write_bytes(saved_manifest)

            manifest = json.loads(saved_manifest)
            manifest["kind"] = "model_report"
            manifest_path.write_text(dumps(manifest), encoding="utf-8")
            try:
                with self.assertRaisesRegex(ValueError, "missing or ambiguous manifest record"):
                    miner._checked_frozen()
            finally:
                manifest_path.write_bytes(saved_manifest)

            completion_path.write_text(dumps({**completion_with_manifest,
                                              "manifest_ref": "a-manifest-other"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or ambiguous manifest record"):
                miner._checked_frozen()

    def test_A_freeze_uses_latest_valid_model_report_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, _, completion, panel = self._prepare_A(directory)
            candidate_id = completion["retained_ids"][0]
            base_report = miner.store._load(miner.store.root / f"{candidate_id}-report.json")
            latest_report_data = {**base_report["data"],
                                  "analysis": "脚本修订版报告，仅检查版本选择。"}
            miner.store.append(f"{candidate_id}-report-revision-0001", "model_report",
                               latest_report_data)

            miner._load_candidates()
            self.assertEqual(miner.candidates[candidate_id]["model_report"], latest_report_data)
            miner.freeze(completion["retained_ids"], panel.universe)
            frozen = json.loads((miner.root / "frozen_batch.json").read_text())
            self.assertEqual(frozen["candidates"][candidate_id]["a_model_report"], latest_report_data)

    def test_action_bound_A_cannot_use_legacy_resume_or_report_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = specification()
            panel = input_panel(spec)
            model = ScenarioModel()
            miner = FactorMiner(spec, model, Path(directory))
            actions = research_actions.ResearchActions(
                miner, a_panel=panel,
                goal_spec={"goal_id": "binding-gate", "objective": spec.objective, "target_ideas": 1})
            actions._ensure_a_initialized("binding-gate-setup")
            self.assertTrue((miner.root / "a-action-input-binding.json").is_file())

            with self.assertRaisesRegex(ValueError, "action-bound A runs"):
                miner.resume_explore(panel)
            with self.assertRaisesRegex(ValueError, "action-bound A runs"):
                miner.complete_reports("A", Path(directory) / "ideas")
            self.assertEqual(model.requests, [])

    def test_checkpoint_reader_rechecks_membership_archives_correction_and_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, model, panel = self._prepare_B(directory)
            miner.validate_numerical(lambda: panel)
            restored = FactorMiner.open(miner.root, model)

            checkpoint = miner.root / "b-numerical-complete.json"
            self._assert_rejected_after_tampering(
                restored, checkpoint,
                lambda value: value["candidate_ids"].__setitem__(0, "candidate-9999"))

            def change_horizon(value):
                horizons = value["retained_horizons"]["candidate-0001"]
                value["retained_horizons"]["candidate-0001"] = [1] if horizons != [1] else [4]

            self._assert_rejected_after_tampering(
                restored, checkpoint, change_horizon)

            membership_path = miner.root / "B-universe.csv"
            original_membership = membership_path.read_bytes()
            changed_membership = original_membership.replace(b",True\n", b",False\n", 1)
            self.assertNotEqual(changed_membership, original_membership)
            membership_path.write_bytes(changed_membership)
            no_load = Mock(side_effect=AssertionError("saved evidence checks must not reload B"))
            try:
                with self.assertRaises(ValueError):
                    restored.validate_numerical(no_load)
                no_load.assert_not_called()
            finally:
                membership_path.write_bytes(original_membership)

            self._assert_rejected_after_tampering(
                restored, miner.root / "b_records" / "candidate-0001-evaluation.json",
                lambda value: value["data"]["horizons"]["24"]["factor_archive"]["identity"].__setitem__(
                    "expanded_expression", "perp_close"))
            self._assert_rejected_after_tampering(
                restored, miner.root / "b_records" / "batch-correction.json",
                lambda value: value.__setitem__("data", {}))

            def corrupt_admission(value):
                value["data"]["eligible_for_idea_pool"] = not value["data"]["eligible_for_idea_pool"]

            self._assert_rejected_after_tampering(
                restored, miner.root / "b_records" / "candidate-0001-validation.json", corrupt_admission)


if __name__ == "__main__":
    unittest.main()
