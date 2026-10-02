import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crypto_quant.research.factor_mining import records as records_module
from crypto_quant.research.factor_mining.factor_archive import EvaluationKey, FactorArchive, FactorIdentity
from crypto_quant.research.factor_mining.records import EvidenceIntegrityError, RecordStore


class FactorArchiveLocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.run = self.project / "experiments" / "factor_mining" / "run-v6"
        self.run.mkdir(parents=True)
        self.archive_root = self.project / "experiments" / "factor_archive_v2"
        self.identity = FactorIdentity("rank(spot_close)", "1", "expr-v1")
        self.key = EvaluationKey("run-v6/A", "contract-v6", "factor-eval-v1", "A", "24h")
        self.payload = {"candidate_id": "candidate-1", "periods": [{"rank_ic": 0.25}]}
        archive = FactorArchive.open_for(self.archive_root, self.identity)
        saved = archive.append_evaluation(self.key, self.payload,
            factor_values_csv=b"symbol,value\nBTCUSDT,1\nETHUSDT,2\n")
        relative_root = os.path.relpath(self.archive_root, self.run)
        self.marker = {"format": "one-factor-one-file-v2", "archive_root": relative_root}
        self.locator = {"root": relative_root, "identity": self.identity.as_dict(),
                        "evaluation_key": self.key.as_dict(), **saved}
        self.record = {"id": "candidate-1-evaluation", "kind": "evaluation",
                       "data": {"candidate_id": "candidate-1", "factor_archive": self.locator}}
        self._write_marker()
        (self.run / "a_records").mkdir()
        self._write_record()

    def _write_marker(self):
        (self.run / "factor_archive.json").write_text(json.dumps(self.marker), encoding="utf-8")

    def _write_record(self):
        path = self.run / "a_records" / "candidate-1-evaluation.json"
        path.write_text(json.dumps(self.record), encoding="utf-8")

    def test_same_v6_artifact_reads_from_primary_and_snapshot_source_locations(self):
        primary = self.project / "src/crypto_quant/research/factor_mining/records.py"
        snapshot = self.project / "source_snapshot/src/crypto_quant/research/factor_mining/records.py"
        with patch.object(records_module, "__file__", str(primary)):
            store = RecordStore(self.run / "a_records")
            primary_records = store.all()
            primary_page = store.read("candidate-1-evaluation", "/factor_values", 1, 1)
        with patch.object(records_module, "__file__", str(snapshot)):
            store = RecordStore(self.run / "a_records")
            self.assertEqual(store.archive_root, self.archive_root.resolve())
            self.assertEqual(store.all(), primary_records)
            self.assertEqual(store.read("candidate-1-evaluation", "/factor_values", 1, 1), primary_page)
        self.assertEqual(primary_records[0]["data"]["periods"], self.payload["periods"])
        self.assertEqual(primary_page, {"total": 2, "offset": 1,
                                       "items": [{"symbol": "ETHUSDT", "value": "2"}]})

    def test_relative_marker_remains_readable_after_moving_run_and_archive_together(self):
        moved = self.root / "portable-project"
        run_relative = self.run.relative_to(self.project)
        archive_relative = self.archive_root.relative_to(self.project)
        shutil.move(self.project, moved)
        store = RecordStore(moved / run_relative / "a_records")
        self.assertEqual(store.archive_root, (moved / archive_relative).resolve())
        self.assertEqual(store.read("candidate-1-evaluation", "/periods", 0, 1), {
            "total": 1, "offset": 0, "items": self.payload["periods"],
        })

    def test_new_archive_location_is_run_sibling_regardless_of_source_or_directory_names(self):
        primary = self.project / "src/crypto_quant/research/factor_mining/records.py"
        snapshot = self.project / "source_snapshot/src/crypto_quant/research/factor_mining/records.py"
        for source in (primary, snapshot):
            with patch.object(records_module, "__file__", str(source)):
                for run in (self.run, self.root / "research-results" / "goal-1" / "runs" / "run-v6"):
                    with self.subTest(source=source, run=run):
                        self.assertEqual(RecordStore._expected_archive_root(run),
                                         run.resolve().parent / "factor_archive_v2")

    def test_marker_format_still_required(self):
        self.marker["format"] = "unsupported-format"
        self._write_marker()
        with self.assertRaises(EvidenceIntegrityError):
            RecordStore(self.run / "a_records")

    def test_absolute_marker_root_still_rejected(self):
        self.marker["archive_root"] = str(self.archive_root.resolve())
        self._write_marker()
        with self.assertRaises(EvidenceIntegrityError):
            RecordStore(self.run / "a_records")

    def test_locator_must_match_marker_even_when_other_root_contains_same_archive(self):
        other_root = self.project / "experiments" / "other_archive"
        shutil.copytree(self.archive_root, other_root)
        self.locator["root"] = os.path.relpath(other_root, self.run)
        self._write_record()
        store = RecordStore(self.run / "a_records")
        with self.assertRaises(EvidenceIntegrityError) as raised:
            store.all()
        self.assertIn("locator differs from the run marker", str(raised.exception.__cause__))

    def test_archive_identity_still_required(self):
        self.locator["identity"] = {**self.identity.as_dict(), "direction": "-1"}
        self._write_record()
        with self.assertRaisesRegex(FileNotFoundError, "factor identity is not registered"):
            RecordStore(self.run / "a_records").all()

    def test_archive_evaluation_and_value_set_ids_still_required(self):
        for field, message in (("evaluation_id", "evaluation ID differs"),
                               ("value_set_id", "value-set ID differs")):
            with self.subTest(field=field):
                original = self.locator[field]
                self.locator[field] = "999"
                self._write_record()
                with self.assertRaises(EvidenceIntegrityError) as raised:
                    RecordStore(self.run / "a_records").all()
                self.assertIn(message, str(raised.exception.__cause__))
                self.locator[field] = original

    def test_current_frozen_A_reference_still_reads_archived_evaluation(self):
        b_root = self.run / "b_records"
        store = RecordStore(b_root)
        store.append("candidate-1-frozen", "frozen_definition_and_A_evidence", {
            "candidate_id": "candidate-1", "a_evaluation_ref": {"record_id": "candidate-1-evaluation"},
        })
        frozen = store.all()[0]
        self.assertEqual(frozen["data"]["a_evaluation_ref"]["data"]["periods"], self.payload["periods"])

    def test_plain_json_store_remains_available_without_factor_marker(self):
        store = RecordStore(self.root / "strategy" / "records")
        store.append("research-note", "strategy_event", {"status": "completed", "rows": [1, 2, 3]})
        self.assertIsNone(store.archive_root)
        self.assertEqual(store.all()[0]["data"]["status"], "completed")
        self.assertEqual(store.read("research-note", "/rows", 1, 1), {
            "total": 3, "offset": 1, "items": [2],
        })


if __name__ == "__main__":
    unittest.main()
