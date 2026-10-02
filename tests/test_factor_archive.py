import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from crypto_quant.research.factor_mining.factor_archive import (
    ArchiveIntegrityError,
    EvaluationConflictError,
    EvaluationKey,
    FactorArchive,
    FactorIdentity,
)


class FactorArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identity = FactorIdentity("Rank(close / delay(close, 1))", "higher_is_better", "expr-v3")
        self.archive = FactorArchive.open_for(self.root, self.identity)
        self.key = EvaluationKey("data-v1", "contract-v1", "eval-v1", "A", "1h")

    def tearDown(self):
        self.temp.cleanup()

    def test_identity_selects_one_sequential_file_and_semantics_change_selects_another(self):
        same = FactorArchive.open_for(self.root, self.identity)
        existing = FactorArchive.open_existing(self.root, self.identity)
        changed = FactorArchive.open_for(self.root, FactorIdentity(
            self.identity.expanded_expression, self.identity.direction, "expr-v4"))
        self.assertEqual(self.archive.path, same.path)
        self.assertEqual(self.archive.path, existing.path)
        self.assertNotEqual(self.archive.path, changed.path)
        self.assertEqual(self.archive.path.name, "factor-000001.sqlite3")
        self.assertEqual(changed.path.name, "factor-000002.sqlite3")
        self.assertTrue((self.root / "registry.sqlite3").is_file())
        self.assertEqual(len(list(self.root.glob("factor-*.sqlite3"))), 2)

    def test_open_existing_fails_without_creating_missing_registry(self):
        missing_root = self.root / "missing"
        with self.assertRaises(FileNotFoundError):
            FactorArchive.open_existing(missing_root, self.identity)
        self.assertFalse(missing_root.exists())

    def test_round_trip_paging_and_lossless_csv(self):
        csv_bytes = b"ts,symbol,value\r\n2024-01-01T00:00:00Z,BTCUSDT,1.2500\r\n"
        result = self.archive.append_evaluation(
            self.key, {"ic": 0.25, "rows": 2},
            factor_values=[{"ts": "t1", "symbol": "BTCUSDT", "value": 1.25},
                           {"ts": "t2", "symbol": "ETHUSDT", "value": -0.5}],
            factor_values_csv=csv_bytes, provenance={"source": "run-1"})
        evidence = self.archive.get_evaluation(self.key)
        self.assertEqual(result["evaluation_id"], evidence["evaluation_id"])
        self.assertNotIn("content_sha256", result)
        self.assertNotIn("content_sha256", evidence)
        self.assertEqual(self.archive.get_factor_values_csv(self.key), csv_bytes)
        self.assertEqual(self.archive.page_factor_values(self.key, offset=1, limit=1), {
            "total": 2, "offset": 1,
            "items": [{"ts": "t2", "symbol": "ETHUSDT", "value": -0.5}],
        })

    def test_same_key_and_content_is_idempotent_but_conflicting_content_fails(self):
        first = self.archive.append_evaluation(self.key, {"score": 1})
        second = self.archive.append_evaluation(self.key, {"score": 1})
        self.assertEqual(first, second)
        with self.assertRaises(EvaluationConflictError):
            self.archive.append_evaluation(self.key, {"score": 2})
        with closing(sqlite3.connect(self.archive.path)) as db, db:
            self.assertEqual(db.execute("SELECT count(*) FROM evaluations").fetchone()[0], 1)

    def test_equal_content_is_stored_separately_for_distinct_evaluation_keys(self):
        csv_bytes = b"same,raw,bytes\n"
        self.archive.append_evaluation(self.key, {"result": [1, 2]}, factor_values_csv=csv_bytes)
        second_key = EvaluationKey("data-v2", "contract-v2", "eval-v2", "A", "1h")
        self.archive.append_evaluation(second_key, {"result": [1, 2]}, factor_values_csv=csv_bytes)
        with closing(sqlite3.connect(self.archive.path)) as db, db:
            self.assertEqual(db.execute("SELECT count(*) FROM evaluations").fetchone()[0], 2)
            self.assertEqual(db.execute("SELECT count(*) FROM value_sets").fetchone()[0], 2)
            self.assertEqual(db.execute("SELECT count(*) FROM blobs").fetchone()[0], 4)

    def test_values_can_be_checkpointed_before_evaluation_and_reused(self):
        csv_bytes = b"ts,symbol,value\n2024-01-01,BTCUSDT,1\n"
        stored = self.archive.append_value_set("data-v1", "eval-v1", csv_bytes,
            provenance={"calculation_record": "calculation-1"})
        self.assertEqual(set(stored), {"value_set_id"})
        self.assertEqual(self.archive.get_value_set("data-v1", "eval-v1"), {
            "value_set_id": stored["value_set_id"], "factor_values_csv": csv_bytes,
            "provenance": [{"calculation_record": "calculation-1"}],
        })
        repeated = self.archive.append_value_set("data-v1", "eval-v1", csv_bytes,
            provenance={"calculation_record": "calculation-2"})
        self.assertEqual(stored, repeated)
        self.assertEqual(set(tuple(sorted(item.items())) for item in
            self.archive.get_value_set("data-v1", "eval-v1")["provenance"]), {
                (("calculation_record", "calculation-1"),), (("calculation_record", "calculation-2"),),
            })
        result = self.archive.append_evaluation(self.key, {"rows": 1}, value_set_id=stored["value_set_id"],
            provenance={"run": "run-1"})
        self.assertEqual(result["value_set_id"], stored["value_set_id"])
        repeated_eval = self.archive.append_evaluation(self.key, {"rows": 1}, value_set_id=stored["value_set_id"],
            provenance={"run": "run-2"})
        self.assertEqual(result, repeated_eval)
        self.assertEqual(self.archive.get_factor_values_csv(self.key), csv_bytes)
        evidence = self.archive.get_evaluation(self.key)
        self.assertEqual(evidence["evaluation_id"], result["evaluation_id"])
        self.assertEqual(evidence["value_set_id"], stored["value_set_id"])
        self.assertNotIn("content_sha256", self.archive.get_evaluation(self.key))
        with self.assertRaises(EvaluationConflictError):
            self.archive.append_value_set("data-v1", "eval-v1", b"different")

    def test_csv_value_set_pages_without_duplicate_row_storage(self):
        csv_bytes = b"timestamp,symbol,factor_value\n2024-01-01,BTCUSDT,1\n2024-01-02,ETHUSDT,2\n"
        self.archive.append_evaluation(self.key, {"summary": {}}, factor_values_csv=csv_bytes)
        page = self.archive.page_factor_values(self.key, offset=1, limit=1)
        self.assertEqual(page, {"total": 2, "offset": 1,
                                "items": [{"timestamp": "2024-01-02", "symbol": "ETHUSDT", "factor_value": "2"}]})
        with closing(sqlite3.connect(self.archive.path)) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM factor_values").fetchone()[0], 0)

    def test_read_fails_when_blob_is_corrupted(self):
        self.archive.append_evaluation(self.key, {"score": 3})
        with closing(sqlite3.connect(self.archive.path)) as db, db:
            blob_id = db.execute("SELECT payload_blob_id FROM evaluations").fetchone()[0]
            db.execute("UPDATE blobs SET data_zlib=? WHERE blob_id=?", (b"bad", blob_id))
        with self.assertRaises(ArchiveIntegrityError):
            self.archive.get_evaluation(self.key)

    def test_read_fails_when_blob_size_metadata_is_wrong(self):
        self.archive.append_evaluation(self.key, {"score": 3})
        with closing(sqlite3.connect(self.archive.path)) as db, db:
            blob_id = db.execute("SELECT payload_blob_id FROM evaluations").fetchone()[0]
            db.execute("UPDATE blobs SET raw_size=raw_size+1 WHERE blob_id=?", (blob_id,))
        with self.assertRaises(ArchiveIntegrityError):
            self.archive.get_evaluation(self.key)

    def test_database_and_public_results_contain_no_hash_fields(self):
        self.archive.append_evaluation(self.key, {"score": 3},
            factor_values=[{"symbol": "BTCUSDT", "value": 0.5}],
            factor_values_csv=b"symbol,value\nBTCUSDT,0.5\n",
            provenance={"calculation_record": "calc-1"})
        with closing(sqlite3.connect(self.archive.path)) as db, db:
            table_names = [row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            column_names = [column[1] for table in table_names
                            for column in db.execute(f"PRAGMA table_info({table})")]
        self.assertFalse(any("hash" in name.lower() or "sha" in name.lower() for name in column_names))
        self.assertFalse(any("hash" in table.lower() or "sha" in table.lower() for table in table_names))
        self.assertNotIn("content_sha256", self.archive.get_evaluation(self.key))
        self.assertNotIn("content_sha256", self.archive.get_value_set("data-v1", "eval-v1"))

    def test_existing_archive_still_rejects_identity_mismatch(self):
        with closing(sqlite3.connect(self.archive.path)) as db, db:
            db.execute("UPDATE archive_meta SET identity_json='{}'")
        with self.assertRaises(ArchiveIntegrityError):
            FactorArchive.open_existing(self.root, self.identity)

    def test_invalid_pages_and_missing_evidence_fail_fast(self):
        with self.assertRaises(ValueError):
            self.archive.page_factor_values(self.key, offset=-1, limit=2)
        with self.assertRaises(KeyError):
            self.archive.get_evaluation(self.key)

    def test_database_has_no_persistent_wal_sidecars(self):
        self.archive.append_evaluation(self.key, {"score": 1})
        self.assertTrue(self.archive.path.exists())
        self.assertFalse(Path(f"{self.archive.path}-wal").exists())
        self.assertFalse(Path(f"{self.archive.path}-shm").exists())


if __name__ == "__main__":
    unittest.main()
