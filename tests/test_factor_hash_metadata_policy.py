import unittest

from crypto_quant.research.factor_mining.contracts import without_hash_metadata


class FactorHashMetadataPolicyTests(unittest.TestCase):
    def test_old_and_new_cards_ignore_hash_metadata_but_keep_decisions(self):
        saved = {"id": "idea-1", "source": {"run_id": "run-1", "frozen_batch_sha256": "old"},
                 "historical_revalidation": {"source_candidates": [{"record_sha256": "old"}]},
                 "admission_evidence": {"validation_status": "passed"}}
        current = {"id": "idea-1", "source": {"run_id": "run-1"},
                   "historical_revalidation": {"source_candidates": [{"record_sha256": "new"}]},
                   "admission_evidence": {"validation_status": "passed"}}
        self.assertEqual(without_hash_metadata(saved), without_hash_metadata(current))
        current["admission_evidence"]["validation_status"] = "not_passed"
        self.assertNotEqual(without_hash_metadata(saved), without_hash_metadata(current))


if __name__ == "__main__":
    unittest.main()
