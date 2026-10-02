"""The active factor-mining contract accepts only the current FM-v6 rules."""

import json
import unittest
from pathlib import Path

from crypto_quant.research.factor_mining.contracts import ResearchSpec, dumps
from crypto_quant.research.factor_mining.evaluation import correct_batch


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/factor_mining/contract.example.json"


class FactorV6ContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_current_engineering_example_roundtrips_without_changing_fields(self):
        spec = ResearchSpec.from_dict(self.contract)
        self.assertEqual(spec.as_dict(), self.contract)
        self.assertEqual(dumps(spec.as_dict()), dumps(self.contract))
        self.assertEqual(spec.min_abs_ic, 0.01)
        self.assertEqual(spec.purpose, "engineering_check")

    def test_formal_v6_parameters_roundtrip_without_changing_fields(self):
        contract = {
            **self.contract,
            "run_id": "formal-v6-contract-fixture",
            "purpose": "research",
            "a_start": "2022-08-01T00:00:00Z",
            "b_start": "2024-08-01T00:00:00Z",
            "c_start": "2025-08-01T00:00:00Z",
            "c_end": "2026-08-01T00:00:00Z",
            "groups": 5,
            "min_symbols": 20,
            "min_periods": 48,
            "stage_hours": 168,
            "rolling_periods": 168,
            "min_abs_ic": 0.02,
        }
        spec = ResearchSpec.from_dict(contract)
        self.assertEqual(spec.as_dict(), contract)
        self.assertEqual(dumps(spec.as_dict()), dumps(contract))
        self.assertEqual(spec.min_abs_ic, 0.02)

    def test_calibrated_research_example_passes_the_selected_threshold(self):
        contract = json.loads((EXAMPLE.parent / "research.contract.json").read_text(encoding="utf-8"))
        spec = ResearchSpec.from_dict(contract)
        self.assertEqual(spec.min_abs_ic, 0.01)
        self.assertEqual(spec.purpose, "research")
        self.assertEqual(spec.b_horizons, (1, 4, 24))
        self.assertEqual(spec.fdr_method, "BH")
        self.assertFalse(spec.plan3_tracks_gate)
        self.assertEqual(spec.as_dict(), contract)

    def test_direct_construction_defaults_to_current_v6_rules(self):
        values = {key: value for key, value in self.contract.items()
                  if key not in {"admission_scheme", "plan3_tracks_gate"}}
        spec = ResearchSpec(**values)
        self.assertEqual(spec.admission_scheme, "plan3")
        self.assertIs(spec.plan3_tracks_gate, False)
        self.assertEqual(spec.as_dict(), self.contract)

    def test_saved_contract_must_explicitly_declare_current_rules(self):
        for field in ("fdr_method", "admission_scheme", "plan3_tracks_gate", "b_horizons"):
            with self.subTest(field=field):
                contract = {key: value for key, value in self.contract.items() if key != field}
                with self.assertRaisesRegex(ValueError, "must explicitly declare"):
                    ResearchSpec.from_dict(contract)

    def test_b_horizon_contract_is_fixed_and_typed(self):
        self.assertEqual(ResearchSpec.from_dict(self.contract).b_horizons, (1, 4, 24))
        for horizons in ([24], [1, 4, 24, 48], [1, 1, 24], [True, 4, 24], [1.0, 4, 24]):
            with self.subTest(horizons=horizons), self.assertRaisesRegex(ValueError, "b_horizons"):
                ResearchSpec.from_dict({**self.contract, "b_horizons": horizons})

    def test_legacy_correction_and_admission_rules_fail_at_entry(self):
        for override in (
            {"fdr_method": "BY"},
            {"admission_scheme": "baseline"},
            {"admission_scheme": "plan2"},
            {"plan3_tracks_gate": True},
        ):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "FM-v6 requires"):
                ResearchSpec.from_dict({**self.contract, **override})

    def test_batch_uses_only_rank_ic_bh_and_keeps_unavailable_tests(self):
        reports = {
            "a": {"summary": {"rank_ic": {"p_value": 0.01},
                              "directional_spread": {"p_value": 0.000001}}},
            "b": {"summary": {"rank_ic": {"p_value": 0.03},
                              "directional_spread": {"p_value": 0.04}}},
            "c": {"summary": {"rank_ic": {"p_value": None},
                              "directional_spread": {"p_value": 0.000001}}},
        }
        reports = {cid: {"horizons": {"24": {"horizon_hours": 24, **report}}}
                   for cid, report in reports.items()}
        correction = correct_batch(reports, ResearchSpec.from_dict(self.contract))
        self.assertEqual(correction["method"], "BH")
        self.assertEqual(correction["family_size"], 3)
        self.assertEqual([test["metric"] for test in correction["tests"]], ["rank_ic"] * 3)
        for test, expected in zip(correction["tests"], (0.03, 0.045, 1.0)):
            self.assertAlmostEqual(test["adjusted_p"], expected)
        self.assertIsNone(correction["tests"][-1]["raw_p"])
        self.assertFalse(correction["tests"][-1]["rejected"])


if __name__ == "__main__":
    unittest.main()
