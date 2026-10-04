"""Synthetic numerical parity checks for the E2/E3/E4 study helper."""

import numpy as np
import pandas as pd

from crypto_quant.features.factor_inputs import FactorInputPanel
from crypto_quant.research.factor_mining.contracts import ResearchSpec
from crypto_quant.research.factor_mining.evaluation import (
    build_labels,
    compare_rank_displacement_experiment,
    evaluate_factor,
)
from scripts.factor_rank_displacement_study import compute_factor_variant_study


SYMBOLS = sorted(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"])


def make_spec(hours=48, *, sample_hours=1, stage_hours=24):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    b_start = start + pd.Timedelta(hours=hours)
    c_start = b_start + pd.Timedelta(hours=hours)
    c_end = c_start + pd.Timedelta(hours=hours)
    hac_lags = 24 // sample_hours - 1
    return ResearchSpec(
        run_id="study-helper-fixture", objective="verify common-support calculations",
        purpose="engineering_check", a_start=start.isoformat(), b_start=b_start.isoformat(),
        c_start=c_start.isoformat(), c_end=c_end.isoformat(),
        universe_provenance="synthetic full hourly universe",
        data_usage_review="synthetic engineering fixture", label="perp_next_open_24h",
        sample_hours=sample_hours, groups=3, min_symbols=6, min_periods=hac_lags + 2, hac_lags=hac_lags,
        confidence=0.95, stage_hours=stage_hours, rolling_periods=2, fdr_method="BH",
        fdr_alpha=0.05, min_abs_ic=0.01, min_directional_spread=0,
        min_stage_share=0.5, max_repairs=1, max_formula_nodes=20,
        max_lookback_hours=24, context_tokens=1000, output_tokens=100,
        b_horizons=[1, 4, 24], admission_scheme="plan3", plan3_tracks_gate=False,
    )


def fixture(hours=48):
    spec = make_spec(hours)
    start, end = spec.bounds("A")
    times = pd.date_range(start, end, freq="h", inclusive="left")
    index = pd.MultiIndex.from_product([times, SYMBOLS], names=["timestamp", "symbol"])
    universe = pd.Series(True, index=index, dtype=bool)
    rng = np.random.default_rng(311)
    reference = rng.normal(size=(len(times), len(SYMBOLS)))
    trial = np.empty_like(reference)
    trial[0] = reference[0]
    for row in range(1, len(times)):
        trial[row] = 0.75 * reference[row] + 0.25 * reference[row - 1]
    tied = reference.copy()
    tied[6] = np.array([0, 0, 0, 0, 1, 2], dtype=float)
    tied[7] = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    tied[8] = np.arange(6, dtype=float)
    tied[10, 5] = np.nan
    prices = np.exp(0.001 * (np.arange(len(times), dtype=float)[:, None] + 1)
                    * np.arange(1, len(SYMBOLS) + 1, dtype=float)[None, :])
    panel_values = pd.DataFrame({"perp_open": prices.reshape(-1)}, index=index)
    panel = FactorInputPanel(panel_values, universe, {})
    labels = {h: build_labels(panel, spec, "A", horizon_hours=h) for h in (1, 4, 24)}
    values = {
        "reference": pd.Series(reference.reshape(-1), index=index, dtype=float),
        "trial": pd.Series(trial.reshape(-1), index=index, dtype=float),
        "tied": pd.Series(tied.reshape(-1), index=index, dtype=float),
    }
    return spec, times, index, universe, panel, values, labels


def design(horizon=1, delta=1):
    return {"experiment_design": {
        "metric": "rank_displacement", "horizon_hours": horizon,
        "displacement_hours": delta, "min_improvement": 0.01,
        "max_ic_loss": 0.05,
    }}


def record_at(records, timestamp):
    return next(row for row in records if row["timestamp"] == timestamp.isoformat())


def test_common_metrics_and_pair_effects_match_production_evaluators_hour_by_hour():
    spec, times, index, universe, panel, values, labels = fixture()
    study = compute_factor_variant_study(
        values, universe, labels, spec, 1, reference_id="reference")
    h1 = study["pair_effects"]["trial"]["horizons"]["1"]["1"]
    production_pair = compare_rank_displacement_experiment(
        values["trial"], values["reference"], labels[1], spec, 1, design(), universe=universe)
    production_reports = {
        version_id: evaluate_factor(series, labels[1], spec, "A", 1, horizon_hours=1)
        for version_id, series in values.items()
    }

    assert len(h1["periods"]) == len(times)
    assert len(h1["stages"]) == 2
    for i, timestamp in enumerate(times):
        study_row = h1["periods"][i]
        pair_row = production_pair["paired_periods"][i]
        assert study_row["timestamp"] == pair_row["timestamp"]
        if study_row["valid_d_ic_pair"]:
            assert np.isclose(study_row["paired_displacement_improvement"], pair_row["paired_improvement"])
            assert np.isclose(study_row["paired_directional_rank_ic_change"], pair_row["paired_ic_change"])
            for version_id, field in (("reference", "reference"), ("trial", "trial")):
                expected = record_at(production_reports[version_id]["periods"], timestamp)
                assert np.isclose(study_row[f"rank_ic_{field}"], expected["rank_ic"])
        if study_row["valid_spread_pair"]:
            for version_id, field in (("reference", "reference"), ("trial", "trial")):
                expected = record_at(production_reports[version_id]["periods"], timestamp)
                assert np.isclose(study_row[f"directional_spread_{field}"], expected["directional_spread"])
        if timestamp == times[6]:
            tied_row = record_at(production_reports["tied"]["periods"], timestamp)
            assert study_row["valid_d_ic_pair"] is True
            assert tied_row["status"] == "empty_group_due_to_ties"
            assert tied_row["rank_ic"] is not None
            assert tied_row["directional_spread"] is None
            assert study["versions"]["tied"]["horizons"]["1"]["1"]["coverage"][
                "directional_spread_unavailable_periods"] > 0
        if timestamp == times[7]:
            assert study_row["valid_d_ic_pair"] is False
            assert study_row["common_symbols"] == 6
        if timestamp in (times[10], times[11]):
            assert study_row["common_symbols"] == 5
            assert study_row["valid_d_ic_pair"] is False

    reference_ic = study["versions"]["reference"]["horizons"]["1"]["1"]["summary"]["rank_ic"]
    expected_ic = production_reports["reference"]["summary"]["rank_ic"]
    assert reference_ic["n"] <= expected_ic["n"]
    assert study["versions"]["tied"]["horizons"]["1"]["1"]["coverage"]["rank_ic_unavailable_periods"] > 0
    assert study["native_coverage"]["tied"]["1"]["expected_periods"] == len(times)
    native_reference = study["native_prediction_coverage"]["reference"]["1"]
    native_tied = study["native_prediction_coverage"]["tied"]["1"]
    assert native_tied["factor_label_common_observations"] < native_reference["factor_label_common_observations"]
    paired_n = study["pair_effects"]["trial"]["horizons"]["1"]["1"]["coverage"]["paired_d_ic_valid_periods"]
    tied_pair_n = study["pair_effects"]["tied"]["horizons"]["1"]["1"]["coverage"]["paired_d_ic_valid_periods"]
    assert paired_n == tied_pair_n
    trial_pair = study["pair_effects"]["trial"]["horizons"]["1"]["1"]
    assert trial_pair["reference_on_pair_support"]["summary"]["rank_displacement"]["n"] == paired_n
    assert trial_pair["trial_on_pair_support"]["summary"]["rank_displacement"]["n"] == paired_n
    assert "tie periods" in study["versions"]["reference"]["horizons"]["1"]["1"]["coverage"]["summary_support"]


def test_subsegment_excludes_both_delta_start_and_horizon_end_boundaries():
    spec, times, index, universe, _, values, labels = fixture()
    start, _ = spec.bounds("A")
    front_start = start + pd.Timedelta(hours=24)
    front_end = front_start + pd.Timedelta(hours=24)
    study = compute_factor_variant_study(
        values, universe, labels, spec, 1, reference_id="reference",
        segment=(front_start, front_end))
    periods = study["pair_effects"]["trial"]["horizons"]["1"]["1"]["periods"]

    assert len(periods) == 24
    assert periods[0]["status"] == "boundary_excluded"
    assert periods[-2]["status"] == "label_boundary_excluded"
    assert periods[-1]["status"] == "label_boundary_excluded"
    assert study["pair_effects"]["trial"]["horizons"]["1"]["1"]["coverage"]["boundary_excluded_periods"] == 1
    assert study["pair_effects"]["trial"]["horizons"]["1"]["1"]["coverage"]["label_boundary_excluded_periods"] == 2


def test_single_executable_version_returns_evidence_without_inventing_a_pair():
    spec, _, _, universe, _, values, labels = fixture()
    study = compute_factor_variant_study(
        {"reference": values["reference"]}, universe, labels, spec, 1,
        reference_id="reference")

    assert study["pair_effects"] == {}
    assert set(study["versions"]["reference"]["horizons"]) == {"1", "4", "24"}
    for horizon in ("1", "4", "24"):
        assert set(study["versions"]["reference"]["horizons"][horizon]) == {"1", "4", "24"}


def test_study_uses_sample_grid_but_looks_up_delta_on_exact_hourly_rows():
    spec = make_spec(hours=48, sample_hours=4)
    start, end = spec.bounds("A")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    index = pd.MultiIndex.from_product([hours, SYMBOLS], names=["timestamp", "symbol"])
    universe = pd.Series(True, index=index, dtype=bool)
    values = np.tile(np.arange(6, dtype=float), (len(hours), 1))
    values[3] = values[3, ::-1]
    series = pd.Series(values.reshape(-1), index=index, dtype=float)
    prices = np.exp(0.001 * (np.arange(len(hours), dtype=float)[:, None] + 1)
                    * np.arange(1, 7, dtype=float)[None, :])
    labels = {h: build_labels(
        FactorInputPanel(pd.DataFrame({"perp_open": prices.reshape(-1)}, index=index), universe, {}),
        spec, "A", horizon_hours=h) for h in (1, 4, 24)}
    study = compute_factor_variant_study(
        {"reference": series, "trial": series}, universe, labels, spec, 1,
        reference_id="reference")
    delta_one = study["pair_effects"]["trial"]["horizons"]["1"]["1"]["periods"]
    delta_four = study["pair_effects"]["trial"]["horizons"]["1"]["4"]["periods"]

    assert len(delta_one) == len(delta_four) == 12
    assert delta_one[1]["timestamp"] == hours[4].isoformat()
    assert delta_one[1]["displacement_reference"] == 0.5
    assert delta_four[1]["displacement_reference"] == 0.0


def test_subsegment_stage_summaries_keep_full_a_week_bucket_alignment():
    spec = make_spec(hours=17520, stage_hours=168)
    start, end = spec.bounds("A")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    index = pd.MultiIndex.from_product([hours, SYMBOLS], names=["timestamp", "symbol"])
    universe = pd.Series(True, index=index, dtype=bool)
    matrix = np.tile(np.arange(6, dtype=float), (len(hours), 1))
    values = pd.Series(matrix.reshape(-1), index=index, dtype=float)
    prices = np.exp(0.001 * (np.arange(len(hours), dtype=float)[:, None] + 1)
                    * np.arange(1, 7, dtype=float)[None, :])
    labels = {h: build_labels(
        FactorInputPanel(pd.DataFrame({"perp_open": prices.reshape(-1)}, index=index), universe, {}),
        spec, "A", horizon_hours=h) for h in (1, 4, 24)}
    segment_start = start + pd.Timedelta(hours=8760)
    study = compute_factor_variant_study(
        {"reference": values}, universe, labels, spec, 1, reference_id="reference",
        segment=(segment_start, end))
    stages = study["versions"]["reference"]["horizons"]["1"]["1"]["stages"]

    assert stages[0]["stage_index"] == 52
    assert stages[0]["start"] == segment_start.isoformat()
    assert stages[0]["end"] == (start + pd.Timedelta(hours=8904)).isoformat()
    assert stages[-1]["stage_index"] == 104
    assert stages[-1]["start"] == (start + pd.Timedelta(hours=17472)).isoformat()
    assert stages[-1]["end"] == end.isoformat()
