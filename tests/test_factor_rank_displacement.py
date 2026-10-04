"""Contract tests for exact-hour rank displacement and paired optimization evidence."""

import numpy as np
import pandas as pd

from crypto_quant.features.factor_inputs import FactorInputPanel
from crypto_quant.research.factor_mining.contracts import ResearchSpec
from crypto_quant.research.factor_mining.evaluation import (
    RANK_DISPLACEMENT_VERSION,
    build_labels,
    compare_rank_displacement_experiment,
    evaluate_rank_displacement,
)


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]


def make_spec(hours=48, *, sample_hours=1, min_periods=None, hac_lags=None, min_symbols=4, groups=2):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    b_start = start + pd.Timedelta(hours=hours)
    c_start = b_start + pd.Timedelta(hours=hours)
    c_end = c_start + pd.Timedelta(hours=hours)
    required_lags = 24 // sample_hours - 1
    hac_lags = required_lags if hac_lags is None else hac_lags
    min_periods = hac_lags + 2 if min_periods is None else min_periods
    return ResearchSpec(
        run_id="rank-displacement-fixture",
        objective="verify deterministic rank displacement definition",
        purpose="engineering_check",
        a_start=start.isoformat(), b_start=b_start.isoformat(),
        c_start=c_start.isoformat(), c_end=c_end.isoformat(),
        universe_provenance="synthetic full hourly universe",
        data_usage_review="synthetic engineering fixture",
        label="perp_next_open_24h", sample_hours=sample_hours, groups=groups,
        min_symbols=min_symbols, min_periods=min_periods, hac_lags=hac_lags,
        confidence=0.95, stage_hours=24, rolling_periods=2, fdr_method="BH",
        fdr_alpha=0.05, min_abs_ic=0.01, min_directional_spread=0.0,
        min_stage_share=0.5, max_repairs=1, max_formula_nodes=20,
        max_lookback_hours=24, context_tokens=1000, output_tokens=100,
        b_horizons=[1, 4, 24], admission_scheme="plan3", plan3_tracks_gate=False,
    )


def panel(spec, symbols=SYMBOLS):
    start, end = spec.bounds("A")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    index = pd.MultiIndex.from_product([hours, symbols], names=["timestamp", "symbol"])
    universe = pd.Series(True, index=index, dtype=bool)
    return hours, index, universe


def series_from_matrix(index, matrix):
    return pd.Series(np.asarray(matrix, dtype=float).reshape(-1), index=index, dtype=float)


def period(report, delta, timestamp):
    return next(row for row in report["deltas"][str(delta)]["periods"]
                if row["timestamp"] == timestamp.isoformat())


def make_labels(index, horizon, values):
    timestamps = index.get_level_values("timestamp")
    labels = pd.DataFrame(index=index)
    labels["label_start"] = timestamps + pd.Timedelta(hours=1)
    labels["label_end"] = timestamps + pd.Timedelta(hours=horizon + 1)
    labels["eligible"] = True
    labels["purged"] = False
    labels["forward_return"] = np.asarray(values, dtype=float)
    return labels


def rank_plan(*, horizon=1, delta=1, min_improvement=0.02, max_ic_loss=0.1):
    return {"experiment_design": {
        "metric": "rank_displacement", "horizon_hours": horizon,
        "displacement_hours": delta, "min_improvement": min_improvement,
        "max_ic_loss": max_ic_loss,
    }}


def test_stable_order_and_affine_or_frozen_direction_reversal_do_not_change_d():
    spec = make_spec(hours=30)
    _, index, universe = panel(spec, SYMBOLS[:4])
    base = np.tile(np.arange(4, dtype=float), (30, 1))
    source = series_from_matrix(index, base)

    report = evaluate_rank_displacement(source, universe, spec, "A")
    scaled = evaluate_rank_displacement(series_from_matrix(index, base * 7 + 11), universe, spec, "A")
    reversed_direction = evaluate_rank_displacement(series_from_matrix(index, -base), universe, spec, "A")

    assert report["definition_version"] == RANK_DISPLACEMENT_VERSION
    for delta in (1, 4, 24):
        assert report["deltas"][str(delta)]["summary"]["mean"] == 0.0
        assert report["deltas"][str(delta)]["summary"]["mean"] == scaled["deltas"][str(delta)]["summary"]["mean"]
        assert report["deltas"][str(delta)]["summary"]["mean"] == reversed_direction["deltas"][str(delta)]["summary"]["mean"]


def test_hand_calculated_reversal_is_half_and_boundaries_are_excluded():
    spec = make_spec(hours=30)
    hours, index, universe = panel(spec, SYMBOLS[:4])
    values = np.tile(np.arange(4, dtype=float), (len(hours), 1))
    values[1] = values[1, ::-1]

    report = evaluate_rank_displacement(series_from_matrix(index, values), universe, spec, "A")
    row = period(report, 1, hours[1])

    # For four symbols, reversal moves normalized ranks by .75, .25, .25, .75.
    assert row["displacement"] == 0.5
    assert report["deltas"]["1"]["coverage"]["boundary_excluded_periods"] == 1
    assert report["deltas"]["4"]["coverage"]["boundary_excluded_periods"] == 4


def test_membership_changes_reduce_common_coverage_without_creating_rank_motion():
    spec = make_spec(hours=30, min_symbols=4)
    hours, index, universe = panel(spec)
    eligible = universe.copy()
    eligible.loc[(hours[0], SYMBOLS[4])] = False
    eligible.loc[(hours[0], SYMBOLS[5])] = False
    values = np.tile(np.arange(6, dtype=float), (len(hours), 1))

    report = evaluate_rank_displacement(series_from_matrix(index, values), eligible, spec, "A")
    row = period(report, 1, hours[1])

    assert row["displacement"] == 0.0
    assert row["common_symbols"] == 4
    assert row["common_coverage_ratio"] == 4 / 6
    assert row["eligible_t"] == 6
    assert row["eligible_t_minus_delta"] == 4


def test_missing_factor_hour_is_not_compressed_and_labels_do_not_filter_independent_d():
    spec = make_spec(hours=30)
    hours, index, universe = panel(spec, SYMBOLS[:4])
    values = np.tile(np.arange(4, dtype=float), (len(hours), 1))
    values[2, :] = np.nan
    report = evaluate_rank_displacement(series_from_matrix(index, values), universe, spec, "A")
    row_after_missing = period(report, 1, hours[3])

    assert row_after_missing["displacement"] is None
    assert row_after_missing["status"] == "insufficient_common_symbols"
    assert period(report, 1, hours[4])["displacement"] == 0.0


def test_nonhour_sample_grid_uses_exact_hour_lookup_for_delta():
    spec = make_spec(hours=48, sample_hours=4, hac_lags=5, min_periods=7)
    hours, index, universe = panel(spec, SYMBOLS[:4])
    values = np.tile(np.arange(4, dtype=float), (len(hours), 1))
    values[3, :] = values[3, ::-1]

    report = evaluate_rank_displacement(series_from_matrix(index, values), universe, spec, "A")
    at_hour_four_delta_one = period(report, 1, hours[4])
    at_hour_four_delta_four = period(report, 4, hours[4])

    assert report["sample_hours"] == 4
    assert report["deltas"]["1"]["summary"]["expected_periods"] == 12
    assert at_hour_four_delta_one["displacement"] == 0.5
    assert at_hour_four_delta_four["displacement"] == 0.0


def test_pair_uses_label_common_symbols_and_hac_full_sample_grid():
    spec = make_spec(hours=120, min_symbols=6)
    hours, index, universe = panel(spec)
    rng = np.random.default_rng(92)
    trial_matrix = np.tile(np.arange(6, dtype=float), (len(hours), 1))
    control_matrix = rng.normal(size=(len(hours), 6))
    labels = make_labels(index, 1, np.tile(np.arange(6, dtype=float), len(hours)))
    labels.loc[(hours[50], SYMBOLS[0]), "forward_return"] = np.nan
    for timestamp in hours[-2:]:
        for symbol in SYMBOLS:
            labels.loc[(timestamp, symbol), "purged"] = False
    trial = series_from_matrix(index, trial_matrix)
    control = series_from_matrix(index, control_matrix)

    result = compare_rank_displacement_experiment(
        trial, control, labels, spec, 1, rank_plan(), universe=universe)

    assert result["decision"] == "continue"
    assert result["paired_improvement"]["grid_periods"] == 120
    assert result["paired_improvement"]["n"] == 116
    assert result["paired_ic_change"]["grid_periods"] == 120
    assert result["paired_ic_change"]["n"] == result["paired_improvement"]["n"]
    assert result["coverage"]["paired_valid_periods"] == 116
    assert result["paired_periods"][50]["common_symbols"] == 5
    assert all((row["paired_improvement"] is None) == (row["paired_ic_change"] is None)
               for row in result["paired_periods"])
    assert result["paired_periods"][-2]["status"] == "label_boundary_excluded"
    assert result["paired_periods"][-1]["status"] == "label_boundary_excluded"


def test_pair_counts_real_build_labels_horizon_boundary_separately_from_delta_boundary():
    spec = make_spec(hours=120, min_symbols=6)
    hours, index, universe = panel(spec, sorted(SYMBOLS))
    levels = np.arange(1, 7, dtype=float)
    time_scale = np.arange(len(hours), dtype=float)[:, None] + 1
    prices = np.exp(0.001 * time_scale * levels[None, :])
    panel_values = pd.DataFrame({"perp_open": prices.reshape(-1)}, index=index)
    labels = build_labels(FactorInputPanel(panel_values, universe, {}), spec, "A", horizon_hours=1)
    factor = series_from_matrix(index, np.tile(np.arange(6, dtype=float), (len(hours), 1)))

    result = compare_rank_displacement_experiment(
        factor, factor, labels, spec, 1, rank_plan(), universe=universe)

    assert result["paired_periods"][0]["status"] == "boundary_excluded"
    assert result["paired_periods"][-2]["status"] == "label_boundary_excluded"
    assert result["paired_periods"][-1]["status"] == "label_boundary_excluded"
    assert result["paired_periods"][-2]["label_boundary_excluded_symbols"] == 6
    assert result["coverage"]["boundary_excluded_periods"] == 1
    assert result["coverage"]["label_boundary_excluded_periods"] == 2


def test_constant_cross_section_cannot_receive_success_from_zero_d():
    spec = make_spec(hours=120, min_symbols=6)
    hours, index, universe = panel(spec)
    constant = series_from_matrix(index, np.ones((len(hours), 6)))
    returns = np.tile(np.arange(6, dtype=float), len(hours))
    labels = make_labels(index, 1, returns)

    result = compare_rank_displacement_experiment(
        constant, constant, labels, spec, 1, rank_plan(), universe=universe)

    assert result["decision"] == "pause_insufficient"
    assert result["coverage"]["rank_ic_unavailable_periods"] > 0
    assert result["paired_improvement"]["status"] == "insufficient_evidence"
    assert result["paired_ic_change"]["status"] == "insufficient_evidence"


def test_d_upper_bound_can_stop_when_common_ic_change_is_degenerate():
    spec = make_spec(hours=120, min_symbols=4)
    hours, index, universe = panel(spec, SYMBOLS[:4])
    trial_matrix = np.tile(np.arange(4, dtype=float), (len(hours), 1))
    states = np.random.default_rng(21).integers(0, 2, len(hours))
    alternatives = np.array([[0, 2, 1, 3], [0, 1, 3, 2]], dtype=float)
    control_matrix = alternatives[states]
    labels = make_labels(index, 1, np.tile(np.arange(4, dtype=float), len(hours)))

    result = compare_rank_displacement_experiment(
        series_from_matrix(index, trial_matrix),
        series_from_matrix(index, control_matrix), labels, spec, 1,
        rank_plan(min_improvement=0.5), universe=universe)

    assert result["decision"] == "stop"
    assert result["paired_improvement"]["status"] == "estimated"
    assert result["paired_improvement"]["ci"][1] < 0.5
    assert result["paired_ic_change"]["status"] == "degenerate_variance"
    assert result["paired_improvement"]["n"] == result["paired_ic_change"]["n"]
    assert result["paired_improvement"]["grid_periods"] == result["paired_ic_change"]["grid_periods"]


def test_tied_factor_ic_requires_the_existing_group_count_of_distinct_values():
    spec = make_spec(hours=120, min_symbols=6, groups=3)
    hours, index, universe = panel(spec)
    two_levels = np.tile(np.array([0, 0, 0, 1, 1, 1], dtype=float), (len(hours), 1))
    labels = make_labels(index, 1, np.tile(np.arange(6, dtype=float), len(hours)))

    result = compare_rank_displacement_experiment(
        series_from_matrix(index, two_levels), series_from_matrix(index, two_levels),
        labels, spec, 1, rank_plan(), universe=universe)

    assert result["decision"] == "pause_insufficient"
    assert result["coverage"]["rank_ic_unavailable_periods"] > 0
    assert result["paired_improvement"]["n"] == result["paired_ic_change"]["n"] == 0
    assert result["paired_periods"][5]["unique_trial_t"] == 2
    assert result["paired_periods"][5]["rank_ic_trial"] is None


def test_removing_volatile_symbols_cannot_create_paired_d_improvement():
    spec = make_spec(hours=120, min_symbols=4)
    hours, index, universe = panel(spec)
    control_matrix = np.tile(np.arange(6, dtype=float), (len(hours), 1))
    control_matrix[1::2, 4] = 2.5
    control_matrix[1::2, 5] = 1.5
    trial_matrix = control_matrix.copy()
    trial_matrix[:, 4:] = np.nan
    labels = make_labels(index, 1, np.tile(np.arange(6, dtype=float), len(hours)))

    control = series_from_matrix(index, control_matrix)
    trial = series_from_matrix(index, trial_matrix)
    control_d = evaluate_rank_displacement(control, universe, spec, "A")["deltas"]["1"]["summary"]
    trial_report = evaluate_rank_displacement(trial, universe, spec, "A")["deltas"]["1"]
    paired = compare_rank_displacement_experiment(
        trial, control, labels, spec, 1, rank_plan(), universe=universe)

    assert control_d["mean"] > trial_report["summary"]["mean"] == 0.0
    assert period({"deltas": {"1": trial_report}}, 1, hours[5])["common_coverage_ratio"] == 4 / 6
    assert paired["paired_periods"][5]["common_symbols"] == 4
    assert paired["paired_periods"][5]["common_coverage_ratio"] == 4 / 6
    assert paired["paired_improvement"]["mean"] == 0.0
    assert paired["decision"] == "pause_insufficient"


def test_lower_d_with_directional_ic_loss_stops_the_route():
    spec = make_spec(hours=120, min_symbols=6)
    hours, index, universe = panel(spec)
    rng = np.random.default_rng(92)
    trial_matrix = np.tile(np.arange(6, dtype=float), (len(hours), 1))
    control_matrix = rng.normal(size=(len(hours), 6))
    labels = make_labels(index, 1, control_matrix.reshape(-1))

    result = compare_rank_displacement_experiment(
        series_from_matrix(index, trial_matrix), series_from_matrix(index, control_matrix),
        labels, spec, 1, rank_plan(min_improvement=0.02, max_ic_loss=0.05), universe=universe)

    assert result["paired_improvement"]["ci"][0] >= 0.02
    assert result["paired_ic_change"]["ci"][1] < -0.05
    assert result["decision"] == "stop"
