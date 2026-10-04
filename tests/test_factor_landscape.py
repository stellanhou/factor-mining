"""A-only HAC landscape contract tests."""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.features.factor_expressions import compile_expression
from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS
from crypto_quant.research.factor_mining import landscape
from crypto_quant.research.factor_mining.landscape import _pairwise_pearson, build_factor_landscape
from test_factor_mining import specification


def landscape_panel(*, sample_hours=1, append_b=False):
    spec = specification(sample_hours=sample_hours, min_periods=26, max_lookback_hours=4)
    start, end = spec.bounds("A")
    stop = end if append_b else end - pd.Timedelta(hours=1)
    times = pd.date_range(start - pd.Timedelta(hours=4), stop, freq="h")
    symbols = sorted(["BTCUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"])
    index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    values = pd.DataFrame(1.0, index=index, columns=INPUT_COLUMNS)
    t = np.arange(len(times), dtype=float)[:, None]
    asset = np.arange(1, len(symbols) + 1, dtype=float)[None, :]
    spot = asset * (1 + 0.0007 * t) + 0.025 * np.sin(t / 4 + asset * 1.7)
    perp = asset * (1 + 0.0006 * t) + 0.04 * np.cos(t / 5 + asset * 1.3)
    values["spot_close"] = spot.reshape(-1)
    values["perp_close"] = perp.reshape(-1)
    values["premium_index"] = (np.sin(t / 7 + asset) * 0.001).reshape(-1)
    universe = pd.Series(True, index=index)
    nonmember = index.get_level_values("symbol") == "ADAUSDT"
    universe.loc[nonmember] = False
    # This outlier must never enter formula outputs or pairwise correlations.
    values.loc[nonmember, "spot_close"] = 1e12
    return spec, FactorInputPanel(values, universe, {"fixture": True})


def member(ref, expression, *, status="computed", executed=True, evidence=None, a_evaluation=None):
    compiled = compile_expression(expression)
    expression_snapshot = compiled.description() if executed else None
    calculation = None if status is None else {
        "status": status,
        "executed_expression": expression_snapshot if status == "computed" and executed else None,
    }
    result = {
        "candidate_ref": ref,
        "definition": {"expression": expression, "meaning": "test-only formula description"},
        "calculation": calculation,
        "A_evaluation": a_evaluation,
        "final_decision": None,
        "evidence_refs": evidence or [],
    }
    return result


def test_hac_uses_mean_hourly_cross_sectional_pearson_and_keeps_window_variants():
    spec, panel = landscape_panel()
    result = build_factor_landscape(panel, spec, [
        member("run-a/spot", "spot_close"),
        member("run-a/reversed", "neg(spot_close)"),
        member("run-a/mean2", "ts_mean(spot_close,2)"),
        member("run-a/mean4", "ts_mean(spot_close,4)"),
    ])

    assert result["status"] == "computed"
    assert result["contract"]["data_segment"] == "A"
    assert result["contract"]["sampled_hours"] == 96
    assert result["tree"]["method"] == "average"
    assert len(result["tree"]["scipy_linkage"]) == 3
    assert {row["candidate_ref"] for row in result["members"]
            if row["inventory_status"] == "eligible_for_similarity"} == {
        "run-a/spot", "run-a/reversed", "run-a/mean2", "run-a/mean4",
    }
    reverse_pair = next(pair for pair in result["pairwise_correlations"]
                        if {pair["left_ref"], pair["right_ref"]} == {"run-a/spot", "run-a/reversed"})
    assert reverse_pair["mean_hourly_cross_sectional_pearson"] == pytest.approx(-1.0)
    assert reverse_pair["distance_1_minus_abs_mean_pearson"] == pytest.approx(0.0)
    assert reverse_pair["valid_periods"] == 96
    for merge in result["tree"]["merges"]:
        assert merge["member_count"] >= 2
        assert merge["minimum_pair_valid_periods"] == 96
        assert merge["similarity_1_minus_height"] == pytest.approx(1 - merge["height"])
    expressions = [row["formula"]["expanded_expression"] for row in result["members"]]
    assert "ts_mean(spot_close, 2)" in expressions
    assert "ts_mean(spot_close, 4)" in expressions


def test_exact_duplicates_fold_sources_but_keep_inventory_and_source_scope():
    spec, panel = landscape_panel()
    a_summary = {"summary": {"rank_ic": {"mean": 0.7}}, "coverage": {"source_run": "old"}}
    result = build_factor_landscape(panel, spec, [
        member("run-1/candidate", "spot_close", evidence=["cycle-1"], a_evaluation=a_summary),
        member("run-2/candidate", "spot_close", evidence=["cycle-2"]),
        member("run-2/reverse", "neg(spot_close)"),
    ])

    alias = next(row for row in result["members"] if row["candidate_ref"] == "run-2/candidate")
    assert alias["inventory_status"] == "duplicate_formula"
    assert alias["duplicate_of"] == "run-1/candidate"
    canonical = next(row for row in result["members"] if row["candidate_ref"] == "run-1/candidate")
    assert canonical["source_members"] == ["run-1/candidate", "run-2/candidate"]
    assert canonical["source_evidence_refs"] == ["cycle-1", "cycle-2"]
    assert canonical["source_A_evaluation"] == a_summary
    assert result["tree"]["leaf_order"] == ["run-1/candidate", "run-2/reverse"]
    assert "not pooled" in result["contract"]["source_metric_scope"]


def test_failed_unattempted_constant_and_sparse_members_remain_visible():
    spec, panel = landscape_panel()
    failed = member("run-a/failed", "perp_close", status="calculation_failed", executed=False)
    failed["definition"]["expression"] = "future_return"
    parsed_failure = member("run-a/parsed-failure", "open_interest_value",
                            status="calculation_failed", executed=False)
    pre_duplicate = member("run-a/pre-duplicate", "spot_close", status="duplicate", executed=False)
    pre_duplicate["calculation"]["duplicate_of"] = "candidate-0001"
    start, _ = spec.bounds("A")
    a_times = panel.values.index.get_level_values("timestamp")
    sparse = (a_times >= start) & (a_times < start + pd.Timedelta(hours=10))
    panel.values.loc[(~sparse) & (a_times >= start), "open_interest_value"] = np.nan
    a_rows = a_times >= start
    time_values = np.repeat(np.arange(int(a_rows.sum() / 7), dtype=float), 7)
    panel.values.loc[a_rows, "perp_close"] = time_values
    panel.values.loc[a_rows, "premium_index"] = 0.1
    t = np.arange(len(panel.values.index.get_level_values("timestamp").unique()), dtype=float)[:, None]
    asset = np.arange(1, 8, dtype=float)[None, :]
    panel.values["open_interest_base"] = (asset * (1 + 0.001 * t)).reshape(-1)
    symbols = panel.values.index.get_level_values("symbol")
    panel.values.loc[a_rows & symbols.isin(["SOLUSDT", "XRPUSDT"]), "open_interest_base"] = np.nan
    result = build_factor_landscape(panel, spec, [
        member("run-a/good", "spot_close"),
        member("run-a/constant", "1"),
        member("run-a/time-constant", "perp_close"),
        member("run-a/floating-constant", "premium_index"),
        member("run-a/sparse", "open_interest_value"),
        member("run-a/too-few-symbols", "open_interest_base"),
        failed,
        parsed_failure,
        pre_duplicate,
        member("run-a/unattempted", "perp_close", status=None),
    ])

    by_ref = {row["candidate_ref"]: row for row in result["members"]}
    assert by_ref["run-a/constant"]["inventory_status"] == "constant_factor"
    assert by_ref["run-a/time-constant"]["inventory_status"] == "constant_factor"
    assert by_ref["run-a/time-constant"]["current_A_value_coverage"]["hours_with_cross_sectional_variation"] == 0
    assert by_ref["run-a/floating-constant"]["inventory_status"] == "constant_factor"
    assert by_ref["run-a/sparse"]["inventory_status"] == "insufficient_A_formula_coverage"
    assert by_ref["run-a/sparse"]["current_A_value_coverage"]["hours_with_minimum_cross_section"] == 10
    assert by_ref["run-a/too-few-symbols"]["inventory_status"] == "insufficient_A_formula_coverage"
    assert by_ref["run-a/too-few-symbols"]["current_A_value_coverage"][
        "hours_with_minimum_cross_section"] == 0
    assert by_ref["run-a/too-few-symbols"]["current_A_value_coverage"]["finite_eligible_rows"] == 4 * 96
    assert by_ref["run-a/failed"]["inventory_status"] == "calculation_failed"
    assert by_ref["run-a/failed"]["formula"]["executed_expression"] is None
    assert by_ref["run-a/parsed-failure"]["inventory_status"] == "calculation_failed"
    assert by_ref["run-a/parsed-failure"]["current_A_input_field_data_coverage"][
        "open_interest_value"]["finite_eligible_rows"] == 10 * 6
    assert by_ref["run-a/pre-duplicate"]["inventory_status"] == "duplicate_candidate"
    assert by_ref["run-a/pre-duplicate"]["calculation_status"] == "duplicate"
    assert by_ref["run-a/pre-duplicate"]["duplicate_of"] == "candidate-0001"
    assert by_ref["run-a/unattempted"]["inventory_status"] == "not_calculated"
    assert by_ref["run-a/failed"]["current_A_input_field_data_coverage"] == {}
    assert result["field_inventory"]["field_use_counts"]["open_interest_value"] == 2
    assert result["inventory_summary"]["members_by_status"]["duplicate_candidate"] == 1
    assert result["inventory_summary"]["members_by_status"]["calculation_failed"] == 2
    assert result["status"] == "single_factor_no_tree"
    assert result["tree"] is None
    no_usable = build_factor_landscape(panel, spec, [
        member("run-a/constant-only", "1"),
        member("run-a/time-constant-only", "perp_close"),
    ])
    assert no_usable["status"] == "no_clusterable_members"


def test_constant_cross_sections_do_not_create_roundoff_correlations():
    result = _pairwise_pearson(
        np.full((30, 6), 0.1), np.full((30, 6), 0.2), np.ones((30, 6), dtype=bool), 3, 26,
    )
    assert result["valid_periods"] == 0
    assert result["mean"] is None
    assert result["distance"] is None


def test_missing_pair_coverage_is_reported_without_filling_distance():
    spec, panel = landscape_panel()
    start, _ = spec.bounds("A")
    times = panel.values.index.get_level_values("timestamp")
    panel.values.loc[(times >= start + pd.Timedelta(hours=40)) & (times < start + pd.Timedelta(hours=96)),
                     "perp_close"] = np.nan
    panel.values.loc[(times >= start) & (times < start + pd.Timedelta(hours=50)), "spot_close"] = np.nan

    result = build_factor_landscape(panel, spec, [
        member("run-a/perp", "perp_close"),
        member("run-a/spot", "spot_close"),
    ])

    pair = result["pairwise_correlations"][0]
    assert result["status"] == "insufficient_pair_coverage"
    assert result["tree"] is None
    assert pair["status"] == "insufficient_pair_coverage"
    assert pair["distance_1_minus_abs_mean_pearson"] is None
    assert pair["valid_periods"] == 0
    assert result["missing_pairs"] == [{
        "left_ref": "run-a/perp", "right_ref": "run-a/spot",
        "status": "insufficient_pair_coverage", "valid_periods": 0,
        "required_periods": 26, "reason_codes": ["insufficient_common_cross_section_hours"],
    }]


def test_computed_record_without_executed_expression_fails_fast():
    spec, panel = landscape_panel()
    malformed = member("run-a/malformed", "spot_close")
    malformed["calculation"]["executed_expression"] = None
    with pytest.raises(ValueError, match="computed calculation is missing executed_expression"):
        build_factor_landscape(panel, spec, [malformed])


def test_expression_beyond_current_lookback_contract_is_not_executed(monkeypatch):
    spec, panel = landscape_panel()
    candidate = member("run-old/long-window", "ts_mean(spot_close,24)")
    monkeypatch.setattr(landscape, "evaluate_expression",
                        lambda *_args, **_kwargs: pytest.fail("incompatible formulas must not be evaluated"))

    result = build_factor_landscape(panel, spec, [candidate])

    row = result["members"][0]
    assert row["inventory_status"] == "incompatible_with_current_A_contract"
    assert "exceeds current A contract maximum" in row["reason"]


def test_sampling_is_a_only_at_declared_cadence_and_nonmembers_do_not_change_correlations():
    spec, panel = landscape_panel(sample_hours=2)
    candidates = [member("run-a/spot", "spot_close"), member("run-a/reverse", "neg(spot_close)")]
    first = build_factor_landscape(panel, spec, candidates)
    changed = panel.values.copy()
    nonmember = panel.universe.eq(False)
    changed.loc[nonmember, "spot_close"] = -1e15
    second = build_factor_landscape(FactorInputPanel(changed, panel.universe.copy(), panel.diagnostics),
                                    spec, candidates)

    assert first["contract"]["sample_hours"] == 2
    assert first["contract"]["sampled_hours"] == 48
    assert first["pairwise_correlations"] == second["pairwise_correlations"]
    assert first["tree"]["scipy_linkage"] == second["tree"]["scipy_linkage"]
    assert first["members"][0]["current_A_value_coverage"]["eligible_sample_rows"] == 48 * 6


def test_panel_that_extends_into_b_is_rejected():
    spec, panel = landscape_panel(append_b=True)
    with pytest.raises(ValueError, match="stop at the end of its A segment"):
        build_factor_landscape(panel, spec, [])
