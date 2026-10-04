"""Forward-return labels and deterministic evidence, isolated from LLM interpretation."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from crypto_quant.features.factor_inputs import FactorInputPanel, validate_universe
from .contracts import ResearchSpec, require


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def hac_mean(series: pd.Series, spec: ResearchSpec) -> dict[str, Any]:
    """Intercept-only Newey-West/Bartlett; missing grid slots retain their lags.

    Covariance is sum(kernel-weighted score cross-products) / n_observed**2,
    with n/(n-1) small-sample correction. Inference uses asymptotic normal tails.
    Missing observations have zero score, rather than compressing elapsed time.
    """
    x = np.asarray(series, dtype=float)
    good = np.isfinite(x)
    n = int(good.sum())
    mean = float(x[good].mean()) if n else None
    std = float(x[good].std(ddof=1)) if n > 1 else None
    result = {"mean": mean, "std": std, "mean_std_ratio": mean / std if std and mean is not None else None,
              "n": n, "grid_periods": len(x), "method": "Newey-West Bartlett, normal approximation",
              "lags": spec.hac_lags, "confidence": spec.confidence, "alternative": "two-sided",
              "se": None, "ci": None, "p_value": None, "status": "insufficient_evidence"}
    if n < spec.min_periods:
        return result
    if not std:
        result["status"] = "degenerate_variance"
        return result
    scores = np.where(good, x - mean, 0.0)
    # Elementwise products avoid platform BLAS overflow warnings observed for
    # bounded IC/spread arrays while preserving the same Bartlett estimator.
    total = float(np.sum(scores * scores, dtype=np.float64))
    for lag in range(1, min(spec.hac_lags, len(scores) - 1) + 1):
        cross_product = float(np.sum(scores[lag:] * scores[:-lag], dtype=np.float64))
        total += 2 * (1 - lag / (spec.hac_lags + 1)) * cross_product
    variance = total / (n * n) * n / (n - 1)
    if variance <= 0:
        result["status"] = "degenerate_variance"
        return result
    se = math.sqrt(variance)
    z = NormalDist().inv_cdf((1 + spec.confidence) / 2)
    result.update(se=se, ci=[mean - z * se, mean + z * se],
                  p_value=math.erfc(abs(mean / se) / math.sqrt(2)), status="estimated")
    return finite(result)


def build_labels(panel: FactorInputPanel, spec: ResearchSpec, stage: str, *, horizon_hours: int = 24) -> pd.DataFrame:
    require(horizon_hours in (1, 4, 24), "supported horizons: 1, 4, 24 hours")
    require(stage == "A" or horizon_hours in spec.b_horizons,
            "B horizon must be declared by the research contract")
    start, end = spec.bounds(stage)
    membership = validate_universe(panel.universe)
    require(panel.values.index.equals(membership.index), "input/universe indices differ")
    hours = membership.index.get_level_values("timestamp")
    require(hours.max() < end, "input panel includes a later protected data segment")
    prices = panel.values["perp_open"].unstack("symbol")
    # An hour t input is available at t+1h-1ms. First executable opening is t+1h.
    entry, exit_price = prices.shift(-1), prices.shift(-(horizon_hours + 1))
    returns = (exit_price / entry - 1).where((entry > 0) & (exit_price > 0))
    output = pd.DataFrame(index=membership.index)
    output["label_start"] = hours + pd.Timedelta(hours=1)
    output["label_end"] = hours + pd.Timedelta(hours=horizon_hours + 1)
    output["signal_available_at"] = hours + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    output["in_segment"] = (hours >= start) & (hours < end)
    # Exact end-boundary quotes belong to the following half-open segment.
    output["purged"] = output["in_segment"] & (output["label_end"] >= end)
    output["forward_return"] = returns.to_numpy().reshape(-1)
    output["eligible"] = membership
    output.loc[~output["in_segment"] | output["purged"] | ~membership, "forward_return"] = np.nan
    return output


def _summary(frame: pd.DataFrame, spec: ResearchSpec, direction: int) -> dict[str, Any]:
    ic, spread = frame["rank_ic"], frame["directional_spread"]
    return finite({
        "rank_ic": hac_mean(ic, spec), "directional_spread": hac_mean(spread, spec),
        "raw_high_low_spread_mean": frame["raw_high_low_spread"].mean(),
        "ic_direction_share": (ic.dropna() * direction > 0).mean(),
        "spread_direction_share": (spread.dropna() > 0).mean(),
        "group_means": {str(i): frame[f"group_{i}"].mean() for i in range(1, spec.groups + 1)},
    })


def evaluate_factor(values: pd.Series, labels: pd.DataFrame, spec: ResearchSpec, stage: str,
                    direction: int, *, horizon_hours: int = 24) -> dict[str, Any]:
    require(horizon_hours in (1, 4, 24), "supported horizons: 1, 4, 24 hours")
    require(stage == "A" or horizon_hours in spec.b_horizons,
            "B horizon must be declared by the research contract")
    require(direction in {-1, 1}, "direction must be frozen before evaluation")
    require(values.index.equals(labels.index), "factor and label indices differ")
    start, end = spec.bounds(stage)
    frame = labels.copy()
    frame["factor"] = values
    times = pd.date_range(start, end, freq=f"{spec.sample_hours}h", inclusive="left")
    rows = []
    sampled = frame.loc[frame.index.get_level_values("timestamp").isin(times)]
    for timestamp in times:
        group = sampled.xs(timestamp, level="timestamp")
        eligible = group.loc[group["eligible"] & ~group["purged"]]
        clean = eligible[["factor", "forward_return"]].replace([np.inf, -np.inf], np.nan).dropna()
        row = {"timestamp": timestamp, "eligible": int(group["eligible"].sum()), "n": len(clean),
               "rank_ic": np.nan, "raw_high_low_spread": np.nan, "directional_spread": np.nan,
               "status": "insufficient_cross_section"}
        for i in range(1, spec.groups + 1):
            row[f"group_{i}"] = np.nan
            row[f"group_{i}_n"] = 0
        if group["purged"].any():
            row["status"] = "purged_boundary"
        elif len(clean) >= spec.min_symbols and clean["factor"].nunique() >= spec.groups:
            ranks = clean["factor"].rank(method="average")
            if clean["forward_return"].nunique() > 1:
                row["rank_ic"] = ranks.corr(clean["forward_return"].rank(method="average"))
            # Equal-width rank quantiles; equal factor values always remain together.
            buckets = np.minimum(((ranks - 1) / len(clean) * spec.groups).astype(int) + 1, spec.groups)
            means = clean["forward_return"].groupby(buckets).mean()
            counts = buckets.value_counts()
            for i in range(1, spec.groups + 1):
                if i in means:
                    row[f"group_{i}"] = means[i]
                    row[f"group_{i}_n"] = int(counts[i])
            if len(means) == spec.groups:
                row["raw_high_low_spread"] = means[spec.groups] - means[1]
                row["directional_spread"] = direction * row["raw_high_low_spread"]
                row["status"] = "computed"
            else:
                row["status"] = "empty_group_due_to_ties"
        rows.append(row)
    periods = pd.DataFrame(rows).set_index("timestamp")
    summary = _summary(periods, spec, direction)
    stages = []
    stage_index = ((periods.index - start) // pd.Timedelta(hours=spec.stage_hours)).astype(int)
    for _, part in periods.groupby(stage_index):
        stages.append({"start": part.index.min(), "end": part.index.max(), **_summary(part, spec, direction)})
    stage_valid = [s for s in stages if s["rank_ic"]["n"] >= spec.min_periods
                   and s["directional_spread"]["n"] >= spec.min_periods]
    summary["positive_stage_share"] = (
        sum(s["rank_ic"]["mean"] * direction > 0 and s["directional_spread"]["mean"] > 0 for s in stage_valid)
        / len(stage_valid) if stage_valid else None)
    summary["valid_stages"] = len(stage_valid)
    rolling = periods[["rank_ic", "directional_spread"]].rolling(spec.rolling_periods, min_periods=spec.rolling_periods).mean()
    rolling.columns = [f"rolling_{name}" for name in rolling.columns]
    periods = periods.join(rolling)
    per_symbol = []
    for symbol, part in sampled.groupby(level="symbol"):
        valid = part.loc[part["eligible"] & ~part["purged"], ["factor", "forward_return"]].dropna()
        per_symbol.append({"symbol": symbol, "n": len(valid), "mean_forward_return": valid["forward_return"].mean(),
                           "factor_mean": valid["factor"].mean()})
    return finite({"segment": stage, "direction": direction, "horizon_hours": horizon_hours,
                   "label": f"perp_next_open_{horizon_hours}h", "sample_hours": spec.sample_hours,
                   "grouping": "equal-weight rank quantiles; average ties kept together",
                   "summary": summary, "periods": periods.reset_index().to_dict("records"),
                   "stages": stages, "per_symbol": per_symbol,
                   "coverage": {"eligible_observations": int(sampled["eligible"].sum()),
                                "purged_observations": int((sampled["purged"] & sampled["eligible"]).sum()),
                                "purged_hours": int((periods["status"] == "purged_boundary").sum()),
                                "status_counts": periods["status"].value_counts().to_dict()},
                   "interpretation": "A is adaptive development evidence; B is a fixed-batch check. No strategy PnL or composite score."})


def evaluate_horizon_comparison(values: pd.Series, labels_by_horizon: dict[int, pd.DataFrame],
                                spec: ResearchSpec, direction: int, *, stage: str = "A") -> dict[str, Any]:
    """Compare a fixed factor on identical observations; retain the contract's HAC bandwidth."""
    common = np.isfinite(values)
    for labels in labels_by_horizon.values():
        common = common & labels["eligible"] & np.isfinite(labels["forward_return"])
    reports = {}
    for horizon in (1, 4, 24):
        labels = labels_by_horizon[horizon].copy()
        # Use the same boundary and asset mask so all three rank sorts are identical.
        labels["purged"] = labels_by_horizon[24]["purged"]
        reports[str(horizon)] = evaluate_factor(
            values.where(common), labels, spec, stage, direction, horizon_hours=horizon)
    return {"interpretation": f"{stage}-stage horizon comparison on common finite observations and the 24h boundary; "
            "same formula, direction, groups and HAC bandwidth. Unadjusted exploratory p-values; "
            "batch admission is computed separately. Returns exclude trading costs and funding.",
            "horizons": reports}


def correct_batch(reports: dict[str, dict[str, Any]], spec: ResearchSpec) -> dict[str, Any]:
    """Every frozen candidate/horizon pair forms one BH correction family.

    Unavailable tests stay in the family (p=1 for adjustment, raw p remains null).
    """
    tests = []
    for candidate_id, candidate_report in sorted(reports.items()):
        horizons = candidate_report["horizons"]
        require(bool(horizons), "each frozen candidate requires horizon evaluations")
        for horizon, report in sorted(horizons.items(), key=lambda item: int(item[0])):
            require(type(report["horizon_hours"]) is int
                    and str(report["horizon_hours"]) == horizon
                    and report["horizon_hours"] in spec.b_horizons,
                    "correction horizon differs from its evaluation")
            raw = report["summary"]["rank_ic"]["p_value"]
            require(raw is None or type(raw) in (int, float)
                    and math.isfinite(raw) and 0 <= raw <= 1,
                    "Rank IC p-value must be null or finite in [0,1]")
            tests.append({"candidate_id": candidate_id, "horizon_hours": report["horizon_hours"],
                          "metric": "rank_ic", "raw_p": raw})
    require(bool(tests), "empty validation batch")
    p = np.array([1.0 if t["raw_p"] is None else t["raw_p"] for t in tests])
    order = np.argsort(p, kind="stable")
    m = len(tests)
    adjusted = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(adjusted, 1)
    for item, value in zip(tests, q):
        item.update(adjusted_p=float(value), rejected=bool(item["raw_p"] is not None and value <= spec.fdr_alpha))
    return {"method": "BH", "alpha": spec.fdr_alpha, "family_size": m,
            "family": "one two-sided Rank IC mean test per frozen candidate/horizon pair, including unavailable tests", "tests": tests}


def compare_experiment(report: dict[str, Any], control: dict[str, Any], plan: dict[str, Any],
                       spec: ResearchSpec) -> dict[str, Any]:
    require(report["segment"] == control["segment"] == "A", "optimization comparisons use A only")
    require(report["direction"] == control["direction"], "a controlled modification cannot flip direction")
    trial = pd.DataFrame(report["periods"]).set_index("timestamp")
    base = pd.DataFrame(control["periods"]).set_index("timestamp")
    require(trial.index.equals(base.index), "comparison must use the same calendar grid")
    design = plan["experiment_design"]
    metric = design["metric"]
    sign = report["direction"] if metric == "rank_ic" else 1
    delta = hac_mean((trial[metric] - base[metric]) * sign, spec)
    ic_delta = hac_mean((trial["rank_ic"] - base["rank_ic"]) * report["direction"], spec)
    state, reason = "pause_insufficient", "paired uncertainty does not yet resolve the predeclared improvement and loss limits"
    if delta["ci"] and ic_delta["ci"]:
        if delta["ci"][1] < design["min_improvement"] or ic_delta["ci"][1] < -design["max_ic_loss"]:
            state, reason = "stop", "upper confidence bound is below the predeclared improvement or IC-loss limit"
        elif delta["ci"][0] >= design["min_improvement"] and ic_delta["ci"][0] >= -design["max_ic_loss"]:
            state, reason = "continue", "paired lower bounds meet the predeclared improvement and IC-loss limits"
    return {"plan": plan, "paired_improvement": delta, "paired_ic_change": ic_delta,
            "decision": state, "reason": reason,
            "scope": "adaptive A-stage diagnostic, not independent validation"}


RANK_DISPLACEMENT_VERSION = "factor-rank-displacement-v1"
RANK_DISPLACEMENT_HOURS = (1, 4, 24)


def _rank_input_matrices(values: pd.Series, universe: pd.Series, spec: ResearchSpec, stage: str):
    require(isinstance(values, pd.Series), "factor values must be a Series")
    require(isinstance(universe, pd.Series), "universe must be a Series")
    require(values.index.equals(universe.index), "factor and universe indices differ")
    membership = validate_universe(universe)
    require(set(universe.index.get_level_values("symbol"))
            == set(membership.index.get_level_values("symbol")),
            "universe symbols must already use canonical names")
    # Keep the input row pairing while applying the validator's sorted index.
    aligned_values = pd.Series(values.to_numpy(), index=universe.index).reindex(membership.index)
    clean_values = aligned_values.astype(float).replace([np.inf, -np.inf], np.nan)
    value_matrix = clean_values.unstack("symbol").sort_index()
    member_matrix = membership.unstack("symbol").sort_index()
    start, end = spec.bounds(stage)
    hourly_times = pd.date_range(start, end, freq="h", inclusive="left")
    times = pd.date_range(start, end, freq=f"{spec.sample_hours}h", inclusive="left")
    symbols = member_matrix.columns
    panel_times = member_matrix.index
    present = hourly_times.isin(panel_times)
    value_matrix = value_matrix.reindex(index=hourly_times, columns=symbols)
    member_matrix = member_matrix.reindex(index=hourly_times, columns=symbols).fillna(False).astype(bool)
    sample_positions = np.arange(0, len(hourly_times), spec.sample_hours, dtype=np.int64)
    return (times, hourly_times, sample_positions, symbols, value_matrix.to_numpy(dtype=float),
            member_matrix.to_numpy(dtype=bool), present)


def _row_rank(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(values).where(mask)
    ranks = frame.rank(axis=1, method="average", na_option="keep")
    unique = ranks.nunique(axis=1, dropna=True).to_numpy(dtype=np.int64)
    return ranks.to_numpy(dtype=float), unique


def _rank_displacement_rows(current_values: np.ndarray, previous_values: np.ndarray,
                            common: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = common.sum(axis=1).astype(np.int64)
    current_ranks, current_unique = _row_rank(current_values, common)
    previous_ranks, previous_unique = _row_rank(previous_values, common)
    rank_difference = np.abs(current_ranks - previous_ranks)
    rank_difference = np.where(common, rank_difference, 0.0)
    displacement = np.divide(rank_difference.sum(axis=1), counts * counts,
                             out=np.full(len(counts), np.nan, dtype=float), where=counts > 0)
    return displacement, current_unique, previous_unique


def _displacement_summary(values: np.ndarray, expected_periods: int) -> dict[str, Any]:
    valid = values[np.isfinite(values)]
    n = int(len(valid))
    return finite({
        "mean": float(valid.mean()) if n else None,
        "median": float(np.median(valid)) if n else None,
        "p90": float(np.quantile(valid, 0.90)) if n else None,
        "valid_periods": n,
        "expected_periods": expected_periods,
        "valid_period_share": n / expected_periods if expected_periods else None,
    })


def _displacement_stages(periods: pd.DataFrame, start: pd.Timestamp, spec: ResearchSpec) -> list[dict[str, Any]]:
    offsets = ((periods.index - start) // pd.Timedelta(hours=spec.stage_hours)).astype(int)
    stages = []
    for _, part in periods.groupby(offsets, sort=True):
        stages.append({
            "start": part.index[0], "end": part.index[-1] + pd.Timedelta(hours=spec.sample_hours),
            **_displacement_summary(part["displacement"].to_numpy(dtype=float), len(part)),
        })
    return finite(stages)


def evaluate_rank_displacement(values: pd.Series, universe: pd.Series, spec: ResearchSpec,
                               stage: str) -> dict[str, Any]:
    """Describe exact-hour cross-sectional rank changes for 1h, 4h and 24h.

    Each endpoint is sorted only over symbols that are eligible and finite at
    both endpoints. Coverage uses the union of endpoint-eligible symbols as
    its denominator, so membership changes and missing factor values remain
    visible instead of being counted as stable ranks.
    """
    times, _, sample_positions, _, hourly_factor, hourly_eligible, hourly_present = _rank_input_matrices(
        values, universe, spec, stage)
    start, _ = spec.bounds(stage)
    factor = hourly_factor[sample_positions]
    eligible = hourly_eligible[sample_positions]
    present = hourly_present[sample_positions]
    eligible_counts = eligible.sum(axis=1).astype(np.int64)
    finite_eligible = eligible & np.isfinite(factor)
    factor_counts = finite_eligible.sum(axis=1).astype(np.int64)
    result: dict[str, Any] = {"definition_version": RANK_DISPLACEMENT_VERSION, "segment": stage, "deltas": {}}

    for delta in RANK_DISPLACEMENT_HOURS:
        n_times = len(times)
        previous_factor = np.full_like(factor, np.nan, dtype=float)
        previous_eligible = np.zeros_like(eligible, dtype=bool)
        previous_present = np.zeros(n_times, dtype=bool)
        previous_positions = sample_positions - delta
        has_previous = previous_positions >= 0
        if has_previous.any():
            destination = np.flatnonzero(has_previous)
            source = previous_positions[has_previous]
            previous_factor[destination] = hourly_factor[source]
            previous_eligible[destination] = hourly_eligible[source]
            previous_present[destination] = hourly_present[source]
        previous_eligible_counts = previous_eligible.sum(axis=1).astype(np.int64)
        previous_finite = previous_eligible & np.isfinite(previous_factor)
        previous_factor_counts = previous_finite.sum(axis=1).astype(np.int64)
        common = finite_eligible & previous_finite
        common_counts = common.sum(axis=1).astype(np.int64)
        displacement, unique_current, unique_previous = _rank_displacement_rows(factor, previous_factor, common)
        boundary = (times - pd.Timedelta(hours=delta)) < start
        missing_hour = ~(present & previous_present) & ~boundary
        sufficient = common_counts >= spec.min_symbols
        status = np.full(n_times, "insufficient_common_symbols", dtype=object)
        status[missing_hour] = "missing_exact_hour"
        status[boundary] = "boundary_excluded"
        status[~boundary & ~missing_hour & sufficient] = "computed"
        displacement[~sufficient | boundary | missing_hour] = np.nan
        eligible_union = eligible_counts + previous_eligible_counts - (eligible & previous_eligible).sum(axis=1)
        coverage_ratio = np.divide(common_counts, eligible_union,
                                    out=np.full(n_times, np.nan, dtype=float), where=eligible_union > 0)
        tie_current = (common_counts > 0) & (unique_current < common_counts)
        tie_previous = (common_counts > 0) & (unique_previous < common_counts)
        constant_current = (common_counts > 0) & (unique_current == 1)
        constant_previous = (common_counts > 0) & (unique_previous == 1)
        frame = pd.DataFrame({
            "displacement": displacement,
            "status": status,
            "eligible_t": eligible_counts,
            "eligible_t_minus_delta": previous_eligible_counts,
            "factor_finite_t": factor_counts,
            "factor_finite_t_minus_delta": previous_factor_counts,
            "eligible_common": (eligible & previous_eligible).sum(axis=1),
            "common_symbols": common_counts,
            "eligible_union": eligible_union,
            "common_coverage_ratio": coverage_ratio,
            "unique_t": unique_current,
            "unique_t_minus_delta": unique_previous,
            "has_ties_t": tie_current,
            "has_ties_t_minus_delta": tie_previous,
            "constant_t": constant_current,
            "constant_t_minus_delta": constant_previous,
        }, index=times)
        summary = _displacement_summary(frame["displacement"].to_numpy(dtype=float), len(frame))
        counts_by_status = frame["status"].value_counts().to_dict()
        coverage = {
            "expected_periods": len(frame),
            "valid_periods": int((frame["status"] == "computed").sum()),
            "valid_period_share": int((frame["status"] == "computed").sum()) / len(frame) if len(frame) else None,
            "boundary_excluded_periods": int((frame["status"] == "boundary_excluded").sum()),
            "missing_exact_hour_periods": int((frame["status"] == "missing_exact_hour").sum()),
            "insufficient_common_symbol_periods": int((frame["status"] == "insufficient_common_symbols").sum()),
            "status_counts": counts_by_status,
            "common_coverage_denominator": "union of endpoint-eligible symbols",
        }
        result["deltas"][str(delta)] = {
            "summary": summary,
            "periods": finite([{"timestamp": timestamp, **row} for timestamp, row in frame.iterrows()]),
            "stages": _displacement_stages(frame, start, spec),
            "coverage": finite({**coverage, "sample_hours": spec.sample_hours}),
        }
    return finite({**result, "sample_hours": spec.sample_hours})


def _rank_correlation(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> np.ndarray:
    left_ranks, _ = _row_rank(left, mask)
    right_ranks, _ = _row_rank(right, mask)
    left_finite = np.isfinite(left_ranks)
    right_finite = np.isfinite(right_ranks)
    valid = left_finite & right_finite
    counts = valid.sum(axis=1)
    left_centered = np.where(valid, left_ranks, 0.0)
    right_centered = np.where(valid, right_ranks, 0.0)
    left_mean = np.divide(left_centered.sum(axis=1), counts,
                          out=np.zeros(len(counts), dtype=float), where=counts > 0)
    right_mean = np.divide(right_centered.sum(axis=1), counts,
                           out=np.zeros(len(counts), dtype=float), where=counts > 0)
    left_centered = np.where(valid, left_ranks - left_mean[:, None], 0.0)
    right_centered = np.where(valid, right_ranks - right_mean[:, None], 0.0)
    covariance = (left_centered * right_centered).sum(axis=1)
    left_variance = (left_centered * left_centered).sum(axis=1)
    right_variance = (right_centered * right_centered).sum(axis=1)
    denominator = np.sqrt(left_variance * right_variance)
    return np.divide(covariance, denominator, out=np.full(len(counts), np.nan, dtype=float),
                     where=(counts > 1) & (denominator > 0))


def compare_rank_displacement_experiment(trial_values: pd.Series, control_values: pd.Series,
                                         labels: pd.DataFrame, spec: ResearchSpec, direction: int,
                                         plan: dict[str, Any], *, universe: pd.Series) -> dict[str, Any]:
    """Compare D and frozen-direction Rank IC on strict common A-stage samples."""
    require(direction in {-1, 1}, "direction must be frozen before evaluation")
    require(isinstance(universe, pd.Series), "universe must be a Series")
    require(trial_values.index.equals(control_values.index), "trial and control indices differ")
    require(trial_values.index.equals(labels.index), "factor and label indices differ")
    require(trial_values.index.equals(universe.index), "factor and universe indices differ")
    design = plan["experiment_design"]
    horizon = design["horizon_hours"]
    delta = design["displacement_hours"]
    require(type(horizon) is int and horizon in RANK_DISPLACEMENT_HOURS,
            "rank-displacement experiment requires horizon_hours in 1, 4, 24")
    require(type(delta) is int and delta in RANK_DISPLACEMENT_HOURS,
            "rank-displacement experiment requires displacement_hours in 1, 4, 24")
    require(math.isfinite(float(design["min_improvement"])) and float(design["min_improvement"]) > 0,
            "minimum absolute D improvement must be positive")
    require(math.isfinite(float(design["max_ic_loss"])) and float(design["max_ic_loss"]) >= 0,
            "maximum IC loss must be nonnegative")
    start, end = spec.bounds("A")
    require(labels.index.names == ["timestamp", "symbol"], "labels require (timestamp, symbol) index")
    require({"eligible", "forward_return", "label_start", "label_end"} <= set(labels.columns),
            "labels do not match build_labels output")
    label_hours = ((labels["label_end"] - labels["label_start"]).dt.total_seconds() / 3600).dropna().unique()
    require(len(label_hours) == 1 and float(label_hours[0]) == horizon,
            "labels horizon differs from experiment_design.horizon_hours")
    require(labels.index.equals(universe.index), "labels and universe indices differ")

    membership = validate_universe(universe)
    require(set(universe.index.get_level_values("symbol"))
            == set(membership.index.get_level_values("symbol")),
            "universe symbols must already use canonical names")
    index = membership.index
    trial = pd.Series(trial_values.to_numpy(), index=universe.index).reindex(index).astype(float).replace([np.inf, -np.inf], np.nan)
    control = pd.Series(control_values.to_numpy(), index=universe.index).reindex(index).astype(float).replace([np.inf, -np.inf], np.nan)
    label_frame = labels.copy()
    label_frame.index = universe.index
    label_frame = label_frame.reindex(index)
    trial_matrix = trial.unstack("symbol").sort_index()
    control_matrix = control.unstack("symbol").sort_index()
    universe_matrix = membership.unstack("symbol").sort_index()
    returns = label_frame["forward_return"].astype(float).replace([np.inf, -np.inf], np.nan)
    return_matrix = returns.unstack("symbol").sort_index()
    label_times = label_frame.index.get_level_values("timestamp")
    label_start = label_frame["label_start"]
    label_end = label_frame["label_end"]
    label_in_segment = pd.Series(
        (label_times >= start) & (label_times < end) & label_start.ge(start) & label_end.lt(end),
        index=label_frame.index,
    )
    require(pd.api.types.is_bool_dtype(label_frame["eligible"].dtype)
            and not label_frame["eligible"].isna().any(),
            "label eligibility must be explicit boolean values")
    eligible_labels = label_frame["eligible"] & label_in_segment
    eligible_label_matrix = eligible_labels.unstack("symbol").sort_index()
    label_boundary_matrix = label_in_segment.unstack("symbol").sort_index()
    hourly_times = pd.date_range(start, end, freq="h", inclusive="left")
    times = pd.date_range(start, end, freq=f"{spec.sample_hours}h", inclusive="left")
    sample_positions = np.arange(0, len(hourly_times), spec.sample_hours, dtype=np.int64)
    symbols = universe_matrix.columns
    present = hourly_times.isin(universe_matrix.index)[sample_positions]
    hourly_trial = trial_matrix.reindex(index=hourly_times, columns=symbols).to_numpy(dtype=float)
    hourly_control = control_matrix.reindex(index=hourly_times, columns=symbols).to_numpy(dtype=float)
    hourly_returns = return_matrix.reindex(index=hourly_times, columns=symbols).to_numpy(dtype=float)
    hourly_eligible = universe_matrix.reindex(index=hourly_times, columns=symbols).fillna(False).to_numpy(dtype=bool)
    hourly_label_eligible = eligible_label_matrix.reindex(index=hourly_times, columns=symbols).fillna(False).to_numpy(dtype=bool)
    hourly_label_boundary_valid = label_boundary_matrix.reindex(index=hourly_times, columns=symbols).fillna(False).to_numpy(dtype=bool)
    trial_values_matrix = hourly_trial[sample_positions]
    control_values_matrix = hourly_control[sample_positions]
    return_values_matrix = hourly_returns[sample_positions]
    eligible = hourly_eligible[sample_positions]
    label_eligible = hourly_label_eligible[sample_positions]
    label_boundary_valid = hourly_label_boundary_valid[sample_positions]
    label_finite = np.isfinite(return_values_matrix)
    n_times = len(times)
    trial_previous = np.full_like(trial_values_matrix, np.nan)
    control_previous = np.full_like(control_values_matrix, np.nan)
    eligible_previous = np.zeros_like(eligible, dtype=bool)
    previous_present = np.zeros(n_times, dtype=bool)
    previous_positions = sample_positions - delta
    has_previous = previous_positions >= 0
    if has_previous.any():
        destination = np.flatnonzero(has_previous)
        source = previous_positions[has_previous]
        trial_previous[destination] = hourly_trial[source]
        control_previous[destination] = hourly_control[source]
        eligible_previous[destination] = hourly_eligible[source]
        previous_present[destination] = hourly_times.isin(universe_matrix.index)[source]
    common = (eligible & eligible_previous & label_eligible & label_finite
              & np.isfinite(trial_values_matrix) & np.isfinite(control_values_matrix)
              & np.isfinite(trial_previous) & np.isfinite(control_previous))
    common_counts = common.sum(axis=1).astype(np.int64)
    trial_displacement, trial_unique, trial_previous_unique = _rank_displacement_rows(
        trial_values_matrix, trial_previous, common)
    control_displacement, control_unique, control_previous_unique = _rank_displacement_rows(
        control_values_matrix, control_previous, common)
    rank_ic_trial = _rank_correlation(trial_values_matrix, return_values_matrix, common)
    rank_ic_control = _rank_correlation(control_values_matrix, return_values_matrix, common)
    _, return_unique = _row_rank(return_values_matrix, common)
    ic_estimable_trial = (common_counts >= spec.min_symbols) & (trial_unique >= spec.groups) & (return_unique > 1)
    ic_estimable_control = (common_counts >= spec.min_symbols) & (control_unique >= spec.groups) & (return_unique > 1)
    rank_ic_trial[~ic_estimable_trial] = np.nan
    rank_ic_control[~ic_estimable_control] = np.nan
    raw_improvement = control_displacement - trial_displacement
    raw_ic_change = (rank_ic_trial - rank_ic_control) * direction
    boundary = (times - pd.Timedelta(hours=delta)) < start
    missing_hour = ~(present & previous_present) & ~boundary
    sufficient = common_counts >= spec.min_symbols
    ic_valid = np.isfinite(rank_ic_trial) & np.isfinite(rank_ic_control)
    displacement_valid = sufficient & ~boundary & ~missing_hour
    paired_valid = displacement_valid & ic_valid
    label_boundary_excluded = (eligible & ~label_boundary_valid).sum(axis=1) > 0
    period_status = np.full(n_times, "insufficient_common_symbols", dtype=object)
    period_status[missing_hour] = "missing_exact_hour"
    period_status[boundary] = "boundary_excluded"
    period_status[~boundary & ~missing_hour & ~sufficient & label_boundary_excluded] = "label_boundary_excluded"
    period_status[~boundary & ~missing_hour & sufficient & ~ic_valid] = "rank_ic_unavailable"
    period_status[paired_valid] = "computed"
    paired_improvement = np.where(paired_valid, raw_improvement, np.nan)
    paired_ic_change = np.where(paired_valid, raw_ic_change, np.nan)
    paired_improvement[(~np.isfinite(raw_improvement))] = np.nan

    paired_series = pd.Series(paired_improvement, index=times)
    ic_series = pd.Series(paired_ic_change, index=times)
    improvement_summary = hac_mean(paired_series, spec)
    ic_summary = hac_mean(ic_series, spec)
    state, reason = "pause_insufficient", "paired uncertainty does not yet resolve the predeclared improvement and loss limits"
    minimum_improvement = float(design["min_improvement"])
    maximum_ic_loss = float(design["max_ic_loss"])
    improvement_ci, ic_ci = improvement_summary["ci"], ic_summary["ci"]
    d_fails = (improvement_summary["status"] == "estimated" and improvement_ci
               and improvement_ci[1] < minimum_improvement)
    ic_fails = (ic_summary["status"] == "estimated" and ic_ci
                and ic_ci[1] < -maximum_ic_loss)
    if d_fails or ic_fails:
        state, reason = "stop", "an estimated upper confidence bound is below its predeclared D improvement or IC-loss limit"
    elif (improvement_summary["status"] == "estimated" and ic_summary["status"] == "estimated"
          and improvement_ci and ic_ci
          and improvement_ci[0] >= minimum_improvement and ic_ci[0] >= -maximum_ic_loss):
        state, reason = "continue", "paired lower bounds meet the predeclared D improvement and IC-loss limits"

    eligible_now = eligible.sum(axis=1)
    eligible_previous_count = eligible_previous.sum(axis=1)
    finite_trial_now = (eligible & np.isfinite(trial_values_matrix)).sum(axis=1)
    finite_control_now = (eligible & np.isfinite(control_values_matrix)).sum(axis=1)
    eligible_union = eligible_now + eligible_previous_count - (eligible & eligible_previous).sum(axis=1)
    common_ratio = np.divide(common_counts, eligible_union,
                             out=np.full(n_times, np.nan, dtype=float), where=eligible_union > 0)
    paired_periods = []
    for i, timestamp in enumerate(times):
        paired_periods.append({
            "timestamp": timestamp,
            "status": period_status[i],
            "eligible_t": int(eligible_now[i]),
            "eligible_t_minus_delta": int(eligible_previous_count[i]),
            "trial_finite_t": int(finite_trial_now[i]),
            "control_finite_t": int(finite_control_now[i]),
            "common_symbols": int(common_counts[i]),
            "common_coverage_ratio": common_ratio[i],
            "label_boundary_excluded_symbols": int((eligible[i] & ~label_boundary_valid[i]).sum()),
            "unique_trial_t": int(trial_unique[i]),
            "unique_trial_t_minus_delta": int(trial_previous_unique[i]),
            "unique_control_t": int(control_unique[i]),
            "unique_control_t_minus_delta": int(control_previous_unique[i]),
            "rank_ic_trial": rank_ic_trial[i],
            "rank_ic_control": rank_ic_control[i],
            "paired_ic_change": paired_ic_change[i],
            "displacement_trial": trial_displacement[i] if sufficient[i] else np.nan,
            "displacement_control": control_displacement[i] if sufficient[i] else np.nan,
            "displacement_improvement": raw_improvement[i] if sufficient[i] else np.nan,
            "paired_improvement": paired_improvement[i],
        })
    statuses = pd.Series(period_status).value_counts().to_dict()
    coverage = finite({
        "expected_periods": n_times,
        "paired_valid_periods": int(paired_valid.sum()),
        "paired_valid_period_share": float(paired_valid.mean()) if n_times else None,
        "boundary_excluded_periods": int((period_status == "boundary_excluded").sum()),
        "missing_exact_hour_periods": int((period_status == "missing_exact_hour").sum()),
        "label_boundary_excluded_periods": int((period_status == "label_boundary_excluded").sum()),
        "insufficient_common_symbol_periods": int((period_status == "insufficient_common_symbols").sum()),
        "rank_ic_unavailable_periods": int((period_status == "rank_ic_unavailable").sum()),
        "status_counts": statuses,
        "common_coverage_denominator": "union of endpoint-eligible symbols",
    })
    return finite({
        "plan": plan,
        "horizon_hours": horizon,
        "displacement_hours": delta,
        "paired_improvement": improvement_summary,
        "paired_ic_change": ic_summary,
        "decision": state,
        "reason": reason,
        "scope": "adaptive A-stage diagnostic, not independent validation",
        "paired_periods": paired_periods,
        "coverage": coverage,
        "sample_hours": spec.sample_hours,
    })
