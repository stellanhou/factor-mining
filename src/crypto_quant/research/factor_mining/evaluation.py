"""24-hour labels and deterministic evidence, isolated from LLM interpretation."""

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
    total = float(scores @ scores)
    for lag in range(1, min(spec.hac_lags, len(scores) - 1) + 1):
        total += 2 * (1 - lag / (spec.hac_lags + 1)) * float(scores[lag:] @ scores[:-lag])
    variance = total / (n * n) * n / (n - 1)
    if variance <= 0:
        result["status"] = "degenerate_variance"
        return result
    se = math.sqrt(variance)
    z = NormalDist().inv_cdf((1 + spec.confidence) / 2)
    result.update(se=se, ci=[mean - z * se, mean + z * se],
                  p_value=math.erfc(abs(mean / se) / math.sqrt(2)), status="estimated")
    return finite(result)


def build_labels(panel: FactorInputPanel, spec: ResearchSpec, stage: str) -> pd.DataFrame:
    start, end = spec.bounds(stage)
    membership = validate_universe(panel.universe)
    require(panel.values.index.equals(membership.index), "input/universe indices differ")
    hours = membership.index.get_level_values("timestamp")
    require(hours.max() < end, "input panel includes a later protected data segment")
    prices = panel.values["perp_open"].unstack("symbol")
    # An hour t input is available at t+1h-1ms. First executable opening is t+1h.
    entry, exit_price = prices.shift(-1), prices.shift(-25)
    returns = (exit_price / entry - 1).where((entry > 0) & (exit_price > 0))
    output = pd.DataFrame(index=membership.index)
    output["label_start"] = hours + pd.Timedelta(hours=1)
    output["label_end"] = hours + pd.Timedelta(hours=25)
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
                    direction: int) -> dict[str, Any]:
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
    return finite({"segment": stage, "direction": direction, "horizon_hours": 24,
                   "label": spec.label, "sample_hours": spec.sample_hours,
                   "grouping": "equal-weight rank quantiles; average ties kept together",
                   "summary": summary, "periods": periods.reset_index().to_dict("records"),
                   "stages": stages, "per_symbol": per_symbol,
                   "coverage": {"eligible_observations": int(sampled["eligible"].sum()),
                                "purged_observations": int((sampled["purged"] & sampled["eligible"]).sum()),
                                "purged_hours": int((periods["status"] == "purged_boundary").sum()),
                                "status_counts": periods["status"].value_counts().to_dict()},
                   "interpretation": "A is adaptive development evidence; B is a fixed-batch check. No strategy PnL or composite score."})


def correct_batch(reports: dict[str, dict[str, Any]], spec: ResearchSpec) -> dict[str, Any]:
    """Both primary tests of EVERY frozen candidate form one correction family.

    Unavailable tests stay in the family (p=1 for adjustment, raw p remains null).
    """
    tests = []
    for candidate_id, report in reports.items():
        for metric in ("rank_ic", "directional_spread"):
            raw = report["summary"][metric]["p_value"]
            tests.append({"candidate_id": candidate_id, "metric": metric, "raw_p": raw})
    require(bool(tests), "empty validation batch")
    p = np.array([1.0 if t["raw_p"] is None else t["raw_p"] for t in tests])
    order = np.argsort(p, kind="stable")
    m = len(tests)
    dependency = sum(1 / i for i in range(1, m + 1)) if spec.fdr_method == "BY" else 1
    adjusted = np.minimum.accumulate((p[order] * m * dependency / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(adjusted, 1)
    for item, value in zip(tests, q):
        item.update(adjusted_p=float(value), rejected=bool(item["raw_p"] is not None and value <= spec.fdr_alpha))
    return {"method": spec.fdr_method, "alpha": spec.fdr_alpha, "family_size": m,
            "family": "two primary two-sided mean tests per frozen candidate, including unavailable tests", "tests": tests}


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
