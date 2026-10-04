"""Common-support numeric summaries for the fixed FM-v6 D studies.

The caller owns data loading, version construction, and front-first sample
selection. This module only evaluates the already-frozen values supplied for
one formula family and one time segment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_quant.features.factor_inputs import validate_universe
from crypto_quant.research.factor_mining.contracts import ResearchSpec, require, utc_hour
from crypto_quant.research.factor_mining.evaluation import (
    RANK_DISPLACEMENT_VERSION,
    finite,
    hac_mean,
)


HORIZONS = (1, 4, 24)
DISPLACEMENTS = (1, 4, 24)


def _segment_bounds(spec: ResearchSpec, segment: tuple[Any, Any] | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    a_start, a_end = spec.bounds("A")
    if segment is None:
        return a_start, a_end
    require(isinstance(segment, tuple) and len(segment) == 2,
            "segment must be a (start, end) UTC-hour tuple")
    start, end = utc_hour(segment[0]), utc_hour(segment[1])
    require(a_start <= start < end <= a_end, "study segment must be inside the A-stage half-open interval")
    return start, end


def _rank_rows(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranks = pd.DataFrame(values).where(mask).rank(axis=1, method="average", na_option="keep")
    unique = ranks.nunique(axis=1, dropna=True).to_numpy(dtype=np.int64)
    return ranks.to_numpy(dtype=float), unique


def _correlation_from_ranks(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(left) & np.isfinite(right)
    counts = valid.sum(axis=1)
    left_values = np.where(valid, left, 0.0)
    right_values = np.where(valid, right, 0.0)
    left_mean = np.divide(left_values.sum(axis=1), counts,
                          out=np.zeros(len(counts), dtype=float), where=counts > 0)
    right_mean = np.divide(right_values.sum(axis=1), counts,
                           out=np.zeros(len(counts), dtype=float), where=counts > 0)
    left_centered = np.where(valid, left - left_mean[:, None], 0.0)
    right_centered = np.where(valid, right - right_mean[:, None], 0.0)
    covariance = (left_centered * right_centered).sum(axis=1)
    left_variance = (left_centered * left_centered).sum(axis=1)
    right_variance = (right_centered * right_centered).sum(axis=1)
    denominator = np.sqrt(left_variance * right_variance)
    return np.divide(covariance, denominator, out=np.full(len(counts), np.nan, dtype=float),
                     where=(counts > 1) & (denominator > 0))


def _rank_displacement(current_ranks: np.ndarray, previous_ranks: np.ndarray,
                       mask: np.ndarray, counts: np.ndarray) -> np.ndarray:
    difference = np.where(mask, np.abs(current_ranks - previous_ranks), 0.0)
    return np.divide(difference.sum(axis=1), counts * counts,
                     out=np.full(len(counts), np.nan, dtype=float), where=counts > 0)


def _directional_spread(ranks: np.ndarray, returns: np.ndarray, mask: np.ndarray,
                        counts: np.ndarray, unique: np.ndarray, spec: ResearchSpec,
                        direction: int) -> tuple[np.ndarray, np.ndarray]:
    buckets = np.full(ranks.shape, 0, dtype=np.int64)
    present = mask & np.isfinite(ranks)
    rank_fraction = np.divide(ranks - 1, counts[:, None],
                              out=np.zeros_like(ranks, dtype=float), where=counts[:, None] > 0)
    scaled = rank_fraction * spec.groups
    buckets[present] = np.minimum(scaled[present].astype(np.int64) + 1, spec.groups)
    group_means = np.full((len(counts), spec.groups), np.nan, dtype=float)
    groups_present = np.zeros((len(counts), spec.groups), dtype=bool)
    for group in range(1, spec.groups + 1):
        group_mask = present & (buckets == group)
        group_counts = group_mask.sum(axis=1)
        group_sum = np.where(group_mask, returns, 0.0).sum(axis=1)
        group_means[:, group - 1] = np.divide(
            group_sum, group_counts, out=np.full(len(counts), np.nan, dtype=float), where=group_counts > 0)
        groups_present[:, group - 1] = group_counts > 0
    valid = ((counts >= spec.min_symbols) & (unique >= spec.groups)
             & groups_present.all(axis=1))
    spread = direction * (group_means[:, -1] - group_means[:, 0])
    spread[~valid] = np.nan
    return spread, valid


def _describe(values: np.ndarray) -> dict[str, Any]:
    valid = values[np.isfinite(values)]
    n = int(len(valid))
    return finite({
        "mean": float(valid.mean()) if n else None,
        "median": float(np.median(valid)) if n else None,
        "p90": float(np.quantile(valid, 0.90)) if n else None,
        "valid_periods": n,
    })


def _stages(times: pd.DatetimeIndex, stage_anchor: pd.Timestamp,
            segment_start: pd.Timestamp, segment_end: pd.Timestamp,
            spec: ResearchSpec, columns: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    stage_numbers = ((times - stage_anchor) // pd.Timedelta(hours=spec.stage_hours)).astype(int)
    result = []
    for stage_number, positions in pd.Series(np.arange(len(times)), index=times).groupby(stage_numbers, sort=True):
        indices = positions.to_numpy(dtype=np.int64)
        full_stage_start = stage_anchor + pd.Timedelta(hours=int(stage_number) * spec.stage_hours)
        full_stage_end = full_stage_start + pd.Timedelta(hours=spec.stage_hours)
        stage_start = max(full_stage_start, segment_start)
        stage_end = min(full_stage_end, segment_end)
        result.append({
            "stage_index": int(stage_number),
            "start": stage_start,
            "end": stage_end,
            "expected_periods": len(indices),
            "metrics": {name: _describe(values[indices]) for name, values in columns.items()},
        })
    return finite(result)


def _summary_hac(values: np.ndarray, times: pd.DatetimeIndex, spec: ResearchSpec) -> dict[str, Any]:
    return hac_mean(pd.Series(values, index=times, dtype=float), spec)


def _prepare_inputs(
    versions: dict[str, pd.Series], universe: pd.Series,
    labels_by_horizon: dict[int, pd.DataFrame], spec: ResearchSpec,
    segment: tuple[pd.Timestamp, pd.Timestamp],
) -> tuple[pd.DatetimeIndex, list[str], list[str], dict[str, np.ndarray], np.ndarray,
           dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    start, end = segment
    require(isinstance(versions, dict) and bool(versions),
            "versions must contain at least one executable version")
    require(all(isinstance(key, str) and key for key in versions), "version IDs must be nonempty strings")
    require(isinstance(universe, pd.Series), "universe must be a boolean Series")
    require(set(labels_by_horizon) == set(HORIZONS), "labels_by_horizon must contain exactly 1, 4 and 24")
    require(isinstance(universe.index, pd.MultiIndex)
            and universe.index.names == ["timestamp", "symbol"],
            "universe requires (timestamp, symbol) index")
    source_index = universe.index
    for version_id, values in versions.items():
        require(isinstance(values, pd.Series) and values.index.equals(source_index),
                f"factor values for {version_id} must align exactly with universe")
    for horizon, labels in labels_by_horizon.items():
        require(type(horizon) is int and isinstance(labels, pd.DataFrame)
                and labels.index.equals(source_index),
                f"labels for {horizon}h must align exactly with universe")

    raw_timestamps = source_index.get_level_values("timestamp")
    require(isinstance(raw_timestamps, pd.DatetimeIndex)
            and raw_timestamps.tz is not None and str(raw_timestamps.tz) == "UTC"
            and raw_timestamps.equals(raw_timestamps.floor("h")),
            "study timestamps must be exact UTC hours")
    in_segment = (raw_timestamps >= start) & (raw_timestamps < end)
    segment_universe = universe.loc[in_segment]
    membership = validate_universe(segment_universe)
    expected_hours = pd.date_range(start, end, freq="h", inclusive="left")
    actual_hours = membership.index.get_level_values("timestamp").unique()
    require(actual_hours.equals(expected_hours), "study inputs must cover every exact hour in the segment")
    raw_symbols = set(segment_universe.index.get_level_values("symbol"))
    canonical_symbols = set(membership.index.get_level_values("symbol"))
    require(raw_symbols == canonical_symbols, "study universe symbols must already be canonical")
    symbols = sorted(canonical_symbols)
    index = membership.index

    matrices: dict[str, np.ndarray] = {}
    for version_id, values in versions.items():
        selected = values.loc[in_segment]
        aligned = pd.Series(selected.to_numpy(), index=selected.index).reindex(index).astype(float)
        aligned = aligned.replace([np.inf, -np.inf], np.nan)
        matrices[version_id] = aligned.unstack("symbol").reindex(
            index=expected_hours, columns=symbols).to_numpy(dtype=float)

    membership_matrix = membership.unstack("symbol").reindex(
        index=expected_hours, columns=symbols).to_numpy(dtype=bool)
    label_returns: dict[int, np.ndarray] = {}
    label_validity: dict[int, np.ndarray] = {}
    label_boundary_validity: dict[int, np.ndarray] = {}
    for horizon, labels in labels_by_horizon.items():
        required = {"eligible", "forward_return", "label_start", "label_end"}
        require(required <= set(labels.columns), f"labels for {horizon}h do not match build_labels output")
        require(pd.api.types.is_bool_dtype(labels["eligible"].dtype) and not labels["eligible"].isna().any(),
                f"labels for {horizon}h require explicit boolean eligibility")
        duration = ((labels["label_end"] - labels["label_start"]).dt.total_seconds() / 3600).dropna().unique()
        require(len(duration) == 1 and float(duration[0]) == horizon,
                f"labels for {horizon}h contain a different horizon")
        selected = labels.loc[in_segment].copy()
        selected.index = selected.index.set_names(["timestamp", "symbol"])
        selected = selected.reindex(index)
        returns = selected["forward_return"].astype(float).replace([np.inf, -np.inf], np.nan)
        label_returns[horizon] = returns.unstack("symbol").reindex(
            index=expected_hours, columns=symbols).to_numpy(dtype=float)
        timestamps = selected.index.get_level_values("timestamp")
        bounds_valid = ((timestamps >= start) & (timestamps < end)
                        & selected["label_start"].ge(start) & selected["label_end"].lt(end))
        valid = selected["eligible"] & pd.Series(bounds_valid, index=selected.index)
        label_validity[horizon] = valid.unstack("symbol").reindex(
            index=expected_hours, columns=symbols).fillna(False).to_numpy(dtype=bool)
        label_boundary_validity[horizon] = pd.Series(bounds_valid, index=selected.index).unstack("symbol").reindex(
            index=expected_hours, columns=symbols).fillna(False).to_numpy(dtype=bool)
    return (expected_hours, symbols, list(versions), matrices, membership_matrix,
            label_returns, label_validity, label_boundary_validity)


def compute_factor_variant_study(
    versions: dict[str, pd.Series], universe: pd.Series,
    labels_by_horizon: dict[int, pd.DataFrame], spec: ResearchSpec, direction: int,
    *, reference_id: str, segment: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Compute full-grid common-support metrics for one fixed formula family.

    Every version shares one asset-time mask per H and Δ. Pairwise D and IC
    effects use the same valid timestamps across all versions. This function
    reports estimates and intervals only; it applies no research threshold or
    route decision.
    """
    require(direction in {-1, 1}, "direction must be frozen as -1 or 1")
    start, end = _segment_bounds(spec, segment)
    require(reference_id in versions, "reference_id must name one supplied version")
    (expected_hours, symbols, version_ids, matrices, membership, label_returns,
     label_validity, label_boundary_validity) = _prepare_inputs(
        versions, universe, labels_by_horizon, spec, (start, end))
    times = pd.date_range(start, end, freq=f"{spec.sample_hours}h", inclusive="left")
    sample_positions = np.arange(0, len(expected_hours), spec.sample_hours, dtype=np.int64)
    sampled_membership = membership[sample_positions]
    expected_periods = len(times)
    stage_anchor = spec.bounds("A")[0]
    result: dict[str, Any] = {
        "definition_version": RANK_DISPLACEMENT_VERSION,
        "segment": {"start": start, "end": end, "meaning": "half-open UTC interval inside A"},
        "sample_hours": spec.sample_hours,
        "direction": direction,
        "reference_id": reference_id,
        "versions": {version_id: {"horizons": {}} for version_id in version_ids},
        "native_coverage": {version_id: {} for version_id in version_ids},
        "native_prediction_coverage": {version_id: {} for version_id in version_ids},
        "pair_effects": {version_id: {"horizons": {}} for version_id in version_ids if version_id != reference_id},
    }

    for horizon in HORIZONS:
        current_returns = label_returns[horizon][sample_positions]
        labels_valid = label_validity[horizon][sample_positions] & np.isfinite(current_returns)
        label_bounds_valid = label_boundary_validity[horizon][sample_positions]
        eligible_counts = sampled_membership.sum(axis=1).astype(np.int64)
        label_counts = (sampled_membership & labels_valid).sum(axis=1).astype(np.int64)
        label_boundary_excluded = (sampled_membership & ~label_bounds_valid).sum(axis=1).astype(np.int64)
        for version_id in version_ids:
            current = matrices[version_id][sample_positions]
            factor_valid = sampled_membership & np.isfinite(current)
            common = factor_valid & labels_valid
            common_counts = common.sum(axis=1).astype(np.int64)
            factor_ranks, factor_unique = _rank_rows(current, common)
            _, label_unique = _rank_rows(current_returns, common)
            ic_valid = ((common_counts >= spec.min_symbols)
                        & (factor_unique >= spec.groups) & (label_unique > 1))
            _, spread_valid = _directional_spread(
                factor_ranks, current_returns, common, common_counts, factor_unique, spec, direction)
            factor_counts = factor_valid.sum(axis=1).astype(np.int64)
            common_periods = common_counts >= spec.min_symbols
            common_observations = int(common.sum())
            eligible_observations = int(eligible_counts.sum())
            result["native_prediction_coverage"][version_id][str(horizon)] = finite({
                "expected_periods": expected_periods,
                "eligible_observations": eligible_observations,
                "factor_finite_eligible_observations": int(factor_valid.sum()),
                "label_valid_eligible_observations": int(label_counts.sum()),
                "factor_label_common_observations": common_observations,
                "factor_finite_eligible_periods": int((factor_counts >= spec.min_symbols).sum()),
                "label_valid_eligible_periods": int((label_counts >= spec.min_symbols).sum()),
                "factor_label_common_periods": int(common_periods.sum()),
                "factor_label_common_period_share": float(common_periods.mean()) if expected_periods else None,
                "rank_ic_valid_periods": int(ic_valid.sum()),
                "directional_spread_valid_periods": int(spread_valid.sum()),
                "label_boundary_excluded_periods": int((label_boundary_excluded > 0).sum()),
                "mean_eligible_symbols": float(eligible_counts.mean()) if expected_periods else None,
                "mean_factor_finite_eligible_symbols": float(factor_counts.mean()) if expected_periods else None,
                "mean_label_valid_eligible_symbols": float(label_counts.mean()) if expected_periods else None,
                "mean_factor_label_common_symbols": float(common_counts.mean()) if expected_periods else None,
                "factor_label_coverage_ratio": (
                    common_observations / eligible_observations if eligible_observations else None),
            })

    for delta in DISPLACEMENTS:
        previous_positions = sample_positions - delta
        has_previous = previous_positions >= 0
        sampled_previous_membership = np.zeros_like(sampled_membership)
        if has_previous.any():
            sampled_previous_membership[np.flatnonzero(has_previous)] = membership[previous_positions[has_previous]]
        boundary = (times - pd.Timedelta(hours=delta)) < start
        eligible_now = sampled_membership.sum(axis=1)
        eligible_previous = sampled_previous_membership.sum(axis=1)
        eligible_union = eligible_now + eligible_previous - (sampled_membership & sampled_previous_membership).sum(axis=1)

        current_by_version: dict[str, np.ndarray] = {}
        previous_by_version: dict[str, np.ndarray] = {}
        native_counts: dict[str, np.ndarray] = {}
        all_versions_finite = np.ones((expected_periods, len(symbols)), dtype=bool)
        for version_id in version_ids:
            current = matrices[version_id][sample_positions]
            previous = np.full_like(current, np.nan)
            if has_previous.any():
                previous[np.flatnonzero(has_previous)] = matrices[version_id][previous_positions[has_previous]]
            current_by_version[version_id] = current
            previous_by_version[version_id] = previous
            finite_pair = np.isfinite(current) & np.isfinite(previous)
            native_common = sampled_membership & sampled_previous_membership & finite_pair
            native_counts[version_id] = native_common.sum(axis=1).astype(np.int64)
            all_versions_finite &= finite_pair
            common_ratio = np.divide(native_counts[version_id], eligible_union,
                                     out=np.full(expected_periods, np.nan, dtype=float), where=eligible_union > 0)
            native_valid = (native_counts[version_id] >= spec.min_symbols) & ~boundary
            result["native_coverage"][version_id][str(delta)] = finite({
                "expected_periods": expected_periods,
                "valid_periods": int(native_valid.sum()),
                "valid_period_share": float(native_valid.mean()) if expected_periods else None,
                "boundary_excluded_periods": int(boundary.sum()),
                "insufficient_common_symbol_periods": int((~boundary & ~native_valid).sum()),
                "mean_eligible_t": float(eligible_now.mean()) if expected_periods else None,
                "mean_eligible_t_minus_delta": float(eligible_previous.mean()) if expected_periods else None,
                "mean_factor_finite_t": float((sampled_membership & np.isfinite(current)).sum(axis=1).mean()) if expected_periods else None,
                "mean_factor_finite_t_minus_delta": float((sampled_previous_membership & np.isfinite(previous)).sum(axis=1).mean()) if expected_periods else None,
                "mean_common_symbols": float(native_counts[version_id].mean()) if expected_periods else None,
                "mean_common_coverage_ratio": float(np.nanmean(common_ratio)) if np.isfinite(common_ratio).any() else None,
            })

        all_version_mask = sampled_membership & sampled_previous_membership & all_versions_finite
        for horizon in HORIZONS:
            current_returns = label_returns[horizon][sample_positions]
            label_mask = label_validity[horizon][sample_positions] & np.isfinite(current_returns)
            label_bounds_valid = label_boundary_validity[horizon][sample_positions]
            common = all_version_mask & label_mask
            counts = common.sum(axis=1).astype(np.int64)
            label_finite_count = (sampled_membership & np.isfinite(current_returns)).sum(axis=1)
            label_boundary_excluded_symbols = (sampled_membership & ~label_bounds_valid).sum(axis=1)
            common_coverage_ratio = np.divide(counts, eligible_union,
                                              out=np.full(expected_periods, np.nan, dtype=float), where=eligible_union > 0)
            sufficient = (counts >= spec.min_symbols) & ~boundary
            return_ranks, return_unique = _rank_rows(current_returns, common)
            metrics: dict[str, dict[str, np.ndarray]] = {}
            unique_current: dict[str, np.ndarray] = {}
            unique_previous: dict[str, np.ndarray] = {}
            spread_valid: dict[str, np.ndarray] = {}
            ic_valid: dict[str, np.ndarray] = {}

            for version_id in version_ids:
                current_ranks, current_unique = _rank_rows(current_by_version[version_id], common)
                previous_ranks, previous_unique = _rank_rows(previous_by_version[version_id], common)
                displacement = _rank_displacement(current_ranks, previous_ranks, common, counts)
                displacement[~sufficient] = np.nan
                rank_ic = _correlation_from_ranks(current_ranks, return_ranks, common)
                ic_ok = sufficient & (current_unique >= spec.groups) & (return_unique > 1)
                rank_ic[~ic_ok] = np.nan
                spread, spread_ok = _directional_spread(
                    current_ranks, current_returns, common, counts, current_unique, spec, direction)
                metrics[version_id] = {
                    "rank_ic": rank_ic,
                    "directional_rank_ic": rank_ic * direction,
                    "directional_spread": spread,
                    "rank_displacement": displacement,
                }
                unique_current[version_id] = current_unique
                unique_previous[version_id] = previous_unique
                ic_valid[version_id] = np.isfinite(rank_ic)
                spread_valid[version_id] = spread_ok

            common_ic = sufficient & (return_unique > 1)
            for version_id in version_ids:
                common_ic &= unique_current[version_id] >= spec.groups
            common_spread = sufficient.copy()
            for version_id in version_ids:
                common_spread &= spread_valid[version_id]
            for version_id in version_ids:
                version_arrays = metrics[version_id]
                valid_counts = {
                    "rank_ic_valid_periods": int(np.isfinite(version_arrays["rank_ic"]).sum()),
                    "directional_spread_valid_periods": int(np.isfinite(version_arrays["directional_spread"]).sum()),
                    "rank_displacement_valid_periods": int(np.isfinite(version_arrays["rank_displacement"]).sum()),
                }
                common_ties = (unique_current[version_id] < counts) & sufficient
                previous_ties = (unique_previous[version_id] < counts) & sufficient
                version_coverage = finite({
                    "expected_periods": expected_periods,
                    "all_version_common_periods": int(sufficient.sum()),
                    "all_version_common_period_share": float(sufficient.mean()) if expected_periods else None,
                    "summary_support": "all-version finite factor and label assets; each metric then excludes its own unestimable tie periods",
                    "valid_pair_d_ic_periods": int(common_ic.sum()),
                    "valid_pair_d_ic_share": float(common_ic.mean()) if expected_periods else None,
                    "rank_ic_unavailable_periods": int((sufficient & ~ic_valid[version_id]).sum()),
                    "directional_spread_unavailable_periods": int((sufficient & ~spread_valid[version_id]).sum()),
                    "tie_periods_current": int(common_ties.sum()),
                    "tie_periods_t_minus_delta": int(previous_ties.sum()),
                    "mean_common_symbols": float(counts.mean()) if expected_periods else None,
                    "mean_common_coverage_ratio": float(np.nanmean(common_coverage_ratio)) if np.isfinite(common_coverage_ratio).any() else None,
                    "mean_label_finite_eligible_symbols": float(label_finite_count.mean()) if expected_periods else None,
                    **valid_counts,
                })
                version_summary = {
                    "rank_ic": _summary_hac(version_arrays["rank_ic"], times, spec),
                    "directional_rank_ic": _summary_hac(version_arrays["directional_rank_ic"], times, spec),
                    "directional_spread": _summary_hac(version_arrays["directional_spread"], times, spec),
                    "rank_displacement": _describe(version_arrays["rank_displacement"]),
                }
                stage_values = {
                    metric_name: metric_values for metric_name, metric_values in version_arrays.items()
                }
                result["versions"][version_id]["horizons"].setdefault(str(horizon), {})[str(delta)] = {
                    "summary": version_summary,
                    "coverage": version_coverage,
                    "stages": _stages(times, stage_anchor, start, end, spec, stage_values),
                }

            common_coverage = finite({
                "expected_periods": expected_periods,
                "boundary_excluded_periods": int(boundary.sum()),
                "label_boundary_excluded_periods": int((label_boundary_excluded_symbols > 0).sum()),
                "label_missing_periods": int((label_finite_count == 0).sum()),
                "all_version_common_periods": int(sufficient.sum()),
                "all_version_common_period_share": float(sufficient.mean()) if expected_periods else None,
                "paired_d_ic_valid_periods": int(common_ic.sum()),
                "all_version_spread_valid_periods": int(common_spread.sum()),
                "mean_common_symbols": float(counts.mean()) if expected_periods else None,
                "mean_common_coverage_ratio": float(np.nanmean(common_coverage_ratio)) if np.isfinite(common_coverage_ratio).any() else None,
            })

            for trial_id in version_ids:
                if trial_id == reference_id:
                    continue
                reference_metrics, trial_metrics = metrics[reference_id], metrics[trial_id]
                paired_d = np.where(common_ic,
                                    reference_metrics["rank_displacement"] - trial_metrics["rank_displacement"],
                                    np.nan)
                paired_ic = np.where(common_ic,
                                     direction * (trial_metrics["rank_ic"] - reference_metrics["rank_ic"]),
                                     np.nan)
                spread_pair_valid = common_spread
                paired_spread = np.where(
                    spread_pair_valid,
                    trial_metrics["directional_spread"] - reference_metrics["directional_spread"],
                    np.nan,
                )
                paired_periods = []
                for i, timestamp in enumerate(times):
                    paired_periods.append({
                        "timestamp": timestamp,
                        "status": ("boundary_excluded" if boundary[i] else
                                   "label_boundary_excluded" if label_boundary_excluded_symbols[i] > 0 and counts[i] < spec.min_symbols else
                                   "label_missing" if label_finite_count[i] == 0 else
                                   "insufficient_common_symbols" if counts[i] < spec.min_symbols else
                                   "rank_ic_unavailable" if not common_ic[i] else "computed"),
                        "common_symbols": int(counts[i]),
                        "common_coverage_ratio": common_coverage_ratio[i],
                        "label_finite_eligible_symbols": int(label_finite_count[i]),
                        "label_boundary_excluded_symbols": int(label_boundary_excluded_symbols[i]),
                        "tie_periods_current": {key: bool(unique_current[key][i] < counts[i]) for key in version_ids},
                        "tie_periods_t_minus_delta": {key: bool(unique_previous[key][i] < counts[i]) for key in version_ids},
                        "valid_d_ic_pair": bool(common_ic[i]),
                        "displacement_reference": reference_metrics["rank_displacement"][i] if common_ic[i] else np.nan,
                        "displacement_trial": trial_metrics["rank_displacement"][i] if common_ic[i] else np.nan,
                        "paired_displacement_improvement": paired_d[i],
                        "rank_ic_reference": reference_metrics["rank_ic"][i] if common_ic[i] else np.nan,
                        "rank_ic_trial": trial_metrics["rank_ic"][i] if common_ic[i] else np.nan,
                        "paired_directional_rank_ic_change": paired_ic[i],
                        "valid_spread_pair": bool(spread_pair_valid[i]),
                        "directional_spread_reference": reference_metrics["directional_spread"][i] if spread_pair_valid[i] else np.nan,
                        "directional_spread_trial": trial_metrics["directional_spread"][i] if spread_pair_valid[i] else np.nan,
                        "paired_directional_spread_change": paired_spread[i],
                    })
                pair_stages = _stages(times, stage_anchor, start, end, spec, {
                    "paired_displacement_improvement": paired_d,
                    "paired_directional_rank_ic_change": paired_ic,
                    "paired_directional_spread_change": paired_spread,
                })
                def on_pair_support(version_metric: dict[str, np.ndarray]) -> dict[str, Any]:
                    d = np.where(common_ic, version_metric["rank_displacement"], np.nan)
                    rank_ic = np.where(common_ic, version_metric["rank_ic"], np.nan)
                    directional_ic = np.where(common_ic, version_metric["directional_rank_ic"], np.nan)
                    spread = np.where(common_spread, version_metric["directional_spread"], np.nan)
                    return {
                        "summary": {
                            "rank_displacement": _summary_hac(d, times, spec),
                            "rank_ic": _summary_hac(rank_ic, times, spec),
                            "directional_rank_ic": _summary_hac(directional_ic, times, spec),
                            "directional_spread": _summary_hac(spread, times, spec),
                        },
                        "coverage": {
                            "paired_d_ic_valid_periods": int(common_ic.sum()),
                            "all_version_spread_valid_periods": int(common_spread.sum()),
                            "expected_periods": expected_periods,
                        },
                        "stages": _stages(times, stage_anchor, start, end, spec, {
                            "rank_displacement": d,
                            "rank_ic": rank_ic,
                            "directional_rank_ic": directional_ic,
                            "directional_spread": spread,
                        }),
                    }
                pair_result = {
                    "reference_id": reference_id,
                    "reference_on_pair_support": on_pair_support(reference_metrics),
                    "trial_on_pair_support": on_pair_support(trial_metrics),
                    "summary": {
                        "paired_displacement_improvement": _summary_hac(paired_d, times, spec),
                        "paired_directional_rank_ic_change": _summary_hac(paired_ic, times, spec),
                        "paired_directional_spread_change": _summary_hac(paired_spread, times, spec),
                    },
                    "coverage": {**common_coverage,
                                 "spread_pair_valid_periods": int(spread_pair_valid.sum()),
                                 "spread_pair_valid_share": float(spread_pair_valid.mean()) if expected_periods else None},
                    "periods": finite(paired_periods),
                    "stages": pair_stages,
                }
                result["pair_effects"][trial_id]["horizons"].setdefault(str(horizon), {})[str(delta)] = finite(pair_result)

    return finite(result)
