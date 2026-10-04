"""A-only factor similarity landscape for ideation context.

This module describes relationships among formulas already computed in A. It
does not inspect labels, B/C evidence, or the formal idea pool.
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from crypto_quant.features.factor_expressions import compile_expression, evaluate_expression
from crypto_quant.features.factor_inputs import (
    FIELD_BY_NAME,
    INPUT_COLUMNS,
    FactorInputPanel,
    validate_universe,
)
from .contracts import ResearchSpec


def _sampled_universe(panel: FactorInputPanel, spec: ResearchSpec) -> tuple[pd.DatetimeIndex, pd.Series]:
    start, end = spec.bounds("A")
    universe = validate_universe(panel.universe)
    if not panel.values.index.equals(universe.index) or not universe.equals(panel.universe):
        raise ValueError("factor landscape requires canonical values and explicit universe indices")
    if set(panel.values.columns) != set(INPUT_COLUMNS):
        raise ValueError("factor landscape requires exactly the shared input-field contract")
    hours = universe.index.get_level_values("timestamp")
    if hours.min() > start - pd.Timedelta(hours=spec.max_lookback_hours):
        raise ValueError("factor landscape panel is missing declared A warm-up hours")
    if hours.max() != end - pd.Timedelta(hours=1):
        raise ValueError("factor landscape panel must stop at the end of its A segment")
    sampled_times = pd.date_range(start, end, freq=f"{spec.sample_hours}h", inclusive="left")
    in_a = (hours >= start) & (hours < end)
    return sampled_times, universe.loc[in_a]


def _finite_ratio(valid: int, total: int) -> float | None:
    return float(valid / total) if total else None


def _pairwise_pearson(left: np.ndarray, right: np.ndarray, eligible: np.ndarray,
                      min_symbols: int, min_periods: int) -> dict[str, Any]:
    common = eligible & np.isfinite(left) & np.isfinite(right)
    common_counts = common.sum(axis=1)
    enough_symbols = common_counts >= min_symbols
    safe_counts = np.maximum(common_counts, 1)
    left_min = np.min(np.where(common, left, np.inf), axis=1)
    left_max = np.max(np.where(common, left, -np.inf), axis=1)
    right_min = np.min(np.where(common, right, np.inf), axis=1)
    right_max = np.max(np.where(common, right, -np.inf), axis=1)
    nonconstant = (left_max > left_min) & (right_max > right_min)
    left_mean = np.where(common, left, 0.0).sum(axis=1) / safe_counts
    right_mean = np.where(common, right, 0.0).sum(axis=1) / safe_counts
    left_centered = np.where(common, left - left_mean[:, None], 0.0)
    right_centered = np.where(common, right - right_mean[:, None], 0.0)
    cross = (left_centered * right_centered).sum(axis=1)
    left_norm = np.square(left_centered).sum(axis=1)
    right_norm = np.square(right_centered).sum(axis=1)
    denominator = np.sqrt(left_norm * right_norm)
    valid = enough_symbols & nonconstant & np.isfinite(denominator) & (denominator > 0)
    constant = enough_symbols & ~nonconstant
    undefined = enough_symbols & nonconstant & ~valid
    correlations = np.full(len(left), np.nan, dtype=float)
    correlations[valid] = np.clip(cross[valid] / denominator[valid], -1.0, 1.0)
    valid_periods = int(valid.sum())
    mean = float(correlations[valid].mean()) if valid_periods >= min_periods else None
    return {
        "valid_periods": valid_periods,
        "periods_with_minimum_common_cross_section": int(enough_symbols.sum()),
        "insufficient_cross_section_hours": int((~enough_symbols).sum()),
        "constant_pair_hours": int(constant.sum()),
        "undefined_pearson_hours": int(undefined.sum()),
        "mean": mean,
        "distance": float(1.0 - abs(mean)) if mean is not None else None,
    }


def _member_metadata(panel: FactorInputPanel, member: dict[str, Any],
                     sampled_universe: pd.Series, spec: ResearchSpec) -> dict[str, Any]:
    if not isinstance(member, dict) or "candidate_ref" not in member:
        raise ValueError("landscape member requires candidate_ref")
    candidate_ref = member["candidate_ref"]
    if not isinstance(candidate_ref, str) or not candidate_ref.strip():
        raise ValueError("candidate_ref must be nonempty text")
    required = {"definition", "calculation", "A_evaluation", "final_decision", "evidence_refs"}
    if not required <= member.keys():
        raise ValueError(f"{candidate_ref}: landscape member is missing fields: {sorted(required - member.keys())}")
    definition = member["definition"]
    if not isinstance(definition, dict) or not isinstance(definition.get("expression"), str):
        raise ValueError(f"{candidate_ref}: definition.expression must be text")
    calculation = member["calculation"]
    if calculation is not None and not isinstance(calculation, dict):
        raise ValueError(f"{candidate_ref}: calculation must be an object or null")
    calculation_status = "not_calculated" if calculation is None else calculation.get("status")
    if calculation is not None and (not isinstance(calculation_status, str) or not calculation_status):
        raise ValueError(f"{candidate_ref}: calculation.status must be nonempty text")
    duplicate_of = None
    if calculation_status == "duplicate":
        duplicate_of = calculation.get("duplicate_of")
        if not isinstance(duplicate_of, str) or not duplicate_of.strip():
            raise ValueError(f"{candidate_ref}: duplicate calculation requires duplicate_of")

    executed = calculation.get("executed_expression") if calculation else None
    if calculation_status == "computed" and not isinstance(executed, dict):
        raise ValueError(f"{candidate_ref}: computed calculation is missing executed_expression")
    if executed is not None and not isinstance(executed, dict):
        raise ValueError(f"{candidate_ref}: executed_expression must be an object or null")
    raw_expression = (
        executed.get("expression") if isinstance(executed, dict) else definition["expression"]
    )
    if not isinstance(raw_expression, str) or not raw_expression.strip():
        raise ValueError(f"{candidate_ref}: formula expression must be nonempty text")

    compiled = None
    compile_error = None
    try:
        if len(raw_expression) > spec.max_formula_nodes * 100:
            raise ValueError("formula exceeds current A contract expression-length budget")
        compiled = compile_expression(raw_expression)
        if sum(1 for _ in ast.walk(compiled.tree)) > spec.max_formula_nodes:
            raise ValueError("formula exceeds current A contract node budget")
        if compiled.lookback_hours > spec.max_lookback_hours:
            raise ValueError(
                f"lookback {compiled.lookback_hours}h exceeds current A contract maximum "
                f"{spec.max_lookback_hours}h"
            )
    except ValueError as exc:
        compile_error = str(exc)
    if compiled is not None and compile_error is None and isinstance(executed, dict):
        if executed.get("expanded_expression") != compiled.expanded_expression:
            raise ValueError(f"{candidate_ref}: executed formula differs from current DSL compilation")
        if executed.get("fields") != list(compiled.fields):
            raise ValueError(f"{candidate_ref}: executed fields differ from current DSL compilation")
        if executed.get("lookback_hours") != compiled.lookback_hours:
            raise ValueError(f"{candidate_ref}: executed lookback differs from current DSL compilation")

    source_refs = member["evidence_refs"]
    if not isinstance(source_refs, list) or any(not isinstance(ref, str) for ref in source_refs):
        raise ValueError(f"{candidate_ref}: evidence_refs must be a list of strings")

    fields = list(compiled.fields) if compiled is not None else []
    field_coverage: dict[str, dict[str, Any]] = {}
    eligible_rows = int(sampled_universe.sum())
    for field in fields:
        raw = pd.to_numeric(panel.values[field].loc[sampled_universe.index], errors="raise")
        valid = np.isfinite(raw.to_numpy(dtype=float)) & sampled_universe.to_numpy(dtype=bool)
        valid_rows = int(valid.sum())
        field_coverage[field] = {
            "eligible_sample_rows": eligible_rows,
            "finite_eligible_rows": valid_rows,
            "coverage_ratio": _finite_ratio(valid_rows, eligible_rows),
        }

    metadata = {
        "candidate_ref": candidate_ref,
        "definition": definition,
        "calculation_status": calculation_status,
        "source_duplicate_of": duplicate_of,
        "source_A_evaluation": member["A_evaluation"],
        "source_final_decision": member["final_decision"],
        "source_evidence_refs": source_refs,
        "formula": {
            "submitted_expression": definition["expression"],
            "executed_expression": executed.get("expression") if isinstance(executed, dict) else None,
            "expanded_expression": compiled.expanded_expression if compiled is not None else None,
            "fields": fields,
            "field_meanings": {field: FIELD_BY_NAME[field].meaning for field in fields},
            "lookback_hours": compiled.lookback_hours if compiled is not None else None,
        },
        "current_A_input_field_data_coverage": field_coverage,
        "_compiled": compiled,
        "_compile_error": compile_error,
        "_executed": executed,
    }
    return metadata


def _clean_member(member: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in member.items() if not key.startswith("_")}


def build_factor_landscape(
    panel: FactorInputPanel,
    spec: ResearchSpec,
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an A-only, sampled factor inventory and complete HAC tree when supported.

    Every computed formula is evaluated against this call's current panel and
    explicit universe. Archived A evaluation summaries remain attached to
    their source candidate and are never pooled into the current correlations.
    """
    if not isinstance(members, list):
        raise ValueError("members must be a list")
    sampled_times, sampled_universe = _sampled_universe(panel, spec)
    min_symbols = max(3, spec.min_symbols)
    sampled_index = pd.MultiIndex.from_product(
        [sampled_times, sampled_universe.index.get_level_values("symbol").unique()],
        names=["timestamp", "symbol"],
    )
    sampled_universe = panel.universe.reindex(sampled_index)
    if sampled_universe.isna().any():
        raise ValueError("explicit universe does not cover all A sample timestamps and symbols")
    sampled_universe = sampled_universe.astype(bool)
    universe_matrix = sampled_universe.to_numpy(dtype=bool).reshape(len(sampled_times), -1)

    prepared = []
    for member in members:
        row = _member_metadata(panel, member, sampled_universe, spec)
        prepared.append(row)
    prepared.sort(key=lambda row: row["candidate_ref"])
    refs = [row["candidate_ref"] for row in prepared]
    if len(refs) != len(set(refs)):
        raise ValueError("landscape candidate_ref values must be unique")

    formula_groups: dict[str, list[dict[str, Any]]] = {}
    for row in prepared:
        if (row["calculation_status"] != "computed" or row["_executed"] is None
                or row["_compile_error"] is not None):
            continue
        if row["_compiled"] is not None:
            formula_groups.setdefault(row["_compiled"].expanded_expression, []).append(row)

    canonical_rows: list[dict[str, Any]] = []
    for expression, sources in sorted(formula_groups.items()):
        canonical = sources[0]
        canonical["source_members"] = [source["candidate_ref"] for source in sources]
        canonical["source_evidence_refs"] = sorted({
            ref for source in sources for ref in source["source_evidence_refs"]
        })
        canonical["source_count"] = len(sources)
        values = evaluate_expression(expression, panel).values.reindex(sampled_universe.index).to_numpy(dtype=float)
        values_matrix = values.reshape(len(sampled_times), -1)
        eligible = sampled_universe.to_numpy(dtype=bool)
        finite_matrix = np.isfinite(values_matrix) & universe_matrix
        valid_rows = int(finite_matrix.sum())
        valid_by_time = finite_matrix.sum(axis=1)
        eligible_by_time = universe_matrix.sum(axis=1)
        adequate_by_time = eligible_by_time >= min_symbols
        adequately_covered = int((valid_by_time >= min_symbols).sum())
        row_min = np.min(np.where(finite_matrix, values_matrix, np.inf), axis=1)
        row_max = np.max(np.where(finite_matrix, values_matrix, -np.inf), axis=1)
        cross_section_variation = (adequate_by_time & (valid_by_time >= min_symbols)
                                   & (row_max > row_min))
        variation_hours = int(cross_section_variation.sum())
        canonical["current_A_value_coverage"] = {
            "sample_hours": int(len(sampled_times)),
            "eligible_sample_rows": int(eligible.sum()),
            "finite_eligible_rows": valid_rows,
            "coverage_ratio": _finite_ratio(valid_rows, int(eligible.sum())),
            "hours_with_minimum_cross_section": adequately_covered,
            "hours_with_cross_sectional_variation": variation_hours,
            "required_hours": int(spec.min_periods),
            "minimum_symbols_per_hour": min_symbols,
        }
        canonical["_adequate_hours"] = adequately_covered
        canonical["_variation_hours"] = variation_hours
        canonical["_valid_rows"] = valid_rows
        canonical["_constant"] = bool(variation_hours == 0)
        canonical["_values_matrix"] = values_matrix
        canonical_rows.append(canonical)
    canonical_rows.sort(key=lambda row: row["candidate_ref"])

    for row in prepared:
        if row["calculation_status"] == "duplicate":
            row["inventory_status"] = "duplicate_candidate"
            row["duplicate_of"] = row["source_duplicate_of"]
            row["reason"] = "candidate was deduplicated before A calculation; no execution result exists"
            continue
        if row["calculation_status"] != "computed":
            row["inventory_status"] = (
                "not_calculated" if row["calculation_status"] in {None, "not_calculated"}
                else "calculation_failed"
            )
            row["reason"] = (
                "A calculation did not complete; the candidate remains in the research inventory"
                if row["calculation_status"] not in {None, "not_calculated"}
                else "A calculation has not been recorded"
            )
            continue
        if row["_compile_error"] is not None:
            row["inventory_status"] = "incompatible_with_current_A_contract"
            row["reason"] = row["_compile_error"]
            continue
        canonical = formula_groups[row["_compiled"].expanded_expression][0]
        row["canonical_ref"] = canonical["candidate_ref"]
        row["source_members"] = canonical["source_members"]
        if canonical["candidate_ref"] != row["candidate_ref"]:
            row["inventory_status"] = "duplicate_formula"
            row["duplicate_of"] = canonical["candidate_ref"]
            row["reason"] = "exact same expanded execution formula; represented by canonical_ref in HAC"
            row["current_A_value_coverage"] = canonical["current_A_value_coverage"]
            continue
        if canonical["_valid_rows"] == 0:
            row["inventory_status"] = "no_finite_A_output"
            row["reason"] = "formula produced no finite values on eligible A samples"
        elif canonical["_adequate_hours"] < spec.min_periods:
            row["inventory_status"] = "insufficient_A_formula_coverage"
            row["reason"] = (
                f"only {canonical['_adequate_hours']} sampled A hours have at least {min_symbols} "
                f"finite eligible values; contract requires {spec.min_periods} hours"
            )
        elif canonical["_constant"]:
            row["inventory_status"] = "constant_factor"
            row["reason"] = "formula has no nonconstant eligible cross-sections on the sampled A interval"
        elif canonical["_variation_hours"] < spec.min_periods:
            row["inventory_status"] = "insufficient_A_cross_sectional_variation"
            row["reason"] = (
                f"only {canonical['_variation_hours']} sampled A hours have nonconstant cross-sections; "
                f"contract requires {spec.min_periods} valid hours"
            )
        else:
            row["inventory_status"] = "eligible_for_similarity"
            row["reason"] = None

    eligible_rows = [
        row for row in canonical_rows
        if row.get("inventory_status") == "eligible_for_similarity"
    ]
    # Status for non-canonical duplicate sources follows their canonical formula,
    # while retaining the per-source archival evaluation and evidence references.
    canonical_by_ref = {row["candidate_ref"]: row for row in canonical_rows}
    for row in prepared:
        if row.get("inventory_status") == "duplicate_formula":
            canonical = canonical_by_ref[row["duplicate_of"]]
            row["canonical_inventory_status"] = canonical["inventory_status"]
            if canonical["inventory_status"] != "eligible_for_similarity":
                row["reason"] = (
                    f"exact duplicate of {row['duplicate_of']}, whose current A status is "
                    f"{canonical['inventory_status']}: {canonical['reason']}"
                )

    pair_rows: list[dict[str, Any]] = []
    pair_valid_periods: dict[tuple[str, str], int] = {}
    distance_matrix = np.zeros((len(eligible_rows), len(eligible_rows)), dtype=float)
    pair_missing = False
    for left_index, left in enumerate(eligible_rows):
        for right_index in range(left_index + 1, len(eligible_rows)):
            right = eligible_rows[right_index]
            stats = _pairwise_pearson(left["_values_matrix"], right["_values_matrix"],
                                      universe_matrix, min_symbols, spec.min_periods)
            valid_periods = stats["valid_periods"]
            mean = stats["mean"]
            distance = stats["distance"]
            periods_with_common_cross_section = stats["periods_with_minimum_common_cross_section"]
            insufficient_cross_section_hours = stats["insufficient_cross_section_hours"]
            constant_pair_hours = stats["constant_pair_hours"]
            undefined_pearson_hours = stats["undefined_pearson_hours"]
            if distance is None:
                pair_missing = True
                reasons = []
                if periods_with_common_cross_section < spec.min_periods:
                    reasons.append("insufficient_common_cross_section_hours")
                if constant_pair_hours:
                    reasons.append("constant_pair_cross_sections")
                if undefined_pearson_hours:
                    reasons.append("undefined_pearson_statistics")
                if not reasons:
                    reasons.append("insufficient_valid_pearson_hours")
            else:
                distance_matrix[left_index, right_index] = distance
                distance_matrix[right_index, left_index] = distance
                reasons = []
            pair_valid_periods[(left["candidate_ref"], right["candidate_ref"])] = valid_periods
            pair_rows.append({
                "left_ref": left["candidate_ref"],
                "right_ref": right["candidate_ref"],
                "left_source_refs": left["source_members"],
                "right_source_refs": right["source_members"],
                "status": "computed" if distance is not None else "insufficient_pair_coverage",
                "valid_periods": valid_periods,
                "required_periods": int(spec.min_periods),
                "periods_with_minimum_common_cross_section": periods_with_common_cross_section,
                "minimum_common_symbols": min_symbols,
                "insufficient_cross_section_hours": insufficient_cross_section_hours,
                "constant_pair_hours": constant_pair_hours,
                "undefined_pearson_hours": undefined_pearson_hours,
                "mean_hourly_cross_sectional_pearson": mean,
                "distance_1_minus_abs_mean_pearson": distance,
                "reason_codes": reasons,
            })

    tree = None
    if len(eligible_rows) >= 2 and not pair_missing:
        linkage_matrix = linkage(squareform(distance_matrix, checks=True), method="average")
        node_members: dict[int, list[int]] = {
            index: [index] for index in range(len(eligible_rows))
        }
        node_pair_stats: dict[int, tuple[int, int | None, int]] = {
            index: (0, None, 0) for index in range(len(eligible_rows))
        }
        merges = []
        for step, merge in enumerate(linkage_matrix, start=1):
            left_node, right_node = int(merge[0]), int(merge[1])
            left_indexes = node_members[left_node]
            right_indexes = node_members[right_node]
            merged_indexes = sorted(left_indexes + right_indexes)
            node_id = len(eligible_rows) + step - 1
            node_members[node_id] = merged_indexes
            branch_members = [eligible_rows[index] for index in merged_indexes]
            height = float(merge[2])
            cross_periods = []
            for left_index in left_indexes:
                for right_index in right_indexes:
                    left_ref = eligible_rows[left_index]["candidate_ref"]
                    right_ref = eligible_rows[right_index]["candidate_ref"]
                    key = (left_ref, right_ref) if left_ref < right_ref else (right_ref, left_ref)
                    cross_periods.append(pair_valid_periods[key])
            left_count, left_min, left_sum = node_pair_stats[left_node]
            right_count, right_min, right_sum = node_pair_stats[right_node]
            known_mins = [value for value in (left_min, right_min) if value is not None]
            known_mins.extend(cross_periods)
            pair_count = left_count + right_count + len(cross_periods)
            period_sum = left_sum + right_sum + sum(cross_periods)
            node_pair_stats[node_id] = (pair_count, min(known_mins), period_sum)
            merges.append({
                "step": step,
                "node_id": node_id,
                "left_node_id": left_node,
                "right_node_id": right_node,
                "height": height,
                "similarity_1_minus_height": float(1 - height),
                "representative_ref": min(item["candidate_ref"] for item in branch_members),
                "member_refs": [item["candidate_ref"] for item in branch_members],
                "member_count": len(branch_members),
                "source_count": sum(item["source_count"] for item in branch_members),
                "minimum_pair_valid_periods": min(known_mins) if known_mins else None,
                "mean_pair_valid_periods": period_sum / pair_count if pair_count else None,
            })
        tree = {
            "method": "average",
            "distance": "1 - abs(mean hourly cross-sectional Pearson)",
            "representative_policy": "lexicographically first candidate_ref in branch; identifier only, not a quality ranking",
            "leaf_order": [row["candidate_ref"] for row in eligible_rows],
            "root_node_id": len(eligible_rows) + len(merges) - 1,
            "scipy_linkage": linkage_matrix.tolist(),
            "merges": merges,
        }

    for row in prepared:
        if row.get("inventory_status") in {"eligible_for_similarity", "duplicate_formula"}:
            canonical = canonical_by_ref[row["canonical_ref"]]
            row["current_A_value_coverage"] = canonical.get("current_A_value_coverage")
            row["canonical_inventory_status"] = canonical.get("inventory_status")
    cleaned = [_clean_member(row) for row in prepared]
    status_counts = dict(sorted(Counter(row["inventory_status"] for row in prepared).items()))
    attempted_parseable = [
        row for row in prepared
        if row["calculation_status"] not in {"not_calculated", "duplicate"}
        and row["_compiled"] is not None
    ]
    field_use_counts = Counter(
        field for row in attempted_parseable for field in row["_compiled"].fields
    )

    if not members:
        status = "no_members"
    elif not eligible_rows:
        status = "no_clusterable_members"
    elif len(eligible_rows) == 1:
        status = "single_factor_no_tree"
    elif pair_missing:
        status = "insufficient_pair_coverage"
    else:
        status = "computed"

    return {
        "status": status,
        "contract": {
            "data_segment": "A",
            "start": spec.a_start,
            "end_exclusive": spec.b_start,
            "sample_hours": spec.sample_hours,
            "sampled_hours": int(len(sampled_times)),
            "minimum_symbols_per_hour": min_symbols,
            "minimum_valid_hours_per_member_and_pair": spec.min_periods,
            "correlation": "mean of hourly pairwise-complete cross-sectional Pearson r",
            "distance": "1 - abs(mean hourly cross-sectional Pearson r)",
            "linkage": "average",
            "formula_value_scope": "re-evaluated on this call's current A panel and explicit current A universe",
            "source_metric_scope": "source A evaluation summaries remain attached to each source; not pooled or treated as same-sample comparable",
            "value_coverage_scope": "finite formula values among explicit eligible rows at sampled A timestamps; describes data availability, not research success",
        },
        "members": cleaned,
        "inventory_summary": {
            "source_member_count": len(prepared),
            "members_by_status": status_counts,
            "unique_computed_formula_count": len(formula_groups),
            "duplicate_formula_source_count": sum(
                row["inventory_status"] == "duplicate_formula" for row in prepared
            ),
        },
        "field_inventory": {
            "scope": "field use among parseable formulas with a calculation attempt recorded (including failures); excludes unattempted and pre-calculation duplicate candidates",
            "attempted_parseable_member_count": len(attempted_parseable),
            "field_use_counts": dict(sorted(field_use_counts.items())),
        },
        "pairwise_correlations": pair_rows,
        "missing_pairs": [
            {key: pair[key] for key in (
                "left_ref", "right_ref", "status", "valid_periods", "required_periods", "reason_codes"
            )}
            for pair in pair_rows if pair["status"] != "computed"
        ],
        "tree": tree,
    }
