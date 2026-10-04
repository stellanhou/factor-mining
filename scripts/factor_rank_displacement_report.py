"""Build compact tables, figures, and a readable report from a completed D study.

The script streams the large E1/E2 JSONL logs and uses the compact E3 CSV. It
does not load the large E2/E3 summary JSON files or recompute research metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


HORIZONS = (1, 4, 24)
DELTAS = (1, 4, 24)
WINDOWS = (4, 12, 24)
CLASS_ORDER = (
    "mixed_structure", "volatility_range", "price_trend_reversal",
    "volume_liquidity", "funding_basis", "positioning_ratio",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_top_level_prefix_field(path: Path, wanted: str) -> Any:
    """Read an early top-level JSON field without loading a following large array."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        text = stream.read(4 * 1024 * 1024)
    pos = 0

    def skip_space(position: int) -> int:
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    pos = skip_space(pos)
    if pos >= len(text) or text[pos] != "{":
        raise ValueError(f"expected top-level object in {path}")
    pos += 1
    while True:
        pos = skip_space(pos)
        key, pos = decoder.raw_decode(text, pos)
        pos = skip_space(pos)
        if text[pos] != ":":
            raise ValueError(f"malformed JSON object in {path}")
        value, pos = decoder.raw_decode(text, skip_space(pos + 1))
        if key == wanted:
            return value
        pos = skip_space(pos)
        if text[pos] == ",":
            pos += 1
            continue
        if text[pos] == "}":
            break
        raise ValueError(f"could not find top-level field {wanted!r} before large values in {path}")
    raise KeyError(f"top-level field {wanted!r} absent from {path}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_preregistration(run_dir: Path) -> dict[str, Any]:
    contract = load_json(run_dir / "preregistration.json")
    expected = contract.pop("preregistration_sha256", None)
    actual = hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    if expected != actual:
        raise ValueError("preregistration hash mismatch")
    contract["preregistration_sha256"] = expected
    parser = contract.get("factor_value_csv_numeric_parser", {})
    if parser.get("float_precision") != "round_trip":
        raise ValueError("readable report requires the corrected round-trip precision run")
    return contract


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": None, "p10": None, "median": None, "p90": None}
    return {"n": int(len(array)), "mean": float(array.mean()),
            "p10": float(np.quantile(array, 0.10)), "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.90))}


def parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    value = float(value)
    if not math.isfinite(value):
        return "—"
    if value != 0 and abs(value) < 10 ** (-digits):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def ci_text(low: Any, high: Any) -> str:
    if low is None or high is None:
        return "—"
    return f"[{fmt(low)}, {fmt(high)}]"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def e1_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    for line in (run_dir / "e1_factors.jsonl").open(encoding="utf-8"):
        if line.strip():
            records.append(json.loads(line))
    if len(records) != 576 or len({record["factor_id"] for record in records}) != 576:
        raise ValueError("E1 report input must contain exactly 576 unique factor records")
    d_values = {str(delta): [] for delta in DELTAS}
    ic_values = {str(horizon): [] for horizon in HORIZONS}
    for record in records:
        for delta in DELTAS:
            value = record["rank_displacement"][str(delta)]["summary"]["mean"]
            if value is not None:
                d_values[str(delta)].append(float(value))
        for horizon in HORIZONS:
            value = record["native_A_horizon_evidence"][str(horizon)].get("directional_rank_ic_mean")
            if value is not None:
                ic_values[str(horizon)].append(float(value))
    coverage = [record["native_A_factor_coverage"] for record in records]
    grid_counts = {item["grid_rows"] for item in coverage}
    eligible_counts = {item["eligible_rows"] for item in coverage}
    if grid_counts != {526320} or eligible_counts != {525183}:
        raise ValueError(f"E1 A-only coverage differs from the frozen period: grid={grid_counts}, eligible={eligible_counts}")
    return records, {"native_D_by_delta": {key: distribution(values) for key, values in d_values.items()},
                     "directional_IC_by_horizon": {key: distribution(values) for key, values in ic_values.items()},
                     "native_A_coverage": {"grid_rows_per_factor": 526320, "eligible_rows_per_factor": 525183,
                                           "eligible_finite_rows": distribution(
                                               [float(item["eligible_finite_rows"]) for item in coverage])}}


def e2_parent_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    scatter: dict[tuple[str, str], dict[str, Any]] = {}
    parents_seen: set[str] = set()
    baseline_by_cell: dict[tuple[str, str, str], tuple[float, float, int]] = {}
    for line in (run_dir / "e2_factors.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        parent = json.loads(line)
        factor_id = parent["factor_id"]
        if factor_id in parents_seen:
            raise ValueError(f"duplicate E2 parent {factor_id}")
        parents_seen.add(factor_id)
        full = parent["segments"]["full_A"]["summary"]
        for version in parent["applicable_versions"]:
            if version == "original":
                continue
            window = int(version.removeprefix("smooth_").removesuffix("h"))
            if window not in WINDOWS:
                raise ValueError(f"unexpected E2 window {version}")
            for horizon in HORIZONS:
                h = str(horizon)
                for delta in DELTAS:
                    d = str(delta)
                    pair = full["pair_effects"][version]["horizons"][h][d]
                    effect = pair["summary"]
                    support = pair["coverage"]
                    reference = pair["reference_on_pair_support"]["summary"]
                    trial = pair["trial_on_pair_support"]["summary"]
                    d_effect = effect["paired_displacement_improvement"]
                    ic_effect = effect["paired_directional_rank_ic_change"]
                    spread_effect = effect["paired_directional_spread_change"]
                    ref_d = reference["rank_displacement"]["mean"]
                    trial_d = trial["rank_displacement"]["mean"]
                    ref_ic = reference["directional_rank_ic"]["mean"]
                    trial_ic = trial["directional_rank_ic"]["mean"]
                    ref_spread = reference["directional_spread"]["mean"]
                    trial_spread = trial["directional_spread"]["mean"]
                    if d_effect["mean"] is not None and not math.isclose(
                            float(ref_d) - float(trial_d), float(d_effect["mean"]), rel_tol=0, abs_tol=1e-10):
                        raise ValueError(f"E2 D pair-support coordinate mismatch: {factor_id} {version} H={h} D={d}")
                    if ic_effect["mean"] is not None and not math.isclose(
                            float(trial_ic) - float(ref_ic), float(ic_effect["mean"]), rel_tol=0, abs_tol=1e-10):
                        raise ValueError(f"E2 IC pair-support coordinate mismatch: {factor_id} {version} H={h} D={d}")
                    if spread_effect["mean"] is not None and not math.isclose(
                            float(trial_spread) - float(ref_spread), float(spread_effect["mean"]), rel_tol=0, abs_tol=1e-10):
                        raise ValueError(f"E2 spread pair-support coordinate mismatch: {factor_id} {version} H={h} D={d}")
                    valid_periods = int(support["paired_d_ic_valid_periods"])
                    if (d_effect["n"] != valid_periods or ic_effect["n"] != valid_periods
                            or reference["rank_displacement"]["n"] != valid_periods
                            or trial["rank_displacement"]["n"] != valid_periods
                            or reference["directional_rank_ic"]["n"] != valid_periods
                            or trial["directional_rank_ic"]["n"] != valid_periods):
                        raise ValueError(f"E2 D/IC pair-support count mismatch: {factor_id} {version} H={h} D={d}")
                    spread_periods = int(support["spread_pair_valid_periods"])
                    if (spread_effect["n"] != spread_periods
                            or reference["directional_spread"]["n"] != spread_periods
                            or trial["directional_spread"]["n"] != spread_periods):
                        raise ValueError(f"E2 spread pair-support count mismatch: {factor_id} {version} H={h} D={d}")
                    baseline_key = (factor_id, h, d)
                    baseline = (float(ref_d), float(ref_ic), valid_periods)
                    previous = baseline_by_cell.setdefault(baseline_key, baseline)
                    if (not math.isclose(previous[0], baseline[0], rel_tol=0, abs_tol=1e-12)
                            or not math.isclose(previous[1], baseline[1], rel_tol=0, abs_tol=1e-12)
                            or previous[2] != baseline[2]):
                        raise ValueError(f"E2 variants do not share the same all-version baseline support: {factor_id} H={h} D={d}")
                    row = {
                        "factor_id": factor_id, "formula_class": parent["formula_class"],
                        "smooth_window_hours": window, "horizon_hours": horizon, "delta_hours": delta,
                        "common_d_ic_periods": valid_periods, "spread_pair_periods": spread_periods,
                        "original_D_on_pair_support": ref_d, "smooth_D_on_pair_support": trial_d,
                        "D_decrease_original_minus_smooth": d_effect["mean"],
                        "D_decrease_ci_low": (d_effect["ci"] or [None, None])[0],
                        "D_decrease_ci_high": (d_effect["ci"] or [None, None])[1],
                        "D_decrease_n": d_effect["n"],
                        "original_directional_IC_on_pair_support": ref_ic,
                        "smooth_directional_IC_on_pair_support": trial_ic,
                        "directional_IC_change_smooth_minus_original": ic_effect["mean"],
                        "directional_IC_change_ci_low": (ic_effect["ci"] or [None, None])[0],
                        "directional_IC_change_ci_high": (ic_effect["ci"] or [None, None])[1],
                        "directional_IC_change_n": ic_effect["n"],
                        "original_directional_spread_on_pair_support": ref_spread,
                        "smooth_directional_spread_on_pair_support": trial_spread,
                        "directional_spread_change_smooth_minus_original": spread_effect["mean"],
                        "directional_spread_change_ci_low": (spread_effect["ci"] or [None, None])[0],
                        "directional_spread_change_ci_high": (spread_effect["ci"] or [None, None])[1],
                        "directional_spread_change_n": spread_effect["n"],
                    }
                    rows.append(row)
                    cell_key = (h, d)
                    version_entry = scatter.setdefault(cell_key, {"original": {}, "trials": defaultdict(dict)})
                    orig_point = {"factor_id": factor_id, "x": float(ref_ic), "y": float(ref_d)}
                    current = version_entry["original"].setdefault(factor_id, orig_point)
                    if current != orig_point:
                        raise ValueError(f"pair-support original coordinate changed across windows: {factor_id} H={h} D={d}")
                    version_entry["trials"][version][factor_id] = {"x": float(trial_ic), "y": float(trial_d)}
    if len(parents_seen) != 24:
        raise ValueError(f"E2 parent log has {len(parents_seen)} identities, expected 24")
    for window in WINDOWS:
        for horizon in HORIZONS:
            for delta in DELTAS:
                count = sum(row["smooth_window_hours"] == window and row["horizon_hours"] == horizon
                            and row["delta_hours"] == delta for row in rows)
                if count != 21:
                    raise ValueError(f"E2 fixed grid H={horizon} D={delta} W={window} has n={count}, expected 21")
    return rows, scatter


def summarize_e2_grid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for window in WINDOWS:
        for horizon in HORIZONS:
            for delta in DELTAS:
                cell = [row for row in rows if row["smooth_window_hours"] == window
                        and row["horizon_hours"] == horizon and row["delta_hours"] == delta]
                d_values = [float(row["D_decrease_original_minus_smooth"]) for row in cell
                            if row["D_decrease_original_minus_smooth"] is not None]
                ic_values = [float(row["directional_IC_change_smooth_minus_original"]) for row in cell
                             if row["directional_IC_change_smooth_minus_original"] is not None]
                spread_values = [float(row["directional_spread_change_smooth_minus_original"]) for row in cell
                                 if row["directional_spread_change_smooth_minus_original"] is not None]
                row = {"smooth_window_hours": window, "horizon_hours": horizon,
                       "delta_hours": delta, "all_smooth_applicable_parent_denominator": 21,
                       "D_effect_n": len(d_values), "D_decrease_positive_parent_count": sum(v > 0 for v in d_values),
                       "D_decrease_negative_parent_count": sum(v < 0 for v in d_values),
                       "D_ci_n": sum(r["D_decrease_ci_low"] is not None and r["D_decrease_ci_high"] is not None
                                     for r in cell),
                       "D_decrease_ci_lower_above_zero_count": sum(
                           (r["D_decrease_ci_low"] is not None and r["D_decrease_ci_low"] > 0) for r in cell),
                       "D_decrease_ci_upper_below_zero_count": sum(
                           (r["D_decrease_ci_high"] is not None and r["D_decrease_ci_high"] < 0) for r in cell),
                       "directional_IC_effect_n": len(ic_values),
                       "directional_IC_change_negative_parent_count": sum(v < 0 for v in ic_values),
                       "directional_IC_ci_n": sum(
                           r["directional_IC_change_ci_low"] is not None
                           and r["directional_IC_change_ci_high"] is not None for r in cell),
                       "directional_IC_ci_upper_below_zero_count": sum(
                           (r["directional_IC_change_ci_high"] is not None and r["directional_IC_change_ci_high"] < 0)
                           for r in cell),
                       "directional_spread_effect_n": len(spread_values),
                       "directional_spread_change_negative_parent_count": sum(v < 0 for v in spread_values),
                       "directional_spread_ci_n": sum(
                           r["directional_spread_change_ci_low"] is not None
                           and r["directional_spread_change_ci_high"] is not None for r in cell),
                       "directional_spread_ci_upper_below_zero_count": sum(
                           (r["directional_spread_change_ci_high"] is not None
                            and r["directional_spread_change_ci_high"] < 0) for r in cell)}
                for metric, values in (("D_decrease", d_values), ("directional_IC_change", ic_values),
                                       ("directional_spread_change", spread_values)):
                    stats = distribution(values)
                    row.update({f"{metric}_{key}": value for key, value in stats.items()})
                summary.append(row)
    return summary


def e3_grid_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = []
    for row in csv.DictReader((run_dir / "e3_effects.csv").open(newline="", encoding="utf-8")):
        parsed = dict(row)
        for key in ("horizon_hours", "delta_hours", "hold_common_periods", "hold_spread_pair_periods"):
            parsed[key] = int(row[key]) if row[key] else None
        for key in ("front_d_improvement", "front_d_ci_low", "front_d_ci_high",
                    "hold_d_improvement", "hold_d_ci_low", "hold_d_ci_high",
                    "hold_directional_ic_change", "hold_directional_ic_ci_low", "hold_directional_ic_ci_high",
                    "hold_directional_spread_change", "hold_directional_spread_ci_low",
                    "hold_directional_spread_ci_high"):
            parsed[key] = parse_optional_float(row[key])
        raw_rows.append(parsed)
    if len(raw_rows) != 189:
        raise ValueError(f"E3 result table has {len(raw_rows)} rows, expected 21 x 9 = 189")
    grid = []
    for horizon in HORIZONS:
        for delta in DELTAS:
            cell = [row for row in raw_rows if row["horizon_hours"] == horizon and row["delta_hours"] == delta]
            if len(cell) != 21:
                raise ValueError(f"E3 H={horizon} D={delta} denominator is {len(cell)}, expected 21")
            d_values = [row["hold_d_improvement"] for row in cell if row["hold_d_improvement"] is not None]
            ic_values = [row["hold_directional_ic_change"] for row in cell
                         if row["hold_directional_ic_change"] is not None]
            spread_values = [row["hold_directional_spread_change"] for row in cell
                             if row["hold_directional_spread_change"] is not None]
            entry = {"horizon_hours": horizon, "delta_hours": delta, "selected_smooth_parent_denominator": 21,
                     "all_parent_denominator": 24, "D_effect_n": len(d_values),
                     "D_decrease_positive_parent_count": sum(value > 0 for value in d_values),
                     "D_decrease_negative_parent_count": sum(value < 0 for value in d_values),
                     "D_ci_n": sum(row["hold_d_ci_low"] is not None and row["hold_d_ci_high"] is not None
                                   for row in cell),
                     "D_decrease_ci_lower_above_zero_count": sum(
                         row["hold_d_ci_low"] is not None and row["hold_d_ci_low"] > 0 for row in cell),
                     "D_decrease_ci_upper_below_zero_count": sum(
                         row["hold_d_ci_high"] is not None and row["hold_d_ci_high"] < 0 for row in cell),
                     "directional_IC_effect_n": len(ic_values),
                     "directional_IC_change_negative_parent_count": sum(value < 0 for value in ic_values),
                     "directional_IC_ci_n": sum(
                         row["hold_directional_ic_ci_low"] is not None
                         and row["hold_directional_ic_ci_high"] is not None for row in cell),
                     "directional_IC_ci_upper_below_zero_count": sum(
                         row["hold_directional_ic_ci_high"] is not None and row["hold_directional_ic_ci_high"] < 0
                         for row in cell),
                     "directional_spread_effect_n": len(spread_values),
                     "directional_spread_change_negative_parent_count": sum(value < 0 for value in spread_values),
                     "directional_spread_ci_n": sum(
                         row["hold_directional_spread_ci_low"] is not None
                         and row["hold_directional_spread_ci_high"] is not None for row in cell),
                     "directional_spread_ci_upper_below_zero_count": sum(
                         row["hold_directional_spread_ci_high"] is not None
                         and row["hold_directional_spread_ci_high"] < 0 for row in cell)}
            for metric, values in (("D_decrease", d_values), ("directional_IC_change", ic_values),
                                   ("directional_spread_change", spread_values)):
                stats = distribution([float(value) for value in values])
                entry.update({f"{metric}_{key}": value for key, value in stats.items()})
            grid.append(entry)
    return raw_rows, grid


def plot_e1(records: list[dict[str, Any]], path: Path) -> None:
    d_data = [[record["rank_displacement"][str(delta)]["summary"]["mean"] for record in records
               if record["rank_displacement"][str(delta)]["summary"]["mean"] is not None] for delta in DELTAS]
    ic_data = [[record["native_A_horizon_evidence"][str(h)].get("directional_rank_ic_mean") for record in records
                if record["native_A_horizon_evidence"][str(h)].get("directional_rank_ic_mean") is not None]
               for h in HORIZONS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].boxplot(d_data, tick_labels=[f"Δ={d}h" for d in DELTAS], showfliers=False, showmeans=True)
    axes[0].set_title("Native D means across the 576 A candidates")
    axes[0].set_ylabel("Rank displacement D")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].boxplot(ic_data, tick_labels=[f"H={h}h" for h in HORIZONS], showfliers=False, showmeans=True)
    axes[1].set_title("Archived directional A rank IC")
    axes[1].set_ylabel("Directional rank IC")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("E1 descriptive factor distributions; formula candidates may be dependent", fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_e1_d_ic_scatter(run_dir: Path, path: Path) -> None:
    records = list(csv.DictReader((run_dir / "e1_scatter.csv").open(newline="", encoding="utf-8")))
    if len(records) != 576 * len(HORIZONS) * len(DELTAS):
        raise ValueError(f"E1 D-IC scatter source has {len(records)} rows, expected 5184")
    cells: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    all_x, all_y = [], []
    for row in records:
        x = parse_optional_float(row["directional_rank_ic_mean"])
        y = parse_optional_float(row["displacement_mean"])
        coverage = parse_optional_float(row["displacement_valid_share"])
        if x is None or y is None or coverage is None:
            continue
        cells[(row["horizon_hours"], row["delta_hours"])].append((x, y, coverage))
        all_x.append(x)
        all_y.append(y)
    if len(all_x) < 500:
        raise ValueError(f"E1 D-IC scatter has too few finite cells: {len(all_x)}")
    xspan = max(all_x) - min(all_x)
    yspan = max(all_y) - min(all_y)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    image = None
    for hi, horizon in enumerate(HORIZONS):
        for di, delta in enumerate(DELTAS):
            ax = axes[hi, di]
            points = cells[(str(horizon), str(delta))]
            image = ax.scatter([point[0] for point in points], [point[1] for point in points],
                               c=[point[2] for point in points], cmap="viridis", vmin=0, vmax=1,
                               s=11, alpha=0.7, edgecolors="none")
            ax.set_xlim(min(all_x) - 0.04 * xspan, max(all_x) + 0.04 * xspan)
            ax.set_ylim(min(all_y) - 0.04 * yspan, max(all_y) + 0.04 * yspan)
            ax.set_title(f"H={horizon}h · Δ={delta}h · n={len(points)}")
            ax.set_xlabel("Directional rank IC")
            if di == 0:
                ax.set_ylabel("Native displacement D")
            ax.grid(alpha=0.18)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02,
                 label="Native D valid-coverage share")
    fig.suptitle("E1 full 576-candidate D-IC grid; native D coverage varies", fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_e2_grid(grid: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    metrics = ("D_decrease_mean", "directional_IC_change_mean")
    for row_index, metric in enumerate(metrics):
        max_abs = max(abs(float(item[metric])) for item in grid if item[metric] is not None)
        norm = TwoSlopeNorm(vcenter=0, vmin=-max_abs, vmax=max_abs) if max_abs else None
        for col_index, window in enumerate(WINDOWS):
            ax = axes[row_index, col_index]
            subset = [item for item in grid if item["smooth_window_hours"] == window]
            matrix = np.full((len(HORIZONS), len(DELTAS)), np.nan)
            count = np.zeros_like(matrix)
            for item in subset:
                i, j = HORIZONS.index(item["horizon_hours"]), DELTAS.index(item["delta_hours"])
                matrix[i, j] = item[metric]
                count[i, j] = item["D_effect_n"] if row_index == 0 else item["directional_IC_effect_n"]
            image = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
            ax.set_xticks(range(len(DELTAS)), [str(d) for d in DELTAS])
            ax.set_yticks(range(len(HORIZONS)), [str(h) for h in HORIZONS])
            ax.set_xlabel("Δ (hours)")
            if row_index == 0:
                ax.set_title(f"Smooth {window}h")
            if col_index == 0:
                ax.set_ylabel(("D decrease\n" if row_index == 0 else "Directional IC change\n") + "H (hours)")
            for i in range(len(HORIZONS)):
                for j in range(len(DELTAS)):
                    item = next(entry for entry in subset if entry["horizon_hours"] == HORIZONS[i]
                                and entry["delta_hours"] == DELTAS[j])
                    count_text = ((f"D↓{item['D_decrease_positive_parent_count']} D↑{item['D_decrease_negative_parent_count']}\n"
                                   f"CI+{item['D_decrease_ci_lower_above_zero_count']}/{item['D_ci_n']} "
                                   f"CI−{item['D_decrease_ci_upper_below_zero_count']}/{item['D_ci_n']}")
                                  if row_index == 0 else
                                  f"IC↓{item['directional_IC_change_negative_parent_count']}/21")
                    ax.text(j, i, f"{matrix[i, j]:+.4f}\nn={int(count[i,j])} · {count_text}",
                            ha="center", va="center", fontsize=8, color="black")
        fig.colorbar(image, ax=axes[row_index, :].tolist(), shrink=0.8, pad=0.02)
    fig.supxlabel("Δ (hours)")
    fig.suptitle("E2 full-A paired effects on all-version common support (21 parents per cell)", fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_pair_support_scatter(scatter: dict[tuple[str, str], dict[str, Any]], path: Path) -> None:
    colors = {"smooth_4h": "#1f77b4", "smooth_12h": "#ff7f0e", "smooth_24h": "#2ca02c"}
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for hi, horizon in enumerate(HORIZONS):
        for di, delta in enumerate(DELTAS):
            ax = axes[hi, di]
            cell = scatter[(str(horizon), str(delta))]
            baselines = cell["original"]
            for factor_id, original in baselines.items():
                for version, points in cell["trials"].items():
                    trial = points[factor_id]
                    ax.plot([original["x"], trial["x"]], [original["y"], trial["y"]],
                            color=colors[version], alpha=0.14, linewidth=0.55)
            ax.scatter([p["x"] for p in baselines.values()], [p["y"] for p in baselines.values()],
                       color="#444444", marker="x", s=18, label="original")
            for version in ("smooth_4h", "smooth_12h", "smooth_24h"):
                points = list(cell["trials"].get(version, {}).values())
                ax.scatter([p["x"] for p in points], [p["y"] for p in points],
                           color=colors[version], s=13, alpha=0.75, label=version)
            ax.set_title(f"H={horizon}h · Δ={delta}h · n={len(baselines)}")
            ax.set_xlabel("Directional rank IC mean")
            if di == 0:
                ax.set_ylabel("Rank displacement D mean")
            ax.grid(alpha=0.18)
            if hi == 0 and di == 0:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle("E2 full-A original/smooth coordinates from each version's paired common support", fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_e3_grid(grid: list[dict[str, Any]], path: Path) -> None:
    metrics = (("D_decrease_mean", "D decrease · positive lowers D"),
               ("directional_IC_change_mean", "Directional IC change · negative declines"),
               ("directional_spread_change_mean", "Directional spread change · negative declines"))
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2), constrained_layout=True)
    for metric_index, (ax, (metric, title)) in enumerate(zip(axes, metrics)):
        max_abs = max(abs(float(item[metric])) for item in grid if item[metric] is not None)
        norm = TwoSlopeNorm(vcenter=0, vmin=-max_abs, vmax=max_abs) if max_abs else None
        matrix = np.full((len(HORIZONS), len(DELTAS)), np.nan)
        for item in grid:
            i, j = HORIZONS.index(item["horizon_hours"]), DELTAS.index(item["delta_hours"])
            matrix[i, j] = item[metric]
        image = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(DELTAS)), [str(d) for d in DELTAS])
        ax.set_yticks(range(len(HORIZONS)), [str(h) for h in HORIZONS])
        ax.set_xlabel("Δ (hours)")
        if metric_index == 0:
            ax.set_ylabel("H (hours)")
        ax.set_title(title)
        for i in range(len(HORIZONS)):
            for j in range(len(DELTAS)):
                cell = next(item for item in grid if item["horizon_hours"] == HORIZONS[i]
                            and item["delta_hours"] == DELTAS[j])
                if metric == "D_decrease_mean":
                    count_text = (f"D↓ {cell['D_decrease_positive_parent_count']} · D↑ {cell['D_decrease_negative_parent_count']}\n"
                                  f"CI+ {cell['D_decrease_ci_lower_above_zero_count']}/{cell['D_ci_n']} · "
                                  f"CI− {cell['D_decrease_ci_upper_below_zero_count']}/{cell['D_ci_n']}")
                    mean_text = f"{matrix[i, j]:+.4f}"
                elif metric == "directional_IC_change_mean":
                    count_text = (f"IC− {cell['directional_IC_change_negative_parent_count']}/21\n"
                                  f"CI− {cell['directional_IC_ci_upper_below_zero_count']}/{cell['directional_IC_ci_n']}")
                    mean_text = f"{matrix[i, j]:+.4f}"
                else:
                    count_text = (f"S− {cell['directional_spread_change_negative_parent_count']}/21\n"
                                  f"CI− {cell['directional_spread_ci_upper_below_zero_count']}/{cell['directional_spread_ci_n']}")
                    mean_text = f"{matrix[i, j]:+.1e}"
                ax.text(j, i, f"{mean_text}\nn=21 · {count_text}",
                        ha="center", va="center", fontsize=7, color="black")
        colorbar_label = {
            "D_decrease_mean": "D improvement (original − smooth)",
            "directional_IC_change_mean": "Directional IC change (smooth − original)",
            "directional_spread_change_mean": "Directional spread change (smooth − original)",
        }[metric]
        fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03, label=colorbar_label)
    fig.suptitle("E3 hold transfer: 21 front-selected executable smooth parents; 3 N/A excluded", fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_e3_front_hold_D(run_dir: Path, path: Path) -> None:
    records = list(csv.DictReader((run_dir / "e3_effects.csv").open(newline="", encoding="utf-8")))
    pairs = []
    values = []
    for row in records:
        front = parse_optional_float(row["front_d_improvement"])
        hold = parse_optional_float(row["hold_d_improvement"])
        if front is None or hold is None:
            continue
        pairs.append((int(row["horizon_hours"]), int(row["delta_hours"]), front, hold))
        values.extend((front, hold))
    if len(pairs) != 189:
        raise ValueError(f"E3 front/hold D source has {len(pairs)} pairs, expected 189")
    limit = max(abs(min(values)), abs(max(values)))
    limit = limit * 1.05 if limit else 1.0
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    for hi, horizon in enumerate(HORIZONS):
        for di, delta in enumerate(DELTAS):
            ax = axes[hi, di]
            cell = [row for row in pairs if row[0] == horizon and row[1] == delta]
            ax.plot([-limit, limit], [-limit, limit], color="#555555", linestyle="--", linewidth=1)
            ax.scatter([row[2] for row in cell], [row[3] for row in cell],
                       color="#2C7FB8", s=19, alpha=0.75)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"H={horizon}h · Δ={delta}h · n={len(cell)}")
            ax.set_xlabel("Front D improvement")
            if di == 0:
                ax.set_ylabel("Hold D improvement")
            ax.grid(alpha=0.18)
    fig.suptitle("E3 paired D improvement in front vs hold (positive lowers D)", fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="completed factor-rank-displacement output directory")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    contract = validate_preregistration(run_dir)
    e1_result = load_json(run_dir / "e1_results.json")
    if e1_result.get("status") != "complete" or e1_result.get("population_count") != 576:
        raise ValueError("report generation requires the completed 576-identity E1 scan")
    e1_sensitivity_path = run_dir / "e1_precision_sensitivity.json"
    e1_sensitivity = load_json(e1_sensitivity_path) if e1_sensitivity_path.is_file() else None
    front_path = run_dir / "e2_front_selection.json"
    front_selection = load_json(front_path)
    front_comparison_path = run_dir / "front_selection_precision_comparison.json"
    front_comparison = load_json(front_comparison_path) if front_comparison_path.is_file() else None
    if front_comparison is not None and not front_comparison.get("all_24_choices_unchanged"):
        raise ValueError("corrected front choices differ; inspect the saved comparison before reporting")
    selected_counts = front_selection["selected_version_counts"]
    preregistered_by_id = {parent["factor_id"]: parent for parent in contract["e2"]["parents"]}
    n_a_ids = sorted(parent["factor_id"] for parent in front_selection["parents"]
                     if parent["selected_version"] == "original"
                     and not any(variant["applicable"]
                                 for variant in preregistered_by_id[parent["factor_id"]]["smooth_variants"]))
    selected_original_with_smoothing = sum(
        parent["selected_version"] == "original"
        and any(variant["applicable"]
                for variant in preregistered_by_id[parent["factor_id"]]["smooth_variants"])
        for parent in front_selection["parents"])
    e2_audit_path = run_dir / "e2_completion_audit.json"
    if e2_audit_path.is_file():
        e2_audit = load_json(e2_audit_path)
    else:
        progress = [json.loads(line) for line in (run_dir / "e2_progress.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if len(progress) != 72 or any(row.get("status") != "complete" for row in progress):
            raise ValueError("report generation requires all 72 E2 parent-segment outputs")
        summary_prefix = read_json_top_level_prefix_field(run_dir / "e2_results.json", "label_source")
        e2_audit = {"status": "E2_complete_from_outputs", "parent_count": 24,
                    "eligible_smoothing_parent_count": 21,
                    "label_coverage_A_only": summary_prefix["label_coverage"],
                    "timing_seconds": None, "storage": None}
    e3_audit_path = run_dir / "e3_completion_audit.json"
    if e3_audit_path.is_file():
        e3_audit = load_json(e3_audit_path)
    else:
        e3_effect_path = run_dir / "e3_effects.csv"
        if not e3_effect_path.is_file() or not (run_dir / "e3_results.json").is_file():
            raise FileNotFoundError("report generation requires completed E3 summary and paired-effects CSV")
        e3_audit = {"status": "E3_complete_from_outputs", "all_parent_denominator": 24,
                    "selected_smooth_parent_count": sum(value for key, value in selected_counts.items()
                                                         if key.startswith("smooth_")),
                    "selected_smooth_window_counts": {key: value for key, value in selected_counts.items()
                                                       if key.startswith("smooth_")},
                    "selected_original_with_smoothing_available_count": selected_original_with_smoothing,
                    "no_smooth_variant_applicable_parent_count": len(n_a_ids), "n_a_parent_ids": n_a_ids}

    e1_records, e1_summary = e1_rows(run_dir)
    e2_effects, scatter = e2_parent_rows(run_dir)
    e2_grid = summarize_e2_grid(e2_effects)
    e3_effects, e3_grid = e3_grid_rows(run_dir)
    if len({row["factor_id"] for row in e3_effects}) != e3_audit["selected_smooth_parent_count"]:
        raise ValueError("E3 selected-smooth parent count differs from its paired-effects CSV")
    selected_smooth_ids = {item["factor_id"] for item in front_selection["parents"]
                           if str(item["selected_version"]).startswith("smooth_")}
    if {row["factor_id"] for row in e3_effects} != selected_smooth_ids:
        raise ValueError("E3 paired-effects identities differ from front-selected executable smooth parents")
    output = run_dir

    e2_effect_fields = list(e2_effects[0])
    write_csv(output / "e2_parent_effects_full_A.csv", e2_effect_fields, e2_effects)
    e2_grid_fields = list(e2_grid[0])
    write_csv(output / "e2_effect_distribution_full_A.csv", e2_grid_fields, e2_grid)
    e3_grid_fields = list(e3_grid[0])
    write_csv(output / "e3_hold_effect_distribution.csv", e3_grid_fields, e3_grid)

    worst_ic = sorted((row for row in e2_effects
                       if row["directional_IC_change_smooth_minus_original"] is not None),
                      key=lambda row: row["directional_IC_change_smooth_minus_original"])[:5]
    write_csv(output / "e2_most_negative_ic_examples.csv", e2_effect_fields, worst_ic)

    e1_d_fig = output / "e1_candidate_distributions.png"
    e1_scatter_fig = output / "e1_native_D_IC_scatter.png"
    e2_fig = output / "e2_full_A_effect_grid.png"
    e2_scatter_fig = output / "e2_full_A_pair_support_scatter.png"
    e3_fig = output / "e3_hold_effect_grid.png"
    e3_front_hold_fig = output / "e3_front_hold_D_comparison.png"
    plot_e1(e1_records, e1_d_fig)
    plot_e1_d_ic_scatter(run_dir, e1_scatter_fig)
    plot_e2_grid(e2_grid, e2_fig)
    plot_pair_support_scatter(scatter, e2_scatter_fig)
    plot_e3_grid(e3_grid, e3_fig)
    plot_e3_front_hold_D(run_dir, e3_front_hold_fig)

    class_counts = contract["formula_classification"]["class_counts"]
    label_coverage = e2_audit["label_coverage_A_only"]
    e0 = load_json(run_dir / "e0_results.json")
    e1_timing = e1_sensitivity["timing_seconds"] if e1_sensitivity else e1_result["timing_seconds"]
    e1_storage = e1_sensitivity["storage"] if e1_sensitivity else e1_result["storage"]
    choice_counts = (front_comparison["corrected_selected_version_counts"]
                     if front_comparison else selected_counts)
    unchanged_vs_default = (front_comparison["all_24_choices_unchanged"]
                            if front_comparison else None)
    max_score_change = (front_comparison["max_absolute_version_score_change"]
                        if front_comparison else None)
    sensitivity_table = ([[str(delta), str(data["factors_with_changed_D_mean"]),
                           fmt(data["default_parser_population_mean_D"]), fmt(data["round_trip_population_mean_D"]),
                           fmt(data["mean_absolute_factor_D_change"]),
                           str(data["max_absolute_factor_D_change"]["factor_id"]),
                           fmt(data["max_absolute_factor_D_change"]["absolute_change"])]
                          for delta, data in e1_sensitivity["round_trip_vs_default_D_mean_sensitivity"].items()]
                         if e1_sensitivity else [])
    if front_comparison:
        precheck_exposure = {
            "first_front_saved_before_first_default_precision_full_A_E1": True,
            "default_precision_full_A_E1_precheck_occurred_before_corrected_front_reselection": True,
            "corrected_front_selection_call_read_front_prefix_only": True,
            "parser_audit_read_full_A_for_factors_00001_00002_including_post_front": True,
            "whole_workflow_post_front_A_exposure": True,
            "interpretation": "A-internal historical temporal-stability diagnostic, not an untouched independent holdout",
        }
        exposure_text = ("首次front选择先于默认精度全A E1；发现精度缺陷后默认精度全A预检已完成，随后同一24身份/规则的round-trip front-only重算保持原选择。精度审计也读取过factor00001/00002的完整A值。整个流程已有front后A暴露，结果是A内时间稳定性诊断，不是独立留出。")
        sensitivity_section = ("### 默认解析预检与round-trip D敏感性\n\n"
                              + markdown_table(['Δ(h)','改变D均值的因子数','默认解析总体均值D','round-trip总体均值D',
                                                '因子绝对差均值','最大差因子','最大绝对差'], sensitivity_table))
    else:
        precheck_exposure = {
            "first_front_saved_before_this_run_full_A_E1": True,
            "default_precision_full_A_E1_precheck_in_this_run": False,
            "corrected_front_selection_call_read_front_prefix_only": True,
            "whole_workflow_post_front_A_exposure": True,
            "interpretation": "A-internal historical temporal-stability diagnostic, not an untouched independent holdout",
        }
        exposure_text = ("本次front选择在本次full-A E1之前保存；该复跑目录没有默认精度E1敏感性预检。E1/E2之后续读A的完整后半历史，因此这不是独立留出验证，结果定位为A内时间稳定性诊断。")
        sensitivity_section = ("### 默认解析预检与round-trip D敏感性\n\n"
                              "此复跑目录未运行默认精度全A预检，没有生成旧/新parser敏感性对照。")
    lite = {
        "status": "complete_readable_summary",
        "preregistration_sha256": contract["preregistration_sha256"],
        "precision_parser": contract["factor_value_csv_numeric_parser"],
        "front_selection": {"choice_counts": choice_counts,
                            "all_24_choices_unchanged_vs_default_precheck": unchanged_vs_default,
                            "max_absolute_version_score_change": max_score_change,
                            "no_smooth_variant_applicable_parent_ids": n_a_ids},
        "e1": {"status": e1_result["status"], "population_count": 576,
               "unique_H_counts": e1_result["aggregate"][
                   "unique_factor_count_with_finite_directional_ic_by_horizon"],
               "native_A_coverage": e1_summary["native_A_coverage"],
               "native_D_distributions_by_delta": e1_summary["native_D_by_delta"],
               "archived_directional_IC_distributions_by_H": e1_summary["directional_IC_by_horizon"],
               "precision_sensitivity": (e1_sensitivity["round_trip_vs_default_D_mean_sensitivity"]
                                         if e1_sensitivity else None),
               "timing_seconds": e1_timing, "storage": e1_storage},
        "e0": {"full_pool_estimate": e0["full_pool_estimate"], "peak_rss_bytes": e0["peak_rss_bytes"]},
        "formula_structural_classes": class_counts,
        "e2": {"status": e2_audit["status"], "parent_count": 24,
               "eligible_smoothing_parent_count": 21,
               "segments": ["front", "hold", "full_A"],
               "common_support":"original and all statically applicable smooth variants share all-version common factor/label support",
               "full_A_effect_distribution_grid": e2_grid,
               "most_negative_directional_IC_examples": worst_ic,
               "timing_seconds": e2_audit.get("timing_seconds"), "storage": e2_audit.get("storage")},
        "e3": {"status": e3_audit["status"], "all_parent_denominator": 24,
               "selected_smooth_parent_count": 21,
               "selected_smooth_window_counts": e3_audit["selected_smooth_window_counts"],
               "selected_original_with_smoothing_available_count": selected_original_with_smoothing,
               "no_smooth_variant_applicable_parent_count": len(n_a_ids),
               "n_a_parent_ids": n_a_ids,
               "hold_effect_distribution_grid": e3_grid,
               "uses_E2_precomputed_all_version_pair_support": True},
        "label_coverage_A_only": label_coverage,
        "exposure_timeline": precheck_exposure,
        "thresholds": {"max_ic_loss_predeclared": False,
                       "negative_IC_changes_are_descriptive": True,
                       "per_parent_HAC_intervals_upper_below_zero_are_counted": True},
        "report_files": {"e2_parent_effects_full_A_csv": str(output / "e2_parent_effects_full_A.csv"),
                         "e2_full_A_grid_csv": str(output / "e2_effect_distribution_full_A.csv"),
                         "e2_worst_ic_examples_csv": str(output / "e2_most_negative_ic_examples.csv"),
                         "e3_hold_grid_csv": str(output / "e3_hold_effect_distribution.csv"),
                         "e3_parent_effects_csv": str(output / "e3_effects.csv"),
                         "e1_chart": str(e1_d_fig), "e1_D_IC_scatter": str(e1_scatter_fig),
                         "e2_grid_chart": str(e2_fig), "e2_pair_support_scatter": str(e2_scatter_fig),
                         "e3_hold_chart": str(e3_fig), "e3_front_hold_D_comparison": str(e3_front_hold_fig)},
    }
    lite_path = output / "e2_e3_lite_summary.json"
    write_json(lite_path, lite)

    e2_d_rows = [[str(row["smooth_window_hours"]), str(row["horizon_hours"]), str(row["delta_hours"]),
                  str(row["D_effect_n"]), fmt(row["D_decrease_mean"]), fmt(row["D_decrease_p10"]),
                  fmt(row["D_decrease_median"]), fmt(row["D_decrease_p90"]),
                  f"{row['D_decrease_positive_parent_count']}/21",
                  f"{row['D_decrease_negative_parent_count']}/21",
                  f"{row['D_decrease_ci_lower_above_zero_count']}/{row['D_ci_n']}",
                  f"{row['D_decrease_ci_upper_below_zero_count']}/{row['D_ci_n']}"] for row in e2_grid]
    e2_ic_rows = [[str(row["smooth_window_hours"]), str(row["horizon_hours"]), str(row["delta_hours"]),
                   str(row["directional_IC_effect_n"]), fmt(row["directional_IC_change_mean"]),
                   fmt(row["directional_IC_change_p10"]), fmt(row["directional_IC_change_median"]),
                   fmt(row["directional_IC_change_p90"]),
                   f"{row['directional_IC_change_negative_parent_count']}/21",
                   f"{row['directional_IC_ci_upper_below_zero_count']}/{row['directional_IC_ci_n']}"] for row in e2_grid]
    e3_d_rows = [[str(row["horizon_hours"]), str(row["delta_hours"]), str(row["D_effect_n"]),
                  fmt(row["D_decrease_mean"]), fmt(row["D_decrease_p10"]), fmt(row["D_decrease_median"]),
                  fmt(row["D_decrease_p90"]), f"{row['D_decrease_positive_parent_count']}/21",
                  f"{row['D_decrease_negative_parent_count']}/21",
                  f"{row['D_decrease_ci_lower_above_zero_count']}/{row['D_ci_n']}",
                  f"{row['D_decrease_ci_upper_below_zero_count']}/{row['D_ci_n']}"] for row in e3_grid]
    e3_ic_rows = [[str(row["horizon_hours"]), str(row["delta_hours"]), str(row["directional_IC_effect_n"]),
                   fmt(row["directional_IC_change_mean"]), fmt(row["directional_IC_change_p10"]),
                   fmt(row["directional_IC_change_median"]), fmt(row["directional_IC_change_p90"]),
                   f"{row['directional_IC_change_negative_parent_count']}/21",
                   f"{row['directional_IC_ci_upper_below_zero_count']}/{row['directional_IC_ci_n']}"] for row in e3_grid]
    e3_spread_rows = [[str(row["horizon_hours"]), str(row["delta_hours"]),
                       fmt(row["directional_spread_change_mean"]),
                       fmt(row["directional_spread_change_p10"]),
                       fmt(row["directional_spread_change_median"]),
                       fmt(row["directional_spread_change_p90"]),
                       f"{row['directional_spread_change_negative_parent_count']}/21",
                       f"{row['directional_spread_ci_upper_below_zero_count']}/{row['directional_spread_ci_n']}"] for row in e3_grid]

    e1_d_table = [[str(delta), str(e1_summary["native_D_by_delta"][str(delta)]["n"]),
                   fmt(e1_summary["native_D_by_delta"][str(delta)]["mean"]),
                   fmt(e1_summary["native_D_by_delta"][str(delta)]["p10"]),
                   fmt(e1_summary["native_D_by_delta"][str(delta)]["median"]),
                   fmt(e1_summary["native_D_by_delta"][str(delta)]["p90"])] for delta in DELTAS]
    e1_ic_table = [[str(h), str(e1_summary["directional_IC_by_horizon"][str(h)]["n"]),
                    fmt(e1_summary["directional_IC_by_horizon"][str(h)]["mean"]),
                    fmt(e1_summary["directional_IC_by_horizon"][str(h)]["p10"]),
                    fmt(e1_summary["directional_IC_by_horizon"][str(h)]["median"]),
                    fmt(e1_summary["directional_IC_by_horizon"][str(h)]["p90"])] for h in HORIZONS]
    class_rows = [[name, str(class_counts[name])] for name in CLASS_ORDER]
    label_rows = [[f"H={h}h", str(label_coverage[str(h)]["eligible_observations"]),
                   str(label_coverage[str(h)]["finite_eligible_observations"]),
                   str(label_coverage[str(h)]["purged_eligible_observations"]),
                   str(label_coverage[str(h)]["warmup_eligible_input_observations"])] for h in HORIZONS]
    top_rows = [[str(row["smooth_window_hours"]), f"H={row['horizon_hours']}h", f"Δ={row['delta_hours']}h",
                 row["factor_id"], row["formula_class"],
                 fmt(row["directional_IC_change_smooth_minus_original"]),
                 ci_text(row["directional_IC_change_ci_low"], row["directional_IC_change_ci_high"]),
                 fmt(row["D_decrease_original_minus_smooth"])] for row in worst_ic]
    worst_path = output / "e2_most_negative_ic_examples.csv"
    report = f"""# FM-v6信号持续性与换手倾向：A历史诊断报告

本报告来自round-trip浮点精度运行，预注册SHA-256为`{contract['preregistration_sha256']}`。Factor-value CSV使用pandas `float_precision='round_trip'`。E1覆盖576个A候选；E2比较4/12/24小时平滑窗、H=1/4/24和Δ=1/4/24的固定网格；E3检查前半选出的平滑方案在hold期的A内时间迁移。历史A区间为2022-08-01至2024-08-01。

## 数据暴露与解释范围

{exposure_text}

B/C评估结果未用于本研究。E2在front、hold、full_A各段，对original和所有静态可执行平滑版本使用同一个all-version common support。E3从已计算的E2 pair effects中提取front锁定smooth相对original的paired effects；hold没有另行只重算两个版本，也没有改成两版本专属样本。

## E1：原版因子分布与精度敏感性

E1包含576个因子身份。原版D为各因子的A原序列rank-displacement均值；H=1/4/24 directional rank IC取自对应A归档报告。下列分布是候选级描述，不把576个因子假设为相互独立。

### Native D按Δ分布

{markdown_table(['Δ(h)','n','mean','p10','median','p90'], e1_d_table)}

### 归档directional IC按H分布

{markdown_table(['H(h)','n','mean','p10','median','p90'], e1_ic_table)}

{sensitivity_section}

E1为576/576，H1/H4/H24各{e1_result['population_count']}个唯一身份记录；每个因子的A网格{e1_summary['native_A_coverage']['grid_rows_per_factor']:,}行、资格行{e1_summary['native_A_coverage']['eligible_rows_per_factor']:,}行。E1总耗时约{e1_timing['total_elapsed']:.1f}秒，峰值RSS {e1_storage['peak_rss_bytes']}字节。

![E1候选D与归档IC分布](e1_candidate_distributions.png)

![E1全576池H-Δ的native D与IC散点，颜色表示D有效覆盖](e1_native_D_IC_scatter.png)

## E2：固定平滑窗的全A配对效应

定义：`D下降 = D_original − D_smooth`，正数表示平滑后D较低；`directional IC变化 = IC_smooth − IC_original`，负数表示预测对齐下降。每个H/Δ/窗口单元汇总21个可执行父因子的父级配对效应均值分布，p10/median/p90是在父级效应均值之间计算。每个父因子使用HAC滞后23的小时级区间；“CI全负数”统计父级95%区间上界小于0的数量，不是跨父因子的独立样本推断。

没有预先登记的`max_ic_loss`阈值。下表与最负实例仅作描述，不代表超限、失败或通过。full_A覆盖整个A区间，包含front选择所用时段；E2是共同支持诊断。

### D下降：27个固定网格单元

{markdown_table(['W(h)','H(h)','Δ(h)','n','D下降均值','p10','中位数','p90','D下降点估计数','D上升点估计数','D CI全正数','D CI全负数'], e2_d_rows)}

### Directional IC变化：27个固定网格单元

{markdown_table(['W(h)','H(h)','Δ(h)','n','IC变化均值','p10','中位数','p90','IC下降点估计数','IC CI全负数'], e2_ic_rows)}

### 固定网格中最负的五个IC点估计

以下只是567个父因子×窗口×H×Δ效应中的五个最负directional-IC变化点估计，CI随行给出，且无预定阈值。

{markdown_table(['W(h)','H','Δ','factor_id','结构类','IC变化','父级HAC 95% CI','D下降'], top_rows)}

![E2全A效应固定网格热图](e2_full_A_effect_grid.png)

### E2 original/variant的IC-D坐标

散点图的original和smooth坐标均来自每个variant的`reference_on_pair_support`与`trial_on_pair_support`。脚本核验坐标差分别与配对D下降和IC变化一致，并核验不同平滑窗共享同一original common-support坐标。

![E2全A配对共同支持IC-D坐标](e2_full_A_pair_support_scatter.png)

## E3：前半选择方案的hold迁移

最终front选择中21个适用父因子均选择smooth_24h；3个父因子（{', '.join(e3_audit['n_a_parent_ids'])}）因固定窗口超过预登记节点/lookback上限而无smooth可执行方案，作为N/A单列，不计作original获胜。E3的配对效应分母为21，总父因子分母为24。

### D下降：H×Δ九格

{markdown_table(['H(h)','Δ(h)','n','D下降均值','p10','中位数','p90','D下降点估计数','D上升点估计数','D CI全正数','D CI全负数'], e3_d_rows)}

### Directional IC变化：H×Δ九格

{markdown_table(['H(h)','Δ(h)','n','IC变化均值','p10','中位数','p90','IC下降点估计数','IC CI全负数'], e3_ic_rows)}

### Directional spread变化：H×Δ九格

{markdown_table(['H(h)','Δ(h)','spread变化均值','p10','中位数','p90','spread下降点估计数','spread CI全负数'], e3_spread_rows)}

![E3 hold迁移效应网格](e3_hold_effect_grid.png)

spread变化使用每个版本全版本共同spread有效期计算，支持期数可与D/IC共同有效期不同。

![E3各H-Δ单元front与hold的配对D下降效应对照](e3_front_hold_D_comparison.png)

三个H的平均directional-IC变化及H1/H4/H24的spread变化见完整网格。D与预测指标方向并不等价：降低D不保证IC或spread上升。报告保留所有H/Δ单元的点估计分布和父级区间支持计数，不以未登记阈值判定成败。front-vs-hold图逐父因子对照两个时期已计算的D下降配对效应。

## 标签与结构分类

标签覆盖只统计A时间段；168小时warm-up eligible输入单列。

{markdown_table(['标签H','A eligible','A finite非purged','A purged','warm-up eligible输入'], label_rows)}

六类仅按公式引用字段的域及显式结构算子归类：多域映射mixed；单域时序离散/区间算子映射volatility/range；否则按price、volume/liquidity、funding/basis或positioning/ratio单域归类。分类计数为：

{markdown_table(['结构类','候选数'], class_rows)}

该分类未按公式谱系、语义等价或相关性独立化；相关候选可共享结构类，576个E1身份不是576个独立样本假设。

## 可复核文件

- E2完整父级全网格：[`e2_parent_effects_full_A.csv`](e2_parent_effects_full_A.csv)（567行）；27格汇总：[`e2_effect_distribution_full_A.csv`](e2_effect_distribution_full_A.csv)。
- 最负IC固定网格实例：[`e2_most_negative_ic_examples.csv`](e2_most_negative_ic_examples.csv)。
- E3 21父×9格明细：[`e3_effects.csv`](e3_effects.csv)（189行）；9格汇总：[`e3_hold_effect_distribution.csv`](e3_hold_effect_distribution.csv)。
- 轻量总览：[`e2_e3_lite_summary.json`](e2_e3_lite_summary.json)。E2大文件`e2_factors.jsonl`和E2/E3完整summary仍保留在目录内，无需加载即可查看上述人读表。
- 图：`e1_candidate_distributions.png`、`e1_native_D_IC_scatter.png`、`e2_full_A_effect_grid.png`、`e2_full_A_pair_support_scatter.png`、`e3_hold_effect_grid.png`、`e3_front_hold_D_comparison.png`。

如需从一个新完成的研究目录重建这份报告，运行：

```sh
.venv/bin/python scripts/factor_rank_displacement_report.py --run-dir <completed-output-directory>
```
"""
    (output / "e2_e3_readable_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "report_complete", "run_dir": str(run_dir),
                      "lite_summary": str(lite_path),
                      "report": str(output / "e2_e3_readable_report.md"),
                      "e2_grid_rows": len(e2_grid), "e2_parent_rows": len(e2_effects),
                      "e3_grid_rows": len(e3_grid), "e3_parent_rows": len(e3_effects),
                      "figures": [str(e1_d_fig), str(e1_scatter_fig), str(e2_fig),
                                  str(e2_scatter_fig), str(e3_fig), str(e3_front_hold_fig)]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
