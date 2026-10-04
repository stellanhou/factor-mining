#!/usr/bin/env python3
"""Audit shared-input failure scope and E4's 30-symbol A inputs.

Run from the repository root with .venv/bin/python. The panel reader filters
to the fixed E4 A interval and declared warmup before retaining row values.
It does not evaluate candidate formulas, D, IC, or B/C statistics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from crypto_quant.data_access.market_data import MarketDataStore
from crypto_quant.features.factor_inputs import (
    DEFERRED_FIELDS,
    INPUT_COLUMNS,
    FactorInputPanel,
    load_factor_inputs,
)
from crypto_quant.features.factor_expressions import compile_expression


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/strategy_research"
NEW = BASE / "selection30_20261003"
OLD = BASE / "ablation_20261003/multifactor-ablation-20261003"
E4 = ROOT / "experiments/factor_rank_displacement/20261003/e4"
OUT = E4 / "precheck.json"
SAMPLE_OUT = E4 / "e4_sample_manifest.json"
FORMAL_OUT = E4 / "formal_e4_precheck.json"
PREREG = ROOT / "experiments/factor_rank_displacement/20261003/preregistration.json"
PRIMARY_DB = ROOT / "market_data/crypto_quant.sqlite"
RAW_DB = NEW / "data/market_data.sqlite"
START = pd.Timestamp("2022-08-01T00:00:00Z")
END = pd.Timestamp("2024-08-01T00:00:00Z")
WARMUP_HOURS = 168
WARMUP_START = START - pd.Timedelta(hours=WARMUP_HOURS)
SYMBOLS = [
    "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "DOTUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT",
]
SYMBOLS_30: list[str] = []
HORIZONS = (1, 4, 24)
DELTAS = (1, 4, 24)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_window(path: Path, *, value_columns: list[str], symbols: list[str] | None = None,
                chunksize: int = 250_000, max_rows: int | None = None) -> pd.DataFrame:
    selected_symbols = SYMBOLS if symbols is None else symbols
    columns = ["timestamp", "symbol", *value_columns]
    dtypes = {"eligible": "string"} if "eligible" in value_columns else None
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, nrows=max_rows,
                             float_precision="round_trip", dtype=dtypes):
        times = pd.to_datetime(chunk["timestamp"], utc=True, format="mixed")
        keep = times.ge(WARMUP_START) & times.lt(END) & chunk["symbol"].isin(selected_symbols)
        if keep.any():
            selected = chunk.loc[keep, columns].copy()
            selected["timestamp"] = times.loc[keep]
            parts.append(selected)
    frame = pd.concat(parts, ignore_index=True).set_index(["timestamp", "symbol"]).sort_index()
    if not frame.index.is_unique:
        raise ValueError(f"duplicate rows in {path}")
    return frame


def read_a_window(path: Path, *, value_columns: list[str], symbols: list[str],
                  grid_symbols: int | None = None) -> pd.DataFrame:
    expected_rows = len(pd.date_range(WARMUP_START, END, freq="h", inclusive="left"))
    total_symbols = len(symbols) if grid_symbols is None else grid_symbols
    first_row = pd.read_csv(path, usecols=["timestamp"], nrows=1)
    if first_row.empty:
        raise ValueError(f"empty panel: {path}")
    first_time = pd.to_datetime(first_row["timestamp"].iloc[0], utc=True, format="mixed")
    hours_before_warmup = (WARMUP_START - first_time) / pd.Timedelta(hours=1)
    if hours_before_warmup < 0 or hours_before_warmup != int(hours_before_warmup):
        raise ValueError(f"panel start is not on or before the frozen warmup grid: {path}")
    max_rows = (expected_rows + int(hours_before_warmup)) * total_symbols
    frame = read_window(path, value_columns=value_columns, symbols=symbols, max_rows=max_rows)
    expected_index = pd.MultiIndex.from_product(
        [pd.date_range(WARMUP_START, END, freq="h", inclusive="left"), symbols],
        names=["timestamp", "symbol"],
    )
    if not frame.index.equals(expected_index):
        raise ValueError(f"panel does not cover exactly the A plus 168-hour warmup grid: {path}")
    return frame


def label_frame(opens: pd.Series, horizon: int, *, symbols: list[str] | None = None) -> pd.DataFrame:
    selected_symbols = SYMBOLS if symbols is None else symbols
    prices = opens.unstack("symbol").reindex(columns=selected_symbols)
    entry = prices.shift(-1)
    exit_price = prices.shift(-(horizon + 1))
    labels = exit_price.div(entry).sub(1.0).where((entry > 0) & (exit_price > 0))
    times = pd.DatetimeIndex(labels.index)
    inside = times + pd.Timedelta(hours=horizon + 1) < END
    labels.loc[~inside, :] = np.nan
    return labels.stack(future_stack=True).rename("forward_return").to_frame()


def mask_frame(path: Path, *, symbols: list[str] | None = None,
               grid_symbols: int | None = None) -> pd.Series:
    if symbols is None:
        frame = read_window(path, value_columns=["eligible"])
    else:
        frame = read_a_window(path, value_columns=["eligible"], symbols=symbols,
                              grid_symbols=grid_symbols)
    raw = frame["eligible"]
    if raw.isna().any():
        raise ValueError(f"missing eligibility values in {path}")
    invalid = sorted(set(raw.unique()) - {"True", "False"})
    if invalid:
        raise ValueError(f"invalid eligibility values in {path}: {invalid}")
    return raw.map({"True": True, "False": False}).astype(bool).rename("eligible")


def metric_counts(mask: pd.Series, labels: pd.DataFrame, *, label: str) -> dict:
    in_a = mask.index.get_level_values("timestamp") >= START
    a_mask = mask.loc[in_a]
    valid_label = labels["forward_return"].reindex(a_mask.index).notna()
    eligible = a_mask & valid_label
    return {
        "eligible_A_cells": int(a_mask.sum()),
        "eligible_labeled_A_cells": int(eligible.sum()),
        "label_cells": int(valid_label.sum()),
        "coverage": float(eligible.sum() / valid_label.sum()) if valid_label.sum() else None,
        "source": label,
    }


def freeze_e4_sample(prereg: dict, *, prepared_fields: set[str]) -> dict:
    parents = prereg["e2"]["parents"]
    frozen = []
    skipped = []
    for parent in parents:
        identity = parent["identity"]
        compiled = compile_expression(identity["expanded_expression"])
        fields = sorted(compiled.fields)
        reasons = []
        if set(fields) != set(parent["fields"]):
            reasons.append("compiled_fields_differ_from_preregistered_fields")
        if not set(fields) <= set(INPUT_COLUMNS):
            reasons.append("field_not_in_factor_input_contract")
        if set(fields) & set(DEFERRED_FIELDS):
            reasons.append("uses_deferred_field")
        if not set(fields) <= prepared_fields:
            reasons.append("field_not_in_rebuilt_panel_header")
        if compiled.lookback_hours > WARMUP_HOURS:
            reasons.append("lookback_exceeds_168_hour_warmup")
        if compiled.lookback_hours != parent["original_lookback_hours"]:
            reasons.append("compiled_lookback_differs_from_preregistration")
        if reasons:
            skipped.append({"factor_id": parent["factor_id"], "reasons": reasons})
            continue
        source_contract_path = Path(parent["a_universe_path"]).parent / "contract.json"
        source_contract = read_json(source_contract_path)
        if source_contract["min_symbols"] != 20 or source_contract["sample_hours"] != 1:
            raise ValueError(f"{parent['factor_id']} source contract differs from min_symbols=20/sample_hours=1")
        frozen.append({
            "selection_position_in_e2_order": int(parent["parent_order"]),
            "factor_id": parent["factor_id"],
            "source_run_id": parent["source_run_id"],
            "candidate_id": parent["candidate_id"],
            "formula_class": parent["formula_class"],
            "expression": identity["expanded_expression"],
            "direction": int(identity["direction"]),
            "semantics_version": identity["semantics_version"],
            "fields": fields,
            "lookback_hours": compiled.lookback_hours,
            "source_label": source_contract["label"],
            "a_start": source_contract["a_start"],
            "a_end_exclusive": source_contract["b_start"],
            "min_symbols": source_contract["min_symbols"],
            "sample_hours": source_contract["sample_hours"],
            "a_universe_path": parent["a_universe_path"],
            "a_archive": parent["archive"],
        })
        if len(frozen) == 6:
            break
    if len(frozen) != 6:
        raise ValueError(f"only {len(frozen)} E2-ordered factors satisfy the frozen E4 field/warmup contract")
    first = frozen[0]
    if any(item["a_start"] != first["a_start"] or item["a_end_exclusive"] != first["a_end_exclusive"]
           or item["min_symbols"] != first["min_symbols"] or item["sample_hours"] != first["sample_hours"]
           for item in frozen):
        raise ValueError("frozen E4 parents do not share the same A contract")
    return {
        "schema_version": 1,
        "status": "frozen_static_identity_and_source_field_audit",
        "preregistration_path": str(PREREG.relative_to(ROOT)),
        "preregistration_sha256": sha256(PREREG),
        "selection_rule": "first six E2 parents in their frozen identity order whose exact expression fields are available in the rebuilt panel, non-deferred, and whose lookback is at most the declared 168-hour warmup; no historical factor performance was read for selection",
        "sample_count": len(frozen),
        "parents": frozen,
        "skipped_before_stop": skipped,
        "evaluation_contract": {
            "universe_identity_count": 30,
            "min_symbols": 20,
            "sample_hours": 1,
            "a_start": first["a_start"],
            "a_end_exclusive": first["a_end_exclusive"],
            "warmup_hours": WARMUP_HOURS,
            "rank_deltas_hours": list(DELTAS),
            "prediction_horizons_hours": list(HORIZONS),
            "label_definition": "perp_next_open_H: perp_open[t+1] to perp_open[t+H+1], label_end < A end",
        },
        "source_comparison": {
            "old": str(PRIMARY_DB.relative_to(ROOT)),
            "old_factor_values": "frozen A value_set referenced by each parent archive record",
            "rebuilt": str(RAW_DB.relative_to(ROOT)),
            "rebuilt_input_panel": str((NEW / "source_inputs/development/inputs/panel.values.csv").relative_to(ROOT)),
            "rebuilt_provenance": str((NEW / "source_inputs/development/inputs/data-provenance.json").relative_to(ROOT)),
        },
        "full_factor_evaluation_gate": {
            "required_artifact": "experiments/factor_rank_displacement/20261003/e2_front_selection.json",
            "available_at_static_freeze": (ROOT / "experiments/factor_rank_displacement/20261003/e2_front_selection.json").is_file(),
            "d_ic_run_started": False,
        },
    }


def strict_boolean_mask(frame: pd.DataFrame, *, label: str) -> pd.Series:
    raw = frame["eligible"]
    if raw.isna().any():
        raise ValueError(f"missing eligibility values in {label}")
    invalid = sorted(set(raw.unique()) - {"True", "False"})
    if invalid:
        raise ValueError(f"invalid eligibility values in {label}: {invalid}")
    return raw.map({"True": True, "False": False}).astype(bool).rename("eligible")


def a_only(series: pd.Series) -> pd.Series:
    timestamps = series.index.get_level_values("timestamp")
    return series.loc[(timestamps >= START) & (timestamps < END)]


def factor_input_support(values: pd.DataFrame, mask: pd.Series, fields: list[str]) -> dict:
    member = a_only(mask)
    panel = values.reindex(member.index)
    finite = np.isfinite(panel[fields].to_numpy(dtype=float)).all(axis=1)
    active = member.to_numpy(dtype=bool)
    times = member.index.get_level_values("timestamp")
    by_hour = pd.Series(active & finite, index=times).groupby(level=0).sum()
    return {
        "eligible_cells": int(active.sum()),
        "required_fields_complete_cells": int((active & finite).sum()),
        "required_fields_complete_share_of_eligible": float((active & finite).sum() / active.sum()) if active.sum() else None,
        "hours_with_at_least_20_complete_symbols": int((by_hour >= 20).sum()),
        "minimum_complete_symbols_per_hour": int(by_hour.min()) if len(by_hour) else None,
        "maximum_complete_symbols_per_hour": int(by_hour.max()) if len(by_hour) else None,
        "interpretation": "field-input support only; no formula values or factor metrics computed",
    }


def field_pair_comparison(old_values: pd.DataFrame, raw_values: pd.DataFrame,
                          old_mask: pd.Series, raw_mask: pd.Series,
                          fields: list[str]) -> dict:
    common = a_only(old_mask & raw_mask)
    common_index = common.index[common.to_numpy(dtype=bool)]
    result = {}
    for field in fields:
        old = old_values[field].reindex(common_index).to_numpy(dtype=float)
        raw = raw_values[field].reindex(common_index).to_numpy(dtype=float)
        finite_old, finite_raw = np.isfinite(old), np.isfinite(raw)
        paired = finite_old & finite_raw
        delta = np.abs(old[paired] - raw[paired])
        scale = np.maximum(np.abs(old[paired]), np.abs(raw[paired]))
        relative = np.divide(delta, scale, out=np.zeros_like(delta), where=scale > 0)
        equal = old[paired] == raw[paired]
        within = np.isclose(old[paired], raw[paired], rtol=1e-12, atol=1e-15)
        result[field] = {
            "old_native_finite_cells": int((a_only(old_mask) & np.isfinite(old_values[field].reindex(a_only(old_mask).index))).sum()),
            "raw_native_finite_cells": int((a_only(raw_mask) & np.isfinite(raw_values[field].reindex(a_only(raw_mask).index))).sum()),
            "common_eligible_cells": int(common.sum()),
            "both_finite_common_cells": int(paired.sum()),
            "exact_equal_cells": int(equal.sum()),
            "within_tolerance_nonexact_cells": int((~equal & within).sum()),
            "different_beyond_tolerance_cells": int((~within).sum()),
            "max_absolute_difference": float(delta.max(initial=0.0)),
            "max_relative_difference": float(relative.max(initial=0.0)),
        }
    return result


def source_label_summary(old_open: pd.Series, raw_open: pd.Series,
                         old_mask: pd.Series, raw_mask: pd.Series,
                         symbols: list[str]) -> tuple[dict, dict[int, tuple[pd.DataFrame, pd.DataFrame]]]:
    old_prices = old_open.unstack("symbol").reindex(columns=symbols)
    raw_prices = raw_open.unstack("symbol").reindex(columns=symbols)
    times = pd.DatetimeIndex(old_prices.index)
    if not times.equals(pd.DatetimeIndex(raw_prices.index)):
        raise ValueError("old and rebuilt A+warmup label time grids differ")
    old_members = old_mask.unstack("symbol").reindex(index=times, columns=symbols)
    raw_members = raw_mask.unstack("symbol").reindex(index=times, columns=symbols)
    common_members = old_members & raw_members
    old_common_cells = raw_members.to_numpy(dtype=bool) & old_members.to_numpy(dtype=bool)
    summary = {}
    label_matrices = {}
    for horizon in HORIZONS:
        entry_old = old_prices.shift(-1)
        exit_old = old_prices.shift(-(horizon + 1))
        entry_raw = raw_prices.shift(-1)
        exit_raw = raw_prices.shift(-(horizon + 1))
        old_labels = exit_old.div(entry_old).sub(1.0).where((entry_old > 0) & (exit_old > 0))
        raw_labels = exit_raw.div(entry_raw).sub(1.0).where((entry_raw > 0) & (exit_raw > 0))
        inside = (times >= START) & (times < END) & (times + pd.Timedelta(hours=horizon + 1) < END)
        old_labels.loc[~inside, :] = np.nan
        raw_labels.loc[~inside, :] = np.nan
        old_valid = np.isfinite(old_labels.to_numpy(dtype=float))
        raw_valid = np.isfinite(raw_labels.to_numpy(dtype=float))
        old_native = old_members.to_numpy(dtype=bool) & old_valid
        raw_native = raw_members.to_numpy(dtype=bool) & raw_valid
        common_valid = old_common_cells & old_valid & raw_valid & inside[:, None]
        both_values = old_labels.to_numpy(dtype=float)[common_valid]
        raw_both_values = raw_labels.to_numpy(dtype=float)[common_valid]
        value_delta = np.abs(both_values - raw_both_values)
        value_equal = both_values == raw_both_values
        value_within = np.isclose(both_values, raw_both_values, rtol=1e-12, atol=1e-15)
        old_hours = old_native.sum(axis=1)
        raw_hours = raw_native.sum(axis=1)
        common_hours = common_valid.sum(axis=1)
        summary[str(horizon)] = {
            "definition": "perp_next_open_H: open[t+1] to open[t+H+1]",
            "purge": "signal timestamps t in A and label_end=t+H+1h < A_end",
            "old_finite_A_label_cells": int(old_valid[inside].sum()),
            "raw_finite_A_label_cells": int(raw_valid[inside].sum()),
            "old_native_eligible_labeled_cells": int(old_native[inside].sum()),
            "raw_native_eligible_labeled_cells": int(raw_native[inside].sum()),
            "common_source_eligible_and_both_labels_valid_cells": int(common_valid.sum()),
            "common_member_label_validity_mismatch_cells": int(
                (old_common_cells & (old_valid != raw_valid) & inside[:, None]).sum()
            ),
            "hours_with_at_least_20_old_native_labels": int(((old_hours >= 20) & inside).sum()),
            "hours_with_at_least_20_raw_native_labels": int(((raw_hours >= 20) & inside).sum()),
            "hours_with_at_least_20_common_labels": int(((common_hours >= 20) & inside).sum()),
            "common_label_value_exact_differences": int((~value_equal).sum()),
            "common_label_value_differences_beyond_tolerance": int((~value_within).sum()),
            "common_label_value_max_absolute_difference": float(value_delta.max(initial=0.0)),
        }
        old_label_rows = old_labels.stack(future_stack=True).rename("forward_return").to_frame()
        raw_label_rows = raw_labels.stack(future_stack=True).rename("forward_return").to_frame()
        row_times = old_label_rows.index.get_level_values("timestamp")
        purged = (row_times >= START) & (row_times < END) & (
            row_times + pd.Timedelta(hours=horizon + 1) >= END
        )
        for labels in (old_label_rows, raw_label_rows):
            labels["purged"] = purged
            labels["eligible"] = False
        old_label_rows.loc[:, "eligible"] = old_mask.reindex(old_label_rows.index).fillna(False).astype(bool)
        raw_label_rows.loc[:, "eligible"] = raw_mask.reindex(raw_label_rows.index).fillna(False).astype(bool)
        label_matrices[horizon] = (old_label_rows, raw_label_rows)

    d_support = {}
    for delta in DELTAS:
        previous_times = times - pd.Timedelta(hours=delta)
        in_segment = (times >= START) & (times < END) & (previous_times >= START)
        old_prev = old_members.reindex(previous_times).to_numpy(dtype=bool)
        raw_prev = raw_members.reindex(previous_times).to_numpy(dtype=bool)
        old_endpoint = old_members.to_numpy(dtype=bool) & old_prev
        raw_endpoint = raw_members.to_numpy(dtype=bool) & raw_prev
        common_endpoint = old_endpoint & raw_endpoint
        old_counts = old_endpoint.sum(axis=1)
        raw_counts = raw_endpoint.sum(axis=1)
        common_counts = common_endpoint.sum(axis=1)
        d_support[str(delta)] = {
            "expected_in_segment_hours_after_boundary": int(in_segment.sum()),
            "old_native_endpoint_hours_with_at_least_20_symbols": int(((old_counts >= 20) & in_segment).sum()),
            "raw_native_endpoint_hours_with_at_least_20_symbols": int(((raw_counts >= 20) & in_segment).sum()),
            "common_eligible_endpoint_hours_with_at_least_20_symbols": int(((common_counts >= 20) & in_segment).sum()),
        }
    summary["eligibility_only_d_endpoint_support"] = d_support
    return summary, label_matrices


def formal_e4_precheck() -> dict:
    prereg = read_json(PREREG)
    rebuilt_panel_path = NEW / "source_inputs/development/inputs/panel.values.csv"
    rebuilt_universe_path = NEW / "source_inputs/development/inputs/universe.csv"
    rebuilt_provenance_path = NEW / "source_inputs/development/inputs/data-provenance.json"
    completion_path = NEW / "source_inputs/preparation_complete.json"
    lifecycle_path = NEW / "source_inputs/lifecycle.json"
    verification_path = NEW / "data/data_verification.json"
    manifest_path = NEW / "data/manifest.json"
    data_usage_path = ROOT / "experiments/factor_mining/long_history_20260919_setup/data-usage.json"
    interpolation_log_path = ROOT / "docs/data/小时行情插值处理记录.md"

    prepared_header = set(pd.read_csv(rebuilt_panel_path, nrows=0).columns)
    prepared_fields = prepared_header - {"timestamp", "symbol"}
    sample_manifest = freeze_e4_sample(prereg, prepared_fields=prepared_fields)
    SAMPLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_OUT.write_text(json.dumps(sample_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
    parents = sample_manifest["parents"]
    symbols = sorted(set(pd.read_csv(parents[0]["a_universe_path"], usecols=["symbol"])["symbol"]))
    if len(symbols) != 30:
        raise ValueError(f"old A universe has {len(symbols)} identities instead of 30")

    old_universe_frame = read_a_window(Path(parents[0]["a_universe_path"]),
                                       value_columns=["eligible"], symbols=symbols)
    raw_universe_frame = read_a_window(rebuilt_universe_path, value_columns=["eligible"], symbols=symbols,
                                       grid_symbols=30)
    old_mask = strict_boolean_mask(old_universe_frame, label="frozen old A-universe")
    raw_mask = strict_boolean_mask(raw_universe_frame, label="rebuilt raw30 source universe")
    if not old_mask.index.equals(raw_mask.index):
        raise ValueError("old and raw30 A plus warmup universe keys differ")
    expected_hours = pd.date_range(WARMUP_START, END, freq="h", inclusive="left", tz="UTC")
    expected_index = pd.MultiIndex.from_product([expected_hours, symbols], names=["timestamp", "symbol"])
    if not old_mask.index.equals(expected_index):
        raise ValueError("30-symbol source inputs do not exactly cover A plus the 168-hour warmup grid")

    verification = read_json(verification_path)
    manifest = read_json(manifest_path)
    provenance = read_json(rebuilt_provenance_path)
    completion = read_json(completion_path)
    lifecycle = read_json(lifecycle_path)
    if verification["status"] != "passed" or not verification["source_bytes_verified"]:
        raise ValueError("raw30 archive verification is not passed")
    if not manifest["complete"] or manifest["interpolation"] is not False:
        raise ValueError("raw30 manifest is incomplete or interpolated")
    if verification["database_signature"] != manifest["database_signature"]:
        raise ValueError("raw30 verification and manifest database signatures differ")
    if provenance["database_sha256"] != manifest["database_signature"]["sha256"]:
        raise ValueError("raw30 prepared input provenance points to a different database")
    if completion["status"] != "passed" or completion["symbol_count"] != 30:
        raise ValueError("raw30 prepared inputs are not frozen for 30 identities")
    if not set(INPUT_COLUMNS) <= prepared_fields:
        raise ValueError(f"raw30 prepared input panel misses fields: {sorted(set(INPUT_COLUMNS) - prepared_fields)}")
    prepared_manifest_key = "development/inputs/panel.values.csv"
    if completion["source_files"].get(prepared_manifest_key) != sha256(rebuilt_panel_path):
        raise ValueError("raw30 A+B input panel no longer matches its preparation snapshot")

    raw_db_before = RAW_DB.stat()
    if raw_db_before.st_size != manifest["database_signature"]["size_bytes"]:
        raise ValueError("raw30 database size differs from its verified source manifest")
    raw_db_sha256 = sha256(RAW_DB)
    if raw_db_sha256 != manifest["database_signature"]["sha256"]:
        raise ValueError("raw30 database content differs from its verified source manifest")
    raw_db_after = RAW_DB.stat()
    if (raw_db_before.st_size, raw_db_before.st_mtime_ns) != (raw_db_after.st_size, raw_db_after.st_mtime_ns):
        raise ValueError("raw30 database changed while checking its A inputs")

    new_contract = read_json(NEW / "source_inputs/development/h1/contract.json")
    old_contract_path = Path(parents[0]["a_universe_path"]).parent / "contract.json"
    old_contract = read_json(old_contract_path)
    old_a_start = old_contract["a_start"]
    old_a_end = old_contract["b_start"]
    if old_a_start != START.isoformat().replace("+00:00", "Z"):
        raise ValueError("old factor archive A start differs from the fixed A boundary")
    if new_contract["start"] != old_a_start:
        raise ValueError("raw30 rebuilt inputs start on a different A contract")
    if old_contract["max_lookback_hours"] > WARMUP_HOURS or new_contract["warmup_hours"] != WARMUP_HOURS:
        raise ValueError("old max lookback or raw30 warmup exceeds the frozen 168-hour contract")
    if old_a_end != END.isoformat().replace("+00:00", "Z"):
        raise ValueError("old factor archive A end differs from the fixed A boundary")
    if pd.Timestamp(new_contract["end"]) <= END:
        raise ValueError("raw30 rebuilt development panel does not cover the full A interval")
    if old_contract["min_symbols"] != 20 or old_contract["sample_hours"] != 1:
        raise ValueError("old A contract differs from frozen min_symbols=20/sample_hours=1")

    raw_input = read_a_window(rebuilt_panel_path, value_columns=list(INPUT_COLUMNS), symbols=symbols,
                              grid_symbols=30)
    if not raw_input.index.equals(old_mask.index) or not raw_input.columns.equals(pd.Index(INPUT_COLUMNS)):
        raise ValueError("raw30 input values are not aligned to the frozen A plus warmup universe")

    primary_stat_before = PRIMARY_DB.stat()
    store = MarketDataStore(PRIMARY_DB)
    with store._connect() as connection:
        connection.execute("PRAGMA query_only=ON")
        primary_page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
        primary_schema_version = int(connection.execute("PRAGMA schema_version").fetchone()[0])
    old_input_panel = load_factor_inputs(store, old_mask, include_liquidations=False)
    primary_stat_after = PRIMARY_DB.stat()
    if (primary_stat_before.st_size, primary_stat_before.st_mtime_ns) != (
        primary_stat_after.st_size, primary_stat_after.st_mtime_ns
    ):
        raise ValueError("primary crypto_quant.sqlite changed during read-only A input reconstruction")
    if not old_input_panel.values.index.equals(old_mask.index):
        raise ValueError("primary input reconstruction does not align to the frozen A universe")
    old_values = old_input_panel.values.loc[:, list(INPUT_COLUMNS)]
    if not np.isfinite(old_values.to_numpy(dtype=float)[~np.isnan(old_values.to_numpy(dtype=float))]).all():
        raise ValueError("primary reconstructed input panel contains infinity")

    common_mask = old_mask & raw_mask
    old_a, raw_a, common_a = a_only(old_mask), a_only(raw_mask), a_only(common_mask)
    timestamp_counts = {
        label: mask.unstack("symbol").sum(axis=1)
        for label, mask in (("old", old_a), ("raw", raw_a), ("common", common_a))
    }
    eligibility_difference = old_a.ne(raw_a)
    mismatch_keys = [
        {"timestamp": ts.isoformat(), "symbol": symbol,
         "old": bool(old_a.loc[(ts, symbol)]), "raw": bool(raw_a.loc[(ts, symbol)])}
        for ts, symbol in eligibility_difference.index[eligibility_difference.to_numpy(dtype=bool)]
    ]
    lifecycle_by_symbol = {event["symbol"]: event for event in lifecycle.get("events", [])}
    mismatches_by_symbol: dict[str, list[pd.Timestamp]] = {}
    for ts, symbol in eligibility_difference.index[eligibility_difference.to_numpy(dtype=bool)]:
        mismatches_by_symbol.setdefault(symbol, []).append(pd.Timestamp(ts))
    lifecycle_alignment = {}
    for symbol, mismatch_times in sorted(mismatches_by_symbol.items()):
        event = lifecycle_by_symbol.get(symbol)
        if event is None:
            lifecycle_alignment[symbol] = {"declared_lifecycle_event": None}
            continue
        cutoff = pd.Timestamp(event["settlement_at"]) - pd.Timedelta(hours=25)
        first, last = min(mismatch_times), max(mismatch_times)
        lifecycle_alignment[symbol] = {
            "published_at": event["published_at"],
            "settlement_at": event["settlement_at"],
            "declared_signal_exclusion_start_25h_before_settlement": cutoff.isoformat(),
            "first_membership_difference": first.isoformat(),
            "last_membership_difference": last.isoformat(),
            "first_difference_matches_declared_cutoff": first == cutoff,
            "difference_cells": len(mismatch_times),
            "interpretation": "alignment with declared lifecycle boundary; later difference extent remains a coverage/membership difference",
        }
    membership_summary = {
        label: {
            "eligible_asset_hours": int(mask.sum()),
            "hours_with_at_least_20_symbols": int((counts >= 20).sum()),
            "hours_below_20_symbols": int((counts < 20).sum()),
            "min_symbols_per_hour": int(counts.min()),
            "max_symbols_per_hour": int(counts.max()),
        }
        for label, (mask, counts) in {
            "old": (old_a, timestamp_counts["old"]),
            "raw": (raw_a, timestamp_counts["raw"]),
            "common": (common_a, timestamp_counts["common"]),
        }.items()
    }

    selected_fields = sorted({field for parent in parents for field in parent["fields"]})
    field_differences = field_pair_comparison(old_values, raw_input, old_mask, raw_mask,
                                              sorted(set(selected_fields) | {"perp_open"}))
    field_support = {}
    common_a_index = common_a.index
    for parent in parents:
        fields = list(parent["fields"])
        old_complete_common = np.isfinite(old_values.loc[common_a_index, fields].to_numpy(dtype=float)).all(axis=1)
        raw_complete_common = np.isfinite(raw_input.loc[common_a_index, fields].to_numpy(dtype=float)).all(axis=1)
        field_support[parent["factor_id"]] = {
            "old_native": factor_input_support(old_values, old_mask, fields),
            "raw_native": factor_input_support(raw_input, raw_mask, fields),
            "common_source_membership": factor_input_support(old_values, common_mask, fields),
            "common_complete_old_and_raw_field_cells": int(
                (common_a.to_numpy(dtype=bool) & old_complete_common & raw_complete_common).sum()
            ),
            "hours_with_at_least_20_common_complete_old_and_raw_field_symbols": int(
                pd.Series(common_a.to_numpy(dtype=bool) & old_complete_common & raw_complete_common,
                          index=common_a.index.get_level_values("timestamp"))
                .groupby(level=0).sum().ge(20).sum()
            ),
        }

    label_summary, _ = source_label_summary(old_values["perp_open"], raw_input["perp_open"],
                                            old_mask, raw_mask, symbols)
    old_usage = read_json(data_usage_path)
    interpolation_log_sha256 = sha256(interpolation_log_path)
    primary_db_identity = {
        "path": str(PRIMARY_DB.relative_to(ROOT)),
        "size_bytes": int(primary_stat_after.st_size),
        "mtime_ns": int(primary_stat_after.st_mtime_ns),
        "page_count": primary_page_count,
        "schema_version": primary_schema_version,
        "full_database_integrity_scan": "not_run; the audit is limited to A source queries and would otherwise scan the entire 32GB primary database",
        "a_source_queries": "completed by read-only FactorEngine input reconstruction",
        "previous_data_usage_snapshot": {
            "path": str(data_usage_path.relative_to(ROOT)),
            "database_size_bytes": old_usage.get("database_size"),
            "database_mtime_ns": old_usage.get("database_mtime_ns"),
            "matches_current_file_stat": (
                old_usage.get("database_size") == primary_stat_after.st_size
                and old_usage.get("database_mtime_ns") == primary_stat_after.st_mtime_ns
            ),
        },
        "interpolation_record": {
            "path": str(interpolation_log_path.relative_to(ROOT)),
            "sha256": interpolation_log_sha256,
        },
    }
    result = {
        "status": "passed_30_symbol_A_input_and_label_alignment_precheck",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "formal_e4_scope": {
            "comparison": "historical A factor archive sourced from interpolated crypto_quant.sqlite versus official raw rebuilt30 source inputs",
            "csv_float_precision": "round_trip; all archived/prepared-panel float CSV reads use pandas float_precision=round_trip; SQLite values are read through its native numeric interface",
            "symbol_count": len(symbols),
            "symbols": symbols,
            "a_start": START.isoformat(),
            "a_end_exclusive": END.isoformat(),
            "warmup_hours": WARMUP_HOURS,
            "warmup_start": WARMUP_START.isoformat(),
            "min_symbols": 20,
            "sample_hours": 1,
            "e4_sample_manifest": str(SAMPLE_OUT.relative_to(ROOT)),
            "e2_front_selection_ready": sample_manifest["full_factor_evaluation_gate"]["available_at_static_freeze"],
            "factor_D_IC_started": False,
        },
        "old_interpolated_source": primary_db_identity,
        "raw_rebuilt_source": {
            "database_path": str(RAW_DB.relative_to(ROOT)),
            "database_sha256": raw_db_sha256,
            "database_size_bytes": int(raw_db_after.st_size),
            "data_verification_status": verification["status"],
            "source_archive_count": verification["source_archive_count"],
            "source_bytes_verified": verification["source_bytes_verified"],
            "archive_parser_issue_count": sum(item.get("issue_count", 0) for item in manifest["source_files"]),
            "missing_archive_404_count": sum(item.get("status") == "archive_missing_404" for item in manifest["source_files"]),
            "interpolation": manifest["interpolation"],
            "historical_causality_certified": manifest["historical_causality_certified"],
            "prepared_panel_path": str(rebuilt_panel_path.relative_to(ROOT)),
            "prepared_panel_sha256": sha256(rebuilt_panel_path),
            "prepared_panel_matches_preparation_manifest": True,
            "provenance_database_sha256": provenance["database_sha256"],
        },
        "a_universe_and_qualification": {
            "old_universe_path": parents[0]["a_universe_path"],
            "raw_universe_path": str(rebuilt_universe_path.relative_to(ROOT)),
            "old_raw_membership_difference_cells_in_a": len(mismatch_keys),
            "membership_difference_cells": mismatch_keys,
            "lifecycle_alignment": lifecycle_alignment,
            "lifecycle_policy": lifecycle.get("policy"),
            "source_native_and_common_coverage": membership_summary,
            "exact_grid_rows_including_warmup": int(len(old_mask)),
        },
        "restored_field_input_support": {
            "fields": field_differences,
            "formula_required_fields": field_support,
            "input_warmup_note": "FactorEngine output was requested only on A plus the frozen 168-hour prewarm grid; its causal input loader reads its documented additional prior support window for derived input fields, with no B/C rows in the returned panel.",
        },
        "forward_label_alignment": label_summary,
        "evidence": {
            "preregistration": {"path": str(PREREG.relative_to(ROOT)), "sha256": sha256(PREREG)},
            "source_data_verification": {"path": str(verification_path.relative_to(ROOT)), "sha256": sha256(verification_path)},
            "source_manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha256(manifest_path)},
            "source_preparation_complete": {"path": str(completion_path.relative_to(ROOT)), "sha256": sha256(completion_path)},
        },
        "limitations": [
            "The separate failed shared_inputs_verification compares only ten shared symbols against a raw archive-derived ablation panel; it is retained as engineering evidence and does not replace this 30-symbol old-interpolated versus raw-rebuilt comparison.",
            "This precheck audits source inputs, eligibility, and label validity/value alignment only. It does not evaluate the six factor expressions or compute D, IC, or spread.",
            "The source manifest does not certify historical publication/receipt timing; E4 conclusions remain historical source sensitivity evidence.",
            "The full 32GB primary SQLite file was not scanned with quick_check; only its A-window source queries and metadata were read in query-only mode.",
        ],
    }
    FORMAL_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
    return result


def main() -> int:
    strict_path = NEW / "shared_inputs_verification.json"
    compat_path = NEW / "eligibility_compatibility.json"
    data_verification_path = NEW / "data/data_verification.json"
    data_manifest_path = NEW / "data/manifest.json"
    provenance_path = NEW / "source_inputs/development/inputs/data-provenance.json"
    completion_path = NEW / "source_inputs/preparation_complete.json"
    strict = read_json(strict_path)
    compatibility = read_json(compat_path)
    verification = read_json(data_verification_path)
    manifest = read_json(data_manifest_path)
    provenance = read_json(provenance_path)
    completion = read_json(completion_path)

    if strict["status"] != "failed":
        raise ValueError("the frozen strict comparison no longer records the expected failure")
    dev = strict["stages"]["development"]
    val = strict["stages"]["internal_validation"]
    expected_difference = {
        "development_eligibility": dev["eligibility"]["exact_difference_count"] == 20
        and dev["eligibility"]["value_difference_beyond_tolerance_count"] == 20,
        "no_development_market_panel_or_funding_difference": all(
            dev[key]["exact_difference_count"] == 0
            and dev[key]["value_difference_beyond_tolerance_count"] == 0
            and dev[key]["missing_key_count"] == 0
            and dev[key]["extra_key_count"] == 0
            for key in ("panel", "funding")
        ) and all(
            item["exact_difference_count"] == 0
            and item["value_difference_beyond_tolerance_count"] == 0
            and item["missing_key_count"] == 0
            and item["extra_key_count"] == 0
            for item in dev["market"].values()
        ),
        "no_internal_validation_difference": val["passed"] is True,
        "eligibility_compatibility_passed": compatibility["status"] == "passed",
    }
    if not all(expected_difference.values()):
        raise ValueError(f"shared-input failure scope changed: {expected_difference}")

    if verification["status"] != "passed":
        raise ValueError("raw archive data verification is not passed")
    if not manifest["complete"] or manifest["interpolation"] is not False:
        raise ValueError("reconstructed source is incomplete or interpolated")
    if verification["database_signature"] != manifest["database_signature"]:
        raise ValueError("data verification and source manifest database signatures differ")
    if provenance["database_sha256"] != manifest["database_signature"]["sha256"]:
        raise ValueError("rebuilt-panel provenance points to a different database")
    if completion["status"] != "passed" or completion["symbol_count"] != 30:
        raise ValueError("rebuilt source inputs are not frozen for 30 identities")

    old_contract = read_json(OLD / "development/h1/contract.json")
    new_contract = read_json(NEW / "source_inputs/development/h1/contract.json")
    if (old_contract["start"], old_contract["end"], old_contract["warmup_hours"]) != (
        new_contract["start"], new_contract["end"], new_contract["warmup_hours"]
    ):
        raise ValueError("source preparation contracts differ")
    if old_contract["warmup_hours"] != WARMUP_HOURS:
        raise ValueError("source contract warmup differs from the E4 contract")

    old_panel = read_a_window(OLD / "development/inputs/panel.values.csv",
                              value_columns=["perp_open"], symbols=SYMBOLS, grid_symbols=10)
    new_panel = read_a_window(NEW / "source_inputs/development/inputs/panel.values.csv",
                              value_columns=["perp_open"], symbols=SYMBOLS, grid_symbols=30)
    if not old_panel.index.equals(new_panel.index):
        raise ValueError("A plus warmup opening-price panel keys differ across sources")
    old_open = old_panel["perp_open"].astype(float)
    new_open = new_panel["perp_open"].astype(float)
    finite_open = np.isfinite(old_open.to_numpy()) & np.isfinite(new_open.to_numpy())
    open_exact = (old_open.to_numpy() == new_open.to_numpy()) | (
        old_open.isna().to_numpy() & new_open.isna().to_numpy()
    )
    if not open_exact.all():
        raise ValueError("A plus warmup opening prices differ across sources")

    old_eligible = mask_frame(OLD / "development/inputs/universe.csv", symbols=SYMBOLS, grid_symbols=10)
    new_eligible = mask_frame(NEW / "source_inputs/development/inputs/universe.csv",
                              symbols=SYMBOLS, grid_symbols=30)
    old_eligible = old_eligible.reindex(old_open.index)
    new_eligible = new_eligible.reindex(new_open.index)
    if old_eligible.isna().any() or new_eligible.isna().any():
        raise ValueError("A plus warmup eligibility does not align to the shared price panel")

    changed = old_eligible.ne(new_eligible)
    in_a = changed.index.get_level_values("timestamp") >= START
    changed_a = changed.loc[changed.to_numpy(dtype=bool) & in_a]
    if len(changed_a) != 20:
        raise ValueError(f"expected exactly 20 A eligibility differences, found {len(changed_a)}")
    expected_keys = {
        (pd.Timestamp(timestamp), symbol)
        for timestamp in ("2023-03-24T12:00:00Z", "2023-03-24T13:00:00Z")
        for symbol in SYMBOLS
    }
    if set(changed_a.index) != expected_keys:
        raise ValueError("A eligibility differences do not match the frozen 2-hour by 10-symbol grid")
    if not old_eligible.loc[changed_a.index].all() or new_eligible.loc[changed_a.index].any():
        raise ValueError("eligibility difference direction changed from reference True to rebuilt False")
    changed_keys = [
        {"timestamp": ts.isoformat(), "symbol": symbol,
         "reference": bool(old_eligible.loc[(ts, symbol)]),
         "rebuilt": bool(new_eligible.loc[(ts, symbol)])}
        for ts, symbol in changed_a.index
    ]

    source_labels = {}
    common_eligible = old_eligible & new_eligible
    for horizon in HORIZONS:
        old_labels = label_frame(old_open, horizon)
        new_labels = label_frame(new_open, horizon)
        if not old_labels.index.equals(new_labels.index):
            raise ValueError(f"h{horizon} label keys differ across sources")
        left = old_labels["forward_return"].to_numpy(dtype=float)
        right = new_labels["forward_return"].to_numpy(dtype=float)
        label_exact = (left == right) | (np.isnan(left) & np.isnan(right))
        if not label_exact.all():
            raise ValueError(f"h{horizon} A labels differ across sources")
        label_times = old_labels.index.get_level_values("timestamp")
        a_label_rows = label_times >= START
        source_labels[str(horizon)] = {
            "definition": "perp_next_open_horizon_hours",
            "a_label_boundary": "label_end < 2024-08-01T00:00:00Z",
            "old_finite_labels": int(np.isfinite(left[a_label_rows]).sum()),
            "rebuilt_finite_labels": int(np.isfinite(right[a_label_rows]).sum()),
            "exact_label_difference_count": 0,
            "old_native_eligible_labeled_cells": metric_counts(
                old_eligible, old_labels, label="old"
            )["eligible_labeled_A_cells"],
            "rebuilt_native_eligible_labeled_cells": metric_counts(
                new_eligible, new_labels, label="rebuilt"
            )["eligible_labeled_A_cells"],
            "common_eligible_labeled_cells": metric_counts(
                common_eligible, old_labels, label="common"
            )["eligible_labeled_A_cells"],
        }

    common_a = common_eligible.loc[common_eligible.index.get_level_values("timestamp") >= START]
    d_pair_coverage = {}
    for delta in DELTAS:
        by_symbol = common_eligible.unstack("symbol").reindex(columns=SYMBOLS)
        times = pd.DatetimeIndex(by_symbol.index)
        previous = by_symbol.reindex(times - pd.Timedelta(hours=delta))
        both_a = (times >= START) & ((times - pd.Timedelta(hours=delta)) >= START)
        pairs = by_symbol & previous.to_numpy()
        pairs.loc[~both_a, :] = False
        d_pair_coverage[str(delta)] = int(pairs.to_numpy(dtype=bool).sum())

    result = {
    "status": "passed_shared_ten_symbol_input_audit",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "contract": {
            "a_start": START.isoformat(), "a_end_exclusive": END.isoformat(),
            "warmup_hours": WARMUP_HOURS, "warmup_start": WARMUP_START.isoformat(),
            "shared_input_comparison_symbol_count": len(SYMBOLS),
            "shared_input_comparison_symbols": SYMBOLS,
            "formal_e4_universe_scope": "fixed 30-symbol cohort with min_symbols=20",
            "label_horizons_hours": list(HORIZONS), "rank_deltas_hours": list(DELTAS),
        },
        "source_data_verification": {
            "status": verification["status"],
            "source_archive_count": verification["source_archive_count"],
            "verified_source_bytes": verification["source_bytes_verified"],
            "zero_archive_parse_issues": sum(
                item.get("issue_count", 0) for item in manifest["source_files"]
            ) == 0,
            "manifest_interpolation": manifest["interpolation"],
            "historical_causality_certified": manifest["historical_causality_certified"],
            "funding_source_note": manifest["source_causality"],
            "database_sha256": manifest["database_signature"]["sha256"],
        },
        "shared_input_failure": {
            "strict_status_retained": strict["status"],
            "development_exact_differences": dev["exact_difference_count"],
            "development_beyond_tolerance_differences": dev["value_difference_beyond_tolerance_count"],
            "eligibility_exact_differences": dev["eligibility"]["exact_difference_count"],
            "eligibility_mismatch_count_in_a_within_shared_ten": len(changed_keys),
            "eligibility_values_parsed_from": ["True", "False"],
            "eligibility_mismatches_in_a": changed_keys,
            "all_other_development_inputs_equal": expected_difference[
                "no_development_market_panel_or_funding_difference"
            ],
            "internal_validation_inputs_equal": val["passed"],
            "multi_factor_rows_excluded_in_existing_compatibility_check": True,
        },
        "a_window_opening_price_alignment": {
            "rows": int(len(old_open)),
            "all_shared_keys_equal": True,
            "exact_open_value_difference_count": 0,
            "finite_price_cells": int(finite_open.sum()),
        },
        "forward_label_alignment": source_labels,
        "common_eligible_a_cells": int(common_a.sum()),
        "common_eligible_d_pair_cells": d_pair_coverage,
        "input_evidence": {
            "shared_inputs_verification": {
                "path": str(strict_path.relative_to(ROOT)), "sha256": sha256(strict_path),
            },
            "eligibility_compatibility": {
                "path": str(compat_path.relative_to(ROOT)), "sha256": sha256(compat_path),
            },
            "data_verification": {
                "path": str(data_verification_path.relative_to(ROOT)), "sha256": sha256(data_verification_path),
            },
            "data_manifest": {
                "path": str(data_manifest_path.relative_to(ROOT)), "sha256": sha256(data_manifest_path),
            },
            "preparation_complete": {
                "path": str(completion_path.relative_to(ROOT)), "sha256": sha256(completion_path),
            },
        },
        "precheck_scope": "shared_ten_symbol_data_engineering_check_only",
        "limitations": [
            "The strict verification remains failed because the development eligibility mask differs at 20 shared-ten cells; this report does not overwrite it.",
            "This comparison is between a raw rebuilt 30-symbol source and a separate raw archive-derived 10-symbol ablation reference; it is not the formal E4 comparison with the interpolated crypto_quant.sqlite source.",
            "Formal E4 must use the frozen 30-symbol identity cohort and min_symbols=20, then intersect source-native eligibility and factor/label validity at relevant endpoints.",
            "The existing multi-factor exclusion proof does not substitute for one-factor native coverage or D/IC checks.",
            "The source manifest explicitly does not certify historical publication/receipt-time causality; funding marks use the declared previous-minute close proxy.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    formal = formal_e4_precheck()
    print(json.dumps({
        "shared_ten_status": result["status"],
        "eligibility_mismatch_count_in_a_within_shared_ten": result["shared_input_failure"]["eligibility_mismatch_count_in_a_within_shared_ten"],
        "opening_rows": result["a_window_opening_price_alignment"]["rows"],
        "label_alignment": {
            horizon: item["exact_label_difference_count"]
            for horizon, item in source_labels.items()
        },
        "formal_30_symbol_status": formal["status"],
        "formal_a_membership_differences": formal["a_universe_and_qualification"]["old_raw_membership_difference_cells_in_a"],
        "formal_label_differences": {
            horizon: item["common_label_value_differences_beyond_tolerance"]
            for horizon, item in formal["forward_label_alignment"].items()
            if horizon in {"1", "4", "24"}
        },
        "sample_manifest": str(SAMPLE_OUT),
        "formal_precheck": str(FORMAL_OUT),
        "shared_ten_precheck": str(OUT),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
