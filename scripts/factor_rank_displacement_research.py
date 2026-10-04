"""Run the pre-registered FM-v6 rank-displacement A-history studies.

This research runner reads archived A factor values and the A-universe sidecars
in place. It never modifies production evidence or reads historical B results.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import gc
import gzip
import hashlib
import io
import json
import math
import os
import resource
import sqlite3
import sys
import time
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "experiments/factor_revalidation/historical_fm_v6_multihorizon_20261002"
ORIGINAL_A = REPO / "experiments/factor_revalidation/historical_fm_v6_rerun_20260930"
ARCHIVE_ROOT = ORIGINAL_A / "runs/factor_archive_v2"
OUT = REPO / "experiments/factor_rank_displacement/20261003"
AUDIT = SOURCE / "archive-audit.json"
CLASS_ORDER = (
    "mixed_structure", "volatility_range", "price_trend_reversal",
    "volume_liquidity", "funding_basis", "positioning_ratio",
)
WINDOWS = (4, 12, 24)
HORIZONS = (1, 4, 24)
DELTAS = (1, 4, 24)
A_START = pd.Timestamp("2022-08-01T00:00:00Z")
A_END = pd.Timestamp("2024-08-01T00:00:00Z")
FRONT_END = pd.Timestamp("2023-08-01T00:00:00Z")
WARMUP_HOURS = 168


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_expression(expression: str) -> tuple[str, list[str]]:
    """Assign one formula-structure class, based only on referenced inputs/operators."""
    from crypto_quant.features.factor_expressions import compile_expression

    compiled = compile_expression(expression)
    fields = list(compiled.fields)
    names = set(fields)
    spot_price = bool(names & {"spot_open", "spot_high", "spot_low", "spot_close"})
    perp_price = bool(names & {"perp_open", "perp_high", "perp_low", "perp_close"})
    basis_fields = bool(names & {"mark_close", "index_close", "premium_index"})
    funding_fields = any(name.startswith("funding_") for name in names)
    funding_basis = funding_fields or basis_fields or (spot_price and perp_price)
    positioning = any(name.startswith(("toptrader_", "global_account_", "open_interest_"))
                       for name in names)
    volume = any("volume" in name or name.endswith("_trades") for name in names)
    price = (spot_price or perp_price) and not (spot_price and perp_price)
    domains = sum((funding_basis, positioning, volume, price))
    if domains > 1:
        family = "mixed_structure"
    elif any(token in expression for token in
             ("ts_std(", "ts_min(", "ts_max(", "ts_mad(", "ts_cov(", "atr(")):
        family = "volatility_range"
    elif price:
        family = "price_trend_reversal"
    elif volume:
        family = "volume_liquidity"
    elif funding_basis:
        family = "funding_basis"
    elif positioning:
        family = "positioning_ratio"
    else:
        family = "mixed_structure"
    return family, fields


def readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def preregister() -> dict[str, Any]:
    from crypto_quant.features.factor_expressions import compile_expression

    if not SOURCE.is_dir() or not AUDIT.is_file():
        raise FileNotFoundError(f"historical A source is missing: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "preregistration.json"
    if destination.exists():
        frozen = json.loads(destination.read_text(encoding="utf-8"))
        expected = frozen.pop("preregistration_sha256", None)
        actual = hashlib.sha256(canonical_json(frozen)).hexdigest()
        if expected != actual:
            raise ValueError("existing preregistration hash does not match its content")
        frozen["preregistration_sha256"] = expected
        return frozen

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    a_items = []
    universe_paths: dict[str, Path] = {}
    registry = readonly_connection(ARCHIVE_ROOT / "registry.sqlite3")
    identity_rows = registry.execute(
        "SELECT factor_id,expanded_expression,direction,semantics_version FROM factor_identities"
    ).fetchall()
    identity_ids = {(row["expanded_expression"], row["direction"], row["semantics_version"]):
                    int(row["factor_id"]) for row in identity_rows}
    if len(identity_ids) != 576:
        raise ValueError(f"expected 576 registered A identities, found {len(identity_ids)}")

    for source_item in audit["factors"]:
        locator = source_item["A_locator"]
        identity = locator["identity"]
        identity_key = (identity["expanded_expression"], str(identity["direction"]),
                        identity["semantics_version"])
        registry_id = identity_ids.get(identity_key)
        if registry_id is None:
            raise ValueError(f"A identity absent from archive registry: {source_item['factor_id']}")
        evaluation_key = locator["evaluation_key"]
        data_version = evaluation_key["data_version"]
        if not data_version.endswith("/A"):
            raise ValueError(f"A locator has unexpected data version: {data_version}")
        run_id = data_version[:-2]
        universe_path = ORIGINAL_A / "runs" / run_id / "A-universe.csv"
        if not universe_path.is_file():
            raise FileNotFoundError(universe_path)
        universe_paths[run_id] = universe_path

        archive_path = ARCHIVE_ROOT / f"factor-{registry_id:06d}.sqlite3"
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        db = readonly_connection(archive_path)
        try:
            meta = db.execute("SELECT identity_json FROM archive_meta WHERE singleton=1").fetchone()
            if meta is None or json.loads(meta["identity_json"]) != {
                    "direction": str(identity["direction"]),
                    "expanded_expression": identity["expanded_expression"],
                    "semantics_version": identity["semantics_version"]}:
                raise ValueError(f"archive identity mismatch for {source_item['factor_id']}")
            value_set_id = int(locator["value_set_id"])
            value_set = db.execute("""SELECT v.data_version,v.computation_semantics,v.blob_id,
                                       b.raw_size,length(b.data_zlib) AS compressed_bytes
                                    FROM value_sets v JOIN blobs b ON b.blob_id=v.blob_id
                                    WHERE v.value_set_id=?""", (value_set_id,)).fetchone()
            if value_set is None or value_set["data_version"] != data_version \
                    or value_set["computation_semantics"] != evaluation_key["evaluator_version"]:
                raise ValueError(f"A value-set pointer mismatch for {source_item['factor_id']}")
            evaluation_id = int(locator["evaluation_id"])
            evaluation = db.execute("""SELECT data_version,evaluator_version,segment,horizon,
                                            value_set_id,factor_values_csv_blob_id
                                         FROM evaluations WHERE evaluation_id=?""",
                                    (evaluation_id,)).fetchone()
            expected = (data_version, evaluation_key["evaluator_version"], "A", "24h", value_set_id)
            actual = tuple(evaluation[k] for k in ("data_version", "evaluator_version", "segment", "horizon", "value_set_id")) \
                if evaluation is not None else None
            if actual != expected or evaluation["factor_values_csv_blob_id"] != value_set["blob_id"]:
                raise ValueError(f"A evaluation pointer mismatch for {source_item['factor_id']}")
            provenance = db.execute(
                "SELECT source_json FROM value_set_sources WHERE value_set_id=? ORDER BY source_json",
                (value_set_id,)).fetchall()
            if not provenance:
                raise ValueError(f"A source provenance missing for {source_item['factor_id']}")
            source_details = [json.loads(row["source_json"]) for row in provenance]
        finally:
            db.close()

        family, fields = classify_expression(identity["expanded_expression"])
        compiled = compile_expression(identity["expanded_expression"])
        base_nodes = sum(1 for _ in ast.walk(compiled.tree))
        variants = []
        for window in WINDOWS:
            smooth_expression = f"ts_mean({identity['expanded_expression']}, {window})"
            smooth = compile_expression(smooth_expression)
            nodes = sum(1 for _ in ast.walk(smooth.tree))
            lookback = int(compiled.lookback_hours + window - 1)
            variants.append({"window_hours": window, "formula_nodes": nodes,
                             "effective_lookback_hours": lookback,
                             "applicable": nodes <= 80 and lookback <= 168,
                             "window_policy": "rolling mean over current and previous window-1 factor observations; min_periods=window"})
        a_items.append({
            "factor_id": source_item["factor_id"],
            "source_run_id": source_item["old_run_id"],
            "candidate_id": source_item["candidate_id"],
            "identity": {"expanded_expression": identity["expanded_expression"],
                         "direction": int(identity["direction"]),
                         "semantics_version": identity["semantics_version"]},
            "formula_class": family,
            "fields": fields,
            "archive": {"registry_factor_id": registry_id, "path": str(archive_path),
                        "evaluation_id": evaluation_id, "value_set_id": value_set_id,
                        "data_version": data_version,
                        "evaluator_version": evaluation_key["evaluator_version"],
                        "raw_bytes": int(value_set["raw_size"]),
                        "compressed_bytes": int(value_set["compressed_bytes"]),
                        "source_provenance": source_details},
            "a_universe_path": str(universe_path),
            "base_formula_nodes": base_nodes,
            "original_lookback_hours": int(compiled.lookback_hours),
            "smooth_variants": variants,
        })
    registry.close()
    if len(a_items) != 576:
        raise ValueError(f"expected 576 A locators, found {len(a_items)}")
    a_items.sort(key=lambda item: (item["identity"]["expanded_expression"],
                                   str(item["identity"]["direction"]),
                                   item["identity"]["semantics_version"]))
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in a_items:
        by_class[item["formula_class"]].append(item)

    e2_parents = []
    e0_samples = []
    for category_index, category in enumerate(CLASS_ORDER, 1):
        members = by_class[category]
        if len(members) < 5:
            raise ValueError(f"class {category} has fewer than five identities for isolated E0 sample")
        for rank, item in enumerate(members[:4], 1):
            e2_parents.append({**item, "parent_order": len(e2_parents) + 1,
                               "class_rank": rank})
        item = members[4]
        e0_samples.append({**item, "sample_order": category_index,
                           "class_rank": 5,
                           "selection_rule": "fifth identity in lexicographic order within class; disjoint from E2 first-four parents"})

    db_path = REPO / "market_data/crypto_quant.sqlite"
    universe_source = REPO / "experiments/factor_mining/long_history_20260919_setup/universe.csv"
    source_signature = audit["input_signatures"]
    db_stat, universe_stat = db_path.stat(), universe_source.stat()
    if (db_stat.st_size != source_signature["db"]["size"]
            or db_stat.st_mtime_ns != source_signature["db"]["mtime_ns"]
            or universe_stat.st_size != source_signature["universe"]["size"]
            or universe_stat.st_mtime_ns != source_signature["universe"]["mtime_ns"]):
        raise ValueError("current historical inputs differ from the revalidation audit signatures")

    universe_inventory = []
    for run_id, path in sorted(universe_paths.items()):
        stat = path.stat()
        universe_inventory.append({"run_id": run_id, "path": str(path), "bytes": stat.st_size,
                                   "sha256": sha256_file(path)})

    contract = {
        "contract_version": "factor-rank-displacement-prereg-v1",
        "preregistered_at": datetime.now(timezone.utc).isoformat(),
        "plan": "docs/plans/FM-v6信号持续性与换手倾向前置Plan_20261003.md",
        "factor_value_csv_numeric_parser": {
            "engine": "pandas C parser",
            "float_precision": "round_trip",
            "scope": "archived factor_value CSV numeric column for full A and streaming front-prefix reads",
            "reason": "preserve distinct archived float64 decimal values during CSV parsing",
        },
        "source_scope": "A archive only; no B or C evaluation results read",
        "source_inventory": {
            "archive_audit": str(AUDIT), "archive_root": str(ARCHIVE_ROOT),
            "archive_output_copy_not_used": str(SOURCE / "runs/factor_archive_v2"),
            "a_reference_count": len(a_items), "registered_identity_count": len(identity_ids),
            "archive_file_count": len(list(ARCHIVE_ROOT.glob("factor-*.sqlite3"))),
            "archive_bytes": sum(path.stat().st_size for path in ARCHIVE_ROOT.glob("factor-*.sqlite3")),
            "archive_references": a_items,
            "historical_input_signatures": source_signature,
            "current_inputs_match_signatures": True,
            "a_universe_files": universe_inventory,
            "a_universe_sha256_count": len({x["sha256"] for x in universe_inventory}),
            "label_source": {"database": str(db_path), "table": "futures_price_bars",
                             "filter": {"data_type": "klines", "interval": "1h"},
                             "price_type": "perpetual trade open", "label": "perp_next_open_Hh",
                             "calculation": "perp_open[t+H+1] / perp_open[t+1] - 1; only within each declared segment; purge label_end >= segment_end"},
        },
        "a_periods": {
            "full": {"start": A_START.isoformat(), "end_exclusive": A_END.isoformat()},
            "front": {"start": A_START.isoformat(), "end_exclusive": FRONT_END.isoformat()},
            "hold": {"start": FRONT_END.isoformat(), "end_exclusive": A_END.isoformat()},
            "source_warmup_hours": WARMUP_HOURS,
            "deltas_hours": list(DELTAS), "prediction_horizons_hours": list(HORIZONS),
        },
        "formula_classification": {
            "rule": "Classify by referenced input fields and explicit formula operators only. More than one field domain maps to mixed_structure; single-domain time-series dispersion/range operators map to volatility_range; otherwise use the sole field domain. Fixed class order: " + ", ".join(CLASS_ORDER),
            "class_counts": {category: len(by_class[category]) for category in CLASS_ORDER},
        },
        "e0": {"fixed_samples": e0_samples,
               "scope": "Full A for six identities disjoint from E2 parents; measure archive read, parse, D compute and peak RSS."},
        "e1": {"fixed_samples": "all 576 A references in source_inventory.archive_references",
               "scope": "Full A native D summaries, hourly D, coverage/ties and same-source A Rank IC associations for H x delta; aggregate by formula class; no filtering by historical acceptance/card status."},
        "e2": {
            "parents": e2_parents,
            "versions": [{"version": "original", "window_hours": 1, "applicable": True}] +
                        [{"version": f"smooth_{window}h", "window_hours": window}
                         for window in WINDOWS],
            "prediction_horizons_hours": list(HORIZONS),
            "displacement_hours": list(DELTAS),
            "rolling_rule": "For each symbol, arithmetic mean of factor values at t, t-1, ..., t-(W-1); require all W finite values; no fill, skip, or interpolation; smoothing precedes cross-sectional ranking/evaluation.",
            "common_support": "For each parent, intersect its original and every applicable smooth version's finite factor values pointwise, universe eligibility, and H-specific label validity. For each delta D also requires common endpoint eligibility at t and exact t-delta. Report each version's native factor and label coverage separately.",
            "front_selection": {"segment": "front", "metric": "mean of D means across delta 1,4,24 on common support; equal weights", "selection": "choose the minimum among original, smooth_4h, smooth_12h, smooth_24h; exclude a version only when the static preflight marks it inapplicable; require all three D means finite; ties prefer original, then shorter window", "ic_use": "IC and spread are not used to select the window"},
            "uncertainty": "Report effect estimates and 95% Newey-West/Bartlett intervals using the source A contract HAC lag 23 on hourly paired differences; no predeclared success threshold and no success declaration.",
        },
        "e3": {
            "front_selection_saved_before_hold_or_full_read": True,
            "selection_artifact": str(OUT / "e2_front_selection.json"),
            "hold_rule": "After persisting front selection, read hold-half A values/labels and evaluate only the frozen selected version plus baseline; then report the full predeclared grid as descriptive analysis.",
            "interpretation": "within-A temporal transfer diagnostic on historically exposed A; not independent out-of-sample proof",
        },
        "stopping_rules": [
            "Stop a parent at the archive boundary on any identity, A locator, source provenance, timestamp grid, or value-set completeness mismatch; never repair from another source.",
            "A missing source value remains missing. Do not fill, interpolate, substitute another database, or alter min_symbols=20.",
            "Preserve inapplicable or failed variants and reasons; do not replace them with candidates from another class.",
            "Do not start a new research batch after 330 elapsed minutes from 2026-10-03T15:49:00Z (hard stop 2026-10-03T21:19:00Z UTC).",
            "If a computation exceeds its staged budget, finish the current safe factor/parent checkpoint and report counts; do not use result-aware sample or formula substitutions.",
        ],
        "execution_order": ["freeze contract", "E0 six isolated A reads and timing", "E2 front half; persist e2_front_selection.json", "E1 full A population", "E2 hold half and predeclared full-A summaries", "E3 frozen transfer comparison", "report"],
    }
    digest = hashlib.sha256(canonical_json(contract)).hexdigest()
    contract["preregistration_sha256"] = digest
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    (OUT / "preregistration.sha256").write_text(digest + "  preregistration.json\n", encoding="ascii")
    return contract


def verify_preregistration() -> dict[str, Any]:
    path = OUT / "preregistration.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    expected = contract.pop("preregistration_sha256", None)
    actual = hashlib.sha256(canonical_json(contract)).hexdigest()
    if expected != actual:
        raise ValueError("preregistration hash mismatch")
    contract["preregistration_sha256"] = expected
    return contract


def amend_source_inventory() -> dict[str, Any]:
    """Add the resolved A source path and source-run checks before any metrics run."""
    path = OUT / "preregistration.json"
    contract = verify_preregistration()
    previous_digest = contract["preregistration_sha256"]
    references = contract["source_inventory"]["archive_references"]
    expected_factor_bytes = 7_672_397_824
    actual_factor_bytes = sum(item.stat().st_size for item in ARCHIVE_ROOT.glob("factor-*.sqlite3"))
    actual_total_bytes = sum(item.stat().st_size for item in ARCHIVE_ROOT.glob("*.sqlite3"))
    if actual_factor_bytes != expected_factor_bytes or actual_total_bytes != 7_672_623_104:
        raise ValueError(f"original A archive bytes changed: factor={actual_factor_bytes}, total={actual_total_bytes}")

    checked_runs: dict[str, dict[str, Any]] = {}
    horizons_by_factor: Counter[str] = Counter()
    for item in references:
        data_version = item["archive"]["data_version"]
        run_id = data_version[:-2]
        if run_id not in checked_runs:
            run_dir = ORIGINAL_A / "runs" / run_id
            spec_path = run_dir / "contract.json"
            if not spec_path.is_file():
                raise FileNotFoundError(spec_path)
            source_spec = json.loads(spec_path.read_text(encoding="utf-8"))
            relevant = {key: source_spec[key] for key in
                        ("a_start", "b_start", "min_symbols", "sample_hours", "max_lookback_hours")}
            expected = {"a_start": "2022-08-01T00:00:00Z",
                        "b_start": "2024-08-01T00:00:00Z",
                        "min_symbols": 20, "sample_hours": 1, "max_lookback_hours": 168}
            if relevant != expected:
                raise ValueError(f"source A contract differs for {run_id}: {relevant}")
            universe_path = run_dir / "A-universe.csv"
            if not universe_path.is_file():
                raise FileNotFoundError(universe_path)
            checked_runs[run_id] = {"path": str(run_dir), "a_start": relevant["a_start"],
                                    "a_end_exclusive": relevant["b_start"],
                                    "min_symbols": relevant["min_symbols"],
                                    "sample_hours": relevant["sample_hours"],
                                    "max_lookback_hours": relevant["max_lookback_hours"],
                                    "a_universe_path": str(universe_path)}
        db = readonly_connection(Path(item["archive"]["path"]))
        try:
            rows = db.execute("SELECT DISTINCT horizon FROM evaluations WHERE segment='A'").fetchall()
            horizons = sorted(row[0] for row in rows)
        finally:
            db.close()
        if horizons != ["24h"]:
            raise ValueError(f"source A horizon inventory differs for {item['factor_id']}: {horizons}")
        horizons_by_factor[",".join(horizons)] += 1

    universe_sample = pd.read_csv(next(iter(checked_runs.values()))["a_universe_path"],
                                  dtype={"symbol": str, "eligible": str})
    universe_sample["timestamp"] = pd.to_datetime(universe_sample["timestamp"], utc=True, errors="raise")
    if set(universe_sample.columns) != {"timestamp", "symbol", "eligible"}:
        raise ValueError("source A-universe file has an unexpected schema")
    universe_shape = {
        "rows": int(len(universe_sample)), "symbols": int(universe_sample["symbol"].nunique()),
        "first_hour": universe_sample["timestamp"].min().isoformat(),
        "last_hour": universe_sample["timestamp"].max().isoformat(),
        "eligible_true": int(universe_sample["eligible"].isin(["true", "True", "1"]).sum()),
    }
    if universe_shape != {"rows": 531_360, "symbols": 30,
                          "first_hour": "2022-07-25T00:00:00+00:00",
                          "last_hour": "2024-07-31T23:00:00+00:00",
                          "eligible_true": 530_223}:
        raise ValueError(f"source A-universe grid differs: {universe_shape}")

    contract["source_path_correction"] = {
        "resolved_locator": "A_locator.root=../factor_archive_v2 relative to historical_fm_v6_rerun_20260930/runs/<source_run_id>",
        "original_a_archive_root": str(ARCHIVE_ROOT),
        "archive_files": len(list(ARCHIVE_ROOT.glob("factor-*.sqlite3"))),
        "archive_factor_files_bytes": actual_factor_bytes,
        "archive_total_bytes_including_registry": actual_total_bytes,
        "multihorizon_output_copy": str(SOURCE / "runs/factor_archive_v2"),
        "copy_policy": "reference original source archive in place; do not duplicate archived payloads",
        "source_runs_checked": len(checked_runs),
        "source_run_a_contracts": list(checked_runs.values()),
        "a_universe_grid_sample": universe_shape,
        "a_universe_sha256_count": contract["source_inventory"]["a_universe_sha256_count"],
    }
    contract["e1"]["native_a_ic_available_horizons"] = [24]
    contract["e1"]["native_a_ic_uncomputed_horizons"] = {
        "1": "No source A evaluation stored for this horizon; do not use B metrics to fill.",
        "4": "No source A evaluation stored for this horizon; do not use B metrics to fill.",
    }
    contract["e1"]["native_a_ic_horizon_inventory"] = dict(horizons_by_factor)
    contract["amendment"] = {
        "amended_at": datetime.now(timezone.utc).isoformat(),
        "supersedes_preregistration_sha256": previous_digest,
        "reason": "resolve A locator root to its original source archive; freeze source-run A contracts, actual archive bytes, and native A horizon availability before numerical computation",
    }
    contract.pop("preregistration_sha256")
    digest = hashlib.sha256(canonical_json(contract)).hexdigest()
    contract["preregistration_sha256"] = digest
    with path.open("w", encoding="utf-8") as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    (OUT / "preregistration.sha256").write_text(digest + "  preregistration.json\n", encoding="ascii")
    (OUT / "preregistration_initial.sha256").write_text(previous_digest + "  initial-before-source-audit-amendment\n", encoding="ascii")
    return contract


def amend_horizon_availability_policy() -> dict[str, Any]:
    """Keep native A horizon availability outcome-dependent and per factor."""
    path = OUT / "preregistration.json"
    contract = verify_preregistration()
    previous_digest = contract["preregistration_sha256"]
    contract["e1"]["scope"] = (
        "Full A native D summaries, hourly D, coverage/ties and same-source A Rank IC associations for H x delta; "
        "read each A evaluation payload's horizon_comparison after E2 front-selection is persisted. "
        "Report H1/H4 as uncomputed per factor when absent; never infer availability from the 24h locator key and never use B to fill."
    )
    contract["e1"].pop("native_a_ic_available_horizons", None)
    contract["e1"].pop("native_a_ic_uncomputed_horizons", None)
    contract["e1"].pop("native_a_ic_horizon_inventory", None)
    contract["e1"]["a_locator_key_horizon_inventory"] = {"24h": 576}
    contract["e1"]["payload_horizon_comparison_availability"] = (
        "Not asserted population-wide at preregistration; resolve per A identity from that same archived A payload during E1."
    )
    contract["evidence_availability_correction"] = {
        "amended_at": datetime.now(timezone.utc).isoformat(),
        "supersedes_preregistration_sha256": previous_digest,
        "reason": "The A natural key is H24, but an A evaluation payload may also contain horizon_comparison H1/H4. Do not infer unavailable horizons from the key alone; freeze a per-factor lookup rule before E0 and inspect population evidence only during E1.",
    }
    contract.pop("preregistration_sha256")
    digest = hashlib.sha256(canonical_json(contract)).hexdigest()
    contract["preregistration_sha256"] = digest
    with path.open("w", encoding="utf-8") as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    (OUT / "preregistration_path_corrected.sha256").write_text(previous_digest + "  source-path-corrected\n", encoding="ascii")
    (OUT / "preregistration.sha256").write_text(digest + "  preregistration.json\n", encoding="ascii")
    log_path = OUT / "preregistration_revisions.jsonl"
    revision = {"at": contract["evidence_availability_correction"]["amended_at"],
                "previous_sha256": previous_digest, "new_sha256": digest,
                "reason": contract["evidence_availability_correction"]["reason"],
                "state": "before E0 numerical computation"}
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(revision, ensure_ascii=False, allow_nan=False) + "\n")
    return contract


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def load_universe(path: Path) -> pd.Series:
    frame = pd.read_csv(path, dtype={"symbol": str, "eligible": str}, parse_dates=["timestamp"])
    if set(frame.columns) != {"timestamp", "symbol", "eligible"}:
        raise ValueError(f"unexpected A-universe schema: {path}")
    if not frame["eligible"].isin(["true", "True", "1", "false", "False", "0"]).all():
        raise ValueError(f"A-universe contains non-boolean membership: {path}")
    frame["eligible"] = frame["eligible"].isin(["true", "True", "1"])
    return frame.set_index(["timestamp", "symbol"])["eligible"].astype(bool)


def load_a_values(item: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    archive = item["archive"]
    db = readonly_connection(Path(archive["path"]))
    try:
        row = db.execute("""SELECT b.raw_size,b.data_zlib FROM value_sets v
                            JOIN blobs b ON b.blob_id=v.blob_id WHERE v.value_set_id=?
                            AND v.data_version=? AND v.computation_semantics=?""",
                         (archive["value_set_id"], archive["data_version"],
                          archive["evaluator_version"])).fetchone()
        if row is None:
            raise ValueError(f"A value set moved or is missing: {item['factor_id']}")
        raw = zlib.decompress(row["data_zlib"])
        if len(raw) != row["raw_size"] or len(raw) != archive["raw_bytes"]:
            raise ValueError(f"A factor-value archive size mismatch: {item['factor_id']}")
    finally:
        db.close()
    frame = pd.read_csv(io.BytesIO(raw), parse_dates=["timestamp"], float_precision="round_trip")
    if list(frame.columns) != ["timestamp", "symbol", "factor_value"]:
        raise ValueError(f"unexpected factor-value CSV schema: {item['factor_id']}")
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError(f"duplicate factor-value key: {item['factor_id']}")
    values = frame.set_index(["timestamp", "symbol"])["factor_value"].astype(float)
    return values, {"compressed_bytes": archive["compressed_bytes"],
                    "raw_bytes": len(raw), "rows": len(frame)}


def load_universe_until(path: Path, end_exclusive: pd.Timestamp) -> pd.Series:
    cutoff = end_exclusive.strftime("%Y-%m-%d %H:%M:%S+00:00").encode("ascii")
    timestamps, symbols, eligible = [], [], []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {"timestamp", "symbol", "eligible"}:
            raise ValueError(f"unexpected A-universe schema: {path}")
        for row in reader:
            if row["timestamp"].encode("ascii") >= cutoff:
                break
            if row["eligible"] not in {"true", "True", "1", "false", "False", "0"}:
                raise ValueError(f"invalid membership token in {path}: {row['eligible']}")
            timestamps.append(pd.Timestamp(row["timestamp"]))
            symbols.append(row["symbol"])
            eligible.append(row["eligible"] in {"true", "True", "1"})
    index = pd.MultiIndex.from_arrays([timestamps, symbols], names=["timestamp", "symbol"])
    return pd.Series(np.asarray(eligible, dtype=bool), index=index, name="eligible")


def load_a_values_until(item: dict[str, Any], end_exclusive: pd.Timestamp) -> tuple[pd.Series, dict[str, Any]]:
    """Stream only the chronological A prefix needed for E2 front selection."""
    archive = item["archive"]
    db = readonly_connection(Path(archive["path"]))
    try:
        row = db.execute("""SELECT v.blob_id,b.raw_size,length(b.data_zlib) AS compressed_bytes
                            FROM value_sets v JOIN blobs b ON b.blob_id=v.blob_id
                            WHERE v.value_set_id=? AND v.data_version=?
                            AND v.computation_semantics=?""",
                         (archive["value_set_id"], archive["data_version"],
                          archive["evaluator_version"])).fetchone()
        if row is None:
            raise ValueError(f"A value set moved or is missing: {item['factor_id']}")
        blob_id, raw_size, compressed_size = int(row["blob_id"]), int(row["raw_size"]), int(row["compressed_bytes"])
        cutoff = end_exclusive.strftime("%Y-%m-%d %H:%M:%S+00:00").encode("ascii")
        prefix = bytearray()
        output_buffer = bytearray()
        header_seen = False
        reached_cutoff = False
        compressed_bytes_read = 0
        decompressor = zlib.decompressobj()
        with db.blobopen("blobs", "data_zlib", blob_id, readonly=True) as blob:
            compressed_pending = b""
            while not reached_cutoff:
                if compressed_pending:
                    chunk = compressed_pending
                else:
                    chunk = blob.read(4096)
                    compressed_bytes_read += len(chunk)
                    if not chunk:
                        output_buffer.extend(decompressor.flush())
                if chunk:
                    output_buffer.extend(decompressor.decompress(chunk, max_length=8192))
                    compressed_pending = decompressor.unconsumed_tail
                else:
                    compressed_pending = b""
                while True:
                    newline = output_buffer.find(b"\n")
                    if newline < 0:
                        break
                    line = bytes(output_buffer[:newline])
                    del output_buffer[:newline + 1]
                    if not header_seen:
                        if line != b"timestamp,symbol,factor_value":
                            raise ValueError(f"A factor CSV header mismatch: {item['factor_id']}")
                        prefix.extend(line + b"\n")
                        header_seen = True
                        continue
                    timestamp = line.split(b",", 1)[0]
                    if timestamp >= cutoff:
                        reached_cutoff = True
                        break
                    prefix.extend(line + b"\n")
                if not chunk and not compressed_pending:
                    break
        if not header_seen or not reached_cutoff:
            raise ValueError(f"A factor CSV ended before front cutoff: {item['factor_id']}")
        frame = pd.read_csv(io.BytesIO(prefix), parse_dates=["timestamp"], float_precision="round_trip")
        if list(frame.columns) != ["timestamp", "symbol", "factor_value"]:
            raise ValueError(f"unexpected factor-value CSV schema: {item['factor_id']}")
        values = frame.set_index(["timestamp", "symbol"])["factor_value"].astype(float)
        return values, {"archive_raw_bytes": raw_size, "archive_compressed_bytes": compressed_size,
                        "compressed_bytes_read_through_front_cutoff": compressed_bytes_read,
                        "prefix_raw_bytes": len(prefix), "prefix_rows": len(frame),
                        "stopped_before_archive_eof": compressed_bytes_read < compressed_size}
    finally:
        db.close()


def _bounded_d_spec(source_spec: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp):
    class Bounded:
        sample_hours = int(source_spec["sample_hours"])
        min_symbols = int(source_spec["min_symbols"])
        stage_hours = int(source_spec["stage_hours"])

        @staticmethod
        def bounds(stage: str):
            if stage != "A":
                raise ValueError("rank-displacement front selection is A-only")
            return start, end

    return Bounded()


def _smooth_from_archived_values(values: pd.Series, universe: pd.Series, window: int) -> pd.Series:
    times = universe.index.get_level_values("timestamp").unique().sort_values()
    symbols = pd.Index(sorted(universe.index.get_level_values("symbol").unique()), name="symbol")
    matrix = values.unstack("symbol").reindex(index=times, columns=symbols)
    smoothed = matrix.rolling(window=window, min_periods=window).mean()
    index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    return pd.Series(smoothed.to_numpy(dtype=float).reshape(-1), index=index, name="factor_value")


def run_e2_front_selection() -> dict[str, Any]:
    from crypto_quant.features.factor_inputs import validate_universe
    from crypto_quant.research.factor_mining.evaluation import evaluate_rank_displacement

    contract = verify_preregistration()
    selection_path = OUT / "e2_front_selection.json"
    if selection_path.exists():
        raise FileExistsError("front-selection artifact already exists; preserve it and choose a new output directory")
    parent_results = []
    hourly_path = OUT / "e2_front_selection_hourly.csv.gz"
    progress_path = OUT / "e2_front_selection_progress.jsonl"
    if hourly_path.exists() or progress_path.exists():
        raise FileExistsError("front-selection partial output already exists; preserve it and choose a new output directory")
    selection_order = ["original", "smooth_4h", "smooth_12h", "smooth_24h"]
    tie_order = {version: index for index, version in enumerate(selection_order)}
    with gzip.open(hourly_path, "wt", newline="", encoding="utf-8", compresslevel=5) as output, \
            progress_path.open("x", encoding="utf-8") as progress:
        columns = ["factor_id", "version", "support", "segment", "delta_hours", "timestamp",
                   "displacement", "status", "eligible_t", "eligible_t_minus_delta",
                   "factor_finite_t", "factor_finite_t_minus_delta", "common_symbols",
                   "eligible_union", "common_coverage_ratio", "unique_t", "unique_t_minus_delta",
                   "has_ties_t", "has_ties_t_minus_delta", "constant_t", "constant_t_minus_delta"]
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for parent in contract["e2"]["parents"]:
            started = time.perf_counter()
            item = {**parent, "archive": parent["archive"]}
            universe_all = load_universe_until(Path(parent["a_universe_path"]), FRONT_END)
            values, read_meta = load_a_values_until(item, FRONT_END)
            if len(values) != len(universe_all) or not values.index.equals(universe_all.index):
                raise ValueError(f"front A factor values and qualification grid differ: {parent['factor_id']}")
            universe = validate_universe(universe_all)
            variants: dict[str, pd.Series] = {"original": values.reindex(universe.index)}
            skipped: dict[str, str] = {}
            for variant in parent["smooth_variants"]:
                window = int(variant["window_hours"])
                version = f"smooth_{window}h"
                if not variant["applicable"]:
                    skipped[version] = (f"static preflight: nodes={variant['formula_nodes']}, "
                                       f"effective_lookback={variant['effective_lookback_hours']} "
                                       "exceeds the frozen 80-node or 168h budget")
                    continue
                variants[version] = _smooth_from_archived_values(values, universe, window).reindex(universe.index)
            common_values = np.ones(len(universe), dtype=bool)
            for series in variants.values():
                common_values &= np.isfinite(series.to_numpy(dtype=float))
            common_series = pd.Series(common_values, index=universe.index)
            source_spec = json.loads((Path(parent["a_universe_path"]).parent / "contract.json").read_text(encoding="utf-8"))
            d_spec = _bounded_d_spec(source_spec, A_START, FRONT_END)
            version_results = {}
            for version in selection_order:
                if version not in variants:
                    continue
                common_factor = variants[version].where(common_series)
                result = evaluate_rank_displacement(common_factor, universe, d_spec, "A")
                version_results[version] = result
                write_hourly_rows(writer, parent["factor_id"], result, segment="front_A",
                                  version=version, support="all_applicable_versions_common")
            output.flush()
            scores = {}
            for version, result in version_results.items():
                delta_means = {str(delta): result["deltas"][str(delta)]["summary"]["mean"] for delta in DELTAS}
                valid = all(value is not None for value in delta_means.values())
                scores[version] = {"delta_means": delta_means,
                                   "eligible_for_selection": valid,
                                   "equal_weight_delta_mean": sum(delta_means.values()) / len(delta_means)
                                   if valid else None}
            eligible_versions = [version for version in selection_order
                                 if version in scores and scores[version]["eligible_for_selection"]]
            selected = min(eligible_versions,
                           key=lambda version: (scores[version]["equal_weight_delta_mean"], tie_order[version])) \
                if eligible_versions else None
            record = {
                "parent_order": parent["parent_order"], "factor_id": parent["factor_id"],
                "formula_class": parent["formula_class"], "identity": parent["identity"],
                "fields": parent["fields"], "status": "selected" if selected else "no_valid_front_D",
                "front_exclusive_end": FRONT_END.isoformat(),
                "warmup_start": (A_START - pd.Timedelta(hours=WARMUP_HOURS)).isoformat(),
                "prefix_read": read_meta,
                "support": {"universe_rows": len(universe), "eligible_rows": int(universe.sum()),
                            "all_applicable_versions_common_factor_rows": int(common_series.sum()),
                            "common_eligible_factor_rows": int((common_series & universe).sum()),
                            "applicable_versions": list(version_results), "skipped_versions": skipped},
                "predeclared_selection_scores": scores, "selected_version": selected,
                "tie_order": selection_order,
                "compute_seconds": time.perf_counter() - started,
            }
            parent_results.append(record)
            progress.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            progress.flush()
    if len(parent_results) != len(contract["e2"]["parents"]):
        raise ValueError("front selection did not complete every frozen E2 parent")
    result = {
        "status": "complete", "stage": "E2-front-only-window-selection",
        "preregistration_sha256": contract["preregistration_sha256"],
        "selection_rule": contract["e2"]["front_selection"],
        "selection_uses_ic": False,
        "post_front_values_read": False,
        "front_exclusive_end": FRONT_END.isoformat(),
        "parents": parent_results,
        "hourly_output": str(hourly_path), "progress_log": str(progress_path),
        "elapsed_seconds": sum(parent["compute_seconds"] for parent in parent_results),
        "selected_version_counts": dict(Counter(parent["selected_version"] or "no_selection"
                                                 for parent in parent_results)),
    }
    digest = hashlib.sha256(canonical_json(result)).hexdigest()
    result["selection_sha256"] = digest
    selection_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")
    return result


def verify_front_selection_gate(contract: dict[str, Any]) -> dict[str, Any]:
    path = OUT / "e2_front_selection.json"
    if not path.is_file():
        raise FileNotFoundError("E1 is gated on the persisted E2 front-selection artifact")
    selection = json.loads(path.read_text(encoding="utf-8"))
    claimed = selection.pop("selection_sha256", None)
    actual = hashlib.sha256(canonical_json(selection)).hexdigest()
    selection["selection_sha256"] = claimed
    if selection.get("status") != "complete":
        raise ValueError("E2 front-selection artifact is incomplete")
    if selection.get("preregistration_sha256") != contract["preregistration_sha256"]:
        raise ValueError("E2 front-selection artifact belongs to a different preregistration")
    if selection.get("post_front_values_read") is not False:
        raise ValueError("front-selection gate does not confirm the front-only read boundary")
    progress_path = OUT / "e2_front_selection_progress.jsonl"
    progress_rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_parents = contract["e2"]["parents"]
    if len(progress_rows) != len(expected_parents) or len(selection.get("parents", [])) != len(expected_parents):
        raise ValueError("front-selection record count differs from the frozen 24-parent list")
    if progress_rows != selection["parents"]:
        raise ValueError("front-selection summary and append-only parent progress records differ")
    canonical_claim_matches = claimed == actual
    normalized_hash = actual
    if not canonical_claim_matches:
        # The original writer hashed integer delta keys; JSON persistence restores
        # those object keys as strings and changes sort order (1,4,24 -> 1,24,4).
        # Accept only this concrete, fully verified serialization difference.
        normalized = copy.deepcopy(selection)
        normalized.pop("selection_sha256", None)
        for parent in normalized["parents"]:
            for score in parent.get("predeclared_selection_scores", {}).values():
                delta_means = score.get("delta_means", {})
                if set(delta_means) != {"1", "4", "24"}:
                    raise ValueError("front-selection hash mismatch is not the known delta-key serialization issue")
                score["delta_means"] = {int(key): delta_means[key] for key in delta_means}
        normalized_hash = hashlib.sha256(canonical_json(normalized)).hexdigest()
        if normalized_hash != claimed:
            raise ValueError("front-selection hash mismatch is not explained by integer-to-string delta-key serialization")
    decisions = []
    for parent, expected in zip(selection["parents"], expected_parents):
        if parent.get("factor_id") != expected["factor_id"] or parent.get("identity") != expected["identity"]:
            raise ValueError("front-selection parent identity differs from the frozen list")
        scores = parent.get("predeclared_selection_scores", {})
        tie_order = {"original": 0, "smooth_4h": 1, "smooth_12h": 2, "smooth_24h": 3}
        applicable = [version for version, score in scores.items()
                      if score.get("eligible_for_selection") and score.get("equal_weight_delta_mean") is not None]
        should_select = min(applicable, key=lambda version: (scores[version]["equal_weight_delta_mean"],
                                                              tie_order[version])) if applicable else None
        if parent.get("selected_version") != should_select:
            raise ValueError(f"frozen D-only selection rule mismatch for {parent['factor_id']}")
        if any("ic" in key.lower() or "spread" in key.lower() for key in scores):
            raise ValueError("front-selection score unexpectedly contains prediction metrics")
        decisions.append({"factor_id": parent["factor_id"], "selected_version": should_select})
    recomputed_counts = dict(Counter(decision["selected_version"] or "no_selection" for decision in decisions))
    if recomputed_counts != selection.get("selected_version_counts"):
        raise ValueError("front-selection aggregate count differs from parent decisions")
    gate_status = "hash_verified" if canonical_claim_matches else "hash_verified_after_known_delta_key_normalization"
    gate_audit = {"status": gate_status, "selection_path": str(path),
                  "claimed_sha256": claimed, "canonical_content_sha256": actual,
                  "normalized_pre_persistence_sha256": normalized_hash,
                  "known_serialization_difference": None if canonical_claim_matches else
                  "delta_means object keys were int in the hashed in-memory result and str after JSON round-trip; sorted key order changes",
                  "selection_progress_parent_records_match": True,
                  "frozen_parent_identities_match": True,
                  "selection_rule_recomputed_for_all_parents": True,
                  "selection_uses_ic": False,
                  "post_front_values_read": False,
                  "selected_version_counts": recomputed_counts,
                  "selection_not_modified": True,
                  "note": "Unknown hash mismatches fail closed. The one verified legacy mismatch is reproduced only after restoring integer delta keys in a temporary copy; the persisted selection file is never modified."}
    audit_path = OUT / "e2_front_hash_correction.json"
    if audit_path.exists():
        existing = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing != gate_audit:
            raise ValueError("front-selection hash-correction audit already exists with different contents")
    else:
        audit_path.write_text(json.dumps(gate_audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")
    selection["selection_sha256"] = claimed
    selection["gate_audit"] = gate_audit
    return selection


def _a_horizon_evidence(payload: dict[str, Any], direction: int) -> dict[str, Any]:
    reports = payload.get("horizon_comparison", {}).get("horizons", {})
    output: dict[str, Any] = {}
    for horizon in HORIZONS:
        report = reports.get(str(horizon))
        if report is None:
            output[str(horizon)] = {"status": "not_computed_in_archived_A_payload"}
            continue
        if report.get("segment") != "A" or report.get("horizon_hours") != horizon \
                or report.get("direction") != direction:
            raise ValueError(f"archived A horizon report identity mismatch at H={horizon}")
        summary = report["summary"]
        rank_ic = summary["rank_ic"]
        raw_ci = rank_ic.get("ci")
        directional_ci = ([min(direction * raw_ci[0], direction * raw_ci[1]),
                           max(direction * raw_ci[0], direction * raw_ci[1])]
                          if raw_ci is not None else None)
        output[str(horizon)] = {
            "status": "available" if rank_ic.get("mean") is not None else "available_but_rank_ic_unavailable",
            "rank_ic": rank_ic,
            "directional_rank_ic_mean": (direction * rank_ic["mean"] if rank_ic.get("mean") is not None else None),
            "directional_rank_ic_ci": directional_ci,
            "directional_spread": summary["directional_spread"],
            "coverage": report["coverage"],
            "sample_hours": report["sample_hours"],
            "label": report["label"],
        }
    return output


def _distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {"n": int(len(array)), "mean": float(array.mean()) if len(array) else None,
            "std": float(array.std(ddof=1)) if len(array) > 1 else None,
            "p10": float(np.quantile(array, 0.10)) if len(array) else None,
            "median": float(np.median(array)) if len(array) else None,
            "p90": float(np.quantile(array, 0.90)) if len(array) else None}


def _persist_e1_summary(result_path: Path, summary: dict[str, Any]) -> tuple[str, float, float]:
    serialize_start = time.perf_counter()
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    serialize_seconds = time.perf_counter() - serialize_start
    write_start = time.perf_counter()
    result_path.write_text(summary_text, encoding="utf-8", newline="")
    write_seconds = time.perf_counter() - write_start
    return summary_text, serialize_seconds, write_seconds


def run_e1_full() -> dict[str, Any]:
    from crypto_quant.features.factor_inputs import validate_universe
    from crypto_quant.research.factor_mining.contracts import ResearchSpec
    from crypto_quant.research.factor_mining.evaluation import evaluate_rank_displacement

    contract = verify_preregistration()
    selection = verify_front_selection_gate(contract)
    del selection
    result_path, progress_path = OUT / "e1_results.json", OUT / "e1_progress.jsonl"
    factor_path, hourly_path = OUT / "e1_factors.jsonl", OUT / "e1_hourly.csv.gz"
    scatter_path = OUT / "e1_scatter.csv"
    if any(path.exists() for path in (result_path, progress_path, factor_path, hourly_path, scatter_path)):
        raise FileExistsError("E1 output already exists; preserve it and use a new output directory for a rerun")
    references = contract["source_inventory"]["archive_references"]
    if len(references) != 576:
        raise ValueError(f"E1 population changed after preregistration: {len(references)}")
    db_signature = contract["source_inventory"]["historical_input_signatures"]["db"]
    db_stat = Path(db_signature["path"]).stat()
    if db_stat.st_size != db_signature["size"] or db_stat.st_mtime_ns != db_signature["mtime_ns"]:
        raise ValueError("label source database differs from the archived A input signature")
    universe_entries = contract["source_inventory"]["a_universe_files"]
    universe_path = Path(universe_entries[0]["path"])
    if sha256_file(universe_path) != universe_entries[0]["sha256"]:
        raise ValueError("source A-universe changed after preregistration")
    universe = validate_universe(load_universe(universe_path))
    if len(universe) != contract["source_path_correction"]["a_universe_grid_sample"]["rows"]:
        raise ValueError("source A-universe no longer covers the frozen full grid")
    source_spec = json.loads((universe_path.parent / "contract.json").read_text(encoding="utf-8"))
    source_spec.setdefault("b_horizons", [1, 4, 24])
    spec = ResearchSpec.from_dict(source_spec)
    spec_adapter = {"source_run_contract": str(universe_path.parent / "contract.json"),
                    "added_only": {"b_horizons": [1, 4, 24]},
                    "unchanged_a_fields": ["a_start", "b_start", "sample_hours", "min_symbols",
                                            "stage_hours", "hac_lags", "groups", "min_periods", "confidence"]}

    records: list[dict[str, Any]] = []
    all_d: dict[str, list[float]] = {str(delta): [] for delta in DELTAS}
    all_ic: dict[str, list[float]] = {str(h): [] for h in HORIZONS}
    family_d: dict[str, dict[str, list[float]]] = defaultdict(lambda: {str(d): [] for d in DELTAS})
    family_ic: dict[str, dict[str, list[float]]] = defaultdict(lambda: {str(h): [] for h in HORIZONS})
    scatter_rows_written = 0
    read_parse_seconds = d_api_seconds = payload_read_seconds = 0.0
    hourly_write_seconds = progress_write_seconds = progress_json_seconds = factor_json_seconds = 0.0
    peak_rss = _peak_rss_bytes()
    scan_started = time.perf_counter()
    with gzip.open(hourly_path, "wt", newline="", encoding="utf-8", compresslevel=5) as hourly, \
            progress_path.open("x", encoding="utf-8") as progress, \
            factor_path.open("x", encoding="utf-8") as factor_log, \
            scatter_path.open("x", newline="", encoding="utf-8") as scatter_file:
        hourly_columns = ["factor_id", "version", "support", "segment", "delta_hours", "timestamp",
                          "displacement", "status", "eligible_t", "eligible_t_minus_delta",
                          "factor_finite_t", "factor_finite_t_minus_delta", "common_symbols",
                          "eligible_union", "common_coverage_ratio", "unique_t", "unique_t_minus_delta",
                          "has_ties_t", "has_ties_t_minus_delta", "constant_t", "constant_t_minus_delta"]
        hourly_writer = csv.DictWriter(hourly, fieldnames=hourly_columns, extrasaction="ignore")
        hourly_writer.writeheader()
        scatter_writer = csv.DictWriter(scatter_file, fieldnames=[
            "factor_id", "formula_class", "horizon_hours", "delta_hours", "displacement_mean",
            "displacement_valid_periods", "displacement_valid_share", "directional_rank_ic_mean",
            "directional_rank_ic_ci_low", "directional_rank_ic_ci_high", "rank_ic_valid_periods",
            "directional_spread_mean", "directional_spread_valid_periods", "label_eligible_observations",
            "source_status", "evidence_scope"], extrasaction="ignore")
        scatter_writer.writeheader()
        for ordinal, item in enumerate(references, 1):
            factor_started = time.perf_counter()
            try:
                read_started = time.perf_counter()
                values, resource_info = load_a_values(item)
                if len(values) != len(universe) or not values.index.equals(universe.index):
                    raise ValueError(f"native A value rows differ from frozen qualification panel: {item['factor_id']}")
                read_parse_done = time.perf_counter()
                payload_started = time.perf_counter()
                payload = load_a_payload(item)
                if payload.get("segment") != "A" or payload.get("direction") != item["identity"]["direction"]:
                    raise ValueError(f"native A payload identity mismatch: {item['factor_id']}")
                horizons = _a_horizon_evidence(payload, item["identity"]["direction"])
                payload_done = time.perf_counter()
                del payload
                d_started = time.perf_counter()
                displacement = evaluate_rank_displacement(values, universe, spec, "A")
                d_done = time.perf_counter()
                write_started = time.perf_counter()
                write_hourly_rows(hourly_writer, item["factor_id"], displacement,
                                  segment="full_A", version="original", support="native")
                hourly.flush()
                hourly_write_seconds += time.perf_counter() - write_started
                native_coverage = native_a_factor_coverage(displacement, universe)
                record = {
                    "factor_id": item["factor_id"], "class": item["formula_class"],
                    "identity": item["identity"], "fields": item["fields"],
                    "source_run_id": item["source_run_id"], "candidate_id": item["candidate_id"],
                    "archive": {key: item["archive"][key] for key in
                                ("registry_factor_id", "path", "evaluation_id", "value_set_id", "data_version",
                                 "evaluator_version", "raw_bytes", "compressed_bytes")},
                    "native_A_factor_coverage": native_coverage,
                    "native_A_horizon_evidence": horizons,
                    "rank_displacement": {
                        delta: {"summary": value["summary"], "stages": value["stages"],
                                "coverage": value["coverage"]}
                        for delta, value in displacement["deltas"].items()},
                    "resource": resource_info,
                    "timing_seconds": {"archive_read_decompress_parse": read_parse_done - read_started,
                                       "A_payload_read_decompress_parse": payload_done - payload_started,
                                       "rank_displacement_api": d_done - d_started,
                                       "factor_total_before_logs": time.perf_counter() - factor_started},
                    "status": "complete",
                }
                for horizon in HORIZONS:
                    horizon_record = horizons[str(horizon)]
                    ic_mean = horizon_record.get("directional_rank_ic_mean")
                    if ic_mean is not None:
                        all_ic[str(horizon)].append(float(ic_mean))
                        family_ic[item["formula_class"]][str(horizon)].append(float(ic_mean))
                for delta, d_result in record["rank_displacement"].items():
                    d_mean = d_result["summary"]["mean"]
                    if d_mean is not None:
                        all_d[delta].append(float(d_mean))
                        family_d[item["formula_class"]][delta].append(float(d_mean))
                    for horizon in HORIZONS:
                        horizon_record = horizons[str(horizon)]
                        ic_mean = horizon_record.get("directional_rank_ic_mean")
                        scatter_writer.writerow({
                            "factor_id": item["factor_id"], "formula_class": item["formula_class"],
                            "horizon_hours": horizon, "delta_hours": delta,
                            "displacement_mean": d_mean,
                            "displacement_valid_periods": d_result["summary"]["valid_periods"],
                            "displacement_valid_share": d_result["summary"]["valid_period_share"],
                            "directional_rank_ic_mean": ic_mean,
                            "directional_rank_ic_ci_low": (horizon_record.get("directional_rank_ic_ci") or [None, None])[0],
                            "directional_rank_ic_ci_high": (horizon_record.get("directional_rank_ic_ci") or [None, None])[1],
                            "rank_ic_valid_periods": horizon_record.get("rank_ic", {}).get("n"),
                            "directional_spread_mean": horizon_record.get("directional_spread", {}).get("mean"),
                            "directional_spread_valid_periods": horizon_record.get("directional_spread", {}).get("n"),
                            "label_eligible_observations": horizon_record.get("coverage", {}).get("eligible_observations"),
                            "source_status": horizon_record.get("status"),
                            "evidence_scope": "same-source archived A H comparison; D is native factor-only diagnostic; descriptive cross-candidate scatter",
                        })
                        scatter_rows_written += 1
                factor_serialize_start = time.perf_counter()
                factor_line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                factor_json_seconds += time.perf_counter() - factor_serialize_start
                serialization_start = time.perf_counter()
                progress_line = json.dumps({"ordinal": ordinal, "factor_id": item["factor_id"],
                                            "status": "complete", "elapsed_seconds": record["timing_seconds"]["factor_total_before_logs"]},
                                           ensure_ascii=False, allow_nan=False) + "\n"
                progress_json_seconds += time.perf_counter() - serialization_start
                progress_start = time.perf_counter()
                factor_log.write(factor_line)
                progress.write(progress_line)
                factor_log.flush()
                progress.flush()
                progress_write_seconds += time.perf_counter() - progress_start
                records.append(record)
                read_parse_seconds += read_parse_done - read_started
                payload_read_seconds += payload_done - payload_started
                d_api_seconds += d_done - d_started
                peak_rss = max(peak_rss, _peak_rss_bytes())
                del values, displacement, record, horizons
                import gc
                gc.collect()
            except Exception as exc:
                failure = {"ordinal": ordinal, "factor_id": item["factor_id"],
                           "error_type": type(exc).__name__, "error": str(exc), "status": "failed"}
                progress.write(json.dumps(failure, ensure_ascii=False, allow_nan=False) + "\n")
                progress.flush()
                raise
    if len(records) != 576:
        raise ValueError(f"E1 population did not complete: {len(records)} of 576")
    directional_ic_distribution = {h: _distribution(values) for h, values in all_ic.items()}
    family_directional_ic_distribution = {
        family: {h: _distribution(values) for h, values in horizons.items()}
        for family, horizons in family_ic.items()}
    unique_ic_counts = {
        str(h): sum(record["native_A_horizon_evidence"][str(h)].get("directional_rank_ic_mean") is not None
                    for record in records)
        for h in HORIZONS
    }
    for horizon, expected_count in unique_ic_counts.items():
        family_count = sum(values.get(horizon, {"n": 0})["n"]
                           for values in family_directional_ic_distribution.values())
        if directional_ic_distribution[horizon]["n"] != expected_count or family_count != expected_count:
            raise ValueError(f"E1 IC aggregation is not unique by factor and horizon H={horizon}")
    aggregates = {
        "all_identity_distribution_by_delta": {delta: _distribution(values) for delta, values in all_d.items()},
        "directional_ic_by_horizon": directional_ic_distribution,
        "unique_factor_count_with_finite_directional_ic_by_horizon": unique_ic_counts,
        "formula_family_d_by_delta": {family: {delta: _distribution(values) for delta, values in deltas.items()}
                                       for family, deltas in family_d.items()},
        "formula_family_directional_ic_by_horizon": family_directional_ic_distribution,
    }
    elapsed = time.perf_counter() - scan_started
    summary = {
        "status": "complete", "stage": "E1-native-A-full-pool",
        "preregistration_sha256": contract["preregistration_sha256"],
        "front_selection_sha256": json.loads((OUT / "e2_front_selection.json").read_text(encoding="utf-8"))["selection_sha256"],
        "population_count": len(records), "included_by_card_or_B_status": False,
        "horizon_evidence_counts": {str(h): sum(record["native_A_horizon_evidence"][str(h)]["status"]
                                                 != "not_computed_in_archived_A_payload" for record in records)
                                    for h in HORIZONS},
        "aggregate": aggregates,
        "interpretation": "Historical A diagnostics only. The cross-candidate D/IC scatter is descriptive, has unequal native coverage, and does not imply causality or eligibility thresholds. H1/H4 are marked per identity when absent; B results are never used.",
        "timing_seconds": {"archive_read_decompress_parse": read_parse_seconds,
                           "A_payload_read_decompress_parse": payload_read_seconds,
                           "rank_displacement_api": d_api_seconds,
                           "hourly_compress_write": hourly_write_seconds,
                           "progress_and_factor_log_write": progress_write_seconds,
                           "progress_json_serialization": progress_json_seconds,
                           "factor_summary_json_serialization": factor_json_seconds,
                           "total_elapsed": elapsed},
        "storage": {"hourly_path": str(hourly_path), "hourly_bytes": hourly_path.stat().st_size,
                    "factor_log_path": str(factor_path), "factor_log_bytes": factor_path.stat().st_size,
                    "progress_log_path": str(progress_path), "progress_log_bytes": progress_path.stat().st_size,
                    "scatter_path": str(scatter_path), "scatter_bytes": scatter_path.stat().st_size,
                    "scatter_rows": scatter_rows_written, "peak_rss_bytes": peak_rss,
                    "archive_copy_bytes": 0},
        "research_spec_adapter": spec_adapter,
        "outputs": {"hourly_d": str(hourly_path), "per_factor": str(factor_path),
                    "progress": str(progress_path), "scatter": str(scatter_path)},
    }
    summary_text, summary_serialize_seconds, summary_write_seconds = _persist_e1_summary(result_path, summary)
    (OUT / "e1_finalization.json").write_text(json.dumps({
        "summary_serialize_seconds": summary_serialize_seconds,
        "summary_write_seconds": summary_write_seconds,
        "summary_bytes": len(summary_text.encode("utf-8")),
        "measured_total_before_finalization_seconds": elapsed,
        "full_elapsed_seconds": time.perf_counter() - scan_started,
        "e1_hourly_bytes": hourly_path.stat().st_size,
        "e1_progress_bytes": progress_path.stat().st_size,
        "e1_factor_summary_bytes": factor_path.stat().st_size,
        "e1_scatter_bytes": scatter_path.stat().st_size,
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def load_a_labels_by_horizon(universe: pd.Series, spec: Any, db_path: Path) -> tuple[dict[int, pd.DataFrame], dict[str, Any]]:
    from crypto_quant.data_access.market_data import MarketDataStore, USD_M_PERPETUAL, resolve_market_symbols
    from crypto_quant.features.factor_inputs import FactorInputPanel, validate_universe
    from crypto_quant.research.factor_mining.evaluation import build_labels

    members = validate_universe(universe)
    timestamps = members.index.get_level_values("timestamp").unique().sort_values()
    symbols = pd.Index(sorted(members.index.get_level_values("symbol").unique()), name="symbol")
    source = MarketDataStore(db_path)
    open_matrix = pd.DataFrame(np.nan, index=timestamps, columns=symbols, dtype=float)
    bar_rows: dict[str, int] = {}
    for symbol in symbols:
        market_symbols = resolve_market_symbols(symbol)
        bars = source.load_bars(
            USD_M_PERPETUAL, market_symbols.perpetual, interval="1h", price_type="trade",
            start=timestamps.min(), end=timestamps.max(), include_incomplete=True, derive=False,
        )
        # The contract multiplier is constant through time, so division matches the
        # FactorEngine perp_open units and leaves simple-return labels unchanged.
        open_values = bars["open"].astype(float) / market_symbols.perpetual_multiplier
        open_matrix[symbol] = open_values.reindex(timestamps)
        bar_rows[symbol] = int(open_values.notna().sum())
    panel_values = pd.DataFrame({"perp_open": open_matrix.to_numpy(dtype=float).reshape(-1)},
                                index=members.index)
    panel = FactorInputPanel(panel_values, members, {"label_source": str(db_path)})
    labels: dict[int, pd.DataFrame] = {}
    coverage: dict[str, Any] = {}
    for horizon in HORIZONS:
        frame = build_labels(panel, spec, "A", horizon_hours=horizon)
        labels[horizon] = frame
        frame_times = frame.index.get_level_values("timestamp")
        in_segment = (frame_times >= A_START) & (frame_times < A_END)
        eligible_in_segment = frame["eligible"] & in_segment
        finite_labels = eligible_in_segment & ~frame["purged"] & np.isfinite(frame["forward_return"])
        coverage[str(horizon)] = {
            "finite_eligible_observations": int(finite_labels.sum()),
            "eligible_observations": int(eligible_in_segment.sum()),
            "purged_eligible_observations": int((eligible_in_segment & frame["purged"]).sum()),
            "warmup_eligible_input_observations": int((frame["eligible"] & ~in_segment).sum()),
            "label": f"perp_next_open_{horizon}h",
        }
    return labels, {"bar_rows_by_symbol": bar_rows, "label_coverage": coverage,
                    "source_table": "futures_price_bars", "data_type": "klines",
                    "interval": "1h", "price_type": "perpetual trade open",
                    "input_signature": {"size": db_path.stat().st_size, "mtime_ns": db_path.stat().st_mtime_ns}}


def _compact_study_result(result: dict[str, Any]) -> dict[str, Any]:
    versions = {}
    for version, entry in result["versions"].items():
        versions[version] = {
            "horizons": {h: {delta: details for delta, details in grid.items()}
                         for h, grid in entry["horizons"].items()},
        }
    pairs = {}
    for version, entry in result["pair_effects"].items():
        pairs[version] = {"horizons": {
            h: {delta: {key: value for key, value in details.items() if key != "periods"}
                for delta, details in grid.items()}
            for h, grid in entry["horizons"].items()}}
    compact = {"definition_version": result["definition_version"], "segment": result["segment"],
            "sample_hours": result["sample_hours"], "direction": result["direction"],
            "reference_id": result["reference_id"], "versions": versions,
            "native_coverage": result["native_coverage"], "pair_effects": pairs}
    if "native_prediction_coverage" in result:
        compact["native_prediction_coverage"] = result["native_prediction_coverage"]
    return compact


def _write_gzip_json(path: Path, value: dict[str, Any]) -> int:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=5) as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    return path.stat().st_size


def run_e2_full() -> dict[str, Any]:
    from crypto_quant.features.factor_inputs import validate_universe
    from crypto_quant.research.factor_mining.contracts import ResearchSpec
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    from factor_rank_displacement_study import compute_factor_variant_study

    contract = verify_preregistration()
    verify_front_selection_gate(contract)
    e1_path = OUT / "e1_results.json"
    if not e1_path.is_file():
        raise FileNotFoundError("E2 common-support evaluation starts after full E1 completes")
    e1 = json.loads(e1_path.read_text(encoding="utf-8"))
    if e1.get("status") != "complete" or e1.get("population_count") != 576:
        raise ValueError("E2 requires the completed frozen 576-factor E1 scan")
    summary_path = OUT / "e2_results.json"
    progress_path = OUT / "e2_progress.jsonl"
    factors_path = OUT / "e2_factors.jsonl"
    if any(path.exists() for path in (summary_path, progress_path, factors_path)):
        raise FileExistsError("E2 output already exists; preserve it and choose a new output directory for a rerun")
    detail_dir = OUT / "e2_periods"
    detail_dir.mkdir(exist_ok=False)

    references = contract["source_inventory"]["archive_references"]
    universe_entry = contract["source_inventory"]["a_universe_files"][0]
    universe_path = Path(universe_entry["path"])
    if sha256_file(universe_path) != universe_entry["sha256"]:
        raise ValueError("A-universe source changed after the front-selection gate")
    universe = validate_universe(load_universe(universe_path))
    source_spec_raw = json.loads((universe_path.parent / "contract.json").read_text(encoding="utf-8"))
    source_spec_raw.setdefault("b_horizons", [1, 4, 24])
    spec = ResearchSpec.from_dict(source_spec_raw)
    db_path = Path(contract["source_inventory"]["label_source"]["database"])
    signature = contract["source_inventory"]["historical_input_signatures"]["db"]
    current = db_path.stat()
    if current.st_size != signature["size"] or current.st_mtime_ns != signature["mtime_ns"]:
        raise ValueError("E2 label source database differs from frozen source signature")
    label_started = time.perf_counter()
    labels, label_source = load_a_labels_by_horizon(universe, spec, db_path)
    label_load_seconds = time.perf_counter() - label_started
    if len(labels) != 3 or any(not label.index.equals(universe.index) for label in labels.values()):
        raise ValueError("E2 label horizons do not align exactly to the A input/universe grid")

    segments = {
        "front": (A_START, FRONT_END),
        "hold": (FRONT_END, A_END),
        "full_A": (A_START, A_END),
    }
    selection = json.loads((OUT / "e2_front_selection.json").read_text(encoding="utf-8"))
    parents_out: list[dict[str, Any]] = []
    write_json_seconds = write_disk_seconds = compute_seconds = read_values_seconds = 0.0
    version_build_seconds = raw_detail_write_seconds = 0.0
    peak_rss = _peak_rss_bytes()
    run_started = time.perf_counter()
    with progress_path.open("x", encoding="utf-8") as progress, factors_path.open("x", encoding="utf-8") as factor_log:
        for parent in contract["e2"]["parents"]:
            parent_started = time.perf_counter()
            values_started = time.perf_counter()
            values, resource_info = load_a_values(parent)
            values_done = time.perf_counter()
            if len(values) != len(universe) or not values.index.equals(universe.index):
                raise ValueError(f"E2 A factor values and universe differ: {parent['factor_id']}")
            version_build_started = time.perf_counter()
            version_values: dict[str, pd.Series] = {"original": values.reindex(universe.index)}
            skipped: dict[str, str] = {}
            for variant in parent["smooth_variants"]:
                window = int(variant["window_hours"])
                version = f"smooth_{window}h"
                if not variant["applicable"]:
                    skipped[version] = "outside frozen formula node/lookback budget"
                    continue
                version_values[version] = _smooth_from_archived_values(values, universe, window).reindex(universe.index)
            version_build_seconds += time.perf_counter() - version_build_started
            selected_parent = selection["parents"][int(parent["parent_order"]) - 1]
            if selected_parent["factor_id"] != parent["factor_id"]:
                raise ValueError("front-selected parent order no longer matches preregistered E2 order")
            version_ids = list(version_values)
            if len(version_ids) != len(selected_parent["support"]["applicable_versions"]):
                raise ValueError(f"front selection and static smooth applicability differ: {parent['factor_id']}")
            segment_outputs = {}
            for segment_name, bounds in segments.items():
                compute_started = time.perf_counter()
                study = compute_factor_variant_study(
                    version_values, universe, labels, spec, int(parent["identity"]["direction"]),
                    reference_id="original", segment=bounds,
                )
                compute_seconds += time.perf_counter() - compute_started
                raw_path = detail_dir / f"parent-{parent['parent_order']:02d}_{segment_name}.json.gz"
                detail_write_started = time.perf_counter()
                raw_size = _write_gzip_json(raw_path, study)
                raw_detail_write_seconds += time.perf_counter() - detail_write_started
                compact = _compact_study_result(study)
                segment_outputs[segment_name] = {
                    "path": str(raw_path), "compressed_bytes": raw_size, "summary": compact,
                    "period_count": int((bounds[1] - bounds[0]) / pd.Timedelta(hours=spec.sample_hours)),
                }
                del study
                gc.collect()
                progress_row = {"parent_order": parent["parent_order"], "factor_id": parent["factor_id"],
                                "segment": segment_name, "status": "complete",
                                "path": str(raw_path), "compressed_bytes": raw_size,
                                "elapsed_seconds": time.perf_counter() - compute_started}
                t0 = time.perf_counter()
                progress.write(json.dumps(progress_row, ensure_ascii=False, allow_nan=False) + "\n")
                progress.flush()
                write_json_seconds += time.perf_counter() - t0
            factor_record = {
                "parent_order": parent["parent_order"], "factor_id": parent["factor_id"],
                "formula_class": parent["formula_class"], "identity": parent["identity"],
                "fields": parent["fields"], "source_run_id": parent["source_run_id"],
                "selected_version_from_front_D": selected_parent["selected_version"],
                "front_selection_score": selected_parent["predeclared_selection_scores"],
                "applicable_versions": version_ids, "inapplicable_versions": skipped,
                "native_value_resource": resource_info,
                "timing_seconds": {"A_value_read_decompress_parse": values_done - values_started,
                                   "three_segment_compute_and_write": time.perf_counter() - parent_started},
                "segments": segment_outputs,
            }
            t0 = time.perf_counter()
            factor_log.write(json.dumps(factor_record, ensure_ascii=False, allow_nan=False) + "\n")
            factor_log.flush()
            write_json_seconds += time.perf_counter() - t0
            write_disk_seconds += sum(entry["compressed_bytes"] for entry in segment_outputs.values())
            read_values_seconds += values_done - values_started
            parents_out.append({"parent_order": parent["parent_order"], "factor_id": parent["factor_id"],
                                "applicable_versions": version_ids,
                                "inapplicable_versions": skipped,
                                "selected_version_from_front_D": selected_parent["selected_version"],
                                "segment_paths": {key: value["path"] for key, value in segment_outputs.items()},
                                "segment_compressed_bytes": {key: value["compressed_bytes"] for key, value in segment_outputs.items()},
                                "compact_results": {key: value["summary"] for key, value in segment_outputs.items()}})
            peak_rss = max(peak_rss, _peak_rss_bytes())
            del values, version_values
            gc.collect()
    if len(parents_out) != len(contract["e2"]["parents"]):
        raise ValueError("E2 did not complete all frozen parents")
    summary = {
        "status": "complete", "stage": "E2-fixed-smoothing-grid",
        "preregistration_sha256": contract["preregistration_sha256"],
        "front_selection_sha256": selection["selection_sha256"],
        "parent_count": len(parents_out),
        "eligible_for_smoothing_parent_count": sum(bool(parent["applicable_versions"][1:]) for parent in parents_out),
        "all_frozen_parent_count_denominator": len(parents_out),
        "segments": {name: {"start": bounds[0].isoformat(), "end_exclusive": bounds[1].isoformat()}
                     for name, bounds in segments.items()},
        "label_source": label_source,
        "label_build_seconds": label_load_seconds,
        "common_support_contract": contract["e2"]["common_support"],
        "effect_reporting": "Report point estimates and HAC intervals only; no success threshold or route decision.",
        "parents": parents_out,
        "timing_seconds": {"A_value_read_decompress_parse_all_parents": read_values_seconds,
                           "smooth_version_build_all_parents": version_build_seconds,
                           "helper_compute_all_segments": compute_seconds,
                           "raw_detail_json_gzip_write_all_segments": raw_detail_write_seconds,
                           "summary_and_progress_json_write": write_json_seconds,
                           "label_source_and_build": label_load_seconds,
                           "total_elapsed_including_labels": time.perf_counter() - run_started + label_load_seconds},
        "storage": {"detail_directory": str(detail_dir), "detail_compressed_bytes": write_disk_seconds,
                    "factor_summary_bytes": factors_path.stat().st_size,
                    "progress_bytes": progress_path.stat().st_size,
                    "peak_rss_bytes": peak_rss, "source_archive_copy_bytes": 0},
        "outputs": {"summary": str(summary_path), "factors": str(factors_path), "progress": str(progress_path),
                    "detail_directory": str(detail_dir)},
    }
    summary_serialize_start = time.perf_counter()
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    summary_serialize_seconds = time.perf_counter() - summary_serialize_start
    summary_write_start = time.perf_counter()
    summary_path.write_text(summary_text, encoding="utf-8", newline="")
    summary_write_seconds = time.perf_counter() - summary_write_start
    (OUT / "e2_finalization.json").write_text(json.dumps({
        "summary_serialize_seconds": summary_serialize_seconds,
        "summary_write_seconds": summary_write_seconds,
        "summary_bytes": len(summary_text.encode("utf-8")),
        "full_elapsed_seconds": time.perf_counter() - run_started,
        "detail_compressed_bytes": write_disk_seconds,
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def _summarize_e3_hold_effects(
        grid_summary: dict[str, dict[str, list[dict[str, Any]]]],
        no_smooth_variant_applicable_count: int,
        selected_original_with_smoothing_available_count: int) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for horizon, by_delta in grid_summary.items():
        aggregate[horizon] = {}
        for delta, rows in by_delta.items():
            metric_summary = {}
            for metric in ("d", "ic", "spread"):
                estimates = [row[metric] for row in rows if row[metric].get("mean") is not None]
                intervals = [row[metric]["ci"] for row in rows if row[metric].get("ci") is not None]
                metric_summary[metric] = {
                    "n_eligible_smooth_parents": len(rows),
                    "n_with_estimate": len(estimates),
                    "mean_of_parent_effect_means": float(np.mean([item["mean"] for item in estimates]))
                    if estimates else None,
                    "positive_parent_point_estimates": sum(item["mean"] > 0 for item in estimates),
                    "nonpositive_parent_point_estimates": sum(item["mean"] <= 0 for item in estimates),
                    "ci_lower_above_zero_count": sum(bounds[0] > 0 for bounds in intervals),
                    "ci_upper_below_zero_count": sum(bounds[1] < 0 for bounds in intervals),
                }
            aggregate[horizon][delta] = {
                "eligible_smooth_parent_denominator": len(rows),
                "all_24_parent_denominator": 24,
                "front_selected_original_with_smoothing_available_count": selected_original_with_smoothing_available_count,
                "no_smooth_variant_applicable_parent_count": no_smooth_variant_applicable_count,
                "effects": metric_summary,
            }
    return aggregate


def run_e3_transfer() -> dict[str, Any]:
    """Read only the already-computed E2 hold segment for front-frozen choices."""
    contract = verify_preregistration()
    verify_front_selection_gate(contract)
    e2_path = OUT / "e2_results.json"
    e1_path = OUT / "e1_results.json"
    if not e1_path.is_file() or not e2_path.is_file():
        raise FileNotFoundError("E3 requires completed E1 and E2 summaries")
    e2 = json.loads(e2_path.read_text(encoding="utf-8"))
    e1 = json.loads(e1_path.read_text(encoding="utf-8"))
    if e2.get("status") != "complete" or e2.get("parent_count") != 24:
        raise ValueError("E3 requires every frozen E2 parent and segment")
    if e1.get("status") != "complete" or e1.get("population_count") != 576:
        raise ValueError("E3 requires completed full-population E1 diagnostics")
    selection = json.loads((OUT / "e2_front_selection.json").read_text(encoding="utf-8"))
    if e2.get("preregistration_sha256") != contract["preregistration_sha256"]:
        raise ValueError("E2 results belong to a different preregistration")
    if e2.get("front_selection_sha256") != selection.get("selection_sha256"):
        raise ValueError("E2 results do not match the persisted front selection")
    result_path, csv_path = OUT / "e3_results.json", OUT / "e3_effects.csv"
    if result_path.exists() or csv_path.exists():
        raise FileExistsError("E3 output already exists; preserve it and choose a new output directory for a rerun")
    selected_by_id = {item["factor_id"]: item for item in selection["parents"]}
    factors_by_id = {item["factor_id"]: item for item in e2["parents"]}
    preregistered_by_id = {item["factor_id"]: item for item in contract["e2"]["parents"]}
    if (len(selected_by_id) != 24 or len(factors_by_id) != 24 or len(preregistered_by_id) != 24
            or set(selected_by_id) != set(factors_by_id)
            or set(selected_by_id) != set(preregistered_by_id)):
        raise ValueError("E3 frozen contract, front selections, and E2 parent identities do not match")
    effects = []
    grid_summary: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(h): {str(delta): [] for delta in DELTAS} for h in HORIZONS}
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "factor_id", "formula_class", "selected_version", "horizon_hours", "delta_hours",
            "front_d_improvement", "front_d_ci_low", "front_d_ci_high",
            "hold_d_improvement", "hold_d_ci_low", "hold_d_ci_high",
            "hold_directional_ic_change", "hold_directional_ic_ci_low", "hold_directional_ic_ci_high",
            "hold_directional_spread_change", "hold_directional_spread_ci_low", "hold_directional_spread_ci_high",
            "hold_common_periods", "hold_spread_pair_periods", "comparison_status"], extrasaction="ignore")
        writer.writeheader()
        for factor_id, selected in selected_by_id.items():
            e2_parent = factors_by_id[factor_id]
            formula_class = preregistered_by_id[factor_id]["formula_class"]
            if e2_parent["parent_order"] != preregistered_by_id[factor_id]["parent_order"]:
                raise ValueError(f"E2 parent order differs from the frozen contract for {factor_id}")
            applicable = e2_parent["applicable_versions"][1:]
            if selected["selected_version"] == "original":
                effects.append({"factor_id": factor_id, "formula_class": formula_class,
                                "selected_version": "original",
                                "status": "original_selected_no_modification" if applicable else
                                         "no_smooth_variant_applicable",
                                "eligible_smooth_parent": False,
                                "smooth_variant_applicable_count": len(applicable),
                                "reason": ("original was available and selected by the pre-registered D-only rule; "
                                           "no smoothing contrast is reported" if applicable else
                                           "all fixed windows exceed the pre-registered node/lookback budget"),
                                "paired_effect_estimates": None,
                                "paired_effect_confidence_intervals": None,
                                "segments": {name: e2_parent["compact_results"][name]["versions"]["original"]
                                             for name in ("front", "hold", "full_A")}})
                continue
            if selected["selected_version"] not in applicable:
                raise ValueError(f"selected E3 window was statically inapplicable for {factor_id}")
            parent_effects = {"factor_id": factor_id, "formula_class": formula_class,
                              "selected_version": selected["selected_version"],
                              "status": "frozen_front_selection_evaluated_on_hold",
                              "eligible_smooth_parent": True, "segment_effects": {}}
            for horizon in HORIZONS:
                h = str(horizon)
                parent_effects["segment_effects"][h] = {}
                for delta in DELTAS:
                    d = str(delta)
                    segment_details = {}
                    for segment_name in ("front", "hold", "full_A"):
                        pair = e2_parent["compact_results"][segment_name]["pair_effects"].get(
                            selected["selected_version"], {}).get("horizons", {}).get(h, {}).get(d)
                        if pair is None:
                            raise ValueError(f"selected version pair evidence missing: {factor_id} H={h} D={d} {segment_name}")
                        segment_details[segment_name] = pair
                    hold_pair = segment_details["hold"]
                    hold_summary = hold_pair["summary"]
                    d_effect = hold_summary["paired_displacement_improvement"]
                    ic_effect = hold_summary["paired_directional_rank_ic_change"]
                    spread_effect = hold_summary["paired_directional_spread_change"]
                    grid_summary[h][d].append({"factor_id": factor_id, "d": d_effect, "ic": ic_effect,
                                               "spread": spread_effect,
                                               "coverage": hold_pair["coverage"]})
                    parent_effects["segment_effects"][h][d] = segment_details
                    writer.writerow({
                        "factor_id": factor_id, "formula_class": formula_class,
                        "selected_version": selected["selected_version"], "horizon_hours": horizon,
                        "delta_hours": delta,
                        "front_d_improvement": segment_details["front"]["summary"]["paired_displacement_improvement"]["mean"],
                        "front_d_ci_low": (segment_details["front"]["summary"]["paired_displacement_improvement"]["ci"] or [None, None])[0],
                        "front_d_ci_high": (segment_details["front"]["summary"]["paired_displacement_improvement"]["ci"] or [None, None])[1],
                        "hold_d_improvement": d_effect["mean"],
                        "hold_d_ci_low": (d_effect["ci"] or [None, None])[0],
                        "hold_d_ci_high": (d_effect["ci"] or [None, None])[1],
                        "hold_directional_ic_change": ic_effect["mean"],
                        "hold_directional_ic_ci_low": (ic_effect["ci"] or [None, None])[0],
                        "hold_directional_ic_ci_high": (ic_effect["ci"] or [None, None])[1],
                        "hold_directional_spread_change": spread_effect["mean"],
                        "hold_directional_spread_ci_low": (spread_effect["ci"] or [None, None])[0],
                        "hold_directional_spread_ci_high": (spread_effect["ci"] or [None, None])[1],
                        "hold_common_periods": hold_pair["coverage"].get("paired_d_ic_valid_periods"),
                        "hold_spread_pair_periods": hold_pair["coverage"].get("spread_pair_valid_periods"),
                        "comparison_status": "paired_effect_estimate_only_no_threshold",
                    })
            effects.append(parent_effects)
    no_smooth_variant_applicable_count = sum(
        not bool(factors_by_id[factor_id]["applicable_versions"][1:]) for factor_id in factors_by_id)
    selected_original_with_smoothing_available_count = sum(
        item["selected_version"] == "original" and item["smooth_variant_applicable_count"] > 0
        for item in effects if item["selected_version"] == "original")
    aggregate = _summarize_e3_hold_effects(
        grid_summary, no_smooth_variant_applicable_count,
        selected_original_with_smoothing_available_count)
    output = {
        "status": "complete", "stage": "E3-front-selected-within-A-transfer",
        "preregistration_sha256": contract["preregistration_sha256"],
        "front_selection_sha256": selection["selection_sha256"],
        "front_hash_correction_audit": str(OUT / "e2_front_hash_correction.json"),
        "selected_smooth_parent_count": sum(e["eligible_smooth_parent"] for e in effects),
        "selected_smooth_window_counts": dict(Counter(
            e["selected_version"] for e in effects if e["eligible_smooth_parent"])),
        "selected_original_with_smoothing_available_count": selected_original_with_smoothing_available_count,
        "no_smooth_variant_applicable_parent_count": no_smooth_variant_applicable_count,
        "all_parent_denominator": 24,
        "aggregate_hold_effects": aggregate,
        "parents": effects,
        "interpretation": "A-internal temporal transfer diagnostic on previously exposed history, not independent out-of-sample evidence. Front choices use D only. Hold effects are paired estimates with HAC intervals and have no predeclared success threshold; positive or negative counts are descriptive.",
        "outputs": {"effects_csv": str(csv_path), "summary": str(result_path)},
    }
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    return output


def load_a_payload(item: dict[str, Any]) -> dict[str, Any]:
    archive = item["archive"]
    db = readonly_connection(Path(archive["path"]))
    try:
        row = db.execute("""SELECT b.raw_size,b.data_zlib FROM evaluations e
                            JOIN blobs b ON b.blob_id=e.payload_blob_id
                            WHERE e.evaluation_id=? AND e.data_version=? AND e.segment='A'
                            AND e.horizon='24h'""",
                         (archive["evaluation_id"], archive["data_version"])).fetchone()
        if row is None:
            raise ValueError(f"A evaluation payload is missing: {item['factor_id']}")
        raw = zlib.decompress(row["data_zlib"])
        if len(raw) != row["raw_size"]:
            raise ValueError(f"A evaluation payload size mismatch: {item['factor_id']}")
        return json.loads(raw.decode("utf-8"))
    finally:
        db.close()


def write_hourly_rows(writer: csv.DictWriter, factor_id: str, result: dict[str, Any], *,
                      segment: str, version: str = "original", support: str = "native") -> None:
    for delta, details in result["deltas"].items():
        for row in details["periods"]:
            writer.writerow({
                "factor_id": factor_id, "version": version, "support": support,
                "segment": segment, "delta_hours": delta,
                **{key: row.get(key) for key in (
                    "timestamp", "displacement", "status", "eligible_t",
                    "eligible_t_minus_delta", "factor_finite_t",
                    "factor_finite_t_minus_delta", "common_symbols",
                    "eligible_union", "common_coverage_ratio", "unique_t",
                    "unique_t_minus_delta", "has_ties_t", "has_ties_t_minus_delta",
                    "constant_t", "constant_t_minus_delta")},
            })


def native_a_factor_coverage(d_result: dict[str, Any], universe: pd.Series) -> dict[str, Any]:
    periods = [row for row in d_result["deltas"]["1"]["periods"]
               if A_START <= pd.Timestamp(row["timestamp"]) < A_END]
    expected_periods = int((A_END - A_START) / pd.Timedelta(hours=1))
    if len(periods) != expected_periods:
        raise ValueError(f"native A coverage timestamp count mismatch: {len(periods)} != {expected_periods}")
    symbol_count = universe.index.get_level_values("symbol").nunique()
    eligible_rows = sum(int(row["eligible_t"]) for row in periods)
    eligible_finite_rows = sum(int(row["factor_finite_t"]) for row in periods)
    return {"grid_rows": expected_periods * symbol_count,
            "eligible_rows": eligible_rows,
            "eligible_finite_rows": eligible_finite_rows,
            "eligible_finite_share": eligible_finite_rows / eligible_rows if eligible_rows else None}


def run_e0() -> dict[str, Any]:
    from crypto_quant.research.factor_mining.contracts import ResearchSpec
    from crypto_quant.research.factor_mining.evaluation import evaluate_rank_displacement

    contract = verify_preregistration()
    if contract["source_inventory"]["current_inputs_match_signatures"] is not True:
        raise ValueError("historical inputs were not frozen against the source signatures")
    if (OUT / "e0_results.json").exists():
        raise FileExistsError("completed E0 output already exists; preserve it and use a new run directory for a rerun")
    prior_runs = [*OUT.glob("e0_progress.jsonl"), *OUT.glob("e0_attempt*_progress.jsonl")]
    attempt = 1 + len(prior_runs)
    samples = contract["e0"]["fixed_samples"]
    universe_path = Path(samples[0]["a_universe_path"])
    expected_universe_hash = contract["source_inventory"]["a_universe_files"][0]["sha256"]
    if sha256_file(universe_path) != expected_universe_hash:
        raise ValueError("source A-universe changed since preregistration")
    universe = load_universe(universe_path)
    expected_rows = contract["source_path_correction"]["a_universe_grid_sample"]["rows"]
    if len(universe) != expected_rows or not universe.index.is_unique:
        raise ValueError(f"source A-universe grid is incomplete: rows={len(universe)}")
    from crypto_quant.features.factor_inputs import validate_universe
    universe = validate_universe(universe)

    compressed_total = sum(item["archive"]["compressed_bytes"]
                           for item in contract["source_inventory"]["archive_references"])
    raw_total = sum(item["archive"]["raw_bytes"]
                    for item in contract["source_inventory"]["archive_references"])
    suffix = "" if attempt == 1 else f"_attempt{attempt:02d}"
    hourly_path = OUT / f"e0{suffix}_hourly.csv.gz"
    progress_path = OUT / ("e0_progress.jsonl" if attempt == 1 else f"e0{suffix}_progress.jsonl")
    results: list[dict[str, Any]] = []
    read_parse_seconds = 0.0
    compute_seconds = 0.0
    compression_write_seconds = 0.0
    progress_serialize_seconds = 0.0
    progress_write_seconds = 0.0
    sampled_compressed = 0
    sampled_raw = 0
    session_start = time.perf_counter()
    with gzip.open(hourly_path, "wt", newline="", encoding="utf-8", compresslevel=5) as output, \
            progress_path.open("x", encoding="utf-8") as progress:
        columns = ["factor_id", "version", "support", "segment", "delta_hours", "timestamp",
                   "displacement", "status", "eligible_t", "eligible_t_minus_delta",
                   "factor_finite_t", "factor_finite_t_minus_delta", "common_symbols",
                   "eligible_union", "common_coverage_ratio", "unique_t", "unique_t_minus_delta",
                   "has_ties_t", "has_ties_t_minus_delta", "constant_t", "constant_t_minus_delta"]
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in samples:
            record: dict[str, Any] = {"factor_id": item["factor_id"], "category": item["formula_class"],
                                      "identity": item["identity"], "archive": item["archive"],
                                      "source_a_universe": item["a_universe_path"]}
            rss_before = _peak_rss_bytes()
            try:
                start = time.perf_counter()
                values, resource_info = load_a_values(item)
                end_load = time.perf_counter()
                if len(values) != expected_rows or not values.index.equals(universe.index):
                    raise ValueError(f"A values do not match full source universe grid: {item['factor_id']}")
                parse_done = time.perf_counter()
                source_spec = json.loads((Path(item["a_universe_path"]).parent / "contract.json").read_text(encoding="utf-8"))
                # This historical contract predates today's required B-horizon declaration.
                # The API uses its unchanged A bounds and sampling values for D.
                source_spec.setdefault("b_horizons", [1, 4, 24])
                spec = ResearchSpec.from_dict(source_spec)
                result = evaluate_rank_displacement(values, universe, spec, "A")
                done = time.perf_counter()
                old_payload = load_a_payload(item)
                if old_payload.get("direction") != item["identity"]["direction"]:
                    raise ValueError(f"source A direction differs from frozen identity: {item['factor_id']}")
                record.update({
                    "status": "complete", "resource": resource_info,
                    "timing_seconds": {"archive_read_decompress_parse": parse_done - start,
                                       "rank_displacement_api": done - parse_done,
                                       "a_payload_read": time.perf_counter() - done},
                    "peak_rss_before_bytes": rss_before, "peak_rss_after_bytes": _peak_rss_bytes(),
                    "native_factor_coverage": native_a_factor_coverage(result, universe),
                    "displacement": {delta: {"summary": value["summary"], "stages": value["stages"],
                                               "coverage": value["coverage"]}
                                     for delta, value in result["deltas"].items()},
                    "source_a_prediction_h24": {
                        "rank_ic": old_payload["summary"]["rank_ic"],
                        "directional_spread": old_payload["summary"]["directional_spread"],
                        "coverage": old_payload["coverage"],
                    },
                    "source_a_horizon_comparison_keys": sorted(
                        old_payload.get("horizon_comparison", {}).get("horizons", {}).keys(),
                        key=int),
                })
                write_start = time.perf_counter()
                bytes_before = hourly_path.stat().st_size if hourly_path.exists() else 0
                write_hourly_rows(writer, item["factor_id"], result, segment="full_A")
                output.flush()
                bytes_after = hourly_path.stat().st_size
                write_end = time.perf_counter()
                compression_write_seconds += write_end - write_start
                record["timing_seconds"]["diagnostic_serialize_compress_write"] = write_end - write_start
                record["diagnostic_output_bytes"] = bytes_after - bytes_before
                read_parse_seconds += parse_done - start
                compute_seconds += done - parse_done
                sampled_compressed += resource_info["compressed_bytes"]
                sampled_raw += resource_info["raw_bytes"]
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                              peak_rss_before_bytes=rss_before, peak_rss_after_bytes=_peak_rss_bytes())
                results.append(record)
                serialization_start = time.perf_counter()
                progress_line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                progress_serialize_seconds += time.perf_counter() - serialization_start
                progress_write_start = time.perf_counter()
                progress.write(progress_line)
                progress.flush()
                progress_write_seconds += time.perf_counter() - progress_write_start
                raise
            results.append(record)
            serialization_start = time.perf_counter()
            progress_line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            progress_serialize_seconds += time.perf_counter() - serialization_start
            progress_write_start = time.perf_counter()
            progress.write(progress_line)
            progress.flush()
            progress_write_seconds += time.perf_counter() - progress_write_start
            output.flush()

    contract_factors = contract["source_inventory"]["archive_references"]
    read_parse_estimate = (read_parse_seconds / sampled_compressed * compressed_total
                           if sampled_compressed else None)
    compute_estimate = compute_seconds / len(results) * len(contract_factors) if results else None
    estimate = {"archive_read_decompress_parse_seconds": read_parse_estimate,
                "D_api_full_pool_seconds": compute_estimate,
                "total_estimated_full_pool_seconds": (read_parse_estimate + compute_estimate)
                if read_parse_estimate is not None and compute_estimate is not None else None,
                "estimation_method": "E0 elapsed throughput scaled by compressed archive bytes for parsing and by factor count for D; includes no E2 IC re-evaluation."}
    prior_failures = []
    for prior_path in prior_runs:
        for line in prior_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("status") == "failed":
                prior_failures.append({"progress_log": str(prior_path),
                                       "factor_id": entry.get("factor_id"),
                                       "error_type": entry.get("error_type"),
                                       "error": entry.get("error")})
    summary = {
        "status": "complete", "preregistration_sha256": contract["preregistration_sha256"],
        "samples_completed": len(results), "elapsed_seconds": time.perf_counter() - session_start,
        "e0_samples": results, "full_pool_estimate": estimate,
        "write_and_storage": {
            "e0_hourly_compress_write_seconds": compression_write_seconds,
            "progress_jsonl_serialize_seconds": progress_serialize_seconds,
            "progress_jsonl_write_seconds": progress_write_seconds,
            "e0_hourly_bytes": hourly_path.stat().st_size,
            "e0_progress_bytes": progress_path.stat().st_size,
            "e0_hourly_rows": sum(sum(delta["coverage"]["expected_periods"]
                                       for delta in item["displacement"].values())
                                   for item in results),
            "e1_full_pool_hourly_rows": 576 * len(DELTAS) * int((A_END - A_START) / pd.Timedelta(hours=1)),
            "source_archives_copied": 0,
            "e1_storage_projection": "calculated from exact E0 compressed bytes per persisted hourly row",
        },
        "peak_rss_bytes": max((r.get("peak_rss_after_bytes", 0) for r in results), default=0),
        "e0_attempt": attempt, "e0_hourly_output": str(hourly_path), "e0_progress_log": str(progress_path),
        "preserved_prior_attempt_failures": prior_failures,
        "production_api": "crypto_quant.research.factor_mining.evaluation.evaluate_rank_displacement",
    }
    write_stats = summary["write_and_storage"]
    serialize_start = time.perf_counter()
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    summary_serialize_seconds = time.perf_counter() - serialize_start
    summary_path = OUT / ("e0_results.json" if attempt == 1 else f"e0{suffix}_results.json")
    summary_write_start = time.perf_counter()
    summary_path.write_text(summary_text, encoding="utf-8", newline="")
    summary_write_seconds = time.perf_counter() - summary_write_start
    projected = (write_stats["e0_hourly_bytes"] * write_stats["e1_full_pool_hourly_rows"]
                 / max(write_stats["e0_hourly_rows"], 1))
    finalization = {
        "summary_path": str(summary_path), "summary_serialize_seconds": summary_serialize_seconds,
        "summary_write_seconds": summary_write_seconds,
        "summary_bytes": len(summary_text.encode("utf-8")),
        "e0_hourly_rows": write_stats["e0_hourly_rows"],
        "e1_full_pool_hourly_rows": write_stats["e1_full_pool_hourly_rows"],
        "e1_estimated_hourly_compressed_bytes": projected,
        "archive_copy_bytes": 0,
        "estimated_e1_archive_read_decompress_parse_seconds": read_parse_estimate,
        "estimated_e1_d_api_seconds": compute_estimate,
    }
    (OUT / ("e0_finalization.json" if attempt == 1 else f"e0{suffix}_finalization.json")).write_text(
        json.dumps(finalization, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "amend-source-audit",
                                             "amend-horizon-availability-policy", "e0", "e2-front", "e1", "e2", "e3"))
    parser.add_argument("--output", type=Path,
                        help="new or existing output directory; default is this run's frozen experiment directory")
    args = parser.parse_args()
    global OUT
    if args.output is not None:
        OUT = args.output.expanduser().resolve()
    if args.action == "freeze":
        contract = preregister()
        print(json.dumps({"preregistration": str(OUT / "preregistration.json"),
                          "sha256": contract["preregistration_sha256"],
                          "a_references": contract["source_inventory"]["a_reference_count"],
                          "universe_hashes": contract["source_inventory"]["a_universe_sha256_count"],
                          "class_counts": contract["formula_classification"]["class_counts"],
                          "e0": [{"order": item["sample_order"], "factor_id": item["factor_id"],
                                  "category": item["formula_class"], "expression": item["identity"]["expanded_expression"],
                                  "direction": item["identity"]["direction"]}
                                 for item in contract["e0"]["fixed_samples"]],
                          "e2": [{"parent_order": item["parent_order"], "factor_id": item["factor_id"],
                                  "category": item["formula_class"], "expression": item["identity"]["expanded_expression"],
                                  "direction": item["identity"]["direction"]}
                                 for item in contract["e2"]["parents"]]}, ensure_ascii=False, indent=2))
    elif args.action == "amend-source-audit":
        contract = amend_source_inventory()
        print(json.dumps({"preregistration": str(OUT / "preregistration.json"),
                          "sha256": contract["preregistration_sha256"],
                          "source_path_correction": contract["source_path_correction"],
                          "native_a_ic_horizons": contract["e1"].get("native_a_ic_horizon_inventory")},
                         ensure_ascii=False, indent=2))
    elif args.action == "amend-horizon-availability-policy":
        contract = amend_horizon_availability_policy()
        print(json.dumps({"preregistration": str(OUT / "preregistration.json"),
                          "sha256": contract["preregistration_sha256"],
                          "e1_availability_policy": contract["e1"]["payload_horizon_comparison_availability"],
                          "state": "amended before E0 numerical computation"},
                         ensure_ascii=False, indent=2))
    elif args.action == "e0":
        result = run_e0()
        print(json.dumps({"status": result["status"], "samples_completed": result["samples_completed"],
                          "elapsed_seconds": result["elapsed_seconds"],
                          "full_pool_estimate": result["full_pool_estimate"],
                          "peak_rss_bytes": result["peak_rss_bytes"]}, ensure_ascii=False, indent=2))
    elif args.action == "e2-front":
        result = run_e2_front_selection()
        print(json.dumps({"status": result["status"], "parents": len(result["parents"]),
                          "elapsed_seconds": result["elapsed_seconds"],
                          "selected_version_counts": result["selected_version_counts"],
                          "selection_sha256": result["selection_sha256"]}, ensure_ascii=False, indent=2))
    elif args.action == "e1":
        result = run_e1_full()
        print(json.dumps({"status": result["status"], "population_count": result["population_count"],
                          "horizon_evidence_counts": result["horizon_evidence_counts"],
                          "timing_seconds": result["timing_seconds"],
                          "storage": result["storage"]}, ensure_ascii=False, indent=2))
    elif args.action == "e2":
        result = run_e2_full()
        print(json.dumps({"status": result["status"], "parent_count": result["parent_count"],
                          "eligible_for_smoothing_parent_count": result["eligible_for_smoothing_parent_count"],
                          "timing_seconds": result["timing_seconds"],
                          "storage": result["storage"]}, ensure_ascii=False, indent=2))
    elif args.action == "e3":
        result = run_e3_transfer()
        print(json.dumps({"status": result["status"],
                          "selected_smooth_parent_count": result["selected_smooth_parent_count"],
                          "selected_smooth_window_counts": result["selected_smooth_window_counts"],
                          "selected_original_with_smoothing_available_count": result[
                              "selected_original_with_smoothing_available_count"],
                          "no_smooth_variant_applicable_parent_count": result[
                              "no_smooth_variant_applicable_parent_count"],
                          "all_parent_denominator": result["all_parent_denominator"],
                          "effects_csv": result["outputs"]["effects_csv"]},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
