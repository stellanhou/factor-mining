#!/usr/bin/env python3
"""Run a fixed-sample FM-v6 24h versus multi-horizon history comparison.

The prepare phase reads only the original Goal's A evidence and makes one
A-only model review of the 26 frozen A-retained factors. Each run-group phase
reuses those saved choices and original formula/direction records, then reads
the shared market database only for B validation. Results go to isolated Goal
directories and idea pools under the worktree.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_GOAL = Path(
    "/Users/stellan/量化投资/experiments/factor_mining/goals/fm-v6-mimo26-pro-20260929"
)
DEFAULT_BASELINE_ENGINE = Path(
    "/var/folders/xc/s5f5mnbn5djgl52h4qhn1hjc0000gn/T/"
    "fm-multihorizon-baseline-jcvc3eua/src"
)
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments/factor_mining/horizon_comparison_20261001"
HORIZONS = (1, 4, 24)
EXPECTED_BATCHES = 7
EXPECTED_FACTORS = 26
EXPECTED_OLD_CARDS = 5


class ComparisonError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot read JSON evidence {path}: {exc}") from exc


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"evidence is not finite JSON: {exc}") from exc


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         indent=2, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise ComparisonError(f"refusing to overwrite existing evidence: {path}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_signature(path: Path, *, hash_contents: bool = False) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    stat = path.stat()
    require(path.is_file(), f"input must be a regular file: {path}")
    signature: dict[str, Any] = {
        "path": str(path), "device": stat.st_dev, "inode": stat.st_ino,
        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }
    if hash_contents:
        signature["sha256"] = sha256_file(path)
    return signature


def verify_signature(saved: dict[str, Any], label: str) -> None:
    path = Path(saved["path"])
    current = file_signature(path, hash_contents="sha256" in saved)
    require(current == saved, f"read-only {label} changed since prepare: {path}")


def factor_key(identity: dict[str, Any]) -> str:
    required = {"expanded_expression", "direction", "semantics_version"}
    require(set(identity) == required, "factor identity must contain formula, direction, and semantics version")
    key = canonical_json(identity)
    return hashlib.sha256(key).hexdigest()[:20]


def stat_archive_identity(path: Path) -> dict[str, Any]:
    return file_signature(path)


def load_archive_identity(path: Path) -> dict[str, Any]:
    uri = f"file:{path.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as db:
            db.execute("PRAGMA query_only = ON")
            row = db.execute(
                "SELECT identity_json FROM archive_meta WHERE singleton=1"
            ).fetchone()
    except sqlite3.Error as exc:
        raise ComparisonError(f"cannot inspect read-only factor archive {path}: {exc}") from exc
    require(row is not None, f"factor archive identity is missing: {path}")
    try:
        identity = json.loads(row[0])
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"invalid identity JSON in {path}") from exc
    return identity


def load_archive_payload(archive_path: Path, evaluation_id: str,
                         expected_segment: str = "A") -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one named evaluation row in read-only mode."""
    uri = f"file:{archive_path.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as db:
            db.execute("PRAGMA query_only = ON")
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT evaluation_id,data_version,contract_version,evaluator_version,segment,horizon,"
                "payload_blob_id,value_set_id,factor_value_count FROM evaluations WHERE evaluation_id=?",
                (int(evaluation_id),),
            ).fetchall()
            require(len(rows) == 1, f"A archive evaluation {evaluation_id} is missing: {archive_path}")
            row = rows[0]
            require(row["segment"] == expected_segment,
                    f"archive lookup resolved to {row['segment']}, expected {expected_segment}")
            blob = db.execute(
                "SELECT data_zlib FROM blobs WHERE blob_id=?", (row["payload_blob_id"],)
            ).fetchone()
            require(blob is not None, f"A archive payload blob is missing: {archive_path}")
    except sqlite3.Error as exc:
        raise ComparisonError(f"cannot read A archive payload {archive_path}: {exc}") from exc
    try:
        payload = json.loads(zlib.decompress(blob[0]).decode("utf-8"))
    except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"invalid A archive payload in {archive_path}: {exc}") from exc
    meta = {key: row[key] for key in (
        "evaluation_id", "data_version", "contract_version", "evaluator_version",
        "segment", "horizon", "value_set_id", "factor_value_count",
    )}
    return payload, meta


def read_fixed_a_factor_values_csv(values_csv: bytes, expected_rows: int):
    """Reconstruct archived binary64 values exactly from their CSV round-trip text."""
    import pandas as pd

    value_rows = pd.read_csv(io.BytesIO(values_csv), float_precision="round_trip")
    require(list(value_rows.columns) == ["timestamp", "symbol", "factor_value"]
            and len(value_rows) == expected_rows,
            "source A factor values CSV shape or row count differs")
    value_index = pd.MultiIndex.from_arrays([
        pd.to_datetime(value_rows["timestamp"], utc=True, errors="raise"),
        value_rows["symbol"].astype(str),
    ], names=["timestamp", "symbol"])
    require(value_index.is_unique, "source A factor values contain duplicate rows")
    return pd.Series(pd.to_numeric(value_rows["factor_value"], errors="raise").to_numpy(dtype=float),
                     index=value_index, name="factor_value")


def imported_a_rank_displacement(miner: Any, fixed: dict[str, Any], source_meta: dict[str, Any],
                                 source_run_dir: Path, source_calculation: dict[str, Any]
                                 ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Recompute D from the fixed A value set only when the target engine requires it."""
    version = getattr(miner, "rank_displacement_version", None)
    if version is None:
        return None, None

    from crypto_quant.features.factor_inputs import validate_universe
    from crypto_quant.research.factor_mining.evaluation import (
        RANK_DISPLACEMENT_VERSION, evaluate_rank_displacement,
    )
    from crypto_quant.research.factor_mining.factor_archive import (
        EvaluationKey, FactorArchive, FactorIdentity,
    )
    import pandas as pd

    require(version == RANK_DISPLACEMENT_VERSION,
            f"target engine declares unsupported rank-displacement version: {version}")
    require(source_run_dir.name == fixed["source_run_id"],
            f"source A run differs for {fixed['factor_key']}")
    archive_path = Path(fixed["source_archive"])
    identity = FactorIdentity(**fixed["identity"])
    target_identity = miner._factor_identity(identity.expanded_expression, identity.direction)
    require(target_identity.as_dict() == identity.as_dict(),
            f"target A factor identity differs for {fixed['factor_key']}")
    require(load_archive_identity(archive_path) == identity.as_dict(),
            f"source A factor archive identity differs for {fixed['factor_key']}")
    source_locator = fixed["a_evaluation_archive"]
    source_id = str(fixed["source_evaluation_id"])
    require(source_locator["identity"] == identity.as_dict()
            and str(source_locator["evaluation_id"]) == source_id
            and str(source_meta["evaluation_id"]) == source_id
            and source_meta["segment"] == "A"
            and source_meta["data_version"] == f"{fixed['source_run_id']}/A"
            and source_meta["horizon"] == "24h",
            f"source A evaluation identity differs for {fixed['factor_key']}")
    source_key = {field: source_meta[field] for field in (
        "data_version", "contract_version", "evaluator_version", "segment", "horizon")}
    require(source_locator["evaluation_key"] == source_key
            and str(source_meta["value_set_id"]) == str(source_locator["value_set_id"]),
            f"source A evaluation key or value set differs for {fixed['factor_key']}")

    values_artifact = source_calculation.get("values_artifact")
    require(isinstance(values_artifact, dict)
            and values_artifact.get("identity") == identity.as_dict()
            and values_artifact.get("data_version") == source_meta["data_version"]
            and values_artifact.get("computation_semantics") == source_meta["evaluator_version"]
            and source_meta["value_set_id"] is not None
            and str(values_artifact.get("value_set_id")) == str(source_meta["value_set_id"])
            and type(values_artifact.get("rows")) is int and values_artifact["rows"] > 0,
            f"source A calculation does not point to the evaluated value set for {fixed['factor_key']}")

    source_archive = FactorArchive.open_existing(archive_path.parent, identity)
    evaluation_key = EvaluationKey(**source_key)
    values_csv = source_archive.get_factor_values_csv(evaluation_key)
    try:
        factor_values = read_fixed_a_factor_values_csv(values_csv, values_artifact["rows"])
    except ComparisonError as exc:
        raise ComparisonError(f"source A factor values are invalid for {fixed['factor_key']}: {exc}") from exc

    universe_path = source_run_dir / "A-universe.csv"
    universe_csv = universe_path.read_bytes()
    universe_rows = pd.read_csv(io.BytesIO(universe_csv))
    require(list(universe_rows.columns) == ["timestamp", "symbol", "eligible"]
            and pd.api.types.is_bool_dtype(universe_rows["eligible"])
            and not universe_rows["eligible"].isna().any(),
            f"source A universe shape or membership differs for {fixed['factor_key']}")
    universe_index = pd.MultiIndex.from_arrays([
        pd.to_datetime(universe_rows["timestamp"], utc=True, errors="raise"),
        universe_rows["symbol"].astype(str),
    ], names=["timestamp", "symbol"])
    universe = validate_universe(pd.Series(universe_rows["eligible"].to_numpy(dtype=bool),
                                            index=universe_index, name="eligible"))
    require(factor_values.index.equals(universe.index),
            f"source A factor values and universe indices differ for {fixed['factor_key']}")
    report = evaluate_rank_displacement(factor_values, universe, miner.spec, "A")
    require(report["definition_version"] == version and report["segment"] == "A",
            f"computed A rank-displacement evidence differs for {fixed['factor_key']}")
    provenance = {
        "source_run_id": fixed["source_run_id"],
        "source_candidate_id": fixed["candidate_id"],
        "source_evaluation_id": source_id,
        "source_evaluation_key": source_key,
        "source_value_set_id": str(source_meta["value_set_id"]),
        "source_archive": str(archive_path),
        "source_values_csv_sha256": hashlib.sha256(values_csv).hexdigest(),
        "source_a_universe_sha256": hashlib.sha256(universe_csv).hexdigest(),
    }
    return report, provenance


def validate_a_payload(payload: dict[str, Any], identity: dict[str, Any], label: str) -> None:
    require(payload.get("segment") == "A" and payload.get("horizon_hours") == 24,
            f"{label} is not a complete primary 24h A report")
    require(str(payload.get("direction")) == str(identity["direction"]),
            f"{label} direction differs from its frozen factor identity")
    comparison = payload.get("horizon_comparison")
    require(isinstance(comparison, dict) and isinstance(comparison.get("horizons"), dict),
            f"{label} lacks archived A horizon_comparison")
    reports = comparison["horizons"]
    require(set(reports) == {str(horizon) for horizon in HORIZONS},
            f"{label} must contain complete 1h, 4h, and 24h A evidence")
    sample_keys: set[tuple[tuple[Any, Any, Any], ...]] = set()
    stage_bounds: set[tuple[tuple[Any, Any], ...]] = set()
    for horizon in HORIZONS:
        report = reports[str(horizon)]
        require(report.get("segment") == "A" and report.get("horizon_hours") == horizon,
                f"{label} has an invalid A report for {horizon}h")
        require(str(report.get("direction")) == str(identity["direction"]),
                f"{label} changes direction at {horizon}h")
        for field in ("summary", "coverage", "grouping", "per_symbol", "stages", "periods"):
            require(field in report, f"{label} {horizon}h A report is missing {field}")
        require(len(report["stages"]) > 0 and len(report["periods"]) > 0,
                f"{label} {horizon}h A report has no stage or period evidence")
        sample_keys.add(tuple((row["timestamp"], row.get("eligible"), row.get("n"))
                              for row in report["periods"]))
        stage_bounds.add(tuple((row["start"], row["end"]) for row in report["stages"]))
    require(len(sample_keys) == 1 and len(stage_bounds) == 1,
            f"{label} A horizons do not share the archived common sample and stage boundaries")


def discover_factor_archives(archive_root: Path, identities: dict[str, dict[str, Any]]) -> dict[str, Path]:
    wanted = {canonical_json(identity): key for key, identity in identities.items()}
    found: dict[str, Path] = {}
    for path in sorted(archive_root.glob("factor-*.sqlite3")):
        identity = load_archive_identity(path)
        key = wanted.get(canonical_json(identity))
        if key is None:
            continue
        require(key not in found, f"more than one archive stores frozen factor {key}")
        found[key] = path.resolve()
    missing = sorted(set(identities) - set(found))
    require(not missing, f"original A factor archive is missing for identities: {missing}")
    return found


def source_a_record(run_dir: Path, candidate_id: str, suffix: str) -> dict[str, Any]:
    path = run_dir / "a_records" / f"{candidate_id}-{suffix}.json"
    record = read_json(path)
    require(record.get("id") == f"{candidate_id}-{suffix}", f"A source record ID differs: {path}")
    return record


def fixed_scope(source_goal: Path) -> dict[str, Any]:
    source_goal = source_goal.expanduser().resolve(strict=True)
    goal_data = read_json(source_goal / "goal.json")
    require(goal_data.get("goal", {}).get("goal_id") == source_goal.name,
            "source Goal directory and saved Goal identity differ")
    input_data = goal_data["inputs"]
    db_signature = file_signature(Path(input_data["db"]))
    universe_signature = file_signature(Path(input_data["universe"]), hash_contents=True)

    run_dirs = sorted(
        (path for path in (source_goal / "runs").glob("goal-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    require(len(run_dirs) == EXPECTED_BATCHES,
            f"source Goal must contain exactly {EXPECTED_BATCHES} fixed cycles, found {len(run_dirs)}")

    batches: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    identities: dict[str, dict[str, Any]] = {}
    by_source_candidate: dict[tuple[str, str], str] = {}
    for batch_index, run_dir in enumerate(run_dirs, start=1):
        a_complete_path = run_dir / "a-complete.json"
        require(a_complete_path.is_file(), f"source A completion checkpoint is missing: {a_complete_path}")
        completed = read_json(a_complete_path)
        candidate_decisions = completed.get("candidate_decisions", {})
        retained_ids = completed.get("retained_ids", [])
        require(len(retained_ids) == len(set(retained_ids)), f"duplicate A retain ID in {run_dir.name}")
        actual_retain_ids = sorted(
            candidate_id for candidate_id, decision in candidate_decisions.items()
            if decision.get("disposition") == "retain"
        )
        require(sorted(retained_ids) == actual_retain_ids,
                f"A retained_ids differs from decisions in {run_dir.name}")
        frozen_path = run_dir / "frozen_batch.json"
        if retained_ids:
            require(frozen_path.is_file(), f"retained cycle lacks its frozen A/B handoff: {frozen_path}")
            frozen = read_json(frozen_path)
            frozen_candidates = frozen.get("candidates", {})
            require(set(frozen_candidates) == set(retained_ids),
                    f"frozen membership differs from A retains in {run_dir.name}")
        else:
            require(not frozen_path.exists(), f"empty A-retain cycle unexpectedly has a frozen batch: {frozen_path}")
            frozen_candidates = {}

        batch_candidates: list[dict[str, Any]] = []
        for candidate_id in sorted(retained_ids):
            item = frozen_candidates[candidate_id]
            decision = candidate_decisions[candidate_id]
            identity = item["a_evaluation_archive"]["identity"]
            require(identity["expanded_expression"] == item["executed"]["expanded_expression"],
                    f"A archive formula differs from frozen calculation for {run_dir.name}/{candidate_id}")
            require(str(identity["direction"]) == str(item["definition"]["direction"]),
                    f"A archive direction differs from frozen definition for {run_dir.name}/{candidate_id}")
            key = factor_key(identity)
            require(key not in identities, f"duplicate expanded formula/direction/semantics in fixed 26: {key}")
            identities[key] = identity
            by_source_candidate[(run_dir.name, candidate_id)] = key

            definition_record = source_a_record(run_dir, candidate_id, "definition")
            calculation_record = source_a_record(run_dir, candidate_id, "calculation")
            evaluation_record = source_a_record(run_dir, candidate_id, "evaluation")
            report_record = source_a_record(run_dir, candidate_id, "report")
            require(definition_record["kind"] == "candidate"
                    and definition_record["data"]["id"] == candidate_id
                    and definition_record["data"]["definition"] == item["definition"],
                    f"A candidate definition differs from frozen handoff in {run_dir.name}/{candidate_id}")
            require(calculation_record["kind"] == "calculation"
                    and calculation_record["data"]["executed_expression"] == item["executed"],
                    f"A calculation differs from frozen handoff in {run_dir.name}/{candidate_id}")
            require(evaluation_record["kind"] == "evaluation"
                    and evaluation_record["data"]["factor_archive"] == item["a_evaluation_archive"],
                    f"A evaluation locator differs from frozen handoff in {run_dir.name}/{candidate_id}")
            require(report_record["kind"] == "model_report"
                    and report_record["data"] == item["a_model_report"],
                    f"A model report differs from frozen handoff in {run_dir.name}/{candidate_id}")

            candidate = {
                "factor_key": key,
                "candidate_id": candidate_id,
                "source_run_id": run_dir.name,
                "batch_index": batch_index,
                "definition": item["definition"],
                "executed": item["executed"],
                "identity": identity,
                "source_decision": decision,
                "a_evaluation_ref": item["a_evaluation_ref"],
                "a_evaluation_archive": item["a_evaluation_archive"],
                "a_model_report": item["a_model_report"],
                "source_record_ids": {
                    "candidate": definition_record["id"],
                    "calculation": calculation_record["id"],
                    "evaluation": evaluation_record["id"],
                    "model_report": report_record["id"],
                },
            }
            batch_candidates.append(candidate)
            candidates.append(candidate)
        batches.append({
            "batch_index": batch_index,
            "source_run_id": run_dir.name,
            "candidate_count": len(batch_candidates),
            "factor_keys": [candidate["factor_key"] for candidate in batch_candidates],
            "candidates": batch_candidates,
        })

    require(len(candidates) == EXPECTED_FACTORS,
            f"source A retain count must be {EXPECTED_FACTORS}, found {len(candidates)}")
    archive_paths = discover_factor_archives(
        source_goal / "runs" / "factor_archive_v2", identities
    )
    for candidate in candidates:
        locator = candidate["a_evaluation_archive"]
        archive_path = archive_paths[candidate["factor_key"]]
        require(locator["identity"] == candidate["identity"], "A evaluation locator identity changed")
        payload, row = load_archive_payload(archive_path, locator["evaluation_id"])
        key = locator["evaluation_key"]
        expected_key = {
            "data_version": row["data_version"],
            "contract_version": row["contract_version"],
            "evaluator_version": row["evaluator_version"],
            "segment": row["segment"],
            "horizon": row["horizon"],
        }
        require(row["segment"] == "A" and row["horizon"] == "24h"
                and key == expected_key,
                f"A archive locator is not the saved 24h A row for {candidate['factor_key']}")
        require(str(row["value_set_id"]) == str(locator.get("value_set_id"))
                and type(locator["factor_value_count"]) is int
                and locator["factor_value_count"] > 0,
                f"A archive locator does not match its saved value-set metadata for {candidate['factor_key']}")
        validate_a_payload(payload, candidate["identity"], candidate["factor_key"])
        candidate["source_archive"] = str(archive_path)
        candidate["source_archive_signature"] = stat_archive_identity(archive_path)
        candidate["source_evaluation_id"] = str(locator["evaluation_id"])
        candidate["a_report_shapes"] = {
            "horizons": {
                horizon: {
                    "periods": len(payload["horizon_comparison"]["horizons"][horizon]["periods"]),
                    "stages": len(payload["horizon_comparison"]["horizons"][horizon]["stages"]),
                }
                for horizon in sorted(payload["horizon_comparison"]["horizons"])
            }
        }

    original_cards = validate_original_cards(source_goal, goal_data, candidates)
    require(len(original_cards) == EXPECTED_OLD_CARDS,
            f"read-only original Goal must contain {EXPECTED_OLD_CARDS} qualified cards")
    return {
        "schema": "fm-v6-fixed-horizon-comparison/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_goal": str(source_goal),
        "source_goal_id": goal_data["goal"]["goal_id"],
        "source_goal_data": goal_data,
        "input_signatures": {"db": db_signature, "universe": universe_signature},
        "research_spec": goal_data["research"],
        "batch_count": len(batches),
        "candidate_count": len(candidates),
        "original_card_count": len(original_cards),
        "original_cards": original_cards,
        "batches": batches,
        "candidates": candidates,
    }


def validate_original_cards(source_goal: Path, goal_data: dict[str, Any],
                            candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    receipts = source_goal / "completion_records"
    admissions: dict[str, dict[str, Any]] = {}
    matches: dict[str, dict[str, Any]] = {}
    for path in sorted(receipts.glob("admission-*.json")):
        record = read_json(path)
        require(record.get("kind") == "program_admission", f"unexpected source admission receipt: {path}")
        for idea in record.get("data", {}).get("ideas", []):
            require(idea.get("admitted_by_program") is True, f"source admission lacks program approval: {path}")
            require(idea["idea_id"] not in admissions, f"duplicate source card admission: {idea['idea_id']}")
            admissions[idea["idea_id"]] = {"idea": idea, "receipt_path": str(path.resolve())}
    for path in sorted(receipts.glob("match-*.json")):
        record = read_json(path)
        require(record.get("kind") == "goal_match", f"unexpected source match receipt: {path}")
        for match in record.get("data", {}).get("matches", []):
            require(match["idea_id"] not in matches, f"duplicate source card match: {match['idea_id']}")
            matches[match["idea_id"]] = match

    pool = Path(goal_data["inputs"]["idea_pool"]).expanduser().resolve(strict=True)
    candidate_lookup = {
        (candidate["source_run_id"], candidate["candidate_id"]): candidate
        for candidate in candidates
    }
    require(set(admissions) == set(matches), "original program admissions and Goal matches differ")
    ideas: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idea_id, admission_record in sorted(admissions.items()):
        admission = admission_record["idea"]
        match = matches[idea_id]
        require(match.get("matches_goal") is True, f"source card does not match its Goal: {idea_id}")
        key = (admission["run_id"], admission["candidate_id"])
        candidate = candidate_lookup.get(key)
        require(candidate is not None, f"source card is outside the fixed 26-factor cohort: {idea_id}")
        identity_key = candidate["factor_key"]
        require(identity_key not in seen, f"source Goal cards duplicate a factor identity: {idea_id}")
        seen.add(identity_key)
        card_path = pool / f"{idea_id}.json"
        card = read_json(card_path)
        require(card.get("id") == idea_id and card.get("source", {}).get("run_id") == key[0]
                and card.get("source", {}).get("candidate_id") == key[1]
                and card.get("b_validation_status") == "passed",
                f"source Goal receipt lacks its actual matching card: {card_path}")
        ideas.append({
            "idea_id": idea_id,
            "candidate_id": key[1],
            "source_run_id": key[0],
            "factor_key": identity_key,
            "card_path": str(card_path),
            "admission_receipt": admission_record["receipt_path"],
            "match": True,
        })
    require(len(ideas) == EXPECTED_OLD_CARDS,
            f"source Goal has {len(ideas)} actual qualified cards, expected {EXPECTED_OLD_CARDS}")
    return ideas


class ArchivePagedReviewStore:
    """AgentGateway store that keeps A period tables paged from the source SQLite."""

    def __init__(self, store: Any, readers: dict[str, tuple[Path, str]]):
        self._store = store
        self._readers = readers
        self.root = store.root

    def all(self) -> list[dict[str, Any]]:
        return self._store.all()

    def append(self, record_id: str, kind: str, data: Any) -> str:
        return self._store.append(record_id, kind, data)

    def read(self, record_id: str, pointer: str, offset: int, limit: int) -> Any:
        require(limit <= 500, "A evidence pages are limited to 500 rows")
        prefix = "/a_evidence/horizon_comparison/horizons/"
        if record_id in self._readers and pointer.startswith(prefix) and pointer.endswith("/periods"):
            archive_path, evaluation_id = self._readers[record_id]
            payload, _ = load_archive_payload(archive_path, evaluation_id)
            source_pointer = pointer[len("/a_evidence"):]
            from crypto_quant.research.factor_mining.records import read_pointer
            return read_pointer(payload, source_pointer, offset, limit)
        return self._store.read(record_id, pointer, offset, limit)


def a_review_record(candidate: dict[str, Any], payload: dict[str, Any], record_id: str) -> dict[str, Any]:
    reports = payload["horizon_comparison"]["horizons"]
    horizons = {}
    for horizon in ("1", "4", "24"):
        report = reports[horizon]
        horizons[horizon] = {
            "segment": report["segment"],
            "horizon_hours": report["horizon_hours"],
            "direction": report["direction"],
            "summary": report["summary"],
            "coverage": report["coverage"],
            "grouping": report["grouping"],
            "per_symbol": report["per_symbol"],
            "stages": report["stages"],
            "periods": {
                "rows": len(report["periods"]),
                "read_records_pointer": f"/a_evidence/horizon_comparison/horizons/{horizon}/periods",
                "read_records_instructions": "Request original rows through AgentGateway read_research_evidence pages.",
            },
        }
    decision = candidate["source_decision"]
    research_basis = decision.get("answers", {}).get("research_basis", {}).get("reason", "")
    return {
        "batch_index": candidate["batch_index"],
        "source_run_id": candidate["source_run_id"],
        "candidate_id": candidate["candidate_id"],
        "factor_key": candidate["factor_key"],
        "definition": candidate["definition"],
        "expanded_expression": candidate["identity"]["expanded_expression"],
        "direction": candidate["identity"]["direction"],
        "semantics_version": candidate["identity"]["semantics_version"],
        "original_retain_reason": decision["reason"],
        "original_research_basis": research_basis,
        "a_model_report": candidate["a_model_report"],
        "a_evidence": {
            "main_24h": {
                "summary": payload["summary"],
                "coverage": payload["coverage"],
                "stages": payload["stages"],
            },
            "horizon_comparison_interpretation": payload["horizon_comparison"].get("interpretation"),
            "horizon_comparison": {"horizons": horizons},
            "period_tables_record_id": record_id,
        },
    }


def a_review_context_measure(records: list[dict[str, Any]], spec: Any,
                             payload: dict[str, Any], schema: dict[str, Any], task: str,
                             engine_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(engine_root))
    from crypto_quant.research.factor_mining.contracts import dumps
    from crypto_quant.research.factor_mining.records import SYSTEM, compact_record

    content = {
        "role": "optimizer", "task": task, "contract": spec.as_dict(),
        "payload": payload, "output_schema": schema,
        "records": [record if record["kind"] == "goal_context" else compact_record(record)
                    for record in records],
    }
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": dumps(content)}]

    def bound(values: list[dict[str, str]]) -> int:
        return sum(len(message["content"].encode("utf-8")) + 32 for message in values) + 128

    before = bound(messages)
    budget = spec.context_tokens - (spec.output_tokens or 0)
    mode = "numeric_tables_paged"
    if before > budget:
        content["records"] = [compact_record(record, page_cycle_candidates=True) for record in records]
        messages[1]["content"] = dumps(content)
        mode = "cycle_candidates_paged"
    after = bound(messages)
    require(after <= budget,
            f"A-only review still exceeds the declared context after paging: {after} > {budget}")
    return {"context_mode": mode, "input_bound_before_paging": before,
            "input_bound_after_paging": after, "input_budget": budget}


def choose_model(scope: dict[str, Any], args: argparse.Namespace,
                 engine_root: Path) -> tuple[Any, dict[str, Any]]:
    sys.path.insert(0, str(engine_root))
    from crypto_quant.research.factor_mining.model_config import model_from_settings
    from crypto_quant.research.factor_mining.mimo_model import MimoModel, PROVIDER as MIMO_PROVIDER
    from crypto_quant.research.factor_mining.codex_model import CodexModel, PROVIDER as CODEX_PROVIDER
    from crypto_quant.research.factor_mining.contracts import require as research_require

    saved = scope["source_goal_data"]["model_settings"]
    explicit = any(getattr(args, name, None) is not None for name in (
        "provider", "model", "reasoning_effort", "thinking", "timeout_seconds", "codex_bin"
    ))
    if not explicit:
        model = model_from_settings(saved)
        return model, model.settings()

    provider_arg = args.provider
    research_require(provider_arg in {"mimo", "codex"},
                     "an explicit model override requires --provider mimo or --provider codex")
    if provider_arg == "mimo":
        research_require(args.codex_bin is None,
                         "--codex-bin is only supported by the Codex provider")
        research_require(args.reasoning_effort is None,
                         "MiMo uses --thinking, not --reasoning-effort")
        saved_mimo = saved.get("provider") == MIMO_PROVIDER
        model_name = args.model or (saved.get("model") if saved_mimo else None)
        research_require(bool(model_name),
                         "--model is required when the saved Goal does not use MiMo")
        thinking = args.thinking or (saved.get("thinking") if saved_mimo else None)
        research_require(thinking in {"enabled", "disabled"},
                         "--thinking is required when selecting MiMo without saved MiMo settings")
        timeout = args.timeout_seconds or saved.get("timeout_seconds")
        research_require(type(timeout) is int and timeout > 0,
                         "--timeout-seconds is required when no saved timeout exists")
        model = MimoModel(model_name, timeout_seconds=timeout, thinking=thinking)
        expected_provider = MIMO_PROVIDER
    else:
        research_require(args.thinking is None, "--thinking is only supported by MiMo")
        research_require(bool(args.model) and bool(args.reasoning_effort),
                         "Codex selection requires explicit --model and --reasoning-effort")
        research_require(args.codex_bin is not None,
                         "Codex selection requires an explicit --codex-bin")
        codex_bin = args.codex_bin.expanduser().resolve(strict=True)
        timeout = args.timeout_seconds or saved.get("timeout_seconds") or 300
        model = CodexModel(args.model, reasoning_effort=args.reasoning_effort,
                           timeout_seconds=timeout, codex_bin=codex_bin)
        expected_provider = CODEX_PROVIDER
    settings = model.settings()
    research_require(settings["provider"] == expected_provider,
                     "selected model adapter returned an unexpected provider")
    return model, settings


def run_a_review(scope: dict[str, Any], output_root: Path,
                 current_engine_root: Path, model_args: argparse.Namespace) -> dict[str, Any]:
    current_engine_root = current_engine_root.expanduser().resolve(strict=True)
    require((current_engine_root / "crypto_quant/research/factor_mining").is_dir(),
            f"current engine root must contain crypto_quant: {current_engine_root}")
    sys.path.insert(0, str(current_engine_root))
    from crypto_quant.research.factor_mining.contracts import (
        ResearchSpec, _array_schema, _object_schema, _text_schema, require as research_require,
    )
    from crypto_quant.research.factor_mining.records import AgentGateway, RecordStore, write_json

    a_review_root = output_root / "a_review"
    a_records_root = a_review_root / "a_records"
    a_records_root.mkdir(parents=True, exist_ok=False)
    base_spec = dict(scope["research_spec"])
    base_spec["run_id"] = "fm-v6-horizon-a-review-20261001"
    base_spec["b_horizons"] = [1, 4, 24]
    review_spec = ResearchSpec.from_dict(base_spec)
    store = RecordStore(a_records_root)
    readers: dict[str, tuple[Path, str]] = {}
    review_payload = []
    for candidate in scope["candidates"]:
        archive_path = Path(candidate["source_archive"])
        verify_signature(candidate["source_archive_signature"], "source A factor archive")
        payload, _ = load_archive_payload(archive_path, candidate["source_evaluation_id"])
        validate_a_payload(payload, candidate["identity"], candidate["factor_key"])
        record_id = f"a-horizon-{candidate['factor_key']}"
        readers[record_id] = (archive_path, candidate["source_evaluation_id"])
        data = a_review_record(candidate, payload, record_id)
        store.append(record_id, "fixed_A_horizon_evidence", data)
        review_payload.append({
            "factor_key": candidate["factor_key"],
            "batch_index": candidate["batch_index"],
            "source_run_id": candidate["source_run_id"],
            "candidate_id": candidate["candidate_id"],
            "evidence_record_id": record_id,
            "expanded_expression": candidate["identity"]["expanded_expression"],
            "direction": candidate["identity"]["direction"],
            "semantics_version": candidate["identity"]["semantics_version"],
            "original_retain_reason": candidate["source_decision"]["reason"],
        })

    result_schema = _object_schema({
        "decisions": _array_schema(_object_schema({
            "factor_key": _text_schema("固定身份清单中的精确factor_key"),
            "candidate_id": _text_schema("原A保留因子的精确 candidate_id"),
            "retained_horizons": _array_schema(
                {"type": "integer", "enum": [1, 4, 24]}, minItems=1, maxItems=3,
            ),
            "reason": _text_schema("仅基于A证据决定是否追加1h/4h，说明观察与不确定性"),
        }), minItems=EXPECTED_FACTORS, maxItems=EXPECTED_FACTORS),
    })
    evidence_store = ArchivePagedReviewStore(store, readers)
    store = evidence_store
    model, model_settings = choose_model(scope, model_args, current_engine_root)
    gateway = AgentGateway(model, review_spec, store, a_review_root / "model_calls", stage="A")

    def validate_result(value: dict[str, Any]) -> None:
        research_require(len(value["decisions"]) == EXPECTED_FACTORS,
                         "A horizon review must cover all 26 fixed factors")
        seen: set[str] = set()
        for item in value["decisions"]:
            factor_identity_key = item["factor_key"]
            candidate_id = item["candidate_id"]
            horizons = item["retained_horizons"]
            research_require(factor_identity_key not in seen,
                             "duplicate A horizon decision")
            seen.add(factor_identity_key)
            research_require(factor_identity_key in {
                candidate["factor_key"] for candidate in scope["candidates"]
            }, "A horizon decision uses an unknown factor identity")
            research_require(type(candidate_id) is str and bool(candidate_id),
                             "A horizon decision needs a candidate ID")
            research_require(all(type(horizon) is int and horizon in HORIZONS for horizon in horizons),
                             "retained_horizons must be drawn from [1,4,24]")
            research_require(len(horizons) == len(set(horizons)) and horizons == sorted(horizons),
                             "retained_horizons must be ordered and contain no duplicates")
            research_require(24 in horizons, "original A retain decision must keep its 24h basis")
            research_require(isinstance(item["reason"], str) and bool(item["reason"].strip()),
                             "A horizon decision needs an A-only rationale")
        returned_ids = [item["candidate_id"] for item in value["decisions"]]
        returned_factor_keys = [item["factor_key"] for item in value["decisions"]]
        expected_ids = [candidate["candidate_id"] for candidate in scope["candidates"]]
        expected_factor_keys = [candidate["factor_key"] for candidate in scope["candidates"]]
        research_require(returned_ids == expected_ids and returned_factor_keys == expected_factor_keys,
                         "A review must return the 26 fixed identities in frozen batch order")

    task = (
        "本次只做一次A证据期限审阅。固定对象是payload中的26个已A-retain因子，按factor_key区分，公式、方向、"
        "语义版本、原批次和原24h retain理由全部固定。每个candidate_id按payload顺序各输出一项，"
        "retained_horizons必须保留24；只决定是否追加1h和/或4h。"
        "逐项依据其A段1h/4h/24h共同样本统计、完整A模型报告、分阶段结果和原retain依据。"
        "允许保留微弱但有重复迹象的期限，不要求统计显著，不提高IC门槛；不得改变公式、方向、"
        "24h原保留依据或任何规则，不生成优化任务。periods和stages表可能按AgentGateway分页；"
        "如需逐期核对，通过read_records请求小页。当前context和records只包含A证据。禁止推断、请求或引用任何B数值、"
        "B通过/失败状态或失败原因。每项reason只说明A期限追加决定及其不确定性。"
    )
    request_measure = a_review_context_measure(
        store.all(), review_spec,
        {"review_kind": "append_A_retained_horizons", "fixed_candidate_count": EXPECTED_FACTORS,
         "allowed_horizons": list(HORIZONS), "preserve_horizon": 24,
         "candidate_evidence_order": review_payload},
        result_schema, task, current_engine_root,
    )
    output = gateway.ask(
        "optimizer", task,
        {"review_kind": "append_A_retained_horizons", "fixed_candidate_count": EXPECTED_FACTORS,
         "allowed_horizons": list(HORIZONS), "preserve_horizon": 24,
         "candidate_evidence_order": review_payload},
        result_schema, validate=validate_result,
    )
    require(output["decisions"] and len(output["decisions"]) == EXPECTED_FACTORS,
            "A review output is incomplete")
    decisions = []
    for candidate, reviewed in zip(scope["candidates"], output["decisions"]):
        require(reviewed["factor_key"] == candidate["factor_key"]
                and reviewed["candidate_id"] == candidate["candidate_id"],
                "A review response order differs from the frozen factor manifest")
        decisions.append({
            "factor_key": candidate["factor_key"],
            "source_run_id": candidate["source_run_id"],
            "candidate_id": candidate["candidate_id"],
            "retained_horizons": reviewed["retained_horizons"],
            "reason": reviewed["reason"],
            "preserved_original_retain_reason": candidate["source_decision"]["reason"],
        })
    review = {
        "schema": "fm-v6-a-horizon-review/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_goal_id": scope["source_goal_id"],
        "fixed_scope_sha256": hashlib.sha256(canonical_json(scope)).hexdigest(),
        "model": {"settings": model_settings, "output_tokens": None},
        "candidate_count": EXPECTED_FACTORS,
        "allowed_horizons": list(HORIZONS),
        "context_measure": request_measure,
        "decisions": decisions,
    }
    write_new_json(a_review_root / "a_review.json", review)
    return review


def prepare(args: argparse.Namespace) -> None:
    source_goal = args.source_goal.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    current_engine_root = args.current_engine_root.expanduser().resolve(strict=True)
    require(output_root.is_relative_to(REPO_ROOT),
            "comparison outputs must be created under this worktree")
    require(not output_root.exists(),
            f"comparison directory already exists; refusing duplicate A review or B run: {output_root}")
    scope = fixed_scope(source_goal)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    write_new_json(output_root / "fixed_scope.json", scope)
    try:
        review = run_a_review(scope, output_root, current_engine_root, args)
    except Exception as exc:
        write_new_json(output_root / "a_review" / "a_review_stopped.json", {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__, "reason": str(exc),
            "action": "Inspect saved A-only evidence and model transcripts; do not repeat the review automatically.",
        })
        raise
    print(json.dumps({
        "status": "prepared",
        "experiment_root": str(output_root),
        "source_goal_read_only": str(source_goal),
        "batches": scope["batch_count"],
        "fixed_A_retains": scope["candidate_count"],
        "original_qualified_cards": scope["original_card_count"],
        "a_review_candidates": review["candidate_count"],
        "a_review_file": str(output_root / "a_review" / "a_review.json"),
        "context_measure": review["context_measure"],
        "next": [
            f"freeze-inputs --output-root {output_root}",
            f"run-group --group old24 --output-root {output_root} --engine-root {DEFAULT_BASELINE_ENGINE}",
            f"run-group --group new_multi --output-root {output_root} --engine-root {REPO_ROOT / 'src'}",
        ],
    }, ensure_ascii=False, indent=2))


def verify_scope_inputs(scope: dict[str, Any]) -> None:
    verify_signature(scope["input_signatures"]["db"], "source market database")
    verify_signature(scope["input_signatures"]["universe"], "source universe CSV")
    for candidate in scope["candidates"]:
        verify_signature(candidate["source_archive_signature"], "source A factor archive")


def verify_frozen_panel(experiment_root: Path) -> dict[str, Any]:
    manifest_path = experiment_root / "frozen_inputs" / "provenance.json"
    require(manifest_path.is_file(),
            "freeze-inputs must load the B panel once before either comparison group")
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "fm-v6-frozen-b-inputs/v1",
            "frozen B input manifest schema differs")
    for name in ("values", "universe"):
        verify_signature(manifest["artifacts"][name], f"frozen B {name} panel")
    return manifest


def load_prepared_scope(experiment_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    scope_path = experiment_root / "fixed_scope.json"
    review_path = experiment_root / "a_review" / "a_review.json"
    require(scope_path.is_file() and review_path.is_file(),
            "run-group requires a completed fixed-scope manifest and saved one-time A review")
    scope = read_json(scope_path)
    review = read_json(review_path)
    require(scope.get("schema") == "fm-v6-fixed-horizon-comparison/v1"
            and review.get("schema") == "fm-v6-a-horizon-review/v1",
            "comparison manifest schema differs")
    require(review["fixed_scope_sha256"] == hashlib.sha256(canonical_json(scope)).hexdigest(),
            "saved A review does not match the fixed-scope manifest")
    require(scope["candidate_count"] == EXPECTED_FACTORS and scope["batch_count"] == EXPECTED_BATCHES,
            "fixed comparison manifest is incomplete")
    verify_scope_inputs(scope)
    by_key = {item["factor_key"]: item for item in review["decisions"]}
    require(set(by_key) == {item["factor_key"] for item in scope["candidates"]},
            "saved A review does not cover every fixed factor exactly once")
    for candidate in scope["candidates"]:
        decision = by_key[candidate["factor_key"]]
        require(decision["retained_horizons"] == sorted(set(decision["retained_horizons"]))
                and 24 in decision["retained_horizons"]
                and set(decision["retained_horizons"]) <= set(HORIZONS),
                f"saved A horizon choice is invalid for {candidate['factor_key']}")
        require(decision["preserved_original_retain_reason"] == candidate["source_decision"]["reason"],
                f"original 24h retain basis changed for {candidate['factor_key']}")
    return scope, review


def run_group(args: argparse.Namespace) -> None:
    experiment_root = args.output_root.expanduser().resolve(strict=True)
    scope, review = load_prepared_scope(experiment_root)
    verify_frozen_panel(experiment_root)
    group = args.group
    engine_root = args.engine_root.expanduser().resolve(strict=True)
    require((engine_root / "crypto_quant/research/factor_mining").is_dir(),
            f"engine root must contain crypto_quant: {engine_root}")
    if group == "old24":
        require(engine_root.resolve() == args.baseline_engine_root.expanduser().resolve(strict=True),
                "old24 must use the saved pre-change baseline source")
    else:
        require(engine_root.resolve() == args.current_engine_root.expanduser().resolve(strict=True),
                "new_multi must use the current source")
    group_dir = experiment_root / group
    if args.resume:
        require(group_dir.is_dir() and not (group_dir / "fixed_group_result.json").exists(),
                f"--resume requires an incomplete saved group without a final result: {group_dir}")
    else:
        require(not group_dir.exists(), f"refusing to repeat B validation in existing group: {group_dir}")
        group_dir.mkdir()
    worker_args = [
        sys.executable, str(Path(__file__).resolve()), "_run_worker",
        "--group", group,
        "--experiment-root", str(experiment_root),
        "--engine-root", str(engine_root),
        "--group-dir", str(group_dir),
        "--frozen-panel-manifest", str(experiment_root / "frozen_inputs" / "provenance.json"),
    ]
    if args.resume:
        worker_args.append("--resume")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(engine_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(worker_args, cwd=Path.cwd(), env=env, check=True)


def recheck_completion(args: argparse.Namespace) -> None:
    experiment_root = args.output_root.expanduser().resolve(strict=True)
    scope, review = load_prepared_scope(experiment_root)
    group_dir = experiment_root / args.group
    group_result_path = group_dir / "fixed_group_result.json"
    require(group_result_path.is_file(),
            f"completion recheck requires a fully completed fixed group: {group_dir}")
    group_result = read_json(group_result_path)
    require(group_result.get("batches_completed") == EXPECTED_BATCHES,
            f"completion recheck requires all {EXPECTED_BATCHES} fixed batches")
    engine_root = args.engine_root.expanduser().resolve(strict=True)
    require((engine_root / "crypto_quant/research/factor_mining").is_dir(),
            f"engine root must contain crypto_quant: {engine_root}")
    if args.group == "old24":
        require(engine_root == args.baseline_engine_root.expanduser().resolve(strict=True),
                "old24 completion recheck must use the pre-change source")
    else:
        require(engine_root == args.current_engine_root.expanduser().resolve(strict=True),
                "new_multi completion recheck must use the current source")
    goal_root = Path(group_result["goal_root"]).resolve(strict=True)
    recheck_root = goal_root / "completion_rechecks" / "source-objective-recheck-v1"
    require(not recheck_root.exists(),
            f"refusing to repeat completion model review: {recheck_root}")
    worker_args = [
        sys.executable, str(Path(__file__).resolve()), "_recheck_worker",
        "--group", args.group,
        "--experiment-root", str(experiment_root),
        "--engine-root", str(engine_root),
        "--goal-root", str(goal_root),
        "--recheck-root", str(recheck_root),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(engine_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(worker_args, cwd=Path.cwd(), env=env, check=True)


def recheck_completion_worker(args: argparse.Namespace) -> None:
    engine_root = args.engine_root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(engine_root))
    from dataclasses import asdict, replace
    from hashlib import sha256
    from crypto_quant.research.factor_mining.contracts import dumps
    from crypto_quant.research.factor_mining.goal import GoalRunner, GoalSpec
    from crypto_quant.research.factor_mining.model_config import model_from_settings
    from crypto_quant.research.factor_mining.records import AgentGateway, RecordStore, write_json
    from crypto_quant.research.factor_mining.workflow import FactorMiner

    experiment_root = Path(args.experiment_root).resolve(strict=True)
    scope, review = load_prepared_scope(experiment_root)
    goal_root = Path(args.goal_root).resolve(strict=True)
    recheck_root = Path(args.recheck_root).resolve()
    require(not recheck_root.exists(), f"refusing to repeat completion model review: {recheck_root}")
    group_dir = experiment_root / args.group
    group_result = read_json(group_dir / "fixed_group_result.json")
    require(Path(group_result["goal_root"]).resolve() == goal_root
            and group_result.get("group") == args.group
            and group_result.get("batches_completed") == EXPECTED_BATCHES,
            "fixed group result does not match the requested Goal root")
    saved_goal_data = read_json(goal_root / "goal.json")
    source_goal = scope["source_goal_data"]["goal"]
    require(source_goal.get("goal_id") == scope["source_goal_id"],
            "source Goal identity differs from the fixed scope")
    corrected_goal = GoalSpec(
        goal_id=saved_goal_data["goal"]["goal_id"],
        objective=source_goal["objective"],
        target_ideas=group_result["target_ideas"],
    )
    model_settings = review["model"]["settings"]
    model = model_from_settings(model_settings)
    runner = GoalRunner(goal_root, model)
    require(runner.goal.goal_id == corrected_goal.goal_id
            and runner.goal.target_ideas == corrected_goal.target_ideas,
            "saved Goal identity or program target differs from the recheck")
    runner.goal = corrected_goal
    recheck_root.mkdir(parents=True, exist_ok=False)
    completion_records = recheck_root / "completion_records"
    runner.receipts = RecordStore(completion_records)
    runner.state = {**runner.state, "qualified_ideas": []}
    model_calls = recheck_root / "model_calls"
    model_calls.mkdir()

    def versioned_gateway(store):
        return AgentGateway(model, replace(runner.spec, run_id=runner.goal.goal_id),
                            store, model_calls, stage="A", progress=runner.progress)

    runner._gateway = versioned_gateway
    manifest = {
        "schema": "fm-v6-source-objective-recheck/v1",
        "group": args.group,
        "goal_root": str(goal_root),
        "source_goal_id": scope["source_goal_id"],
        "corrected_goal": asdict(corrected_goal),
        "corrected_objective_sha256": sha256(corrected_goal.objective.encode("utf-8")).hexdigest(),
        "previous_goal_objective": saved_goal_data["goal"]["objective"],
        "previous_completion_records": str((goal_root / "completion_records").resolve()),
        "recheck_completion_records": str(completion_records.resolve()),
        "model_settings": model_settings,
        "method": "call GoalRunner._review_completion for existing complete validations; no B loader or evaluator",
        "status": "started",
    }
    write_json(recheck_root / "recheck_manifest.json", manifest)
    batches_root = group_dir / "fixed_batches"
    batch_records = [read_json(batches_root / f"batch-{i:08d}.json")
                     for i in range(1, EXPECTED_BATCHES + 1)]
    require([item.get("batch_index") for item in batch_records] == list(range(1, EXPECTED_BATCHES + 1)),
            "completion recheck requires the seven saved batch boundaries")
    rechecked_runs = []
    for batch in batch_records:
        if batch["candidate_count"] == 0:
            continue
        run_id = batch["run_id"]
        run_dir = (goal_root / "runs" / run_id).resolve(strict=True)
        require(run_dir.is_relative_to(goal_root / "runs"),
                f"completion recheck run escaped Goal root: {run_dir}")
        require((run_dir / "b-access-started.json").is_file()
                and (run_dir / "b-numerical-complete.json").is_file()
                and (run_dir / "validation.json").is_file()
                and (run_dir / "B-report.md").is_file(),
                f"completion recheck requires a completed B checkpoint: {run_dir}")
        validation = read_json(run_dir / "validation.json")
        require(validation.get("status") == "complete",
                f"completion recheck requires complete B reports: {run_dir}")
        miner = FactorMiner.open(run_dir, model)
        runner._save(status="active", phase="completion_recheck", cycle=batch["batch_index"],
                     current_run=run_id, error=None,
                     task={"source_run_id": batch["source_run_id"],
                           "purpose": "recheck Goal match under the original research objective"})
        runner._review_completion(miner, validation)
        rechecked_runs.append(run_id)
    qualified = runner.state.get("qualified_ideas", [])
    target_met = len(qualified) >= corrected_goal.target_ideas
    runner._save(status="complete" if target_met else "active",
                 phase="completion_recheck_complete", current_run=None, error=None)
    result = {
        "schema": "fm-v6-source-objective-recheck-result/v1",
        "status": "complete",
        "group": args.group,
        "goal_root": str(goal_root),
        "corrected_goal": asdict(corrected_goal),
        "receipt_store": str(completion_records.resolve()),
        "model_calls": str(model_calls.resolve()),
        "qualified_idea_ids": qualified,
        "qualified_count": len(qualified),
        "target_met": target_met,
        "target_ideas": corrected_goal.target_ideas,
        "batches_completed": EXPECTED_BATCHES,
        "b_runs_rechecked": len(rechecked_runs),
        "runs": rechecked_runs,
    }
    write_json(recheck_root / "recheck_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def freeze_inputs(args: argparse.Namespace) -> None:
    experiment_root = args.output_root.expanduser().resolve(strict=True)
    scope, _ = load_prepared_scope(experiment_root)
    current_engine_root = args.current_engine_root.expanduser().resolve(strict=True)
    frozen_root = experiment_root / "frozen_inputs"
    require(not frozen_root.exists(), f"refusing to reload B inputs into existing directory: {frozen_root}")
    frozen_root.mkdir()
    worker_args = [
        sys.executable, str(Path(__file__).resolve()), "_freeze_worker",
        "--experiment-root", str(experiment_root),
        "--engine-root", str(current_engine_root),
        "--frozen-root", str(frozen_root),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(current_engine_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run(worker_args, cwd=Path.cwd(), env=env, check=True)


def freeze_inputs_worker(args: argparse.Namespace) -> None:
    engine_root = args.engine_root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(engine_root))
    from crypto_quant.research.factor_mining.cli import load_stage
    from crypto_quant.research.factor_mining.contracts import ResearchSpec

    experiment_root = Path(args.experiment_root).resolve(strict=True)
    scope, _ = load_prepared_scope(experiment_root)
    frozen_root = Path(args.frozen_root).resolve()
    require(frozen_root.is_dir() and not any(frozen_root.iterdir()),
            f"frozen B input output must be an empty prepared directory: {frozen_root}")
    verify_scope_inputs(scope)
    db_path = Path(scope["input_signatures"]["db"]["path"])
    universe_path = Path(scope["input_signatures"]["universe"]["path"])
    input_before = file_signature(db_path)
    universe_before = file_signature(universe_path, hash_contents=True)
    spec_data = dict(scope["research_spec"])
    spec_data["run_id"] = "fm-v6-frozen-b-inputs-20261001"
    spec_data["b_horizons"] = list(HORIZONS)
    spec = ResearchSpec.from_dict(spec_data)
    panel = load_stage(
        db_path, universe_path, spec, "B",
        include_liquidations=bool(scope["source_goal_data"]["inputs"]["include_liquidations"]),
    )
    values_path = frozen_root / "values.parquet"
    universe_frame_path = frozen_root / "universe.parquet"
    panel.values.to_parquet(values_path, index=True)
    panel.universe.rename("eligible").to_frame().to_parquet(universe_frame_path, index=True)
    input_after = file_signature(db_path)
    universe_after = file_signature(universe_path, hash_contents=True)
    require(input_before == input_after and universe_before == universe_after,
            "source DB or universe changed while the frozen B panel was being loaded")
    data_usage = panel.diagnostics.get("data_usage", {})
    provenance = {
        "schema": "fm-v6-frozen-b-inputs/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_goal_id": scope["source_goal_id"],
        "source_db_before": input_before,
        "source_db_after": input_after,
        "source_universe_before": universe_before,
        "source_universe_after": universe_after,
        "data_usage": data_usage,
        "rows": len(panel.values),
        "columns": list(panel.values.columns),
        "symbols": int(panel.universe.index.get_level_values("symbol").nunique()),
        "artifacts": {
            "values": file_signature(values_path, hash_contents=True),
            "universe": file_signature(universe_frame_path, hash_contents=True),
        },
    }
    write_new_json(frozen_root / "provenance.json", provenance)
    print(json.dumps({
        "status": "frozen",
        "rows": provenance["rows"],
        "symbols": provenance["symbols"],
        "values": provenance["artifacts"]["values"],
        "universe": provenance["artifacts"]["universe"],
    }, ensure_ascii=False, indent=2))


def import_fixed_a_batch(miner: Any, scope: dict[str, Any], batch: dict[str, Any],
                         review_by_key: dict[str, dict[str, Any]], old_policy: bool,
                         model_settings: dict[str, Any], b_membership: Any,
                         write_json: Any) -> list[str]:
    write_json(miner.root / "model-settings.json", model_settings)
    source_run_dir = Path(scope["source_goal"]) / "runs" / batch["source_run_id"]
    (miner.root / "A-universe.csv").write_bytes((source_run_dir / "A-universe.csv").read_bytes())
    candidate_ids: list[str] = []
    new_decisions = []
    for fixed in batch["candidates"]:
        candidate_id = fixed["candidate_id"]
        candidate_ids.append(candidate_id)
        source_a_records = {}
        for suffix, expected_kind in (("definition", "candidate"),
                                      ("calculation", "calculation"),
                                      ("report", "model_report")):
            source_record = read_json(source_run_dir / "a_records" / f"{candidate_id}-{suffix}.json")
            require(source_record.get("kind") == expected_kind,
                    f"source A {expected_kind} record differs for {fixed['factor_key']}")
            source_a_records[suffix] = source_record
            miner.store.append(source_record["id"], source_record["kind"], source_record["data"])

        archive_path = Path(fixed["source_archive"])
        verify_signature(fixed["source_archive_signature"], "source A factor archive")
        a_payload, source_meta = load_archive_payload(archive_path, fixed["source_evaluation_id"])
        validate_a_payload(a_payload, fixed["identity"], fixed["factor_key"])
        factor_identity = miner._factor_identity(fixed["identity"]["expanded_expression"],
                                                  fixed["identity"]["direction"])
        locator = miner._write_evaluation(candidate_id, "A", a_payload, factor_identity,
                                          definition=fixed["definition"])
        rank_displacement, displacement_source = imported_a_rank_displacement(
            miner, fixed, source_meta, source_run_dir, source_a_records["calculation"]["data"])
        if rank_displacement is None:
            evaluation_data = miner._evaluation_record_data(candidate_id, a_payload, locator)
        else:
            displacement_locator = miner._write_rank_displacement(
                candidate_id, "A", rank_displacement, factor_identity, source=displacement_source)
            evaluation_data = miner._evaluation_record_data(
                candidate_id, a_payload, locator, rank_displacement, displacement_locator)
        miner.store.append(f"{candidate_id}-evaluation", "evaluation", evaluation_data)

        decision = json.loads(json.dumps(fixed["source_decision"]))
        if not old_policy:
            saved = review_by_key[fixed["factor_key"]]
            decision["retained_horizons"] = saved["retained_horizons"]
            if model_settings["provider"] == "xiaomi-token-plan":
                review_model = f"{model_settings['model']} (thinking={model_settings['thinking']})"
            else:
                review_model = (f"{model_settings['model']} "
                                f"(reasoning_effort={model_settings['reasoning_effort']})")
            decision["reason"] = (
                f"{decision['reason']}\n\n"
                f"多期限A审阅追加依据（{review_model}，仅依据A证据）：{saved['reason']}"
            )
        new_decisions.append(decision)

    optimization = {
        "analysis": "固定历史A retain决策；不生成、修改或优化公式。",
        "decisions": new_decisions,
        "diagnostics": [], "proposals": [], "route_states": {},
    }
    miner.store.append("fixed-optimization", "optimization", optimization)
    completion = {
        "completed_rounds": 0,
        "candidate_ids": candidate_ids,
        "evaluated_ids": candidate_ids,
        "retained_ids": candidate_ids,
        "pending_report_ids": [],
        "candidate_decisions": {decision["candidate_id"]: decision for decision in new_decisions},
        "routes": {}, "proposals": [],
        "stop_reason": "fixed historical A evidence imported without new formula generation or optimization",
    }
    write_json(miner.root / "a-complete.json", completion)
    miner.freeze(candidate_ids, b_membership)
    return candidate_ids


def run_group_worker(args: argparse.Namespace) -> None:
    engine_root = args.engine_root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(engine_root))
    from crypto_quant.research.factor_mining.model_config import model_from_settings
    from crypto_quant.features.factor_inputs import FactorInputPanel, validate_universe
    from crypto_quant.research.factor_mining.contracts import ResearchSpec, digest
    from crypto_quant.research.factor_mining.goal import GoalRunner, GoalSpec
    from crypto_quant.research.factor_mining.records import RecordStore, write_json
    from crypto_quant.research.factor_mining.workflow import FactorMiner

    experiment_root = Path(args.experiment_root).resolve(strict=True)
    scope, review = load_prepared_scope(experiment_root)
    frozen_manifest_path = Path(args.frozen_panel_manifest).resolve(strict=True)
    frozen_manifest = read_json(frozen_manifest_path)
    require(frozen_manifest.get("schema") == "fm-v6-frozen-b-inputs/v1",
            "frozen B input manifest schema differs")
    verify_scope_inputs(scope)
    require(frozen_manifest["source_db_before"] == scope["input_signatures"]["db"]
            and frozen_manifest["source_db_after"] == scope["input_signatures"]["db"]
            and frozen_manifest["source_universe_before"] == scope["input_signatures"]["universe"]
            and frozen_manifest["source_universe_after"] == scope["input_signatures"]["universe"],
            "frozen B input does not use the unchanged source DB and universe")
    for name in ("values", "universe"):
        verify_signature(frozen_manifest["artifacts"][name], f"frozen B {name} panel")
    import pandas as pd
    values = pd.read_parquet(frozen_manifest["artifacts"]["values"]["path"])
    universe_frame = pd.read_parquet(frozen_manifest["artifacts"]["universe"]["path"])
    require(list(universe_frame.columns) == ["eligible"],
            "frozen B universe parquet must contain only the eligible column")
    b_universe = validate_universe(universe_frame["eligible"])
    b_panel = FactorInputPanel(values, b_universe, {"data_usage": frozen_manifest["data_usage"]})
    require(len(b_panel.values) == frozen_manifest["rows"]
            and list(b_panel.values.columns) == frozen_manifest["columns"],
            "frozen B panel differs from its provenance")
    require(b_panel.values.index.equals(b_panel.universe.index),
            "frozen B panel values and universe indices differ")
    group_dir = Path(args.group_dir).resolve()
    if args.resume:
        require(group_dir.is_dir() and not (group_dir / "fixed_group_result.json").exists(),
                f"resume requires an incomplete saved group: {group_dir}")
    else:
        require(group_dir.is_dir() and not any(group_dir.iterdir()),
                f"new group output directory must be empty: {group_dir}")
    old_policy = args.group == "old24"
    goal_id = f"fm-v6-horizon-{args.group}-20261001"
    goal_target = EXPECTED_OLD_CARDS if old_policy else EXPECTED_OLD_CARDS + 1
    objective = scope["source_goal_data"]["goal"]["objective"]
    goal = GoalSpec(goal_id=goal_id, objective=objective, target_ideas=goal_target)
    spec_payload = dict(scope["research_spec"])
    spec_payload["run_id"] = goal_id
    if old_policy:
        spec_payload.pop("b_horizons", None)
    else:
        spec_payload["b_horizons"] = list(HORIZONS)
    spec = ResearchSpec.from_dict(spec_payload)
    goal_output = group_dir / "goal"
    idea_pool = group_dir / "idea_pool"
    idea_pool.mkdir(exist_ok=args.resume)
    db_path = Path(scope["input_signatures"]["db"]["path"])
    universe_path = Path(scope["input_signatures"]["universe"]["path"])
    inputs = {
        "db": str(db_path), "universe": str(universe_path),
        "idea_pool": str(idea_pool.resolve()),
        "include_liquidations": bool(scope["source_goal_data"]["inputs"]["include_liquidations"]),
        "poll_seconds": 60,
        "frozen_b_panel": str(frozen_manifest_path),
        "frozen_b_values_sha256": frozen_manifest["artifacts"]["values"]["sha256"],
        "frozen_b_universe_sha256": frozen_manifest["artifacts"]["universe"]["sha256"],
    }
    require(Path(inputs["idea_pool"]).resolve() != Path(scope["source_goal_data"]["inputs"]["idea_pool"]).resolve(),
            "comparison idea pool must be isolated from the original pool")
    model_settings = review["model"]["settings"]
    model = model_from_settings(model_settings)
    saved_goal_root = goal_output / goal_id
    if args.resume:
        require(saved_goal_root.is_dir(), f"saved Goal root is missing: {saved_goal_root}")
        runner = GoalRunner(saved_goal_root, model)
        require(runner.goal == goal and runner.spec == spec and runner.inputs == inputs,
                "saved fixed Goal contract or inputs differ from the requested replay")
        require(idea_pool.is_dir(), f"saved isolated idea pool is missing: {idea_pool}")
    else:
        runner = GoalRunner.create(
            goal, spec, model, goal_output, inputs=inputs,
            model_settings=model.settings(),
        )

    b_membership = b_panel.universe
    review_by_key = {decision["factor_key"]: decision for decision in review["decisions"]}
    batches_root = group_dir / "fixed_batches"
    batches_root.mkdir(exist_ok=args.resume)
    all_run_ids: list[str] = []
    for batch in scope["batches"]:
        index = batch["batch_index"]
        run_id = f"goal-{digest({**goal.__dict__, 'batch': index})[:12]}-{index:06d}"
        runner._save(status="active", phase="fixed_batch_replay", cycle=index,
                     current_run=run_id if batch["candidate_count"] else None,
                     error=None, task={"source_run_id": batch["source_run_id"],
                                       "fixed_candidate_count": batch["candidate_count"]})
        batch_result = {
            "batch_index": index,
            "source_run_id": batch["source_run_id"],
            "run_id": run_id if batch["candidate_count"] else None,
            "candidate_count": batch["candidate_count"],
            "factor_keys": batch["factor_keys"],
            "policy": "24h_only" if old_policy else "A_B_horizon_intersection",
        }
        if not batch["candidate_count"]:
            batch_path = batches_root / f"batch-{index:08d}.json"
            if batch_path.exists():
                require(read_json(batch_path) == batch_result,
                        f"saved empty batch metadata changed during resume: {batch_path}")
            else:
                write_json(batch_path, batch_result)
            continue

        run_spec = ResearchSpec.from_dict({**spec.as_dict(), "run_id": run_id})
        run_dir = runner.root / "runs" / run_id
        if run_dir.exists():
            require(args.resume, f"existing run requires explicit --resume: {run_dir}")
            miner = FactorMiner.open(run_dir, model)
            require(miner.spec == run_spec and (run_dir / "frozen_batch.json").is_file(),
                    f"saved run is not a complete frozen batch: {run_dir}")
            frozen = miner._checked_frozen()
            require(set(frozen["candidates"]) == {item["candidate_id"] for item in batch["candidates"]},
                    f"saved frozen candidates differ for batch {index}")
        else:
            miner = FactorMiner(run_spec, model, runner.root / "runs")
            candidate_ids = import_fixed_a_batch(
                miner, scope, batch, review_by_key, old_policy, model_settings,
                b_membership, write_json,
            )
            require(set(candidate_ids) == {item["candidate_id"] for item in batch["candidates"]},
                    f"imported A candidates differ for batch {index}")
        all_run_ids.append(run_id)

        if (miner.root / "b-access-started.json").exists():
            require((miner.root / "b-numerical-complete.json").is_file(),
                    "B was previously accessed without a complete numerical checkpoint; refusing another read")
            validation = miner.complete_reports("B", idea_pool)
        else:
            validation = miner.validate(lambda: b_panel, idea_pool)
        require(validation["status"] == "complete",
                f"B model reports remain incomplete in {run_id}; numerical data stays checkpointed")
        runner._review_completion(miner, validation)
        batch_result.update({
            "run_path": str(miner.root.resolve()),
            "b_numerical_complete": (miner.root / "b-numerical-complete.json").is_file(),
            "validation_complete": (miner.root / "validation.json").is_file(),
            "b_report": str(miner.root / "B-report.md"),
            "admission_receipt": str(runner.root / "completion_records" / f"admission-{index:08d}.json"),
            "match_receipt": str(runner.root / "completion_records" / f"match-{index:08d}.json"),
        })
        batch_path = batches_root / f"batch-{index:08d}.json"
        if batch_path.exists():
            require(read_json(batch_path) == batch_result,
                    f"saved fixed-batch metadata changed during resume: {batch_path}")
        else:
            write_json(batch_path, batch_result)

    qualified = runner.state.get("qualified_ideas", [])
    target_met = len(qualified) >= goal_target
    runner._save(status="complete" if target_met else "active",
                 phase="fixed_batch_replay_complete", current_run=None, error=None)
    group_result = {
        "schema": "fm-v6-fixed-group-result/v1",
        "group": args.group,
        "policy": "24h_only" if old_policy else "A_B_horizon_intersection",
        "engine_root": str(engine_root),
        "goal_root": str(runner.root.resolve()),
        "idea_pool": str(idea_pool.resolve()),
        "goal_id": goal_id,
        "target_ideas": goal_target,
        "qualified_idea_ids": qualified,
        "qualified_count": len(qualified),
        "target_met": target_met,
        "batches_completed": EXPECTED_BATCHES,
        "frozen_b_panel": str(frozen_manifest_path),
        "frozen_b_values_sha256": frozen_manifest["artifacts"]["values"]["sha256"],
        "frozen_b_universe_sha256": frozen_manifest["artifacts"]["universe"]["sha256"],
        "runs": all_run_ids,
    }
    write_json(group_dir / "fixed_group_result.json", group_result)
    print(json.dumps(group_result, ensure_ascii=False, indent=2))


def checked_record(path: Path, expected_kind: str, candidate_id: str | None = None) -> dict[str, Any]:
    record = read_json(path)
    require(record.get("kind") == expected_kind, f"unexpected record kind in {path}")
    require(isinstance(record.get("data"), dict), f"record data is missing in {path}")
    if candidate_id is not None:
        require(record["data"].get("candidate_id") == candidate_id,
                f"record belongs to another candidate: {path}")
    return record


def archive_map_for_group(goal_root: Path, run_dirs: list[Path]) -> tuple[Path, dict[bytes, Path]]:
    archive_root: Path | None = None
    for run_dir in run_dirs:
        marker = read_json(run_dir / "factor_archive.json")
        require(marker.get("format") == "one-factor-one-file-v2",
                f"unsupported archive marker in {run_dir}")
        relative = Path(marker["archive_root"])
        require(not relative.is_absolute(), "factor archive marker must be relative")
        current = (run_dir / relative).resolve(strict=True)
        if archive_root is None:
            archive_root = current
        else:
            require(archive_root == current,
                    f"group run archives differ: {archive_root} versus {current}")
    require(archive_root is not None and archive_root.is_dir(),
            f"factor archive directory is missing under {goal_root}")
    archives: dict[bytes, Path] = {}
    for path in sorted(archive_root.glob("factor-*.sqlite3")):
        identity = load_archive_identity(path)
        identity_bytes = canonical_json(identity)
        require(identity_bytes not in archives, f"duplicate factor archive identity in {archive_root}")
        archives[identity_bytes] = path.resolve()
    return archive_root, archives


def validate_b_archive(run_dir: Path, archive_root: Path, archives: dict[bytes, Path],
                       locator: dict[str, Any], identity: dict[str, Any], horizon: int,
                       compact_report: dict[str, Any]) -> dict[str, Any]:
    require(locator.get("identity") == identity,
            f"B archive factor identity differs for {run_dir.name}/{horizon}h")
    relative = Path(locator["root"])
    require(not relative.is_absolute() and (run_dir / relative).resolve() == archive_root,
            f"B locator archive root differs for {run_dir.name}/{horizon}h")
    archive_path = archives.get(canonical_json(identity))
    require(archive_path is not None, f"B factor archive is missing for {run_dir.name}/{horizon}h")
    payload, row = load_archive_payload(archive_path, locator["evaluation_id"], expected_segment="B")
    key = locator["evaluation_key"]
    expected_key = {
        "data_version": row["data_version"],
        "contract_version": row["contract_version"],
        "evaluator_version": row["evaluator_version"],
        "segment": row["segment"],
        "horizon": row["horizon"],
    }
    require(key == expected_key and row["horizon"] == f"{horizon}h",
            f"B archive evaluation key differs for {run_dir.name}/{horizon}h")
    require(payload.get("segment") == "B" and payload.get("horizon_hours") == horizon,
            f"B archive report horizon differs for {run_dir.name}/{horizon}h")
    require(str(payload.get("direction")) == str(identity["direction"]),
            f"B archive direction differs for {run_dir.name}/{horizon}h")
    require(payload.get("summary") == compact_report.get("summary")
            and payload.get("coverage") == compact_report.get("coverage"),
            f"compact B summary differs from archived report for {run_dir.name}/{horizon}h")
    for field in ("summary", "coverage", "grouping", "per_symbol", "stages", "periods"):
        require(field in payload, f"full B numerical report omits {field}: {run_dir.name}/{horizon}h")
    require(payload["periods"] and payload["stages"],
            f"full B numerical report has no period or stage rows: {run_dir.name}/{horizon}h")
    return payload


def report_horizon_summary(payload: dict[str, Any], test: dict[str, Any],
                            model_report_path: Path) -> dict[str, Any]:
    summary = payload["summary"]
    rank_ic = summary["rank_ic"]
    spread = summary["directional_spread"]
    coverage = payload["coverage"]
    return {
        "horizon_hours": payload["horizon_hours"],
        "rank_ic_mean": rank_ic.get("mean"),
        "rank_ic_p_value": rank_ic.get("p_value"),
        "rank_ic_adjusted_p": test.get("adjusted_p"),
        "rank_ic_bh_rejected": test.get("rejected"),
        "directional_spread_mean": spread.get("mean"),
        "eligible_observations": coverage.get("eligible_observations"),
        "purged_hours": coverage.get("purged_hours"),
        "purged_observations": coverage.get("purged_observations"),
        "period_rows": len(payload["periods"]),
        "stage_rows": len(payload["stages"]),
        "model_report_path": str(model_report_path.resolve()),
    }


def collect_group_results(experiment_root: Path, group_name: str,
                          scope: dict[str, Any], review: dict[str, Any],
                          frozen_panel: dict[str, Any]) -> dict[str, Any]:
    group_dir = experiment_root / group_name
    group_result = read_json(group_dir / "fixed_group_result.json")
    require(group_result.get("group") == group_name
            and group_result.get("batches_completed") == EXPECTED_BATCHES,
            f"{group_name} has not completed all seven fixed batches")
    goal_root = Path(group_result["goal_root"]).resolve(strict=True)
    idea_pool = Path(group_result["idea_pool"]).resolve(strict=True)
    require(goal_root.is_relative_to(group_dir.resolve()) and idea_pool.is_relative_to(group_dir.resolve()),
            f"{group_name} Goal and idea pool must be isolated below its group directory")
    require(idea_pool != Path(scope["source_goal_data"]["inputs"]["idea_pool"]).resolve()
            and idea_pool != (experiment_root / ("new_multi" if group_name == "old24" else "old24") / "idea_pool").resolve(),
            f"{group_name} idea pool overlaps another comparison pool")
    goal_data = read_json(goal_root / "goal.json")
    inputs = goal_data["inputs"]
    require(Path(inputs["db"]).resolve() == Path(scope["input_signatures"]["db"]["path"]).resolve()
            and Path(inputs["universe"]).resolve() == Path(scope["input_signatures"]["universe"]["path"]).resolve()
            and Path(inputs["idea_pool"]).resolve() == idea_pool,
            f"{group_name} Goal does not use the fixed source files and isolated pool")
    require(inputs.get("frozen_b_panel") == str((experiment_root / "frozen_inputs" / "provenance.json").resolve())
            and inputs.get("frozen_b_values_sha256") == frozen_panel["artifacts"]["values"]["sha256"]
            and inputs.get("frozen_b_universe_sha256") == frozen_panel["artifacts"]["universe"]["sha256"],
            f"{group_name} Goal does not use the shared frozen B panel")
    require(goal_data["research"]["label"] == scope["research_spec"]["label"]
            and goal_data["research"]["a_start"] == scope["research_spec"]["a_start"]
            and goal_data["research"]["b_start"] == scope["research_spec"]["b_start"]
            and goal_data["research"]["c_start"] == scope["research_spec"]["c_start"],
            f"{group_name} research time bounds differ from the frozen scope")
    verify_scope_inputs(scope)
    require(file_signature(Path(inputs["db"])) == frozen_panel["source_db_after"]
            and file_signature(Path(inputs["universe"]), hash_contents=True)
            == frozen_panel["source_universe_after"],
            f"source inputs changed before {group_name} comparison completed")

    batches_root = group_dir / "fixed_batches"
    require(batches_root.is_dir(), f"{group_name} fixed-batch index is missing")
    batch_records = [read_json(batches_root / f"batch-{i:08d}.json")
                     for i in range(1, EXPECTED_BATCHES + 1)]
    require([item.get("batch_index") for item in batch_records] == list(range(1, EXPECTED_BATCHES + 1)),
            f"{group_name} does not preserve all seven original batch positions")
    expected_batch_map = {batch["batch_index"]: batch for batch in scope["batches"]}
    expected_run_ids = {item["run_id"] for item in batch_records if item["run_id"] is not None}
    require(set(group_result["runs"]) == expected_run_ids,
            f"{group_name} run list differs from its fixed-batch index")
    run_dirs: list[Path] = []
    for batch_record in batch_records:
        expected = expected_batch_map[batch_record["batch_index"]]
        require(batch_record["source_run_id"] == expected["source_run_id"]
                and batch_record["candidate_count"] == expected["candidate_count"]
                and batch_record["factor_keys"] == expected["factor_keys"],
                f"{group_name} changed the original candidate batch {expected['batch_index']}")
        if expected["candidate_count"] == 0:
            require(batch_record["run_id"] is None,
                    f"{group_name} created a nonempty run for the originally empty cycle")
            continue
        require(isinstance(batch_record.get("run_id"), str),
                f"{group_name} run ID is missing for batch {expected['batch_index']}")
        run_dir = (goal_root / "runs" / batch_record["run_id"]).resolve(strict=True)
        require(run_dir.is_relative_to(goal_root / "runs"),
                f"{group_name} batch run escaped its Goal root")
        run_dirs.append(run_dir)
        require(batch_record.get("run_path") == str(run_dir)
                and batch_record.get("b_numerical_complete") is True
                and batch_record.get("validation_complete") is True,
                f"{group_name} batch report is incomplete: {run_dir}")
    archive_root, archives = archive_map_for_group(goal_root, run_dirs)
    review_by_key = {item["factor_key"]: item for item in review["decisions"]}
    group_states: dict[str, dict[str, Any]] = {}
    candidate_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for batch_record in batch_records:
        batch = expected_batch_map[batch_record["batch_index"]]
        if not batch["candidate_count"]:
            continue
        run_id = batch_record["run_id"]
        run_dir = goal_root / "runs" / run_id
        frozen = read_json(run_dir / "frozen_batch.json")
        fixed_candidates = {candidate["candidate_id"]: candidate for candidate in batch["candidates"]}
        require(set(frozen.get("candidates", {})) == set(fixed_candidates),
                f"{group_name} frozen B membership differs for batch {batch['batch_index']}")
        a_complete = read_json(run_dir / "a-complete.json")
        require(set(a_complete.get("retained_ids", [])) == set(fixed_candidates),
                f"{group_name} imported A retain IDs differ for batch {batch['batch_index']}")
        require((run_dir / "b-access-started.json").is_file(),
                f"{group_name} did not record B access for {run_dir}")
        numerical_checkpoint = read_json(run_dir / "b-numerical-complete.json")
        require(set(numerical_checkpoint.get("candidate_ids", [])) == set(fixed_candidates),
                f"{group_name} B numerical checkpoint omits a fixed candidate in {run_dir}")
        validation_output = read_json(run_dir / "validation.json")
        require(validation_output.get("status") == "complete"
                and set(validation_output.get("decisions", {})) == set(fixed_candidates),
                f"{group_name} B report checkpoint is incomplete in {run_dir}")
        require((run_dir / "B-report.md").is_file()
                and (run_dir / "B-report.md").stat().st_size > 0,
                f"{group_name} aggregate B report is missing in {run_dir}")

        correction_record = checked_record(
            run_dir / "b_records" / "batch-correction.json", "multiple_testing"
        )
        correction = correction_record["data"]
        require(correction.get("method") == "BH", f"{group_name} batch correction is not BH")
        correction_tests = correction.get("tests", [])
        test_map: dict[tuple[str, int], dict[str, Any]] = {}
        for test in correction_tests:
            candidate_id = test.get("candidate_id")
            horizon = test.get("horizon_hours", 24)
            key = (candidate_id, horizon)
            require(key not in test_map, f"duplicate BH test in {run_dir}: {key}")
            test_map[key] = test

        for candidate_id, fixed in fixed_candidates.items():
            frozen_item = frozen["candidates"][candidate_id]
            identity = fixed["identity"]
            require(frozen_item["definition"] == fixed["definition"]
                    and frozen_item["executed"]["expanded_expression"] == identity["expanded_expression"]
                    and str(frozen_item["definition"]["direction"]) == str(identity["direction"])
                    and frozen_item["a_evaluation_archive"]["identity"] == identity,
                    f"{group_name} changed formula, direction, or semantics for {fixed['factor_key']}")
            if group_name == "new_multi":
                retained_horizons = frozen_item.get("retained_horizons")
                saved_horizons = review_by_key[fixed["factor_key"]]["retained_horizons"]
                require(retained_horizons == saved_horizons
                        and frozen_item["a_decision"]["retained_horizons"] == saved_horizons,
                        f"{group_name} changed the one-time A horizon decision for {fixed['factor_key']}")
            else:
                retained_horizons = [24]
            expected_horizon_set = set(retained_horizons)

            evaluation_record = checked_record(
                run_dir / "b_records" / f"{candidate_id}-evaluation.json", "evaluation", candidate_id
            )
            evaluation_data = evaluation_record["data"]
            require(evaluation_data.get("segment") == "B"
                    and str(evaluation_data.get("direction")) == str(identity["direction"]),
                    f"{group_name} B evaluation identity differs for {fixed['factor_key']}")
            if group_name == "old24":
                require(evaluation_data.get("horizon_hours") == 24,
                        f"{group_name} must use its original 24h evaluation for {fixed['factor_key']}")
                evaluation_horizons = {"24": {
                    "summary": evaluation_data["summary"],
                    "coverage": evaluation_data["coverage"],
                    "factor_archive": evaluation_data["factor_archive"],
                }}
            else:
                require(evaluation_data.get("retained_horizons") == retained_horizons,
                        f"{group_name} B evaluation horizon set differs for {fixed['factor_key']}")
                evaluation_horizons = evaluation_data.get("horizons", {})
                require(set(evaluation_horizons) == {str(horizon) for horizon in retained_horizons},
                        f"{group_name} B evaluation omits a retained horizon for {fixed['factor_key']}")
            require(set(evaluation_horizons) == {str(horizon) for horizon in retained_horizons},
                    f"{group_name} B horizon report set differs for {fixed['factor_key']}")
            payloads = {}
            tests_by_horizon = {}
            for horizon in retained_horizons:
                horizon_text = str(horizon)
                compact = evaluation_horizons[horizon_text]
                locator = compact["factor_archive"]
                payloads[horizon] = validate_b_archive(
                    run_dir, archive_root, archives, locator, identity, horizon, compact
                )
                test = test_map.get((candidate_id, horizon))
                require(test is not None and test.get("metric") == "rank_ic",
                        f"{group_name} BH family omits {candidate_id} {horizon}h")
                tests_by_horizon[horizon] = test
            if group_name == "new_multi":
                require(set(test_map) == {
                    (cid, horizon)
                    for cid, batch_fixed in fixed_candidates.items()
                    for horizon in frozen["candidates"][cid]["retained_horizons"]
                }, f"{group_name} BH family differs from the frozen candidate-horizon pairs")
                require(len(correction_tests) == sum(len(item["retained_horizons"])
                                                     for item in frozen["candidates"].values())
                        and correction.get("family_size") == len(correction_tests),
                        f"{group_name} BH family size differs from the frozen tests")
                require(numerical_checkpoint.get("retained_horizons") == {
                    cid: frozen["candidates"][cid]["retained_horizons"] for cid in fixed_candidates
                }, f"{group_name} B checkpoint horizons differ from frozen A choices")
            else:
                require(set(test_map) == {(cid, 24) for cid in fixed_candidates}
                        and len(correction_tests) == len(fixed_candidates)
                        and correction.get("family_size") == len(fixed_candidates),
                        f"{group_name} 24h BH family differs from the fixed batch")

            validation_record = checked_record(
                run_dir / "b_records" / f"{candidate_id}-validation.json", "validation_result", candidate_id
            )
            program_decision = validation_record["data"]
            final_decision = validation_output["decisions"][candidate_id]
            saved_program_decision = {key: value for key, value in final_decision.items() if key != "idea_card"}
            program_record_decision = {key: value for key, value in program_decision.items()
                                       if key != "candidate_id"}
            require(saved_program_decision == program_record_decision,
                    f"{group_name} final validation differs from its saved B decision for {fixed['factor_key']}")
            idea_card_path = final_decision.get("idea_card")
            if program_decision["eligible_for_idea_pool"]:
                require(isinstance(idea_card_path, str) and Path(idea_card_path).is_file(),
                        f"{group_name} eligible B verdict lacks an actual card file for {fixed['factor_key']}")
            else:
                require(idea_card_path is None,
                        f"{group_name} ineligible B verdict unexpectedly has a card for {fixed['factor_key']}")
            require(program_decision.get("tests") == [test_map[(candidate_id, horizon)]
                                                       for horizon in retained_horizons],
                    f"{group_name} B verdict tests differ from correction records for {fixed['factor_key']}")
            if group_name == "new_multi":
                passed_horizons = program_decision.get("passed_horizons", [])
                horizon_results = program_decision.get("horizon_results", {})
                require(set(horizon_results) == {str(h) for h in retained_horizons}
                        and passed_horizons == [h for h in retained_horizons
                            if horizon_results[str(h)]["validation_status"] == "passed"],
                        f"{group_name} per-horizon verdicts do not match passed_horizons for {fixed['factor_key']}")
            else:
                passed_horizons = [24] if program_decision.get("validation_status") == "passed" else []
                require(program_decision.get("eligible_for_idea_pool") is bool(passed_horizons),
                        f"{group_name} 24h verdict and card eligibility differ for {fixed['factor_key']}")

            model_report_path = run_dir / "b_records" / f"{candidate_id}-report.json"
            model_record = checked_record(model_report_path, "model_report", candidate_id)
            model_data = model_record["data"]
            narrative_fields = {"analysis", "mechanism", "conditions", "falsifiers", "limitations", "next_steps"}
            require(narrative_fields <= model_data.keys()
                    and all(model_data[key] for key in narrative_fields),
                    f"{group_name} candidate model report is incomplete for {fixed['factor_key']}")

            decision_summary = {
                "factor_key": fixed["factor_key"],
                "batch_index": fixed["batch_index"],
                "candidate_id": candidate_id,
                "source_run_id": batch["source_run_id"],
                "run_id": run_id,
                "identity": identity,
                "definition": fixed["definition"],
                "retained_horizons": retained_horizons,
                "passed_horizons": passed_horizons,
                "validation_status": program_decision["validation_status"],
                "eligible_for_idea_pool": program_decision["eligible_for_idea_pool"],
                "idea_card_path": str(Path(idea_card_path).resolve()) if idea_card_path else None,
                "reasons": program_decision.get("reasons", []),
                "horizon_results": program_decision.get("horizon_results", {}),
                "horizons": {
                    str(horizon): report_horizon_summary(payloads[horizon], tests_by_horizon[horizon],
                                                         model_report_path)
                    for horizon in retained_horizons
                },
                "run_path": str(run_dir.resolve()),
                "evaluation_record_path": str((run_dir / "b_records" / f"{candidate_id}-evaluation.json").resolve()),
                "validation_record_path": str((run_dir / "b_records" / f"{candidate_id}-validation.json").resolve()),
                "model_report_path": str(model_report_path.resolve()),
                "model_report": model_data,
            }
            group_states[fixed["factor_key"]] = decision_summary
            candidate_lookup[(run_id, candidate_id)] = fixed

    recheck_root = goal_root / "completion_rechecks" / "source-objective-recheck-v1"
    recheck_manifest = read_json(recheck_root / "recheck_manifest.json")
    recheck_result = read_json(recheck_root / "recheck_result.json")
    source_objective = scope["source_goal_data"]["goal"]["objective"]
    require(recheck_manifest.get("corrected_goal", {}).get("objective") == source_objective
            and recheck_manifest.get("corrected_objective_sha256")
            == hashlib.sha256(source_objective.encode("utf-8")).hexdigest()
            and recheck_result.get("corrected_goal", {}).get("objective") == source_objective
            and recheck_result.get("status") == "complete"
            and recheck_result.get("batches_completed") == EXPECTED_BATCHES
            and recheck_result.get("target_ideas") == group_result.get("target_ideas"),
            f"{group_name} completion recheck is not bound to the original research objective")
    receipts_root = Path(recheck_result["receipt_store"]).resolve(strict=True)
    require(receipts_root == (recheck_root / "completion_records").resolve(),
            f"{group_name} corrected completion receipts are outside their versioned store")
    admissions: dict[str, dict[str, Any]] = {}
    matches: dict[str, dict[str, Any]] = {}
    for path in sorted(receipts_root.glob("admission-*.json")):
        record = read_json(path)
        require(record.get("kind") == "program_admission", f"unexpected {group_name} admission receipt: {path}")
        for idea in record.get("data", {}).get("ideas", []):
            require(idea.get("admitted_by_program") is True,
                    f"{group_name} admission receipt lacks program approval: {path}")
            require(idea["idea_id"] not in admissions,
                    f"duplicate {group_name} program admission: {idea['idea_id']}")
            admissions[idea["idea_id"]] = {**idea, "receipt_path": str(path.resolve())}
    for path in sorted(receipts_root.glob("match-*.json")):
        record = read_json(path)
        require(record.get("kind") == "goal_match", f"unexpected {group_name} Goal match receipt: {path}")
        for match in record.get("data", {}).get("matches", []):
            require(match["idea_id"] not in matches,
                    f"duplicate {group_name} Goal match receipt: {match['idea_id']}")
            matches[match["idea_id"]] = {**match, "receipt_path": str(path.resolve())}
    require(set(admissions) == set(matches),
            f"{group_name} program admissions and Goal match receipts differ")

    pool_cards = {path.stem: path.resolve() for path in idea_pool.glob("*.json")}
    require(set(pool_cards) == set(admissions),
            f"{group_name} actual idea pool differs from its program admission receipts")
    delivery_by_id: dict[str, dict[str, Any]] = {}
    if group_name == "new_multi":
        for path in sorted((goal_root / "delivery_evidence").glob("admission-*.json")):
            record = read_json(path)
            for item in record.get("ideas", []):
                require(item["idea_id"] not in delivery_by_id,
                        f"duplicate new_multi delivery evidence: {item['idea_id']}")
                delivery_by_id[item["idea_id"]] = {**item, "delivery_path": str(path.resolve())}
        require(set(delivery_by_id) == set(admissions),
                "new_multi delivery evidence does not cover every admitted card")

    qualified: dict[str, dict[str, Any]] = {}
    all_cards: list[dict[str, Any]] = []
    for idea_id, admission in admissions.items():
        run_id, candidate_id = admission["run_id"], admission["candidate_id"]
        fixed = candidate_lookup.get((run_id, candidate_id))
        require(fixed is not None, f"{group_name} admission references a factor outside the fixed scope: {idea_id}")
        state = group_states[fixed["factor_key"]]
        require(idea_id == f"{run_id}--{candidate_id}",
                f"{group_name} card ID differs from its frozen run and candidate")
        require(admission.get("definition") == fixed["definition"],
                f"{group_name} admission definition differs from the fixed identity: {idea_id}")
        card_path = pool_cards[idea_id]
        card = read_json(card_path)
        require(card.get("id") == idea_id
                and card.get("source", {}).get("run_id") == run_id
                and card.get("source", {}).get("candidate_id") == candidate_id
                and card.get("b_validation_status") == "passed",
                f"{group_name} admission does not have its actual passing card: {card_path}")
        require(state["eligible_for_idea_pool"] is True,
                f"{group_name} program admitted a candidate the B records mark ineligible: {idea_id}")
        require(state["idea_card_path"] == str(card_path),
                f"{group_name} B decision points to a different card path: {idea_id}")
        if group_name == "new_multi":
            require(admission.get("retained_horizons") == state["retained_horizons"],
                    f"new_multi admission A horizons differ: {idea_id}")
            delivery = delivery_by_id[idea_id]
            require(Path(delivery["card_path"]).resolve() == card_path
                    and delivery.get("run_id") == run_id
                    and delivery.get("candidate_id") == candidate_id
                    and delivery.get("retained_horizons") == state["retained_horizons"]
                    and delivery.get("passed_horizons") == state["passed_horizons"],
                    f"new_multi delivery evidence differs from its card or B verdict: {idea_id}")
            require(set(delivery.get("horizon_evidence", {})) == {
                str(horizon) for horizon in state["retained_horizons"]
            }, f"new_multi delivery evidence omits B horizons: {idea_id}")
        match = matches[idea_id]
        card_evidence = {
            "idea_id": idea_id,
            "card_path": str(card_path),
            "admission_receipt": admission["receipt_path"],
            "match_receipt": match["receipt_path"],
            "matches_goal": match["matches_goal"],
            "match_reason": match["reason"],
            "delivery_evidence": delivery_by_id[idea_id] if group_name == "new_multi" else None,
        }
        all_cards.append({**card_evidence, "factor_key": fixed["factor_key"]})
        state.update(card_evidence)
        if match["matches_goal"] is True:
            require(fixed["factor_key"] not in qualified,
                    f"{group_name} has multiple matching cards for one factor identity")
            qualified[fixed["factor_key"]] = state
    qualified_ids = {card["idea_id"] for card in all_cards if card["matches_goal"] is True}
    require(set(recheck_result.get("qualified_idea_ids", [])) == qualified_ids
            and recheck_result.get("qualified_count") == len(qualified_ids),
            f"{group_name} rechecked Goal count differs from versioned matching receipts")
    return {
        "group": group_name,
        "policy": group_result["policy"],
        "engine_root": group_result["engine_root"],
        "goal_root": str(goal_root),
        "idea_pool": str(idea_pool),
        "target_ideas": group_result["target_ideas"],
        "qualified_count": len(qualified),
        "qualified_identities": sorted(qualified),
        "qualified": qualified,
        "cards": all_cards,
        "corrected_goal": recheck_result["corrected_goal"],
        "previous_goal_objective": recheck_manifest["previous_goal_objective"],
        "recheck_manifest": str((recheck_root / "recheck_manifest.json").resolve()),
        "recheck_result": str((recheck_root / "recheck_result.json").resolve()),
        "prior_completion_records": str((goal_root / "completion_records").resolve()),
        "rechecked_completion_records": str(receipts_root),
        "batches": batch_records,
        "run_dirs": [str(path.resolve()) for path in run_dirs],
        "factor_archive_root": str(archive_root),
        "fixed_b_panel": str((experiment_root / "frozen_inputs" / "provenance.json").resolve()),
        "frozen_b_values_sha256": frozen_panel["artifacts"]["values"]["sha256"],
        "frozen_b_universe_sha256": frozen_panel["artifacts"]["universe"]["sha256"],
        "all_candidates": group_states,
    }


def markdown_report(report: dict[str, Any]) -> str:
    checks = report["acceptance"]
    lines = [
        "# FM-v6 固定因子多期限历史对照",
        "",
        f"旧正式Goal只读基线：{report['original_goal_reference']['qualified_card_count']} 张卡；"
        f"A样本：{report['fixed_scope']['candidate_count']} 个因子、{report['fixed_scope']['batch_count']} 批。",
        f"24h重放：{report['groups']['old24']['qualified_count']} 张；"
        f"多期限重放：{report['groups']['new_multi']['qualified_count']} 张。",
        f"同身份保留 {len(report['identity_sets']['retained'])}，新增 {len(report['identity_sets']['new'])}，"
        f"流失 {len(report['identity_sets']['lost'])}，净增 {report['identity_sets']['net_gain']}。",
        f"验收：旧基线复现5张={'通过' if checks['old24_reproduced_five'] else '失败'}；"
        f"原5张因子身份一致={'通过' if checks['old24_reproduced_original_identities'] else '失败'}；"
        f"新规则至少6张={'通过' if checks['new_multi_at_least_six'] else '未达标'}；"
        f"净增至少1={'通过' if checks['net_gain_at_least_one'] else '失败'}。",
        "",
        "两组使用同一份冻结B面板，输入路径和SHA-256记录在下方产物中。完整逐期数值、"
        "模型报告、卡片和Goal收据保存在各自隔离目录；本表按公式、方向和语义版本去重。",
        f"Goal匹配使用原研究目标：{report['groups']['old24']['corrected_goal']['objective']}",
        "",
        "## 身份变化",
        "",
        "| 状态 | 批次 | 因子身份 | 方向 | 24h重放 | A保留期限 | 多期限B通过期限 | 结论或失败原因 |",
        "| --- | ---: | --- | ---: | --- | --- | --- | --- |",
    ]
    factors = report["factor_results"]
    for key, item in sorted(factors.items(), key=lambda pair: (pair[1]["batch_index"], pair[0])):
        lines.append(
            f"| {item['transition']} | {item['batch_index']} | `{item['expanded_expression']}` "
            f"(`{key}`) | {item['direction']} | {item['old24']['qualified']} | "
            f"{','.join(map(str, item['new_multi']['retained_horizons']))} | "
            f"{','.join(map(str, item['new_multi']['passed_horizons']))} | "
            f"{'; '.join(item['new_multi']['reasons']) or '程序准入'} |"
        )
    lines.extend([
        "",
        "## 可复核产物",
        "",
        f"- 24h Goal：`{report['groups']['old24']['goal_root']}`",
        f"- 多期限 Goal：`{report['groups']['new_multi']['goal_root']}`",
        f"- 24h更正目标核验：`{report['groups']['old24']['recheck_result']}`",
        f"- 多期限更正目标核验：`{report['groups']['new_multi']['recheck_result']}`",
        f"- 原错误目标（留痕）：`{report['groups']['old24']['previous_goal_objective']}`",
        f"- 冻结B面板：`{report['groups']['old24']['fixed_b_panel']}`",
        f"- 机器可读明细：`{report['json_path']}`",
        "",
    ])
    return "\n".join(lines)


def compare(args: argparse.Namespace) -> None:
    experiment_root = args.output_root.expanduser().resolve(strict=True)
    scope, review = load_prepared_scope(experiment_root)
    frozen_panel = verify_frozen_panel(experiment_root)
    old24 = collect_group_results(experiment_root, "old24", scope, review, frozen_panel)
    new_multi = collect_group_results(experiment_root, "new_multi", scope, review, frozen_panel)
    old_ids = set(old24["qualified_identities"])
    new_ids = set(new_multi["qualified_identities"])
    retained = sorted(old_ids & new_ids)
    added = sorted(new_ids - old_ids)
    lost = sorted(old_ids - new_ids)
    factors = {candidate["factor_key"]: candidate for candidate in scope["candidates"]}
    factor_results: dict[str, Any] = {}
    for key, candidate in factors.items():
        old = old24["all_candidates"][key]
        new = new_multi["all_candidates"][key]
        if key in retained:
            transition = "保留"
        elif key in added:
            transition = "新增"
        elif key in lost:
            transition = "流失"
        else:
            transition = "未入池"
        factor_results[key] = {
            "transition": transition,
            "batch_index": candidate["batch_index"],
            "candidate_id": candidate["candidate_id"],
            "source_run_id": candidate["source_run_id"],
            "expanded_expression": candidate["identity"]["expanded_expression"],
            "direction": candidate["identity"]["direction"],
            "semantics_version": candidate["identity"]["semantics_version"],
            "old24": {
                "qualified": key in old_ids,
                "validation_status": old["validation_status"],
                "eligible_for_idea_pool": old["eligible_for_idea_pool"],
                "reasons": old["reasons"],
                "passed_horizons": old["passed_horizons"],
                "horizons": old["horizons"],
                "card": {field: old.get(field) for field in (
                    "idea_id", "card_path", "admission_receipt", "match_receipt", "matches_goal"
                ) if field in old},
                "run_path": old["run_path"],
                "evaluation_record_path": old["evaluation_record_path"],
                "validation_record_path": old["validation_record_path"],
                "model_report_path": old["model_report_path"],
            },
            "new_multi": {
                "qualified": key in new_ids,
                "validation_status": new["validation_status"],
                "eligible_for_idea_pool": new["eligible_for_idea_pool"],
                "reasons": new["reasons"],
                "retained_horizons": new["retained_horizons"],
                "passed_horizons": new["passed_horizons"],
                "horizon_results": new["horizon_results"],
                "horizons": new["horizons"],
                "card": {field: new.get(field) for field in (
                    "idea_id", "card_path", "admission_receipt", "match_receipt", "matches_goal",
                    "delivery_evidence"
                ) if field in new},
                "run_path": new["run_path"],
                "evaluation_record_path": new["evaluation_record_path"],
                "validation_record_path": new["validation_record_path"],
                "model_report_path": new["model_report_path"],
            },
        }
    old_count = old24["qualified_count"]
    new_count = new_multi["qualified_count"]
    net_gain = new_count - old_count
    acceptance = {
        "old24_reproduced_five": old_count == EXPECTED_OLD_CARDS,
        "old24_reproduced_original_identities": old_ids == {card["factor_key"] for card in scope["original_cards"]},
        "new_multi_at_least_six": new_count >= EXPECTED_OLD_CARDS + 1,
        "net_gain_at_least_one": net_gain >= 1,
    }
    report = {
        "schema": "fm-v6-horizon-comparison-result/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(acceptance.values()) else "not_met",
        "source_goal": scope["source_goal"],
        "original_goal_reference": {
            "goal_id": scope["source_goal_id"],
            "qualified_card_count": scope["original_card_count"],
            "cards": scope["original_cards"],
        },
        "fixed_scope": {
            "batch_count": scope["batch_count"],
            "candidate_count": scope["candidate_count"],
            "batches": [{key: batch[key] for key in ("batch_index", "source_run_id", "candidate_count", "factor_keys")}
                        for batch in scope["batches"]],
        },
        "groups": {"old24": old24, "new_multi": new_multi},
        "identity_sets": {
            "retained": retained, "new": added, "lost": lost,
            "net_gain": net_gain,
        },
        "acceptance": acceptance,
        "factor_results": factor_results,
        "json_path": str((experiment_root / "comparison.json").resolve()),
    }
    write_new_json(experiment_root / "comparison.json", report)
    markdown = markdown_report(report)
    try:
        with (experiment_root / "comparison.md").open("x", encoding="utf-8") as handle:
            handle.write(markdown)
    except FileExistsError as exc:
        raise ComparisonError(f"refusing to overwrite existing comparison report: {experiment_root / 'comparison.md'}") from exc
    print(json.dumps({
        "status": report["status"],
        "old24_cards": old_count,
        "new_multi_cards": new_count,
        "retained": len(retained), "new": len(added), "lost": len(lost), "net_gain": net_gain,
        "acceptance": acceptance,
        "json": str(experiment_root / "comparison.json"),
        "markdown": str(experiment_root / "comparison.md"),
    }, ensure_ascii=False, indent=2))
    if not all(acceptance.values()):
        raise ComparisonError("card-count acceptance failed; see comparison.md and comparison.json")


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-goal", type=Path, default=DEFAULT_SOURCE_GOAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--baseline-engine-root", type=Path, default=DEFAULT_BASELINE_ENGINE)
    parser.add_argument("--current-engine-root", type=Path, default=REPO_ROOT / "src")


def add_model_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("mimo", "codex"),
                        help="model provider; default: source Goal's saved provider/model/settings")
    parser.add_argument("--model", help="explicit model override; Codex requires this with --reasoning-effort")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"),
                        help="Codex only; required with an explicit Codex model")
    parser.add_argument("--codex-bin", type=Path,
                        help="Codex only; explicit native codex executable path")
    parser.add_argument("--thinking", choices=("enabled", "disabled"),
                        help="MiMo only; default to the source Goal setting when available")
    parser.add_argument("--timeout-seconds", type=int,
                        help="timeout override; defaults to the source Goal setting")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="freeze the 26 A identities and make the one-time A-only review")
    add_shared_arguments(prepare_parser)
    add_model_selection_arguments(prepare_parser)

    run_parser = subparsers.add_parser("run-group", help="run one isolated fixed-batch B comparison group")
    run_parser.add_argument("--group", choices=("old24", "new_multi"), required=True)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    run_parser.add_argument("--engine-root", type=Path, required=True)
    run_parser.add_argument("--baseline-engine-root", type=Path, default=DEFAULT_BASELINE_ENGINE)
    run_parser.add_argument("--current-engine-root", type=Path, default=REPO_ROOT / "src")
    run_parser.add_argument("--resume", action="store_true",
                            help="continue saved batches; complete reports only after a full B numerical checkpoint")

    recheck_parser = subparsers.add_parser(
        "recheck-completion",
        help="recheck Goal matches under the source Goal objective using completed checkpoints",
    )
    recheck_parser.add_argument("--group", choices=("old24", "new_multi"), required=True)
    recheck_parser.add_argument("--output-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    recheck_parser.add_argument("--engine-root", type=Path, required=True)
    recheck_parser.add_argument("--baseline-engine-root", type=Path, default=DEFAULT_BASELINE_ENGINE)
    recheck_parser.add_argument("--current-engine-root", type=Path, default=REPO_ROOT / "src")

    freeze_parser = subparsers.add_parser(
        "freeze-inputs", help="load B market data once and persist the shared read-only panel"
    )
    freeze_parser.add_argument("--output-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    freeze_parser.add_argument("--current-engine-root", type=Path, default=REPO_ROOT / "src")

    compare_parser = subparsers.add_parser(
        "compare", help="validate complete group artifacts and evaluate card identity net gain"
    )
    compare_parser.add_argument("--output-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)

    worker = subparsers.add_parser("_run_worker", help=argparse.SUPPRESS)
    worker.add_argument("--group", choices=("old24", "new_multi"), required=True)
    worker.add_argument("--experiment-root", type=Path, required=True)
    worker.add_argument("--engine-root", type=Path, required=True)
    worker.add_argument("--group-dir", type=Path, required=True)
    worker.add_argument("--frozen-panel-manifest", type=Path, required=True)
    worker.add_argument("--resume", action="store_true")

    recheck_worker = subparsers.add_parser("_recheck_worker", help=argparse.SUPPRESS)
    recheck_worker.add_argument("--group", choices=("old24", "new_multi"), required=True)
    recheck_worker.add_argument("--experiment-root", type=Path, required=True)
    recheck_worker.add_argument("--engine-root", type=Path, required=True)
    recheck_worker.add_argument("--goal-root", type=Path, required=True)
    recheck_worker.add_argument("--recheck-root", type=Path, required=True)

    freeze_worker = subparsers.add_parser("_freeze_worker", help=argparse.SUPPRESS)
    freeze_worker.add_argument("--experiment-root", type=Path, required=True)
    freeze_worker.add_argument("--engine-root", type=Path, required=True)
    freeze_worker.add_argument("--frozen-root", type=Path, required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "freeze-inputs":
        freeze_inputs(args)
    elif args.command == "compare":
        compare(args)
    elif args.command == "run-group":
        run_group(args)
    elif args.command == "recheck-completion":
        recheck_completion(args)
    elif args.command == "_run_worker":
        run_group_worker(args)
    elif args.command == "_freeze_worker":
        freeze_inputs_worker(args)
    elif args.command == "_recheck_worker":
        recheck_completion_worker(args)
    else:
        raise ComparisonError(f"unknown command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except (ComparisonError, subprocess.CalledProcessError) as exc:
        print(f"fm-v6 horizon comparison stopped: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
