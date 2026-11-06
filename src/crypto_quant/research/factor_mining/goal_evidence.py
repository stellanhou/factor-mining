"""Pure quantity, quality, semantic-match, and delivery evidence checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import identifier, number, require, without_hash_metadata
from .records import (RecordStore, load_record_reference, record_reference, write_json)


def quality_evidence(goal, miner, candidate_id: str,
                     retained_horizons: list[int]) -> dict[str, Any]:
    target = goal.quality_target
    require(target is not None, "quality evidence requires a quality goal")
    result = {"passed": False, "target": target, "segment": "A"}
    candidate = miner.candidates[candidate_id]
    if candidate["experiment"] is None:
        return {**result, "reason": "candidate has no predeclared paired experiment"}
    ref = {"record_id": f"{candidate_id}-comparison"}
    record = load_record_reference(miner.store.root, ref)
    comparison = record["data"]
    require(record["kind"] == "experiment_result" and comparison["candidate_id"] == candidate_id
            and comparison["plan"] == candidate["experiment"],
            "quality comparison differs from its candidate's predeclared experiment")
    design = comparison["plan"]["experiment_design"]
    if design["metric"] != target["metric"]:
        return {**result, "comparison_ref": ref, "reason": "paired metric differs from quality target"}
    require(comparison["horizon_hours"] == design["horizon_hours"]
            and comparison["displacement_hours"] == design["displacement_hours"],
            "quality comparison intervals differ from its plan")
    result.update(comparison_ref=ref, control_id=comparison["plan"]["control_id"],
                  experiment_design=design, paired_improvement=comparison["paired_improvement"],
                  paired_ic_change=comparison["paired_ic_change"])
    if (design["horizon_hours"] != target["horizon_hours"]
            or design["displacement_hours"] != target["displacement_hours"]
            or target["horizon_hours"] not in retained_horizons
            or design["min_improvement"] < target["min_improvement"]
            or design["max_ic_loss"] > target["max_ic_loss"]):
        return {**result, "reason": "predeclared experiment does not meet the frozen quality target"}
    improvement, ic = comparison["paired_improvement"], comparison["paired_ic_change"]
    if improvement["status"] != "estimated" or ic["status"] != "estimated":
        return {**result, "reason": "paired uncertainty is unavailable"}
    lower_bounds = []
    for metric in (improvement, ic):
        require(isinstance(metric["ci"], list) and len(metric["ci"]) == 2,
                "estimated quality evidence requires two confidence bounds")
        lower, upper = (number(value, "quality confidence bound") for value in metric["ci"])
        require(lower <= upper, "quality confidence bounds are reversed")
        lower_bounds.append(lower)
    improvement_mean = number(improvement["mean"], "quality paired improvement mean")
    passed = (comparison["route_decision"] == "continue"
              and improvement_mean >= target["min_improvement"] and lower_bounds[0] > 0
              and lower_bounds[1] >= -target["max_ic_loss"])
    return {**result, "passed": passed,
            "reason": "paired mean meets quality target with positive improvement lower bound and acceptable IC loss"
                      if passed else "paired evidence does not meet the quality target"}


def archive_a_cycle(runner, miner) -> None:
    """Save one immutable compact A cycle summary and exact record references."""
    record_id = f"cycle-{runner.state['cycle']:08d}"
    if record_id in {record["id"] for record in runner.research.all()}:
        return
    records = [_read(path) for path in sorted(miner.store.root.glob("*.json"))]
    records = [record for record in records if record["kind"] != "goal_context"]
    records = [load_record_reference(miner.store.root, record_reference(record))
               if record["kind"] == "evaluation" else record for record in records]
    context = runner._cycle_context(record_id, {"run_id": miner.spec.run_id, "records": records})
    run_path = miner.root.resolve().relative_to(runner.root.parent.resolve())
    items = []
    for index, record in enumerate(records):
        data = record["data"]
        candidate_id = data.get("candidate_id")
        if record["kind"] == "candidate":
            candidate_id = data["id"]
        item = {"record_id": record["id"], "kind": record["kind"],
                "candidate_id": candidate_id, "ref": record_reference(record),
                "pointer": f"/records/{index}"}
        if record["kind"] == "evaluation":
            item["factor_archive_ref"] = data["factor_archive"]
        items.append(item)
    completion_path = miner.root / "a-complete.json"
    completion = _read(completion_path) if completion_path.exists() else None
    data = {"run_id": miner.spec.run_id, "context": context,
            "task": runner.state["task"],
            "candidate_ids": [record["data"]["id"] for record in records
                              if record["kind"] == "candidate"],
            "records": {"run_path": str(run_path), "items": items}}
    if completion is not None:
        data["a_completion"] = {key: completion[key] for key in ("completed_rounds", "stop_reason")}
    runner.research.append(record_id, "research_cycle", data)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_goal_evidence(root: str | Path) -> dict[str, Any]:
    """Recompute Goal completion from saved reports, receipts, and delivered cards."""
    root = Path(root)
    saved_goal = _read(root / "goal.json")
    goal = saved_goal["goal"]
    state = RecordStore(root / "events").all()[-1]["data"]
    receipt_records = RecordStore(root / "completion_records").all()
    admissions = [record for record in receipt_records if record["kind"] == "program_admission"]
    match_records = [record for record in receipt_records if record["kind"] == "goal_match"]
    match_by_idea: dict[str, bool] = {}
    for record in match_records:
        matches = record["data"].get("matches", [])
        for match in matches:
            idea_id = match.get("idea_id")
            require(isinstance(idea_id, str) and type(match.get("matches_goal")) is bool,
                    "saved semantic-match report is malformed")
            require(idea_id not in match_by_idea, "duplicate saved semantic-match decision")
            match_by_idea[idea_id] = match["matches_goal"]

    delivery_rows: dict[str, dict[str, Any]] = {}
    delivery_files = sorted((root / "delivery_evidence").glob("admission-*.json"))
    for path in delivery_files:
        data = _read(path)
        for item in data.get("ideas", []):
            idea_id = item.get("idea_id")
            require(isinstance(idea_id, str) and idea_id not in delivery_rows,
                    "saved Goal delivery evidence has duplicate or invalid idea IDs")
            delivery_rows[idea_id] = item

    admitted_ids = {idea["idea_id"] for record in admissions
                    for idea in record["data"].get("ideas", [])}
    gaps: list[str] = []
    if not admitted_ids and not match_records:
        gaps.append("missing semantic-match report")
    for idea_id in sorted(admitted_ids):
        if idea_id not in match_by_idea:
            gaps.append(f"missing semantic-match report for {idea_id}")
    if not admitted_ids and not delivery_files:
        gaps.append("missing actual delivery evidence")
    verified_delivery_ids: set[str] = set()
    for idea_id in sorted(admitted_ids):
        item = delivery_rows.get(idea_id)
        if item is None:
            gaps.append(f"missing actual delivery evidence for {idea_id}")
            continue
        card_path = Path(item["card_path"])
        if not card_path.is_file():
            gaps.append(f"delivered card file is missing for {idea_id}: {card_path}")
        else:
            card = _read(card_path)
            if card.get("id") != idea_id:
                gaps.append(f"delivered card identity differs for {idea_id}: {card_path}")
            else:
                verified_delivery_ids.add(idea_id)

    matched = {idea_id for idea_id, value in match_by_idea.items()
               if value and idea_id in admitted_ids}
    delivered = matched & verified_delivery_ids
    quality_target = goal.get("quality_target")
    quality_qualified: set[str] = set()
    if quality_target is not None:
        delivery_by_id = delivery_rows
        for idea_id in delivered:
            item = delivery_by_id[idea_id]
            evidence = item.get("quality_evidence")
            decision = item.get("admission_evidence", {})
            if (isinstance(evidence, dict) and evidence.get("passed") is True
                    and evidence.get("target") == quality_target
                    and quality_target["horizon_hours"] in item.get("passed_horizons", [])
                    and decision.get("eligible_for_idea_pool") is True):
                quality_qualified.add(idea_id)
        met = bool(quality_qualified)
        if not met:
            gaps.append("quality target is not supported by paired A evidence and same-horizon B admission")
        qualified = quality_qualified
        target = quality_target
    else:
        target_count = goal["target_ideas"]
        met = len(delivered) >= target_count
        if not met:
            gaps.append(f"quantity target unmet: {len(delivered)} of {target_count} ideas have match and delivery evidence")
        qualified = delivered
        target = {"target_ideas": target_count}

    expected_qualified = set(state.get("qualified_ideas", []))
    if not expected_qualified <= matched:
        gaps.append("Goal state includes ideas without a positive semantic-match report")
    if not expected_qualified <= verified_delivery_ids:
        gaps.append("Goal state includes ideas without verified actual delivery evidence")
    complete = met and not gaps
    return {"status": "complete" if complete else "incomplete",
            "objective_met": met, "target": target,
            "qualified_idea_ids": sorted(qualified), "gaps": gaps,
            "evidence": {"program_admission_receipts": [record["id"] for record in admissions],
                         "semantic_match_reports": [record["id"] for record in match_records],
                         "delivery_files": [str(path) for path in delivery_files]}}


def check_native_goal_evidence(miner, goal_spec: dict[str, Any]) -> dict[str, Any]:
    """Recompute a native run's goal from its sealed A evidence and delivered B cards."""
    from .goal import GoalSpec
    from .research_actions import _latest_model_reports, read_b_checkpoint
    gaps: list[str] = []
    a_manifest_ref = None
    b_delivery_ref = str(miner.root / "b-delivery.json")
    candidate_ids: list[str] = []
    eligible_ids: list[str] = []
    qualified: set[str] = set()
    target: dict[str, Any] = {}
    goal = None
    try:
        require(isinstance(goal_spec, dict), "fixed GoalSpec binding must be an object")
        goal = GoalSpec(**goal_spec)
        normalized_goal = goal.as_dict()
        target = (goal.quality_target if goal.quality_target is not None
                  else {"target_ideas": goal.target_ideas})
    except (TypeError, ValueError, KeyError) as exc:
        gaps.append(f"fixed GoalSpec is missing or inconsistent: {exc}")
        normalized_goal = None

    def finish(objective_met: bool = False) -> dict[str, Any]:
        complete = objective_met and not gaps
        return {"status": "complete" if complete else "incomplete",
                "objective_met": complete, "target": target,
                "qualified_idea_ids": sorted(qualified), "gaps": gaps,
                "evidence": {"a_manifest_ref": a_manifest_ref,
                             "b_delivery_ref": b_delivery_ref,
                             "candidate_ids": candidate_ids,
                             "eligible_candidate_ids": eligible_ids}}

    if normalized_goal is None:
        return finish()
    if goal.objective != miner.spec.objective:
        gaps.append("fixed GoalSpec objective differs from the saved research contract")
    if miner.spec.purpose != "research":
        gaps.append("engineering_check runs cannot satisfy a research goal")

    try:
        a_records = miner.store.all()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        gaps.append(f"A records cannot be verified: {exc}")
        return finish()
    a_by_id = {record["id"]: record for record in a_records}
    latest_a_reports = _latest_model_reports(a_records)
    saved_goal = a_by_id.get("goal-spec")
    if (saved_goal is None or saved_goal.get("kind") != "goal_contract"
            or saved_goal.get("data") != normalized_goal):
        gaps.append("fixed GoalSpec differs from the saved A goal contract")

    completion_path = miner.root / "a-complete.json"
    if not completion_path.is_file():
        gaps.append("missing sealed A completion checkpoint")
        return finish()
    try:
        completion = _read(completion_path)
    except (OSError, ValueError, TypeError) as exc:
        gaps.append(f"A completion checkpoint cannot be read: {exc}")
        return finish()
    if not isinstance(completion, dict):
        gaps.append("A completion checkpoint is malformed")
        return finish()
    a_manifest_ref = completion.get("manifest_ref")
    if (completion.get("status") != "complete"
            or completion.get("run_id") != miner.spec.run_id
            or completion.get("semantic_match_ref") != a_manifest_ref
            or completion.get("goal_status") != "awaiting_B"
            or completion.get("objective_met") is not False):
        gaps.append("A completion checkpoint is incomplete or inconsistent with the native seal contract")
    manifest_record = a_by_id.get(a_manifest_ref) if isinstance(a_manifest_ref, str) else None
    if manifest_record is None or manifest_record.get("kind") != "a_manifest":
        gaps.append("A completion references a missing or invalid manifest record")
        return finish()
    manifest = manifest_record["data"]
    if not isinstance(manifest, dict):
        gaps.append("sealed A manifest is malformed")
        return finish()
    try:
        frozen = miner._checked_frozen()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        gaps.append(f"B freeze cannot be verified: {exc}")
        frozen = None
    if frozen is None:
        return finish()
    if frozen.get("a_manifest_ref") != a_manifest_ref:
        gaps.append("B freeze does not reference the exact sealed A manifest")

    candidate_ids = manifest.get("candidate_ids", [])
    retained_ids = manifest.get("retained_ids", [])
    if (not isinstance(candidate_ids, list) or not candidate_ids
            or not all(isinstance(candidate_id, str) for candidate_id in candidate_ids)
            or len(candidate_ids) != len(set(candidate_ids))
            or set(candidate_ids) != {record["data"].get("id") for record in a_records
                                     if record.get("kind") == "candidate"}):
        gaps.append("A manifest candidate set differs from the saved A candidate records")
        candidate_ids = []
    else:
        candidate_ids = list(candidate_ids)
    if (not isinstance(retained_ids, list) or not retained_ids
            or not all(isinstance(candidate_id, str) for candidate_id in retained_ids)
            or len(retained_ids) != len(set(retained_ids))
            or set(retained_ids) != set(frozen.get("candidates", {}))):
        gaps.append("A retained candidate set differs from the frozen B batch")
        retained_ids = []
    if (completion.get("candidate_ids") != candidate_ids
            or completion.get("retained_ids") != retained_ids
            or completion.get("evaluated_ids") != manifest.get("evaluated_ids")):
        gaps.append("A completion checkpoint differs from its manifest")
    if (manifest.get("run_id") != miner.spec.run_id
            or manifest.get("objective") != goal.objective
            or manifest.get("goal_spec") != normalized_goal):
        gaps.append("sealed A manifest differs from the fixed run or GoalSpec")
    declared_a_ids = manifest.get("a_record_ids")
    if (not isinstance(declared_a_ids, list)
            or not all(isinstance(record_id, str) for record_id in declared_a_ids)
            or len(declared_a_ids) != len(set(declared_a_ids))
            or set(declared_a_ids) != set(a_by_id) - {a_manifest_ref}):
        gaps.append("A manifest record references differ from the saved A records")

    evidence_by_candidate = manifest.get("candidate_evidence")
    match_report = manifest.get("semantic_match_report")
    match_rows: dict[str, dict[str, Any]] = {}
    if (not isinstance(match_report, dict) or match_report.get("objective") != goal.objective
            or not isinstance(match_report.get("matches"), list)):
        gaps.append("A manifest is missing its semantic-match report")
    else:
        for row in match_report["matches"]:
            if (not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str)
                    or type(row.get("matches_goal")) is not bool
                    or not isinstance(row.get("reason"), str) or not row["reason"].strip()
                    or not isinstance(row.get("evidence_refs"), list)
                    or not all(isinstance(ref, str) for ref in row["evidence_refs"])
                    or not row["evidence_refs"] or row["candidate_id"] in match_rows):
                gaps.append("A semantic-match report contains a malformed or duplicate candidate row")
                continue
            match_rows[row["candidate_id"]] = row
        if set(match_rows) != set(candidate_ids):
            gaps.append("A semantic-match report does not cover the sealed candidate set")
    if not isinstance(evidence_by_candidate, dict) or set(evidence_by_candidate) != set(candidate_ids):
        gaps.append("A manifest is missing candidate-level program evidence")
        evidence_by_candidate = {}
    evidence_retained = {candidate_id for candidate_id, evidence in evidence_by_candidate.items()
                         if isinstance(evidence, dict) and evidence.get("disposition") == "retain"}
    if evidence_retained != set(retained_ids):
        gaps.append("A retained set differs from the candidate-level research decisions")
    matched_candidate_ids: set[str] = set()
    for candidate_id in candidate_ids:
        evidence = evidence_by_candidate.get(candidate_id)
        match = match_rows.get(candidate_id)
        if not isinstance(evidence, dict) or match is None:
            continue
        program_refs = evidence.get("program_evidence_refs")
        supplied_refs = match.get("evidence_refs")
        if (not isinstance(program_refs, list) or not program_refs
                or not all(isinstance(ref, str) for ref in program_refs)
                or len(program_refs) != len(set(program_refs))
                or not isinstance(supplied_refs, list)
                or not all(isinstance(ref, str) for ref in supplied_refs)
                or len(supplied_refs) != len(set(supplied_refs))
                or not set(program_refs) <= set(supplied_refs)
                or not set(supplied_refs) <= set(a_by_id)
                or not set(program_refs) <= set(a_by_id)):
            gaps.append(f"A semantic-match evidence references are incomplete for {candidate_id}")
            continue
        report_ref, decision_ref = evidence.get("report_ref"), evidence.get("decision_ref")
        if (not isinstance(report_ref, str) or not isinstance(decision_ref, str)
                or report_ref not in program_refs or decision_ref not in program_refs):
            gaps.append(f"A program evidence references omit the final report or decision for {candidate_id}")
            continue
        report = a_by_id.get(report_ref)
        decision = a_by_id.get(decision_ref)
        decision_rows = (decision.get("data", {}).get("candidate_decisions", [])
                         if isinstance(decision, dict) else [])
        candidate_decisions = [row for row in decision_rows
                               if isinstance(row, dict) and row.get("candidate_id") == candidate_id]
        latest_decisions = [record for record in a_records if record.get("kind") == "research_review"
                            and any(isinstance(row, dict) and row.get("candidate_id") == candidate_id
                                    for row in record.get("data", {}).get("candidate_decisions", []))]
        if (report is None or report.get("kind") != "model_report"
                or report.get("data", {}).get("candidate_id") != candidate_id
                or decision is None or decision.get("kind") != "research_review"
                or not isinstance(decision.get("data", {}).get("candidate_decisions"), list)
                or len(candidate_decisions) != 1
                or latest_a_reports.get(candidate_id, {}).get("id") != report_ref
                or not latest_decisions or latest_decisions[-1]["id"] != decision_ref
                or evidence.get("disposition") != candidate_decisions[0].get("disposition")
                or evidence.get("retained_horizons") != candidate_decisions[0].get("retained_horizons")):
            gaps.append(f"A report or decision reference is invalid for {candidate_id}")
            continue
        if evidence.get("semantic_match") != match:
            gaps.append(f"A candidate semantic match differs from the sealed report for {candidate_id}")
            continue
        if match["matches_goal"] and candidate_id in retained_ids:
            matched_candidate_ids.add(candidate_id)

    try:
        checkpoint = read_b_checkpoint(miner)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        gaps.append(f"B numerical checkpoint is missing or inconsistent: {exc}")
        return finish()
    b_candidate_ids = checkpoint["candidate_ids"]
    if b_candidate_ids != retained_ids:
        gaps.append("B numerical checkpoint candidate set differs from the sealed A retained set")

    b_store = RecordStore(miner.root / "b_records", run_root=miner.root.resolve(),
                          archive_root=miner.archive_root)
    try:
        b_records = b_store.all()
        b_by_id = {record["id"]: record for record in b_records}
        latest_b_reports = _latest_model_reports(b_records)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        gaps.append(f"B interpretation records cannot be verified: {exc}")
        return finish()
    validation_path = miner.root / "validation.json"
    delivery_path = miner.root / "b-delivery.json"
    report_path = miner.root / "B-report.md"
    if not validation_path.is_file():
        gaps.append("missing completed B validation")
        return finish()
    try:
        if not report_path.is_file() or not report_path.read_text(encoding="utf-8").strip():
            gaps.append("missing completed B interpretation report")
    except (OSError, ValueError) as exc:
        gaps.append(f"B interpretation report cannot be read: {exc}")
    if not delivery_path.is_file():
        gaps.append("missing formal B delivery receipt")
        return finish()
    try:
        validation = _read(validation_path)
        delivery = _read(delivery_path)
    except (OSError, ValueError, TypeError) as exc:
        gaps.append(f"B validation or delivery receipt cannot be read: {exc}")
        return finish()
    if not isinstance(validation, dict) or not isinstance(delivery, dict):
        gaps.append("B validation or delivery receipt is malformed")
        return finish()
    decisions = checkpoint["decisions"]
    saved_decisions = validation.get("decisions")
    if (validation.get("status") != "complete" or validation.get("pending_report_ids") != []
            or not isinstance(saved_decisions, dict) or set(saved_decisions) != set(b_candidate_ids)):
        gaps.append("B validation is incomplete or does not cover the frozen candidate set")
        saved_decisions = {}
    for candidate_id in b_candidate_ids:
        saved_decision = saved_decisions.get(candidate_id)
        decision = decisions[candidate_id]
        if not isinstance(saved_decision, dict) or {
                key: value for key, value in saved_decision.items() if key != "idea_card"} != decision:
            gaps.append(f"saved B validation differs from the verified numerical decision for {candidate_id}")

    if (delivery.get("status") != "complete"
            or delivery.get("run_id") != miner.spec.run_id
            or delivery.get("a_manifest_ref") != a_manifest_ref
            or delivery.get("candidate_ids") != b_candidate_ids
            or delivery.get("batch_correction_ref") != "batch-correction"
            or delivery.get("validation_ref") != "validation.json"
            or delivery.get("report_file") != "B-report.md"):
        gaps.append("B delivery receipt differs from the sealed A and verified B checkpoints")
    delivery_rows = delivery.get("candidates")
    if not isinstance(delivery_rows, list):
        gaps.append("B delivery receipt is missing candidate delivery rows")
        delivery_rows = []
    rows_by_candidate: dict[str, dict[str, Any]] = {}
    for row in delivery_rows:
        if (not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str)
                or row["candidate_id"] in rows_by_candidate):
            gaps.append("B delivery receipt contains a malformed or duplicate candidate row")
            continue
        rows_by_candidate[row["candidate_id"]] = row
    if set(rows_by_candidate) != set(b_candidate_ids):
        gaps.append("B delivery receipt does not cover the complete frozen candidate set")
    eligible_ids = sorted(candidate_id for candidate_id in b_candidate_ids
                          if decisions[candidate_id]["eligible_for_idea_pool"])
    if delivery.get("eligible_candidate_ids") != eligible_ids:
        gaps.append("B delivery eligible candidate set differs from the verified program decisions")
    candidate_to_idea: dict[str, str] = {}
    for candidate_id in b_candidate_ids:
        row = rows_by_candidate.get(candidate_id)
        if row is None:
            continue
        decision = decisions[candidate_id]
        latest_report = latest_b_reports.get(candidate_id)
        report_ref = row.get("report_ref")
        report = b_by_id.get(report_ref) if isinstance(report_ref, str) else None
        validation_record = b_by_id.get(f"{candidate_id}-validation")
        expected_eligible = decision["eligible_for_idea_pool"]
        if (latest_report is None or latest_report["id"] != report_ref
                or report is None or report.get("kind") != "model_report"
                or report.get("data", {}).get("candidate_id") != candidate_id
                or row.get("validation_ref") != f"{candidate_id}-validation"
                or validation_record is None or validation_record.get("kind") != "validation_result"
                or row.get("eligible_for_idea_pool") is not expected_eligible):
            gaps.append(f"B report or validation reference is invalid for {candidate_id}")
            continue
        base_keys = {"candidate_id", "report_ref", "validation_ref", "eligible_for_idea_pool"}
        if not expected_eligible:
            saved_decision = saved_decisions.get(candidate_id, {})
            if (set(row) != base_keys or not isinstance(saved_decision, dict)
                    or "idea_card" in saved_decision):
                gaps.append(f"ineligible B candidate has a formal card reference: {candidate_id}")
            continue
        expected_keys = base_keys | {"card_path", "card_sha256", "idea_id"}
        saved_decision = saved_decisions.get(candidate_id, {})
        if (set(row) != expected_keys or not isinstance(saved_decision, dict)
                or not isinstance(row.get("card_path"), str)
                or not isinstance(row.get("card_sha256"), str)
                or not isinstance(row.get("idea_id"), str)):
            gaps.append(f"eligible B candidate is missing an exact card delivery reference: {candidate_id}")
            continue
        idea_id = f"{miner.spec.run_id}--{candidate_id}"
        candidate_to_idea[candidate_id] = idea_id
        card_path = Path(row["card_path"])
        validation_card_path = saved_decision.get("idea_card")
        if (not isinstance(validation_card_path, str)
                or Path(validation_card_path).resolve() != card_path.resolve()
                or row.get("idea_id") != idea_id or card_path.name != f"{idea_id}.json"
                or not card_path.is_file()):
            gaps.append(f"formal idea card is missing or has the wrong identity for {candidate_id}")
            continue
        try:
            card_bytes = card_path.read_bytes()
        except OSError as exc:
            gaps.append(f"formal idea card cannot be read for {candidate_id}: {exc}")
            continue
        if hashlib.sha256(card_bytes).hexdigest() != row.get("card_sha256"):
            gaps.append(f"formal idea card hash differs from its B delivery receipt for {candidate_id}")
            continue
        try:
            card = json.loads(card_bytes.decode("utf-8"))
            expected_card = miner._idea_card(
                candidate_id, frozen["candidates"][candidate_id],
                b_by_id[f"{candidate_id}-evaluation"]["data"], report["data"], decision,
                report["created_at"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            gaps.append(f"formal idea card cannot be recomputed for {candidate_id}: {exc}")
            continue
        if (card.get("id") != idea_id
                or without_hash_metadata(card) != without_hash_metadata(expected_card)):
            gaps.append(f"formal idea card differs from the verified B evidence for {candidate_id}")
            continue
        if candidate_id in matched_candidate_ids:
            qualified.add(idea_id)

    if goal.quality_target is not None:
        quality_evidence_by_candidate: dict[str, dict[str, Any]] = {}
        if isinstance(manifest.get("quality_evidence"), dict):
            quality_evidence_by_candidate = manifest["quality_evidence"]
        if set(quality_evidence_by_candidate) != set(retained_ids):
            gaps.append("A manifest quality evidence does not cover every retained candidate")
        quality_qualified: set[str] = set()
        try:
            miner._load_candidates()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            gaps.append(f"A quality candidate definitions cannot be verified: {exc}")
        else:
            for candidate_id in retained_ids:
                retained_horizons = frozen["candidates"][candidate_id]["retained_horizons"]
                try:
                    recomputed = quality_evidence(goal, miner, candidate_id, retained_horizons)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    gaps.append(f"A paired quality evidence cannot be verified for {candidate_id}: {exc}")
                    continue
                if quality_evidence_by_candidate.get(candidate_id) != recomputed:
                    gaps.append(f"saved A quality evidence differs from the recomputed result for {candidate_id}")
                if (candidate_id in matched_candidate_ids and recomputed["passed"]
                        and goal.quality_target["horizon_hours"] in retained_horizons
                        and goal.quality_target["horizon_hours"]
                        in decisions[candidate_id]["passed_horizons"]
                        and candidate_id in candidate_to_idea
                        and candidate_to_idea[candidate_id] in qualified):
                    quality_qualified.add(candidate_to_idea[candidate_id])
        qualified = quality_qualified
        if not qualified:
            gaps.append("quality target lacks recomputed paired A evidence and same-horizon B admission")

    if goal.target_ideas is not None and len(qualified) < goal.target_ideas:
        gaps.append(f"quantity target unmet: {len(qualified)} of {goal.target_ideas} matched ideas have verified B cards")
    if goal.quality_target is None:
        objective_met = len(qualified) >= goal.target_ideas
    else:
        objective_met = bool(qualified)
    return finish(objective_met)


def prepare_goal_review(runner, miner, validation: dict[str, Any]) -> dict[str, Any] | None:
    """Verify admissions, archives, reports, cards, and quality before semantic review."""
    require(validation["status"] == "complete", "Goal admission requires complete B reports")
    ideas = []
    delivery_ideas = []
    quality_eligible_ids = set()
    frozen = miner._checked_frozen()
    receipt_id = f"admission-{runner.state['cycle']:08d}"
    records = {r["id"]: r["data"] for r in runner.receipts.all()}
    b_store = RecordStore(miner.root / "b_records", run_root=miner.root.resolve(),
                          archive_root=miner.archive_root)
    b_records = {record["id"]: record for record in b_store.all()}
    for cid, decision in validation["decisions"].items():
        if not decision["eligible_for_idea_pool"]:
            require("idea_card" not in decision,
                    "ineligible B verdict cannot have an idea card")
            continue
        require("idea_card" in decision, "eligible B verdict is missing its idea card")
        card = _read(Path(decision["idea_card"]))
        item = frozen["candidates"][cid]
        b_evaluation = b_records[f"{cid}-evaluation"]
        b_report = b_records[f"{cid}-report"]
        b_validation = b_records[f"{cid}-validation"]
        require(b_evaluation["kind"] == "evaluation" and b_report["kind"] == "model_report"
                and b_validation["kind"] == "validation_result",
                "Goal admission B evidence has unexpected record kinds")
        evaluation = b_evaluation["data"]
        require(b_report["data"]["candidate_id"] == cid,
                "Goal admission model report belongs to another candidate")
        program = {key: value for key, value in decision.items() if key != "idea_card"}
        saved_program = {key: value for key, value in b_validation["data"].items()
                         if key != "candidate_id"}
        require(b_validation["data"]["candidate_id"] == cid and program == saved_program,
                "Goal admission verdict differs from the saved B program result")
        require(program["decision_source"] == "program"
                and program["admission_scheme"] == miner.spec.admission_scheme,
                "Goal admission verdict has an unexpected source or scheme")
        retained_horizons = item["a_decision"]["retained_horizons"]
        require(retained_horizons == decision["retained_horizons"]
                and len(retained_horizons) == len(set(retained_horizons))
                and all(type(h) is int and h in miner.spec.b_horizons for h in retained_horizons),
                "Goal admission horizons differ from the frozen A decision")
        passed_horizons = decision["passed_horizons"]
        require(len(passed_horizons) == len(set(passed_horizons))
                and all(type(h) is int and h in retained_horizons for h in passed_horizons)
                and set(passed_horizons) & set(retained_horizons)
                and decision["validation_status"] == "passed"
                and decision["eligible_for_idea_pool"],
                "Goal admission has no passed A/B horizon intersection")
        require(evaluation["candidate_id"] == cid and evaluation["segment"] == "B"
                and evaluation["direction"] == item["definition"]["direction"]
                and evaluation["retained_horizons"] == retained_horizons,
                "Goal admission evaluation differs from the frozen candidate")
        horizons = evaluation["horizons"]
        require(set(horizons) == {str(h) for h in retained_horizons},
                "Goal admission evaluations differ from the frozen horizon set")
        tests_by_horizon = {test["horizon_hours"]: test for test in decision["tests"]}
        require(len(tests_by_horizon) == len(decision["tests"])
                and set(tests_by_horizon) == set(retained_horizons)
                and all(test["candidate_id"] == cid for test in tests_by_horizon.values()),
                "Goal admission BH tests differ from the frozen candidate horizons")
        horizon_results = decision["horizon_results"]
        require(set(horizon_results) == {str(h) for h in retained_horizons},
                "Goal admission verdict omits a frozen horizon")
        require(passed_horizons == [h for h in retained_horizons
                                    if horizon_results[str(h)]["validation_status"] == "passed"],
                "Goal admission passed horizons differ from per-horizon verdicts")
        factor_identity = miner._factor_identity(
            item["executed"]["expanded_expression"], item["definition"]["direction"]).as_dict()
        factor_archives = {}
        horizon_evidence = {}
        for horizon in retained_horizons:
            result = horizons[str(horizon)]
            locator = result["factor_archive"]
            require(result["segment"] == "B" and result["direction"] == item["definition"]["direction"]
                    and result["horizon_hours"] == horizon
                    and locator["identity"] == factor_identity
                    and locator["evaluation_key"] == miner._evaluation_key("B", horizon).as_dict(),
                    "Goal admission factor archive differs from formula, direction, or horizon")
            horizon_verdict = horizon_results[str(horizon)]
            require(horizon_verdict["tests"] == [tests_by_horizon[horizon]],
                    "Goal admission horizon tests differ from the batch correction")
            factor_archives[str(horizon)] = locator
            horizon_evidence[str(horizon)] = {
                "summary": result["summary"], "coverage": result["coverage"],
                "verdict": horizon_verdict, "factor_archive": locator}
        a_evaluation_record_id = identifier(item["a_evaluation_ref"]["record_id"])
        a_evaluation = load_record_reference(miner.store.root,
                                             {"record_id": a_evaluation_record_id})
        require(a_evaluation["kind"] == "evaluation"
                and a_evaluation["data"]["candidate_id"] == cid
                and a_evaluation["data"]["factor_archive"] == item["a_evaluation_archive"],
                "Goal admission A evaluation reference differs from candidate")
        a_horizons = a_evaluation["data"]["horizon_comparison"]["horizons"]
        require(set(map(str, retained_horizons)) <= set(a_horizons),
                "Goal admission A evaluation omits a retained horizon")
        a_horizon_evidence = {str(h): a_horizons[str(h)]["summary"]
                              for h in retained_horizons}
        expected_card = miner._idea_card(cid, item, evaluation, b_report["data"],
                                         program, b_report["created_at"])
        require(without_hash_metadata(card) == without_hash_metadata(expected_card),
                "Goal admission card differs from its candidate, evaluation, verdict, or model report")
        ideas.append({"idea_id": card["id"], "candidate_id": cid,
                      "run_id": miner.spec.run_id,
                      "run_path": str(miner.root.resolve().relative_to(runner.root.resolve())),
                      "definition": item["definition"],
                      "a_evaluation_ref": {"record_id": a_evaluation_record_id},
                      "factor_archive_ref": item["a_evaluation_archive"],
                      "retained_horizons": retained_horizons,
                      "a_horizon_evidence": a_horizon_evidence,
                      "a_model_report": item["a_model_report"],
                      "admitted_by_program": True})
        delivery_ideas.append({"idea_id": card["id"], "candidate_id": cid,
                               "run_id": miner.spec.run_id,
                               "card_path": str(Path(decision["idea_card"]).resolve()),
                               "source": card["source"],
                               "executed_formula": item["executed"],
                               "direction": item["definition"]["direction"],
                               "retained_horizons": retained_horizons,
                               "passed_horizons": passed_horizons,
                               "a_evaluation_ref": {"record_id": a_evaluation_record_id},
                               "factor_archive_ref": item["a_evaluation_archive"],
                               "b_evaluation_ref": {"record_id": b_evaluation["id"]},
                               "b_validation_ref": {"record_id": b_validation["id"]},
                               "horizon_evidence": horizon_evidence,
                               "admission_evidence": program})
        if runner.goal.quality_target is not None:
            quality = runner._quality_evidence(miner, cid, retained_horizons)
            ideas[-1]["quality_evidence"] = quality
            delivery_ideas[-1]["quality_evidence"] = quality
            if quality["passed"] and runner.goal.quality_target["horizon_hours"] in passed_horizons:
                quality_eligible_ids.add(card["id"])
    if not ideas:
        return None
    require(len({idea["idea_id"] for idea in ideas}) == len(ideas),
            "Goal admission contains duplicate idea cards")
    delivery_path = runner.root / "delivery_evidence" / f"admission-{runner.state['cycle']:08d}.json"
    delivery_record = {"ideas": delivery_ideas}
    if delivery_path.exists():
        require(_read(delivery_path) == delivery_record,
                "saved Goal delivery evidence differs from the program admission")
    else:
        write_json(delivery_path, delivery_record)
    if receipt_id not in records:
        runner.receipts.append(receipt_id, "program_admission", {"ideas": ideas})
    else:
        require(records[receipt_id] == {"ideas": ideas}, "goal admission evidence changed")
    match_id = f"match-{runner.state['cycle']:08d}"
    ids = {item["idea_id"] for item in ideas}

    return {"ids": ids, "match_id": match_id, "records": records,
            "quality_eligible_ids": quality_eligible_ids}


def check_goal_match(value: dict[str, Any], idea_ids: set[str]) -> None:
    matches = value["matches"]
    require(len(matches) == len(idea_ids) and {match["idea_id"] for match in matches} == idea_ids,
            "review each admitted idea exactly once")
    for match in matches:
        require(type(match["matches_goal"]) is bool, "goal match must be boolean")
        require(isinstance(match["reason"], str) and bool(match["reason"].strip()),
                "goal match reason must be non-empty text")


def save_goal_match(receipts: RecordStore, match_id: str, idea_ids: set[str],
                    value: dict[str, Any]) -> dict[str, Any]:
    """Validate and store semantic-match evidence submitted by an external reviewer."""
    check_goal_match(value, idea_ids)
    records = {record["id"]: record for record in receipts.all()}
    if match_id in records:
        require(records[match_id]["kind"] == "goal_match"
                and records[match_id]["data"] == value,
                "saved semantic-match report differs from submitted report")
    else:
        receipts.append(match_id, "goal_match", value)
    return value
