"""Program-only factor research actions shared with the legacy role workflow."""
from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from crypto_quant.features.factor_expressions import (ExpressionResult, compile_expression,
                                                       evaluate_expression)
from crypto_quant.features.factor_inputs import FactorInputPanel
from .contracts import (candidate, digest, dumps, identifier, model_fields,
                        modification_plan, require, text, without_hash_metadata)
from .action_ledger import ResearchActionLedger
from .landscape import build_factor_landscape
from .evaluation import (RANK_DISPLACEMENT_VERSION, build_labels, compare_experiment,
                         compare_rank_displacement_experiment, correct_batch,
                         evaluate_factor, evaluate_horizon_comparison,
                         evaluate_rank_displacement)
from .factor_archive import FactorArchive, FactorIdentity, EvaluationKey
from .records import InvalidRecordPointer, RecordStore, write_json
from .model import RecoverableToolError
from .reporting import write_research_report
from .agent_evaluator import CANDIDATE_DECISION_SCHEMA, REPORT_SCHEMA
from .agent_optimizer import EXPERIMENT_DESIGN_SCHEMA, MODIFICATION_TASK_SCHEMA
from .research_tools import AResearchEvidenceTools


class FormulaBudgetError(ValueError):
    """A valid formula exceeds its contract; simplifying it is new research."""


_ACTION_ID = {"type": "string", "minLength": 1, "maxLength": 80,
              "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]*$"}
_CANDIDATE_DEFINITION = {"type": "object", "additionalProperties": False,
    "required": ["name", "expression", "meaning", "hypothesis", "direction",
                 "parent_id", "proposal_id", "change_reason"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "expression": {"type": "string", "minLength": 1},
        "meaning": {"type": "string"},
        "hypothesis": {"type": "string", "minLength": 1},
        "direction": {"type": "integer", "enum": [-1, 1]},
        "parent_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "proposal_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "change_reason": {"type": "string", "minLength": 1},
    }}
_EXPERIMENT_PLAN = {"type": "object", "additionalProperties": False,
    "required": ["route_id", "control_id", "evidence_refs", "modification_task",
                 "experiment_design", "restart_of", "new_evidence"],
    "properties": {
        "route_id": {"type": "string", "minLength": 1, "maxLength": 60},
        "control_id": {"type": "string", "minLength": 1},
        "evidence_refs": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
        "modification_task": MODIFICATION_TASK_SCHEMA,
        "experiment_design": EXPERIMENT_DESIGN_SCHEMA,
        "restart_of": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "new_evidence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    }}

_MATCH_REPORT = {"type": "object", "additionalProperties": False,
    "required": ["objective", "matches"],
    "properties": {
        "objective": {"type": "string", "minLength": 1},
        "matches": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["candidate_id", "matches_goal", "reason", "evidence_refs"],
            "properties": {
                "candidate_id": {"type": "string", "minLength": 1, "maxLength": 50},
                "matches_goal": {"type": "boolean"},
                "reason": {"type": "string", "minLength": 1},
                "evidence_refs": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
            }}}}}


def _tool(name: str, description: str, required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "name": name, "description": description,
            "inputSchema": {"type": "object", "required": required,
                            "properties": properties, "additionalProperties": False}}


def _report_order(record: dict[str, Any]) -> tuple[int, str, str]:
    record_id = record["id"]
    marker = "-report-revision-"
    if marker in record_id:
        suffix = record_id.rsplit(marker, 1)[1]
        revision = int(suffix) if suffix.isdigit() else 0
    else:
        revision = 0
    return revision, record.get("created_at", ""), record_id


def _latest_model_reports(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["kind"] != "model_report":
            continue
        candidate_id = record["data"]["candidate_id"]
        prior = latest.get(candidate_id)
        if prior is None or _report_order(record) > _report_order(prior):
            latest[candidate_id] = record
    return latest


A_TOOL_SCHEMAS = [
    _tool("get_research_status", "Return A-only research status; never inspect or disclose B records.", [], {}),
    _tool("get_a_landscape", "Return the current A-only factor landscape and evidence references.", [], {}),
    _tool("register_experiment", "Predeclare a controlled A experiment before its child candidate is evaluated.",
          ["action_id", "plan"], {"action_id": _ACTION_ID, "plan": _EXPERIMENT_PLAN}),
    _tool("resolve_experiment", "Pause or stop a predeclared A route that has no pending calculation.",
          ["action_id", "route_id", "status", "reason"], {
              "action_id": _ACTION_ID, "route_id": {"type": "string", "minLength": 1, "maxLength": 60},
              "status": {"type": "string", "enum": ["paused", "stopped"]},
              "reason": {"type": "string", "minLength": 1}}),
    _tool("evaluate_candidate", "Register and evaluate one A candidate with the fixed A panel. "
          "The result includes exact evidence_pointers for its compact evaluation and comparison fields; "
          "use those record IDs and JSON pointers when reading source evidence.",
          ["action_id", "candidate_id", "definition", "experiment_id"], {
              "action_id": _ACTION_ID, "candidate_id": {"type": "string", "minLength": 1, "maxLength": 50},
              "definition": _CANDIDATE_DEFINITION,
              "experiment_id": {"anyOf": [{"type": "string", "maxLength": 60}, {"type": "null"}]}}),
    _tool("repair_candidate", "Repair a saved formula error using a new expression supplied by A.",
          ["action_id", "candidate_id", "program_error", "repair_expression", "reason"], {
              "action_id": _ACTION_ID, "candidate_id": {"type": "string", "minLength": 1, "maxLength": 50},
              "program_error": {"type": "string", "minLength": 1},
              "repair_expression": {"type": "string", "minLength": 1},
              "reason": {"type": "string", "minLength": 1}}),
    _tool("save_research_decision", "Save an A interpretation and the existing candidate decision contract.",
          ["action_id", "candidate_id", "decision", "narrative"], {
              "action_id": _ACTION_ID, "candidate_id": {"type": "string", "minLength": 1, "maxLength": 50},
              "decision": CANDIDATE_DECISION_SCHEMA, "narrative": REPORT_SCHEMA}),
    _tool("finish_a_research", "Validate all A evidence and seal a complete immutable manifest.",
          ["action_id", "semantic_match_report"], {
              "action_id": _ACTION_ID, "semantic_match_report": _MATCH_REPORT}),
]

B_TOOL_SCHEMAS = [
    _tool("get_research_status", "Return frozen-batch and B-processing status for the fixed run.", [], {}),
    _tool("read_b_result", "Read the saved decision, numerical summaries and exact B record references for one frozen candidate; never reload market data. Use this after numerical completion before saving interpretation.",
          ["candidate_id"], {"candidate_id": {"type": "string", "minLength": 1}}),
    _tool("validate_frozen_batch", "Run the single numerical validation for the fixed frozen batch.",
          ["action_id"], {"action_id": _ACTION_ID}),
    _tool("read_b_evidence", "Read one saved B numerical or interpretation record by ID and JSON pointer.",
          ["record_id", "pointer", "offset", "limit"], {
              "record_id": {"type": "string", "minLength": 1},
              "pointer": {"type": "string"}, "offset": {"type": "integer", "minimum": 0},
              "limit": {"type": "integer", "minimum": 1}}),
    _tool("save_b_interpretation", "Save B interpretation only after a complete numerical checkpoint.",
          ["action_id", "candidate_id", "narrative"], {
              "action_id": _ACTION_ID, "candidate_id": {"type": "string", "minLength": 1, "maxLength": 50},
              "narrative": REPORT_SCHEMA}),
    _tool("finalize_b_delivery", "Verify B evidence and save the final report and eligible cards.",
          ["action_id"], {"action_id": _ACTION_ID}),
]

COORDINATOR_TOOL_SCHEMAS = [
    _tool("prepare_a_task", "Return the fixed objective and current A-only evidence snapshot.", [], {}),
    _tool("get_research_status", "Return the status visible to the configured research role.", [], {}),
    _tool("read_a_result", "Read one current-run A candidate's complete latest decision and report, with exact source references; a turn ending does not prove A is sealed.",
          ["candidate_id"], {"candidate_id": {"type": "string", "minLength": 1}}),
    _tool("read_a_evidence", "Read a bounded current-run A record by exact ID and JSON pointer. No paths or B/C/history records. Narrow the pointer if the result exceeds 16 KiB.",
          ["record_id", "pointer", "offset", "limit"], {
              "record_id": {"type": "string", "minLength": 1}, "pointer": {"type": "string"},
              "offset": {"type": "integer", "minimum": 0},
              "limit": {"type": "integer", "minimum": 1}}),
    _tool("read_b_result", "Read a frozen B candidate's saved decision, numerical summaries and interpretation without loading market data; requires a complete checkpoint.",
          ["candidate_id"], {"candidate_id": {"type": "string", "minLength": 1}}),
    _tool("read_b_evidence", "Read a saved B record by exact ID and narrow JSON pointer, at most 16 KiB; never reload B market data.",
          ["record_id", "pointer", "offset", "limit"], {
              "record_id": {"type": "string", "minLength": 1}, "pointer": {"type": "string"},
              "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1}}),
    _tool("freeze_a_manifest", "Freeze every retained candidate in the completed A manifest.",
          ["action_id", "manifest_ref"], {"action_id": _ACTION_ID,
                                             "manifest_ref": {"type": "string", "minLength": 1}}),
    _tool("check_goal_evidence", "Recompute the saved quantity or quality goal evidence.", [], {}),
]

_A_ACTION_NAMES = {
    "register_experiment", "resolve_experiment", "evaluate_candidate",
    "repair_candidate", "save_research_decision", "finish_a_research",
}
_NATIVE_A_EVIDENCE_RESPONSE_BYTES = 256 * 1024


def compile_factor_formula(expression: str, spec):
    if len(expression) > spec.max_formula_nodes * 100:
        raise FormulaBudgetError("formula exceeds complexity budget")
    compiled = compile_expression(expression)
    if sum(1 for _ in ast.walk(compiled.tree)) > spec.max_formula_nodes:
        raise FormulaBudgetError("formula exceeds node budget")
    if compiled.lookback_hours > spec.max_lookback_hours:
        raise FormulaBudgetError("formula exceeds declared history budget")
    return compiled


def archive_factor_values(archive_root: Path, run_root: Path, spec, candidate_id: str,
                          definition: dict[str, Any], result, *, action_id: str | None = None) -> dict[str, Any]:
    native_bound = (Path(run_root) / "a-action-input-binding.json").is_file()
    saved_goal = (Path(run_root) / "a_records" / "goal-spec.json").is_file()
    require(not (native_bound or saved_goal) or action_id is not None,
            "native A value archives require an action_id")
    identity = FactorIdentity(result.definition["expanded_expression"],
                              str(definition["direction"]), "expr-v1")
    archive = FactorArchive.open_for(archive_root, identity)
    data_version = f"{spec.run_id}/A"
    value_set = archive.append_value_set(
        data_version, "factor-eval-v1", result.values.to_csv().encode("utf-8"),
        provenance={"run_id": spec.run_id, "candidate_id": candidate_id, "segment": "A",
                    "definition": definition, "executed_expression": result.definition,
                    **({"action_id": action_id} if action_id is not None else {})})
    return {"root": os.path.relpath(Path(archive_root).resolve(), Path(run_root).resolve()),
            "identity": identity.as_dict(), "data_version": data_version,
            "computation_semantics": "factor-eval-v1",
            "value_set_id": value_set["value_set_id"], "rows": len(result.values)}


def evaluate_a_candidate(values: pd.Series, labels_by_horizon: dict[int, pd.DataFrame],
                         spec, direction: int, *, evaluate_fn=None, horizon_fn=None,
                         stop_check: Callable[[], Any] | None = None) -> dict[str, Any]:
    require(set(labels_by_horizon) == {1, 4, 24},
            "A evaluation requires the fixed 1h, 4h and 24h labels")
    if stop_check is not None:
        stop_check()
    if evaluate_fn is None:
        report = evaluate_factor(values, labels_by_horizon[24], spec, "A", direction,
                                 stop_check=stop_check)
    else:
        report = evaluate_fn(values, labels_by_horizon[24], spec, "A", direction)
    if horizon_fn is None:
        report["horizon_comparison"] = evaluate_horizon_comparison(
            values, labels_by_horizon, spec, direction, stop_check=stop_check)
    else:
        report["horizon_comparison"] = horizon_fn(values, labels_by_horizon, spec, direction)
    return report


def compare_candidates(trial_values: pd.Series, control_values: pd.Series,
                       panel: FactorInputPanel, labels: pd.DataFrame, spec,
                       trial_direction: int, control_direction: int,
                       plan: dict[str, Any], rank_displacement_version: str | None,
                       *, evaluate_fn=None, compare_fn=None, rank_compare_fn=None,
                       build_labels_fn=None,
                       stop_check: Callable[[], Any] | None = None) -> dict[str, Any]:
    if stop_check is not None:
        stop_check()
    design = plan["experiment_design"]
    if design["metric"] == "rank_displacement":
        require(rank_displacement_version == RANK_DISPLACEMENT_VERSION,
                "rank-displacement experiments require the declared evaluator definition")
        require(trial_direction == control_direction,
                "rank-displacement comparisons keep the frozen direction")
        comparison_labels = (build_labels_fn or build_labels)(
            panel, spec, "A", horizon_hours=design["horizon_hours"])
        if rank_compare_fn is None:
            return compare_rank_displacement_experiment(
                trial_values, control_values, comparison_labels, spec,
                trial_direction, plan, universe=panel.universe, stop_check=stop_check)
        return rank_compare_fn(trial_values, control_values, comparison_labels, spec,
                               trial_direction, plan, universe=panel.universe)
    common = trial_values.notna() & control_values.notna()
    if evaluate_fn is None:
        trial = evaluate_factor(trial_values.where(common), labels, spec, "A", trial_direction,
                                stop_check=stop_check)
        control = evaluate_factor(control_values.where(common), labels, spec, "A", control_direction,
                                  stop_check=stop_check)
    else:
        trial = evaluate_fn(trial_values.where(common), labels, spec, "A", trial_direction)
        control = evaluate_fn(control_values.where(common), labels, spec, "A", control_direction)
    if compare_fn is None:
        comparison = compare_experiment(trial, control, plan, spec, stop_check=stop_check)
    else:
        comparison = compare_fn(trial, control, plan, spec)
    if stop_check is not None:
        stop_check()
    comparison["common_asset_observations"] = int(common.sum())
    comparison["trial_on_common_assets"] = trial
    comparison["control_on_common_assets"] = control
    return comparison


def read_b_checkpoint(miner) -> dict[str, Any]:
    """Verify and summarize the immutable B numerical checkpoint without loading data."""
    checkpoint_path = miner.root / "b-numerical-complete.json"
    marker_path = miner.root / "b-access-started.json"
    require(checkpoint_path.is_file(), "B numerical checkpoint is not complete")
    require(marker_path.is_file(), "B access marker is missing")
    frozen = miner._checked_frozen()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    candidate_ids = list(frozen["candidates"])
    expected_checkpoint = {
        "candidate_ids": candidate_ids,
        "retained_horizons": {cid: frozen["candidates"][cid]["retained_horizons"]
                              for cid in candidate_ids},
    }
    require(checkpoint == expected_checkpoint,
            "B numerical checkpoint differs from the frozen candidate batch")
    require(marker.get("candidate_ids") == candidate_ids,
            "B access marker differs from the frozen candidate batch")
    frozen_membership = miner.root / "B-membership-frozen.csv"
    b_universe = miner.root / "B-universe.csv"
    require(frozen_membership.is_file() and b_universe.is_file()
            and frozen_membership.read_bytes() == b_universe.read_bytes(),
            "B membership differs from its frozen validation-universe checkpoint")
    store = RecordStore(miner.root / "b_records", run_root=miner.root.resolve(),
                        archive_root=miner.archive_root)
    records = store.all()
    evaluations = {record["data"]["candidate_id"]: record
                   for record in records if record["kind"] == "evaluation"}
    validations = {record["data"]["candidate_id"]: record
                   for record in records if record["kind"] == "validation_result"}
    corrections = [record for record in records if record["kind"] == "multiple_testing"]
    require(set(evaluations) == set(validations) == set(candidate_ids)
            and len(evaluations) == len(validations) == len(candidate_ids)
            and len(corrections) == 1,
            "B numerical records are incomplete or repeated")
    for cid in candidate_ids:
        item = frozen["candidates"][cid]
        data = evaluations[cid]["data"]
        require(evaluations[cid]["id"] == f"{cid}-evaluation"
                and data["segment"] == "B"
                and data["direction"] == item["definition"]["direction"]
                and data["retained_horizons"] == item["retained_horizons"]
                and set(data["horizons"]) == {str(h) for h in item["retained_horizons"]},
                "B evaluation differs from its frozen formula, direction or horizon set")
        for horizon in item["retained_horizons"]:
            horizon_data = data["horizons"][str(horizon)]
            locator = horizon_data["factor_archive"]
            identity = miner._factor_identity(
                item["executed"]["expanded_expression"], item["definition"]["direction"]).as_dict()
            key = miner._evaluation_key("B", horizon)
            require(horizon_data["summary"] and horizon_data["coverage"]
                    and locator["identity"] == identity
                    and locator["evaluation_key"] == key.as_dict(),
                    "B archived evaluation differs from its frozen formula or horizon")
            archive = FactorArchive.open_existing(miner.archive_root, FactorIdentity(**identity))
            archived = archive.get_evaluation(key)
            archived_report = archived["payload"]
            provenance = [source for source in archived["provenance"]
                          if source.get("run_id") == miner.spec.run_id
                          and source.get("candidate_id") == cid
                          and source.get("segment") == "B"
                          and source.get("expanded_expression") == identity["expanded_expression"]
                          and source.get("direction") == identity["direction"]]
            require(archived["evaluation_id"] == locator["evaluation_id"]
                    and archived["value_set_id"] == locator["value_set_id"]
                    and archived_report["segment"] == "B"
                    and archived_report["direction"] == item["definition"]["direction"]
                    and archived_report["horizon_hours"] == horizon
                    and archived_report["summary"] == horizon_data["summary"]
                    and archived_report["coverage"] == horizon_data["coverage"]
                    and provenance,
                    "B evaluation archive payload or provenance differs from its record")
        displacement_version = item.get("rank_displacement_version")
        if displacement_version is None:
            require(miner.rank_displacement_version is None
                    and "rank_displacement" not in data
                    and "rank_displacement_archive" not in data,
                    "legacy frozen B cannot receive new rank-displacement diagnostics")
        else:
            locator = data.get("rank_displacement_archive")
            displacement = data.get("rank_displacement")
            expected_key = EvaluationKey(
                f"{miner.spec.run_id}/B", dumps(miner.spec.as_dict()),
                displacement_version, "B", "rank-displacement")
            require(displacement_version == miner.rank_displacement_version
                    and displacement_version == RANK_DISPLACEMENT_VERSION
                    and isinstance(locator, dict) and isinstance(displacement, dict)
                    and locator["identity"] == identity
                    and locator["evaluation_key"] == expected_key.as_dict(),
                    "B rank-displacement archive differs from the frozen definition")
            displacement_archive = FactorArchive.open_existing(
                miner.archive_root, FactorIdentity(**identity))
            archived_d = displacement_archive.get_evaluation(expected_key)
            d_provenance = [source for source in archived_d["provenance"]
                            if source.get("run_id") == miner.spec.run_id
                            and source.get("candidate_id") == cid
                            and source.get("segment") == "B"
                            and source.get("definition_version") == displacement_version]
            require(archived_d["evaluation_id"] == locator["evaluation_id"]
                    and archived_d["value_set_id"] == locator["value_set_id"]
                    and miner._rank_displacement_summary(archived_d["payload"])
                    == miner._rank_displacement_summary(displacement)
                    and d_provenance,
                    "B rank-displacement payload or provenance differs from its record")
    from .workflow import b_candidate_decision

    reports = {cid: {key: value for key, value in evaluations[cid]["data"].items()
                     if key != "candidate_id"} for cid in candidate_ids}
    correction = correct_batch(reports, miner.spec)
    require(corrections[0]["id"] == "batch-correction"
            and corrections[0]["data"] == correction,
            "saved B batch correction differs from the fixed family recomputation")
    decisions = {}
    for cid in candidate_ids:
        expected = b_candidate_decision(cid, reports[cid], correction, miner.spec)
        saved = validations[cid]
        require(saved["id"] == f"{cid}-validation"
                and saved["data"] == {"candidate_id": cid, **expected},
                "saved B program decision differs from the numerical evidence")
        decisions[cid] = expected
    validation_path = miner.root / "validation.json"
    if validation_path.is_file():
        completed = json.loads(validation_path.read_text(encoding="utf-8"))
        prior_decisions = {cid: {key: value for key, value in decision.items()
                                 if key != "idea_card"}
                           for cid, decision in completed["decisions"].items()}
        require(prior_decisions == decisions
                and completed["batch_correction"] == correction,
                "completed B delivery differs from its immutable numerical evidence")
    reports_by_candidate = {record["data"]["candidate_id"]
                            for record in records if record["kind"] == "model_report"}
    return {"status": "numerical_complete", "candidate_ids": candidate_ids,
            "retained_horizons": expected_checkpoint["retained_horizons"],
            "batch_correction": correction, "decisions": decisions,
            "pending_report_ids": sorted(set(candidate_ids) - reports_by_candidate)}


class ResearchActions:
    """Program operations with no access to role-model request methods."""

    def __init__(self, miner, *, a_panel: FactorInputPanel | None = None,
                 b_panel_loader=None, validation_universe: pd.Series | None = None,
                 idea_pool: Path | None = None, goal_spec: dict[str, Any] | None = None,
                 goal_root: Path | None = None,
                 history_index=(), allowed_history_record_ids=None, history_reader=None):
        self.miner = miner
        self.ledger = ResearchActionLedger(miner.root)
        self.a_panel = a_panel
        self.b_panel_loader = b_panel_loader
        self.validation_universe = validation_universe
        self.idea_pool = None if idea_pool is None else Path(idea_pool)
        self.goal_root = None if goal_root is None else Path(goal_root)
        self.history_index = tuple(history_index)
        self.allowed_history_record_ids = set(allowed_history_record_ids or set())
        self.history_reader = history_reader
        self.goal_spec = None
        if goal_spec is not None:
            from .goal import GoalSpec

            require(isinstance(goal_spec, dict), "fixed GoalSpec binding must be an object")
            self.goal_spec = GoalSpec(**goal_spec).as_dict()
            require(self.goal_spec["objective"] == miner.spec.objective,
                    "GoalSpec objective differs from the saved research contract")
            goal_record = miner.store.root / "goal-spec.json"
            if goal_record.exists():
                saved = miner.store._load(goal_record)
                require(saved["kind"] == "goal_contract" and saved["data"] == self.goal_spec,
                        "fixed GoalSpec binding differs from the saved A goal contract")

    def _run_action(self, action_id: str, name: str, parameters: dict[str, Any],
                    operation, recover=None, preflight=None):
        return self.ledger.run(action_id, name, parameters, operation,
                               recover=recover, preflight=preflight)

    def _require_no_pending_a_actions(self, action_id: str) -> None:
        pending = [row for row in self.ledger.pending_actions()
                   if row["name"] in _A_ACTION_NAMES and row["action_id"] != action_id]
        require(not pending,
                "cannot seal or freeze A while earlier A actions lack committed receipts; "
                "replay each original action_id to reconcile its saved evidence first: "
                + ", ".join(f"{row['name']}:{row['action_id']}" for row in pending))

    def tool_definitions(self, role: str) -> list[dict[str, Any]]:
        """Return the static allowlist for one service-bound native role."""
        require(role in {"coordinator", "A", "B"}, "unknown fixed research role")
        if role == "A":
            return [*self._a_evidence_tools().tool_definitions(), *copy.deepcopy(A_TOOL_SCHEMAS)]
        if role == "B":
            return copy.deepcopy(B_TOOL_SCHEMAS)
        return copy.deepcopy(COORDINATOR_TOOL_SCHEMAS)

    def handle(self, role: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch only actions declared for the configured role."""
        require(role in {"coordinator", "A", "B"}, "unknown fixed research role")
        require(isinstance(arguments, dict), "research tool arguments must be an object")
        definitions = self.tool_definitions(role)
        definition = next((item for item in definitions if item["name"] == name), None)
        require(definition is not None, f"tool is not available to fixed role {role}: {name}")
        schema = definition["inputSchema"]
        require(set(arguments) == set(schema.get("required", [])),
                f"tool arguments do not match the fixed {role} schema for {name}")

        if role == "A" and name in {"search_a_evidence", "read_a_evidence"}:
            return self._a_evidence_tools().handle(name, arguments)
        if name == "get_research_status":
            return self.get_research_status_for(role)
        if role == "coordinator" and name == "read_a_result":
            return self.read_a_result(**arguments)
        if role == "coordinator" and name == "read_a_evidence":
            return self._coordinator_a_evidence_tools().read(**arguments)
        if name == "read_b_result":
            return self.read_b_result(**arguments)
        if name == "get_a_landscape":
            return self.get_a_landscape()
        if name == "prepare_a_task":
            return self.prepare_a_task()
        if name == "register_experiment":
            return self.register_experiment(**arguments)
        if name == "resolve_experiment":
            return self.resolve_experiment(**arguments)
        if name == "evaluate_candidate":
            return self.evaluate_candidate(**arguments)
        if name == "repair_candidate":
            return self.repair_candidate(**arguments)
        if name == "save_research_decision":
            return self.save_research_decision(**arguments)
        if name == "finish_a_research":
            return self.finish_a_research(**arguments)
        if name == "freeze_a_manifest":
            return self.freeze_a_manifest(**arguments)
        if name == "validate_frozen_batch":
            return self.validate_frozen_batch(**arguments)
        if name == "read_b_evidence":
            return self.read_b_evidence(**arguments)
        if name == "save_b_interpretation":
            return self.save_b_interpretation(**arguments)
        if name == "finalize_b_delivery":
            return self.finalize_b_delivery(**arguments)
        if name == "check_goal_evidence":
            return self.check_goal_evidence()
        raise ValueError(f"tool is not available to fixed role {role}: {name}")

    def _commit(self, action_id: str, operation):
        return self.ledger.commit(action_id, operation)

    def _require_legacy_writer_unbound(self) -> None:
        require(self.goal_spec is None
                and "goal-spec" not in self._a_record_map()
                and not (self.miner.root / "a-action-input-binding.json").exists()
                and "a-manifest" not in self._a_record_map(),
                "legacy writer is disabled for a native action-bound or sealed run")

    def _require_a_open(self) -> None:
        self.ledger.require_active()
        require(not (self.miner.root / "a-complete.json").exists(),
                "A is sealed; program actions cannot reopen or modify the completed manifest")
        require("a-manifest" not in self._a_record_map(),
                "A manifest commit is pending or sealed; only its original completion action may recover it")
        require(not (self.miner.root / "frozen_batch.json").exists(),
                "A evidence cannot change after B is frozen")

    def _require_goal_spec(self) -> dict[str, Any]:
        require(self.goal_spec is not None,
                "native research actions require a fixed GoalSpec binding")
        return copy.deepcopy(self.goal_spec)

    @staticmethod
    def _panel_fingerprint(panel: FactorInputPanel) -> str:
        digest_state = hashlib.sha256()
        digest_state.update(pd.util.hash_pandas_object(panel.values, index=True).values.tobytes())
        digest_state.update(pd.util.hash_pandas_object(panel.universe, index=True).values.tobytes())
        digest_state.update(dumps(panel.diagnostics).encode("utf-8"))
        return digest_state.hexdigest()

    def _a_records(self) -> list[dict[str, Any]]:
        return self.miner.store.all()

    def _a_record_map(self) -> dict[str, dict[str, Any]]:
        return {record["id"]: record for record in self._a_records()}

    def _ensure_a_initialized(self, action_id: str) -> None:
        self._require_a_open()
        require(self.a_panel is not None, "A panel is not fixed in the research service binding")
        self._require_goal_spec()
        self.miner._validate_panel(self.a_panel, "A")
        fingerprint = self._panel_fingerprint(self.a_panel)
        source_contract_sha256 = hashlib.sha256(
            (self.miner.root / "contract.json").read_bytes()).hexdigest()
        binding_path = self.miner.root / "a-action-input-binding.json"
        universe_path = self.miner.root / "A-universe.csv"
        universe_csv = self.a_panel.universe.rename("eligible").to_csv()
        binding = {"run_id": self.miner.spec.run_id,
                   "contract_sha256": digest(self.miner.spec.as_dict()),
                   "panel_sha256": fingerprint}
        if binding_path.exists():
            require(json.loads(binding_path.read_text(encoding="utf-8")) == binding,
                    "fixed A panel or research contract differs from its saved action binding")
            require(universe_path.is_file() and universe_path.read_text(encoding="utf-8") == universe_csv,
                    "A membership differs from its saved action binding")
            return
        records = self._a_records()
        if records:
            existing = {record["id"]: record for record in records}
            environment_records = [record for record in existing.values()
                                   if record["kind"] == "a_environment"]
            allowed_record_ids = {"inputs", "goal-spec"}
            if environment_records:
                require(len(environment_records) == 1,
                        "A run has multiple saved environment records")
                environment = environment_records[0]
                data = environment["data"]
                require(isinstance(data, dict)
                        and set(data) == {
                            "source_segment", "research_id", "contract_sha256",
                            "panel_sha256", "field_catalog", "operators",
                            "templates", "data_coverage",
                        }
                        and data["source_segment"] == "A"
                        and data["research_id"] == self.miner.spec.run_id
                        and data["contract_sha256"] == source_contract_sha256
                        and data["panel_sha256"] == fingerprint
                        and environment["id"] == f"a-environment-{digest(data)}",
                        "saved A environment record differs from the fixed contract or panel")
                allowed_record_ids.add(environment["id"])
            require(set(existing) <= allowed_record_ids,
                    "cannot attach native actions to an A run with unrelated or completed records")
            require("goal-spec" in existing
                    and existing["goal-spec"]["data"] == self.goal_spec,
                    "fixed GoalSpec binding differs from the saved A goal contract")
            if "inputs" in existing:
                require(existing["inputs"]["data"].get("source_fingerprint") == fingerprint,
                        "fixed A panel differs from its saved action input record")
                require(universe_path.is_file()
                        and universe_path.read_text(encoding="utf-8") == universe_csv,
                        "A membership differs from its saved action binding")
                self._commit(action_id, lambda: write_json(binding_path, binding))
                return
            initialization_record_ids = {"goal-spec"} | {
                record["id"] for record in environment_records}
            require(set(existing) == initialization_record_ids,
                    "native initialization requires either a goal-only binding or a complete A input record")

        def persist():
            from .workflow import _panel_metadata

            if universe_path.exists():
                require(universe_path.read_text(encoding="utf-8") == universe_csv,
                        "A membership differs during action-service initialization")
            else:
                with universe_path.open("x", encoding="utf-8") as handle:
                    handle.write(universe_csv)
            records_now = self._a_record_map()
            if "inputs" not in records_now:
                self.miner.store.append("inputs", "data_provenance", {
                    "panel": _panel_metadata(self.a_panel),
                    "catalog": self.a_panel.ideation_context(),
                    "source_fingerprint": fingerprint})
            if "goal-spec" not in records_now:
                self.miner.store.append("goal-spec", "goal_contract", self.goal_spec)
            write_json(binding_path, binding)

        self._commit(action_id, persist)

    def _load_candidates(self) -> None:
        self.miner._load_candidates()
        records = self._a_records()
        by_id = {record["id"]: record for record in records}
        for candidate_id, item in self.miner.candidates.items():
            duplicate = by_id.get(f"{candidate_id}-duplicate")
            if duplicate is not None:
                item["duplicate_of"] = duplicate["data"]["duplicate_of"]
            comparison = by_id.get(f"{candidate_id}-comparison")
            if comparison is not None:
                item["comparison"] = comparison["data"]
            attempts = [record["data"] for record in records
                        if record["kind"] == "calculation_attempt"
                        and record["data"].get("candidate_id") == candidate_id]
            if attempts:
                item["calculation_attempts"] = sorted(attempts, key=lambda data: data["attempt"])
        self.miner.routes = {}
        self.miner.proposals = {}
        for record in records:
            if record["kind"] == "modification_experiment":
                data = record["data"]
                plan = data["plan"]
                route = {"plan": plan, "attempts": data["attempts"],
                         "decision": data["decision"]}
                self.miner.routes[plan["route_id"]] = route
            elif record["kind"] == "experiment_result":
                route_id = record["data"]["plan"]["route_id"]
                if route_id in self.miner.routes:
                    self.miner.routes[route_id]["decision"] = record["data"]["route_decision"]
            elif record["kind"] == "route_resolution":
                data = record["data"]
                if data["route_id"] in self.miner.routes:
                    self.miner.routes[data["route_id"]]["decision"] = data["status"]

    def _a_evidence_tools(self) -> AResearchEvidenceTools:
        from .research_tools import _require_source_a_only

        records = self._a_records()
        for record in records:
            _require_source_a_only(record)
        return AResearchEvidenceTools(
            self.miner.store, {record["id"] for record in records},
            history_index=self.history_index,
            allowed_history_record_ids=self.allowed_history_record_ids,
            history_reader=self.history_reader,
            max_single_output_bytes=_NATIVE_A_EVIDENCE_RESPONSE_BYTES)

    def search_a_evidence(self, *, query: str, limit: int) -> dict[str, Any]:
        return self._a_evidence_tools().search(query=query, limit=limit)

    def _coordinator_a_evidence_tools(self) -> AResearchEvidenceTools:
        return AResearchEvidenceTools(
            self.miner.store, {record["id"] for record in self._a_records()},
            max_single_output_bytes=16 * 1024)

    def read_a_result(self, *, candidate_id: str) -> dict[str, Any]:
        candidate_id = identifier(candidate_id)
        records = self._a_record_map()
        definition = records.get(f"{candidate_id}-definition")
        require(definition is not None and definition["kind"] == "candidate",
                "candidate ID is outside the current A research")
        self._coordinator_a_evidence_tools()  # Validate current-run A provenance.
        report_id = self._latest_report_id(candidate_id)
        decision_result = self._latest_decision(candidate_id)

        def source(record_id, pointer, data):
            return {"record_id": record_id, "pointer": pointer,
                    "record_sha256": digest(records[record_id]), "data": copy.deepcopy(data)}

        decision = None
        if decision_result is not None:
            decision_id, value = decision_result
            rows = records[decision_id]["data"]["candidate_decisions"]
            index = next(i for i, row in enumerate(rows) if row["candidate_id"] == candidate_id)
            decision = source(decision_id, f"/candidate_decisions/{index}", value)
        completion = self.miner.root / "a-complete.json"
        result = {"run_id": self.miner.spec.run_id, "candidate_id": candidate_id,
                  "a_status": "sealed" if completion.is_file() else
                  "sealing" if "a-manifest" in records else "open",
                  "manifest_ref": json.loads(completion.read_text())["manifest_ref"]
                  if completion.is_file() else None,
                  "definition": source(definition["id"], "", definition["data"]),
                  "decision": decision,
                  "report": source(report_id, "", records[report_id]["data"])
                  if report_id is not None else None,
                  "program_evidence_refs": [{"record_id": record["id"], "kind": record["kind"],
                                             "record_sha256": digest(record)}
                                            for record in records.values()
                                            if record["data"].get("candidate_id") == candidate_id
                                            and record["kind"] not in {"model_report", "research_review"}]}
        if len(dumps(result).encode("utf-8")) > 32 * 1024:
            refs = [f"record_id={item['record_id']}, pointer={item['pointer']!r}"
                    for item in (decision, result["report"]) if item is not None]
            raise RecoverableToolError(
                "A result exceeds 32 KiB; no content returned. Use read_a_evidence with "
                "narrower JSON pointers under these exact sources: " + "; ".join(refs))
        return result

    def read_a_evidence(self, *, record_id: str, pointer: str,
                        offset: int, limit: int) -> dict[str, Any]:
        return self._a_evidence_tools().read(
            record_id=record_id, pointer=pointer, offset=offset, limit=limit)

    def prepare_a_task(self) -> dict[str, Any]:
        records = self._a_records()
        record_map = {record["id"]: record for record in records}
        candidates = sorted((record["data"]["id"] for record in records
                            if record["kind"] == "candidate"))
        evaluated = sorted(cid for cid in candidates if f"{cid}-evaluation" in record_map)
        reports = sorted(cid for cid in candidates
                         if any(record["kind"] == "model_report"
                                and record["data"].get("candidate_id") == cid
                                for record in records))
        decisions = {}
        for record in sorted((record for record in records if record["kind"] == "research_review"),
                             key=lambda item: item["created_at"]):
            for decision in record["data"].get("candidate_decisions", []):
                decisions[decision["candidate_id"]] = decision["disposition"]
        retained = sorted(candidate_id for candidate_id, disposition in decisions.items()
                          if disposition == "retain")
        return {"run_id": self.miner.spec.run_id,
                "objective": (self.goal_spec["objective"] if self.goal_spec else self.miner.spec.objective),
                "goal_spec": copy.deepcopy(self.goal_spec),
                "research_contract": self.miner.spec.as_dict(),
                "a_status": ("sealed" if (self.miner.root / "a-complete.json").exists()
                             else "sealing" if "a-manifest" in record_map else "open"),
                "candidate_ids": candidates, "evaluated_ids": evaluated,
                "reported_ids": reports, "decided_ids": sorted(decisions),
                "retained_ids": retained, "retained_count": len(retained),
                "a_evidence_refs": [{"record_id": record["id"], "kind": record["kind"]}
                                    for record in records],
                "pending_candidate_ids": [cid for cid in candidates
                                          if cid not in reports or cid not in decisions]}

    def get_research_status_for(self, role: str) -> dict[str, Any]:
        require(role in {"A", "B", "coordinator"}, "unknown fixed research role")
        records = self._a_records()
        candidates = sorted(record["data"]["id"] for record in records if record["kind"] == "candidate")
        completion_path = self.miner.root / "a-complete.json"
        a_records = {record["id"]: record for record in records}
        a_status = ("sealed" if completion_path.exists()
                    else "sealing" if "a-manifest" in a_records else "open")
        all_pending = self.ledger.pending_actions()
        b_action_names = {"freeze_a_manifest", "validate_frozen_batch",
                          "save_b_interpretation", "finalize_b_delivery"}
        base = {"run_id": self.miner.spec.run_id,
                "objective": (self.goal_spec["objective"] if self.goal_spec
                              else self.miner.spec.objective),
                "program_actions": self.ledger.status(),
                "model_status": {"status": "unknown", "source": "native_host_required"},
                "a_status": a_status,
                "candidate_count": len(candidates),
                "candidate_ids": candidates}
        if role == "A":
            return {**base,
                    "calculation_status": ("pending" if any(row["name"] in _A_ACTION_NAMES
                                                               for row in all_pending) else "idle"),
                    "evaluated_ids": sorted(record["data"]["candidate_id"] for record in records
                                             if record["kind"] == "evaluation"),
                    "reported_ids": sorted(record["data"]["candidate_id"] for record in records
                                            if record["kind"] == "model_report"),
                    "pending_actions": [row for row in all_pending if row["name"] in _A_ACTION_NAMES],
                    "manifest_ref": json.loads(completion_path.read_text())["manifest_ref"]
                    if completion_path.exists() else None}
        frozen_path = self.miner.root / "frozen_batch.json"
        numeric_path = self.miner.root / "b-numerical-complete.json"
        validation_path = self.miner.root / "validation.json"
        b_records_root = self.miner.root / "b_records"
        if role == "B":
            require(frozen_path.is_file(), "B actions are unavailable before A manifest freeze")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else None
        b_records = (RecordStore(b_records_root, run_root=self.miner.root.resolve(),
                                 archive_root=self.miner.archive_root).all()
                     if b_records_root.exists() and role != "A" else [])
        visible_action_names = (b_action_names | _A_ACTION_NAMES
                                if role == "coordinator" else b_action_names)
        status = {**base,
                  "calculation_status": ("pending" if any(row["name"] in visible_action_names
                                                            for row in all_pending) else "idle"),
                  "pending_actions": [row for row in all_pending
                                      if row["name"] in visible_action_names],
                  "frozen_candidate_ids": list(frozen["candidates"]) if frozen else [],
                  "manifest_ref": frozen.get("a_manifest_ref") if frozen else
                  json.loads(completion_path.read_text())["manifest_ref"] if completion_path.exists() else None,
                  "b_access_started": (self.miner.root / "b-access-started.json").exists(),
                  "b_numerical_complete": numeric_path.exists(),
                  "b_interpreted_ids": sorted(record["data"]["candidate_id"] for record in b_records
                                              if record["kind"] == "model_report"),
                  "b_delivery_complete": validation_path.exists(),
                  "goal_evidence": self.check_goal_evidence()
                  if role == "coordinator" and self.goal_spec is not None else None}
        return status

    def get_a_landscape(self) -> dict[str, Any]:
        require(self.a_panel is not None, "A panel is not fixed in the research service binding")
        self._load_candidates()
        records = self._a_records()
        saved = {record["id"]: record["data"] for record in records}
        max_round = max((item["round"] for item in self.miner.candidates.values()), default=0)
        members = self.miner._landscape_members(max_round + 1, saved) if self.miner.candidates else []
        return {"source_segment": "A", "run_id": self.miner.spec.run_id,
                "snapshot": build_factor_landscape(self.a_panel, self.miner.spec, members),
                "source_record_ids": sorted(record["id"] for record in records)}

    def register_experiment(self, *, action_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        plan = modification_plan(plan)
        require(len(plan["route_id"]) <= 60,
                "route_id is too long for the immutable experiment record IDs")
        parameters = {"plan": plan}

        def operation():
            self._require_a_open()
            self._ensure_a_initialized(action_id)
            self._load_candidates()
            record_id = f"experiment-{plan['route_id']}"
            existing = self._a_record_map().get(record_id)
            routes = self.miner.routes
            if existing is not None:
                require(existing["kind"] == "modification_experiment"
                        and existing["data"]["plan"] == plan,
                        "registered experiment route differs from its saved plan")
                return {"route_id": plan["route_id"], "plan_ref": record_id,
                        "status": routes[plan["route_id"]]["decision"]}
            self.miner._register_route(plan, routes)
            route = routes[plan["route_id"]]
            data = {"plan": plan, "attempts": route["attempts"],
                    "decision": route["decision"], "registered_by_action": action_id}
            self._commit(action_id, lambda: self.miner.store.append(
                record_id, "modification_experiment", data))
            self._load_candidates()
            return {"route_id": plan["route_id"], "plan_ref": record_id, "status": "continue"}

        def recover():
            record = self._a_record_map().get(f"experiment-{plan['route_id']}")
            if record is None:
                require(f"route-resolution-{plan['route_id']}" not in self._a_record_map(),
                        "experiment resolution exists without its registered route")
                return operation()
            require(record is not None and record["kind"] == "modification_experiment"
                    and record["data"]["plan"] == plan,
                    "experiment action is pending without a complete saved plan")
            return {"route_id": plan["route_id"], "plan_ref": record["id"],
                    "status": record["data"]["decision"]}

        return self._run_action(action_id, "register_experiment", parameters, operation, recover=recover)

    def resolve_experiment(self, *, action_id: str, route_id: str,
                           status: str, reason: str) -> dict[str, Any]:
        route_id = identifier(route_id)
        require(len(route_id) <= 60,
                "route_id is too long for the immutable experiment record IDs")
        require(status in {"paused", "stopped"}, "experiment resolution must pause or stop the route")
        text(reason, "experiment resolution reason")
        parameters = {"route_id": route_id, "status": status, "reason": reason}
        record_id = f"route-resolution-{route_id}"
        data = {**parameters, "action_id": action_id}

        def child_resolution():
            self._load_candidates()
            children = [item for item in self.miner.candidates.values()
                        if isinstance(item.get("experiment"), dict)
                        and item["experiment"].get("route_id") == route_id]
            if not children:
                return None
            saved = self._a_record_map().get(record_id)
            require(saved is not None and saved["kind"] == "route_resolution"
                    and saved["data"].get("candidate_id") in {item["id"] for item in children}
                    and saved["data"].get("status") == status,
                    "resolve the experiment through its candidate decision after registering a child")
            child_id = saved["data"]["candidate_id"]
            decision = self._latest_decision(child_id)
            source_review_id = self._review_id_for_action(saved["data"]["action_id"])
            source_review = self._a_record_map().get(source_review_id)
            source_decision = next((row for row in source_review["data"]["candidate_decisions"]
                                    if row["candidate_id"] == child_id), None) if source_review else None
            require(decision is not None
                    and source_decision is not None and not source_decision["continue_optimization"]
                    and not decision[1]["continue_optimization"]
                    and ("paused" if decision[1]["disposition"] == "pause" else "stopped") == status
                    and ("paused" if source_decision["disposition"] == "pause" else "stopped") == status,
                    "route resolution must reference the child's final saved decision")
            return {"route_id": route_id, "resolution_ref": record_id, "status": status,
                    "already_resolved": True, "decision_ref": decision[0],
                    "resolution_decision_ref": source_review_id}

        def preflight():
            self._require_a_open()
            require(f"experiment-{route_id}" in self._a_record_map(),
                    "cannot resolve an unregistered experiment")
            child_resolution()

        def resolve():
            self._require_a_open()
            self._load_candidates()
            confirmed = child_resolution()
            if confirmed is not None:
                return confirmed
            plan_record = self._a_record_map().get(f"experiment-{route_id}")
            require(plan_record is not None and plan_record["kind"] == "modification_experiment",
                    "cannot resolve an unregistered experiment")
            existing = self._a_record_map().get(record_id)
            if existing is not None:
                require(existing["kind"] == "route_resolution"
                        and existing["data"]["status"] == status
                        and existing["data"]["reason"] == reason,
                        "experiment route has an immutable resolution")
                return {"route_id": route_id, "resolution_ref": record_id, "status": status}
            self.miner.store.append(record_id, "route_resolution", data)
            self.miner.routes[route_id]["decision"] = status
            return {"route_id": route_id, "resolution_ref": record_id, "status": status}

        def operation():
            return self._commit(action_id, resolve)

        def recover():
            confirmed = child_resolution()
            if confirmed is not None:
                return confirmed
            record = self._a_record_map().get(record_id)
            if record is None:
                return operation()
            require(record is not None and record["kind"] == "route_resolution"
                    and record["data"] == data,
                    "experiment resolution is pending without a matching saved decision")
            return {"route_id": route_id, "resolution_ref": record_id, "status": status}

        return self._run_action(action_id, "resolve_experiment", parameters,
                                operation, recover=recover, preflight=preflight)

    def _candidate_result(self, candidate_id: str) -> dict[str, Any]:
        self._load_candidates()
        require(candidate_id in self.miner.candidates, "candidate is missing from the A record store")
        item = self.miner.candidates[candidate_id]
        records = self._a_record_map()
        duplicate = records.get(f"{candidate_id}-duplicate")
        if duplicate is not None:
            return {"candidate_id": candidate_id, "status": "duplicate",
                    "duplicate_of": duplicate["data"]["duplicate_of"],
                    "evidence_refs": [f"{candidate_id}-definition", duplicate["id"]]}
        calculation_record = records.get(f"{candidate_id}-calculation")
        evaluation_record = records.get(f"{candidate_id}-evaluation")
        comparison_record = records.get(f"{candidate_id}-comparison")
        attempts = [record for record in records.values()
                    if record["kind"] == "calculation_attempt"
                    and record["data"].get("candidate_id") == candidate_id]
        if calculation_record is None:
            if attempts:
                return {"candidate_id": candidate_id, "status": "repair_pending",
                        "calculation_attempts": [record["data"] for record in sorted(
                            attempts, key=lambda record: record["data"]["attempt"])],
                        "evidence_refs": [f"{candidate_id}-definition",
                                          *[record["id"] for record in attempts]]}
            return {"candidate_id": candidate_id, "status": "pending",
                    "evidence_refs": [f"{candidate_id}-definition"]}
        calculation = calculation_record["data"]
        result = {"candidate_id": candidate_id, "status": calculation["status"],
                  "calculation": calculation,
                  "evidence_refs": [f"{candidate_id}-definition", calculation_record["id"]]}
        if evaluation_record is not None:
            report = evaluation_record["data"]
            result["evaluation"] = {
                "summary": report["summary"], "coverage": report["coverage"],
                "horizons": {horizon: {"summary": data["summary"],
                                        "coverage": data["coverage"]}
                             for horizon, data in report["horizon_comparison"]["horizons"].items()}}
            result["evidence_refs"].append(evaluation_record["id"])
            result.setdefault("evidence_pointers", {})["evaluation"] = {
                "record_id": evaluation_record["id"],
                "field_pointers": {
                    "summary": "/summary",
                    "coverage": "/coverage",
                    "horizons": {
                        horizon: {
                            "summary": f"/horizon_comparison/horizons/{horizon}/summary",
                            "coverage": f"/horizon_comparison/horizons/{horizon}/coverage",
                        }
                        for horizon in report["horizon_comparison"]["horizons"]
                    },
                },
            }
        if comparison_record is not None:
            comparison = comparison_record["data"]
            result["comparison"] = {
                "route_id": comparison["plan"]["route_id"],
                "metric": comparison["plan"]["experiment_design"]["metric"],
                "paired_improvement": comparison.get("paired_improvement"),
                "paired_ic_change": comparison.get("paired_ic_change"),
                "route_decision": comparison["route_decision"]}
            result["evidence_refs"].append(comparison_record["id"])
            result.setdefault("evidence_pointers", {})["comparison"] = {
                "record_id": comparison_record["id"],
                "field_pointers": {
                    "route_id": "/plan/route_id",
                    "metric": "/plan/experiment_design/metric",
                    "paired_improvement": "/paired_improvement",
                    "paired_ic_change": "/paired_ic_change",
                    "route_decision": "/route_decision",
                },
            }
        return result

    def _experiment(self, experiment_id: str | None,
                    definition: dict[str, Any]) -> dict[str, Any] | None:
        if experiment_id is None:
            require(definition["parent_id"] is None,
                    "a modified candidate requires a registered experiment before evaluation")
            return None
        route_id = identifier(experiment_id)
        require(len(route_id) <= 60,
                "route_id is too long for the immutable experiment record IDs")
        record = self._a_record_map().get(f"experiment-{route_id}")
        require(record is not None and record["kind"] == "modification_experiment",
                "candidate experiment must be predeclared and registered before evaluation")
        plan = record["data"]["plan"]
        require(definition["parent_id"] == plan["control_id"],
                "modified candidate parent differs from the predeclared experiment control")
        self._load_candidates()
        control = self.miner.candidates[plan["control_id"]]
        require(definition["direction"] == control["definition"]["direction"],
                "controlled modifications keep the frozen direction")
        require(self.miner.routes[route_id]["decision"] == "continue",
                "a paused or stopped experiment route requires a newly registered restart")
        return plan

    def _persist_failed_calculation(self, action_id: str, candidate_id: str,
                                    definition: dict[str, Any], expression: str,
                                    attempt: int, error: ValueError,
                                    compiled) -> dict[str, Any]:
        record_id = f"{candidate_id}-calculation-attempt-{attempt:04d}"
        data = {"candidate_id": candidate_id, "attempt": attempt, "expression": expression,
                "actual_steps": compiled.calculation_steps() if compiled is not None else None,
                "program_error": str(error), "error_type": type(error).__name__,
                "action_id": action_id}
        existing = self._a_record_map().get(record_id)
        if existing is not None:
            require(existing["kind"] == "calculation_attempt" and existing["data"] == data,
                    "saved calculation attempt differs from the submitted action")
        else:
            self.miner.store.append(record_id, "calculation_attempt", data)
        max_repairs = self.miner.spec.max_repairs
        if isinstance(error, FormulaBudgetError) or attempt >= max_repairs:
            attempts = [record["data"] for record in self._a_records()
                        if record["kind"] == "calculation_attempt"
                        and record["data"].get("candidate_id") == candidate_id]
            attempts = sorted(attempts, key=lambda row: row["attempt"])
            calculation = {"status": "calculation_failed",
                           "original_expression": definition["expression"],
                           "original_meaning": definition["meaning"],
                           "checks": [{"attempt": row["attempt"], "expression": row["expression"],
                                       "actual_steps": row["actual_steps"],
                                       "program_error": row["program_error"]} for row in attempts],
                           "executed_expression": None}
            calc_record = {"candidate_id": candidate_id, **calculation}
            self.miner.store.append(f"{candidate_id}-calculation", "calculation", calc_record)
            self.miner.candidates[candidate_id]["calculation"] = calculation
            self._close_uncomputed_route(action_id, self.miner.candidates[candidate_id],
                                         "candidate calculation failed within its declared repair budget")
            return {"candidate_id": candidate_id, "status": "calculation_failed",
                    "calculation": calculation,
                    "evidence_refs": [f"{candidate_id}-definition", record_id,
                                      f"{candidate_id}-calculation"]}
        return {"candidate_id": candidate_id, "status": "repair_pending",
                "attempt": attempt, "program_error": str(error),
                "evidence_refs": [f"{candidate_id}-definition", record_id]}

    def _close_uncomputed_route(self, action_id: str, item: dict[str, Any], reason: str) -> None:
        plan = item.get("experiment")
        if not isinstance(plan, dict):
            return
        route_id = plan["route_id"]
        record_id = f"route-resolution-{route_id}"
        data = {"route_id": route_id, "status": "pause_insufficient",
                "reason": reason, "candidate_id": item["id"], "action_id": action_id}
        existing = self._a_record_map().get(record_id)
        if existing is None:
            self.miner.store.append(record_id, "route_resolution", data)
        else:
            require(existing["kind"] == "route_resolution"
                    and existing["data"] == data,
                    "uncomputed experiment route already has a different resolution")
        self.miner.routes[route_id]["decision"] = "pause_insufficient"

    def _load_action_values(self, candidate_id: str, definition: dict[str, Any],
                            compiled, action_id: str) -> tuple[dict[str, Any], ExpressionResult] | None:
        identity = FactorIdentity(compiled.expanded_expression,
                                  str(definition["direction"]), "expr-v1")
        try:
            archive = FactorArchive.open_existing(self.miner.archive_root, identity)
            saved = archive.get_value_set(f"{self.miner.spec.run_id}/A", "factor-eval-v1")
        except (FileNotFoundError, KeyError):
            return None
        provenance = [source for source in saved["provenance"]
                      if source.get("run_id") == self.miner.spec.run_id
                      and source.get("candidate_id") == candidate_id
                      and source.get("action_id") == action_id
                      and source.get("definition") == definition]
        if not provenance:
            return None
        frame = pd.read_csv(io.BytesIO(saved["factor_values_csv"]), index_col=[0, 1],
                            parse_dates=[0], float_precision="round_trip")
        require(frame.shape[1] == 1, "saved A value set must contain one factor column")
        timestamps = pd.to_datetime(frame.index.get_level_values(0), utc=True)
        index = pd.MultiIndex.from_arrays(
            [timestamps, frame.index.get_level_values(1)], names=["timestamp", "symbol"])
        values = pd.Series(frame.iloc[:, 0].to_numpy(dtype=float), index=index,
                           name="factor_value")
        artifact = {"root": os.path.relpath(self.miner.archive_root.resolve(),
                                             self.miner.root.resolve()),
                    "identity": identity.as_dict(), "data_version": f"{self.miner.spec.run_id}/A",
                    "computation_semantics": "factor-eval-v1",
                    "value_set_id": saved["value_set_id"], "rows": len(values)}
        return artifact, ExpressionResult(values, compiled.description(), pd.DataFrame())

    def _require_no_foreign_candidate_value_provenance(self, candidate_id: str,
                                                        definition: dict[str, Any],
                                                        compiled, action_id: str) -> None:
        identity = FactorIdentity(compiled.expanded_expression,
                                  str(definition["direction"]), "expr-v1")
        try:
            archive = FactorArchive.open_existing(self.miner.archive_root, identity)
            saved = archive.get_value_set(f"{self.miner.spec.run_id}/A", "factor-eval-v1")
        except (FileNotFoundError, KeyError):
            return
        foreign = [source for source in saved["provenance"]
                   if source.get("run_id") == self.miner.spec.run_id
                   and source.get("candidate_id") == candidate_id
                   and source.get("action_id") != action_id]
        require(not foreign,
                "candidate value set is committed under a different action ID")

    def _persist_candidate_evaluation(self, *, action_id: str, candidate_id: str,
                                     definition: dict[str, Any], plan: dict[str, Any] | None,
                                     compiled, evaluated: ExpressionResult,
                                     artifact: dict[str, Any], checks: list[dict[str, Any]],
                                     repair_refs: list[str] | None = None) -> dict[str, Any]:
        labels_by_horizon = {}
        for horizon in (1, 4, 24):
            self.ledger.require_active()
            labels_by_horizon[horizon] = build_labels(
                self.a_panel, self.miner.spec, "A", horizon_hours=horizon)
        report = evaluate_a_candidate(evaluated.values, labels_by_horizon,
                                      self.miner.spec, definition["direction"],
                                      stop_check=self.ledger.require_active)
        displacement = None
        if self.miner.rank_displacement_version is not None:
            displacement = evaluate_rank_displacement(
                evaluated.values, self.a_panel.universe, self.miner.spec, "A",
                stop_check=self.ledger.require_active)
            require(displacement["definition_version"] == self.miner.rank_displacement_version,
                    "rank-displacement implementation differs from the run marker")
        comparison = None
        if plan is not None:
            self._load_candidates()
            control = self.miner.candidates[plan["control_id"]]
            require(control.get("calculation", {}).get("status") == "computed",
                    "experiment control requires a saved A calculation")
            control_values = self._load_archived_values(plan["control_id"])
            comparison = compare_candidates(
                evaluated.values, control_values, self.a_panel,
                build_labels(self.a_panel, self.miner.spec, "A"), self.miner.spec,
                definition["direction"], control["definition"]["direction"],
                plan, self.miner.rank_displacement_version,
                stop_check=self.ledger.require_active)
            comparison["route_decision"] = comparison.pop("decision")
            comparison["plan"] = plan
        calculation = {"status": "computed", "original_expression": definition["expression"],
                       "original_meaning": definition["meaning"], "checks": checks,
                       "executed_expression": evaluated.definition,
                       "valid_values": int(evaluated.values.notna().sum()),
                       "values_artifact": artifact}
        if repair_refs:
            calculation["repair_refs"] = repair_refs

        def persist():
            records = self._a_record_map()
            calculation_data = {"candidate_id": candidate_id, **calculation}
            prior_calculation = records.get(f"{candidate_id}-calculation")
            if prior_calculation is None:
                self.miner.store.append(f"{candidate_id}-calculation", "calculation", calculation_data)
            else:
                require(prior_calculation["kind"] == "calculation"
                        and prior_calculation["data"] == calculation_data,
                        "saved calculation differs from the recovered A calculation")
            self._load_candidates()
            item = self.miner.candidates[candidate_id]
            item["calculation"] = copy.deepcopy(calculation)
            item["evaluation"] = report
            identity = FactorIdentity(evaluated.definition["expanded_expression"],
                                      str(definition["direction"]), "expr-v1")
            locator = self.miner._write_evaluation(
                candidate_id, "A", report, identity,
                value_set_id=artifact["value_set_id"], factor_value_count=artifact["rows"],
                definition=definition)
            displacement_locator = None
            if displacement is not None:
                displacement_locator = self.miner._write_rank_displacement(
                    candidate_id, "A", displacement, identity,
                    source={"calculation_artifact": artifact})
                item["rank_displacement"] = displacement
                item["rank_displacement_archive"] = displacement_locator
            evaluation_data = self.miner._evaluation_record_data(
                candidate_id, report, locator, displacement, displacement_locator)
            existing_evaluation = records.get(f"{candidate_id}-evaluation")
            if existing_evaluation is None:
                self.miner.store.append(f"{candidate_id}-evaluation", "evaluation", evaluation_data)
            else:
                require(existing_evaluation["kind"] == "evaluation"
                        and existing_evaluation["data"] == evaluation_data,
                        "saved A evaluation differs from its recovered calculation")
            if comparison is not None:
                comparison_data = {"candidate_id": candidate_id, **comparison}
                existing_comparison = records.get(f"{candidate_id}-comparison")
                if existing_comparison is None:
                    self.miner.store.append(f"{candidate_id}-comparison", "experiment_result",
                                            comparison_data)
                else:
                    require(existing_comparison["kind"] == "experiment_result"
                            and existing_comparison["data"] == comparison_data,
                            "saved experiment result differs from its recovered comparison")
                self.miner.routes[plan["route_id"]]["decision"] = comparison["route_decision"]
                item["comparison"] = comparison_data
        self._commit(action_id, persist)
        return self._candidate_result(candidate_id)

    def evaluate_candidate(self, *, action_id: str, candidate_id: str,
                          definition: dict[str, Any],
                          experiment_id: str | None = None) -> dict[str, Any]:
        candidate_id = identifier(candidate_id)
        require(len(candidate_id) <= 50,
                "candidate_id is too long for the immutable A record IDs")
        definition = candidate(definition)
        if experiment_id is not None:
            experiment_id = identifier(experiment_id)
        require(self.a_panel is not None, "A panel is not fixed in the research service binding")
        parameters = {"candidate_id": candidate_id, "definition": definition,
                      "experiment_id": experiment_id,
                      "a_panel_sha256": self._panel_fingerprint(self.a_panel)}

        def operation():
            self._require_a_open()
            self._ensure_a_initialized(action_id)
            self._load_candidates()
            plan = self._experiment(experiment_id, definition)
            record_map = self._a_record_map()
            definition_record = record_map.get(f"{candidate_id}-definition")
            item = {"id": candidate_id, "round": 1, "definition": definition,
                    "experiment": plan, "created_by_action_id": action_id}
            if definition_record is not None:
                require(definition_record["kind"] == "candidate"
                        and definition_record["data"] == item,
                        "candidate ID is already bound to a different or incomplete action")
            else:
                require(candidate_id not in self.miner.candidates,
                        "candidate ID is already used by this research run")
                self._commit(action_id, lambda: self.miner.store.append(
                    f"{candidate_id}-definition", "candidate", item))
            self._load_candidates()
            record_map = self._a_record_map()
            if f"{candidate_id}-calculation" in record_map:
                require(f"{candidate_id}-evaluation" in record_map
                        and (plan is None or f"{candidate_id}-comparison" in record_map),
                        "candidate has an incomplete committed action; recover with its original action_id")
                raise ValueError("candidate has already been evaluated; use its saved A evidence")
            if f"{candidate_id}-calculation-attempt-0000" in record_map:
                require(record_map[f"{candidate_id}-calculation-attempt-0000"]["data"]["action_id"] == action_id,
                        "candidate has an unfinished formula error; repair it using the saved attempt")
            if (f"{candidate_id}-calculation" not in record_map
                    and f"{candidate_id}-calculation-attempt-0000" not in record_map):
                try:
                    compiled = compile_factor_formula(definition["expression"], self.miner.spec)
                    evaluated = evaluate_expression(
                        definition["expression"], self.a_panel,
                        stop_check=self.ledger.require_active)
                except ValueError as exc:
                    self.ledger.require_active()
                    return self._commit(action_id, lambda: self._persist_failed_calculation(
                        action_id, candidate_id, definition, definition["expression"], 0, exc,
                        locals().get("compiled")))
                self.ledger.require_active()
                duplicate_of = None
                for other_id, other in self.miner.candidates.items():
                    if other_id == candidate_id:
                        continue
                    other_calc = other.get("calculation")
                    if other_calc and other_calc.get("status") == "computed":
                        other_expression = other_calc["executed_expression"]["expanded_expression"]
                    else:
                        try:
                            other_expression = compile_factor_formula(
                                other["definition"]["expression"], self.miner.spec).expanded_expression
                        except ValueError:
                            continue
                    if (other_expression == compiled.expanded_expression
                            and other["definition"]["direction"] == definition["direction"]):
                        duplicate_of = other_id
                        break
                if duplicate_of is not None:
                    duplicate_data = {"candidate_id": candidate_id, "duplicate_of": duplicate_of}
                    def save_duplicate():
                        self.miner.store.append(f"{candidate_id}-duplicate", "duplicate", duplicate_data)
                        self._close_uncomputed_route(
                            action_id, self.miner.candidates[candidate_id],
                            "candidate is a duplicate and has no paired result")
                    self._commit(action_id, save_duplicate)
                    return self._candidate_result(candidate_id)
                artifact = self._commit(action_id, lambda: archive_factor_values(
                    self.miner.archive_root, self.miner.root, self.miner.spec,
                    candidate_id, definition, evaluated, action_id=action_id))
                self.ledger.require_active()
                return self._persist_candidate_evaluation(
                    action_id=action_id, candidate_id=candidate_id, definition=definition,
                    plan=plan, compiled=compiled, evaluated=evaluated, artifact=artifact,
                    checks=[{"attempt": 0, "expression": definition["expression"],
                             "actual_steps": compiled.calculation_steps(), "program_error": None}])
            attempt = record_map.get(f"{candidate_id}-calculation-attempt-0000")
            require(attempt is None or attempt["data"]["action_id"] == action_id,
                    "candidate has an unfinished formula error; repair it using the saved attempt")
            return self._candidate_result(candidate_id)

        def recover():
            if f"{candidate_id}-definition" not in self._a_record_map():
                return operation()
            result = self._candidate_result(candidate_id)
            self._load_candidates()
            item = self.miner.candidates[candidate_id]
            require(item["created_by_action_id"] == action_id
                    and item["definition"] == definition
                    and ((experiment_id is None and item.get("experiment") is None)
                         or (isinstance(item.get("experiment"), dict)
                             and item["experiment"]["route_id"] == experiment_id)),
                    "pending candidate action differs from its saved definition or experiment")
            if (self.miner.root / "a-complete.json").is_file():
                require(result["status"] in {"computed", "duplicate", "calculation_failed"},
                        "sealed A action lacks complete saved evidence for read-only recovery")
                if result["status"] == "computed":
                    require("evaluation" in result
                            and (item.get("experiment") is None or "comparison" in result),
                            "sealed candidate action is missing its evaluation or comparison")
                if result["status"] == "calculation_failed":
                    require(any(record["data"].get("action_id") == action_id
                                for record in self._a_record_map().values()
                                if record["kind"] == "calculation_attempt"
                                and record["data"].get("candidate_id") == candidate_id),
                            "sealed failed calculation does not belong to the pending action")
                return result
            self._require_a_open()
            self._ensure_a_initialized(action_id)
            if result["status"] in {"duplicate", "calculation_failed", "repair_pending"}:
                if result["status"] == "repair_pending":
                    require(result["calculation_attempts"][-1]["action_id"] == action_id,
                            "pending formula error belongs to a different action")
                elif item.get("experiment") is not None:
                    reason = ("candidate is a duplicate and has no paired result"
                              if result["status"] == "duplicate"
                              else "candidate calculation failed within its declared repair budget")
                    self._commit(action_id, lambda: self._close_uncomputed_route(
                        action_id, item, reason))
                return result
            self._load_candidates()
            item = self.miner.candidates.get(candidate_id)
            require(item is not None and item["definition"] == definition,
                    "pending candidate action has no matching candidate definition")
            compiled = compile_factor_formula(definition["expression"], self.miner.spec)
            self._require_no_foreign_candidate_value_provenance(
                candidate_id, definition, compiled, action_id)
            saved = self._load_action_values(candidate_id, definition, compiled, action_id)
            if saved is None:
                require(result["status"] == "pending",
                        "candidate action has committed calculations but no recoverable value set")
                return operation()
            artifact, evaluated = saved
            plan = self._experiment(experiment_id, definition)
            if (result["status"] == "computed" and "evaluation" in result
                    and (plan is None or "comparison" in result)):
                return result
            self.ledger.require_active()
            checks = (item.get("calculation", {}).get("checks") or
                      [{"attempt": 0, "expression": definition["expression"],
                        "actual_steps": compiled.calculation_steps(), "program_error": None}])
            return self._persist_candidate_evaluation(
                action_id=action_id, candidate_id=candidate_id, definition=definition,
                plan=plan, compiled=compiled, evaluated=evaluated, artifact=artifact,
                checks=checks)

        return self._run_action(action_id, "evaluate_candidate", parameters, operation, recover=recover)

    def repair_candidate(self, *, action_id: str, candidate_id: str,
                         program_error: str, repair_expression: str,
                         reason: str) -> dict[str, Any]:
        candidate_id = identifier(candidate_id)
        require(len(candidate_id) <= 50,
                "candidate_id is too long for the immutable A record IDs")
        text(program_error, "program error")
        text(repair_expression, "repair expression")
        text(reason, "formula repair reason")
        require(self.a_panel is not None, "A panel is not fixed in the research service binding")
        parameters = {"candidate_id": candidate_id, "program_error": program_error,
                      "repair_expression": repair_expression, "reason": reason,
                      "a_panel_sha256": self._panel_fingerprint(self.a_panel)}

        def operation():
            self._require_a_open()
            self._ensure_a_initialized(action_id)
            self._load_candidates()
            require(candidate_id in self.miner.candidates, "repair candidate is not in this A run")
            item = self.miner.candidates[candidate_id]
            definition = item["definition"]
            records = self._a_record_map()
            require(f"{candidate_id}-calculation" not in records,
                    "candidate calculation is complete and cannot be repaired")
            attempts = sorted((record["data"] for record in records.values()
                               if record["kind"] == "calculation_attempt"
                               and record["data"].get("candidate_id") == candidate_id),
                              key=lambda row: row["attempt"])
            require(attempts and attempts[-1]["program_error"] == program_error,
                    "repair must address the latest saved program error exactly")
            require(attempts[-1]["error_type"] != "FormulaBudgetError",
                    "a formula budget failure requires a new candidate, not formula repair")
            repair_count = len(attempts) - 1
            require(repair_count < self.miner.spec.max_repairs,
                    "candidate has exhausted the declared formula repair budget")
            repair_id = f"{candidate_id}-repair-{repair_count + 1:04d}"
            repair_record = {"candidate_id": candidate_id, "attempt": repair_count + 1,
                             "original_expression": definition["expression"],
                             "program_error": program_error,
                             "repair_expression": repair_expression,
                             "reason": reason, "action_id": action_id}
            old_repair = records.get(repair_id)
            if old_repair is not None:
                require(old_repair["kind"] == "formula_repair"
                        and old_repair["data"] == repair_record,
                        "saved formula repair differs from the submitted action")
            try:
                compiled = compile_factor_formula(repair_expression, self.miner.spec)
                evaluated = evaluate_expression(
                    repair_expression, self.a_panel,
                    stop_check=self.ledger.require_active)
            except ValueError as exc:
                self.ledger.require_active()
                def save_failure():
                    if old_repair is None:
                        self.miner.store.append(repair_id, "formula_repair", repair_record)
                    result = self._persist_failed_calculation(
                        action_id, candidate_id, definition, repair_expression,
                        repair_count + 1, exc, locals().get("compiled"))
                    return result
                return self._commit(action_id, save_failure)
            self.ledger.require_active()
            artifact = self._commit(action_id, lambda: archive_factor_values(
                self.miner.archive_root, self.miner.root, self.miner.spec,
                candidate_id, definition, evaluated, action_id=action_id))
            self.ledger.require_active()
            if old_repair is None:
                self._commit(action_id, lambda: self.miner.store.append(
                    repair_id, "formula_repair", repair_record))
            checks = [{"attempt": row["attempt"], "expression": row["expression"],
                       "actual_steps": row["actual_steps"], "program_error": row["program_error"]}
                      for row in attempts]
            checks.append({"attempt": repair_count + 1, "expression": repair_expression,
                           "actual_steps": compiled.calculation_steps(), "program_error": None})
            plan = item.get("experiment")
            return self._persist_candidate_evaluation(
                action_id=action_id, candidate_id=candidate_id, definition=definition,
                plan=plan, compiled=compiled, evaluated=evaluated, artifact=artifact,
                checks=checks, repair_refs=[*[f"{candidate_id}-repair-{n:04d}"
                                              for n in range(1, repair_count + 1)], repair_id])

        def recover():
            result = self._candidate_result(candidate_id)
            if result["status"] == "computed":
                repair_refs = result.get("calculation", {}).get("repair_refs", [])
                require(repair_refs, "computed repair action is missing its repair reference")
                repair = self._a_record_map().get(repair_refs[-1])
                expected = {"candidate_id": candidate_id,
                            "attempt": len(result.get("calculation", {}).get("checks", [])) - 1,
                            "original_expression": self.miner.candidates[candidate_id]["definition"]["expression"],
                            "program_error": program_error,
                            "repair_expression": repair_expression,
                            "reason": reason, "action_id": action_id}
                require(repair is not None and repair["kind"] == "formula_repair"
                        and repair["id"] == repair_refs[-1] and repair["data"] == expected,
                        "computed repair does not belong to the pending action")
                return result
            if result["status"] == "calculation_failed":
                records = self._a_record_map()
                attempts = [record for record in records.values()
                            if record["kind"] == "calculation_attempt"
                            and record["data"].get("candidate_id") == candidate_id]
                repair_id = f"{candidate_id}-repair-{len(attempts):04d}"
                repair = records.get(repair_id)
                expected = {"candidate_id": candidate_id,
                            "attempt": len(attempts),
                            "original_expression": self.miner.candidates[candidate_id]["definition"]["expression"],
                            "program_error": program_error,
                            "repair_expression": repair_expression,
                            "reason": reason, "action_id": action_id}
                require(repair is not None and repair["kind"] == "formula_repair"
                        and repair["data"] == expected,
                        "repair action lacks a matching committed repair record")
                self._load_candidates()
                item = self.miner.candidates[candidate_id]
                if item.get("experiment") is not None:
                    self._commit(action_id, lambda: self._close_uncomputed_route(
                        action_id, item,
                        "candidate calculation failed within its declared repair budget"))
                return result
            self._require_a_open()
            self._ensure_a_initialized(action_id)
            self._load_candidates()
            item = self.miner.candidates.get(candidate_id)
            require(item is not None, "pending repair candidate is missing")
            require(result["status"] == "repair_pending",
                    "pending repair action has no saved formula error to address")
            compiled = compile_factor_formula(repair_expression, self.miner.spec)
            attempts = sorted((record["data"] for record in self._a_records()
                               if record["kind"] == "calculation_attempt"
                               and record["data"].get("candidate_id") == candidate_id),
                              key=lambda row: row["attempt"])
            require(attempts and attempts[-1]["program_error"] == program_error,
                    "pending repair action does not match the latest saved formula error")
            repair_count = len(attempts) - 1
            repair_id = f"{candidate_id}-repair-{repair_count + 1:04d}"
            repair_record = self._a_record_map().get(repair_id)
            expected_repair = {"candidate_id": candidate_id, "attempt": repair_count + 1,
                               "original_expression": item["definition"]["expression"],
                               "program_error": program_error,
                               "repair_expression": repair_expression,
                               "reason": reason, "action_id": action_id}
            self._require_no_foreign_candidate_value_provenance(
                candidate_id, item["definition"], compiled, action_id)
            saved = self._load_action_values(candidate_id, item["definition"], compiled, action_id)
            if saved is None:
                if attempts[-1]["action_id"] == action_id:
                    require(repair_record is not None
                            and repair_record["kind"] == "formula_repair"
                            and repair_record["data"] == expected_repair,
                            "failed repair evidence is incomplete or belongs to another action")
                    return result
                return operation()
            artifact, evaluated = saved
            checks = [{"attempt": row["attempt"], "expression": row["expression"],
                       "actual_steps": row["actual_steps"], "program_error": row["program_error"]}
                      for row in attempts]
            checks.append({"attempt": repair_count + 1, "expression": repair_expression,
                           "actual_steps": compiled.calculation_steps(), "program_error": None})
            if repair_record is None:
                self._commit(action_id, lambda: self.miner.store.append(
                    repair_id, "formula_repair", expected_repair))
            else:
                require(repair_record["kind"] == "formula_repair"
                        and repair_record["data"] == expected_repair,
                        "repair value set has a mismatched repair record")
            return self._persist_candidate_evaluation(
                action_id=action_id, candidate_id=candidate_id,
                definition=item["definition"], plan=item.get("experiment"),
                compiled=compiled, evaluated=evaluated, artifact=artifact,
                checks=checks, repair_refs=[f"{candidate_id}-repair-{n:04d}"
                                            for n in range(1, repair_count + 2)])

        return self._run_action(action_id, "repair_candidate", parameters, operation, recover=recover)

    def _load_archived_values(self, candidate_id: str) -> pd.Series:
        self._load_candidates()
        item = self.miner.candidates[candidate_id]
        locator = item["calculation"]["values_artifact"]
        archive = FactorArchive.open_existing(
            self.miner.archive_root, FactorIdentity(**locator["identity"]))
        saved = archive.get_value_set(locator["data_version"], locator["computation_semantics"])
        values = pd.read_csv(io.BytesIO(saved["factor_values_csv"]), index_col=[0, 1],
                             parse_dates=[0], float_precision="round_trip")
        values.index = pd.MultiIndex.from_arrays([
            pd.to_datetime(values.index.get_level_values(0), utc=True),
            values.index.get_level_values(1)], names=["timestamp", "symbol"])
        require(values.shape[1] == 1, "saved factor value set must contain one value column")
        return values.iloc[:, 0].astype(float)

    def _save_narrative(self, stage: str, candidate_id: str,
                        narrative: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        require(stage in {"A", "B"}, "report stage must be A or B")
        records = self._a_record_map() if stage == "A" else {
            record["id"]: record for record in self._b_store().all()}
        require(candidate_id in self.miner.candidates,
                "report candidate is not in this research run")
        candidate_item = self.miner.candidates[candidate_id]
        if stage == "A":
            self._require_a_open()
            require("evaluation" in candidate_item or "duplicate_of" in candidate_item
                    or candidate_item.get("calculation", {}).get("status") == "calculation_failed",
                    "A report requires saved program evidence")
            store = self.miner.store
        else:
            require((self.miner.root / "b-numerical-complete.json").is_file(),
                    "B report requires a complete numerical checkpoint")
            frozen = self.miner._checked_frozen()
            require(candidate_id in frozen["candidates"],
                    "B report candidate is outside the frozen batch")
            store = self._b_store()
        extras: list[str] = []
        parsed = model_fields(narrative, REPORT_SCHEMA, extras)
        require(not extras, "research report contains undeclared fields")
        data = {"candidate_id": candidate_id, **parsed}
        report_records = [record for record in records.values()
                          if record["kind"] == "model_report"
                          and record["data"].get("candidate_id") == candidate_id]
        same = max((record for record in report_records if record["data"] == data),
                   key=_report_order, default=None)
        if same is not None:
            return same["id"], data
        if not report_records:
            record_id = f"{candidate_id}-report"
        else:
            revision = sum("-report-revision-" in record["id"] for record in report_records) + 1
            record_id = f"{candidate_id}-report-revision-{revision:04d}"
        store.append(record_id, "model_report", data)
        if stage == "A":
            candidate_item["model_report"] = data
        return record_id, data

    def _b_store(self) -> RecordStore:
        return RecordStore(self.miner.root / "b_records", run_root=self.miner.root.resolve(),
                           archive_root=self.miner.archive_root)

    def _decision_records(self, candidate_id: str | None = None) -> list[dict[str, Any]]:
        rows = [record for record in self._a_records() if record["kind"] == "research_review"]
        if candidate_id is not None:
            rows = [record for record in rows
                    if any(decision["candidate_id"] == candidate_id
                           for decision in record["data"].get("candidate_decisions", []))]
        return rows

    def _review_id_for_action(self, action_id: str) -> str | None:
        return next((record["id"] for record in self._decision_records()
                     if record["data"].get("action_id") == action_id), None)

    def save_research_decision(self, *, action_id: str, candidate_id: str,
                               decision: dict[str, Any], narrative: dict[str, Any]) -> dict[str, Any]:
        candidate_id = identifier(candidate_id)
        require(len(candidate_id) <= 50,
                "candidate_id is too long for the immutable A record IDs")
        require(isinstance(decision, dict)
                and set(decision) == set(CANDIDATE_DECISION_SCHEMA["properties"]),
                "research decision fields do not match the existing decision contract")
        require(decision["candidate_id"] == candidate_id,
                "research decision belongs to a different candidate")
        extras: list[str] = []
        narrative_data = model_fields(narrative, REPORT_SCHEMA, extras)
        require(not extras, "research report contains undeclared fields")
        params = {"candidate_id": candidate_id, "decision": decision,
                  "narrative": narrative_data}

        def finish():
            self._require_a_open()
            self._load_candidates()
            require(candidate_id in self.miner.candidates, "unknown A candidate")
            report_ref, _ = self._save_narrative("A", candidate_id, narrative_data)
            self._load_candidates()
            decisions = self.miner._check_candidate_decisions([decision], [candidate_id])
            current = decisions[candidate_id]
            candidate_item = self.miner.candidates[candidate_id]
            plan = candidate_item.get("experiment")
            if isinstance(plan, dict) and not current["continue_optimization"]:
                route_id = plan["route_id"]
                resolution_id = f"route-resolution-{route_id}"
                resolution_status = "paused" if current["disposition"] == "pause" else "stopped"
                resolution = self._a_record_map().get(resolution_id)
                if resolution is None:
                    self.miner.store.append(resolution_id, "route_resolution", {
                        "route_id": route_id, "status": resolution_status,
                        "reason": current["reason"], "candidate_id": candidate_id,
                        "action_id": action_id})
                else:
                    require(resolution["kind"] == "route_resolution"
                            and resolution["data"]["status"] == resolution_status,
                            "experiment route already has a different resolution")
                if route_id in self.miner.routes:
                    self.miner.routes[route_id]["decision"] = resolution_status
            existing_review = self._review_id_for_action(action_id)
            if existing_review is not None:
                record = self._a_record_map()[existing_review]
                require(record["data"]["candidate_decisions"] == [current],
                        "saved A decision differs from the submitted action")
                return {"candidate_id": candidate_id, "decision_ref": existing_review,
                        "report_ref": self._latest_report_id(candidate_id), "decision": current}
            review_records = self._decision_records()
            review_id = f"a-review-{len(review_records) + 1:06d}"
            round_no = self.miner.candidates[candidate_id]["round"]
            review_data = {
                "round_action": ("continue_optimization" if current["continue_optimization"]
                                 else "end_direction"),
                "candidate_decisions": [current], "information_request": None,
                "candidate_reports": [{"candidate_id": candidate_id, **narrative_data}],
                "analysis": "A research decision saved by the fixed domain action service.",
                "round_no": round_no, "round_candidate_ids": [candidate_id],
                "review_candidate_ids": [candidate_id], "report_candidate_ids": [candidate_id],
                "action_id": action_id}
            saved = self._a_record_map().get(review_id)
            if saved is None:
                self.miner.store.append(review_id, "research_review", review_data)
            else:
                require(saved["kind"] == "research_review" and saved["data"] == review_data,
                        "saved A decision record differs from the submitted action")
            self._load_candidates()
            return {"candidate_id": candidate_id, "decision_ref": review_id,
                    "report_ref": report_ref, "decision": current}

        def operation():
            return self._commit(action_id, finish)

        def recover():
            expected_report = {"candidate_id": candidate_id, **narrative_data}
            reports = [record for record in self._a_records()
                       if record["kind"] == "model_report"
                       and record["data"] == expected_report]
            if not reports:
                return operation()
            report_id = max(reports, key=_report_order)["id"]
            existing_review = self._review_id_for_action(action_id)
            if existing_review is not None:
                record = self._a_record_map()[existing_review]
                require(record["data"]["candidate_decisions"] == [decision],
                        "saved A decision differs from the submitted action")
                return {"candidate_id": candidate_id, "decision_ref": existing_review,
                        "report_ref": report_id, "decision": decision}
            return self._commit(action_id, finish)

        return self._run_action(action_id, "save_research_decision", params,
                                operation, recover=recover)

    def _latest_report_id(self, candidate_id: str) -> str | None:
        record = _latest_model_reports(self._a_records()).get(candidate_id)
        return None if record is None else record["id"]

    def _latest_decision(self, candidate_id: str) -> tuple[str, dict[str, Any]] | None:
        records = self._decision_records(candidate_id)
        if not records:
            return None
        record = records[-1]
        decision = next(row for row in record["data"]["candidate_decisions"]
                        if row["candidate_id"] == candidate_id)
        return record["id"], decision

    def _build_a_manifest(self, semantic_match_report: dict[str, Any]) -> dict[str, Any]:
        self._load_candidates()
        records = self._a_record_map()
        candidate_ids = sorted(record["data"]["id"] for record in records.values()
                               if record["kind"] == "candidate")
        require(semantic_match_report["objective"] == self._require_goal_spec()["objective"],
                "A semantic-match report objective differs from the fixed GoalSpec")
        matches = semantic_match_report["matches"]
        require(len(matches) == len(candidate_ids)
                and {match["candidate_id"] for match in matches} == set(candidate_ids),
                "A semantic-match report must cover every candidate exactly once")
        self._a_evidence_tools()
        candidate_evidence = {}
        evaluated_ids = []
        retained_ids = []
        quality = {}
        goal_spec = self._require_goal_spec()
        from .goal import GoalSpec
        from .goal_evidence import quality_evidence

        goal = GoalSpec(**goal_spec)
        for candidate_id in candidate_ids:
            item = self.miner.candidates[candidate_id]
            calculation = records.get(f"{candidate_id}-calculation")
            evaluation = records.get(f"{candidate_id}-evaluation")
            duplicate = records.get(f"{candidate_id}-duplicate")
            attempts = [record for record in records.values()
                        if record["kind"] == "calculation_attempt"
                        and record["data"].get("candidate_id") == candidate_id]
            require(calculation is not None or duplicate is not None,
                    "A manifest contains a candidate with an unfinished calculation")
            require(not attempts or calculation is not None,
                    "A manifest contains a formula repair that is still pending")
            if calculation is not None and calculation["data"]["status"] == "computed":
                require(evaluation is not None,
                        "computed A candidate is missing its evaluation record")
                evaluated_ids.append(candidate_id)
            report_id = self._latest_report_id(candidate_id)
            decision_result = self._latest_decision(candidate_id)
            require(report_id is not None and decision_result is not None,
                    "A manifest requires a final interpretation and decision for every candidate")
            decision_id, decision = decision_result
            require(not decision["continue_optimization"],
                    "A cannot be sealed while a candidate still requests optimization")
            plan = item.get("experiment")
            if isinstance(plan, dict):
                route_id = plan["route_id"]
                comparison = records.get(f"{candidate_id}-comparison")
                resolution = records.get(f"route-resolution-{route_id}")
                route_state = self.miner.routes[route_id]["decision"]
                require(comparison is not None or resolution is not None,
                        "A manifest contains an unresolved predeclared experiment")
                require(route_state not in {"continue", "pending"},
                        "A manifest cannot seal an open experiment route")
            refs = [f"{candidate_id}-definition"]
            if calculation is not None:
                refs.append(calculation["id"])
            if duplicate is not None:
                refs.append(duplicate["id"])
            refs.extend(record["id"] for record in attempts)
            refs.extend(record["id"] for record in records.values()
                        if record["kind"] == "formula_repair"
                        and record["data"].get("candidate_id") == candidate_id)
            if evaluation is not None:
                refs.append(evaluation["id"])
            comparison = records.get(f"{candidate_id}-comparison")
            if comparison is not None:
                refs.append(comparison["id"])
            if isinstance(plan, dict):
                refs.append(f"experiment-{plan['route_id']}")
                resolution = records.get(f"route-resolution-{plan['route_id']}")
                if resolution is not None:
                    refs.append(resolution["id"])
            refs.extend([report_id, decision_id])
            required_refs = set(refs)
            match = next(row for row in matches if row["candidate_id"] == candidate_id)
            supplied_refs = match["evidence_refs"]
            require(len(supplied_refs) == len(set(supplied_refs))
                    and required_refs <= set(supplied_refs)
                    and set(supplied_refs) <= set(records),
                    "semantic-match evidence must cite this candidate's complete A evidence")
            candidate_evidence[candidate_id] = {
                "decision_ref": decision_id, "report_ref": report_id,
                "program_evidence_refs": sorted(set(refs)),
                "semantic_match": {key: match[key] for key in
                                   ("candidate_id", "matches_goal", "reason", "evidence_refs")},
                "disposition": decision["disposition"],
                "retained_horizons": list(decision["retained_horizons"])}
            if decision["disposition"] == "retain":
                retained_ids.append(candidate_id)
                if goal.quality_target is not None:
                    quality[candidate_id] = quality_evidence(
                        goal, self.miner, candidate_id, decision["retained_horizons"])
        experiments = [{"record_id": record["id"], "plan": record["data"]["plan"],
                        "status": self.miner.routes[record["data"]["plan"]["route_id"]]["decision"]}
                       for record in records.values()
                       if record["kind"] == "modification_experiment"]
        for row in experiments:
            require(row["status"] not in {"continue", "pending"},
                    "A manifest cannot seal while a registered experiment route remains open")
        return {"run_id": self.miner.spec.run_id, "objective": goal.objective,
                "goal_spec": goal_spec, "candidate_ids": candidate_ids,
                "evaluated_ids": sorted(evaluated_ids),
                "retained_ids": sorted(retained_ids), "experiments": experiments,
                "candidate_evidence": candidate_evidence,
                "semantic_match_report": semantic_match_report,
                "quality_evidence": quality,
                "a_record_ids": sorted(records)}

    def finish_a_research(self, *, action_id: str,
                          semantic_match_report: dict[str, Any]) -> dict[str, Any]:
        require(isinstance(semantic_match_report, dict)
                and set(semantic_match_report) == {"objective", "matches"},
                "semantic-match report fields do not match the A completion contract")
        text(semantic_match_report["objective"], "semantic-match objective")
        require(isinstance(semantic_match_report["matches"], list),
                "semantic-match matches must be a list")
        normalized_matches = []
        for row in semantic_match_report["matches"]:
            require(isinstance(row, dict)
                    and set(row) == {"candidate_id", "matches_goal", "reason", "evidence_refs"},
                    "semantic-match row fields do not match the A completion contract")
            identifier(row["candidate_id"])
            require(len(row["candidate_id"]) <= 50,
                    "candidate_id is too long for the immutable A record IDs")
            require(type(row["matches_goal"]) is bool, "matches_goal must be boolean")
            text(row["reason"], "semantic-match reason")
            require(isinstance(row["evidence_refs"], list) and row["evidence_refs"],
                    "semantic-match evidence references must be nonempty")
            normalized_matches.append({**row, "evidence_refs": [identifier(ref) for ref in row["evidence_refs"]]})
        semantic_match_report = {"objective": semantic_match_report["objective"],
                                 "matches": normalized_matches}
        params = {"semantic_match_report": semantic_match_report,
                  "goal_spec": self._require_goal_spec()}

        def write_completion(manifest: dict[str, Any]):
            manifest_ref = "a-manifest"
            completion = {"status": "complete", "run_id": self.miner.spec.run_id,
                          "candidate_ids": manifest["candidate_ids"],
                          "evaluated_ids": manifest["evaluated_ids"],
                          "retained_ids": manifest["retained_ids"],
                          "pending_report_ids": [], "semantic_match_ref": manifest_ref,
                          "manifest_ref": manifest_ref, "frozen": False,
                          "goal_status": "awaiting_B", "objective_met": False}
            checkpoint = self.miner.root / "a-complete.json"
            if checkpoint.exists():
                require(json.loads(checkpoint.read_text(encoding="utf-8")) == completion,
                        "saved A completion checkpoint differs from its manifest")
            else:
                write_json(checkpoint, completion)
            return completion

        def complete():
            manifest = self._build_a_manifest(semantic_match_report)
            prior = self._a_record_map().get("a-manifest")
            if prior is None:
                self.miner.store.append("a-manifest", "a_manifest", manifest)
            else:
                require(prior["kind"] == "a_manifest" and prior["data"] == manifest,
                        "saved A manifest differs from the completed candidate evidence")
                manifest = prior["data"]
            return write_completion(manifest)

        def operation():
            self._require_a_open()
            self._ensure_a_initialized(action_id)
            return self._commit(action_id, complete)

        def recover():
            completion_path = self.miner.root / "a-complete.json"
            if completion_path.exists():
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                require(completion["manifest_ref"] == "a-manifest",
                        "saved A completion points to another manifest")
                manifest = self._a_record_map().get("a-manifest")
                require(manifest is not None and manifest["kind"] == "a_manifest"
                        and manifest["data"]["semantic_match_report"] == semantic_match_report,
                        "saved A manifest differs from the pending completion action")
                return completion
            prior = self._a_record_map().get("a-manifest")
            if prior is not None:
                require(prior["kind"] == "a_manifest"
                        and prior["data"]["semantic_match_report"] == semantic_match_report
                        and prior["data"]["goal_spec"] == params["goal_spec"],
                        "pending A seal has a different immutable manifest")
                return self._commit(action_id, lambda: write_completion(prior["data"]))
            self._require_a_open()
            self._ensure_a_initialized(action_id)
            return self._commit(action_id, complete)

        def preflight():
            self._require_no_pending_a_actions(action_id)
            self._require_a_open()
            self._build_a_manifest(semantic_match_report)

        return self._run_action(action_id, "finish_a_research", params,
                                operation, recover=recover, preflight=preflight)

    def _completed_a_manifest(self, manifest_ref: str) -> dict[str, Any]:
        manifest_ref = identifier(manifest_ref)
        completion_path = self.miner.root / "a-complete.json"
        require(completion_path.is_file(), "complete and seal A before freezing B")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        require(completion.get("manifest_ref") == manifest_ref,
                "freeze must use the exact manifest referenced by A completion")
        record = self._a_record_map().get(manifest_ref)
        require(record is not None and record["kind"] == "a_manifest",
                "A completion references a missing or invalid manifest record")
        manifest = record["data"]
        require(completion["candidate_ids"] == manifest["candidate_ids"]
                and completion["retained_ids"] == manifest["retained_ids"]
                and completion["evaluated_ids"] == manifest["evaluated_ids"],
                "A completion checkpoint differs from its manifest")
        return manifest

    def freeze_a_manifest(self, *, action_id: str, manifest_ref: str) -> dict[str, Any]:
        manifest_ref = identifier(manifest_ref)
        require(self.validation_universe is not None,
                "complete B membership is not fixed in the coordinator service binding")
        parameters = {"manifest_ref": manifest_ref,
                      "universe_sha256": hashlib.sha256(
                          self.validation_universe.rename("eligible").to_csv().encode("utf-8")).hexdigest()}

        def freeze():
            manifest = self._completed_a_manifest(manifest_ref)
            candidate_ids = list(manifest["retained_ids"])
            require(candidate_ids, "A completed without retained candidates; B cannot be opened")
            frozen_path = self.miner.root / "frozen_batch.json"
            if frozen_path.exists():
                frozen = self.miner._checked_frozen()
                require(frozen.get("a_manifest_ref") == manifest_ref
                        and list(frozen["candidates"]) == candidate_ids,
                        "existing B freeze differs from the completed A manifest")
                return {"manifest_ref": manifest_ref, "candidate_ids": candidate_ids,
                        "frozen": True}
            self.miner.freeze(candidate_ids, self.validation_universe,
                              a_manifest_ref=manifest_ref)
            return {"manifest_ref": manifest_ref, "candidate_ids": candidate_ids,
                    "frozen": True}

        def operation():
            return self._commit(action_id, freeze)

        def recover():
            frozen_path = self.miner.root / "frozen_batch.json"
            if frozen_path.exists():
                frozen = self.miner._checked_frozen()
                manifest = self._completed_a_manifest(manifest_ref)
                require(frozen.get("a_manifest_ref") == manifest_ref
                        and list(frozen["candidates"]) == list(manifest["retained_ids"]),
                        "saved frozen batch differs from the pending manifest action")
                return {"manifest_ref": manifest_ref,
                        "candidate_ids": list(manifest["retained_ids"]), "frozen": True}
            return self._commit(action_id, freeze)

        return self._run_action(action_id, "freeze_a_manifest", parameters,
                                operation, recover=recover,
                                preflight=lambda: self._require_no_pending_a_actions(action_id))

    def validate_frozen_batch(self, *, action_id: str) -> dict[str, Any]:
        require((self.miner.root / "frozen_batch.json").is_file(),
                "freeze the completed A manifest before B validation")
        require(self.b_panel_loader is not None,
                "B panel loader is not fixed in the research service binding")
        frozen = self.miner._checked_frozen()
        parameters = {"manifest_ref": frozen.get("a_manifest_ref"),
                      "candidate_ids": list(frozen["candidates"]),
                      "validation_universe_sha256": (
                          hashlib.sha256(self.validation_universe.rename("eligible").to_csv()
                                         .encode("utf-8")).hexdigest()
                          if self.validation_universe is not None else None)}

        def operation():
            return self.miner.validate_numerical(
                self.b_panel_loader,
                commit=lambda write: self._commit(action_id, write),
                stop_check=self.ledger.require_active)

        def recover():
            if (self.miner.root / "b-numerical-complete.json").is_file():
                return read_b_checkpoint(self.miner)
            require(not (self.miner.root / "b-access-started.json").exists(),
                    "B access started without a complete numerical checkpoint; inspect saved evidence before retrying")
            return operation()

        return self._run_action(action_id, "validate_frozen_batch", parameters,
                                operation, recover=recover)

    def read_b_evidence(self, *, record_id: str, pointer: str,
                        offset: int, limit: int) -> dict[str, Any]:
        read_b_checkpoint(self.miner)
        store = self._b_store()
        saved_ids = {record["id"] for record in store.all()}
        record_id = identifier(record_id)
        require(record_id in saved_ids, "B evidence record is not in the fixed run store")
        try:
            value = store.read(record_id, pointer, offset, limit)
        except InvalidRecordPointer as exc:
            raise RecoverableToolError(
                f"JSON pointer {pointer!r} does not resolve in B record {record_id}; "
                "read_b_result supplies exact numerical and decision references; "
                "root keys: " + ", ".join(sorted(store.read(record_id, "", 0, 1)))) from exc
        result = {"record_id": record_id, "value": value}
        if len(dumps(result).encode("utf-8")) > 16 * 1024:
            raise RecoverableToolError(
                f"B evidence response exceeds 16 KiB; no content returned. "
                f"Use a narrower pointer in record_id={record_id}; root keys: "
                + ", ".join(sorted(store.read(record_id, "", 0, 1))))
        return result

    def read_b_result(self, *, candidate_id: str) -> dict[str, Any]:
        candidate_id = identifier(candidate_id)
        checkpoint = read_b_checkpoint(self.miner)
        require(candidate_id in checkpoint["candidate_ids"], "candidate is outside the frozen B batch")
        records = {record["id"]: record for record in self._b_store().all()}
        evaluation = records[f"{candidate_id}-evaluation"]
        decision = records[f"{candidate_id}-validation"]
        report = _latest_model_reports(list(records.values())).get(candidate_id)
        result = {"run_id": self.miner.spec.run_id, "candidate_id": candidate_id,
                  "numerical_complete": True,
                  "delivery_complete": (self.miner.root / "b-delivery.json").is_file(),
                  "decision": {"record_id": decision["id"], "record_sha256": digest(decision),
                               "data": decision["data"]},
                  "report": {"record_id": report["id"], "record_sha256": digest(report),
                             "data": report["data"]} if report else None,
                  "numerical": {"record_id": evaluation["id"], "record_sha256": digest(evaluation),
                                "horizons": {key: {"summary": value["summary"], "coverage": value["coverage"]}
                                             for key, value in evaluation["data"]["horizons"].items()}},
                  "batch_correction_ref": "batch-correction"}
        if len(dumps(result).encode("utf-8")) > 32 * 1024:
            raise RecoverableToolError(f"B result exceeds 32 KiB; no content returned. "
                                       f"Read {candidate_id}-validation and {candidate_id}-evaluation "
                                       "through read_b_evidence with narrow pointers.")
        return result

    def save_b_interpretation(self, *, action_id: str, candidate_id: str,
                              narrative: dict[str, Any]) -> dict[str, Any]:
        candidate_id = identifier(candidate_id)
        require(len(candidate_id) <= 50,
                "candidate_id is too long for the immutable B record IDs")
        extras: list[str] = []
        narrative_data = model_fields(narrative, REPORT_SCHEMA, extras)
        require(not extras, "B interpretation contains undeclared fields")
        parameters = {"candidate_id": candidate_id, "narrative": narrative_data}
        action_record_id = f"b-action-{hashlib.sha256(action_id.encode('utf-8')).hexdigest()[:24]}"

        def save_action_record(report_ref: str):
            data = {"action_id": action_id, "candidate_id": candidate_id,
                    "report_ref": report_ref, "narrative_sha256": digest(narrative_data)}
            records = {record["id"]: record for record in self._b_store().all()}
            prior = records.get(action_record_id)
            if prior is None:
                self._commit(action_id, lambda: self._b_store().append(
                    action_record_id, "b_interpretation_action", data))
            else:
                require(prior["kind"] == "b_interpretation_action" and prior["data"] == data,
                        "B interpretation action receipt differs from its submitted evidence")
            return {"candidate_id": candidate_id, "report_ref": report_ref}

        def operation():
            require(not (self.miner.root / "b-delivery.json").exists(),
                    "B interpretations are immutable after final delivery")
            read_b_checkpoint(self.miner)
            self._load_candidates()
            report_ref, _ = self._commit(
                action_id, lambda: self._save_narrative("B", candidate_id, narrative_data))
            return save_action_record(report_ref)

        def recover():
            require(not (self.miner.root / "b-delivery.json").exists(),
                    "B interpretations are immutable after final delivery")
            records = self._b_store().all()
            by_id = {record["id"]: record for record in records}
            saved_action = by_id.get(action_record_id)
            if saved_action is not None:
                data = saved_action["data"]
                require(saved_action["kind"] == "b_interpretation_action"
                        and data["action_id"] == action_id
                        and data["candidate_id"] == candidate_id
                        and data["narrative_sha256"] == digest(narrative_data),
                        "pending B interpretation differs from its saved action receipt")
                report = by_id.get(data["report_ref"])
                require(report is not None and report["kind"] == "model_report"
                        and report["data"] == {"candidate_id": candidate_id, **narrative_data},
                        "B interpretation action points to a missing or different report")
                return {"candidate_id": candidate_id, "report_ref": data["report_ref"]}
            expected_report = {"candidate_id": candidate_id, **narrative_data}
            matches = [record for record in records if record["kind"] == "model_report"
                       and record["data"] == expected_report]
            if matches:
                return save_action_record(max(matches, key=_report_order)["id"])
            return operation()

        return self._run_action(action_id, "save_b_interpretation", parameters,
                                operation, recover=recover)

    def finalize_b_delivery(self, *, action_id: str) -> dict[str, Any]:
        require(self.idea_pool is not None,
                "idea-pool destination is not fixed in the research service binding")
        parameters = {"idea_pool": str(self.idea_pool.resolve())}

        def finish():
            numeric = read_b_checkpoint(self.miner)
            result = complete_b_reports_program(
                self.miner, self.idea_pool,
                commit=lambda write: self._commit(action_id, write),
                stop_check=self.ledger.require_active)
            if result["status"] != "complete":
                return result
            validation = json.loads((self.miner.root / "validation.json").read_text(encoding="utf-8"))
            records = {record["id"]: record for record in self._b_store().all()}
            delivery_candidates = []
            eligible_ids = []
            for candidate_id, decision in validation["decisions"].items():
                candidate_report = _latest_model_reports(list(records.values())).get(candidate_id)
                require(candidate_report is not None,
                        "completed B delivery is missing a candidate interpretation")
                item = {"candidate_id": candidate_id,
                        "report_ref": candidate_report["id"],
                        "validation_ref": f"{candidate_id}-validation",
                        "eligible_for_idea_pool": decision["eligible_for_idea_pool"]}
                if decision["eligible_for_idea_pool"]:
                    card_path = Path(decision["idea_card"]).resolve()
                    require(card_path.is_file(), "eligible B candidate is missing its delivered idea card")
                    card = json.loads(card_path.read_text(encoding="utf-8"))
                    require(card.get("id") == f"{self.miner.spec.run_id}--{candidate_id}",
                            "delivered idea card does not match its frozen candidate")
                    item.update(card_path=str(card_path),
                                card_sha256=hashlib.sha256(card_path.read_bytes()).hexdigest(),
                                idea_id=card["id"])
                    eligible_ids.append(candidate_id)
                delivery_candidates.append(item)
            manifest = {"status": "complete", "run_id": self.miner.spec.run_id,
                        "a_manifest_ref": self.miner._checked_frozen().get("a_manifest_ref"),
                        "candidate_ids": numeric["candidate_ids"],
                        "batch_correction_ref": "batch-correction",
                        "validation_ref": "validation.json", "report_file": "B-report.md",
                        "candidates": delivery_candidates,
                        "eligible_candidate_ids": sorted(eligible_ids)}
            delivery_path = self.miner.root / "b-delivery.json"
            if delivery_path.exists():
                require(json.loads(delivery_path.read_text(encoding="utf-8")) == manifest,
                        "saved B delivery receipt differs from the verified result")
            else:
                self._commit(action_id, lambda: write_json(delivery_path, manifest))
            return {**result, "delivery_ref": "b-delivery.json",
                    "eligible_candidate_ids": sorted(eligible_ids)}

        def operation():
            return finish()

        def recover():
            path = self.miner.root / "b-delivery.json"
            if path.is_file():
                saved = json.loads(path.read_text(encoding="utf-8"))
                require(saved.get("status") == "complete"
                        and saved.get("run_id") == self.miner.spec.run_id
                        and saved.get("a_manifest_ref")
                        == self.miner._checked_frozen().get("a_manifest_ref"),
                        "pending B delivery action has a mismatched saved receipt")
            return finish()

        return self._run_action(action_id, "finalize_b_delivery", parameters,
                                operation, recover=recover)

    def request_stop(self, *, action_id: str, reason: str) -> dict[str, Any]:
        return self.ledger.control(action_id, "stop", reason=reason)

    def resume_actions(self, *, action_id: str) -> dict[str, Any]:
        return self.ledger.control(action_id, "resume")

    def calculate_formula(self, expression: str, panel: FactorInputPanel):
        compiled = compile_factor_formula(expression, self.miner.spec)
        return compiled, evaluate_expression(expression, panel)

    def calculate_candidate(self, candidate_id: str, panel: FactorInputPanel):
        self._require_legacy_writer_unbound()
        if candidate_id not in self.miner.candidates:
            self.miner._load_candidates()
        require(candidate_id in self.miner.candidates,
                "calculation candidate is not in this research run")
        item = self.miner.candidates[candidate_id]
        definition = item["definition"]
        expression = definition["expression"]
        compiled = None
        result = None
        try:
            compiled = compile_factor_formula(expression, self.miner.spec)
            result = evaluate_expression(expression, panel)
        except ValueError as exc:
            calculation = {"status": "calculation_failed",
                           "original_expression": expression,
                           "original_meaning": definition["meaning"],
                           "checks": [{"attempt": 0, "expression": expression,
                                       "actual_steps": compiled.calculation_steps() if compiled else None,
                                       "program_error": str(exc)}],
                           "executed_expression": None}
        else:
            calculation = {"status": "computed", "original_expression": expression,
                           "original_meaning": definition["meaning"],
                           "checks": [{"attempt": 0, "expression": expression,
                                       "actual_steps": compiled.calculation_steps(),
                                       "program_error": None}],
                           "executed_expression": result.definition,
                           "valid_values": int(result.values.notna().sum()),
                           "values_artifact": archive_factor_values(
                               self.miner.archive_root, self.miner.root, self.miner.spec,
                               candidate_id, definition, result)}
        data = {"candidate_id": candidate_id, **calculation}
        record_id = f"{candidate_id}-calculation"
        saved = {record["id"]: record for record in self.miner.store.all()}
        if record_id in saved:
            require(saved[record_id]["kind"] == "calculation"
                    and saved[record_id]["data"] == data,
                    "saved calculation differs from program execution")
        else:
            self.miner.store.append(record_id, "calculation", data)
        item["calculation"] = calculation
        return result.values if result is not None else None

    def evaluate_a(self, values: pd.Series, panel: FactorInputPanel, direction: int):
        labels = {horizon: build_labels(panel, self.miner.spec, "A", horizon_hours=horizon)
                  for horizon in (1, 4, 24)}
        return evaluate_a_candidate(values, labels, self.miner.spec, direction)

    def evaluate_d(self, values: pd.Series, panel: FactorInputPanel, stage: str = "A"):
        return evaluate_rank_displacement(values, panel.universe, self.miner.spec, stage,
                                          stop_check=self.ledger.require_active)

    def compare_candidates(self, trial_values: pd.Series, control_values: pd.Series,
                           panel: FactorInputPanel, labels: pd.DataFrame,
                           trial_direction: int, control_direction: int,
                           plan: dict[str, Any]) -> dict[str, Any]:
        return compare_candidates(trial_values, control_values, panel, labels, self.miner.spec,
                                  trial_direction, control_direction, plan,
                                  self.miner.rank_displacement_version)

    def archive_factor_values(self, candidate_id: str, definition: dict[str, Any], result):
        self._require_legacy_writer_unbound()
        return {"values_artifact": archive_factor_values(
            self.miner.archive_root, self.miner.root, self.miner.spec,
            candidate_id, definition, result)}

    def freeze(self, candidate_ids: list[str], validation_universe: pd.Series):
        self._require_legacy_writer_unbound()
        return self.miner.freeze(candidate_ids, validation_universe)

    def submit_report(self, stage: str, candidate_id: str, narrative: dict[str, Any]):
        self._require_legacy_writer_unbound()
        require(stage in {"A", "B"}, "report stage must be A or B")
        if candidate_id not in self.miner.candidates:
            self.miner._load_candidates()
        require(candidate_id in self.miner.candidates,
                "report candidate is not in this research run")
        candidate = self.miner.candidates[candidate_id]
        if stage == "A":
            require(not (self.miner.root / "frozen_batch.json").exists(),
                    "A evidence cannot change after freezing B")
            require("evaluation" in candidate, "A report requires saved evaluation evidence")
            store = self.miner.store
        else:
            require((self.miner.root / "b-numerical-complete.json").exists(),
                    "B report requires complete numerical validation")
            frozen = self.miner._checked_frozen()
            require(candidate_id in frozen["candidates"],
                    "B report candidate is not in the frozen batch")
            store = RecordStore(self.miner.root / "b_records", run_root=self.miner.root.resolve(),
                                archive_root=self.miner.archive_root)
        extras: list[str] = []
        parsed = model_fields(narrative, REPORT_SCHEMA, extras)
        require(not extras, "research report contains undeclared fields")
        data = {"candidate_id": candidate_id, **parsed}
        record_id = f"{candidate_id}-report"
        saved = {record["id"]: record for record in store.all()}
        if record_id in saved:
            require(saved[record_id]["kind"] == "model_report" and saved[record_id]["data"] == data,
                    "saved research report differs from submitted report")
        else:
            store.append(record_id, "model_report", data)
        if stage == "A":
            candidate["model_report"] = data
        return data

    def complete_b_reports(self, idea_pool: Path):
        self._require_legacy_writer_unbound()
        return complete_b_reports_program(self.miner, idea_pool)

    def prepare_goal_review(self, goal_runner, validation: dict[str, Any]):
        from .goal_evidence import prepare_goal_review

        return prepare_goal_review(goal_runner, self.miner, validation)

    def submit_goal_match(self, goal_runner, match_id: str, idea_ids: set[str],
                          report: dict[str, Any]):
        from .goal_evidence import save_goal_match

        return save_goal_match(goal_runner.receipts, match_id, idea_ids, report)

    def check_goal_evidence(self, goal_runner=None):
        if goal_runner is not None:
            from .goal_evidence import check_goal_evidence

            return check_goal_evidence(goal_runner.root)
        require(self.goal_spec is not None,
                "native goal evidence requires the fixed GoalSpec service binding")
        from .goal_evidence import check_native_goal_evidence

        return check_native_goal_evidence(self.miner, self.goal_spec)

    def correct_and_admit(self, reports: dict[str, dict[str, Any]]):
        from .workflow import b_candidate_decision

        correction = correct_batch(reports, self.miner.spec)
        decisions = {candidate_id: b_candidate_decision(
            candidate_id, report, correction, self.miner.spec)
            for candidate_id, report in reports.items()}
        return {"batch_correction": correction, "decisions": decisions}


def complete_b_reports_program(miner, idea_pool: Path, *, commit=None,
                               stop_check=None) -> dict[str, Any]:
    """Verify saved B numbers and submitted reports, then write deterministic delivery artifacts."""
    from .workflow import b_candidate_decision

    def commit_write(operation):
        if stop_check is not None:
            stop_check()
        return operation() if commit is None else commit(operation)

    frozen = miner._checked_frozen()
    if frozen.get("a_manifest_ref") is not None:
        require(commit is not None and stop_check is not None,
                "native B delivery requires the action-bound commit and stop hooks")
    require((miner.root / "b-numerical-complete.json").exists(), "complete B numerical validation before completing reports")
    require((miner.root / "b-access-started.json").exists(), "B access marker is missing")
    store = RecordStore(miner.root / "b_records", run_root=miner.root.resolve(),
                        archive_root=miner.archive_root)
    records = store.all()
    evaluation_records = [record for record in records if record["kind"] == "evaluation"]
    validation_records = [record for record in records if record["kind"] == "validation_result"]
    correction_records = [record for record in records if record["kind"] == "multiple_testing"]
    require(len(evaluation_records) == len(validation_records) == len(frozen["candidates"])
            and len(correction_records) == 1, "B numerical records are incomplete or repeated")
    reports = {record["data"]["candidate_id"]: {
        key: value for key, value in record["data"].items() if key != "candidate_id"}
        for record in evaluation_records}
    decisions = {record["data"]["candidate_id"]: {
        key: value for key, value in record["data"].items() if key != "candidate_id"}
        for record in validation_records}
    require(set(reports) == set(decisions) == set(frozen["candidates"]), "B numerical batch is incomplete")
    expected_checkpoint = {"candidate_ids": list(frozen["candidates"]),
                           "retained_horizons": {cid: item["retained_horizons"]
                                                 for cid, item in frozen["candidates"].items()}}
    require(json.loads((miner.root / "b-numerical-complete.json").read_text()) == expected_checkpoint,
            "B numerical checkpoint differs from the frozen batch")
    for cid, report in reports.items():
        frozen_item = frozen["candidates"][cid]
        require(report["segment"] == "B"
                and report["direction"] == frozen["candidates"][cid]["definition"]["direction"]
                and report["retained_horizons"] == frozen["candidates"][cid]["retained_horizons"]
                and set(report["horizons"]) == {str(horizon) for horizon in report["retained_horizons"]},
                "B evaluation horizons differ from the frozen batch")
        if frozen_item.get("rank_displacement_version") is not None:
            displacement = report.get("rank_displacement")
            displacement_locator = report.get("rank_displacement_archive")
            expected_displacement_key = EvaluationKey(
                f"{miner.spec.run_id}/B", dumps(miner.spec.as_dict()),
                frozen_item["rank_displacement_version"], "B", "rank-displacement").as_dict()
            require(frozen_item["rank_displacement_version"] == miner.rank_displacement_version
                    == RANK_DISPLACEMENT_VERSION
                    and displacement is not None and displacement_locator is not None
                    and displacement.get("definition_version") == frozen_item["rank_displacement_version"]
                    and displacement.get("segment") == "B"
                    and displacement_locator["identity"]
                    == report["horizons"][str(report["retained_horizons"][0])]["factor_archive"]["identity"]
                    and displacement_locator["evaluation_key"] == expected_displacement_key,
                    "B rank-displacement diagnostic differs from frozen A definition")
        else:
            require("rank_displacement" not in report and "rank_displacement_archive" not in report
                    and miner.rank_displacement_version is None,
                    "legacy frozen B cannot receive new rank-displacement diagnostics")
        identity = miner._factor_identity(
            frozen["candidates"][cid]["executed"]["expanded_expression"], report["direction"]).as_dict()
        for horizon in report["retained_horizons"]:
            horizon_report = report["horizons"][str(horizon)]
            locator = horizon_report["factor_archive"]
            require(horizon_report["segment"] == "B"
                    and horizon_report["direction"] == frozen["candidates"][cid]["definition"]["direction"]
                    and horizon_report["horizon_hours"] == horizon
                    and locator["identity"] == identity
                    and locator["evaluation_key"] == miner._evaluation_key("B", horizon).as_dict(),
                    "B archive identity differs from its frozen formula, direction or horizon")
    correction = correction_records[0]["data"]
    require(correction == correct_batch(reports, miner.spec), "B batch correction differs from saved evaluations")
    for cid, report in reports.items():
        expected = b_candidate_decision(cid, report, correction, miner.spec)
        require(decisions[cid] == expected, "B program decision differs from saved evaluation")
    completed_path = miner.root / "validation.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text())
        prior_decisions = {cid: {key: value for key, value in decision.items() if key != "idea_card"}
                           for cid, decision in completed["decisions"].items()}
        require(prior_decisions == decisions and completed["batch_correction"] == correction,
                "completed validation differs from saved evidence")
    narratives = _latest_model_reports(records)
    pending = [cid for cid in reports if cid not in narratives]
    for cid, decision in decisions.items():
        if decision["eligible_for_idea_pool"] and cid in narratives:
            narrative = narratives[cid]
            card = miner._idea_card(cid, frozen["candidates"][cid], reports[cid], narrative["data"], decision,
                                   narrative["created_at"])
            path = Path(idea_pool) / f"{miner.spec.run_id}--{cid}.json"
            if path.exists():
                existing_card = json.loads(path.read_text())
                require(without_hash_metadata(existing_card) == without_hash_metadata(card),
                        "existing idea card differs from saved research evidence")
            else:
                commit_write(lambda path=path, card=card: write_json(path, card))
            decision["idea_card"] = str(path)
    output = {"batch_correction": correction, "decisions": decisions,
              "status": "reports_pending" if pending else "complete", "pending_report_ids": pending,
              "B_use": "used for fixed-candidate selection; not untouched data for subsequent strategy development",
              "C_use": "reserved; never loaded by factor mining"}
    commit_write(lambda: write_research_report(
        miner.root / "B-report.md", miner.spec.run_id, miner.spec.purpose,
        records, "B", replace=True))
    if not pending:
        path = miner.root / "validation.json"
        if path.exists():
            prior = json.loads(path.read_text())
            require(prior == output, "completed validation differs from saved evidence")
        else:
            commit_write(lambda: write_json(path, output))
    return output
