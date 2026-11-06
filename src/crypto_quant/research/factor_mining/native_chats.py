"""Fixed bindings and role-scoped adapters for the native research chats.

The binding is operator supplied at MCP process startup. No tool accepts paths,
chat IDs, or a caller-selected role.
"""

from __future__ import annotations

import hashlib
import json
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from crypto_quant.features.factor_expressions import operator_catalog, template_catalog
from crypto_quant.features.factor_inputs import FactorInputPanel, input_catalog

from .contracts import ResearchSpec, digest, require, text
from .records import write_json


NativeRole = Literal["coordinator", "A", "B"]
_ROLES: tuple[NativeRole, ...] = ("coordinator", "A", "B")
_RECOVERY_TASK_TOOL = {
    "type": "function",
    "name": "prepare_a_recovery_task",
    "description": ("Prepare an immutable A recovery ticket from a previously accepted A task. "
                    "The new ticket references the prior dispatch, current A evidence, and pending action IDs."),
    "inputSchema": {
        "type": "object",
        "required": ["task_id"],
        "properties": {
            "task_id": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        },
        "additionalProperties": False,
    },
}
_PREPARE_B_TASK_TOOL = {
    "type": "function",
    "name": "prepare_b_task",
    "description": ("Prepare an immutable B validation ticket from the completed A manifest "
                    "and the exact frozen batch, without opening B market data."),
    "inputSchema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class ChatReference:
    thread_id: str
    host_id: str


@dataclass(frozen=True)
class NativeChatBinding:
    """Immutable target, contract, inputs and three native chat identities."""

    research_id: str
    target: dict[str, Any]
    goal: dict[str, Any]
    run_root: Path
    contract_sha256: str
    market_db: Path
    universe_csv: Path
    include_liquidations: bool
    idea_pool: Path
    chats: dict[NativeRole, ChatReference]
    binding_path: Path
    spec: ResearchSpec

    @classmethod
    def load(cls, binding_path: str | Path) -> "NativeChatBinding":
        path = Path(binding_path).resolve(strict=True)
        raw = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(raw, dict), "native chat binding must be an object")
        require(set(raw) == {
            "format_version", "research_id", "target", "goal", "run_root",
            "contract_sha256", "inputs", "chats",
        }, "native chat binding fields do not match the supported format")
        require(raw["format_version"] == 1, "unsupported native chat binding format")

        research_id = text(raw["research_id"], "research_id")
        target = raw["target"]
        require(isinstance(target, dict)
                and set(target) == {"version", "objective"}
                and type(target["version"]) is int and target["version"] > 0,
                "native chat target must declare a positive version and objective")
        text(target["objective"], "target objective")

        from .goal import GoalSpec

        goal_data = raw["goal"]
        require(isinstance(goal_data, dict), "bound GoalSpec must be an object")
        goal = GoalSpec(**goal_data).as_dict()
        require(goal["objective"] == target["objective"],
                "fixed chat target must match the bound GoalSpec objective")

        run_root = _resolve(path.parent, raw["run_root"])
        require(run_root.is_dir(), "bound factor-mining run directory does not exist")
        contract_path = run_root / "contract.json"
        require(contract_path.is_file(), "bound factor-mining contract is missing")
        contract_bytes = contract_path.read_bytes()
        contract_sha256 = raw["contract_sha256"]
        require(isinstance(contract_sha256, str)
                and hashlib.sha256(contract_bytes).hexdigest() == contract_sha256,
                "bound factor-mining contract digest does not match")
        spec = ResearchSpec.from_dict(json.loads(contract_bytes))
        require(spec.run_id == research_id, "research_id differs from the saved contract run_id")
        require(spec.objective == target["objective"],
                "fixed chat target differs from the saved research contract")

        inputs = raw["inputs"]
        require(isinstance(inputs, dict) and set(inputs) == {
            "market_db", "universe_csv", "include_liquidations", "idea_pool",
        }, "native chat input binding fields do not match the supported format")
        require(type(inputs["include_liquidations"]) is bool,
                "include_liquidations must be explicitly boolean")
        market_db = _resolve(path.parent, inputs["market_db"])
        universe_csv = _resolve(path.parent, inputs["universe_csv"])
        idea_pool = _resolve(path.parent, inputs["idea_pool"])
        require(market_db.is_file(), "bound market database does not exist")
        require(universe_csv.is_file(), "bound membership file does not exist")
        require(idea_pool.is_dir(), "bound idea pool directory does not exist")

        chat_data = raw["chats"]
        require(isinstance(chat_data, dict) and set(chat_data) == set(_ROLES),
                "binding must contain coordinator, A and B native chats")
        chats: dict[NativeRole, ChatReference] = {}
        for role in _ROLES:
            reference = chat_data[role]
            require(isinstance(reference, dict)
                    and set(reference) == {"thread_id", "host_id"},
                    f"{role} chat binding must declare thread_id and host_id")
            thread_id = text(reference["thread_id"], f"{role} thread_id")
            host_id = text(reference["host_id"], f"{role} host_id")
            chats[role] = ChatReference(thread_id=thread_id, host_id=host_id)
        require(len({item.thread_id for item in chats.values()}) == len(_ROLES),
                "coordinator, A and B must bind distinct native threads")

        return cls(
            research_id=research_id, target=dict(target), goal=goal, run_root=run_root,
            contract_sha256=contract_sha256, market_db=market_db,
            universe_csv=universe_csv,
            include_liquidations=inputs["include_liquidations"], idea_pool=idea_pool,
            chats=chats, binding_path=path, spec=spec,
        )


def _resolve(base: Path, configured: Any) -> Path:
    require(isinstance(configured, str) and configured.strip(),
            "bound paths must be non-empty strings")
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _content_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _a_coverage(panel: FactorInputPanel) -> dict[str, Any]:
    diagnostics = panel.diagnostics
    symbols = diagnostics.get("symbols")
    require(isinstance(symbols, dict), "A panel coverage diagnostics are malformed")
    catalog = input_catalog()
    field_names = [field["name"] for field in catalog["fields"]]
    field_coverage = {}
    for field_name in field_names:
        eligible_rows = 0
        eligible_valid_rows = 0
        missing_reasons: Counter[str] = Counter()
        for symbol in symbols.values():
            coverage = symbol.get("coverage", {}).get(field_name)
            require(isinstance(coverage, dict), f"A panel coverage is missing field {field_name}")
            eligible_rows += coverage["eligible_rows"]
            eligible_valid_rows += coverage["eligible_valid_rows"]
            missing_reasons.update(coverage["missing_reasons"])
        field_coverage[field_name] = {
            "eligible_rows": eligible_rows,
            "eligible_valid_rows": eligible_valid_rows,
            "eligible_coverage_ratio": (eligible_valid_rows / eligible_rows
                                         if eligible_rows else None),
            "missing_reasons": dict(sorted(missing_reasons.items())),
        }
    covered = sorted(name for name, item in field_coverage.items()
                     if item["eligible_valid_rows"] > 0)
    return {
        "loaded_start": diagnostics["start"],
        "loaded_end_inclusive": diagnostics["end"],
        "eligible_rows": int(panel.universe.sum()),
        "eligible_symbols": int(panel.universe.groupby(level="symbol").any().sum()),
        "fields": len(field_names),
        "covered_fields": len(covered),
        "covered_field_names": covered,
        "uncovered_fields": sorted(set(field_names) - set(covered)),
        "field_coverage": field_coverage,
    }


def _save_a_environment_record(binding: NativeChatBinding, actions: Any,
                               panel: FactorInputPanel) -> dict[str, str]:
    panel_sha256 = actions._panel_fingerprint(panel)
    action_binding_path = actions.miner.root / "a-action-input-binding.json"
    if action_binding_path.is_file():
        action_binding = json.loads(action_binding_path.read_text(encoding="utf-8"))
        require(action_binding["run_id"] == binding.research_id
                and action_binding["contract_sha256"] == digest(binding.spec.as_dict())
                and action_binding["panel_sha256"] == panel_sha256,
                "A task input differs from the panel already bound to A actions")
    field_catalog = input_catalog()
    data = {
        "source_segment": "A",
        "research_id": binding.research_id,
        "contract_sha256": binding.contract_sha256,
        "panel_sha256": panel_sha256,
        "field_catalog": field_catalog,
        "operators": operator_catalog(),
        "templates": template_catalog(),
        "data_coverage": _a_coverage(panel),
    }
    content_sha256 = _content_sha256(data)
    record_id = f"a-environment-{content_sha256}"
    existing = next((record for record in actions.miner.store.all()
                     if record["id"] == record_id), None)
    if existing is None:
        actions.miner.store.append(record_id, "a_environment", data)
    else:
        require(existing["kind"] == "a_environment" and existing["data"] == data,
                "saved A environment record differs from its content reference")
    return {
        "record_id": record_id,
        "content_sha256": content_sha256,
        "panel_sha256": panel_sha256,
        "fields_pointer": "/field_catalog/fields",
        "operators_pointer": "/operators",
        "templates_pointer": "/templates",
        "coverage_pointer": "/data_coverage/field_coverage",
    }


class _NoModelCalls:
    """A model port that makes an accidental legacy model call fail immediately."""

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("native research tools must not call a role model")


class NativeChatAdapter:
    """Bind one server process to a fixed role, run, and the real domain actions."""

    def __init__(self, binding: NativeChatBinding, role: NativeRole, actions: Any,
                 *, a_panel_loader: Callable[[], FactorInputPanel] | None = None):
        require(role in _ROLES, "native MCP server role is invalid")
        self.binding = binding
        self.role = role
        self.actions = actions
        self._a_panel = actions.a_panel
        self._a_panel_loader = a_panel_loader
        self._tool_definitions = actions.tool_definitions(role)
        if role == "coordinator":
            self._tool_definitions.extend((_RECOVERY_TASK_TOOL, _PREPARE_B_TASK_TOOL))
        self._tools = {item["name"]: item for item in self._tool_definitions}
        require(len(self._tools) == len(self._tool_definitions),
                "role tool definitions contain duplicate names")
        for definition in self._tool_definitions:
            require(definition.get("type") == "function"
                    and isinstance(definition.get("inputSchema"), dict),
                    "domain tool definition is malformed")
            properties = definition["inputSchema"].get("properties", {})
            require("role" not in properties and "thread_id" not in properties
                    and "host_id" not in properties and "path" not in properties,
                    "domain tool schema exposes an untrusted identity or path parameter")

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        return list(self._tool_definitions)

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        require(name in self._tools, f"tool {name!r} is not available to role {self.role}")
        require(isinstance(arguments, dict), "tool arguments must be an object")
        from jsonschema import Draft202012Validator

        schema = self._tools[name]["inputSchema"]
        Draft202012Validator.check_schema(schema)
        errors = sorted(Draft202012Validator(schema).iter_errors(arguments),
                        key=lambda error: (list(error.path), error.message))
        require(not errors, f"invalid arguments for {name}: {errors[0].message if errors else ''}")
        if name == "prepare_a_task":
            return self.prepare_a_task()
        if name == "prepare_a_recovery_task":
            return self.prepare_a_recovery_task(arguments["task_id"])
        if name == "prepare_b_task":
            return self.prepare_b_task()
        return self.actions.handle(self.role, name, arguments)

    def prepare_a_task(self) -> dict[str, Any]:
        return self._prepare_a_task()

    def prepare_b_task(self) -> dict[str, Any]:
        require(self.role == "coordinator", "prepare_b_task is only available to coordinator")
        run_root = self.binding.run_root
        for name in ("b-access-started.json", "b-numerical-complete.json", "validation.json"):
            state_path = run_root / name
            require(not state_path.exists() and not state_path.is_symlink(),
                    "cannot prepare a B ticket after B access or delivery has started")

        completion_path = run_root / "a-complete.json"
        completion_info = completion_path.lstat() if completion_path.exists() else None
        require(completion_info is not None and stat.S_ISREG(completion_info.st_mode)
                and not completion_path.is_symlink(),
                "complete and seal A before preparing a B ticket")
        completion_bytes = completion_path.read_bytes()
        completion = json.loads(completion_bytes)

        frozen_path = run_root / "frozen_batch.json"
        frozen_info = frozen_path.lstat() if frozen_path.exists() else None
        require(frozen_info is not None and stat.S_ISREG(frozen_info.st_mode)
                and not frozen_path.is_symlink(),
                "freeze the completed A manifest before preparing a B ticket")
        frozen_bytes = frozen_path.read_bytes()
        frozen = json.loads(frozen_bytes)
        checked_frozen = self.actions.miner._checked_frozen()
        require(frozen == checked_frozen,
                "saved B freeze differs from the validated domain batch")

        candidate_ids = list(frozen["candidates"])
        manifest_ref = frozen.get("a_manifest_ref")
        require(isinstance(manifest_ref, str) and bool(manifest_ref)
                and completion.get("manifest_ref") == manifest_ref
                and completion.get("retained_ids") == candidate_ids
                and candidate_ids,
                "B ticket requires the exact nonempty retained A manifest")
        require(frozen.get("contract") == self.binding.spec.as_dict(),
                "frozen B batch differs from the bound research contract")

        body = {
            "format_version": 1,
            "research_id": self.binding.research_id,
            "batch_id": self.binding.research_id,
            "task_type": "frozen_b_validation",
            "target_version": self.binding.target["version"],
            "contract_version": "FM-v6",
            "contract_sha256": self.binding.contract_sha256,
            "target": dict(self.binding.target),
            "goal": dict(self.binding.goal),
            "contract": self.binding.spec.as_dict(),
            "a_completion": {
                "manifest_ref": manifest_ref,
                "frozen_candidate_ids": candidate_ids,
                "content_sha256": hashlib.sha256(completion_bytes).hexdigest(),
            },
            "frozen_batch_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
            "frozen_batch": frozen,
            "thin_spec": {
                "data_environment": {
                    "segment": "B",
                    "purpose": self.binding.spec.purpose,
                    "include_liquidations": self.binding.include_liquidations,
                    "interval_hours": self.binding.spec.sample_hours,
                    "timezone": "UTC",
                    "start_inclusive": self.binding.spec.b_start,
                    "end_exclusive": self.binding.spec.c_start,
                    "warmup_hours": self.binding.spec.max_lookback_hours,
                    "universe_provenance": self.binding.spec.universe_provenance,
                    "coverage": "not loaded; the bound B validator reports coverage after its single access",
                },
                "validation_policy": {
                    "candidate_set": "exact frozen_batch.candidates; no additions, removals, or substitutions",
                    "definitions": "frozen formula, direction, and retained horizons are immutable",
                    "numerical_access": "one B validation; persist the checkpoint before interpretation",
                    "incomplete_access": "if access started without a complete checkpoint, inspect evidence and stop",
                    "feedback_to_a": "never return B metrics to A or use them to modify this frozen batch",
                },
            },
            "expected_deliverables": {
                "numerical_checkpoint": "single frozen-batch B validation with complete saved checkpoint",
                "interpretation": "evidence-grounded report for every frozen candidate",
                "delivery": "complete B delivery from the saved numerical and interpretation records",
            },
        }
        task_id = _content_sha256(body)
        ticket = {**body, "task_id": task_id, "content_sha256": task_id,
                  "action_id": f"prepare-b-task-{task_id}"}
        ticket_root = run_root / "task-tickets"
        b_ticket_root = ticket_root / "B"
        for directory in (ticket_root, b_ticket_root):
            if directory.exists() or directory.is_symlink():
                info = directory.lstat()
                require(stat.S_ISDIR(info.st_mode) and not directory.is_symlink(),
                        "B task ticket directory must be a regular directory")
            else:
                directory.mkdir()
        ticket_path = b_ticket_root / f"{task_id}.json"
        if ticket_path.exists() or ticket_path.is_symlink():
            info = ticket_path.lstat()
            require(stat.S_ISREG(info.st_mode) and not ticket_path.is_symlink(),
                    "saved B task ticket must be a regular file")
            require(json.loads(ticket_path.read_text(encoding="utf-8")) == ticket,
                    "saved B task ticket differs from its content hash")
        else:
            write_json(ticket_path, ticket)
        ticket_ref = {"task_id": task_id, "content_sha256": task_id}
        return {"task_id": task_id, "ticket_ref": ticket_ref, "task_ticket": ticket}

    def prepare_a_recovery_task(self, task_id: str) -> dict[str, Any]:
        require(self.role == "coordinator",
                "prepare_a_recovery_task is only available to coordinator")
        recovery_ref = self._load_accepted_a_recovery_reference(task_id)
        return self._prepare_a_task(recovery_ref=recovery_ref)

    def _load_accepted_a_recovery_reference(self, task_id: str) -> dict[str, Any]:
        require(isinstance(task_id, str) and len(task_id) == 64
                and all(character in "0123456789abcdef" for character in task_id),
                "task_id must be a lowercase SHA-256 digest")
        ticket_path = self.binding.run_root / "task-tickets" / "A" / f"{task_id}.json"
        require(ticket_path.exists(), "saved A task ticket does not exist")
        ticket_info = ticket_path.lstat()
        require(stat.S_ISREG(ticket_info.st_mode) and not ticket_path.is_symlink(),
                "saved A task ticket must be a regular file")
        ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
        require(isinstance(ticket, dict)
                and ticket.get("task_id") == task_id
                and ticket.get("content_sha256") == task_id
                and ticket.get("action_id") == f"prepare-a-task-{task_id}",
                "saved A task ticket identity differs from its reference")
        body = {key: value for key, value in ticket.items()
                if key not in {"task_id", "content_sha256", "action_id"}}
        require(_content_sha256(body) == task_id,
                "saved A task ticket content hash is invalid")
        require(ticket.get("research_id") == self.binding.research_id
                and ticket.get("contract_sha256") == self.binding.contract_sha256
                and ticket.get("contract") == self.binding.spec.as_dict()
                and ticket.get("target") == self.binding.target
                and ticket.get("goal") == self.binding.goal
                and ticket.get("task_type") in {
                    "initial_a_research", "continue_a_research", "recovery_a_research"},
                "saved A task ticket belongs to a different run or contract")

        receipt_path = (self.binding.run_root / "dispatch-receipts" / "A"
                        / f"{task_id}.json")
        require(receipt_path.exists(), "accepted A dispatch receipt does not exist")
        receipt_info = receipt_path.lstat()
        require(stat.S_ISREG(receipt_info.st_mode) and not receipt_path.is_symlink(),
                "saved A dispatch receipt must be a regular file")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(isinstance(receipt, dict)
                and set(receipt) == {
                    "format_version", "task_id", "ticket_sha256", "target_role",
                    "session_id", "request_id", "status",
                }
                and receipt["format_version"] == 1
                and receipt["task_id"] == task_id
                and receipt["ticket_sha256"] == task_id
                and receipt["target_role"] == "A"
                and isinstance(receipt["session_id"], str)
                and bool(receipt["session_id"])
                and isinstance(receipt["request_id"], str)
                and bool(receipt["request_id"])
                and receipt["status"] == "accepted",
                "A recovery requires a matching accepted dispatch receipt")
        status = self.actions.get_research_status_for("A")
        pending_actions = status["pending_actions"]
        pending_refs = sorted(({
            "action_id": row["action_id"],
            "name": row["name"],
            "parameters_sha256": row["parameters_sha256"],
        } for row in pending_actions), key=lambda item: item["action_id"])
        return {
            "parent_task_id": task_id,
            "parent_content_sha256": task_id,
            "parent_task_type": ticket["task_type"],
            "parent_a_snapshot_sha256": _content_sha256(ticket["a_snapshot"]),
            "parent_dispatch_receipt_sha256": _content_sha256(receipt),
            "pending_actions": pending_refs,
        }

    def _load_human_change_reference(self) -> dict[str, Any] | None:
        path = self.binding.run_root / "change-record.json"
        if not path.exists() and not path.is_symlink():
            return None
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and not path.is_symlink(),
                "saved human change record must be a regular file")
        record = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "format_version", "record_id", "content_sha256", "parent", "child",
            "change", "approval_source", "approval_sha256", "recorded_at",
        }
        require(isinstance(record, dict) and set(record) == required
                and record["format_version"] == 1
                and record["approval_source"] == "local_operator_cli"
                and isinstance(record["approval_sha256"], str)
                and len(record["approval_sha256"]) == 64,
                "saved human change record fields are malformed")
        body = {key: value for key, value in record.items()
                if key not in {"record_id", "content_sha256"}}
        require(record["content_sha256"] == _content_sha256(body),
                "saved human change record content hash is invalid")
        parent, child = record["parent"], record["child"]
        change = record["change"]
        require(isinstance(change, dict) and set(change) == {
            "original_intent", "adopted_change", "new_objective",
            "contract_changes", "goal_changes",
        }
                and isinstance(change["original_intent"], str) and bool(change["original_intent"])
                and isinstance(change["adopted_change"], str) and bool(change["adopted_change"])
                and change["new_objective"] == self.binding.target["objective"]
                and isinstance(change["contract_changes"], dict)
                and isinstance(change["goal_changes"], dict),
                "saved human change description is malformed")
        require(isinstance(parent, dict) and isinstance(parent.get("target"), dict)
                and isinstance(child, dict)
                and child.get("research_id") == self.binding.research_id
                and child.get("target") == self.binding.target
                and child.get("contract") == self.binding.spec.as_dict()
                and child.get("goal") == self.binding.goal
                and child.get("contract_sha256") == self.binding.contract_sha256
                and parent.get("research_id") != self.binding.research_id
                and isinstance(parent.get("contract_sha256"), str)
                and isinstance(parent.get("a_snapshot_sha256"), str)
                and parent.get("a_snapshot_sha256") == _content_sha256(parent.get("a_snapshot"))
                and isinstance(parent.get("evidence_refs"), list)
                and isinstance(parent.get("task_refs"), list),
                "saved human change record does not match the child research binding")
        require(record["record_id"]
                == f"change-{self.binding.research_id}-v{self.binding.target['version']}"
                and self.binding.target["version"]
                == parent.get("target", {}).get("version", 0) + 1,
                "saved human change record version lineage is invalid")
        return {
            "record_id": record["record_id"],
            "content_sha256": record["content_sha256"],
            "parent_research_id": parent["research_id"],
            "parent_contract_sha256": parent["contract_sha256"],
            "parent_target_version": parent["target"]["version"],
            "parent_a_snapshot_sha256": parent["a_snapshot_sha256"],
            "affected_task_refs": parent["task_refs"],
            "affected_evidence_refs": parent["evidence_refs"],
        }

    def _prepare_a_task(self, *, recovery_ref: dict[str, Any] | None = None) -> dict[str, Any]:
        require(self.role == "coordinator", "prepare_a_task is only available to coordinator")
        snapshot = self.actions.prepare_a_task()
        require(isinstance(snapshot, dict), "A task preparation did not return a domain snapshot")
        require(snapshot.get("run_id") == self.binding.research_id,
                "A task snapshot belongs to a different research run")
        require(snapshot["a_status"] == "open",
                "cannot prepare an A task after the A manifest is sealed")
        require(not (self.binding.run_root / "b-access-started.json").exists(),
                "cannot prepare an A task after B access has started")
        if self._a_panel is None:
            require(self._a_panel_loader is not None,
                    "coordinator A task preparation requires the bound A input panel")
            self._a_panel = self._a_panel_loader()
        environment_ref = _save_a_environment_record(self.binding, self.actions, self._a_panel)
        snapshot = self.actions.prepare_a_task()
        task_type = ("recovery_a_research" if recovery_ref is not None
                     else "initial_a_research" if not snapshot["candidate_ids"]
                     else "continue_a_research")
        coverage = _a_coverage(self._a_panel)
        catalog = input_catalog()
        operators = operator_catalog()
        goal = dict(self.binding.goal)
        thin_spec = {
            "data_environment": {
                "segment": "A",
                "purpose": self.binding.spec.purpose,
                "include_liquidations": self.binding.include_liquidations,
                "interval_hours": self.binding.spec.sample_hours,
                "timezone": "UTC",
                "start_inclusive": self.binding.spec.a_start,
                "end_exclusive": self.binding.spec.b_start,
                "warmup_hours": self.binding.spec.max_lookback_hours,
                "loaded_start": coverage["loaded_start"],
                "loaded_end_inclusive": coverage["loaded_end_inclusive"],
                "universe_provenance": self.binding.spec.universe_provenance,
                "eligible_rows": coverage["eligible_rows"],
                "eligible_symbols": coverage["eligible_symbols"],
                "panel_sha256": environment_ref["panel_sha256"],
                "coverage_summary": {key: coverage[key] for key in
                                     ("fields", "covered_fields", "uncovered_fields")},
                "coverage_ref": {"record_id": environment_ref["record_id"],
                                 "pointer": environment_ref["coverage_pointer"],
                                 "content_sha256": environment_ref["content_sha256"]},
                "label": {
                    "name": self.binding.spec.label,
                    "horizon_hours": 24,
                    "availability": "future returns are used only by program evaluation and are not formula inputs",
                },
            "unit_rules": catalog["unit_rules"],
            "availability_assumption": catalog["availability_assumption"],
            },
            "expression_capabilities": {
                "fields": [{key: field[key] for key in ("name", "meaning", "unit", "source", "native_frequency")}
                           for field in catalog["fields"]
                           if field["name"] in coverage["covered_field_names"]
                           and (field["source"] != "liquidations"
                                or self.binding.include_liquidations)],
                "operator_names": [operator["name"] for operator in operators],
                "formula_budget": {
                    "max_formula_nodes": self.binding.spec.max_formula_nodes,
                    "max_lookback_hours": self.binding.spec.max_lookback_hours,
                    "max_repairs": self.binding.spec.max_repairs,
                },
                "dictionary_ref": {
                    "record_id": environment_ref["record_id"],
                    "content_sha256": environment_ref["content_sha256"],
                    "fields_pointer": environment_ref["fields_pointer"],
                    "operators_pointer": environment_ref["operators_pointer"],
                    "templates_pointer": environment_ref["templates_pointer"],
                },
            },
            "goal_counts": {
                "target_ideas": goal.get("target_ideas"),
                "quality_target": goal.get("quality_target"),
                "a_retained_candidates": snapshot["retained_count"],
                "a_retained_definition": "A-stage retain decisions; these are not B-qualified deliverables",
                "b_qualified_deliverables": None,
                "b_count_availability": "not available to A before frozen-batch validation",
            },
            "decision_policy": {
                "choose_next_action_from_evidence": True,
                "predeclare_modifications_before_candidate_submission": True,
                "candidate_direction_is_fixed_during_modification": True,
                "mean_improvement": ">= min_improvement",
                "improvement_lower_bound": "> 0",
                "ic_change_lower_bound": ">= -max_ic_loss",
                "stop": "estimated improvement upper bound < min_improvement or estimated IC-change upper bound < -max_ic_loss",
                "otherwise": "pause_insufficient",
                "b_feedback_to_a": "never reuse B results to select, filter or modify this A research batch",
            },
        }
        body = {
            "format_version": 1,
            "research_id": self.binding.research_id,
            "batch_id": self.binding.research_id,
            "task_type": task_type,
            "target_version": self.binding.target["version"],
            "contract_version": "FM-v6",
            "contract_sha256": self.binding.contract_sha256,
            "target": dict(self.binding.target),
            "goal": goal,
            "contract": self.binding.spec.as_dict(),
            "thin_spec": thin_spec,
            "a_snapshot": snapshot,
            "expected_deliverables": {
                "a_manifest": "complete candidate list, frozen definitions, retained horizons and A evidence references",
                "a_decisions": ["retain", "optimize", "pause", "discard"],
                "b_access": "not performed by A; coordinator freezes the complete A manifest before one B validation",
            },
        }
        change_ref = self._load_human_change_reference()
        if change_ref is not None:
            body["change_ref"] = change_ref
        if recovery_ref is not None:
            body["recovery_ref"] = recovery_ref
        task_id = _content_sha256(body)
        ticket = {**body, "task_id": task_id, "content_sha256": task_id,
                  "action_id": f"prepare-a-task-{task_id}"}
        ticket_path = self.binding.run_root / "task-tickets" / "A" / f"{task_id}.json"
        if ticket_path.is_file():
            require(json.loads(ticket_path.read_text(encoding="utf-8")) == ticket,
                    "saved A task ticket differs from its content hash")
        else:
            write_json(ticket_path, ticket)
        ticket_ref = {"task_id": task_id, "content_sha256": task_id}
        return {"task_id": task_id, "ticket_ref": ticket_ref, "task_ticket": ticket}


def load_native_adapter(binding_path: str | Path, role: NativeRole) -> NativeChatAdapter:
    """Construct the bound program interface; market paths come only from config."""
    binding = NativeChatBinding.load(binding_path)
    from .cli import load_membership, load_stage
    from .research_actions import ResearchActions
    from .workflow import FactorMiner

    miner = FactorMiner.open(binding.run_root, _NoModelCalls())
    require(miner.spec.as_dict() == binding.spec.as_dict(),
            "opened FactorMiner contract differs from its binding")
    a_panel = None
    validation_universe = None
    b_panel_loader = None
    a_panel_loader = None
    idea_pool = None
    if role == "A":
        a_panel = load_stage(binding.market_db, binding.universe_csv, miner.spec, "A",
                             include_liquidations=binding.include_liquidations)
    if role == "coordinator":
        a_panel_loader = lambda: load_stage(
            binding.market_db, binding.universe_csv, miner.spec, "A",
            include_liquidations=binding.include_liquidations,
        )
    if role == "coordinator":
        validation_universe = load_membership(binding.universe_csv, miner.spec, "B")
    if role == "B":
        b_panel_loader = lambda: load_stage(
            binding.market_db, binding.universe_csv, miner.spec, "B",
            include_liquidations=binding.include_liquidations,
        )
        idea_pool = binding.idea_pool
    actions = ResearchActions(
        miner, a_panel=a_panel, b_panel_loader=b_panel_loader,
        validation_universe=validation_universe, idea_pool=idea_pool,
        goal_spec=binding.goal,
    )
    return NativeChatAdapter(binding, role, actions, a_panel_loader=a_panel_loader)
