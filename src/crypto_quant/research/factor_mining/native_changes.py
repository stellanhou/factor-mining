"""Human-approved version branches for native factor research runs."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import ResearchSpec, digest, identifier, require, text
from .records import write_json


_APPROVAL_FIELDS = {
    "format_version", "child_research_id", "original_intent", "adopted_change",
    "new_objective", "contract_changes", "goal_changes",
}


def _read_json_regular(path: Path, label: str) -> Any:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink(),
            f"{label} must be a regular file")
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_approval(approval_path: Path, parent: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    approval = _read_json_regular(approval_path, "human change approval")
    require(isinstance(approval, dict) and set(approval) == _APPROVAL_FIELDS
            and approval["format_version"] == 1,
            "human change approval fields do not match format 1")
    child_id = text(approval["child_research_id"], "child_research_id")
    identifier(child_id)
    require(child_id != parent.research_id, "child research_id must differ from its parent")
    original_intent = text(approval["original_intent"], "original_intent")
    adopted_change = text(approval["adopted_change"], "adopted_change")
    new_objective = text(approval["new_objective"], "new_objective")
    contract_changes = approval["contract_changes"]
    goal_changes = approval["goal_changes"]
    require(isinstance(contract_changes, dict), "contract_changes must be an object")
    require(isinstance(goal_changes, dict), "goal_changes must be an object")
    allowed_contract_fields = set(parent.spec.as_dict()) - {"run_id", "objective"}
    require(set(contract_changes) <= allowed_contract_fields,
            "contract_changes contains a field outside the fixed FM-v6 contract")
    allowed_goal_fields = {"target_ideas", "quality_target", "max_cycles"}
    require(set(goal_changes) <= allowed_goal_fields,
            "goal_changes contains an unsupported GoalSpec field")

    new_contract_data = {
        **parent.spec.as_dict(), **contract_changes,
        "run_id": child_id, "objective": new_objective,
    }
    new_spec = ResearchSpec.from_dict(new_contract_data)
    old_goal = dict(parent.goal)
    goal_data = {**old_goal, "goal_id": child_id, "objective": new_objective}
    if "target_ideas" in goal_changes:
        goal_data.pop("quality_target", None)
        goal_data.update(goal_changes)
    elif "quality_target" in goal_changes:
        goal_data.pop("target_ideas", None)
        goal_data.update(goal_changes)
    else:
        goal_data.update(goal_changes)
    from .goal import GoalSpec

    new_goal = GoalSpec(**goal_data).as_dict()
    changed_contract = {
        key: value for key, value in new_spec.as_dict().items() if key != "run_id"
    } != {
        key: value for key, value in parent.spec.as_dict().items() if key != "run_id"
    }
    require(changed_contract or new_goal != old_goal,
            "human change must change the target, research contract, or Goal")
    new_target = {"version": parent.target["version"] + 1, "objective": new_objective}
    return {"spec": new_spec, "goal": new_goal, "target": new_target}, {
        "original_intent": original_intent,
        "adopted_change": adopted_change,
        "new_objective": new_objective,
        "contract_changes": contract_changes,
        "goal_changes": goal_changes,
    }


def _parent_refs(binding: Any, miner: Any, actions: Any) -> dict[str, Any]:
    from .native_chats import _content_sha256

    status = actions.get_research_status_for("A")
    require(status["a_status"] == "open",
            "a human change requires the parent A run to remain open")
    require(not (binding.run_root / "b-access-started.json").exists()
            and not (binding.run_root / "frozen_batch.json").exists(),
            "a human change is not allowed after B access or freeze")
    b_records = binding.run_root / "b_records"
    require(not b_records.is_symlink(), "parent B record directory must not be a symlink")
    if b_records.is_dir():
        require(not list(b_records.glob("*.json")),
                "a human change is not allowed after B evidence exists")
    pending = actions.ledger.pending_actions()
    require(not pending,
            "resolve the parent run's pending domain action before recording a change")
    snapshot = actions.prepare_a_task()
    require(snapshot["run_id"] == binding.research_id and snapshot["a_status"] == "open",
            "parent A snapshot changed while preparing the human change")

    evidence_refs = [
        {"record_id": row["id"], "kind": row["kind"], "sha256": digest(row)}
        for row in miner.store.all()
    ]
    task_refs = []
    ticket_root = binding.run_root / "task-tickets" / "A"
    if ticket_root.exists():
        for ticket_path in sorted(ticket_root.glob("*.json")):
            ticket = _read_json_regular(ticket_path, "saved A task ticket")
            require(isinstance(ticket, dict), "saved A task ticket must be an object")
            task_id = ticket.get("task_id")
            body = {key: value for key, value in ticket.items()
                    if key not in {"task_id", "content_sha256", "action_id"}}
            require(isinstance(task_id, str) and ticket_path.name == f"{task_id}.json"
                    and ticket.get("format_version") == 1
                    and ticket.get("content_sha256") == task_id
                    and ticket.get("action_id") == f"prepare-a-task-{task_id}"
                    and _content_sha256(body) == task_id
                    and ticket.get("research_id") == binding.research_id
                    and ticket.get("contract_sha256") == binding.contract_sha256
                    and ticket.get("target") == binding.target
                    and ticket.get("goal") == binding.goal
                    and ticket.get("contract") == binding.spec.as_dict()
                    and ticket.get("task_type") in {
                        "initial_a_research", "continue_a_research", "recovery_a_research",
                    }
                    and isinstance(ticket.get("a_snapshot"), dict)
                    and ticket["a_snapshot"].get("a_status") == "open"
                    and ticket["a_snapshot"].get("research_contract") == binding.spec.as_dict(),
                    "saved A task ticket is not bound to the parent research contract")
            task_refs.append({"task_id": task_id, "task_type": ticket.get("task_type")})

    action_refs = []
    receipt_root = binding.run_root / "action_receipts"
    if receipt_root.exists():
        for receipt_path in sorted(receipt_root.glob("*.committed.json")):
            receipt = _read_json_regular(receipt_path, "committed action receipt")
            require(isinstance(receipt, dict)
                    and receipt_path.name == f"{receipt.get('action_id', '')}.committed.json"
                    and isinstance(receipt.get("name"), str)
                    and isinstance(receipt.get("parameters_sha256"), str)
                    and len(receipt["parameters_sha256"]) == 64
                    and "result" in receipt,
                    "committed action receipt identity is malformed")
            action_refs.append({
                "action_id": receipt["action_id"],
                "name": receipt["name"],
                "parameters_sha256": receipt["parameters_sha256"],
                "result_sha256": digest(receipt["result"]),
            })
    return {
        "research_id": binding.research_id,
        "target": dict(binding.target),
        "contract_sha256": binding.contract_sha256,
        "a_snapshot": snapshot,
        "a_snapshot_sha256": digest(snapshot),
        "evidence_refs": evidence_refs,
        "task_refs": task_refs,
        "action_refs": action_refs,
    }


def _change_ref(record: dict[str, Any]) -> dict[str, Any]:
    parent = record["parent"]
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


def _validate_saved_record(record: Any, child_id: str, approval_sha256: str) -> dict[str, Any]:
    require(isinstance(record, dict)
            and set(record) == {
                "format_version", "record_id", "content_sha256", "parent", "child",
                "change", "approval_source", "approval_sha256", "recorded_at",
            }
            and record["format_version"] == 1
            and record["approval_sha256"] == approval_sha256
            and record["approval_source"] == "local_operator_cli",
            "saved human change record does not match the approved child run")
    child = record["child"]
    require(isinstance(child, dict) and isinstance(child.get("target"), dict)
            and type(child["target"].get("version")) is int
            and record["record_id"] == f"change-{child_id}-v{child['target']['version']}",
            "saved human change record child target is malformed")
    body = {key: value for key, value in record.items()
            if key not in {"record_id", "content_sha256"}}
    require(digest(body) == record["content_sha256"]
            and child["research_id"] == child_id,
            "saved human change record content hash is invalid")
    return _change_ref(record)


def record_human_change(binding_path: str | Path,
                        approval_path: str | Path) -> dict[str, Any]:
    """Create a linked child run from an explicit local human approval file.

    The approval file is consumed only by this operator CLI path; it is never a
    model-callable MCP tool. Existing parent artifacts are referenced by digest
    and left in place.
    """
    from .native_chats import NativeChatAdapter, NativeChatBinding, _NoModelCalls
    from .research_actions import ResearchActions
    from .workflow import FactorMiner

    binding = NativeChatBinding.load(binding_path)
    approval = Path(approval_path).expanduser().absolute()
    proposed, descriptions = _validated_approval(approval, binding)
    approval_sha256 = hashlib.sha256(approval.read_bytes()).hexdigest()
    child_id = proposed["spec"].run_id
    child_root = binding.run_root.parent / child_id
    parent_miner = FactorMiner.open(binding.run_root, _NoModelCalls())
    parent_actions = ResearchActions(parent_miner, goal_spec=binding.goal)

    with parent_actions.ledger.exclusive():
        reused_existing_child = child_root.exists() or child_root.is_symlink()
        if reused_existing_child:
            saved_record_path = child_root / "change-record.json"
            run_info = child_root.lstat()
            require(stat.S_ISDIR(run_info.st_mode) and not child_root.is_symlink(),
                    "child research run must be a regular directory")
            require(saved_record_path.is_file() and not saved_record_path.is_symlink(),
                    "child run exists without a complete human change record; inspect it manually")
            change_ref = _validate_saved_record(
                _read_json_regular(saved_record_path, "saved human change record"),
                child_id, approval_sha256,
            )
            expected_contract = proposed["spec"].as_dict()
            contract_path = child_root / "contract.json"
            require(_read_json_regular(contract_path, "saved child contract") == expected_contract,
                    "saved child contract differs from the approved change")
        else:
            parent_ref = _parent_refs(binding, parent_miner, parent_actions)
            new_target = proposed["target"]
            contract_sha256 = hashlib.sha256(
                json.dumps(proposed["spec"].as_dict(), ensure_ascii=False,
                           allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                + b"\n"
            ).hexdigest()
            record_body = {
                "format_version": 1,
                "parent": parent_ref,
                "child": {
                    "research_id": child_id,
                    "target": new_target,
                    "contract": proposed["spec"].as_dict(),
                    "contract_sha256": contract_sha256,
                    "goal": proposed["goal"],
                },
                "change": descriptions,
                "approval_source": "local_operator_cli",
                "approval_sha256": approval_sha256,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            record_id = f"change-{child_id}-v{new_target['version']}"
            record = {**record_body, "record_id": record_id,
                      "content_sha256": digest(record_body)}
            miner = FactorMiner(proposed["spec"], _NoModelCalls(), child_root.parent)
            miner.store.append("goal-spec", "goal_contract", proposed["goal"])
            write_json(child_root / "change-record.json", record)
            change_ref = _change_ref(record)

        child_binding = replace(
            binding, research_id=child_id, target=proposed["target"], goal=proposed["goal"],
            run_root=child_root, contract_sha256=hashlib.sha256(
                (child_root / "contract.json").read_bytes()).hexdigest(),
            spec=proposed["spec"],
        )
        from .cli import load_stage

        a_panel = load_stage(
            child_binding.market_db, child_binding.universe_csv, child_binding.spec, "A",
            include_liquidations=child_binding.include_liquidations,
        )
        child_miner = FactorMiner.open(child_root, _NoModelCalls())
        actions = ResearchActions(child_miner, a_panel=a_panel,
                                  goal_spec=child_binding.goal)
        adapter = NativeChatAdapter(child_binding, "coordinator", actions,
                                    a_panel_loader=lambda: a_panel)
        prepared = adapter.prepare_a_task()
        require(prepared["task_ticket"].get("change_ref") == change_ref,
                "child A ticket does not bind the human change record")
        binding_source = Path(binding_path).expanduser().absolute()
        saved_binding = _read_json_regular(binding_source, "parent binding")
        child_binding_data = {
            **saved_binding, "research_id": child_id, "target": proposed["target"],
            "goal": proposed["goal"],
            "run_root": os.path.relpath(child_root, binding_source.parent),
            "contract_sha256": child_binding.contract_sha256,
        }
        child_binding_path = binding_source.parent / f"binding-{child_id}.json"
        if child_binding_path.exists():
            require(_read_json_regular(child_binding_path, "child binding") == child_binding_data,
                    "saved child binding differs from the approved change")
        else:
            write_json(child_binding_path, child_binding_data)
        return {
            "parent_research_id": binding.research_id,
            "child_research_id": child_id,
            "change_record_id": change_ref["record_id"],
            "change_record_sha256": change_ref["content_sha256"],
            "task_id": prepared["task_id"],
            "task_ticket_path": str(child_root / "task-tickets" / "A" / f"{prepared['task_id']}.json"),
            "child_run_root": str(child_root),
            "child_binding_path": str(child_binding_path),
            "reused_existing_child": reused_existing_child,
        }
