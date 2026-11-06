"""Durable, run-scoped idempotency and stop control for program actions."""
from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from .contracts import digest, identifier, require
from .records import write_json


class ResearchActionLedger:
    """Serialize domain writes and bind each action ID to one immutable request."""

    def __init__(self, run_root: Path):
        self.run_root = Path(run_root).resolve()
        self.root = self.run_root / "action_receipts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.control_root = self.run_root / "action_control"
        self.control_root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lock(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @contextmanager
    def exclusive(self):
        """Hold the cross-process research write lock."""
        with self._lock(self.run_root / "research-actions.lock"):
            yield

    def _action_paths(self, action_id: str) -> tuple[Path, Path]:
        action_id = identifier(action_id)
        return (self.root / f"{action_id}.started.json",
                self.root / f"{action_id}.committed.json")

    def _control_events(self) -> list[dict[str, Any]]:
        events = []
        for path in sorted(self.control_root.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-*.json")):
            events.append(json.loads(path.read_text(encoding="utf-8")))
        return events

    def status(self) -> dict[str, Any]:
        events = self._control_events()
        if not events:
            return {"status": "active", "source": "program_action_control"}
        latest = events[-1]
        status = {"stop_requested": "stopping", "stop": "stopped", "resume": "active"}[latest["event"]]
        return {"status": status,
                "source": "program_action_control", "event_id": latest["event_id"],
                "reason": latest.get("reason")}

    def pending_actions(self) -> list[dict[str, Any]]:
        rows = []
        for started in sorted(self.root.glob("*.started.json")):
            action_id = started.name.removesuffix(".started.json")
            if (self.root / f"{action_id}.committed.json").exists():
                continue
            rows.append(json.loads(started.read_text(encoding="utf-8")))
        return rows

    def require_active(self) -> None:
        status = self.status()
        require(status["status"] == "active",
                "program actions are stopped; explicitly resume before submitting new actions")

    def commit(self, action_id: str, operation: Callable[[], Any]) -> Any:
        """Make one evidence write atomic with respect to stop requests."""
        self.require_active()
        with self._lock(self.control_root / "control.lock"):
            self.require_active()
            return operation()

    def run(self, action_id: str, name: str, parameters: dict[str, Any],
            operation: Callable[[], Any], *,
            recover: Callable[[], Any] | None = None,
            preflight: Callable[[], None] | None = None) -> Any:
        action_id = identifier(action_id)
        request = {"action_id": action_id, "name": identifier(name),
                   "parameters_sha256": digest(parameters)}
        started_path, committed_path = self._action_paths(action_id)
        with self.exclusive():
            pending = False
            with self._lock(self.control_root / "control.lock"):
                if committed_path.exists():
                    saved = json.loads(committed_path.read_text(encoding="utf-8"))
                    require({key: saved[key] for key in request} == request,
                            "action_id was already committed with different parameters")
                    return saved["result"]
                self.require_active()
                if started_path.exists():
                    saved_start = json.loads(started_path.read_text(encoding="utf-8"))
                    require(saved_start == request,
                            "action_id is pending with different parameters")
                    pending = True
                else:
                    pending = False
                if preflight is not None:
                    preflight()
                if not pending:
                    write_json(started_path, request)
            if pending:
                require(recover is not None,
                        "action is pending and has no domain-specific recovery evidence")
                result = recover()
            else:
                result = operation()
            with self._lock(self.control_root / "control.lock"):
                if committed_path.exists():
                    saved = json.loads(committed_path.read_text(encoding="utf-8"))
                    require({key: saved[key] for key in request} == request,
                            "action_id was already committed with different parameters")
                    return saved["result"]
                write_json(committed_path, {**request, "result": result})
            return result

    def control(self, action_id: str, event: str, *, reason: str | None = None) -> dict[str, Any]:
        action_id = identifier(action_id)
        require(event in {"stop", "resume"}, "unknown action-control event")
        if event == "stop":
            require(isinstance(reason, str) and bool(reason.strip()), "stop reason must be nonempty")
            self.begin_stop(action_id, reason=reason)
            return self.finish_stop(action_id, reason=reason)
        else:
            require(reason is None, "resume does not take a reason")
        request = {"action_id": action_id, "name": f"{event}_actions",
                   "parameters_sha256": digest({"reason": reason})}
        started_path, committed_path = self._action_paths(action_id)
        with self._lock(self.control_root / "control.lock"):
            if committed_path.exists():
                saved = json.loads(committed_path.read_text(encoding="utf-8"))
                require({key: saved[key] for key in request} == request,
                        "action_id was already committed with different parameters")
                return saved["result"]
            if started_path.exists():
                saved_start = json.loads(started_path.read_text(encoding="utf-8"))
                require(saved_start == request, "action_id is pending with different parameters")
            else:
                write_json(started_path, request)
        with self.exclusive():
            with self._lock(self.control_root / "control.lock"):
                if committed_path.exists():
                    saved = json.loads(committed_path.read_text(encoding="utf-8"))
                    require({key: saved[key] for key in request} == request,
                            "action_id was already committed with different parameters")
                    return saved["result"]
                require(self.status()["status"] == "stopped", "program actions are not stopped")
                event_id = self._append_control_event(action_id, "resume", None)
                result = {"status": "active", "source": "program_action_control",
                          "event_id": event_id}
                write_json(committed_path, {**request, "result": result})
                return result

    def begin_stop(self, action_id: str, *, reason: str) -> dict[str, Any]:
        """Persist stop intent immediately so new actions fail before quiescence."""
        action_id = identifier(action_id)
        require(isinstance(reason, str) and bool(reason.strip()), "stop reason must be nonempty")
        request = {"action_id": action_id, "name": "stop_actions",
                   "parameters_sha256": digest({"reason": reason})}
        started_path, committed_path = self._action_paths(action_id)
        with self._lock(self.control_root / "control.lock"):
            if committed_path.exists():
                saved = json.loads(committed_path.read_text(encoding="utf-8"))
                require({key: saved[key] for key in request} == request,
                        "action_id was already committed with different parameters")
                return saved["result"]
            if started_path.exists():
                saved_start = json.loads(started_path.read_text(encoding="utf-8"))
                require(saved_start == request,
                        "action_id is pending with different parameters")
            else:
                write_json(started_path, request)
            prior_event = next((item for item in self._control_events()
                                if item.get("action_id") == action_id), None)
            if prior_event is None:
                require(self.status()["status"] == "active",
                        "program actions are already stopping or stopped")
                event_id = self._append_control_event(action_id, "stop_requested", reason)
            else:
                require(prior_event["event"] in {"stop_requested", "stop"}
                        and prior_event.get("reason") == reason,
                        "action_id was already used for a different control event")
                event_id = prior_event["event_id"]
            status = self.status()["status"]
            return {"status": "stopped" if status == "stopped" else "stopping",
                    "source": "program_action_control", "event_id": event_id,
                    "reason": reason}

    def finish_stop(self, action_id: str, *, reason: str) -> dict[str, Any]:
        """Acknowledge stop only after every active domain write releases its lock."""
        action_id = identifier(action_id)
        require(isinstance(reason, str) and bool(reason.strip()), "stop reason must be nonempty")
        request = {"action_id": action_id, "name": "stop_actions",
                   "parameters_sha256": digest({"reason": reason})}
        started_path, committed_path = self._action_paths(action_id)
        if not started_path.exists():
            raise ValueError("stop completion has no matching persisted stop request")
        saved_start = json.loads(started_path.read_text(encoding="utf-8"))
        require(saved_start == request, "pending stop request differs from its completion")
        with self.exclusive():
            with self._lock(self.control_root / "control.lock"):
                if committed_path.exists():
                    saved = json.loads(committed_path.read_text(encoding="utf-8"))
                    require({key: saved[key] for key in request} == request,
                            "action_id was already committed with different parameters")
                    return saved["result"]
                prior = next((item for item in self._control_events()
                              if item.get("action_id") == action_id), None)
                require(prior is not None and prior["event"] in {"stop_requested", "stop"}
                        and prior.get("reason") == reason,
                        "stop completion has no matching persisted stop request")
                if prior["event"] == "stop_requested":
                    event_id = self._append_control_event(action_id, "stop", reason)
                else:
                    event_id = prior["event_id"]
                result = {"status": "stopped", "source": "program_action_control",
                          "event_id": event_id, "reason": reason}
                write_json(committed_path, {**request, "result": result})
                return result

    def _append_control_event(self, action_id: str, event: str,
                              reason: str | None) -> str:
        sequence = len(self._control_events()) + 1
        event_id = f"{sequence:08d}-{action_id}"
        control_event = {"event_id": event_id, "event": event, "action_id": action_id,
                         **({"reason": reason} if reason is not None else {})}
        write_json(self.control_root / f"{event_id}.json", control_event)
        return event_id
