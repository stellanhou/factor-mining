"""A-only evidence tools exposed to the Codex ideator's native tool loop."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from .contracts import dumps, identifier, require
from .model import RecoverableToolError


_SEARCH_LIMIT = 20
_SUMMARY_CHARS = 1200


class AResearchEvidenceTools:
    """Search compact A-history indexes and page only authorized saved records.

    The current-run store and the Goal history reader are both read-only. Historical
    record IDs are kept in a separate whitelist because they live outside the active
    FactorMiner run's ``a_records`` directory.
    """

    def __init__(self, store: Any, allowed_record_ids: set[str], *,
                 history_index: Iterable[dict[str, Any]] = (),
                 allowed_history_record_ids: set[str] | None = None,
                 history_reader: Callable[[str, str, int, int], Any] | None = None,
                 max_single_output_bytes: int | None = None):
        require(isinstance(allowed_record_ids, set), "A evidence record whitelist is invalid")
        self.store = store
        self.allowed_record_ids = frozenset(allowed_record_ids)
        self.allowed_history_record_ids = frozenset(allowed_history_record_ids or set())
        require(not (self.allowed_record_ids & self.allowed_history_record_ids),
                "current and historical A record IDs must be disjoint")
        require(not self.allowed_history_record_ids or history_reader is not None,
                "historical A record access requires the Goal evidence reader")
        require(max_single_output_bytes is None
                or (type(max_single_output_bytes) is int and max_single_output_bytes >= 0),
                "A evidence single-response budget is invalid")
        self.history_reader = history_reader
        self.max_single_output_bytes = max_single_output_bytes
        self._records = {record["id"]: record for record in self.store.all()
                         if record["id"] in self.allowed_record_ids}
        require(set(self._records) == self.allowed_record_ids,
                "A evidence whitelist references a missing current-run record")
        for record in self._records.values():
            _require_source_a_only(record)
        self._validated_history_ids: set[str] = set()
        self.history_index = tuple(self._validate_history_item(item) for item in history_index)
        require(all(item["record_id"] in self.allowed_history_record_ids
                    for item in self.history_index),
                "history search index contains an unauthorized record ID")

    @staticmethod
    def _validate_history_item(item: dict[str, Any]) -> dict[str, Any]:
        require(isinstance(item, dict), "history search item is invalid")
        record_id = identifier(item.get("record_id"))
        pointer = item.get("pointer")
        require(isinstance(pointer, str) and (pointer == "" or pointer.startswith("/")),
                "history search pointer must be a JSON pointer")
        kind = item.get("kind")
        summary = item.get("summary")
        search_text = item.get("search_text", summary)
        require(isinstance(kind, str) and isinstance(summary, str)
                and isinstance(search_text, str), "history search metadata is invalid")
        segment = item.get("segment")
        require(segment in {None, "A"}, "history search index contains non-A evidence")
        result = {"record_id": record_id, "pointer": pointer, "kind": kind,
                  "summary": summary, "search_text": search_text, "segment": segment}
        candidate_ref = item.get("candidate_ref")
        if candidate_ref is not None:
            require(isinstance(candidate_ref, str), "candidate reference is invalid")
            result["candidate_ref"] = candidate_ref
        return result

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "search_a_evidence",
                "description": (
                    "Search the authorized A-segment task, cycle, candidate and factor-landscape "
                    "summaries. Returns short matches with record IDs and JSON pointers; it does "
                    "not read B/C evidence or arbitrary files."
                ),
                "inputSchema": {
                    "type": "object",
                    "required": ["query", "limit"],
                    "additionalProperties": False,
                    "properties": {
                        "query": {"type": "string", "minLength": 1,
                                   "description": "Exact phrase, field, candidate ID, or research topic."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": _SEARCH_LIMIT,
                                  "description": "Maximum number of short matches to return."},
                    },
                },
            },
            {
                "type": "function",
                "name": "read_a_evidence",
                "description": (
                    "Read one authorized immutable A-segment record by its record ID, JSON pointer "
                    "and page. offset/limit page only arrays; for large objects, request a narrower "
                    "JSON pointer. No filesystem paths are accepted."
                ),
                "inputSchema": {
                    "type": "object",
                    "required": ["record_id", "pointer", "offset", "limit"],
                    "additionalProperties": False,
                    "properties": {
                        "record_id": {"type": "string", "minLength": 1},
                        "pointer": {"type": "string",
                                    "description": "JSON pointer into the record's data; empty string selects all data."},
                        "offset": {"type": "integer", "minimum": 0},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                },
            },
        ]

    def handle(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        require(isinstance(arguments, dict), "research tool arguments must be an object")
        if name == "search_a_evidence":
            return self.search(**arguments)
        if name == "read_a_evidence":
            return self.read(**arguments)
        raise ValueError(f"unknown A research tool: {name}")

    def search(self, *, query: str, limit: int) -> dict[str, Any]:
        require(isinstance(query, str) and query.strip(), "search query must be non-empty")
        require(type(limit) is int and 1 <= limit <= _SEARCH_LIMIT, "invalid A evidence search limit")
        terms = [part.casefold() for part in re.findall(r"[^\s,，;；]+", query.strip())]
        matches: list[tuple[int, dict[str, Any]]] = []
        seen: set[tuple[str, str]] = set()
        for item in (*self._current_search_items(), *self.history_index):
            key = (item["record_id"], item["pointer"])
            if key in seen:
                continue
            seen.add(key)
            haystack = " ".join((item["record_id"], item["pointer"], item["kind"],
                                 item.get("candidate_ref", ""), item["search_text"])).casefold()
            score = sum(term in haystack for term in terms)
            if score:
                matches.append((score, item))
        matches.sort(key=lambda pair: (-pair[0], pair[1]["record_id"], pair[1]["pointer"]))
        total = len(matches)
        result = []
        for _, item in matches[:limit]:
            match = {"record_id": item["record_id"], "pointer": item["pointer"],
                     "kind": item["kind"], "summary": item["summary"][:_SUMMARY_CHARS],
                     "summary_truncated": len(item["summary"]) > _SUMMARY_CHARS}
            if "candidate_ref" in item:
                match["candidate_ref"] = item["candidate_ref"]
            result.append(match)
        response = {"matches": result, "total_matches": total, "truncated": total > limit}
        self._reserve_response(response, suggestion="reduce the search result limit")
        return response

    def read(self, *, record_id: str, pointer: str, offset: int, limit: int) -> dict[str, Any]:
        record_id = identifier(record_id)
        require(isinstance(pointer, str) and (pointer == "" or pointer.startswith("/")),
                "use a JSON pointer into A record data")
        require(type(offset) is int and offset >= 0 and type(limit) is int and limit > 0,
                "invalid A evidence page")
        if record_id in self.allowed_record_ids:
            _require_source_a_only(self._records[record_id])
            data = self.store.read(record_id, pointer, offset, limit)
        elif record_id in self.allowed_history_record_ids and self.history_reader is not None:
            self._require_history_record_a_only(record_id)
            data = self.history_reader(record_id, pointer, offset, limit)
        else:
            raise ValueError("record ID is outside the authorized A evidence set")
        _require_a_only(data)
        response = {"record_id": record_id, "pointer": pointer, "offset": offset,
                    "limit": limit, "data": data}
        suggestion = _narrowing_hint(pointer, offset, data)
        self._reserve_response(response, suggestion=suggestion)
        return response

    def _reserve_response(self, response: dict[str, Any], *, suggestion: str) -> None:
        if self.max_single_output_bytes is None:
            return
        size = len(dumps(response).encode("utf-8"))
        if size > self.max_single_output_bytes:
            raise RecoverableToolError(
                f"A evidence response is larger than the available context allowance; "
                f"no content was returned ({size} UTF-8 bytes, {self.max_single_output_bytes} available); "
                f"{suggestion}.")

    def _require_history_record_a_only(self, record_id: str) -> None:
        if record_id in self._validated_history_ids:
            return
        require(self.history_reader is not None, "historical A evidence reader is unavailable")
        source = self.history_reader(record_id, "", 0, 1)
        _require_a_only(source)
        if isinstance(source, dict) and isinstance(source.get("records"), dict):
            items = source["records"].get("items")
            if items is not None:
                require(isinstance(items, list), "historical cycle record index is malformed")
                for item in items:
                    require(isinstance(item, dict), "historical cycle record reference is malformed")
                    if item.get("kind") == "evaluation":
                        locator = item.get("factor_archive_ref")
                        if locator is not None:
                            key = locator.get("evaluation_key") if isinstance(locator, dict) else None
                            require(isinstance(key, dict) and key.get("segment") == "A",
                                    "historical cycle includes a non-A evaluation reference")
        self._validated_history_ids.add(record_id)

    def _current_search_items(self) -> list[dict[str, Any]]:
        records = list(self._records.values())
        items: list[dict[str, Any]] = []
        for record in records:
            data = record.get("data")
            _require_source_a_only(record)
            record_id, kind = record["id"], record["kind"]
            if kind == "goal_context" and isinstance(data, dict):
                prior = data.get("prior_A_research", {})
                require(isinstance(prior, dict), "goal A-history context is malformed")
                for collection in ("tasks", "cycles"):
                    rows = prior.get(collection, [])
                    if isinstance(rows, dict):
                        require(set(rows) == {"record_id", "pointer", "rows", "read_records"}
                                and rows["record_id"] == record_id
                                and rows["pointer"] == f"/prior_A_research/{collection}"
                                and type(rows["rows"]) is int and rows["rows"] >= 0,
                                f"goal A-history {collection} page reference is malformed")
                        continue
                    require(isinstance(rows, list), f"goal A-history {collection} index is malformed")
                    for index, row in enumerate(rows):
                        require(isinstance(row, dict), f"goal A-history {collection} row is malformed")
                        pointer = f"/prior_A_research/{collection}/{index}"
                        candidate_rows = row.get("candidates") if collection == "cycles" else None
                        if candidate_rows is not None:
                            require(isinstance(candidate_rows, list),
                                    "goal cycle candidate index is malformed")
                            for candidate_index, candidate in enumerate(candidate_rows):
                                require(isinstance(candidate, dict),
                                        "goal cycle candidate summary is malformed")
                                candidate_pointer = f"{pointer}/candidates/{candidate_index}"
                                candidate_with_cycle = {**candidate,
                                                        "source_record_id": row.get("source_record_id")}
                                items.append(_search_item(
                                    record_id, candidate_pointer, "historical_candidate_summary",
                                    candidate_with_cycle,
                                    candidate_ref=candidate.get("candidate_ref")))
                        else:
                            items.append(_search_item(record_id, pointer,
                                                      f"historical_{collection[:-1]}", row))
            elif kind == "factor_landscape" and isinstance(data, dict):
                snapshot = data.get("snapshot")
                require(isinstance(snapshot, dict), "A factor landscape snapshot is malformed")
                members = snapshot.get("members", [])
                if isinstance(members, dict):
                    require(set(members) == {"record_id", "pointer", "rows", "read_records"},
                            "A factor landscape member reference is malformed")
                else:
                    require(isinstance(members, list), "A factor landscape member list is malformed")
                    for index, member in enumerate(members):
                        require(isinstance(member, dict), "A factor landscape member is malformed")
                        items.append(_search_item(record_id, f"/snapshot/members/{index}",
                                                  "factor_landscape_member", member,
                                                  candidate_ref=member.get("candidate_ref")))
                items.append(_search_item(record_id, "/snapshot", kind, snapshot))
            else:
                items.append(_search_item(record_id, "", kind, data))
        return items


def _search_item(record_id: str, pointer: str, kind: str, value: Any, *,
                 candidate_ref: str | None = None) -> dict[str, Any]:
    serialized = dumps(value)
    summary = _summary_text(value)
    item = {"record_id": record_id, "pointer": pointer, "kind": kind,
            "summary": summary, "search_text": serialized, "segment": "A"}
    if candidate_ref is not None:
        item["candidate_ref"] = candidate_ref
    return item


def _summary_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        selected = {key: value[key] for key in (
            "source_record_id", "task", "reason", "candidate_ref", "definition",
            "calculation", "A_evaluation", "paired_comparison", "summary", "analysis",
            "field_inventory", "inventory_summary") if key in value}
        value = selected or value
    return dumps(value)


def _narrowing_hint(pointer: str, offset: int, data: Any) -> str:
    if isinstance(data, dict) and set(data) == {"total", "offset", "items"}:
        items = data["items"]
        if isinstance(items, list) and items and isinstance(items[0], dict):
            base = pointer.rstrip("/") + f"/{offset}"
            fields = [base + "/" + key.replace("~", "~0").replace("/", "~1")
                      for key in sorted(items[0])]
            return "narrow the JSON pointer to one of: " + ", ".join(fields[:20])
        return "reduce the requested array page limit"
    if isinstance(data, dict):
        fields = [pointer.rstrip("/") + "/" + key.replace("~", "~0").replace("/", "~1")
                  for key in sorted(data)]
        return "narrow the JSON pointer to one of: " + ", ".join(fields[:20])
    return "reduce the requested array page limit or select a smaller field"


def _require_a_only(value: Any) -> None:
    """Reject any returned record fragment that is explicitly marked B or C."""
    if isinstance(value, dict):
        segment = value.get("segment")
        require(segment not in {"B", "C"}, "A research tools cannot return B/C evidence")
        for nested in value.values():
            _require_a_only(nested)
    elif isinstance(value, list):
        for nested in value:
            _require_a_only(nested)


def _require_source_a_only(record: dict[str, Any]) -> None:
    require(isinstance(record, dict) and isinstance(record.get("data"), dict),
            "research record source is malformed")
    data = record["data"]
    if record.get("kind") == "evaluation":
        require(data.get("segment") == "A", "research tools can read only A evaluations")
    elif record.get("kind") == "factor_landscape":
        require(data.get("source_segment") == "A", "research tools can read only A factor landscapes")
    _require_a_only(data)
