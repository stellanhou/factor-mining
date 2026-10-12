from __future__ import annotations

import json

import pytest

from crypto_quant.research.factor_mining.model import ModelReply, RoleModels
from crypto_quant.research.factor_mining.codex_model import CodexModel
from crypto_quant.research.factor_mining.agent_ideator import CANDIDATE_SCHEMA
from crypto_quant.research.factor_mining.agent_ideator import IdeatorRole
from crypto_quant.research.factor_mining.records import (
    AgentGateway, ContextBudgetError, ENVELOPE_SCHEMA, RecordStore,
    compact_ideator_record, compact_record,
)
from crypto_quant.research.factor_mining.research_tools import AResearchEvidenceTools
from crypto_quant.research.factor_mining.model import RecoverableToolError


def _store(tmp_path):
    store = RecordStore(tmp_path / "a_records")
    store.append("goal-context", "goal_context", {
        "research_task": {"task": "研究相邻24小时波动状态", "reason": "检验排序稳定性"},
        "prior_A_research": {
            "tasks": [{"source_record_id": "task-00001010",
                       "task": "相邻、不重叠24小时波动变化",
                       "reason": "已有共同波动和下行风险研究"}],
            "cycles": [{"candidates": [{
                "candidate_ref": "goal-390db63370a1-000289/candidate-0001",
                "definition": {"expression": "sub(ts_return(spot_close,4),ts_return(perp_close,4))",
                               "direction": 1},
                "A_evaluation": {"summary": {"rank_ic_mean": 0.01297}},
            }]}],
        },
    })
    store.append("factor-landscape", "factor_landscape", {
        "source_segment": "A",
        "snapshot": {"members": [{"candidate_ref": "source-run/candidate-1",
                                   "expression": "ts_return(perp_close,4)",
                                   "direction": 1}]},
    })
    store.append("a-result", "experiment_result", {
        "segment": "A", "paired_periods": [{"period": 1}, {"period": 2}],
    })
    return store


def test_search_indexes_cycle_and_candidate_ids_and_returns_readable_pointers(tmp_path):
    store = _store(tmp_path)
    history_index = [{
        "record_id": "cycle-00000289",
        "pointer": "/context/candidates/0",
        "kind": "historical_candidate_summary",
        "candidate_ref": "goal-390db63370a1-000289/candidate-0001",
        "summary": "4h现货—永续收益差父因子；A Rank IC 0.01297",
        "search_text": "cycle-00000289 goal-390db63370a1-000289/candidate-0001 spot perp return spread",
        "segment": "A",
    }]
    tools = AResearchEvidenceTools(
        store, {"goal-context", "factor-landscape", "a-result"},
        history_index=history_index,
        allowed_history_record_ids={"cycle-00000289"},
        history_reader=lambda *_: {"segment": "A"},
    )

    result = tools.handle("search_a_evidence", {
        "query": "cycle-00000289 goal-390db63370a1-000289/candidate-0001", "limit": 5,
    })

    assert result["total_matches"] >= 1
    match = next(item for item in result["matches"] if item["record_id"] == "cycle-00000289")
    assert match["pointer"] == "/context/candidates/0"
    assert match["candidate_ref"] == "goal-390db63370a1-000289/candidate-0001"


def test_read_returns_exact_page_and_provenance_without_mutating_record(tmp_path):
    store = _store(tmp_path)
    before = store._load(store.root / "a-result.json")["data"]
    tools = AResearchEvidenceTools(store, {"goal-context", "factor-landscape", "a-result"})

    page = tools.handle("read_a_evidence", {
        "record_id": "a-result", "pointer": "/paired_periods", "offset": 1, "limit": 1,
    })

    assert page == {"record_id": "a-result", "pointer": "/paired_periods", "offset": 1,
            "limit": 1, "data": {"total": 2, "offset": 1, "items": [{"period": 2}]}}
    assert store._load(store.root / "a-result.json")["data"] == before


def test_ideator_catalog_symbol_coverage_has_exact_read_refs_and_keeps_original(tmp_path):
    store = RecordStore(tmp_path / "a_records")
    fields = [{"name": "spot_close", "unit": "USDT / base_asset"}]
    symbols = {
        "BTCUSDT": {"coverage": {"spot_close": {"eligible_rows": 10,
                                                   "eligible_valid_rows": 9}}},
        "ETHUSDT": {"coverage": {"spot_close": {"eligible_rows": 10,
                                                   "eligible_valid_rows": 8}}},
    }
    store.append("inputs", "data_provenance", {
        "panel": {"start": "2026-01-01T00:00:00+00:00"},
        "catalog": {"fields": fields, "data": {
            "start": "2026-01-01T00:00:00+00:00", "symbols": symbols,
        }},
    })
    original = store._load(store.root / "inputs.json")["data"]

    model_view = compact_ideator_record(store._load(store.root / "inputs.json"))
    symbol_view = model_view["data"]["catalog"]["data"]["symbols"]
    entries = symbol_view["entries"]

    assert [entry["name"] for entry in entries] == list(symbols)
    assert [entry["read_ref"] for entry in entries] == [
        {"record_id": "inputs", "pointer": "/catalog/data/symbols/BTCUSDT"},
        {"record_id": "inputs", "pointer": "/catalog/data/symbols/ETHUSDT"},
    ]
    assert model_view["data"]["catalog"]["fields"] == fields
    tools = AResearchEvidenceTools(store, {"inputs"})
    for entry in entries:
        ref = entry["read_ref"]
        result = tools.handle("read_a_evidence", {
            "record_id": ref["record_id"], "pointer": ref["pointer"],
            "offset": 0, "limit": 1,
        })
        assert result["data"] == symbols[entry["name"]]
    assert store._load(store.root / "inputs.json")["data"] == original


def test_full_current_record_is_checked_before_pointer_can_hide_B_segment(tmp_path):
    store = RecordStore(tmp_path / "a_records")
    store.append("b-evaluation", "evaluation", {
        "segment": "B", "summary": {"rank_ic": 0.1},
    })

    with pytest.raises(ValueError, match="only A evaluations"):
        AResearchEvidenceTools(store, {"b-evaluation"})


def test_historical_source_is_checked_before_reading_a_segmentless_field(tmp_path):
    store = RecordStore(tmp_path / "a_records")
    calls = []

    def history_reader(record_id, pointer, offset, limit):
        calls.append((record_id, pointer, offset, limit))
        if pointer == "":
            return {"records": {"items": [{"kind": "evaluation",
                "factor_archive_ref": {"evaluation_key": {"segment": "B"}}}]}}
        return {"summary": {"rank_ic": 0.1}}

    tools = AResearchEvidenceTools(
        store, set(), history_index=[{"record_id": "cycle-00000001", "pointer": "/records/0",
            "kind": "historical_cycle", "summary": "cycle", "segment": "A"}],
        allowed_history_record_ids={"cycle-00000001"}, history_reader=history_reader,
    )

    with pytest.raises(ValueError, match="B/C evidence"):
        tools.read(record_id="cycle-00000001", pointer="/records/0/summary", offset=0, limit=1)
    assert calls == [("cycle-00000001", "", 0, 1)]


def test_oversized_page_fails_with_narrowing_hint_and_returns_no_partial_data(tmp_path):
    store = RecordStore(tmp_path / "a_records")
    store.append("a-evaluation", "evaluation", {
        "segment": "A", "summary": {"rank_ic": 0.01},
        "coverage": {"periods": 300}, "paired_periods": [{"payload": "x" * 400}],
    })
    tools = AResearchEvidenceTools(store, {"a-evaluation"}, max_single_output_bytes=64)

    with pytest.raises(RecoverableToolError,
                       match="no content was returned.*narrow the JSON pointer") as exc:
        tools.read(record_id="a-evaluation", pointer="/paired_periods", offset=0, limit=1)
    assert "/paired_periods/0/payload" in str(exc.value)
    assert store._load(store.root / "a-evaluation.json")["data"]["paired_periods"][0]["payload"] == "x" * 400


def test_context_budget_is_per_response_not_a_cumulative_read_quota(tmp_path):
    store = _store(tmp_path)
    tools = AResearchEvidenceTools(
        store, {"goal-context", "factor-landscape", "a-result"}, max_single_output_bytes=300,
    )
    request = {"record_id": "a-result", "pointer": "/paired_periods", "offset": 0, "limit": 1}

    first = tools.read(**request)
    second = tools.read(**request)

    assert first == second


def test_native_tools_are_explicit_and_unknown_tools_fail_closed(tmp_path):
    store = _store(tmp_path)
    tools = AResearchEvidenceTools(store, {"goal-context", "factor-landscape", "a-result"})
    definitions = tools.tool_definitions()

    assert [tool["name"] for tool in definitions] == ["search_a_evidence", "read_a_evidence"]
    assert all(tool["type"] == "function" and "inputSchema" in tool for tool in definitions)
    with pytest.raises(ValueError, match="unknown A research tool"):
        tools.handle("read_any_file", {"path": "/tmp/secret"})


def test_ideator_context_pages_full_history_without_mutating_saved_goal_context():
    record = {"id": "goal-context", "kind": "goal_context", "data": {
        "prior_A_research": {"tasks": [{"task": f"task {index}"} for index in range(334)],
                             "cycles": [{"candidates": [{"candidate_ref": "run/candidate-1"}],
                                         "cycle": index} for index in range(333)]},
    }}

    compacted = compact_record(record, page_cycle_candidates=True, page_goal_tasks=True)

    assert compacted["data"]["prior_A_research"]["tasks"] == {
        "record_id": "goal-context", "pointer": "/prior_A_research/tasks", "rows": 334,
        "read_records": "search A history or request an explicit offset and limit for original task summaries",
    }
    assert compacted["data"]["prior_A_research"]["cycles"]["rows"] == 333
    assert isinstance(record["data"]["prior_A_research"]["tasks"], list)
    assert isinstance(record["data"]["prior_A_research"]["cycles"], list)


def test_ideator_preloads_only_task_referenced_parent_formula_and_target_A_evidence():
    parent = {"candidate_ref": "goal-run-000289/candidate-0001",
              "definition": {"expression": "sub(ts_return(spot_close,4),ts_return(perp_close,4))",
                             "direction": 1},
              "A_evaluation": {"horizon_comparison": {
                  "4": {"horizon_hours": 4, "label": "perp_next_open_4h",
                        "direction": 1, "summary": {"rank_ic_mean": 0.01297},
                        "coverage": {"n": 17_500}},
                  "24": {"horizon_hours": 24, "summary": {"rank_ic_mean": 0.006}}},
                  "rank_displacement": {"deltas": {
                      "4": {"summary": {"mean": 0.40542}},
                      "24": {"summary": {"mean": 0.33}},
                  }}},
              "paired_comparison": None}
    record = {"id": "goal-context", "kind": "goal_context", "data": {
        "goal": {"quality_target": {"horizon_hours": 4, "displacement_hours": 4}},
        "research_task": {"task": "研究波动变化",
                          "evidence_refs": ["cycle-00000289", "task-old"]},
        "prior_A_research": {
            "tasks": [{"source_record_id": "task-old", "task": "相关父因子任务",
                       "reason": "基准", "evidence_refs": ["cycle-00000289"]},
                      {"source_record_id": "task-other", "task": "无关任务", "evidence_refs": []}],
            "cycles": [{"source_record_id": "cycle-00000289", "candidates": [parent]},
                       {"source_record_id": "cycle-00000290", "candidates": [
                           {"candidate_ref": "other/candidate", "definition": {"expression": "unrelated"}}]}],
        },
    }}

    compacted = compact_ideator_record(record)["data"]
    evidence = compacted["research_task_evidence"]

    assert compacted["prior_A_research"]["tasks"]["rows"] == 2
    assert compacted["prior_A_research"]["cycles"]["rows"] == 2
    assert [item["source_record_id"] for item in evidence["matched_tasks"]] == ["task-old"]
    assert [item["source_record_id"] for item in evidence["matched_cycles"]] == ["cycle-00000289"]
    candidate = evidence["matched_cycles"][0]["candidates"][0]
    assert candidate["definition"] == parent["definition"]
    assert candidate["A_horizon_evidence"]["horizon_hours"] == 4
    assert candidate["A_rank_displacement"]["summary"]["mean"] == 0.40542
    assert candidate["read_ref"] == {
        "record_id": "goal-context",
        "pointer": "/prior_A_research/cycles/0/candidates/0",
    }
    assert evidence["unmatched_evidence_refs"] == []
    assert len(compacted["research_task_evidence"]["matched_cycles"]) == 1
    assert len(record["data"]["prior_A_research"]["cycles"]) == 2


class _GatewaySpec:
    context_tokens = 100_000
    output_tokens = None
    run_id = "ideator-tool-test"

    def as_dict(self):
        return {"run_id": self.run_id, "label": "perp_next_open_4h"}


class _NativeCodex(CodexModel):
    def __init__(self):
        super().__init__("gpt-5.6-luna")
        self.tool_calls = []
        self.complete_calls = 0

    def complete(self, *args, **kwargs):
        self.complete_calls += 1
        raise AssertionError("native Codex ideator must not use the old one-shot transport")

    def complete_with_tools(self, messages, *, tools, tool_handler, max_output_tokens,
                            session_id, output_schema=None):
        self.messages = messages
        self.tools = tools
        self.output_schema = output_schema
        found = tool_handler("search_a_evidence", {"query": "goal-context", "limit": 2})
        page = tool_handler("read_a_evidence", {
            "record_id": "a-result", "pointer": "/paired_periods", "offset": 0, "limit": 1,
        })
        self.tool_calls.extend([found, page])
        return ModelReply(
            '{"result":{"candidates":[],"dispositions":[],"analysis":"已读取A证据"},'
            '"read_records":[]}', {}, self.model,
        )


def _ideator_schema():
    return {
        "type": "object", "additionalProperties": False,
        "required": ["candidates", "dispositions", "analysis"],
        "properties": {
            "candidates": {"type": "array", "items": CANDIDATE_SCHEMA},
            "dispositions": {"type": "array", "maxItems": 0,
                             "items": {"type": "object", "properties": {}, "required": [],
                                       "additionalProperties": False}},
            "analysis": {"type": "string"},
        },
    }


def test_codex_ideator_uses_native_search_and_page_tools_in_one_call(tmp_path):
    store = _store(tmp_path)
    model = _NativeCodex()
    role_models = RoleModels({"ideator": model, "calculator": object(),
                              "evaluator": object(), "optimizer": object()})
    gateway = AgentGateway(role_models, _GatewaySpec(), store, tmp_path / "model_calls")
    allowed = {"goal-context", "factor-landscape", "a-result"}
    result = gateway.ask(
        "ideator", "读取历史父因子", {
            "ideation_id": "round-001-ideation",
            "factor_landscape_ref": {"record_id": "factor-landscape", "pointer": "/snapshot"},
        }, _ideator_schema(), allowed_record_ids=allowed,
    )

    assert result == {"candidates": [], "dispositions": [], "analysis": "已读取A证据"}
    assert model.complete_calls == 0
    assert len(model.tool_calls) == 2
    assert [tool["name"] for tool in model.tools] == ["search_a_evidence", "read_a_evidence"]
    assert model.output_schema["properties"]["read_records"]["maxItems"] == 0
    assert model.output_schema["properties"]["read_records"]["items"]["additionalProperties"] is False
    assert "additionalProperties" not in ENVELOPE_SCHEMA["properties"]["read_records"]["items"]
    assert "search_a_evidence" in model.messages[0]["content"]
    assert "需要原始证据时输出" not in model.messages[0]["content"]


def test_ideator_over_budget_recompression_keeps_symbol_refs_and_task_preload(tmp_path, monkeypatch):
    store = RecordStore(tmp_path / "a_records")
    store.append("factor-landscape", "factor_landscape", {
        "source_segment": "A", "snapshot": {"members": []},
    })
    symbols = {}
    for symbol_index in range(30):
        coverage = {}
        for field_index in range(28):
            coverage[f"field_{field_index:02d}"] = {
                "eligible_rows": 4000,
                "eligible_valid_rows": 3000 + ((symbol_index + field_index) % 1000),
                "missing_reasons": {f"reason_{(symbol_index + field_index) % 17}": 1000},
            }
        symbols[f"ASSET{symbol_index:02d}USDT"] = {"coverage": coverage}
    fields = [{"name": f"field_{index:04d}", "description": "field contract " * 8}
              for index in range(500)]
    store.append("inputs", "data_provenance", {
        "catalog": {"fields": fields, "data": {
            "start": "2026-01-01T00:00:00+00:00", "symbols": symbols,
        }},
    })
    parent = {
        "candidate_ref": "prior-goal/candidate-0001",
        "definition": {"expression": "ts_return(spot_close,4)", "direction": 1},
        "calculation": {"status": "computed", "executed_expression": "ts_return(spot_close,4)"},
        "A_evaluation": {
            "horizon_comparison": {"4": {"horizon_hours": 4,
                "label": "perp_next_open_4h", "direction": 1,
                "summary": {"rank_ic_mean": 0.012}, "coverage": {"valid_rows": 100}}},
            "rank_displacement": {"deltas": {"4": {"summary": {"mean": 0.4}}}},
        },
    }
    store.append("goal-context", "goal_context", {
        "goal": {"quality_target": {"horizon_hours": 4, "displacement_hours": 4}},
        "research_task": {"task": "围绕已引用的历史父因子研究", "evidence_refs": ["cycle-00000289"]},
        "prior_A_research": {
            "tasks": [{"source_record_id": f"task-{index:08d}", "task": "旧任务 " + str(index)}
                      for index in range(334)],
            "cycles": [{"source_record_id": "cycle-00000289", "candidates": [parent]}]
                      + [{"source_record_id": f"cycle-{index:08d}", "candidates": []}
                         for index in range(1, 334) if index != 289],
        },
    })
    model = _NativeCodex()
    role_models = RoleModels({"ideator": model, "calculator": object(),
                              "evaluator": object(), "optimizer": object()})
    spec = _GatewaySpec()
    spec.context_tokens = 80_000
    gateway = AgentGateway(role_models, spec, store, tmp_path / "model_calls")
    allowed = {record["id"] for record in store.all()}
    original_bound = AgentGateway._input_bound
    seen = []

    def capture_bound(self, messages):
        bound = original_bound(self, messages)
        if messages and messages[-1]["role"] == "user":
            content = json.loads(messages[-1]["content"])
            if content.get("role") == "ideator":
                seen.append((bound, content))
        return bound

    monkeypatch.setattr(AgentGateway, "_input_bound", capture_bound)
    with pytest.raises(ContextBudgetError, match="no model call was started"):
        gateway.ask(
            "ideator", "核对当前研究依据", {
                "catalog_record_id": "inputs", "candidate_decisions": {},
                "pending_proposals": {}, "ideation_id": "round-001-ideation",
                "factor_landscape_ref": {"record_id": "factor-landscape", "pointer": "/snapshot"},
            }, _ideator_schema(), allowed_record_ids=allowed,
        )

    assert len(seen) == 3
    assert all(bound == seen[0][0] for bound, _ in seen)
    assert seen[0][0] > spec.context_tokens
    for _, content in seen:
        input_view = next(record for record in content["records"] if record["id"] == "inputs")
        symbol_view = input_view["data"]["catalog"]["data"]["symbols"]
        assert len(symbol_view["entries"]) == len(symbols)
        assert "coverage" not in symbol_view
        goal_view = next(record for record in content["records"] if record["id"] == "goal-context")
        candidate = goal_view["data"]["research_task_evidence"]["matched_cycles"][0]["candidates"][0]
        assert candidate["definition"] == parent["definition"]
        assert candidate["read_ref"] == {
            "record_id": "goal-context",
            "pointer": "/prior_A_research/cycles/0/candidates/0",
        }
    assert model.complete_calls == 0


def test_non_codex_ideator_keeps_existing_read_records_transport(tmp_path):
    store = _store(tmp_path)

    class PlainModel:
        def __init__(self):
            self.calls = []

        def complete(self, messages, *, max_output_tokens, session_id):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return ModelReply('{"result":null,"read_records":[{"record_id":"a-result",'
                    '"pointer":"/paired_periods","offset":0,"limit":1}]}', {}, "fixture")
            return ModelReply('{"result":{"candidates":[],"dispositions":[],"analysis":"已补读"},'
                              '"read_records":[]}', {}, "fixture")

    model = PlainModel()
    gateway = AgentGateway(model, _GatewaySpec(), store, tmp_path / "model_calls")
    result = gateway.ask(
        "ideator", "读取A证据", {
            "ideation_id": "round-001-ideation",
            "factor_landscape_ref": {"record_id": "factor-landscape", "pointer": "/snapshot"},
        }, _ideator_schema(), allowed_record_ids={"goal-context", "factor-landscape", "a-result"},
    )

    assert result["analysis"] == "已补读"
    assert len(model.calls) == 2


def test_evaluator_information_request_is_passed_as_A_reference_without_decision_authority():
    class CaptureGateway:
        def ask(self, role, task, payload, schema, **kwargs):
            self.role, self.payload, self.schema, self.kwargs = role, payload, schema, kwargs
            return {"candidates": [], "dispositions": [], "analysis": "记录待核验事实缺口"}

    ideator = IdeatorRole()
    ideator.gateway = CaptureGateway()
    ideator.decisions = {}
    ideator.routes = {}
    ideator.proposals = {}
    ideator.candidates = {}
    review_ref = {"record_id": "round-001-review", "pointer": "/information_request"}

    result = ideator._ask_ideator(
        {}, ideation_id="round-002-ideation",
        factor_landscape_ref={"record_id": "factor-landscape", "pointer": "/snapshot"},
        allowed_record_ids={"round-001-review", "factor-landscape"},
        evaluator_review_ref=review_ref,
    )

    assert result["analysis"] == "记录待核验事实缺口"
    assert ideator.gateway.payload["evaluator_review_ref"] == review_ref
    assert "action" not in ideator.gateway.schema["properties"]
    assert ideator.gateway.kwargs["allowed_record_ids"] == {"round-001-review", "factor-landscape"}

    with pytest.raises(ValueError, match="authorized A review record"):
        ideator._ask_ideator(
            {}, ideation_id="round-002-ideation",
            factor_landscape_ref={"record_id": "factor-landscape", "pointer": "/snapshot"},
            allowed_record_ids={"factor-landscape"}, evaluator_review_ref=review_ref,
        )
