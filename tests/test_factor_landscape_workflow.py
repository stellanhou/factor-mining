"""A-only factor landscape integration and recovery checks."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto_quant.features.factor_expressions import compile_expression
from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS
from crypto_quant.research.factor_mining.agent_optimizer import RESEARCH_QUESTIONS
from crypto_quant.research.factor_mining.contracts import ResearchSpec, dumps
from crypto_quant.research.factor_mining.landscape import build_factor_landscape
from crypto_quant.research.factor_mining.model import ModelReply
from crypto_quant.research.factor_mining.records import RecordStore, compact_record, write_json
from crypto_quant.research.factor_mining.workflow import FactorMiner


def specification() -> ResearchSpec:
    return ResearchSpec(
        run_id="landscape-fixture", objective="核对A段因子树上下文", purpose="engineering_check",
        a_start="2026-07-01T00:00:00Z", b_start="2026-07-05T00:00:00Z",
        c_start="2026-07-09T00:00:00Z", c_end="2026-07-13T00:00:00Z",
        universe_provenance="synthetic six-symbol fixture",
        data_usage_review="synthetic values for integration checks only",
        label="perp_next_open_24h", sample_hours=1, groups=3, min_symbols=6,
        min_periods=26, hac_lags=23, confidence=0.95, stage_hours=48,
        rolling_periods=26, fdr_method="BH", fdr_alpha=0.05,
        min_abs_ic=0.02, min_directional_spread=0.0001, min_stage_share=0.5,
        max_repairs=2, max_formula_nodes=80, max_lookback_hours=7,
        context_tokens=2_000_000, output_tokens=4000,
        b_horizons=(1, 4, 24), admission_scheme="plan3", plan3_tracks_gate=False,
    )


def panel(spec: ResearchSpec) -> FactorInputPanel:
    start, end = spec.bounds("A")
    hours = pd.date_range(start - pd.Timedelta(hours=spec.max_lookback_hours), end,
                          freq="h", inclusive="left")
    symbols = sorted(["BTCUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    index = pd.MultiIndex.from_product([hours, symbols], names=["timestamp", "symbol"])
    row_count = len(hours)
    t = np.arange(row_count, dtype=float)[:, None]
    s = np.arange(1, len(symbols) + 1, dtype=float)[None, :]
    price = s * np.exp(0.0002 * t + 0.035 * np.sin(t * (0.12 + s * 0.013)))
    values = pd.DataFrame(1.0, index=index, columns=INPUT_COLUMNS)
    values["perp_open"] = price.reshape(-1)
    values["perp_close"] = (price * (1 + 0.002 * np.cos(t * (0.08 + s * 0.01)))).reshape(-1)
    values["spot_close"] = values["perp_close"] * 0.998
    values["premium_index"] = np.broadcast_to(np.sin(t * (0.1 + s * 0.02)), price.shape).reshape(-1)
    universe = pd.Series(True, index=index, dtype=bool)
    return FactorInputPanel(values, universe, {"fixture": True, "segment": "A"})


def definition(expression: str) -> dict:
    return {
        "name": "fixture_factor", "expression": expression,
        "meaning": "按每个币自身价格窗口构造相对偏离，无量纲",
        "hypothesis": "价格偏离在短期内存在可检验的反转关系",
        "direction": 1, "parent_id": None, "proposal_id": None, "change_reason": "基准公式",
    }


def prior_rank_displacement() -> dict:
    return {"status": "available", "definition_version": "fixture-D-v1",
            "deltas": {key: {"summary": {"mean": 0.2}, "coverage": {"valid": 80}}
                       for key in ("1", "4", "24")}}


class CaptureModel:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.read_prior_page = False
        self.did_read_prior_page = False
        self.received_evidence = None

    def complete(self, messages, *, max_output_tokens, session_id):
        request = json.loads(messages[1]["content"])
        self.requests.append(copy.deepcopy(request))
        if len(messages) > 2:
            self.received_evidence = json.loads(messages[-1]["content"])
        if self.read_prior_page and not self.did_read_prior_page:
            self.did_read_prior_page = True
            return ModelReply(dumps({"result": None, "read_records": [{
                "record_id": "goal-context",
                "pointer": "/prior_A_research/cycles/0/candidates/0",
                "offset": 0, "limit": 1,
            }]}), {}, "fixture")
        result = {"candidates": [], "dispositions": [{
            "proposal_id": "proposal-D", "action": "abandon",
            "reason": "当前模拟响应放弃该配对任务", "candidate_index": None,
        }], "analysis": "模拟模型用于核对请求携带的树和原任务边界。"}
        return ModelReply(dumps({"result": result, "read_records": []}), {}, "fixture")


class ExploreGraphModel:
    """Script the actual two-round graph, including one failed A formula."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.first_proposal = None

    @staticmethod
    def _answers(supported: bool) -> dict:
        return {key: {"supported": supported, "reason": "脚本场景核对流水线字段，不代表研究结论。"}
                for key in RESEARCH_QUESTIONS}

    @staticmethod
    def _decision(candidate_id: str, disposition: str, continuing: bool,
                  evidence_ref: str) -> dict:
        return {
            "candidate_id": candidate_id, "disposition": disposition,
            "retained_horizons": [24] if disposition == "retain" else [],
            "continue_optimization": continuing, "answers": ExploreGraphModel._answers(continuing),
            "evidence_refs": [evidence_ref], "reason": "脚本模型供图流程集成验证。",
            "resume_condition": "补充公式诊断后再评估" if disposition == "pause" else None,
        }

    @staticmethod
    def _candidate(expression: str) -> dict:
        return {"name": "landscape_fixture", "expression": expression,
                "meaning": "当前价格相对自身窗口均值的偏离，无量纲",
                "hypothesis": "短期价格偏离可能包含可检验的反转信息",
                "direction": 1, "parent_id": None, "proposal_id": None,
                "change_reason": "脚本模型只用于流程测试。"}

    def complete(self, messages, *, max_output_tokens, session_id):
        request = json.loads(messages[1]["content"])
        self.requests.append(copy.deepcopy(request))
        role, payload = request["role"], request["payload"]
        if role == "ideator":
            if payload["pending_proposals"]:
                proposal_id = next(iter(payload["pending_proposals"]))
                result = {"candidates": [], "dispositions": [{
                    "proposal_id": proposal_id, "action": "abandon",
                    "reason": "模拟放弃，验证rank displacement任务原文保持不变。",
                    "candidate_index": None,
                }], "analysis": "第二轮模拟构想。"}
            else:
                result = {"candidates": [
                    self._candidate("div(perp_close,ts_mean(perp_close,3))"),
                    self._candidate("ts_mean(perp_close)"),
                ], "dispositions": [], "analysis": "第一轮模拟构想，第二个公式用于验证失败候选仍入库存。"}
        elif role == "calculator":
            result = {"candidate_id": payload["candidate_id"],
                      "repair_expression": payload["current_expression"],
                      "reason": "脚本模型保留无效公式以验证失败证据。"}
        elif role == "evaluator":
            result = {"analysis": "脚本报告用于验证A历史保留。", "mechanism": "反转假设待检验。",
                      "conditions": ["合成数据"], "falsifiers": ["不具研究效力"],
                      "limitations": ["脚本模型不作有效性结论"], "next_steps": ["不开展真实研究"]}
        elif role == "optimizer":
            if payload["round_candidate_ids"]:
                result = {
                    "analysis": "脚本模型提供完整结构，仅检查任务流转。",
                    "decisions": [
                        self._decision("candidate-0001", "retain", True, "candidate-0001-evaluation"),
                        self._decision("candidate-0002", "pause", False, "candidate-0002-calculation"),
                    ],
                    "diagnostics": ["仅为集成夹具"],
                    "proposals": [{
                        "proposal_id": "proposal-D", "route_id": "rank-displacement-route",
                        "control_id": "candidate-0001", "evidence_refs": ["candidate-0001-evaluation"],
                        "modification_task": {
                            "core_hypothesis": "相对价格偏离可能反转",
                            "observed_problem": "脚本配对任务用于核对已有A证据链接。",
                            "modification_hypothesis": "排名变化下降可能改善信号持续性代理指标。",
                            "change_target": "降低4小时预测期限下1小时排名位移，保留其他构件",
                            "fixed_components": "原始窗口偏离、币池和方向固定",
                        },
                        "experiment_design": {
                            "question": "指定平滑能否降低排名位移并保持预测证据？",
                            "metric": "rank_displacement", "min_improvement": 0.01,
                            "max_ic_loss": 0.02, "expected_outcome": "D下降且有向IC损失不超过容限",
                            "stop_condition": "不能达到预设D改善则停止",
                            "pause_condition": "覆盖不足或区间不确定时暂停",
                            "horizon_hours": 4, "displacement_hours": 1,
                        },
                        "restart_of": None, "new_evidence": None,
                    }],
                }
                self.first_proposal = copy.deepcopy(result["proposals"][0])
            else:
                result = {"analysis": "脚本模型结束第二轮。",
                          "decisions": [self._decision("candidate-0001", "retain", False,
                                                        "candidate-0001-evaluation")],
                          "diagnostics": ["仅为集成夹具"], "proposals": []}
        else:
            raise AssertionError(f"unexpected role: {role}")
        return ModelReply(dumps({"result": result, "read_records": []}), {}, "fixture")


class FactorLandscapeWorkflowTests(unittest.TestCase):
    def add_current_candidate(self, miner: FactorMiner, *, candidate_id="candidate-0001",
                              round_no=1, expression="div(perp_close,ts_mean(perp_close,3))"):
        definition_data = definition(expression)
        executed = compile_expression(expression).description()
        summary = {"rank_ic": {"mean": 0.04}, "directional_spread": {"mean": 0.0002}}
        item = {
            "id": candidate_id, "round": round_no, "definition": definition_data,
            "experiment": None,
            "calculation": {"status": "computed", "executed_expression": executed,
                            "checks": [{"attempt": 0, "program_error": None}]},
            "evaluation": {"segment": "A", "summary": summary, "coverage": {"rows": 96}},
        }
        miner.candidates[candidate_id] = item
        miner.decisions[candidate_id] = {
            "disposition": "retain", "continue_optimization": True,
            "reason": "A段保留，允许当前配对优化", "resume_condition": None,
        }
        miner.store.append(f"{candidate_id}-definition", "candidate", item)
        miner.store.append(f"{candidate_id}-calculation", "calculation", {
            "candidate_id": candidate_id, **item["calculation"]})
        miner.store.append(f"{candidate_id}-evaluation", "evaluation", {
            "candidate_id": candidate_id, "segment": "A", "summary": summary,
            "coverage": {"rows": 96}})

    def add_goal_context(self, miner: FactorMiner):
        expression = "div(perp_close,ts_mean(perp_close,5))"
        executed = compile_expression(expression).description()
        candidate = {
            "candidate_ref": "prior-run/candidate-0001",
            "definition": definition(expression),
            "calculation": {"status": "computed", "executed_expression": executed},
            "A_evaluation": {"summary": {"rank_ic": {"mean": 0.031}},
                             "coverage": {"rows": 120},
                             "rank_displacement": prior_rank_displacement()},
            "final_decision": {"disposition": "retain", "continue_optimization": False,
                               "reason": "prior A-only decision", "resume_condition": None},
            "paired_comparison": None,
        }
        context = {
            "goal": {"goal_id": "goal-fixture", "objective": "探索未覆盖的信号"},
            "research_task": {"task": "检查订单流之外的量价偏离方向"},
            "prior_A_research": {"tasks": [], "cycles": [{
                "source_record_id": "cycle-00000001", "run_id": "prior-run",
                "candidates": [candidate],
            }]},
            "previous_expressions": [],
        }
        miner.store.append("goal-context", "goal_context", context)

    def test_ideator_request_carries_current_tree_prior_A_and_no_B_or_future(self):
        spec = specification()
        input_panel = panel(spec)
        model = CaptureModel()
        model.read_prior_page = True
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory) / "runs")
            miner.store.append("inputs", "data_provenance", {"segment": "A"})
            self.add_current_candidate(miner)
            self.add_goal_context(miner)
            # This record is deliberately outside the A evidence store.
            write_json(miner.root / "b_records" / "b-secret.json", {"value": "B_SENTINEL_987654"})
            saved = {record["id"]: record["data"] for record in miner.store.all()}

            landscape_ref, context_ids = miner._landscape_snapshot(
                input_panel, 2, "round-002-ideation", saved)
            saved_snapshot = miner.store._load(
                miner.store.root / "round-002-factor-landscape.json")["data"]
            future = {"id": "candidate-0002", "round": 3,
                      "definition": definition("perp_close")}
            miner.candidates["candidate-0002"] = future
            miner.store.append("candidate-0002-definition", "candidate", future)
            miner.store.append("round-003-ideation", "ideation", {"future_only": "FUTURE_SENTINEL"})

            replay_ref, replay_ids = miner._landscape_snapshot(
                input_panel, 2, "round-002-ideation", saved)
            self.assertEqual(replay_ref, landscape_ref)
            self.assertEqual(replay_ids, context_ids)
            replayed_snapshot = miner.store._load(
                miner.store.root / "round-002-factor-landscape.json")["data"]
            self.assertEqual(replayed_snapshot, saved_snapshot)

            pending = {"proposal-D": {
                "proposal_id": "proposal-D", "route_id": "D-route",
                "control_id": "candidate-0001", "evidence_refs": ["candidate-0001-evaluation"],
                "modification_task": {"change_target": "降低指定期限的排名位移"},
                "experiment_design": {"metric": "rank_displacement", "horizon_hours": 4,
                                      "displacement_hours": 1},
            }}
            miner.proposals.update(copy.deepcopy(pending))
            miner.proposals["proposal-D"]["status"] = "pending"
            visible_ids = miner._ideator_visible_record_ids(landscape_ref["record_id"], context_ids)
            response = miner._ask_ideator(
                pending, ideation_id="round-002-ideation", factor_landscape_ref=landscape_ref,
                allowed_record_ids=visible_ids)
            self.assertEqual(response["dispositions"][0]["action"], "abandon")

            request = model.requests[0]
            self.assertEqual(request["payload"]["factor_landscape_ref"], landscape_ref)
            self.assertEqual(request["payload"]["pending_proposals"], pending)
            self.assertIn("拥挤分支不是禁入条件", request["task"])
            self.assertIn("rank_displacement", request["task"])
            self.assertIn("horizon_hours和displacement_hours", request["task"])
            self.assertNotIn("B_SENTINEL_987654", dumps(request))
            self.assertNotIn("FUTURE_SENTINEL", dumps(request))
            self.assertNotIn("candidate-0002", dumps(request))
            landscape_record = next(record for record in request["records"]
                                    if record["id"] == landscape_ref["record_id"])
            snapshot = landscape_record["data"]["snapshot"]
            self.assertEqual(snapshot["status"], "computed")
            self.assertEqual({member["candidate_ref"] for member in snapshot["members"]}, {
                "landscape-fixture/candidate-0001", "prior-run/candidate-0001"})
            prior_member = next(member for member in snapshot["members"]
                                if member["candidate_ref"] == "prior-run/candidate-0001")
            self.assertEqual(prior_member["source_A_evaluation"]["summary"]["rank_ic"]["mean"], 0.031)
            self.assertEqual(prior_member["source_A_evaluation"]["rank_displacement"], prior_rank_displacement())
            self.assertIn("goal-context#/prior_A_research/cycles/0/candidates/0",
                          prior_member["source_evidence_refs"])
            self.assertTrue(any(record["id"] == "candidate-0001-evaluation" for record in request["records"]))
            self.assertEqual(model.received_evidence["requested_original_evidence"][0]["record_id"],
                             "goal-context")
            self.assertEqual(model.received_evidence["requested_original_evidence"][0]["data"][
                "candidate_ref"], "prior-run/candidate-0001")

    def test_empty_history_is_a_normal_explicit_empty_landscape(self):
        spec = specification()
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, CaptureModel(), Path(directory) / "runs")
            miner.store.append("inputs", "data_provenance", {"segment": "A"})
            saved = {record["id"]: record["data"] for record in miner.store.all()}
            ref, _ = miner._landscape_snapshot(panel(spec), 1, "round-001-ideation", saved)
            record = miner.store._load(miner.store.root / f"{ref['record_id']}.json")
            self.assertEqual(record["data"]["snapshot"]["status"], "no_members")
            self.assertEqual(record["data"]["snapshot"]["members"], [])
            self.assertIsNone(record["data"]["snapshot"]["tree"])

    def test_landscape_arrays_use_record_paging_when_context_is_compacted(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RecordStore(Path(directory) / "records")
            snapshot = {"status": "insufficient_pair_coverage", "contract": {"data_segment": "A"},
                        "members": [{"candidate_ref": f"run/candidate-{i:04d}"} for i in range(4)],
                        "pairwise_correlations": [{"left_ref": "a", "right_ref": "b"}],
                        "missing_pairs": [{"left_ref": "a", "right_ref": "b"}],
                        "tree": {"method": "average", "leaf_order": ["a", "b"],
                                 "scipy_linkage": [[0, 1, 0.2, 2]], "merges": []}}
            store.append("round-001-factor-landscape", "factor_landscape",
                         {"round_no": 1, "snapshot": snapshot})
            record = store.all()[0]
            compacted = compact_record(record, page_cycle_candidates=True)
            members_ref = compacted["data"]["snapshot"]["members"]
            pairs_ref = compacted["data"]["snapshot"]["pairwise_correlations"]
            linkage_ref = compacted["data"]["snapshot"]["tree"]["scipy_linkage"]
            self.assertEqual((members_ref["record_id"], members_ref["pointer"], members_ref["rows"]),
                             ("round-001-factor-landscape", "/snapshot/members", 4))
            self.assertEqual(pairs_ref["pointer"], "/snapshot/pairwise_correlations")
            self.assertEqual(linkage_ref["pointer"], "/snapshot/tree/scipy_linkage")
            self.assertEqual(store.read(members_ref["record_id"], members_ref["pointer"], 2, 1)["items"],
                             snapshot["members"][2:3])

    def test_real_explore_graph_saves_each_round_and_reuses_snapshots_on_resume(self):
        spec = specification()
        input_panel = panel(spec)
        model = ExploreGraphModel()
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory) / "runs")
            result = miner.explore(input_panel)
            self.assertEqual(result["completed_rounds"], 2)

            ideations = [request for request in model.requests if request["role"] == "ideator"]
            self.assertEqual(len(ideations), 2)
            for request in model.requests:
                landscape_ids = [record["id"] for record in request["records"]
                                 if record["kind"] == "factor_landscape"]
                if request["role"] == "ideator":
                    self.assertEqual(landscape_ids, [request["payload"]["factor_landscape_ref"]["record_id"]])
                else:
                    self.assertEqual(landscape_ids, [])
            first_ref = ideations[0]["payload"]["factor_landscape_ref"]
            second_ref = ideations[1]["payload"]["factor_landscape_ref"]
            records = {record["id"]: record for record in miner.store.all()}
            first_snapshot = records[first_ref["record_id"]]["data"]["snapshot"]
            second_snapshot = records[second_ref["record_id"]]["data"]["snapshot"]
            self.assertEqual(first_snapshot["status"], "no_members")
            self.assertEqual({member["candidate_ref"] for member in second_snapshot["members"]}, {
                "landscape-fixture/candidate-0001", "landscape-fixture/candidate-0002"})
            by_ref = {member["candidate_ref"]: member for member in second_snapshot["members"]}
            self.assertEqual(by_ref["landscape-fixture/candidate-0001"]["calculation_status"], "computed")
            self.assertEqual(by_ref["landscape-fixture/candidate-0002"]["calculation_status"],
                             "calculation_failed")
            self.assertIn("candidate-0001-calculation",
                          by_ref["landscape-fixture/candidate-0001"]["source_evidence_refs"])
            self.assertIn("candidate-0001-evaluation",
                          by_ref["landscape-fixture/candidate-0001"]["source_evidence_refs"])
            self.assertIn("candidate-0002-calculation",
                          by_ref["landscape-fixture/candidate-0002"]["source_evidence_refs"])
            second_request_record_ids = {record["id"] for record in ideations[1]["records"]}
            self.assertTrue({"candidate-0001-calculation", "candidate-0001-evaluation",
                             "candidate-0002-calculation"} <= second_request_record_ids)

            pending = ideations[1]["payload"]["pending_proposals"]["proposal-D"]
            expected_pending = {**model.first_proposal, "status": "pending"}
            self.assertEqual(pending, expected_pending)
            self.assertEqual(pending["experiment_design"]["metric"], "rank_displacement")
            self.assertEqual((pending["experiment_design"]["horizon_hours"],
                              pending["experiment_design"]["displacement_hours"]), (4, 1))

            snapshot_bytes = {record_id: (miner.store.root / f"{record_id}.json").read_bytes()
                              for record_id in (first_ref["record_id"], second_ref["record_id"])}
            request_count = len(model.requests)
            with patch(
                    "crypto_quant.research.factor_mining.landscape.build_factor_landscape",
                    side_effect=AssertionError("saved snapshots must be reused on resume")):
                reopened = FactorMiner.open(miner.root, model)
                resumed = reopened.resume_explore(input_panel)
            self.assertEqual(resumed["completed_rounds"], 2)
            self.assertEqual(len(model.requests), request_count)
            for record_id, data in snapshot_bytes.items():
                self.assertEqual((miner.store.root / f"{record_id}.json").read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
