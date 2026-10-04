"""Deterministic rank-displacement integration and archive recovery checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS
from crypto_quant.research.factor_mining.contracts import ResearchSpec, dumps
from crypto_quant.research.factor_mining.evaluation import RANK_DISPLACEMENT_VERSION
from crypto_quant.research.factor_mining.factor_archive import EvaluationKey, FactorArchive, FactorIdentity
from crypto_quant.research.factor_mining.model import ModelReply
from crypto_quant.research.factor_mining.records import EvidenceIntegrityError, RecordStore, compact_record
from crypto_quant.research.factor_mining.reporting import write_research_report
from crypto_quant.research.factor_mining.workflow import FactorMiner


def _spec() -> ResearchSpec:
    return ResearchSpec(
        run_id="rank-displacement-flow", objective="检验排名变化的确定性归档和A/B流程",
        purpose="research", a_start="2026-09-01T00:00:00Z", b_start="2026-09-05T00:00:00Z",
        c_start="2026-09-09T00:00:00Z", c_end="2026-09-13T00:00:00Z",
        universe_provenance="six-asset deterministic engineering fixture",
        data_usage_review="synthetic data for workflow validation only",
        label="perp_next_open_24h", sample_hours=1, groups=3, min_symbols=6,
        min_periods=26, hac_lags=23, confidence=0.95, stage_hours=48,
        rolling_periods=26, fdr_method="BH", fdr_alpha=0.05, min_abs_ic=0.02,
        min_directional_spread=0.0001, min_stage_share=0.5, max_repairs=1,
        max_formula_nodes=80, max_lookback_hours=3, context_tokens=2_000_000,
        output_tokens=4000, b_horizons=(1, 4, 24))


def _panel(spec: ResearchSpec, stage: str) -> FactorInputPanel:
    start, end = spec.bounds(stage)
    hours = pd.date_range(start - pd.Timedelta(hours=spec.max_lookback_hours), end,
                          freq="h", inclusive="left")
    symbols = sorted(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"])
    index = pd.MultiIndex.from_product([hours, symbols], names=["timestamp", "symbol"])
    rng = np.random.default_rng(20261003 + (stage == "B"))
    drift = np.linspace(0.00003, 0.00009, len(symbols))
    log_returns = drift[None, :] + rng.normal(0.0, 0.00007, (len(hours), len(symbols)))
    base = np.power(1.7, np.arange(1, len(symbols) + 1, dtype=float)) * 100
    price = base[None, :] * np.exp(np.cumsum(log_returns, axis=0))
    values = pd.DataFrame(1.0, index=index, columns=INPUT_COLUMNS)
    values["perp_open"] = price.reshape(-1)
    values["perp_close"] = (price * np.exp(rng.normal(0.0, 0.0001, price.shape))).reshape(-1)
    values["spot_close"] = values["perp_close"] * 0.998
    values["premium_index"] = rng.normal(0.0, 0.0001, len(index))
    universe = pd.Series(True, index=index, name="eligible")
    return FactorInputPanel(values, universe, {"synthetic": True, "segment": stage})


def _candidate(expression: str, *, parent: str | None = None,
               proposal: str | None = None) -> dict:
    return {"name": "合成价格排序信号", "expression": expression,
            "meaning": "工程样本中当前永续价格的横截面排序输入。",
            "hypothesis": "只验证流程接线，不代表预测研究结论。", "direction": 1,
            "parent_id": parent, "proposal_id": proposal,
            "change_reason": "确定性回放采用预先声明的受控版本。"}


def _decision(candidate_id: str, *, continuing: bool) -> dict:
    return {"candidate_id": candidate_id, "disposition": "retain",
            "retained_horizons": [24], "continue_optimization": continuing,
            "answers": {key: {"supported": continuing, "reason": "仅用于确定性闭环回放。"}
                        for key in ("research_basis", "modification_hypothesis",
                                    "verifiable_improvement", "attempt_value")},
            "evidence_refs": [f"{candidate_id}-evaluation"],
            "reason": "脚本固定当前候选去向，结果不表示预测有效。",
            "resume_condition": None}


class DeterministicModel:
    """A no-network model that exercises both legacy and new workflow roles."""

    def __init__(self):
        self.optimizations = 0
        self.requests = []

    def complete(self, messages, *, max_output_tokens, session_id):
        request = json.loads(messages[1]["content"])
        self.requests.append(request)
        role = request["role"]
        if role == "ideator":
            pending = request["payload"]["pending_proposals"]
            if not pending:
                result = {"candidates": [_candidate("perp_close")], "dispositions": [],
                          "analysis": "脚本生成控制候选。"}
            else:
                proposal_id, proposal = next(iter(pending.items()))
                result = {"candidates": [_candidate("ts_mean(perp_close,3)",
                                                      parent=proposal["control_id"],
                                                      proposal=proposal_id)],
                          "dispositions": [{"proposal_id": proposal_id, "action": "adopt",
                                            "reason": "脚本采用已经冻结的修改任务。",
                                            "candidate_index": 0}],
                          "analysis": "脚本生成受控排序平滑版本。"}
        elif role == "optimizer":
            self.optimizations += 1
            candidate_ids = request["payload"]["review_candidate_ids"]
            record_ids = {record["id"] for record in request["records"]}
            proposals = []
            if self.optimizations == 1 and "candidate-0001-evaluation" in record_ids:
                proposals = [{
                    "proposal_id": "rank-displacement-proposal", "route_id": "rank-displacement-route",
                    "control_id": "candidate-0001", "evidence_refs": ["candidate-0001-evaluation"],
                    "modification_task": {
                        "core_hypothesis": "合成价格排序用于闭环回放。",
                        "observed_problem": "逐时排序变化需要确定性诊断。",
                        "modification_hypothesis": "因子值均值平滑可能降低排序变化。",
                        "change_target": "对永续价格因子使用三小时因果均值。",
                        "fixed_components": "字段、方向和原假设保持不变。"},
                    "experiment_design": {
                        "question": "平滑是否减少1小时排名变化并保留24小时Rank IC？",
                        "metric": "rank_displacement", "horizon_hours": 24,
                        "displacement_hours": 1, "min_improvement": 0.01,
                        "max_ic_loss": 0.2,
                        "expected_outcome": "共同样本上的D降低且允许的有向IC损失未超限。",
                        "stop_condition": "D改善或IC损失上界不满足预设条件时停止。",
                        "pause_condition": "区间无法识别结果或共同样本不足时暂停。"},
                    "restart_of": None, "new_evidence": None}]
            decisions = [_decision(cid, continuing=bool(proposals and cid == "candidate-0001"))
                         for cid in candidate_ids]
            result = {"analysis": "脚本只校验程序边界。", "decisions": decisions,
                      "diagnostics": ["研究结论待真实数据复验。"], "proposals": proposals}
        elif role == "evaluator":
            result = {"analysis": "确定性模型解释仅用于流程回放。", "mechanism": "合成信号无金融结论。",
                      "conditions": ["工程样本。"], "falsifiers": ["不适用于真实预测结论。"],
                      "limitations": ["未使用真实市场数据。"], "next_steps": ["独立评价固定样本。"]}
        else:
            raise AssertionError(role)
        return ModelReply(dumps({"result": result, "read_records": []}), {"fixture": True}, "scripted")


class RankDisplacementWorkflowTests(unittest.TestCase):
    def test_diagnostic_closed_loop_archive_recovery_and_card(self):
        spec = _spec()
        panel_a, panel_b = _panel(spec, "A"), _panel(spec, "B")
        model = DeterministicModel()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            miner = FactorMiner(spec, model, root / "runs")
            completion = miner.explore(panel_a)
            self.assertEqual(completion["candidate_ids"], ["candidate-0001", "candidate-0002"])
            comparison_record = next(record for record in miner.store.all()
                                     if record["id"] == "candidate-0002-comparison")
            comparison = comparison_record["data"]
            self.assertEqual(comparison["plan"]["experiment_design"]["metric"], "rank_displacement")
            self.assertEqual(comparison["horizon_hours"], 24)
            self.assertEqual(comparison["displacement_hours"], 1)
            self.assertEqual(completion["routes"]["rank-displacement-route"]["decision"],
                             comparison["decision"])
            self.assertIn(comparison["decision"], {"continue", "stop", "pause_insufficient"})
            pair_page = miner.store.read("candidate-0002-comparison", "/paired_periods", 0, 3)
            self.assertEqual(pair_page["total"], len(comparison["paired_periods"]))
            self.assertEqual(len(pair_page["items"]), 3)
            compact_pair = compact_record(comparison_record)
            self.assertEqual(compact_pair["data"]["paired_periods"]["pointer"], "/paired_periods")

            a_record = next(record for record in miner.store.all()
                            if record["id"] == "candidate-0001-evaluation")
            a_data = a_record["data"]
            displacement = a_data["rank_displacement"]
            self.assertEqual(displacement["definition_version"], RANK_DISPLACEMENT_VERSION)
            self.assertEqual(set(displacement["deltas"]), {"1", "4", "24"})
            self.assertEqual(a_data["rank_displacement_archive"]["evaluation_key"]["horizon"],
                             "rank-displacement")
            self.assertEqual(a_data["factor_archive"]["evaluation_key"]["evaluator_version"],
                             "factor-eval-v1")
            a_archive = FactorArchive.open_existing(
                miner.archive_root, FactorIdentity(**a_data["factor_archive"]["identity"]))
            old_payload = a_archive.get_evaluation(EvaluationKey(**a_data["factor_archive"]["evaluation_key"]))["payload"]
            self.assertNotIn("rank_displacement", old_payload)
            diagnostic_page = miner.store.read(
                "candidate-0001-evaluation", "/rank_displacement/deltas/1/periods", 0, 3)
            self.assertEqual(diagnostic_page["total"], int(
                (spec.bounds("A")[1] - spec.bounds("A")[0]) / pd.Timedelta(hours=1)))
            self.assertEqual(len(diagnostic_page["items"]), 3)
            compact_a = compact_record(a_record)
            self.assertEqual(compact_a["data"]["rank_displacement"]["deltas"]["1"]["periods"]["pointer"],
                             "/rank_displacement/deltas/1/periods")
            a_report_text = (miner.root / "A-report.md").read_text()
            self.assertIn("信号持续性与换手倾向代理", a_report_text)
            self.assertIn("排名变化受控修改配对结果", a_report_text)
            self.assertIn("逐期配对证据与合同", a_report_text)

            restored = FactorMiner.open(miner.root, model)
            resumed = restored.resume_explore(panel_a)
            self.assertEqual(resumed["routes"], completion["routes"])
            restored.freeze(["candidate-0001"], panel_b.universe)
            frozen = json.loads((miner.root / "frozen_batch.json").read_text())
            frozen_candidate = frozen["candidates"]["candidate-0001"]
            self.assertEqual(frozen_candidate["rank_displacement_version"], RANK_DISPLACEMENT_VERSION)
            self.assertEqual(frozen_candidate["a_rank_displacement_archive"],
                             a_data["rank_displacement_archive"])

            candidate_two = next(record for record in restored.store.all()
                                 if record["id"] == "candidate-0002-evaluation")["data"]
            raw_a_path = miner.root / "a_records" / "candidate-0001-evaluation.json"
            original = raw_a_path.read_text()
            tampered = json.loads(original)
            tampered["data"]["rank_displacement_archive"] = candidate_two["rank_displacement_archive"]
            raw_a_path.write_text(dumps(tampered) + "\n")
            with self.assertRaises(EvidenceIntegrityError):
                restored.store._load(raw_a_path)
            raw_a_path.write_text(original)
            restored._checked_frozen()

            idea_pool = root / "ideas"
            validation = restored.validate(lambda: panel_b, idea_pool)
            self.assertEqual(validation["batch_correction"]["family_size"], 1)
            card_path = next(idea_pool.glob("*.json"))
            card = json.loads(card_path.read_text())
            self.assertIn("signal_persistence", card)
            self.assertIn("coverage", card["signal_persistence"]["A"]["deltas"]["1"])
            self.assertIn("coverage", card["signal_persistence"]["B"]["deltas"]["1"])
            self.assertEqual(card["evidence_references"]["B"]["rank_displacement_archive"]["evaluation_key"]["horizon"],
                             "rank-displacement")
            self.assertIn("信号持续性与换手倾向代理", (miner.root / "B-report.md").read_text())

            raw_b_path = miner.root / "b_records" / "candidate-0001-evaluation.json"
            original_b = raw_b_path.read_text()
            tampered_b = json.loads(original_b)
            tampered_b["data"]["rank_displacement_archive"] = candidate_two["rank_displacement_archive"]
            raw_b_path.write_text(dumps(tampered_b) + "\n")
            with self.assertRaises(EvidenceIntegrityError):
                RecordStore(miner.root / "b_records")._load(raw_b_path)
            raw_b_path.write_text(original_b)

            recovered = FactorMiner.open(miner.root, model)
            completed = recovered.complete_reports("B", idea_pool)
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(json.loads(card_path.read_text()), card)

    def test_legacy_report_marks_rank_displacement_uncomputed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = {"id": "candidate-legacy-evaluation", "kind": "evaluation", "data": {
                "candidate_id": "candidate-legacy", "segment": "A", "direction": 1,
                "horizon_hours": 24,
                "summary": {"rank_ic": {"mean": 0.1, "mean_std_ratio": 0.3, "n": 26},
                            "directional_spread": {"mean": 0.02}},
                "coverage": {"purged_hours": 0}}}
            path = root / "legacy-report.md"
            write_research_report(path, "legacy-run", "research", [record], "A")
            self.assertIn("未计算：该次运行没有保存排名变化诊断。", path.read_text())


if __name__ == "__main__":
    unittest.main()
