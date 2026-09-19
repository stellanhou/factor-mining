"""Research-boundary tests and an explicitly scripted two-round integration run."""

import copy
import json
import math
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from crypto_quant.features.factor_expressions import compile_expression, evaluate_expression
from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS
from crypto_quant.research.factor_mining.contracts import ResearchSpec, digest, dumps
from crypto_quant.research.factor_mining.evaluation import build_labels, compare_experiment, correct_batch, evaluate_factor, hac_mean
from crypto_quant.research.factor_mining.model import ApiCallError, ModelReply, OpenCodeGoModel, read_api_key
from crypto_quant.research.factor_mining.records import AgentGateway, EvidenceIntegrityError, ContextBudgetError, ModelResponseError, RecordStore, compact_record
from crypto_quant.research.factor_mining.workflow import CHECK_DIMENSIONS, RESEARCH_QUESTIONS, FactorMiner, _check_report


def specification(**overrides):
    values = dict(
        run_id="engineering-fixture", objective="检查两轮上下文和时间边界", purpose="engineering_check",
        a_start="2026-07-01T00:00:00Z", b_start="2026-07-05T00:00:00Z",
        c_start="2026-07-09T00:00:00Z", c_end="2026-07-13T00:00:00Z",
        universe_provenance="synthetic six-asset fixture, not a historical selection rule",
        data_usage_review="engineering fixtures only; these dates are not held-out research evidence",
        label="perp_next_open_24h", sample_hours=1, groups=3, min_symbols=6,
        min_periods=26, hac_lags=23, confidence=0.95, stage_hours=48, rolling_periods=26,
        fdr_method="BY", fdr_alpha=0.05, min_abs_ic=0.02, min_directional_spread=0.0001,
        min_stage_share=0.5, max_repairs=2, max_formula_nodes=80, max_lookback_hours=7,
        context_tokens=2000000, output_tokens=4000)
    values.update(overrides)
    return ResearchSpec(**values)


def input_panel(spec, stage="A"):
    start, end = spec.bounds(stage)
    hours = pd.date_range(start - pd.Timedelta(hours=spec.max_lookback_hours), end, freq="h", inclusive="left")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]
    index = pd.MultiIndex.from_product([hours, sorted(symbols)], names=["timestamp", "symbol"])
    rng = np.random.default_rng(314)
    returns = rng.normal(0.0001, 0.003, (len(hours), len(symbols)))
    price = np.exp(np.cumsum(returns, axis=0)) * np.arange(1, 7)[None, :]
    values = pd.DataFrame(1.0, index=index, columns=INPUT_COLUMNS)
    values["perp_open"] = price.reshape(-1)
    values["perp_close"] = (price * np.exp(rng.normal(0, 0.001, price.shape))).reshape(-1)
    values["spot_close"] = values["perp_close"] * 0.998
    values["premium_index"] = rng.normal(0, 0.001, len(index))
    return FactorInputPanel(values, pd.Series(True, index=index), {"fixture": True, "segment": stage})


def definition(expression="div(perp_close,ts_mean(spot_close,3))", parent=None, proposal=None, meaning=None):
    return {"name": "price_deviation", "expression": expression,
            "meaning": meaning or "每个历史币池成员的当前已收盘永续价格除以包含当前小时的三小时永续价格均值，无量纲",
            "hypothesis": "偏离可能产生反转，待验证", "direction": -1, "parent_id": parent,
            "proposal_id": proposal, "change_reason": "依据明确计算定义检验"}


def modification_task(change_target="将近期价格基准窗口设为6小时，并对偏离值使用3小时均值"):
    return {"core_hypothesis": "价格偏离近期基准反映拥挤，未来可能反转",
            "observed_problem": "原候选的A段分阶段IC显示短期波动可能掩盖拥挤关系",
            "modification_hypothesis": "较稳定的近期基准可能更好表达同一反转假设",
            "change_target": change_target,
            "fixed_components": "价格字段、偏离比值、预测方向和24小时标签保持不变"}


def experiment_design(question="短期波动是否掩盖偏离所代表的拥挤"):
    return {"question": question, "metric": "rank_ic", "min_improvement": 0.01, "max_ic_loss": 0.02,
            "expected_outcome": "在同币池、日期、方向和样本中观察有向IC改善",
            "stop_condition": "改善上限低于0.01或有向IC损失超过上限则停止",
            "pause_condition": "区间不能区分改善与恶化或数据不足则暂停"}


def optimization_proposal(proposal_id="proposal-1", control="candidate-0001", route="window-route",
                          change_target=None, question=None):
    return {"proposal_id": proposal_id, "route_id": route, "control_id": control,
            "evidence_refs": [f"{control}-evaluation"],
            "modification_task": modification_task(change_target) if change_target else modification_task(),
            "experiment_design": experiment_design(question) if question else experiment_design(),
            "restart_of": None, "new_evidence": None}


def experiment(**overrides):
    proposal = optimization_proposal(**overrides)
    proposal.pop("proposal_id")
    return proposal


def definition_from_task(proposal_id, proposal):
    target = proposal["modification_task"]["change_target"]
    windows = [int(value) for value in re.findall(r"(\d+)小时", target)]
    window = windows[0]
    if len(windows) > 1:
        expression = f"ts_mean(div(perp_close,ts_mean(perp_close,{window})),{windows[1]})"
        meaning = (f"每个历史币池成员的当前已收盘永续价格除以包含当前小时的{window}小时永续价格均值，"
                   f"再对该偏离做包含当前小时的{windows[1]}小时均值，无量纲")
    else:
        expression = f"div(perp_close,ts_mean(perp_close,{window}))"
        meaning = f"每个历史币池成员的当前已收盘永续价格除以包含当前小时的{window}小时永续价格均值，无量纲"
    return definition(expression, proposal["control_id"], proposal_id, meaning)


def narrative():
    return {"analysis": "这是脚本模型返回，用于验证数据流，不是模型研究结论。",
            "mechanism": "反转解释待检验", "conditions": ["稳定市场环境待确认"], "falsifiers": ["关系跨阶段反向"],
            "limitations": ["脚本模型不具备推理能力，研究有效性未验证"], "next_steps": ["确认真实模型与研究合同后开展研究"]}


def research_decision(cid, disposition, continuing=False, evidence=None):
    return {"candidate_id": cid, "disposition": disposition, "continue_optimization": continuing,
            "answers": {key: {"supported": continuing, "reason": "脚本场景用于检查决策分流，不是实际研究判断"}
                        for key in RESEARCH_QUESTIONS},
            "evidence_refs": [evidence or f"{cid}-evaluation"],
            "reason": "脚本指定当前去向，核对程序是否按决定执行；不表示因子有效。",
            "resume_condition": "补齐定义或研究证据后另行决定恢复" if disposition == "pause" else None}


class ScenarioModel:
    """No network. Responses encode known scenarios, never stand in for live LLM quality."""
    def __init__(self, transform=None, *, propose_once=True):
        self.requests = []
        self.ideations = 0
        self.optimizations = 0
        self.transform = transform
        self.propose_once = propose_once
        self.responses = {}

    def complete(self, messages, *, max_output_tokens, session_id):
        request = json.loads(messages[1]["content"])
        self.requests.append(copy.deepcopy(request))
        role = request["role"]
        cache_key = messages[1]["content"]
        if cache_key not in self.responses:
            if role == "ideator":
                self.ideations += 1
                if self.ideations == 1:
                    result = {"candidates": [definition(), definition("ts_mean(perp_close)", meaning="近期均值，窗口未定义")],
                              "dispositions": [], "analysis": "构造一个市场错误和一个定义不足的例子"}
                else:
                    proposal_id, proposed = next(iter(request["payload"]["pending_proposals"].items()))
                    result = {"candidates": [definition_from_task(proposal_id, proposed)], "dispositions": [
                        {"proposal_id": proposal_id, "action": "adopt", "reason": "检验预先声明的窗口原因",
                         "candidate_index": 0}], "analysis": "根据修改任务首次生成公式，不改写配对判断标准"}
            elif role == "calculator":
                payload = request["payload"]
                expression = payload["current_expression"]
                insufficient = "未定义" in payload["original_definition"]["meaning"]
                mismatch = "spot_close" in expression
                checks = [{"dimension": d, "original_meaning": payload["original_definition"]["meaning"],
                           "actual": "脚本核对已提供的程序步骤", "matches": not insufficient and not (mismatch and d == "fields_units"),
                           "reason": "模拟核对该已知错误"} for d in sorted(CHECK_DIMENSIONS)]
                result = {"candidate_id": payload["candidate_id"],
                          "status": "insufficient_definition" if insufficient else "inconsistent" if mismatch else "consistent",
                          "checks": checks, "repair_expression": expression.replace("spot_close", "perp_close") if mismatch else None,
                          "reason": "按原始含义核对，不修改含义"}
            elif role == "evaluator":
                result = narrative()
            elif role == "optimizer":
                self.optimizations += 1
                proposals = []
                if (self.propose_once and self.optimizations == 1
                        and any(r["id"] == "candidate-0001-evaluation" for r in request["records"])):
                    proposals = [optimization_proposal()]
                record_ids = {r["id"] for r in request["records"]}
                decisions = []
                for cid in request["payload"]["review_candidate_ids"]:
                    evaluated = f"{cid}-evaluation" in record_ids
                    decisions.append(research_decision(cid, "retain" if evaluated else "pause",
                        bool(proposals) and cid == "candidate-0001",
                        f"{cid}-evaluation" if evaluated else f"{cid}-calculation"))
                result = {"analysis": "保留完整分析，脚本场景只进行一次有依据的窗口实验", "decisions": decisions,
                          "diagnostics": ["真实有效性待研究"], "proposals": proposals}
            else:
                raise AssertionError(role)
            self.responses[cache_key] = copy.deepcopy(result)
        result = copy.deepcopy(self.responses[cache_key])
        if self.transform is not None:
            self.transform(request, result)
        return ModelReply(dumps({"result": result, "read_records": []}), {"fixture": True}, "scripted-fixture")


def four_destination_scenario(request, result):
    if request["role"] == "ideator" and not request["payload"]["pending_proposals"]:
        result["candidates"] = [definition(f"div(perp_close,ts_mean(perp_close,{window}))",
            meaning=f"当前已收盘永续价格除以包含当前小时的{window}小时永续价格均值，无量纲")
            for window in (3, 2, 4, 5)]
    if request["role"] == "optimizer" and request["payload"]["round_candidate_ids"][0] == "candidate-0001":
        result["decisions"] = [research_decision("candidate-0001", "optimize", True),
                               research_decision("candidate-0002", "retain"),
                               research_decision("candidate-0003", "pause"),
                               research_decision("candidate-0004", "discard")]


class EvaluationTests(unittest.TestCase):
    def test_24_hour_prices_exact_shift_and_boundary(self):
        spec = specification()
        panel = input_panel(spec)
        labels = build_labels(panel, spec, "A")
        t = pd.Timestamp(spec.a_start)
        symbol = "BTCUSDT"
        expected = panel.values.loc[(t + pd.Timedelta(hours=25), symbol), "perp_open"] / panel.values.loc[(t + pd.Timedelta(hours=1), symbol), "perp_open"] - 1
        self.assertAlmostEqual(labels.loc[(t, symbol), "forward_return"], expected)
        self.assertEqual(labels.loc[(t, symbol), "label_end"] - labels.loc[(t, symbol), "label_start"], pd.Timedelta(hours=24))
        self.assertTrue(labels.loc[(pd.Timestamp(spec.b_start) - pd.Timedelta(hours=25), symbol), "purged"])
        self.assertEqual(labels.groupby(level="symbol")["purged"].sum().iloc[0], 25)
        self.assertTrue((labels["signal_available_at"] < labels["label_start"]).all())

    def test_missing_price_not_zero_or_compressed_time(self):
        spec = specification()
        panel = input_panel(spec)
        t = pd.Timestamp(spec.a_start)
        panel.values.loc[(t + pd.Timedelta(hours=25), "BTCUSDT"), "perp_open"] = np.nan
        self.assertTrue(pd.isna(build_labels(panel, spec, "A").loc[(t, "BTCUSDT"), "forward_return"]))

    def test_cannot_build_labels_from_future_segment(self):
        with self.assertRaisesRegex(ValueError, "protected"):
            build_labels(input_panel(specification(), "B"), specification(), "A")
        with self.assertRaisesRegex(ValueError, "reserved"):
            specification().bounds("C")

    def test_hac_matches_independent_bartlett_matrix_with_missing_time(self):
        spec = specification(min_periods=26)
        rng = np.random.default_rng(71)
        x = rng.normal(size=70)
        x[[4, 5, 30, 51]] = np.nan
        n = np.isfinite(x).sum()
        residual = np.nan_to_num(x - np.nanmean(x))
        distance = np.abs(np.arange(len(x))[:, None] - np.arange(len(x))[None, :])
        kernel = np.maximum(1 - distance / (spec.hac_lags + 1), 0)
        quadratic = math.fsum(float(residual[i] * kernel[i, j] * residual[j])
                              for i in range(len(x)) for j in range(len(x)))
        expected = quadratic / n**2 * n / (n - 1)
        result = hac_mean(pd.Series(x), spec)
        self.assertAlmostEqual(result["se"]**2, expected)
        compressed = hac_mean(pd.Series(x).dropna(), spec)
        self.assertNotAlmostEqual(result["se"], compressed["se"])

    def test_zero_lag_hac_is_sample_standard_error(self):
        spec = specification(sample_hours=24, hac_lags=0, min_periods=4)
        x = pd.Series([1., 2., 3., 4.])
        self.assertAlmostEqual(hac_mean(x, spec)["se"], x.std(ddof=1) / 2)
        self.assertIsNone(hac_mean(pd.Series([1.] * 40), spec)["p_value"])

    def test_tied_values_not_arbitrarily_split_into_groups(self):
        spec = specification()
        panel = input_panel(spec)
        labels = build_labels(panel, spec, "A")
        values = pd.Series(np.tile([1., 1., 1., 1., 2., 3.], len(panel.values) // 6), index=panel.values.index)
        report = evaluate_factor(values, labels, spec, "A", 1)
        first = report["periods"][0]
        self.assertEqual(first["group_1_n"], 4)
        self.assertEqual(sum(first[f"group_{i}_n"] for i in range(1, 4)), 6)

    def test_membership_is_point_in_time_and_future_changes_leave_past_factor_alone(self):
        spec = specification()
        panel = input_panel(spec)
        t = pd.Timestamp(spec.a_start) + pd.Timedelta(hours=10)
        panel.universe.loc[(t, "BTCUSDT")] = False
        expr = "cross_rank(ts_return(perp_close,3))"
        before = evaluate_expression(expr, panel).values
        changed = copy.deepcopy(panel)
        changed.values.loc[changed.values.index.get_level_values("timestamp") > t, "perp_close"] *= 100
        after = evaluate_expression(expr, changed).values
        pd.testing.assert_series_equal(before.loc[:t], after.loc[:t])
        self.assertTrue(pd.isna(before.loc[(t, "BTCUSDT")]))

    def test_batch_keeps_missing_tests_in_family(self):
        spec = specification(fdr_method="BH")
        reports = {"a": {"summary": {"rank_ic": {"p_value": .01}, "directional_spread": {"p_value": .04}}},
                   "b": {"summary": {"rank_ic": {"p_value": .03}, "directional_spread": {"p_value": None}}}}
        bh = correct_batch(reports, spec)
        self.assertEqual(bh["family_size"], 4)
        np.testing.assert_allclose([t["adjusted_p"] for t in bh["tests"]], [.04, .04 * 4 / 3, .04 * 4 / 3, 1])
        self.assertIsNone(bh["tests"][-1]["raw_p"])
        self.assertFalse(bh["tests"][-1]["rejected"])
        by = correct_batch(reports, replace(spec, fdr_method="BY"))
        self.assertGreater(by["tests"][0]["adjusted_p"], bh["tests"][0]["adjusted_p"])
        dumps(by)

    def test_compiler_steps_preserve_nested_order_and_market(self):
        compiled = compile_expression("ts_delay(ts_mean(spot_close,3),2)")
        steps = compiled.calculation_steps()
        self.assertEqual(steps[0]["market"], "spot")
        self.assertEqual(steps[-1]["window_hours"], 2)
        self.assertEqual(steps[-1]["arguments"][0], 3)
        self.assertEqual(steps[2]["window_hours"], 3)


class WorkflowTests(unittest.TestCase):
    def test_all_roles_and_nested_objects_accept_extras_without_changing_research(self):
        scripted = ScenarioModel()
        def decorate(value):
            if isinstance(value, dict):
                return {**{k: decorate(v) for k, v in value.items()}, "extra_note": "unused-extra-value"}
            if isinstance(value, list):
                return [decorate(v) for v in value]
            return value
        class ExtendedReplies:
            def complete(self, messages, **kwargs):
                reply = scripted.complete(messages, **kwargs)
                envelope = decorate(json.loads(reply.text))
                if json.loads(messages[1]["content"])["role"] == "ideator":
                    envelope["result"]["read_records"] = []
                return replace(reply, text=dumps(envelope))
        spec = specification()
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, ExtendedReplies(), Path(directory))
            result = miner.explore(input_panel(spec))
            self.assertEqual(result["completed_rounds"], 2)
            self.assertEqual(result["evaluated_ids"], ["candidate-0001", "candidate-0003"])
            records = miner.store.all()
            notes = [r["data"] for r in records if r["kind"] == "format_deviation"]
            self.assertEqual({n["role"] for n in notes}, {"ideator", "calculator", "evaluator", "optimizer"})
            paths = {p for note in notes for p in note["extra_fields"]}
            for path in ("/extra_note", "/result/read_records", "/result/candidates/0/extra_note",
                         "/result/checks/0/extra_note", "/result/proposals/0/modification_task/extra_note",
                         "/result/proposals/0/experiment_design/extra_note",
                         "/result/dispositions/0/extra_note",
                         "/result/decisions/0/answers/research_basis/extra_note"):
                self.assertIn(path, paths)
            for record in records:
                if record["kind"] != "format_deviation":
                    self.assertNotIn("unused-extra-value", dumps(record))
            for request in scripted.requests:
                self.assertNotIn("unused-extra-value", dumps(request["records"]))
            for note in notes:
                self.assertIn("unused-extra-value", (miner.root / note["raw_response"]).read_text())
            self.assertIn("模型回复格式记录", (miner.root / "A-report.md").read_text())

    def test_read_requests_and_envelope_extras_do_not_enter_followup_context(self):
        spec, model = specification(), Mock()
        envelope = {"result": None, "read_records": [{"record_id": "facts", "pointer": "/values",
                    "offset": 0, "limit": 1, "note": "unused-extra-value"}], "note": "unused-extra-value"}
        model.complete.side_effect = [ModelReply(dumps(envelope), {}, "fixture"),
                                     ModelReply(dumps({"result": {}, "read_records": []}), {}, "fixture")]
        with tempfile.TemporaryDirectory() as directory:
            store = RecordStore(Path(directory) / "records")
            store.append("facts", "fact", {"values": [123]})
            gateway = AgentGateway(model, spec, store, Path(directory) / "calls")
            self.assertEqual(gateway.ask("optimizer", "read", {}, {}), {})
            second_messages = model.complete.call_args_list[1].args[0]
            self.assertNotIn("unused-extra-value", dumps(second_messages))
            self.assertEqual(json.loads(second_messages[-1]["content"])["requested_original_evidence"][0]["data"]["items"], [123])
            note = next(r for r in store.all() if r["kind"] == "format_deviation")
            self.assertEqual(note["data"]["extra_fields"], ["/note", "/read_records/0/note"])

    def test_nested_missing_fields_and_types_are_rejected_with_local_calculator_failure(self):
        cases = [
            ("ideator", lambda r: r["candidates"][0].pop("expression"), "missing required fields: expression"),
            ("ideator", lambda r: r["candidates"][0].update(direction="-1"), "direction must be integer"),
            ("calculator", lambda r: r["checks"][0].pop("matches"), "missing required fields: matches"),
            ("calculator", lambda r: r["checks"][0].update(matches="true"), "matches must be boolean"),
            ("optimizer", lambda r: r["proposals"][0]["experiment_design"].pop("expected_outcome"),
             "missing required fields: expected_outcome"),
            ("optimizer", lambda r: r["proposals"][0]["modification_task"].pop("change_target"),
             "missing required fields: change_target"),
            ("optimizer", lambda r: r["decisions"][0]["answers"]["research_basis"].pop("reason"), "missing required fields: reason"),
        ]
        for role, mutate, error in cases:
            def transform(request, result):
                if request["role"] == role:
                    result["extra_note"] = "unused-extra-value"
                    mutate(result)
            with self.subTest(role=role, error=error), tempfile.TemporaryDirectory() as directory:
                miner = FactorMiner(specification(), ScenarioModel(transform), Path(directory))
                if role == "calculator":
                    result = miner.explore(input_panel(miner.spec))
                    self.assertEqual(result["evaluated_ids"], [])
                    self.assertTrue(all(item["calculation"]["status"] == "model_review_failed"
                                        for item in miner.candidates.values()))
                    self.assertFalse((miner.root / "exploration-stopped.json").exists())
                else:
                    with self.assertRaisesRegex(ModelResponseError, error):
                        miner.explore(input_panel(miner.spec))
                    self.assertTrue((miner.root / "exploration-stopped.json").exists())
                    self.assertFalse((miner.root / "a-complete.json").exists())
                invalid = [r for r in miner.store.all() if r["kind"] == "invalid_model_response"]
                self.assertTrue(any(error in r["data"]["error"] for r in invalid))
                self.assertTrue(any(r["kind"] == "format_deviation" for r in miner.store.all()))

    def test_extra_report_fields_are_preserved_raw_but_not_forwarded_in_A_or_B(self):
        def add_fields(request, result):
            if request["role"] == "evaluator":
                result.update(conditions_note=None, decision="discard", extra_decision={"continue_optimization": True})
        spec, model = specification(), ScenarioModel(add_fields, propose_once=False)
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            result = miner.explore(input_panel(spec))
            optimizer = next(r for r in model.requests if r["role"] == "optimizer")
            report = next(r["data"] for r in optimizer["records"] if r["kind"] == "model_report")
            self.assertEqual(set(report), set(narrative()) | {"candidate_id"})
            self.assertNotIn("extra_decision", report)
            self.assertNotIn("decision", report)
            miner.freeze(result["retained_ids"], input_panel(spec, "B").universe)
            miner.validate(lambda: input_panel(spec, "B"), Path(directory) / "ideas")
            for stage in ("A", "B"):
                note = json.loads(next((miner.root / f"{stage.lower()}_records").glob("model-*-format.json")).read_text())
                self.assertEqual(note["data"]["extra_fields"], ["/result/conditions_note", "/result/decision", "/result/extra_decision"])
                self.assertNotIn("extra_decision", note["data"])
                raw = json.loads((miner.root / note["data"]["raw_response"]).read_text())
                response = json.loads(raw["text"])["result"]
                self.assertIsNone(response["conditions_note"])
                self.assertEqual(response["decision"], "discard")
                self.assertTrue(response["extra_decision"]["continue_optimization"])
                self.assertIn("conditions_note", (miner.root / f"{stage}-report.md").read_text())

    def test_missing_or_invalid_core_report_fields_still_fail(self):
        for field in narrative():
            bad = {**narrative(), "conditions_note": None}
            del bad[field]
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, f"missing required fields: {field}"):
                _check_report(bad)
        for field, wrong in (("analysis", None), ("mechanism", ""),
                             ("conditions", "text"), ("falsifiers", []),
                             ("limitations", [None]), ("next_steps", [""])):
            with self.subTest(field=field, wrong=wrong), self.assertRaisesRegex(ValueError, field):
                _check_report({**narrative(), field: wrong, "conditions_note": None})

    def test_invalid_report_is_pending_and_preserves_evidence(self):
        def invalid_report(request, result):
            if request["role"] == "evaluator":
                result.update(analysis=None, conditions_note=None)
        spec, model = specification(), ScenarioModel(invalid_report)
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            result = miner.explore(input_panel(spec))
            self.assertTrue(any(r["role"] == "optimizer" for r in model.requests))
            self.assertFalse((miner.root / "exploration-stopped.json").exists())
            self.assertEqual(result["pending_report_ids"], result["evaluated_ids"])
            self.assertTrue(list((miner.root / "a_records").glob("model-*-format.json")))
            self.assertFalse((miner.root / "a_records/candidate-0001-report.json").exists())

    def test_four_destinations_control_next_round_and_freeze(self):
        spec = specification()
        model = ScenarioModel(four_destination_scenario)
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            result = miner.explore(input_panel(spec))
            second = [r for r in model.requests if r["role"] == "ideator"][1]
            states = second["payload"]["candidate_decisions"]
            self.assertEqual([states[f"candidate-000{i}"]["disposition"] for i in range(1, 5)],
                             ["optimize", "retain", "pause", "discard"])
            self.assertEqual({p["control_id"] for p in second["payload"]["pending_proposals"].values()},
                             {"candidate-0001"})
            second_review = [r for r in model.requests if r["role"] == "optimizer"][1]
            self.assertEqual(second_review["payload"]["review_candidate_ids"], ["candidate-0005", "candidate-0001"])
            self.assertEqual(len(miner.candidates), 5)
            self.assertEqual(miner.candidates["candidate-0005"]["definition"]["parent_id"], "candidate-0001")
            self.assertEqual(set(result["retained_ids"]), {"candidate-0001", "candidate-0002", "candidate-0005"})
            restored = FactorMiner.open(miner.root, model=None)
            for cid in ("candidate-0003", "candidate-0004"):
                with self.subTest(cid=cid), self.assertRaisesRegex(ValueError, "explicitly retained"):
                    restored.freeze([cid], input_panel(spec, "B").universe)
            self.assertFalse((miner.root / "b-access-started.json").exists())
            restored.freeze(result["retained_ids"], input_panel(spec, "B").universe)
            frozen = json.loads((miner.root / "frozen_batch.json").read_text())
            self.assertEqual(frozen["candidates"]["candidate-0002"]["a_decision"], states["candidate-0002"])
            report = (miner.root / "A-report.md").read_text()
            for label in ("保留待B验证", "暂停", "淘汰", "恢复条件", "有研究依据"):
                self.assertIn(label, report)

    def test_retaining_parent_and_stopping_modification_are_independent(self):
        spec, model = specification(), ScenarioModel()
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            from crypto_quant.research.factor_mining import workflow
            original = workflow.compare_experiment
            def stopped(*args):
                return {**original(*args), "decision": "stop"}
            with patch.object(workflow, "compare_experiment", side_effect=stopped):
                result = miner.explore(input_panel(spec))
            first_optimization = next(r for r in miner.store.all() if r["id"] == "round-001-optimization")
            first = first_optimization["data"]["decisions"][0]
            self.assertEqual(first["disposition"], "retain")
            self.assertTrue(first["continue_optimization"])
            self.assertEqual(result["routes"]["window-route"]["decision"], "stop")
            self.assertIn("candidate-0001", result["retained_ids"])

    def test_no_authorized_optimization_ends_loop_naturally(self):
        def pause_all(request, result):
            if request["role"] == "optimizer":
                result["proposals"] = []
                for d in result["decisions"]:
                    d.update(disposition="pause", continue_optimization=False, resume_condition="补充样本和诊断")
        spec, model = specification(), ScenarioModel(pause_all)
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            result = miner.explore(input_panel(spec))
            self.assertEqual(result["completed_rounds"], 1)
            self.assertEqual(result["retained_ids"], [])
            self.assertEqual(model.ideations, 1)
            self.assertFalse((miner.root / "b-access-started.json").exists())

    def test_bad_decisions_stop_before_next_round(self):
        cases = [
            (lambda r: r["decisions"].pop(), "needs one decision"),
            (lambda r: r["decisions"].__setitem__(1, copy.deepcopy(r["decisions"][0])), "repeated candidate"),
            (lambda r: r["decisions"][0]["answers"].pop("research_basis"), "missing required fields: research_basis"),
            (lambda r: r["decisions"][0]["answers"]["research_basis"].update(supported=False), "all four questions"),
            (lambda r: r["decisions"][0]["answers"]["attempt_value"].update(supported="true"), "supported must be boolean"),
            (lambda r: r["decisions"][0].update(evidence_refs=["invented-evidence"]), "existing records"),
            (lambda r: r["decisions"][0].update(evidence_refs=["candidate-0002-calculation"]), "this candidate's program evidence"),
            (lambda r: r["decisions"][1].update(resume_condition=None), "pause resume condition"),
            (lambda r: r["decisions"][1].update(disposition="retain", resume_condition=None), "evaluated candidate evidence"),
            (lambda r: r["decisions"][1].update(disposition="discard", resume_condition=None), "failure alone cannot discard"),
            (lambda r: r["decisions"][0].update(continue_optimization=False), "proposal is not authorized"),
            (lambda r: r.update(proposals=[]), "requires an experiment proposal"),
            (lambda r: r["proposals"][0].update(evidence_refs=["invented-evidence"]), "existing records"),
            (lambda r: r["proposals"][0].update(evidence_refs=["candidate-0001-report"]), "control evaluation"),
        ]
        for mutate, error in cases:
            def transform(request, result):
                if request["role"] == "optimizer":
                    mutate(result)
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                spec, model = specification(), ScenarioModel(transform)
                miner = FactorMiner(spec, model, Path(directory))
                with self.assertRaisesRegex(ValueError, error):
                    miner.explore(input_panel(spec))
                self.assertEqual(model.ideations, 1)
                self.assertTrue((miner.root / "exploration-stopped.json").exists())
                self.assertFalse((miner.root / "a-complete.json").exists())

    def test_paused_candidate_cannot_be_revived_by_ideator(self):
        def revive(request, result):
            if request["role"] == "ideator" and request["payload"]["pending_proposals"]:
                result["candidates"].append(definition("ts_mean(perp_close,2)", parent="candidate-0002"))
        spec, model = specification(), ScenarioModel(revive)
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            with self.assertRaisesRegex(ValueError, "does not authorize a new version"):
                miner.explore(input_panel(spec))
            self.assertFalse(any(r["payload"].get("candidate_id") == "candidate-0004" for r in model.requests))

    def test_natural_loop_can_exceed_previous_round_and_candidate_caps(self):
        model = None
        def continue_five_times(request, result):
            if request["role"] != "optimizer":
                return
            records = {record["id"] for record in request["records"]}
            if model.optimizations <= 5:
                control = request["payload"]["round_candidate_ids"][0]
                proposal_id = f"proposal-{model.optimizations}"
                window = 3 + model.optimizations
                result["proposals"] = [optimization_proposal(
                    proposal_id, control, f"route-{model.optimizations}",
                    f"将近期价格基准窗口改为{window}小时，其他部分不变",
                    f"{window}小时窗口是否带来新的可验证信息")]
                result["decisions"] = [research_decision(
                    cid, "optimize" if cid == control else "retain" if f"{cid}-evaluation" in records else "pause",
                    cid == control, f"{cid}-evaluation" if f"{cid}-evaluation" in records else f"{cid}-calculation")
                    for cid in request["payload"]["review_candidate_ids"]]
            else:
                result["proposals"] = []
                result["decisions"] = [research_decision(
                    cid, "retain" if f"{cid}-evaluation" in records else "pause", False,
                    f"{cid}-evaluation" if f"{cid}-evaluation" in records else f"{cid}-calculation")
                    for cid in request["payload"]["review_candidate_ids"]]
        spec = specification()
        model = ScenarioModel(continue_five_times)
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            result = miner.explore(input_panel(spec))
            self.assertEqual(result["completed_rounds"], 6)
            self.assertEqual(len(result["candidate_ids"]), 7)
            self.assertEqual((model.ideations, model.optimizations), (6, 6))
            self.assertEqual(result["stop_reason"], "optimizer returned no authorized optimization proposals")

    def test_optimization_continues_after_many_prior_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, model = specification(), ScenarioModel()
            miner = FactorMiner(spec, model, Path(directory))
            for _ in range(35):
                miner.gateway.ask("evaluator", "prior call fixture", {}, {})
            result = miner.explore(input_panel(spec))
            self.assertEqual(result["completed_rounds"], 2)
            self.assertEqual(len(model.requests), 45)
            for field in ("rounds", "max_candidates", "candidates_per_round", "max_route_attempts", "max_model_calls"):
                self.assertNotIn(field, spec.as_dict())
            decision_request = next(r for r in model.requests if r["role"] == "optimizer")
            self.assertNotIn("remaining_budget", decision_request["payload"])

    def test_semantic_review_cannot_be_attached_to_another_candidate(self):
        spec, model = specification(), Mock()
        wrong = {"candidate_id": "candidate-9999", "status": "consistent", "checks": [], "repair_expression": None,
                 "reason": "wrong candidate"}
        model.complete.return_value = ModelReply(dumps({"result": wrong, "read_records": []}), {}, "fixture")
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            miner.candidates["candidate-0001"] = {"definition": definition("div(perp_close,ts_mean(perp_close,3))")}
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_expression") as compute:
                self.assertIsNone(miner._calculate("candidate-0001", input_panel(spec)))
            compute.assert_not_called()
            self.assertEqual(model.complete.call_count, 3)
            self.assertEqual(miner.candidates["candidate-0001"]["calculation"]["status"], "model_review_failed")

    def test_B_card_admission_is_independent_of_legacy_evaluator_verdict(self):
        for legacy in ("keep", "discard"):
            for p_value, should_admit in ((0.000001, True), (0.8, False)):
                def old_verdict(request, result):
                    if request["role"] == "evaluator":
                        self.assertNotIn("decision", request["output_schema"]["properties"])
                        result["decision"] = legacy
                with self.subTest(legacy=legacy, p=p_value), tempfile.TemporaryDirectory() as directory:
                    spec, model = specification(purpose="research"), ScenarioModel(old_verdict, propose_once=False)
                    miner = FactorMiner(spec, model, Path(directory))
                    miner.explore(input_panel(spec))
                    b_panel = input_panel(spec, "B")
                    miner.freeze(["candidate-0001"], b_panel.universe)
                    controlled_report = copy.deepcopy(miner.candidates["candidate-0001"]["evaluation"])
                    controlled_report["segment"] = "B"
                    controlled_report["summary"]["rank_ic"].update(mean=-0.2, p_value=p_value)
                    controlled_report["summary"]["directional_spread"].update(mean=0.003, p_value=p_value)
                    controlled_report["summary"].update(valid_stages=2, positive_stage_share=1.0)
                    with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", return_value=controlled_report):
                        result = miner.validate(lambda: b_panel, Path(directory) / "ideas")
                    decision = result["decisions"]["candidate-0001"]
                    self.assertEqual(decision["eligible_for_idea_pool"], should_admit)
                    self.assertEqual(decision["validation_status"], "passed" if should_admit else "not_passed")
                    self.assertEqual(decision["decision_source"], "program")
                    b_request = model.requests[-1]
                    self.assertEqual(b_request["role"], "evaluator")
                    facts = next(r["data"] for r in b_request["records"] if r["kind"] == "validation_result")
                    self.assertEqual(facts["validation_status"], decision["validation_status"])
                    self.assertEqual(facts["eligible_for_idea_pool"], should_admit)
                    b_report = json.loads((miner.root / "b_records/candidate-0001-report.json").read_text())
                    self.assertNotIn("decision", b_report["data"])
                    if should_admit:
                        card = json.loads(next((Path(directory) / "ideas").glob("*.json")).read_text())
                        self.assertEqual(card["source_type"], "factor_mining")
                        self.assertEqual(card["status"], "research_idea")
                        for key in ("source", "original_claim", "economic_mechanism", "market_and_horizon",
                                    "data_and_coverage", "falsification_conditions", "unverified_assumptions", "next_research_plan"):
                            self.assertIn(key, card)
                        self.assertNotIn("spot_close", card["original_claim"]["formula"]["expression"])
                        self.assertEqual(card["a_research_decision"]["disposition"], "retain")
                        self.assertEqual(card["b_validation_status"], "passed")
                        self.assertEqual(card["admission_evidence"]["decision_source"], "program")
                        self.assertEqual(card["strategy_validation_status"], "not_started")
                    else:
                        self.assertFalse((Path(directory) / "ideas").exists())
                    self.assertIn("B段程序验证结果", (miner.root / "B-report.md").read_text())
                    calls = len(model.requests)
                    self.assertEqual(miner.complete_reports("B", Path(directory) / "ideas"), result)
                    self.assertEqual(len(model.requests), calls)
                    if should_admit:
                        path = next((Path(directory) / "ideas").glob("*.json"))
                        self.assertEqual(json.loads(path.read_text()), card)
                        path.write_text(dumps({**card, "status": "changed"}))
                        with self.assertRaisesRegex(ValueError, "existing idea card differs"):
                            miner.complete_reports("B", Path(directory) / "ideas")

    def test_B_thresholds_and_engineering_exclusion_remain_enforced(self):
        cases = [
            ("research", {"rank_ic": {"mean": 0.2}}, "directional IC", "not_passed"),
            ("research", {"directional_spread": {"mean": 0.0}}, "directional return spread", "not_passed"),
            ("research", {"valid_stages": 1}, "stage repetition", "not_passed"),
            ("research", {"positive_stage_share": 0.0}, "stage repetition", "not_passed"),
            ("engineering_check", {}, "engineering checks", "passed"),
        ]
        for purpose, overrides, reason, status in cases:
            with self.subTest(reason=reason, overrides=overrides), tempfile.TemporaryDirectory() as directory:
                spec, model = specification(purpose=purpose), ScenarioModel(propose_once=False)
                miner = FactorMiner(spec, model, Path(directory))
                miner.explore(input_panel(spec))
                panel = input_panel(spec, "B")
                miner.freeze(["candidate-0001"], panel.universe)
                report = copy.deepcopy(miner.candidates["candidate-0001"]["evaluation"])
                report["segment"] = "B"
                report["summary"]["rank_ic"].update(mean=-0.2, p_value=0.000001)
                report["summary"]["directional_spread"].update(mean=0.003, p_value=0.000001)
                report["summary"].update(valid_stages=2, positive_stage_share=1.0)
                for key, value in overrides.items():
                    if isinstance(value, dict):
                        report["summary"][key].update(value)
                    else:
                        report["summary"][key] = value
                with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", return_value=report):
                    result = miner.validate(lambda: panel, Path(directory) / "ideas")
                decision = result["decisions"]["candidate-0001"]
                self.assertFalse(decision["eligible_for_idea_pool"])
                self.assertEqual(decision["validation_status"], status)
                self.assertTrue(any(reason in item for item in decision["reasons"]))
                self.assertFalse((Path(directory) / "ideas").exists())

    def test_optimizer_task_is_materialized_as_a_linked_formula_only_by_ideator(self):
        spec, model = specification(), ScenarioModel()
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            result = miner.explore(input_panel(spec))
            child = miner.candidates["candidate-0003"]
            self.assertEqual(child["definition"]["expression"],
                             "ts_mean(div(perp_close,ts_mean(perp_close,6)),3)")
            self.assertIn("candidate-0003", result["evaluated_ids"])
            proposal = miner.proposals["proposal-1"]
            self.assertNotIn("candidate", proposal)
            self.assertNotIn("expression", dumps(proposal))
            self.assertEqual(proposal["modification_task"], modification_task())
            self.assertEqual(proposal["experiment_design"], experiment_design())
            child_review = next(request for request in model.requests
                                if request["role"] == "calculator"
                                and request["payload"]["candidate_id"] == "candidate-0003")
            self.assertEqual(child_review["payload"]["modification_plan"], experiment())
            self.assertFalse(any(record["kind"] == "experiment_scope_error" for record in miner.store.all()))
            report = (miner.root / "A-report.md").read_text()
            for label in ("修改任务与配对检验合同", "原核心假设", "改动目标", "判断标准"):
                self.assertIn(label, report)

    def test_old_optimizer_candidate_field_is_excluded_before_ideator(self):
        leaked = definition("div(perp_close,ts_mean(perp_close,7))", "candidate-0001", "proposal-1")
        def add_old_candidate(request, result):
            if request["role"] == "optimizer" and result["proposals"]:
                result["proposals"][0]["candidate"] = leaked
        spec, model = specification(), ScenarioModel(add_old_candidate)
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            miner.explore(input_panel(spec))
            second = [request for request in model.requests if request["role"] == "ideator"][1]
            pending = second["payload"]["pending_proposals"]["proposal-1"]
            self.assertNotIn("candidate", pending)
            self.assertNotIn(leaked["expression"], dumps(pending))
            self.assertNotEqual(miner.candidates["candidate-0003"]["definition"]["expression"], leaked["expression"])
            note = next(record for record in miner.store.all()
                        if record["kind"] == "format_deviation" and record["data"]["role"] == "optimizer")
            self.assertIn("/result/proposals/0/candidate", note["data"]["extra_fields"])

    def test_predeclared_improvement_stops_continues_or_pauses(self):
        spec = specification()
        rng = np.random.default_rng(55)
        grid = pd.date_range("2026-01-01", periods=300, freq="h", tz="UTC")
        base_values = rng.normal(0, 0.1, 300)
        def report(values):
            return {"segment": "A", "direction": 1, "periods": [
                {"timestamp": t.isoformat(), "rank_ic": float(x), "directional_spread": float(x / 100)}
                for t, x in zip(grid, values)]}
        base = report(base_values)
        small_noise = rng.normal(0, .001, 300)
        plans = experiment()
        supported = compare_experiment(report(base_values + .03 + small_noise), base, plans, spec)
        stopped = compare_experiment(report(base_values - .03 + small_noise), base, plans, spec)
        uncertain = compare_experiment(report(base_values + rng.normal(0, .5, 300)), base, plans, spec)
        self.assertEqual(supported["decision"], "continue")
        self.assertEqual(stopped["decision"], "stop")
        self.assertEqual(uncertain["decision"], "pause_insufficient")

    def test_missing_meaning_returns_without_repair(self):
        spec, model = specification(), Mock()
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            data = definition()
            data["meaning"] = ""
            miner.candidates["candidate-0001"] = {"definition": data}
            result = miner._calculate("candidate-0001", input_panel(spec))
            self.assertIsNone(result)
            model.complete.assert_not_called()
            self.assertEqual(miner.store.all()[0]["data"]["status"], "returned_to_ideator")

    def test_frozen_membership_cannot_be_changed_before_B(self):
        spec, model = specification(), ScenarioModel(propose_once=False)
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            miner.explore(input_panel(spec))
            planned = input_panel(spec, "B")
            miner.freeze(["candidate-0001"], planned.universe)
            changed = copy.deepcopy(planned)
            changed.universe.iloc[20] = False
            with self.assertRaisesRegex(ValueError, "membership changed"):
                miner.validate(lambda: changed, Path(directory) / "ideas")
            self.assertTrue((miner.root / "b-access-started.json").exists())

    def test_model_calls_remain_recorded_without_a_total_limit(self):
        model = Mock()
        model.complete.return_value = ModelReply(dumps({"result": {}, "read_records": []}), {}, "fixture")
        with tempfile.TemporaryDirectory() as directory:
            gateway = AgentGateway(model, specification(), RecordStore(Path(directory) / "records"), Path(directory) / "calls")
            for number in range(40):
                gateway.ask("ideator" if number % 2 else "optimizer", "fixture", {}, {})
            self.assertEqual(model.complete.call_count, 40)
            self.assertEqual(len(list(gateway.transcripts.glob("*-request.json"))), 40)
            self.assertTrue((gateway.transcripts / "00040-response.json").exists())

    def test_two_round_context_repairs_failures_experiments_freeze_and_B_isolation(self):
        spec, model = specification(), ScenarioModel()
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            result = miner.explore(input_panel(spec))
            self.assertEqual(result["completed_rounds"], 2)
            self.assertEqual(result["evaluated_ids"], ["candidate-0001", "candidate-0003"])
            second = [r for r in model.requests if r["role"] == "ideator"][1]
            records = {r["id"]: r["data"] for r in second["records"]}
            calc = records["candidate-0001-calculation"]
            self.assertEqual(len(calc["checks"]), 2)
            self.assertIn("spot_close", calc["original_expression"])
            self.assertNotIn("spot_close", calc["executed_expression"]["expression"])
            self.assertEqual(records["candidate-0002-calculation"]["status"], "returned_to_ideator")
            self.assertTrue(records["candidate-0001-evaluation"]["periods"])
            self.assertIn("candidate-0001-report", records)
            self.assertIn("round-001-optimization", records)
            third = miner.candidates["candidate-0003"]
            self.assertEqual(third["experiment"], experiment())
            self.assertIn("paired_improvement", third["comparison"])
            before = digest(miner.store.all())
            miner.freeze(result["evaluated_ids"], input_panel(spec, "B").universe)
            restored = FactorMiner.open(miner.root, model)
            called = []
            def loader():
                self.assertTrue((miner.root / "b-access-started.json").exists())
                called.append(True)
                return input_panel(spec, "B")
            validation = restored.validate(loader, Path(directory) / "ideas")
            self.assertEqual(validation["batch_correction"]["family_size"], 4)
            self.assertEqual(digest(miner.store.all()), before)
            self.assertFalse((Path(directory) / "ideas").exists())
            for request in model.requests:
                if request["role"] in {"ideator", "optimizer", "calculator"}:
                    self.assertNotIn('"segment":"B"', dumps(request["records"]))
            with self.assertRaises(FileExistsError):
                restored.validate(loader, Path(directory) / "ideas")
            self.assertEqual(len(called), 1)
            with self.assertRaisesRegex(ValueError, "new research"):
                restored.explore(input_panel(spec))

    def test_invalid_model_response_is_saved_and_stops_after_two_corrections(self):
        model = Mock()
        model.complete.return_value = ModelReply("not json", {}, "broken")
        spec = specification()
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(spec, model, Path(directory))
            with self.assertRaisesRegex(ModelResponseError, "JSONDecodeError"):
                miner.explore(input_panel(spec))
            self.assertEqual(model.complete.call_count, 3)
            self.assertTrue((miner.root / "model_calls/00001-response.json").exists())
            self.assertTrue((miner.root / "exploration-stopped.json").exists())

    def test_inherited_model_failure_id_does_not_block_a_new_correction(self):
        spec = specification(run_id="successor-run")
        good = dumps({"result": {"ok": True}, "read_records": []})
        model = Mock()
        model.complete.side_effect = [ModelReply("not json", {}, "fixture"),
                                      ModelReply(good, {}, "fixture")]
        with tempfile.TemporaryDirectory() as directory:
            store = RecordStore(Path(directory) / "records")
            store.append("model-00001-invalid", "invalid_model_response", {"inherited": True})
            gateway = AgentGateway(model, spec, store, Path(directory) / "calls")

            self.assertEqual(gateway.ask("evaluator", "check", {}, {}), {"ok": True})

            failures = [record for record in store.all()
                        if record["kind"] == "invalid_model_response"]
            self.assertEqual(len(failures), 2)
            self.assertTrue(any(record["id"].startswith("model-00001-")
                                and record["id"].endswith("-invalid")
                                and record["id"] != "model-00001-invalid"
                                for record in failures))

    def test_research_parameters_fail_before_running(self):
        for kwargs in ({"hac_lags": 0}, {"sample_hours": 5},
                       {"groups": 4, "min_symbols": 6}, {"label": "next_bar"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                specification(**kwargs)

    def test_record_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RecordStore(Path(directory))
            store.append("record", "fact", {"number": 1})
            path = Path(directory) / "record.json"
            value = json.loads(path.read_text())
            value["data"]["number"] = 2
            path.write_text(dumps(value))
            with self.assertRaisesRegex(ValueError, "changed"):
                store.all()

    def test_context_pages_original_rows_only_when_budget_requires_it(self):
        spec = specification(context_tokens=7000, output_tokens=1000)
        with tempfile.TemporaryDirectory() as directory:
            store = RecordStore(Path(directory) / "records")
            store.append("evidence", "evaluation", {"summary": {"mean": .2}, "periods": [{"value": i} for i in range(1000)]})
            model = Mock()
            model.complete.side_effect = [
                ModelReply(dumps({"result": None, "read_records": [{"record_id": "evidence", "pointer": "/periods", "offset": 995, "limit": 5}]}), {}, "fixture"),
                ModelReply(dumps({"result": {"checked": True}, "read_records": []}), {}, "fixture")]
            gateway = AgentGateway(model, spec, store, Path(directory) / "calls")
            self.assertTrue(gateway.ask("evaluator", "核对原文", {}, {})["checked"])
            first = json.loads((Path(directory) / "calls/00001-request.json").read_text())
            self.assertEqual(first["context_mode"], "numeric_tables_paged")
            second_messages = model.complete.call_args_list[1].args[0]
            self.assertEqual(json.loads(second_messages[-1]["content"])["requested_original_evidence"][0]["data"]["items"][-1], {"value": 999})

    def test_context_pages_stage_rows_as_numeric_evidence(self):
        record = {"id": "evaluation", "kind": "evaluation", "data": {
            "summary": {"mean": .2},
            "stages": [{"stage": number, "rank_ic": number / 10} for number in range(20)],
        }}
        compacted = compact_record(record)
        self.assertEqual(compacted["data"]["stages"], {
            "record_id": "evaluation", "pointer": "/stages", "rows": 20,
            "read_records": "request an explicit offset and limit to read original rows",
        })

    def test_context_pages_numeric_rows_nested_in_goal_cycle_records(self):
        spec = specification(context_tokens=7000, output_tokens=1000)
        with tempfile.TemporaryDirectory() as directory:
            store = RecordStore(Path(directory) / "records")
            store.append("cycle-00000001", "research_cycle", {"records": [{
                "id": "candidate-0001-evaluation", "kind": "evaluation",
                "data": {"summary": {"mean": .2}, "periods": [{"value": i} for i in range(1000)]},
            }]})
            pointer = "/records/0/data/periods"
            model = Mock()
            model.complete.side_effect = [
                ModelReply(dumps({"result": None, "read_records": [{
                    "record_id": "cycle-00000001", "pointer": pointer, "offset": 998, "limit": 2,
                }]}), {}, "fixture"),
                ModelReply(dumps({"result": {"checked": True}, "read_records": []}), {}, "fixture"),
            ]
            gateway = AgentGateway(model, spec, store, Path(directory) / "calls")
            self.assertTrue(gateway.ask("optimizer", "核对Goal历史", {}, {})["checked"])
            first = json.loads((Path(directory) / "calls/00001-request.json").read_text())
            request = json.loads(first["messages"][1]["content"])
            paged = request["records"][0]["data"]["records"][0]["data"]["periods"]
            self.assertEqual(paged["record_id"], "cycle-00000001")
            self.assertEqual(paged["pointer"], pointer)
            last_messages = model.complete.call_args_list[-1].args[0]
            rows = json.loads(last_messages[-1]["content"])["requested_original_evidence"][0]["data"]
            self.assertEqual(rows["items"], [{"value": 998}, {"value": 999}])

    def test_context_uses_saved_goal_cycle_handoff_before_nested_raw_records(self):
        record = {"id": "cycle-00000001", "kind": "research_cycle", "data": {
            "run_id": "prior-run",
            "records": [{"id": "candidate-0001-evaluation", "kind": "evaluation",
                         "data": {"periods": [{"value": i} for i in range(1000)]}}],
            "context": {"source_record_id": "cycle-00000001", "run_id": "prior-run",
                        "candidates": [{"candidate_ref": "prior-run/candidate-0001"}],
                        "previous_expressions": []},
        }}
        compacted = compact_record(record)
        self.assertEqual(compacted["data"]["records"]["record_id"], "cycle-00000001")
        self.assertEqual(compacted["data"]["records"]["pointer"], "/records")
        self.assertEqual(compacted["data"]["records"]["rows"], 1)
        self.assertEqual(compacted["data"]["context"]["run_id"], "prior-run")

    def test_context_overflow_does_not_make_a_request(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Mock()
            spec = specification(context_tokens=1200, output_tokens=1000)
            gateway = AgentGateway(model, spec, RecordStore(Path(directory) / "records"), Path(directory) / "calls")
            with self.assertRaisesRegex(ValueError, "context budget"):
                gateway.ask("ideator", "work", {}, {})
            model.complete.assert_not_called()

    def test_B_gateway_refuses_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Mock()
            gateway = AgentGateway(model, specification(), RecordStore(Path(directory) / "records"), Path(directory) / "calls", stage="B")
            with self.assertRaisesRegex(ValueError, "cannot feed"):
                gateway.ask("optimizer", "work", {}, {})
            model.complete.assert_not_called()


class GoAdapterTests(unittest.TestCase):
    @patch.dict("os.environ", {"TEST_GO_KEY": "fixture-secret"})
    @patch("crypto_quant.research.factor_mining.model.requests.post")
    def test_provider_default_output_limit_is_omitted_from_actual_payload(self, post):
        example = Path(__file__).resolve().parents[1] / "examples/factor_mining/contract.example.json"
        spec = ResearchSpec.from_dict(json.loads(example.read_text()))
        self.assertIsNone(spec.output_tokens)
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {"choices": [{"finish_reason": "stop", "message": {
            "content": dumps({"result": {"ok": True}, "read_records": []})}}], "usage": {"completion_tokens": 20000}}
        model = OpenCodeGoModel("deepseek-v4.1-flash", "chat", api_key_env="TEST_GO_KEY", timeout_seconds=30,
                                reasoning_effort="low")
        with tempfile.TemporaryDirectory() as directory:
            gateway = AgentGateway(model, spec, RecordStore(Path(directory) / "records"), Path(directory) / "calls")
            self.assertTrue(gateway.ask("calculator", "fixture", {}, {})["ok"])
            payload = post.call_args.kwargs["json"]
            self.assertNotIn("max_tokens", payload)
            self.assertNotIn("max_completion_tokens", payload)
            self.assertEqual(payload["reasoning_effort"], "low")
            saved = json.loads((Path(directory) / "calls/00001-request.json").read_text())
            self.assertIsNone(saved["output_tokens"])
        for invalid in (0, -1, True, "16384"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "output_tokens"):
                specification(output_tokens=invalid)

    @patch.dict("os.environ", {"TEST_GO_KEY": "fixture-secret"})
    @patch("crypto_quant.research.factor_mining.model.requests.post")
    def test_explicit_deepseek_effort_reaches_request_and_trace(self, post):
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {}}
        model = OpenCodeGoModel("deepseek-v4.1-flash", "chat", api_key_env="TEST_GO_KEY", timeout_seconds=30, reasoning_effort="low")
        reply = model.complete([], max_output_tokens=16384, session_id="fixture")
        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "low")
        self.assertEqual(reply.requested_reasoning_effort, "low")
        with self.assertRaises(ValueError):
            OpenCodeGoModel("minimax-m2.7", "messages", api_key_env="TEST_GO_KEY", timeout_seconds=30, reasoning_effort="low")

    @patch.dict("os.environ", {}, clear=True)
    def test_local_env_reads_literal_key_without_exporting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('# Local setting\nOPENCODE_GO_API_KEY="fixture-$literal#part=="\n')
            self.assertEqual(read_api_key("OPENCODE_GO_API_KEY", path), "fixture-$literal#part==")
            import os
            self.assertNotIn("OPENCODE_GO_API_KEY", os.environ)

    @patch.dict("os.environ", {"TEST_GO_KEY": "environment-fixture"})
    def test_existing_environment_does_not_read_local_file(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("should not read file")):
            self.assertEqual(read_api_key("TEST_GO_KEY", Path(".env")), "environment-fixture")

    @patch.dict("os.environ", {}, clear=True)
    def test_empty_duplicate_or_bad_quoted_keys_fail_without_exposing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            for content in ('OPENCODE_GO_API_KEY=\n',
                            'OPENCODE_GO_API_KEY=fixture-secret\nOPENCODE_GO_API_KEY=duplicate\n',
                            'OPENCODE_GO_API_KEY="fixture-secret\n'):
                with self.subTest(case=content.count("\n")):
                    path.write_text(content)
                    with self.assertRaises(ValueError) as caught:
                        read_api_key("OPENCODE_GO_API_KEY", path)
                    self.assertNotIn("fixture-secret", str(caught.exception))

    @patch.dict("os.environ", {"TEST_GO_KEY": "fixture-secret"})
    @patch("crypto_quant.research.factor_mining.model.requests.post")
    def test_chat_and_messages_payloads(self, post):
        messages = [{"role": "system", "content": "JSON"}, {"role": "user", "content": "task"}]
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {"total_tokens": 10}}
        model = OpenCodeGoModel("kimi-k2.6", "chat", api_key_env="TEST_GO_KEY", timeout_seconds=30)
        self.assertEqual(model.complete(messages, max_output_tokens=100, session_id="session").text, "{}")
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["x-opencode-session"], "session")
        self.assertEqual(kwargs["json"]["messages"], messages)
        self.assertFalse(kwargs["allow_redirects"])
        post.return_value.json.return_value = {"content": [{"type": "thinking", "thinking": "internal"}, {"type": "text", "text": "{}"}],
                                               "stop_reason": "end_turn", "usage": {"output_tokens": 3}}
        model = OpenCodeGoModel("minimax-m2.7", "messages", api_key_env="TEST_GO_KEY", timeout_seconds=30)
        self.assertEqual(model.complete(messages, max_output_tokens=100, session_id="session").text, "{}")
        self.assertEqual(post.call_args.kwargs["json"]["system"], "JSON")
        self.assertEqual(post.call_args.kwargs["json"]["messages"], messages[1:])

    @patch.dict("os.environ", {"TEST_GO_KEY": "fixture-secret"})
    @patch("crypto_quant.research.factor_mining.model.requests.post")
    def test_transport_identifies_retryable_http_failure(self, post):
        post.return_value = Mock(status_code=429, text="rate limited")
        model = OpenCodeGoModel("kimi-k2.6", "chat", api_key_env="TEST_GO_KEY", timeout_seconds=30)
        with self.assertRaisesRegex(ApiCallError, "429") as caught:
            model.complete([], max_output_tokens=100, session_id="fixture")
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(post.call_count, 1)


@patch.dict("os.environ", {"TEST_GO_KEY": "fixture-secret"})
@patch("crypto_quant.research.factor_mining.records.time.sleep")
class ApiRetryTests(unittest.TestCase):
    def gateway(self, directory, *, protocol="chat"):
        model = OpenCodeGoModel("fixture-model", protocol, api_key_env="TEST_GO_KEY", timeout_seconds=30)
        return AgentGateway(model, specification(), RecordStore(Path(directory) / "records"),
                            Path(directory) / "calls")

    def response(self, content, *, protocol="chat", status=200):
        response = Mock(status_code=status, text="temporary service error")
        usage = {"completion_tokens": 100, "completion_tokens_details": {"reasoning_tokens": 90}}
        if protocol == "chat":
            response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {
                "content": content, "reasoning_content": "recorded fixture reasoning"}}], "usage": usage,
                "echoed_key": "fixture-secret"}
        else:
            response.json.return_value = {"content": [{"type": "text", "text": content}],
                                          "stop_reason": "end_turn", "usage": usage}
        return response

    def test_empty_response_saved_before_retry_on_both_protocols(self, sleep):
        for protocol in ("chat", "messages"):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as directory, patch(
                    "crypto_quant.research.factor_mining.model.requests.post") as post:
                empty = self.response(None if protocol == "chat" else "", protocol=protocol)
                good = self.response(dumps({"result": {"ok": True}, "read_records": []}), protocol=protocol)
                post.side_effect = [empty, good]
                gateway = self.gateway(directory, protocol=protocol)
                self.assertTrue(gateway.ask("optimizer", "fixture", {}, {})["ok"])
                first = json.loads((Path(directory) / "calls/00001-response.json").read_text())
                self.assertEqual(first["usage"]["completion_tokens_details"]["reasoning_tokens"], 90)
                self.assertIsNotNone(first["raw_response"])
                error = json.loads((Path(directory) / "calls/00001-error.json").read_text())
                self.assertTrue(error["will_retry"])
                retry = json.loads((Path(directory) / "calls/00002-request.json").read_text())
                self.assertEqual(retry["retry_of"], "00001")
                self.assertEqual(retry["attempt"], 2)
                self.assertEqual(post.call_args_list[0].kwargs, post.call_args_list[1].kwargs)
                for path in (Path(directory) / "calls").glob("*.json"):
                    self.assertNotIn("fixture-secret", path.read_text())

    def test_all_five_retries_can_recover_without_changing_request(self, sleep):
        with tempfile.TemporaryDirectory() as directory, patch(
                "crypto_quant.research.factor_mining.model.requests.post") as post:
            post.side_effect = [requests.Timeout("fixture-secret"), requests.ConnectionError("fixture-secret"),
                                self.response("", status=429), self.response("", status=503), self.response(""),
                                self.response(dumps({"result": {"ok": True}, "read_records": []}))]
            self.assertTrue(self.gateway(directory).ask("ideator", "fixture", {}, {})["ok"])
            self.assertEqual(post.call_count, 6)
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2, 4, 8, 16])
            self.assertTrue(all(c.kwargs == post.call_args_list[0].kwargs for c in post.call_args_list))
            self.assertEqual(len(list((Path(directory) / "calls").glob("*-error.json"))), 5)
            for path in (Path(directory) / "calls").glob("*.json"):
                self.assertNotIn("fixture-secret", path.read_text())

    def test_retry_cap_remains_after_many_prior_calls(self, sleep):
        with tempfile.TemporaryDirectory() as directory, patch(
                "crypto_quant.research.factor_mining.model.requests.post") as post:
            post.return_value = self.response(dumps({"result": {}, "read_records": []}))
            gateway = self.gateway(directory)
            for _ in range(35):
                gateway.ask("optimizer", "fixture", {}, {})
            post.return_value = self.response(" ")
            with self.assertRaisesRegex(ApiCallError, "empty"):
                gateway.ask("optimizer", "fixture", {}, {})
            self.assertEqual(post.call_count, 41)
            last = json.loads((Path(directory) / "calls/00041-error.json").read_text())
            self.assertFalse(last["will_retry"])
            self.assertEqual(last["stop_reason"], "retries_exhausted")
            self.assertEqual(sleep.call_count, 5)

    def test_permanent_http_and_content_filter_do_not_retry(self, sleep):
        for status in (400, 401, 403, 404):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory, patch(
                    "crypto_quant.research.factor_mining.model.requests.post") as post:
                post.return_value = self.response("", status=status)
                post.return_value.text = "error echo fixture-secret"
                with self.assertRaisesRegex(ApiCallError, str(status)):
                    self.gateway(directory).ask("optimizer", "fixture", {}, {})
                self.assertEqual(post.call_count, 1)
                error = (Path(directory) / "calls/00001-error.json").read_text()
                self.assertNotIn("fixture-secret", error)
                self.assertEqual(json.loads(error)["stop_reason"], "permanent_error")
        with tempfile.TemporaryDirectory() as directory, patch(
                "crypto_quant.research.factor_mining.model.requests.post") as post:
            response = self.response("")
            response.json.return_value["choices"][0]["finish_reason"] = "content_filter"
            post.return_value = response
            with self.assertRaises(ApiCallError):
                self.gateway(directory).ask("optimizer", "fixture", {}, {})
            self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_malformed_provider_response_is_recorded_and_retried(self, sleep):
        for invalid_json in (True, False):
            with self.subTest(invalid_json=invalid_json), tempfile.TemporaryDirectory() as directory, patch(
                    "crypto_quant.research.factor_mining.model.requests.post") as post:
                bad = self.response("")
                if invalid_json:
                    bad.json.side_effect = ValueError("invalid JSON")
                else:
                    bad.json.return_value = {"choices": []}
                post.side_effect = [bad, self.response(dumps({"result": {}, "read_records": []}))]
                self.assertEqual(self.gateway(directory).ask("optimizer", "fixture", {}, {}), {})
                self.assertEqual(post.call_count, 2)
                self.assertTrue((Path(directory) / "calls/00001-error.json").exists())

    def test_retry_resumes_current_agent_without_restarting_research(self, sleep):
        scripted = ScenarioModel()
        class EmptyOnce:
            def __init__(self):
                self.failed_stages = set()
            def complete(self, messages, **kwargs):
                stage = "B" if "当前仅解释冻结候选的B段" in messages[0]["content"] else "A"
                if json.loads(messages[1]["content"])["role"] == "evaluator" and stage not in self.failed_stages:
                    self.failed_stages.add(stage)
                    return ModelReply("", {"completion_tokens": 100}, "empty-fixture")
                return scripted.complete(messages, **kwargs)
        spec = specification()
        with tempfile.TemporaryDirectory() as directory:
            scripted.propose_once = False
            miner = FactorMiner(spec, EmptyOnce(), Path(directory))
            result = miner.explore(input_panel(spec))
            self.assertEqual(result["completed_rounds"], 1)
            self.assertEqual(scripted.ideations, 1)
            self.assertEqual(scripted.optimizations, 1)
            self.assertEqual(len(list((miner.root / "a_records").glob("*-evaluation.json"))), 1)
            self.assertEqual(len(list((miner.root / "model_calls").glob("*-error.json"))), 1)
            panel = input_panel(spec, "B")
            miner.freeze(result["retained_ids"], panel.universe)
            loader = Mock(return_value=panel)
            miner.validate(loader, Path(directory) / "ideas")
            loader.assert_called_once_with()
            self.assertEqual(len(list((miner.root / "b_records").glob("*-evaluation.json"))), 1)
            self.assertEqual(len(list((miner.root / "model_calls").glob("*-error.json"))), 2)

class ResponseRecoveryTests(unittest.TestCase):
    def test_ideator_prompt_requires_statistical_only_oi_conditioning(self):
        with tempfile.TemporaryDirectory() as directory:
            model = ScenarioModel()
            miner = FactorMiner(specification(), model, Path(directory))
            miner.explore(input_panel(miner.spec))

            task = next(request["task"] for request in model.requests
                        if request["role"] == "ideator")
            self.assertIn("可证伪统计条件关联", task)
            self.assertIn("无法写成纯统计假设时必须不生成该候选", task)
            self.assertIn("纠错时必须删除全部越界机制", task)

    def test_ideator_rejects_reversed_shifted_correlation_orientation(self):
        rejected = []
        def transform(request, result):
            if request["role"] == "ideator" and not rejected:
                rejected.append(True)
                result["candidates"] = [definition(
                    "cross_rank(ts_corr(spot_log_return_1bar,ts_delay(perp_log_return_1bar,1),24))",
                    meaning="现货当期收益与永续上一小时收益的相关表示现货对下一小时永续的领先")]
                result["candidates"][0].update(
                    name="spot_lead_perp_lag_corr",
                    hypothesis="现货领先永续的强度可能预测未来收益",
                )
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(specification(), ScenarioModel(transform), Path(directory))
            result = miner.explore(input_panel(miner.spec))
            self.assertEqual(result["completed_rounds"], 2)
            failures = [r for r in miner.store.all() if r["kind"] == "invalid_model_response"]
            self.assertEqual(len(failures), 1)
            self.assertIn("corr(current x, delay(y, n)) means y leads x", failures[0]["data"]["error"])

    def test_ideator_accepts_correct_shifted_correlation_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(specification(), ScenarioModel(), Path(directory))
            value = definition(
                "cross_rank(ts_corr(spot_log_return_1bar,ts_delay(perp_log_return_1bar,1),24))",
                meaning="现货当期收益与永续上一小时收益的相关，即永续领先现货")
            value.update(name="perp_lead_spot_lag_corr", hypothesis="永续领先现货的强度可能预测未来收益")
            definitions, _, _, _ = miner._check_ideation(
                {"candidates": [value], "dispositions": [], "analysis": "测试时间对齐"}, {})
            self.assertEqual(definitions[0]["name"], "perp_lead_spot_lag_corr")

    def test_ideator_rejects_misstated_maximum_lookback(self):
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(specification(max_lookback_hours=168), ScenarioModel(), Path(directory))
            expression = "cross_rank(ts_corr(perp_log_return_1bar,ts_delay(spot_log_return_1bar,1),24))"
            value = definition(expression, meaning="24小时窗口与1小时滞后；最大回看24小时")
            value.update(name="spot_lead_perp_lag_corr", hypothesis="现货领先永续的强度可能预测未来收益")
            response = {"candidates": [value], "dispositions": [], "analysis": "测试回看口径"}
            with self.assertRaisesRegex(ValueError, "claimed=\\[24\\], actual=25"):
                miner._check_ideation(response, {})
            value["meaning"] = "24小时窗口与1小时滞后；最大回看25小时"
            definitions, _, _, _ = miner._check_ideation(response, {})
            self.assertEqual(definitions[0]["meaning"], value["meaning"])

            # Test precalculated feature lookback: funding_7d_sum accepts both 0 and 168 hours
            funding_expr = "mul(cross_rank(basis_trade_spot), neg(cross_zscore(funding_7d_sum)))"
            funding_val = definition(funding_expr, meaning="永续基差与7天资金费率状态；最大回看168小时")
            funding_val.update(name="basis_funding_state", hypothesis="基差与7天资金费率交互可能预测未来收益")
            res_funding = {"candidates": [funding_val], "dispositions": [], "analysis": "测试预计算特征回看"}
            defs, _, _, _ = miner._check_ideation(res_funding, {})
            self.assertEqual(defs[0]["name"], "basis_funding_state")

            funding_val["meaning"] = "永续基差与7天资金费率状态；最大回看0小时"
            defs, _, _, _ = miner._check_ideation(res_funding, {})
            self.assertEqual(defs[0]["name"], "basis_funding_state")

            funding_val["meaning"] = "永续基差与7天资金费率状态；最大回看50小时"
            with self.assertRaisesRegex(ValueError, "claimed=\\[50\\], actual=0"):
                miner._check_ideation(res_funding, {})

    def test_evaluator_and_optimizer_prompts_do_not_invite_oi_denial_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            model = ScenarioModel()
            miner = FactorMiner(specification(), model, Path(directory))
            miner.explore(input_panel(miner.spec))

            tasks = {request["role"]: request["task"] for request in model.requests
                     if request["role"] in {"evaluator", "optimizer"}}
            for role in ("evaluator", "optimizer"):
                self.assertIn("聚合OI数量/名义价值变化的统计条件关联", tasks[role])
                self.assertIn("不列举被排除的解释，即便是否定句", tasks[role])

    def test_each_role_corrects_once_before_research_state_is_committed(self):
        mutations = {
            "ideator": lambda r: r["candidates"][0].update(direction="-1"),
            "calculator": lambda r: r.update(candidate_id="candidate-9999"),
            "evaluator": lambda r: r.update(limitations=[]),
            "optimizer": lambda r: r["decisions"][0].update(evidence_refs=["unknown-record"]),
        }
        for role, mutate in mutations.items():
            failed = []
            def transform(request, result):
                if request["role"] == role and not failed:
                    failed.append(True)
                    mutate(result)
                    result["untrusted_extra"] = "DO-NOT-FORWARD-EXTRA"
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                model = ScenarioModel(transform)
                miner = FactorMiner(specification(), model, Path(directory))
                result = miner.explore(input_panel(miner.spec))
                self.assertEqual(result["completed_rounds"], 2)
                self.assertEqual(len(result["candidate_ids"]), 3)
                self.assertEqual((model.ideations, model.optimizations), (2, 2))
                self.assertEqual(miner.routes["window-route"]["attempts"], 1)
                failures = [r for r in miner.store.all() if r["kind"] == "invalid_model_response"]
                self.assertEqual(len(failures), 1)
                requests = [json.loads(p.read_text()) for p in (miner.root / "model_calls").glob("*-request.json")]
                self.assertTrue(any("response_error" in dumps(r["messages"]) for r in requests))
                self.assertTrue(all("DO-NOT-FORWARD-EXTRA" not in dumps(r["messages"]) for r in requests))

    def test_rejected_optimizer_proposals_do_not_reserve_ids_or_routes(self):
        rejected = []
        def transform(request, result):
            if request["role"] == "optimizer" and result["proposals"] and not rejected:
                rejected.append(True)
                result["proposals"].append(copy.deepcopy(result["proposals"][0]))
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(specification(), ScenarioModel(transform), Path(directory))
            result = miner.explore(input_panel(miner.spec))
            self.assertEqual(result["completed_rounds"], 2)
            self.assertEqual(set(miner.proposals), {"proposal-1"})
            self.assertEqual(miner.routes["window-route"]["attempts"], 1)
            failures = [r for r in miner.store.all() if r["kind"] == "invalid_model_response"]
            self.assertEqual(len(failures), 1)
            self.assertIn("proposal IDs must be unique", failures[0]["data"]["error"])

    def test_rejected_ideation_does_not_consume_route_attempt(self):
        rejected = []
        def transform(request, result):
            if request["role"] == "ideator" and request["payload"]["pending_proposals"] and not rejected:
                rejected.append(True)
                result["candidates"].append(definition(parent="unknown-parent"))
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(specification(), ScenarioModel(transform), Path(directory))
            result = miner.explore(input_panel(miner.spec))
            self.assertEqual(result["completed_rounds"], 2)
            self.assertEqual(miner.routes["window-route"]["attempts"], 1)
            self.assertEqual(len(miner.candidates), 3)

    def test_optimizer_cannot_request_unsupported_residualization(self):
        rejected = []
        def transform(request, result):
            if request["role"] == "optimizer" and result["proposals"] and not rejected:
                rejected.append(True)
                result["proposals"][0]["modification_task"]["change_target"] = "对候选做截面正交化"
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(specification(), ScenarioModel(transform), Path(directory))
            result = miner.explore(input_panel(miner.spec))
            self.assertEqual(result["completed_rounds"], 2)
            failures = [r for r in miner.store.all() if r["kind"] == "invalid_model_response"]
            self.assertEqual(len(failures), 1)
            self.assertIn("no regression/residualization operator", failures[0]["data"]["error"])

    def test_optimizer_cannot_infer_leverage_change_from_open_interest(self):
        rejected = []
        def transform(request, result):
            if request["role"] == "optimizer" and result["proposals"] and not rejected:
                rejected.append(True)
                result["proposals"][0]["modification_task"].update(
                    modification_hypothesis="open_interest_base 上升说明大户加杠杆",
                    change_target="用 OI 变化识别新杠杆头寸",
                )
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(specification(), ScenarioModel(transform), Path(directory))
            result = miner.explore(input_panel(miner.spec))
            self.assertEqual(result["completed_rounds"], 2)
            failures = [r for r in miner.store.all() if r["kind"] == "invalid_model_response"]
            self.assertEqual(len(failures), 1)
            self.assertIn("does not identify leverage ratios", failures[0]["data"]["error"])

    def test_ideator_cannot_label_fixed_arithmetic_as_residualization(self):
        rejected = []
        def transform(request, result):
            if request["role"] == "ideator" and request["payload"]["pending_proposals"] and not rejected:
                rejected.append(True)
                result["candidates"][0]["name"] = "orthogonalized_price_deviation"
                result["candidates"][0]["change_reason"] = "用固定系数的 rank 加减实现正交化"
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(specification(), ScenarioModel(transform), Path(directory))
            result = miner.explore(input_panel(miner.spec))
            self.assertEqual(result["completed_rounds"], 2)
            failures = [r for r in miner.store.all() if r["kind"] == "invalid_model_response"]
            self.assertEqual(len(failures), 1)
            self.assertIn("no regression/residualization operator", failures[0]["data"]["error"])

    def test_evaluator_cannot_claim_unsupported_residualization_in_any_report_field(self):
        for field in ("analysis", "mechanism", "conditions", "falsifiers", "limitations", "next_steps"):
            rejected = []
            def transform(request, result):
                if request["role"] == "evaluator" and not rejected:
                    rejected.append(True)
                    claim = "对控制变量做正交化并报告残差 rank_ic"
                    if isinstance(result[field], list):
                        result[field].append(claim)
                    else:
                        result[field] += claim
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                miner = FactorMiner(specification(), ScenarioModel(transform), Path(directory))
                result = miner.explore(input_panel(miner.spec))
                self.assertEqual(result["completed_rounds"], 2)
                failures = [r for r in miner.store.all() if r["kind"] == "invalid_model_response"]
                self.assertEqual(len(failures), 1)
                self.assertIn("no regression/residualization operator", failures[0]["data"]["error"])

    def test_evaluator_can_record_unavailable_residualization_as_a_limitation(self):
        noted = []
        def transform(request, result):
            if request["role"] == "evaluator" and not noted:
                noted.append(True)
                result["limitations"].append(
                    "当前表达式白名单没有回归或残差算子，未执行正交化或残差化。")
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(specification(), ScenarioModel(transform), Path(directory))
            result = miner.explore(input_panel(miner.spec))
            self.assertEqual(result["completed_rounds"], 2)
            failures = [r for r in miner.store.all() if r["kind"] == "invalid_model_response"]
            self.assertEqual(failures, [])

    def test_evaluator_cannot_infer_leverage_change_from_open_interest(self):
        rejected = []
        def transform(request, result):
            if request["role"] == "evaluator" and not rejected:
                rejected.append(True)
                result["mechanism"] = "open_interest_base 上升说明大户继续加杠杆"
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(specification(), ScenarioModel(transform), Path(directory))
            result = miner.explore(input_panel(miner.spec))
            self.assertEqual(result["completed_rounds"], 2)
            failures = [r for r in miner.store.all() if r["kind"] == "invalid_model_response"]
            self.assertEqual(len(failures), 1)
            self.assertIn("does not identify leverage ratios", failures[0]["data"]["error"])

    def test_evaluator_cannot_infer_position_building_from_open_interest(self):
        rejected = []
        def transform(request, result):
            if request["role"] == "evaluator" and not rejected:
                rejected.append(True)
                result["mechanism"] = "open_interest_base 上升表示方向性头寸建立阶段"
        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(specification(), ScenarioModel(transform), Path(directory))
            result = miner.explore(input_panel(miner.spec))
            self.assertEqual(result["completed_rounds"], 2)
            failures = [r for r in miner.store.all() if r["kind"] == "invalid_model_response"]
            self.assertEqual(len(failures), 1)
            self.assertIn("open/close and long/short position direction", failures[0]["data"]["error"])

    def test_truncated_and_fenced_json_are_corrected_without_accepting_invalid_output(self):
        good = dumps({"result": {"ok": True}, "read_records": []})
        for text, finish in ((good, "length"), (good[:15], "max_tokens"), (f"```json\n{good}\n```", "stop")):
            with self.subTest(finish=finish), tempfile.TemporaryDirectory() as directory:
                model = Mock()
                model.complete.side_effect = [ModelReply(text, {}, "fixture", finish), ModelReply(good, {}, "fixture")]
                store = RecordStore(Path(directory) / "records")
                gateway = AgentGateway(model, specification(), store, Path(directory) / "calls")
                self.assertEqual(gateway.ask("evaluator", "check", {}, {}), {"ok": True})
                self.assertEqual(model.complete.call_count, 2)
                self.assertEqual(len([r for r in store.all() if r["kind"] == "invalid_model_response"]), 1)

    def test_bad_evidence_request_can_be_corrected_and_read(self):
        for record_id, pointer in (("missing", "/values"), ("facts", "/missing"), ("facts", "/values/99")):
            with self.subTest(record_id=record_id, pointer=pointer), tempfile.TemporaryDirectory() as directory:
                store = RecordStore(Path(directory) / "records")
                store.append("facts", "fact", {"values": [123]})
                def read(rid, ptr):
                    return ModelReply(dumps({"result": None, "read_records": [
                        {"record_id": rid, "pointer": ptr, "offset": 0, "limit": 1}]}), {}, "fixture")
                model = Mock()
                model.complete.side_effect = [read(record_id, pointer), read("facts", "/values"),
                                             ModelReply(dumps({"result": {}, "read_records": []}), {}, "fixture")]
                gateway = AgentGateway(model, specification(), store, Path(directory) / "calls")
                self.assertEqual(gateway.ask("evaluator", "read facts", {}, {}), {})
                last_messages = model.complete.call_args_list[-1].args[0]
                self.assertEqual(json.loads(last_messages[-1]["content"])["requested_original_evidence"][0]["data"]["items"], [123])

    def test_evidence_corruption_is_fatal_during_correction_or_read(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RecordStore(Path(directory) / "records")
            store.append("facts", "fact", {"value": 1})
            def corrupt(*args, **kwargs):
                path = store.root / "facts.json"
                record = json.loads(path.read_text()); record["data"]["value"] = 2
                path.write_text(dumps(record))
                return ModelReply(dumps({"result": None, "read_records": [
                    {"record_id": "facts", "pointer": "", "offset": 0, "limit": 1}]}), {}, "fixture")
            model = Mock(); model.complete.side_effect = corrupt
            gateway = AgentGateway(model, specification(), store, Path(directory) / "calls")
            with self.assertRaises(EvidenceIntegrityError):
                gateway.ask("evaluator", "read", {}, {})
            self.assertEqual(model.complete.call_count, 1)
            self.assertFalse(list(store.root.glob("*-invalid.json")))

    def test_malformed_saved_record_is_fatal_inside_model_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RecordStore(Path(directory) / "records")
            store.append("facts", "fact", {"value": 1})
            def corrupt(*args, **kwargs):
                (store.root / "facts.json").write_text("{")
                return ModelReply('{"result":{},"read_records":[]}', {}, "fixture")
            model = Mock(); model.complete.side_effect = corrupt
            gateway = AgentGateway(model, specification(), store, Path(directory) / "calls")
            with self.assertRaises(EvidenceIntegrityError):
                gateway.ask("optimizer", "check", {}, {}, validate=lambda result: store.all())
            self.assertEqual(model.complete.call_count, 1)
            self.assertFalse(list(store.root.glob("*-invalid.json")))

    @patch("crypto_quant.research.factor_mining.records.time.sleep")
    def test_exhausted_temporary_api_failure_is_local_to_one_report(self, sleep):
        def two_candidates(request, result):
            if request["role"] == "ideator":
                result["candidates"] = [definition(f"div(perp_close,ts_mean(perp_close,{window}))") for window in (3, 2)]
        scripted = ScenarioModel(two_candidates)
        empty_calls = []
        class EmptyFirstReport:
            def complete(self, messages, **kwargs):
                request = json.loads(messages[1]["content"])
                if request["role"] == "evaluator" and request["payload"]["candidate_id"] == "candidate-0001":
                    empty_calls.append(request)
                    return ModelReply("", {}, "empty-fixture")
                return scripted.complete(messages, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            scripted.propose_once = False
            miner = FactorMiner(specification(), EmptyFirstReport(), Path(directory))
            result = miner.explore(input_panel(miner.spec))
            self.assertEqual(len(empty_calls), 6)
            self.assertEqual(sleep.call_count, 5)
            self.assertEqual(result["pending_report_ids"], ["candidate-0001"])
            self.assertTrue((miner.root / "a_records/candidate-0002-report.json").exists())

    def test_unexpected_validator_bug_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Mock(); model.complete.return_value = ModelReply('{"result":{},"read_records":[]}', {}, "fixture")
            gateway = AgentGateway(model, specification(), RecordStore(Path(directory) / "records"), Path(directory) / "calls")
            with self.assertRaises(KeyError):
                gateway.ask("ideator", "work", {}, {}, validate=lambda result: result["internal-invariant"])
            self.assertEqual(model.complete.call_count, 1)

    def test_each_task_keeps_two_corrections_without_a_total_call_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Mock(); model.complete.return_value = ModelReply("invalid", {}, "fixture")
            gateway = AgentGateway(model, specification(), RecordStore(Path(directory) / "records"), Path(directory) / "calls")
            for task in range(12):
                with self.assertRaises(ModelResponseError):
                    gateway.ask("ideator", f"work {task}", {}, {})
                self.assertEqual(model.complete.call_count, (task + 1) * 3)
                self.assertFalse(gateway.store.all()[-1]["data"]["will_correct"])
            self.assertEqual(model.complete.call_count, 36)

    def test_failed_calculator_or_report_does_not_block_second_candidate(self):
        for role in ("calculator", "evaluator"):
            def transform(request, result):
                if request["role"] == "ideator":
                    result["candidates"] = [definition(f"div(perp_close,ts_mean(perp_close,{window}))") for window in (3, 2)]
                if request["role"] == role and request["payload"]["candidate_id"] == "candidate-0001":
                    result.pop("candidate_id" if role == "calculator" else "limitations")
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                miner = FactorMiner(specification(), ScenarioModel(transform, propose_once=False), Path(directory))
                result = miner.explore(input_panel(miner.spec))
                self.assertIn("candidate-0002", result["evaluated_ids"])
                self.assertTrue((miner.root / "a_records/candidate-0002-report.json").exists())
                self.assertFalse((miner.root / "exploration-stopped.json").exists())
                self.assertEqual(result["candidate_decisions"]["candidate-0001"]["disposition"],
                                 "pause" if role == "calculator" else "retain")

    def test_A_report_completion_does_not_recalculate_and_freeze_requires_report(self):
        def bad_report(request, result):
            if request["role"] == "evaluator": result.pop("limitations")
        with tempfile.TemporaryDirectory() as directory:
            model = ScenarioModel(bad_report)
            model.propose_once = False
            miner = FactorMiner(specification(), model, Path(directory))
            done = miner.explore(input_panel(miner.spec))
            panel_b = input_panel(miner.spec, "B")
            with self.assertRaisesRegex(ValueError, "complete.*report"):
                miner.freeze(done["retained_ids"], panel_b.universe)
            before = {r["id"]: r["sha256"] for r in miner.store.all()}
            model.transform = None
            restored = FactorMiner.open(miner.root, model)
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_expression") as compute:
                completed = restored.complete_reports("A", Path(directory) / "ideas")
            compute.assert_not_called()
            self.assertEqual(completed["status"], "complete")
            after = {r["id"]: r["sha256"] for r in restored.store.all()}
            self.assertTrue(all(after[key] == value for key, value in before.items()))
            count = len(model.requests)
            restored.complete_reports("A", Path(directory) / "ideas")
            self.assertEqual(len(model.requests), count)
            restored.freeze(done["retained_ids"], panel_b.universe)
            with self.assertRaisesRegex(ValueError, "after freezing"):
                restored.complete_reports("A", Path(directory) / "ideas")

    def prepare_B(self, directory):
        def two_candidates(request, result):
            if request["role"] == "ideator":
                result["candidates"] = [definition(f"div(perp_close,ts_mean(perp_close,{window}))") for window in (3, 2)]
        model = ScenarioModel(two_candidates)
        model.propose_once = False
        miner = FactorMiner(specification(), model, Path(directory))
        completed = miner.explore(input_panel(miner.spec))
        panel = input_panel(miner.spec, "B")
        miner.freeze(completed["retained_ids"], panel.universe)
        return miner, model, panel

    def test_B_saves_all_numerical_decisions_and_completes_only_missing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, model, panel = self.prepare_B(directory)
            def bad_first(request, result):
                if request["role"] == "evaluator" and request["payload"]["candidate_id"] == "candidate-0001":
                    result.pop("limitations")
            model.transform = bad_first
            loader = Mock(return_value=panel)
            result = miner.validate(loader, Path(directory) / "ideas")
            self.assertEqual(result["status"], "reports_pending")
            self.assertEqual(result["pending_report_ids"], ["candidate-0001"])
            self.assertEqual(len(list((miner.root / "b_records").glob("*-validation.json"))), 2)
            self.assertTrue((miner.root / "b_records/candidate-0002-report.json").exists())
            self.assertFalse((miner.root / "validation.json").exists())
            checkpoint = (miner.root / "b-numerical-complete.json").read_bytes()
            before = {p.name: p.read_bytes() for p in (miner.root / "b_records").glob("*.json")}
            model.transform = None
            restored = FactorMiner.open(miner.root, model)
            call_count = len(model.requests)
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_expression") as compute, patch(
                    "crypto_quant.research.factor_mining.workflow.evaluate_factor") as evaluate:
                completed = restored.complete_reports("B", Path(directory) / "ideas")
            compute.assert_not_called(); evaluate.assert_not_called(); loader.assert_called_once_with()
            self.assertEqual(len(model.requests), call_count + 1)
            self.assertEqual(completed["status"], "complete")
            self.assertEqual((miner.root / "b-numerical-complete.json").read_bytes(), checkpoint)
            self.assertTrue(all((miner.root / "b_records" / name).read_bytes() == value for name, value in before.items()))
            self.assertTrue((miner.root / "validation.json").exists())
            self.assertEqual(restored.complete_reports("B", Path(directory) / "ideas"), completed)
            self.assertEqual(len(model.requests), call_count + 1)
            with self.assertRaises(FileExistsError):
                restored.validate(loader, Path(directory) / "ideas")
            loader.assert_called_once_with()

    def test_B_can_complete_after_permanent_api_failure_without_reaccess(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, model, panel = self.prepare_B(directory)
            def fail(request, result):
                raise ApiCallError("authorization failed", retryable=False, diagnostics={})
            model.transform = fail
            loader = Mock(return_value=panel)
            with self.assertRaises(ApiCallError):
                miner.validate(loader, Path(directory) / "ideas")
            self.assertTrue((miner.root / "b-numerical-complete.json").exists())
            self.assertTrue((miner.root / "validation-stopped.json").exists())
            model.transform = None
            self.assertEqual(miner.complete_reports("B", Path(directory) / "ideas")["status"], "complete")
            loader.assert_called_once_with()

    def test_B_completion_refuses_changed_numerical_evidence_and_incomplete_batch(self):
        for change in ("record", "checkpoint", "incomplete"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                miner, model, panel = self.prepare_B(directory)
                if change == "incomplete":
                    with self.assertRaisesRegex(ValueError, "numerical validation"):
                        miner.complete_reports("B", Path(directory) / "ideas")
                    continue
                miner.validate(lambda: panel, Path(directory) / "ideas")
                if change == "record":
                    path = miner.root / "b_records/candidate-0001-validation.json"
                    value = json.loads(path.read_text())
                    value["data"]["eligible_for_idea_pool"] = True
                    value["sha256"] = digest(value["data"])  # Even a rehashed edit violates the numerical checkpoint.
                else:
                    path = miner.root / "b-numerical-complete.json"
                    value = json.loads(path.read_text()); value["b_records"] = {}
                path.write_text(dumps(value))
                calls = len(model.requests)
                with self.assertRaisesRegex(ValueError, "changed"):
                    miner.complete_reports("B", Path(directory) / "ideas")
                self.assertEqual(len(model.requests), calls)

    def test_report_completion_can_continue_after_many_prior_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            miner, model, panel = self.prepare_B(directory)
            def bad(request, result):
                if request["role"] == "evaluator": result.pop("limitations")
            model.transform = bad
            pending = miner.validate(lambda: panel, Path(directory) / "ideas")
            self.assertEqual(pending["status"], "reports_pending")
            previous = Mock()
            previous.complete.return_value = ModelReply('{"result":{},"read_records":[]}', {}, "fixture")
            gateway = AgentGateway(previous, miner.spec, RecordStore(miner.root / "b_records"),
                                   miner.root / "model_calls", stage="B")
            for _ in range(35):
                gateway.ask("evaluator", "prior call fixture", {}, {})
            before = len(model.requests)
            model.transform = None
            result = miner.complete_reports("B", Path(directory) / "ideas")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(model.requests), before + 2)
            self.assertTrue((miner.root / "validation.json").exists())

    def test_cli_report_completion_needs_no_database_or_universe(self):
        import argparse
        from crypto_quant.research.factor_mining.cli import add_arguments, execute
        parser = argparse.ArgumentParser(); add_arguments(parser)
        args = parser.parse_args(["complete-reports", "--run-dir", "saved-run", "--stage", "B",
                                  "--model", "fixture"])
        with patch("crypto_quant.research.factor_mining.cli.FactorMiner.open") as opened, patch(
                "crypto_quant.research.factor_mining.cli.load_stage") as loader:
            execute(args)
            opened.return_value.complete_reports.assert_called_once_with("B", Path("experiments/idea_pool"))
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
