"""Ideate -> semantically check/compute -> evaluate -> decide -> design an experiment.

Exploration and fixed-batch validation have separate record stores. The latter
cannot repair a formula or send feedback into the exploration loop.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from crypto_quant.features.factor_expressions import compile_expression, evaluate_expression
from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS, validate_universe
from .contracts import ResearchSpec, candidate, design, digest, identifier, require, text
from .evaluation import build_labels, compare_experiment, correct_batch, evaluate_factor, finite
from .model import ApiCallError, JsonModel
from .records import AgentGateway, ContextBudgetError, ModelResponseError, RecordStore, write_json
from .reporting import write_group_plot, write_research_report


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _text_schema(description: str) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "description": description}


def _array_schema(items: dict[str, Any], **limits: int) -> dict[str, Any]:
    return {"type": "array", "items": items, **limits}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


CANDIDATE_SCHEMA = _object_schema({
    **{name: _text_schema(description) for name, description in {
        "name": "因子名称", "expression": "白名单公式", "meaning": "字段、单位、窗口、顺序、当前期与截面范围",
        "hypothesis": "待检验的金融解释", "change_reason": "生成或修改理由"}.items()},
    "direction": {"type": "integer", "enum": [-1, 1], "description": "预先固定预测方向"},
    "parent_id": _nullable(_text_schema("原候选ID；无父候选时为null")),
    "proposal_id": _nullable(_text_schema("采用的优化建议ID；自主构想时为null")),
})
DESIGN_SCHEMA = _object_schema({
    **{name: _text_schema(description) for name, description in {
        "route_id": "新路线或已有继续路线的英文ID", "control_id": "已评估的原候选ID",
        "evidence": "引用原候选ID及具体指标、阶段和本地证据", "question": "此次只检验哪个原因",
        "change": "具体单项改动与预先限定的范围", "expected": "预期出现的变化",
        "stop_condition": "解释预定改善或IC损失区间上限不足时的停止条件",
        "pause_condition": "解释证据不精确、缺数据或预算用尽时的暂停条件"}.items()},
    "change_kind": {"type": "string", "enum": ["window", "smoothing", "field", "operator", "conditioning"],
                    "description": "还必须属于本次合同allowed_changes"},
    "metric": {"type": "string", "enum": ["rank_ic", "directional_spread"]},
    "min_improvement": {"type": "number", "exclusiveMinimum": 0, "description": "配对改善最低要求"},
    "max_ic_loss": {"type": "number", "minimum": 0, "description": "允许的有向IC损失"},
    "attempt_budget": {"type": "integer", "minimum": 1, "description": "不超过合同max_route_attempts"},
    "restart_of": _nullable(_text_schema("重启的已停止/暂停路线ID，否则null")),
    "new_evidence": _nullable(_text_schema("重启所依赖的新增依据及原记录位置，否则null")),
})
REPORT_SCHEMA = _object_schema({
    "analysis": _text_schema("完整解释并引用具体证据"), "mechanism": _text_schema("待检验经济机制"),
    **{name: _array_schema(_text_schema(description), minItems=1) for name, description in {
        "conditions": "适用及失效条件", "falsifiers": "可推翻条件",
        "limitations": "不确定性及实际检查的数据覆盖", "next_steps": "下一步研究"}.items()},
})
CHECK_DIMENSIONS = {"fields_units", "windows", "operation_order", "current_period", "cross_section", "definition"}

RESEARCH_QUESTIONS = {
    "research_basis": "已有结果中什么现象支持继续？引用证据，区分观察与推测，不限定现象类型。",
    "modification_hypothesis": "准备改什么，为什么可能改善？",
    "verifiable_improvement": "与谁比较，预期什么变化，什么结果使修改路线停止？",
    "attempt_value": "是否重复失败尝试？剩余数据、轮数和候选预算是否足够？",
}
DECISION_SCHEMA = _object_schema({
    "candidate_id": _text_schema("本次需要决策的候选ID"),
    "disposition": {"type": "string", "enum": ["optimize", "retain", "pause", "discard"],
                    "description": "继续优化／保留待B验证／暂停／淘汰；retain可以同时继续优化"},
    "continue_optimization": {"type": "boolean"},
    "answers": _object_schema({key: {**_object_schema({
        "supported": {"type": "boolean"}, "reason": _text_schema(question)}), "description": question}
        for key, question in RESEARCH_QUESTIONS.items()}),
    "evidence_refs": _array_schema(_text_schema("已有records中的精确记录ID，至少包含当前候选的程序结果"), minItems=1),
    "reason": _text_schema("解释当前版本的保留价值、证据是否充分及去向；不能把预算用尽或单个不显著结果当作无效"),
    "resume_condition": _nullable(_text_schema("pause时写明恢复研究所需的新数据、诊断或预算；其他去向为null")),
})


def _code_fingerprint() -> dict[str, str]:
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((Path(__file__).parents[2] / "features").glob("*.py"))
    return {**{str(path.relative_to(Path(__file__).parents[2])): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)},
            "runtime.python": sys.version.split()[0], "runtime.numpy": np.__version__, "runtime.pandas": pd.__version__}


def _panel_fingerprint(panel: FactorInputPanel) -> dict[str, Any]:
    values = pd.util.hash_pandas_object(panel.values, index=True).to_numpy().tobytes()
    members = pd.util.hash_pandas_object(panel.universe, index=True).to_numpy().tobytes()
    return {"values_sha256": hashlib.sha256(values).hexdigest(), "membership_sha256": hashlib.sha256(members).hexdigest(),
            "columns": list(panel.values.columns), "rows": len(panel.values),
            "dtypes": {k: str(v) for k, v in panel.values.dtypes.items()}}


def validate_modification(control: str, proposed: str, kind: str) -> None:
    """Enforce one structural change, in addition to the model's semantic review."""
    left, right = compile_expression(control).tree, compile_expression(proposed).tree
    same = lambda a, b: ast.dump(a) == ast.dump(b)
    if kind == "smoothing":
        require(isinstance(right, ast.Call) and right.func.id == "ts_mean" and same(right.args[0], left),
                "smoothing must add one ts_mean around the original formula")
        return
    if kind == "conditioning":
        require(isinstance(right, ast.Call) and right.func.id == "mul" and same(right.args[0], left),
                "conditioning must preserve the original formula as the first argument of mul")
        return
    changes = []

    def visit(a: ast.AST, b: ast.AST, context: str = "") -> None:
        if same(a, b):
            return
        if isinstance(a, ast.Call) and isinstance(b, ast.Call) and len(a.args) == len(b.args):
            if a.func.id != b.func.id:
                changes.append("operator")
            for index, (x, y) in enumerate(zip(a.args, b.args)):
                where = "window" if a.func.id.startswith("ts_") and index == len(a.args) - 1 else ""
                visit(x, y, where)
        elif isinstance(a, ast.Name) and isinstance(b, ast.Name):
            changes.append("field")
        elif isinstance(a, ast.Constant) and isinstance(b, ast.Constant) and context == "window":
            changes.append("window")
        else:
            changes.append("structure")
    visit(left, right)
    require(changes == [kind], f"experiment must change exactly one declared {kind}; observed {changes}")


def _check_report(value: dict[str, Any]) -> dict[str, Any]:
    required = REPORT_SCHEMA["properties"]
    missing = sorted(set(required) - set(value))
    require(not missing, f"evaluation report missing required fields: {', '.join(missing)}")
    for name in ("analysis", "mechanism"):
        text(value[name], name)
    for name in ("conditions", "falsifiers", "limitations", "next_steps"):
        require(isinstance(value[name], list) and bool(value[name]), f"{name} must be a nonempty list")
        for item in value[name]:
            text(item, name)
    return {name: value[name] for name in required}


def _check_review(review: dict[str, Any], cid: str, compiled: Any, error: str | None) -> None:
    require(set(review) == {"candidate_id", "status", "checks", "repair_expression", "reason"}, "invalid semantic review schema")
    require(review["candidate_id"] == cid, "semantic review belongs to a different candidate")
    require(review["status"] in {"consistent", "inconsistent", "insufficient_definition"}, "unknown semantic status")
    text(review["reason"], "semantic reason")
    require(isinstance(review["checks"], list) and len(review["checks"]) == len(CHECK_DIMENSIONS), "semantic review requires six dimensions")
    require({c["dimension"] for c in review["checks"]} == CHECK_DIMENSIONS, "semantic dimensions incomplete")
    for check in review["checks"]:
        require(set(check) == {"dimension", "original_meaning", "actual", "matches", "reason"}, "invalid semantic evidence")
        require(type(check["matches"]) is bool, "matches must be boolean")
        for key in ("original_meaning", "actual", "reason"):
            text(check[key], key)
    if review["status"] == "insufficient_definition":
        require(review["repair_expression"] is None, "insufficient meaning cannot authorize repair")
    elif review["status"] == "consistent":
        require(compiled is not None and not error and all(c["matches"] for c in review["checks"])
                and review["repair_expression"] is None, "consistent verdict contradicts program or item checks")
    else:
        require(not all(c["matches"] for c in review["checks"]) or error is not None,
                "inconsistent verdict must identify a discrepancy")
        text(review["repair_expression"], "repair_expression")


class FactorMiner:
    def __init__(self, spec: ResearchSpec, model: JsonModel, output_root: Path):
        self.spec, self.model = spec, model
        self.root = Path(output_root) / spec.run_id
        self.root.mkdir(parents=True, exist_ok=False)
        write_json(self.root / "contract.json", spec.as_dict())
        write_json(self.root / "code.json", _code_fingerprint())
        self._attach()

    @classmethod
    def open(cls, run_directory: Path, model: JsonModel) -> FactorMiner:
        self = cls.__new__(cls)
        self.root, self.model = Path(run_directory), model
        self.spec = ResearchSpec.from_dict(json.loads((self.root / "contract.json").read_text()))
        require(json.loads((self.root / "code.json").read_text()) == _code_fingerprint(),
                "code changed since exploration; the frozen research run cannot silently use different code")
        self._attach()
        return self

    def _attach(self) -> None:
        self.store = RecordStore(self.root / "a_records")
        self.gateway = AgentGateway(self.model, self.spec, self.store, self.root / "model_calls")
        self.candidates: dict[str, dict[str, Any]] = {}
        self.routes: dict[str, dict[str, Any]] = {}
        self.proposals: dict[str, dict[str, Any]] = {}
        self.decisions: dict[str, dict[str, Any]] = {}

    def _validate_panel(self, panel: FactorInputPanel, stage: str) -> None:
        start, end = self.spec.bounds(stage)
        members = validate_universe(panel.universe)
        require(members.index.equals(panel.values.index) and members.equals(panel.universe), "panel must have canonical explicit membership")
        hours = members.index.get_level_values("timestamp")
        require(hours.min() <= start - pd.Timedelta(hours=self.spec.max_lookback_hours), "panel is missing declared expression warm-up hours")
        require(hours.max() == end - pd.Timedelta(hours=1), "panel must stop at the end of its allowed data segment")
        require("perp_open" in panel.values, "24h target requires perpetual opening prices")
        require(set(panel.values.columns) == set(INPUT_COLUMNS), "mining uses exactly the shared input-field contract")

    def _compile(self, expression: str):
        require(len(expression) <= self.spec.max_formula_nodes * 100, "formula exceeds complexity budget")
        compiled = compile_expression(expression)
        require(sum(1 for _ in ast.walk(compiled.tree)) <= self.spec.max_formula_nodes, "formula exceeds node budget")
        require(compiled.lookback_hours <= self.spec.max_lookback_hours, "formula exceeds declared history budget")
        require(not compiled.unit.powers[1], "cross-sectional factor output cannot compare different base-asset units")
        return compiled

    def _calculate(self, cid: str, panel: FactorInputPanel) -> pd.Series | None:
        item = self.candidates[cid]
        definition = item["definition"]
        expression = definition["expression"]
        checks = []
        result = None
        status = "repair_budget_exhausted"
        execution_error = None
        if not definition["meaning"].strip():
            calculation = {"status": "returned_to_ideator", "original_expression": expression,
                           "original_meaning": definition["meaning"], "checks": [], "executed_expression": None,
                           "reason": "missing formula meaning; no repair is authorized"}
            item["calculation"] = calculation
            self.store.append(f"{cid}-calculation", "calculation", {"candidate_id": cid, **calculation})
            return None
        for attempt in range(self.spec.max_repairs + 1):
            error, compiled = execution_error, None
            execution_error = None
            try:
                compiled = self._compile(expression)
            except ValueError as exc:
                error = str(exc)
            try:
                review = self.gateway.ask("calculator",
                    "逐项核对原始meaning与程序实际步骤，不能按预测效果改写meaning。定义不足则退回构想。"
                    "只有原始含义明确时才给出修复公式，修复后程序会再次检查。检查维度必须完整。",
                    {"candidate_id": cid, "original_definition": definition, "current_expression": expression,
                     "program_error": error, "actual_steps": compiled.calculation_steps() if compiled else None,
                     "catalog_record_id": "inputs"},
                    _object_schema({"candidate_id": {"type": "string", "const": cid},
                        "status": {"type": "string", "enum": ["consistent", "inconsistent", "insufficient_definition"]},
                        "checks": _array_schema(_object_schema({
                            "dimension": {"type": "string", "enum": sorted(CHECK_DIMENSIONS)},
                            "original_meaning": _text_schema("原文具体依据"), "actual": _text_schema("程序实际步骤"),
                            "matches": {"type": "boolean"}, "reason": _text_schema("逐项理由")}), minItems=6, maxItems=6),
                        "repair_expression": _nullable(_text_schema("修复公式；无需修复或定义不足时为null")),
                        "reason": _text_schema("整体判断依据")}),
                    validate=lambda review: _check_review(review, cid, compiled, error))
            except (ModelResponseError, ApiCallError) as exc:
                if isinstance(exc, ApiCallError) and not exc.retryable:
                    raise
                checks.append({"attempt": attempt, "expression": expression, "program_error": error,
                               "model_error": str(exc)})
                status = "model_review_failed"
                break
            record = {"attempt": attempt, "expression": expression, "program_error": error,
                      "actual_steps": compiled.calculation_steps() if compiled else None, "model_review": review}
            checks.append(record)
            if review["status"] == "insufficient_definition":
                status = "returned_to_ideator"
                break
            if review["status"] == "consistent":
                try:
                    result = evaluate_expression(expression, panel)
                except ValueError as exc:
                    record["calculation_error"] = str(exc)
                    status = "calculation_failed"
                    execution_error = str(exc)
                    continue
                status = "computed"
                break
            expression = text(review["repair_expression"], "repair_expression")
        calculation = {"status": status, "original_expression": definition["expression"],
                       "original_meaning": definition["meaning"], "checks": checks,
                       "executed_expression": result.definition if result is not None else None}
        if result is not None:
            calculation["valid_values"] = int(result.values.notna().sum())
            calculation["cross_section_counts"] = finite(result.cross_section_counts.reset_index().to_dict("records"))
            path = self.root / "factor_values" / f"{cid}-A.csv"
            path.parent.mkdir(exist_ok=True)
            with path.open("x") as handle:
                result.values.to_csv(handle)
            calculation["values_artifact"] = str(path.relative_to(self.root))
        item["calculation"] = calculation
        self.store.append(f"{cid}-calculation", "calculation", {"candidate_id": cid, **calculation})
        return result.values if result is not None else None

    def _interpret(self, cid: str, gateway: AgentGateway) -> dict[str, Any]:
        return gateway.ask("evaluator",
            "依据程序完整评估、语义检查和修复记录，解释方向、幅度、稳定性和可信性。"
            "只提供证据分析、适用条件、限制和研究建议，不输出保留/优化/暂停/淘汰或创意卡准入决定。"
            "A段候选去向由优化Agent回答四问后决定；B段通过与交付资格由程序按冻结规则计算，"
            "B段解释须参考validation_result记录，不能用模型意见覆盖程序结论。"
            "不能凭综合分数或单个p值判断有效或无效；同时覆盖失败、证据不足和失效条件。"
            "如数值表分页，按需要读取原文并在limitations中说明实际检查覆盖。",
            {"candidate_id": cid}, REPORT_SCHEMA, validate=_check_report)

    def _try_report(self, cid: str, gateway: AgentGateway) -> dict[str, Any] | None:
        try:
            narrative = self._interpret(cid, gateway)
        except (ModelResponseError, ContextBudgetError, ApiCallError) as exc:
            if isinstance(exc, ApiCallError) and not exc.retryable:
                raise
            attempt = len([r for r in gateway.store.all() if r["kind"] == "report_pending"
                           and r["data"]["candidate_id"] == cid]) + 1
            gateway.store.append(f"{cid}-report-pending-{attempt:03d}", "report_pending", {
                "candidate_id": cid, "error_type": type(exc).__name__, "reason": str(exc),
                "status": "pending", "action": "complete-reports from saved numerical evidence"})
            return None
        gateway.store.append(f"{cid}-report", "model_report", {"candidate_id": cid, **narrative})
        return narrative

    def explore(self, panel: FactorInputPanel) -> dict[str, Any]:
        require(not self.store.all() and not (self.root / "frozen_batch.json").exists(), "exploration can start only in a new research run")
        self._validate_panel(panel, "A")
        with (self.root / "A-universe.csv").open("x") as handle:
            panel.universe.rename("eligible").to_csv(handle)
        self.store.append("inputs", "data_provenance", {"fingerprint": _panel_fingerprint(panel), "catalog": panel.ideation_context()})
        labels = build_labels(panel, self.spec, "A")
        try:
            return self._explore_rounds(panel, labels)
        except Exception as exc:
            # API retries happen inside the gateway. An exhausted or invalid run stops here.
            write_json(self.root / "exploration-stopped.json", {"error_type": type(exc).__name__, "reason": str(exc)})
            raise

    def _explore_rounds(self, panel: FactorInputPanel, labels: pd.DataFrame) -> dict[str, Any]:
        seen: dict[tuple[str, int], str] = {}
        completed_rounds = 0
        for round_no in range(1, self.spec.rounds + 1):
            capacity = min(self.spec.candidates_per_round, self.spec.max_candidates - len(self.candidates))
            if capacity == 0:
                break
            pending = {pid: p for pid, p in self.proposals.items() if p["status"] == "pending"}
            response = self.gateway.ask("ideator",
                "构造可证伪的候选公式，写清计算含义。阅读全部历史及优化建议，逐条记录沿用/调整/放弃。"
                "沿用必须保持建议定义与检验设计一致；调整时在计算前提供更新后的完整设计。"
                "遵守candidate_decisions：暂停或淘汰的候选不自动重启；保留但未获准优化的版本不修改。"
                "不可重复已停止路线且不给新增依据。",
                {"catalog_record_id": "inputs", "candidate_capacity": capacity, "pending_proposals": pending,
                 "candidate_decisions": self.decisions},
                _object_schema({"candidates": _array_schema(CANDIDATE_SCHEMA, maxItems=capacity),
                    "dispositions": _array_schema(_object_schema({
                        "proposal_id": _text_schema("每条pending建议的ID"),
                        "action": {"type": "string", "enum": ["adopt", "adjust", "abandon"]},
                        "reason": _text_schema("理由"),
                        "candidate_index": _nullable({"type": "integer", "minimum": 0}),
                        "design": {**_nullable(DESIGN_SCHEMA), "description": "adjust时提供完整设计，否则null"}}),
                        minItems=len(pending), maxItems=len(pending)),
                    "analysis": _text_schema("本轮构想依据")}),
                validate=lambda response: self._check_ideation(response, capacity, pending))
            definitions, experiments, self.routes, self.proposals = self._check_ideation(response, capacity, pending)
            self.store.append(f"round-{round_no:03d}-ideation", "ideation", response)
            round_ids = []
            for index, definition in enumerate(definitions):
                cid = f"candidate-{len(self.candidates) + 1:04d}"
                item: dict[str, Any] = {"id": cid, "round": round_no, "definition": definition,
                                       "experiment": experiments.get(index)}
                self.candidates[cid] = item
                round_ids.append(cid)
                self.store.append(f"{cid}-definition", "candidate", item)
                try:
                    key = (self._compile(definition["expression"]).expanded_expression, definition["direction"])
                except ValueError:
                    key = None  # Invalid formulas still enter semantic diagnosis and repair.
                if key in seen:
                    item["duplicate_of"] = seen[key]
                    self.store.append(f"{cid}-duplicate", "duplicate", {"candidate_id": cid, "duplicate_of": seen[key]})
                    self._finish_uncomputed_route(item)
                    continue
                values = self._calculate(cid, panel)
                if values is None:
                    self._finish_uncomputed_route(item)
                    continue
                expression = item["calculation"]["executed_expression"]["expanded_expression"]
                if (expression, definition["direction"]) in seen:
                    item["duplicate_of"] = seen[(expression, definition["direction"])]
                    self.store.append(f"{cid}-duplicate", "duplicate", {"candidate_id": cid, "duplicate_of": item["duplicate_of"]})
                    self._finish_uncomputed_route(item)
                    continue
                seen[(expression, definition["direction"])] = cid
                if item["experiment"]:
                    control = self.candidates[item["experiment"]["control_id"]]
                    try:
                        validate_modification(control["calculation"]["executed_expression"]["expression"],
                                              expression, item["experiment"]["change_kind"])
                    except ValueError as exc:
                        self.store.append(f"{cid}-scope-error", "experiment_scope_error", {"candidate_id": cid, "reason": str(exc)})
                        self._finish_uncomputed_route(item)
                        continue
                report = evaluate_factor(values, labels, self.spec, "A", definition["direction"])
                item["evaluation"] = report
                self.store.append(f"{cid}-evaluation", "evaluation", {"candidate_id": cid, **report})
                write_group_plot(self.root / "plots" / f"{cid}-A.svg", report)
                if item["experiment"]:
                    self._compare(cid, values, panel, labels)
                narrative = self._try_report(cid, self.gateway)
                if narrative is not None:
                    item["model_report"] = narrative
            review_ids = list(dict.fromkeys(round_ids + [cid for cid, decision in self.decisions.items()
                                                        if decision["continue_optimization"]]))
            remaining = {"rounds": self.spec.rounds - round_no,
                         "candidates": self.spec.max_candidates - len(self.candidates)}
            optimization = self.gateway.ask("optimizer",
                "先阅读本轮所有候选、失败及完整评估，再判断哪里有值得继续的具体依据。"
                "评估Agent只提供证据解释，不作候选去向决定；由你独立综合证据回答四问并决定去向。"
                "先逐个review_candidate_ids回答四个研究问题并决定去向，decisions必须完整且不重复。"
                "研究依据由你根据证据分析，不限于预设现象；引用真实record ID并说明观察与推测。"
                "区分当前版本与修改路线：retain可同时优化；修改路线停止不自动淘汰原版。"
                "保留须解释A证据为何值得固定版本送B，A阶段不宣称通过独立验证；"
                "证据不足或预算不够则pause并写恢复条件，有充分否定证据或无增量的重复才discard。"
                "不得仅因低分、不显著、预算用尽而淘汰。continue_optimization为true须四个回答均获支持，"
                "并为该候选提供至少一条proposal；false时禁止proposal。没有下一轮或候选预算时不得继续。"
                "依据不足先给diagnostics，不强行生成新公式。每条建议仅检验一个原因，必须给出对照、"
                "最低改善、允许IC损失、停止/暂停条件和预算。evidence必须引用control_id。"
                "程序配对比较的决定不能覆盖；已停止路线若重新启动必须使用新路线并说明新增依据及旧路线。",
                {"round_candidate_ids": round_ids, "review_candidate_ids": review_ids,
                 "candidate_decisions": self.decisions, "route_states": self.routes, "remaining_budget": remaining},
                _object_schema({"analysis": _text_schema("完整研究反馈"),
                    "decisions": _array_schema(DECISION_SCHEMA, minItems=len(review_ids), maxItems=len(review_ids)),
                    "diagnostics": _array_schema(_text_schema("证据缺口和所需诊断")),
                    "proposals": _array_schema(_object_schema({"proposal_id": _text_schema("唯一英文ID"),
                        "candidate": CANDIDATE_SCHEMA, "design": DESIGN_SCHEMA}), maxItems=self.spec.candidates_per_round)}),
                validate=lambda response: self._check_optimization(response, review_ids, remaining))
            decisions, self.routes, self.proposals = self._check_optimization(optimization, review_ids, remaining)
            self.decisions.update(decisions)
            self.store.append(f"round-{round_no:03d}-optimization", "optimization", {**optimization, "route_states": self.routes})
            completed_rounds = round_no
            if not optimization["proposals"]:
                break
        completion = {"completed_rounds": completed_rounds, "candidate_ids": list(self.candidates),
                      "evaluated_ids": [cid for cid, item in self.candidates.items() if "evaluation" in item],
                      "retained_ids": [cid for cid, decision in self.decisions.items() if decision["disposition"] == "retain"],
                      "pending_report_ids": [cid for cid, item in self.candidates.items()
                                             if "evaluation" in item and "model_report" not in item],
                      "candidate_decisions": self.decisions,
                      "routes": self.routes, "proposals": self.proposals,
                      "stop_reason": "no authorized optimization remains within the exploration budget"}
        write_research_report(self.root / "A-report.md", self.spec.run_id, self.spec.purpose, self.store.all(), "A")
        write_json(self.root / "a-complete.json", completion)
        return completion

    def _check_ideation(self, response: dict[str, Any], capacity: int, pending: dict[str, Any]):
        require(set(response) == {"candidates", "dispositions", "analysis"}, "invalid ideation schema")
        text(response["analysis"], "ideation analysis")
        require(isinstance(response["candidates"], list) and len(response["candidates"]) <= capacity, "candidate budget exceeded")
        definitions = [candidate(value) for value in response["candidates"]]
        dispositions = response["dispositions"]
        require(isinstance(dispositions, list) and len(dispositions) == len(pending)
                and {d["proposal_id"] for d in dispositions} == set(pending), "every pending proposal needs one disposition")
        experiments: dict[int, dict[str, Any]] = {}
        routes, proposals = copy.deepcopy(self.routes), copy.deepcopy(self.proposals)
        for disposition in dispositions:
            require(set(disposition) == {"proposal_id", "action", "reason", "candidate_index", "design"}, "invalid disposition")
            text(disposition["reason"], "disposition reason")
            action, index = disposition["action"], disposition["candidate_index"]
            require(action in {"adopt", "adjust", "abandon"}, "unknown disposition")
            proposal = pending[disposition["proposal_id"]]
            require(self.decisions[proposal["design"]["control_id"]]["continue_optimization"],
                    "candidate decision does not authorize optimization")
            if action == "abandon":
                require(index is None and disposition["design"] is None, "abandon has no candidate/design")
            else:
                require(type(index) is int and 0 <= index < len(definitions) and index not in experiments, "invalid adopted candidate index")
                definition = definitions[index]
                require(definition["proposal_id"] == disposition["proposal_id"], "candidate/proposal linkage differs")
                plan = proposal["design"] if action == "adopt" else design(disposition["design"], self.spec)
                if action == "adopt":
                    require(definition == proposal["candidate"] and disposition["design"] is None, "adoption changed the proposed definition")
                require(definition["parent_id"] == plan["control_id"], "experiment must reference its control")
                require(definition["direction"] == self.candidates[plan["control_id"]]["definition"]["direction"], "controlled modifications keep direction fixed")
                self._register_route(plan, routes)
                route = routes[plan["route_id"]]
                require(route["decision"] == "continue" and route["attempts"] < plan["attempt_budget"], "route stopped or budget exhausted")
                route["attempts"] += 1
                experiments[index] = plan
            proposals[disposition["proposal_id"]]["status"] = action
        for index, definition in enumerate(definitions):
            parent = definition["parent_id"]
            if parent is not None:
                require(parent in self.candidates, "unknown parent candidate")
                require(parent in self.decisions and self.decisions[parent]["continue_optimization"],
                        "candidate decision does not authorize a new version")
                require(index in experiments,
                        "modifying an evaluated candidate requires a predeclared experiment")
            require(definition["proposal_id"] is None or index in experiments, "proposal has no adopted experiment")
        return definitions, experiments, routes, proposals

    def _check_optimization(self, optimization: dict[str, Any], review_ids: list[str], remaining: dict[str, int]):
        require(set(optimization) == {"analysis", "decisions", "diagnostics", "proposals"}, "invalid optimizer schema")
        text(optimization["analysis"], "optimization analysis")
        require(isinstance(optimization["diagnostics"], list) and isinstance(optimization["proposals"], list), "optimizer lists are required")
        for diagnostic in optimization["diagnostics"]:
            text(diagnostic, "diagnostic")
        require(len(optimization["proposals"]) <= self.spec.candidates_per_round, "optimization proposal budget exceeded")
        decisions = self._check_decisions(optimization["decisions"], review_ids, remaining)
        proposal_controls = set()
        routes, proposals = copy.deepcopy(self.routes), copy.deepcopy(self.proposals)
        for proposal in optimization["proposals"]:
            require(set(proposal) == {"proposal_id", "candidate", "design"}, "invalid optimization proposal")
            pid = identifier(proposal["proposal_id"])
            require(pid not in proposals, "proposal IDs must be unique")
            proposed = candidate(proposal["candidate"])
            plan = design(proposal["design"], self.spec)
            require(plan["control_id"] in decisions and decisions[plan["control_id"]]["continue_optimization"],
                    "proposal is not authorized by the candidate decision")
            proposal_controls.add(plan["control_id"])
            require(proposed["parent_id"] == plan["control_id"] and proposed["proposal_id"] == pid, "invalid proposal version links")
            self._register_route(plan, routes)
            require(routes[plan["route_id"]]["decision"] == "continue", "cannot propose on a stopped/paused route without new evidence")
            proposals[pid] = {**proposal, "status": "pending"}
        require(proposal_controls == {cid for cid, decision in decisions.items() if decision["continue_optimization"]},
                "every continuing candidate requires an experiment proposal")
        require(len(optimization["proposals"]) <= remaining["candidates"], "proposals exceed remaining candidate budget")
        return decisions, routes, proposals

    def _check_decisions(self, values: Any, review_ids: list[str], remaining: dict[str, int]) -> dict[str, dict[str, Any]]:
        require(isinstance(values, list) and len(values) == len(review_ids), "every reviewed candidate needs one decision")
        records = {record["id"]: record for record in self.store.all()}
        decisions = {}
        for value in values:
            require(isinstance(value, dict) and set(value) == set(DECISION_SCHEMA["properties"]), "invalid candidate decision schema")
            cid = value["candidate_id"]
            require(isinstance(cid, str) and cid in review_ids and cid not in decisions, "unknown or repeated candidate decision")
            disposition, continuing = value["disposition"], value["continue_optimization"]
            require(disposition in {"optimize", "retain", "pause", "discard"} and type(continuing) is bool, "invalid candidate disposition")
            require((disposition != "optimize" or continuing) and (disposition not in {"pause", "discard"} or not continuing),
                    "candidate disposition contradicts optimization decision")
            text(value["reason"], "candidate decision reason")
            answers = value["answers"]
            require(isinstance(answers, dict) and set(answers) == set(RESEARCH_QUESTIONS), "answer all four research questions")
            for answer in answers.values():
                require(isinstance(answer, dict) and set(answer) == {"supported", "reason"}
                        and type(answer["supported"]) is bool, "invalid research answer")
                text(answer["reason"], "research answer reason")
            refs = value["evidence_refs"]
            require(isinstance(refs, list) and refs and all(isinstance(ref, str) and ref in records for ref in refs),
                    "decision evidence must reference existing records")
            own_results = [records[ref] for ref in refs if records[ref]["kind"] in {
                "evaluation", "calculation", "duplicate", "experiment_scope_error", "experiment_result"}
                and records[ref]["data"]["candidate_id"] == cid]
            require(bool(own_results), "decision requires this candidate's program evidence")
            if disposition == "retain" or continuing:
                require("evaluation" in self.candidates[cid] and any(r["kind"] == "evaluation" for r in own_results),
                        "retention or optimization requires evaluated candidate evidence")
            if disposition == "discard":
                require("evaluation" in self.candidates[cid] or "duplicate_of" in self.candidates[cid],
                        "calculation failure alone cannot discard a candidate; pause for diagnosis")
            if continuing:
                require(all(answer["supported"] for answer in answers.values()), "optimization requires support for all four questions")
                require(remaining["rounds"] > 0 and remaining["candidates"] > 0, "no exploration budget for optimization")
            if disposition == "pause":
                text(value["resume_condition"], "pause resume condition")
            else:
                require(value["resume_condition"] is None, "resume condition belongs only to paused candidates")
            decisions[cid] = value
        return decisions

    def _register_route(self, plan: dict[str, Any], routes: dict[str, Any]) -> None:
        plan = design(plan, self.spec)
        control = plan["control_id"]
        require(control in self.candidates and "evaluation" in self.candidates[control], "experiment control must have A evidence")
        require(control in plan["evidence"], "experiment evidence must identify its control candidate")
        rid = plan["route_id"]
        if rid in routes:
            require(routes[rid]["design"] == plan, "cannot rewrite an existing route's success criteria")
        else:
            if plan["restart_of"] is not None:
                require(plan["restart_of"] in routes, "restarted route must link an existing route")
                require(routes[plan["restart_of"]]["decision"] != "continue", "only stopped or paused routes can restart")
            routes[rid] = {"design": plan, "attempts": 0, "decision": "continue"}

    def _finish_uncomputed_route(self, item: dict[str, Any]) -> None:
        if item["experiment"]:
            route = self.routes[item["experiment"]["route_id"]]
            route["decision"] = "pause_budget" if route["attempts"] >= route["design"]["attempt_budget"] else "pause_insufficient"

    def _compare(self, cid: str, values: pd.Series, panel: FactorInputPanel, labels: pd.DataFrame) -> None:
        item = self.candidates[cid]
        plan = item["experiment"]
        control = self.candidates[plan["control_id"]]
        baseline = evaluate_expression(control["calculation"]["executed_expression"]["expression"], panel).values
        common = values.notna() & baseline.notna()
        # Both candidates are ranked/grouped again on the identical valid assets.
        trial = evaluate_factor(values.where(common), labels, self.spec, "A", item["definition"]["direction"])
        base = evaluate_factor(baseline.where(common), labels, self.spec, "A", control["definition"]["direction"])
        comparison = compare_experiment(trial, base, plan, self.spec)
        comparison["common_asset_observations"] = int(common.sum())
        comparison["trial_on_common_assets"] = trial
        comparison["control_on_common_assets"] = base
        route = self.routes[plan["route_id"]]
        route["decision"] = comparison["decision"]
        if route["attempts"] >= plan["attempt_budget"] and route["decision"] == "pause_insufficient":
            route["decision"] = "pause_budget"
        elif route["attempts"] >= plan["attempt_budget"] and route["decision"] == "continue":
            route["decision"] = "complete_supported"
        comparison["route_decision"] = route["decision"]
        item["comparison"] = comparison
        self.store.append(f"{cid}-comparison", "experiment_result", {"candidate_id": cid, **comparison})

    def _load_candidates(self) -> None:
        self.decisions = {}
        for record in self.store.all():
            data = record["data"]
            if record["kind"] == "candidate":
                self.candidates[data["id"]] = data
        for record in self.store.all():
            if record["kind"] in {"calculation", "evaluation", "model_report"}:
                data = record["data"]
                key = "model_report" if record["kind"] == "model_report" else record["kind"]
                self.candidates[data["candidate_id"]][key] = data
            elif record["kind"] == "optimization":
                self.decisions.update({decision["candidate_id"]: decision for decision in record["data"]["decisions"]})

    def freeze(self, candidate_ids: list[str], validation_universe: pd.Series) -> dict[str, Any]:
        require((self.root / "a-complete.json").exists(), "complete A exploration before freezing B")
        require(candidate_ids and len(set(candidate_ids)) == len(candidate_ids), "freeze a nonempty unique candidate batch")
        validation_universe = validate_universe(validation_universe)
        start, end = self.spec.bounds("B")
        times = validation_universe.index.get_level_values("timestamp")
        require(times.min() == start - pd.Timedelta(hours=self.spec.max_lookback_hours)
                and times.max() == end - pd.Timedelta(hours=1), "freeze the complete B membership and warm-up")
        old = pd.read_csv(self.root / "A-universe.csv")
        old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
        old_members = old.set_index(["timestamp", "symbol"])["eligible"]
        overlap = validation_universe.index[validation_universe.index.get_level_values("timestamp") < start]
        require(set(overlap) <= set(old_members.index), "B warm-up universe contains undeclared historical assets")
        require(np.array_equal(old_members.loc[overlap].to_numpy(), validation_universe.loc[overlap].to_numpy()),
                "B warm-up membership differs from its recorded A history")
        self._load_candidates()
        definitions = {}
        for cid in candidate_ids:
            require(cid in self.candidates and "evaluation" in self.candidates[cid], "only computed and evaluated candidates can be frozen")
            require(cid in self.decisions and self.decisions[cid]["disposition"] == "retain",
                    "only explicitly retained candidates can be frozen for B")
            item = self.candidates[cid]
            require("model_report" in item, "complete the retained candidate report before freezing B")
            definitions[cid] = {"definition": item["definition"], "executed": item["calculation"]["executed_expression"],
                                "calculation_sha256": digest(item["calculation"]), "a_evaluation": item["evaluation"],
                                "a_model_report": item["model_report"], "a_decision": self.decisions[cid]}
        frozen = {"contract": self.spec.as_dict(), "candidates": definitions,
                  "validation_membership_sha256": hashlib.sha256(pd.util.hash_pandas_object(validation_universe, index=True).to_numpy().tobytes()).hexdigest(),
                  "a_records": {r["id"]: r["sha256"] for r in self.store.all()},
                  "code": _code_fingerprint(), "frozen_at": datetime.now(timezone.utc).isoformat()}
        frozen["sha256"] = digest(frozen)
        write_json(self.root / "frozen_batch.json", frozen)
        return {"candidate_ids": candidate_ids, "sha256": frozen["sha256"]}

    def validate(self, load_panel: Callable[[], FactorInputPanel], idea_pool: Path) -> dict[str, Any]:
        already_accessed = (self.root / "b-access-started.json").exists()
        try:
            return self._validate_once(load_panel, idea_pool)
        except Exception as exc:
            if not already_accessed and (self.root / "b-access-started.json").exists():
                write_json(self.root / "validation-stopped.json", {"error_type": type(exc).__name__, "reason": str(exc),
                           "B_status": "access was attempted; inspect existing evidence before any subsequent research"})
            raise

    def _checked_frozen(self) -> tuple[dict[str, Any], str]:
        frozen = json.loads((self.root / "frozen_batch.json").read_text())
        expected_hash = frozen.pop("sha256")
        require(digest(frozen) == expected_hash and frozen["code"] == _code_fingerprint(), "frozen batch/code changed")
        require(frozen["contract"] == json.loads((self.root / "contract.json").read_text()), "research contract changed")
        require(frozen["a_records"] == {r["id"]: r["sha256"] for r in self.store.all()}, "A evidence changed after freezing")
        require(digest(frozen["contract"]) == digest(self.spec.as_dict()), "loaded research contract changed")
        return frozen, expected_hash

    def _validate_once(self, load_panel: Callable[[], FactorInputPanel], idea_pool: Path) -> dict[str, Any]:
        frozen, expected_hash = self._checked_frozen()
        # This marker is committed before even invoking the B data loader.
        write_json(self.root / "b-access-started.json", {"frozen_sha256": expected_hash,
                   "time": datetime.now(timezone.utc).isoformat(), "use": "fixed_batch_validation"})
        panel = load_panel()
        self._validate_panel(panel, "B")
        require(_panel_fingerprint(panel)["membership_sha256"] == frozen["validation_membership_sha256"],
                "B membership changed after the batch was frozen")
        with (self.root / "B-universe.csv").open("x") as handle:
            panel.universe.rename("eligible").to_csv(handle)
        labels = build_labels(panel, self.spec, "B")
        store = RecordStore(self.root / "b_records")
        store.append("inputs", "data_provenance", {"fingerprint": _panel_fingerprint(panel), "catalog": panel.ideation_context()})
        reports = {}
        for cid, item in frozen["candidates"].items():
            store.append(f"{cid}-frozen", "frozen_definition_and_A_evidence", item)
            result = evaluate_expression(item["executed"]["expression"], panel)
            require(result.definition == item["executed"], "B execution differs from frozen formula")
            reports[cid] = evaluate_factor(result.values, labels, self.spec, "B", item["definition"]["direction"])
            store.append(f"{cid}-evaluation", "evaluation", {"candidate_id": cid, **reports[cid]})
            write_group_plot(self.root / "plots" / f"{cid}-B.svg", reports[cid])
        correction = correct_batch(reports, self.spec)
        store.append("batch-correction", "multiple_testing", correction)
        decisions = {}
        for cid, report in reports.items():
            summary = report["summary"]
            reasons = []
            tests = [t for t in correction["tests"] if t["candidate_id"] == cid]
            if not all(t["rejected"] for t in tests):
                reasons.append("primary tests lack batch-corrected support")
            ic, spread = summary["rank_ic"]["mean"], summary["directional_spread"]["mean"]
            if ic is None or ic * report["direction"] < self.spec.min_abs_ic:
                reasons.append("directional IC is below the predeclared minimum")
            if spread is None or spread < self.spec.min_directional_spread:
                reasons.append("directional return spread is below the predeclared minimum")
            if summary["valid_stages"] < 2 or summary["positive_stage_share"] < self.spec.min_stage_share:
                reasons.append("stage repetition is insufficient")
            validation_status = "not_passed" if reasons else "passed"
            if self.spec.purpose != "research":
                reasons.append("engineering checks cannot enter the idea pool")
            decisions[cid] = {"validation_status": validation_status, "decision_source": "program",
                              "eligible_for_idea_pool": not reasons, "reasons": reasons, "tests": tests}
            store.append(f"{cid}-validation", "validation_result", {"candidate_id": cid, **decisions[cid]})
        checkpoint = {"frozen_sha256": expected_hash, "b_records": {r["id"]: r["sha256"] for r in store.all()}}
        write_json(self.root / "b-numerical-complete.json", {**checkpoint, "sha256": digest(checkpoint)})
        return self._complete_b_reports(idea_pool)

    def complete_reports(self, stage: str, idea_pool: Path) -> dict[str, Any]:
        """Fill missing narratives from saved evidence only; never reload A/B prices."""
        require(stage in {"A", "B"}, "report stage must be A or B")
        require(json.loads((self.root / "code.json").read_text()) == _code_fingerprint(), "research code changed")
        require(digest(json.loads((self.root / "contract.json").read_text())) == digest(self.spec.as_dict()),
                "research contract changed")
        if stage == "B":
            return self._complete_b_reports(idea_pool)
        require(not (self.root / "frozen_batch.json").exists(), "A evidence cannot change after freezing B")
        records = self.store.all()
        evaluated = [r["data"]["candidate_id"] for r in records if r["kind"] == "evaluation"]
        require(bool(evaluated), "no saved A evaluations to explain")
        reports = {r["data"]["candidate_id"] for r in records if r["kind"] == "model_report"}
        for cid in evaluated:
            if cid not in reports:
                self._try_report(cid, self.gateway)
        records = self.store.all()
        reports = {r["data"]["candidate_id"] for r in records if r["kind"] == "model_report"}
        pending = [cid for cid in evaluated if cid not in reports]
        write_research_report(self.root / "A-report.md", self.spec.run_id, self.spec.purpose, records, "A", replace=True)
        return {"stage": "A", "status": "reports_pending" if pending else "complete", "pending_report_ids": pending,
                "exploration_complete": (self.root / "a-complete.json").exists()}

    def _complete_b_reports(self, idea_pool: Path) -> dict[str, Any]:
        frozen, expected_hash = self._checked_frozen()
        require((self.root / "b-numerical-complete.json").exists(), "complete B numerical validation before completing reports")
        checkpoint = json.loads((self.root / "b-numerical-complete.json").read_text())
        checkpoint_hash = checkpoint.pop("sha256")
        require(digest(checkpoint) == checkpoint_hash and checkpoint["frozen_sha256"] == expected_hash,
                "B numerical checkpoint changed")
        access = json.loads((self.root / "b-access-started.json").read_text())
        require(access["frozen_sha256"] == expected_hash, "B access marker differs from the frozen batch")
        store = RecordStore(self.root / "b_records")
        records = store.all()
        numerical_kinds = {"data_provenance", "frozen_definition_and_A_evidence", "evaluation", "multiple_testing", "validation_result"}
        require({r["id"]: r["sha256"] for r in records if r["kind"] in numerical_kinds} == checkpoint["b_records"],
                "B numerical evidence changed after calculation")
        reports = {r["data"]["candidate_id"]: r["data"] for r in records if r["kind"] == "evaluation"}
        decisions = {r["data"]["candidate_id"]: {k: v for k, v in r["data"].items() if k != "candidate_id"}
                     for r in records if r["kind"] == "validation_result"}
        require(set(reports) == set(decisions) == set(frozen["candidates"]), "B numerical batch is incomplete")
        correction = next(r["data"] for r in records if r["kind"] == "multiple_testing")
        narratives = {r["data"]["candidate_id"]: r for r in records if r["kind"] == "model_report"}
        gateway = AgentGateway(self.model, self.spec, store, self.root / "model_calls", stage="B")
        for cid in reports:
            if cid not in narratives:
                self._try_report(cid, gateway)
        records = store.all()
        narratives = {r["data"]["candidate_id"]: r for r in records if r["kind"] == "model_report"}
        pending = [cid for cid in reports if cid not in narratives]
        for cid, decision in decisions.items():
            if decision["eligible_for_idea_pool"] and cid in narratives:
                narrative = narratives[cid]
                card = self._idea_card(cid, frozen["candidates"][cid], reports[cid], narrative["data"], decision,
                                       expected_hash, narrative["created_at"])
                path = Path(idea_pool) / f"{self.spec.run_id}--{cid}.json"
                if path.exists():
                    require(json.loads(path.read_text()) == card, "existing idea card differs from saved research evidence")
                else:
                    write_json(path, card)
                decision["idea_card"] = str(path)
        output = {"frozen_sha256": expected_hash, "batch_correction": correction, "decisions": decisions,
                  "status": "reports_pending" if pending else "complete", "pending_report_ids": pending,
                  "B_use": "used for fixed-candidate selection; not untouched data for subsequent strategy development",
                  "C_use": "reserved; never loaded by factor mining"}
        write_research_report(self.root / "B-report.md", self.spec.run_id, self.spec.purpose, records, "B", replace=True)
        if not pending:
            path = self.root / "validation.json"
            if path.exists():
                require(json.loads(path.read_text()) == output, "completed validation differs from saved evidence")
            else:
                write_json(path, output)
        return output

    def _idea_card(self, cid: str, item: dict[str, Any], report: dict[str, Any], narrative: dict[str, Any],
                   decision: dict[str, Any], frozen_hash: str, report_created_at: str) -> dict[str, Any]:
        definition = item["definition"]
        return {"id": f"{self.spec.run_id}--{cid}", "source_type": "factor_mining", "status": "research_idea",
                "source": {"run_id": self.spec.run_id, "candidate_id": cid, "date": report_created_at,
                           "research_directory": str(self.root.resolve()), "frozen_batch_sha256": frozen_hash},
                "original_claim": {"formula": item["executed"], "meaning": definition["meaning"],
                                   "hypothesis": definition["hypothesis"], "initial_findings": report["summary"]},
                "economic_mechanism": narrative["mechanism"],
                "market_and_horizon": {"venue": "Binance", "market": "USD-M perpetual", "inputs": "1h", "target": "24h"},
                "data_and_coverage": {"fields": item["executed"]["fields"], "B": report["coverage"],
                                      "universe_provenance": self.spec.universe_provenance},
                "falsification_conditions": narrative["falsifiers"],
                "unverified_assumptions": narrative["limitations"] + ["因子参数未经最优性证明；完整选币、调仓、仓位、成本及风险规则待研究"],
                "next_research_plan": narrative["next_steps"], "applicability": narrative["conditions"],
                "admission_evidence": decision, "a_research_decision": item["a_decision"],
                "b_validation_status": decision["validation_status"],
                "strategy_validation_status": "not_started"}
