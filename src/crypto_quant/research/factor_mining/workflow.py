"""Ideate -> semantically check/compute -> evaluate -> decide -> design an experiment.

Exploration and fixed-batch validation have separate record stores. The latter
cannot repair a formula or send feedback into the exploration loop.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from crypto_quant.features.factor_expressions import compile_expression, evaluate_expression
from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS, validate_universe
from .contracts import ResearchSpec, candidate, digest, identifier, modification_plan, require, text
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
        "hypothesis": "待检验的金融解释",
        "change_reason": "生成理由；采用修改任务时说明公式如何实现change_target并遵守fixed_components"}.items()},
    "direction": {"type": "integer", "enum": [-1, 1], "description": "预先固定预测方向"},
    "parent_id": _nullable(_text_schema("原候选ID；无父候选时为null")),
    "proposal_id": _nullable(_text_schema("采用的优化建议ID；自主构想时为null")),
})
MODIFICATION_TASK_SCHEMA = _object_schema({
    "core_hypothesis": _text_schema("原候选的核心金融假设"),
    "observed_problem": _text_schema("引用已有证据说明当前版本的具体问题"),
    "modification_hypothesis": _text_schema("为什么指定改动可能改善或更好检验核心假设"),
    "change_target": _text_schema("允许Ideator实现的语义改动目标；不得包含最终可执行公式"),
    "fixed_components": _text_schema("除被检验改动外必须保持不变的定义部分"),
})
EXPERIMENT_DESIGN_SCHEMA = _object_schema({
    "question": _text_schema("新候选与原候选的配对检验问题"),
    "metric": {"type": "string", "enum": ["rank_ic", "directional_spread"]},
    "min_improvement": {"type": "number", "exclusiveMinimum": 0, "description": "配对改善最低要求"},
    "max_ic_loss": {"type": "number", "minimum": 0, "description": "允许的有向IC损失"},
    "expected_outcome": _text_schema("修改假设得到支持时预期出现的可观测变化"),
    "stop_condition": _text_schema("预定改善或IC损失区间上限不足时的停止条件"),
    "pause_condition": _text_schema("证据不精确或缺数据时的暂停条件"),
})
PROPOSAL_SCHEMA = _object_schema({
    "proposal_id": _text_schema("唯一英文ID"),
    "route_id": _text_schema("新路线或已有继续路线的英文ID"),
    "control_id": _text_schema("已评估的原候选ID"),
    "evidence_refs": _array_schema(_text_schema("已有records中的精确记录ID"), minItems=1),
    "modification_task": MODIFICATION_TASK_SCHEMA,
    "experiment_design": EXPERIMENT_DESIGN_SCHEMA,
    "restart_of": _nullable(_text_schema("重启的已停止/暂停路线ID，否则null")),
    "new_evidence": _nullable(_text_schema("重启所依赖的新增依据及原记录位置，否则null")),
})
MODIFICATION_PLAN_FIELDS = (
    "route_id", "control_id", "evidence_refs", "modification_task", "experiment_design",
    "restart_of", "new_evidence",
)
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
    "attempt_value": "是否属于无新增信息的重复失败尝试？继续能否带来新的可验证信息？",
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
    "reason": _text_schema("解释当前版本的保留价值、证据是否充分及去向；不能把单个不显著结果当作无效"),
    "resume_condition": _nullable(_text_schema("pause时写明恢复研究所需的新数据或诊断；其他去向为null")),
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


def _reject_unsupported_oi_cost_basis(field_references: str, *claims: str) -> None:
    """Keep aggregate OI quantity/value within the meanings exposed by the catalog."""
    oi_fields = ("open_interest_value", "open_interest_base")
    referenced = tuple(field for field in oi_fields if field in field_references)
    if not referenced:
        return
    description = " ".join(claims).casefold()
    if len(referenced) == 2:
        unsupported_cost = (
            "平均持仓", "平均开仓", "平均入场", "入场价", "平均名义价格",
            "持仓成本", "开仓成本", "浮盈", "浮亏",
            "average entry", "entry price", "entry level", "average position price",
            "average notional price", "cost basis", "break-even", "breakeven",
            "unrealized pnl", "unrealized profit", "unrealized loss",
        )
        require(not any(term in description for term in unsupported_cost),
                "open_interest_value/open_interest_base is current aggregate notional per base unit; "
                "it does not provide entry price, cost basis, or unrealized PnL")
    unsupported_position_inference = (
        "加杠杆", "去杠杆", "提高杠杆", "降低杠杆", "杠杆率", "杠杆倍数", "杠杆头寸",
        "新建头寸", "新增头寸", "新头寸", "头寸建立", "头寸的建立", "头寸加速建立",
        "方向性头寸建立", "建仓", "开仓方向", "平仓方向", "头寸平仓",
        "lever up", "levering up", "deleverag", "leverage ratio", "higher leverage", "lower leverage",
        "new position", "position building", "position establishment", "open position", "close position",
    )
    negations = (
        "不表示", "不代表", "不推断", "不用于推断", "不能推断", "不可推断", "无法推断",
        "不能据此推断", "不能解释", "不解释", "不应", "不得", "禁止",
        "does not", "do not", "cannot", "can't", "must not", "not infer", "no evidence",
    )
    for term in unsupported_position_inference:
        for match in re.finditer(re.escape(term), description):
            context = description[max(0, match.start() - 48):match.end() + 24]
            require(any(negation in context for negation in negations),
                    "open interest is an aggregate current quantity/value; its change does not identify "
                    "leverage ratios, participant identity, or open/close and long/short position direction")


def _reject_unsupported_residualization(*claims: str) -> None:
    """Do not label fixed arithmetic as regression-based residualization."""
    description = " ".join(claims).casefold()
    unsupported = (
        "正交化", "中性化", "残差化", "回归残差", "剔除共线", "去除共线",
        "orthogonal", "neutraliz", "residualiz", "regression residual", "de-correlat",
    )
    negations = (
        "没有", "不支持", "不使用", "未使用", "未执行", "未提供", "不得", "不能", "不可", "无法", "禁止",
        "不引入", "不涉及", "不做", "不作", "未做", "未作", "不包含", "未包含", "不进行", "未进行", "并非", "排除", "不存在", "无需", "不需要",
        "no ", "not ", "without ", "cannot", "can't", "must not", "unsupported", "does not",
        "did not", "never ", "neither", "nor ",
    )
    for term in unsupported:
        for match in re.finditer(re.escape(term), description):
            context = description[max(0, match.start() - 48):match.end() + 24]
            require(any(negation in context for negation in negations),
                    "the expression catalog has no regression/residualization operator; fixed-coefficient "
                    "rank or z-score arithmetic cannot be described as orthogonalization, neutralization, "
                    "residualization, or removal of collinearity")


def _hourly_return_market(node: ast.AST) -> str | None:
    """Recognize the catalog's one-hour close-to-close return expansion."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "log" and len(node.args) == 1):
        return None
    ratio = node.args[0]
    if not (isinstance(ratio, ast.Call) and isinstance(ratio.func, ast.Name)
            and ratio.func.id == "div" and len(ratio.args) == 2):
        return None
    current, delayed = ratio.args
    if not (isinstance(current, ast.Name)
            and isinstance(delayed, ast.Call) and isinstance(delayed.func, ast.Name)
            and delayed.func.id == "ts_delay" and len(delayed.args) == 2
            and isinstance(delayed.args[0], ast.Name)
            and isinstance(delayed.args[1], ast.Constant) and delayed.args[1].value == 1
            and current.id == delayed.args[0].id):
        return None
    return {"spot_close": "spot", "perp_close": "perp"}.get(current.id)


def _lead_lag_orientation(node: ast.AST) -> str | None:
    """Return the leading market in corr(current_x, delay(return_y, n))."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "ts_corr" and len(node.args) == 3):
        return None
    current = _hourly_return_market(node.args[0])
    delayed = node.args[1]
    if not (current and isinstance(delayed, ast.Call) and isinstance(delayed.func, ast.Name)
            and delayed.func.id == "ts_delay" and len(delayed.args) == 2
            and isinstance(delayed.args[1], ast.Constant)
            and type(delayed.args[1].value) is int and delayed.args[1].value > 0):
        return None
    previous = _hourly_return_market(delayed.args[0])
    if previous is None or previous == current:
        return None
    return f"{previous}_leads_{current}"


def _reject_reversed_lead_lag_semantics(expression: str, *claims: str) -> None:
    """Reject the observed reversal of leader/follower labels in shifted correlations."""
    try:
        tree = compile_expression(expression).tree
    except ValueError:
        return  # Formula errors are diagnosed by the calculator with program evidence.
    pair = next(((_lead_lag_orientation(node.args[0]), _lead_lag_orientation(node.args[1]))
                 for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "sub" and len(node.args) == 2
                 and _lead_lag_orientation(node.args[0]) and _lead_lag_orientation(node.args[1])), None)
    orientations = [value for value in (_lead_lag_orientation(node) for node in ast.walk(tree)) if value]
    if not orientations:
        return
    name = claims[0].casefold().replace("-", "_") if claims else ""
    description = " ".join(claims).casefold()
    expected = pair[0] if pair else orientations[0]
    wrong = "spot_leads_perp" if expected == "perp_leads_spot" else "perp_leads_spot"
    expected_token = "spot_lead" if expected == "spot_leads_perp" else "perp_lead"
    wrong_token = "spot_lead" if wrong == "spot_leads_perp" else "perp_lead"
    name_order_is_reversed = wrong_token in name and (
        expected_token not in name or name.index(wrong_token) < name.index(expected_token))
    chinese_reversal = (
        expected == "perp_leads_spot" and (
            "现货领先项为现货当期" in description or "现货对下一小时永续的领先" in description)
        or expected == "spot_leads_perp" and (
            "永续领先项为永续当期" in description or "永续对下一小时现货的领先" in description)
    )
    require(not (name_order_is_reversed or chinese_reversal),
            "lead-lag orientation is reversed: corr(current x, delay(y, n)) means y leads x; "
            f"the first shifted correlation in this expression is {expected}")


def _reject_misstated_lookback(expression: str, meaning: str) -> None:
    """Keep an explicitly claimed maximum lookback equal to the executable tree."""
    try:
        compiled = compile_expression(expression)
        actual = compiled.lookback_hours
    except ValueError:
        return  # Formula errors are diagnosed by the calculator with program evidence.
    claimed = [int(value) for value in re.findall(r"最大(?:显式)?回看(?:为)?\s*(\d+)\s*小时", meaning)]
    if not claimed:
        return
    feature_horizons = {"funding_7d_sum": 168, "funding_24h_sum": 24}
    inherent = max([feature_horizons[f] for f in compiled.fields if f in feature_horizons] or [0])
    valid_lookbacks = {actual, actual + inherent} if inherent else {actual}
    require(all(value in valid_lookbacks for value in claimed),
            f"stated maximum lookback differs from executable expression: claimed={claimed}, actual={actual}")


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
    _reject_unsupported_residualization(
        value["analysis"], value["mechanism"],
        *value["conditions"], *value["falsifiers"],
        *value["limitations"], *value["next_steps"])
    report_text = " ".join((value["analysis"], value["mechanism"],
                            *value["conditions"], *value["falsifiers"],
                            *value["limitations"], *value["next_steps"]))
    _reject_unsupported_oi_cost_basis(report_text, report_text)
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


def validate_panel(panel: FactorInputPanel, spec: ResearchSpec, stage: str) -> None:
    start, end = spec.bounds(stage)
    members = validate_universe(panel.universe)
    require(members.index.equals(panel.values.index) and members.equals(panel.universe), "panel must have canonical explicit membership")
    hours = members.index.get_level_values("timestamp")
    require(hours.min() <= start - pd.Timedelta(hours=spec.max_lookback_hours), "panel is missing declared expression warm-up hours")
    require(hours.max() == end - pd.Timedelta(hours=1), "panel must stop at the end of its allowed data segment")
    require("perp_open" in panel.values, "24h target requires perpetual opening prices")
    require(set(panel.values.columns) == set(INPUT_COLUMNS), "mining uses exactly the shared input-field contract")


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
        validate_panel(panel, self.spec, stage)

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
                    "modification_plan不为null时，还须在definition维度核对候选定义和实际步骤是否落实"
                    "modification_task.change_target并遵守fixed_components；不得改写任务或配对判断标准。"
                    "只有原始含义明确时才给出修复公式，修复后程序会再次检查。检查维度必须完整。",
                    {"candidate_id": cid, "original_definition": definition, "current_expression": expression,
                     "program_error": error, "actual_steps": compiled.calculation_steps() if compiled else None,
                     "modification_plan": item.get("experiment"), "catalog_record_id": "inputs"},
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
            csv = result.values.to_csv()
            if path.exists():
                require(path.read_text() == csv, "saved factor values differ during checkpoint recovery")
            else:
                with path.open("x") as handle:
                    handle.write(csv)
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
            "候选涉及OI时，各报告字段只用‘聚合OI数量/名义价值变化的统计条件关联’"
            "这一中性语义；不讨论交易行为或身份机制，也不列举被排除的解释，即便是否定句。"
            "当前表达式白名单没有回归或残差算子；不得建议正交化、中性化、残差化、"
            "剔除共线或报告残差统计量。"
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

    def explore(self, panel: FactorInputPanel, *, goal_context: dict[str, Any] | None = None) -> dict[str, Any]:
        require(not self.store.all() and not (self.root / "frozen_batch.json").exists(), "exploration can start only in a new research run")
        self._validate_panel(panel, "A")
        universe_path = self.root / "A-universe.csv"
        csv = panel.universe.rename("eligible").to_csv()
        if universe_path.exists():
            require(universe_path.read_text() == csv, "saved A universe differs during initialization recovery")
        else:
            with universe_path.open("x") as handle:
                handle.write(csv)
        self.store.append("inputs", "data_provenance", {"fingerprint": _panel_fingerprint(panel), "catalog": panel.ideation_context()})
        if goal_context is not None:
            self.store.append("goal-context", "goal_context", goal_context)
        return self._run_exploration(panel)

    def resume_explore(self, panel: FactorInputPanel) -> dict[str, Any]:
        """Replay committed A checkpoints; never reopen a run after B was frozen."""
        require(not (self.root / "frozen_batch.json").exists(), "cannot resume A after freezing B")
        self._validate_panel(panel, "A")
        records = {r["id"]: r for r in self.store.all()}
        require("inputs" in records and records["inputs"]["data"]["fingerprint"] == _panel_fingerprint(panel),
                "A inputs changed; cannot resume the saved research run")
        self.candidates, self.decisions, self.routes, self.proposals = {}, {}, {}, {}
        return self._run_exploration(panel)

    def _run_exploration(self, panel: FactorInputPanel) -> dict[str, Any]:
        labels = build_labels(panel, self.spec, "A")
        try:
            return self._explore_rounds(panel, labels)
        except Exception as exc:
            # API retries happen inside the gateway. An exhausted or invalid run stops here.
            path = self.root / "exploration-stopped.json"
            if not path.exists():
                write_json(path, {"error_type": type(exc).__name__, "reason": str(exc)})
            raise

    def _explore_rounds(self, panel: FactorInputPanel, labels: pd.DataFrame) -> dict[str, Any]:
        saved = {r["id"]: r["data"] for r in self.store.all()}
        context = saved.get("goal-context", {})
        seen: dict[tuple[str, int], str] = {
            (item["expression"], item["direction"]): item["candidate_ref"]
            for item in context.get("previous_expressions", [])}
        completed_rounds = 0
        round_no = 0
        while True:
            round_no += 1
            pending = {pid: p for pid, p in self.proposals.items() if p["status"] == "pending"}
            ideation_id = f"round-{round_no:03d}-ideation"
            response = saved[ideation_id] if ideation_id in saved else self.gateway.ask("ideator",
                "构造当前有研究依据的一组可证伪候选公式，写清计算含义，不为凑数量生成候选。"
                "阅读全部历史及优化任务，逐条记录采用或放弃。"
                "严格按inputs字段目录解释数据；open_interest_value/open_interest_base只是"
                "当前聚合名义价值与基础币数量之比，不得将其称为平均名义价格，也不得推断"
                "历史开仓成本、平均入场价、平均持仓价或浮动盈亏。OI变化只能表示当前"
                "聚合未平仓数量/名义价值变化，不能推断杠杆率、加杠杆/去杠杆、"
                "开平仓或多空方向，也不能推断由哪类参与者造成。"
                "若使用OI状态，hypothesis和meaning只能陈述交互项的可证伪统计条件关联，"
                "不得补造建仓、平仓、杠杆或参与者机制；无法写成纯统计假设时必须不生成该候选。"
                "纠错时必须删除全部越界机制，不得只在其他字段追加否定句。"
                "Optimizer只给出修改任务和配对检验合同；最终候选定义和可执行公式必须由你首次生成。"
                "采用任务时，候选parent_id必须引用control_id，proposal_id必须引用当前任务，"
                "预测方向保持不变；change_reason须具体说明公式如何实现change_target并遵守fixed_components。"
                "不得改写Optimizer预先声明的修改任务或判断标准；无法一致实现时放弃，由Optimizer另立新任务。"
                "领先-滞后公式中，corr(当期x, delay(y,n))表示y领先x；候选名称、meaning、"
                "hypothesis与做差顺序必须与这一时间对齐一致。"
                "meaning若声明最大回看小时数，必须累加底层收益、外层delay与滚动窗口的"
                "完整依赖，并与程序可执行表达式一致（funding_7d_sum等底座预计算字段的算子回看为0，声明0或168均可接受）。"
                "当前白名单没有回归或残差算子；不得把固定系数的rank/z-score加减称为正交化、"
                "中性化、残差化或剔除共线。修改任务需要这些操作时必须abandon。"
                "遵守candidate_decisions：暂停或淘汰的候选不自动重启；保留但未获准优化的版本不修改。"
                "不可重复已停止路线且不给新增依据。",
                {"catalog_record_id": "inputs", "pending_proposals": pending,
                 "candidate_decisions": self.decisions},
                _object_schema({"candidates": _array_schema(CANDIDATE_SCHEMA),
                    "dispositions": _array_schema(_object_schema({
                        "proposal_id": _text_schema("每条pending建议的ID"),
                        "action": {"type": "string", "enum": ["adopt", "abandon"]},
                        "reason": _text_schema("采用时说明候选如何落实任务；放弃时说明不能一致实现的原因"),
                        "candidate_index": _nullable({"type": "integer", "minimum": 0})}),
                        minItems=len(pending), maxItems=len(pending)),
                    "analysis": _text_schema("本轮构想依据")}),
                validate=lambda response: self._check_ideation(response, pending))
            definitions, experiments, self.routes, self.proposals = self._check_ideation(response, pending)
            if ideation_id not in saved:
                self.store.append(ideation_id, "ideation", response)
            round_ids = []
            for index, definition in enumerate(definitions):
                cid = f"candidate-{len(self.candidates) + 1:04d}"
                item: dict[str, Any] = {"id": cid, "round": round_no, "definition": definition,
                                       "experiment": experiments.get(index)}
                self.candidates[cid] = item
                round_ids.append(cid)
                if f"{cid}-definition" in saved:
                    require(saved[f"{cid}-definition"] == item, "saved candidate differs from accepted ideation")
                else:
                    self.store.append(f"{cid}-definition", "candidate", item)
                try:
                    key = (self._compile(definition["expression"]).expanded_expression, definition["direction"])
                except ValueError:
                    key = None  # Invalid formulas still enter semantic diagnosis and repair.
                if key in seen:
                    item["duplicate_of"] = seen[key]
                    if f"{cid}-duplicate" not in saved:
                        self.store.append(f"{cid}-duplicate", "duplicate", {"candidate_id": cid, "duplicate_of": seen[key]})
                    self._finish_uncomputed_route(item)
                    continue
                if f"{cid}-calculation" in saved:
                    item["calculation"] = saved[f"{cid}-calculation"]
                    executed = item["calculation"]["executed_expression"]
                    values = evaluate_expression(executed["expression"], panel).values if executed else None
                else:
                    values = self._calculate(cid, panel)
                if values is None:
                    self._finish_uncomputed_route(item)
                    continue
                expression = item["calculation"]["executed_expression"]["expanded_expression"]
                if (expression, definition["direction"]) in seen:
                    item["duplicate_of"] = seen[(expression, definition["direction"])]
                    if f"{cid}-duplicate" not in saved:
                        self.store.append(f"{cid}-duplicate", "duplicate", {"candidate_id": cid, "duplicate_of": item["duplicate_of"]})
                    self._finish_uncomputed_route(item)
                    continue
                seen[(expression, definition["direction"])] = cid
                report = (saved[f"{cid}-evaluation"] if f"{cid}-evaluation" in saved else
                          evaluate_factor(values, labels, self.spec, "A", definition["direction"]))
                item["evaluation"] = report
                if f"{cid}-evaluation" not in saved:
                    self.store.append(f"{cid}-evaluation", "evaluation", {"candidate_id": cid, **report})
                plot = self.root / "plots" / f"{cid}-A.svg"
                if not plot.exists():
                    write_group_plot(plot, report)
                if item["experiment"]:
                    if f"{cid}-comparison" in saved:
                        item["comparison"] = saved[f"{cid}-comparison"]
                        self.routes[item["experiment"]["route_id"]]["decision"] = item["comparison"]["route_decision"]
                    else:
                        self._compare(cid, values, panel, labels)
                narrative = saved[f"{cid}-report"] if f"{cid}-report" in saved else self._try_report(cid, self.gateway)
                if narrative is not None:
                    item["model_report"] = narrative
            review_ids = list(dict.fromkeys(round_ids + [cid for cid, decision in self.decisions.items()
                                                        if decision["continue_optimization"]]))
            optimization_id = f"round-{round_no:03d}-optimization"
            optimization = ({k: v for k, v in saved[optimization_id].items() if k != "route_states"}
                            if optimization_id in saved else self.gateway.ask("optimizer",
                "先阅读本轮所有候选、失败及完整评估，再判断哪里有值得继续的具体依据。"
                "评估Agent只提供证据解释，不作候选去向决定；由你独立综合证据回答四问并决定去向。"
                "先逐个review_candidate_ids回答四个研究问题并决定去向，decisions必须完整且不重复。"
                "研究依据由你根据证据分析，不限于预设现象；引用真实record ID并说明观察与推测。"
                "区分当前版本与修改路线：retain可同时优化；修改路线停止不自动淘汰原版。"
                "去向决策准则：全批次BY FDR多重校正与样本外独立检验是B段程序的专职职责，"
                "不得将‘A段未做全批次FDR、未做正交化或A段非独立样本’作为拒绝送B的理由。"
                "当候选在A段呈现出与假设一致的方向、Rank IC具有统计显著性（如点估计|IC|>=0.02且p<0.05）"
                "或分组收益呈现合理单调趋势时，应当积极判定为retain保留送B，交由B段冻结批次接受客观检验。"
                "若认为当前候选仍有改进空间，支持在判定retain的同时将continue_optimization设为true继续优化。"
                "证据不足、IC接近零或修改路线受阻缺乏明确方向时pause并写恢复条件；"
                "有充分否定证据或无增量的重复才discard；不得仅因单一统计量未达完美而淘汰。"
                "continue_optimization为true须四个回答均获支持，"
                "并为该候选提供至少一条proposal；false时禁止proposal。"
                "依据不足先给diagnostics，不强行生成修改任务。不得输出新候选定义、名称、最终公式、"
                "公式含义或方向；这些由下一轮Ideator首次生成。"
                "每条proposal只输出：精确证据引用、原核心假设、已观察问题、修改假设、语义改动目标和固定部分；"
                "以及与原候选的配对检验问题、主指标、最低改善、允许IC损失、预期、停止和暂停条件。"
                "同一假设下可在change_target中指定多个相互关联的语义改动，但不写可执行表达式。"
                "候选涉及OI时，所有决策、diagnostics和proposal只用‘聚合OI数量/"
                "名义价值变化的统计条件关联’这一中性语义；不讨论交易行为或身份机制，"
                "也不列举被排除的解释，即便是否定句。"
                "当前表达式白名单没有回归或残差算子；不得提议正交化、中性化、残差化、"
                "剔除共线或其他无法由当前白名单真实执行的统计操作；"
                "固定系数的rank/z-score加减不属于上述操作。需要时应pause并记录恢复条件。"
                "判断条件须与experiment_design.metric、最低改善和允许IC损失一致。"
                "control_id必须是获准继续的已评估候选；evidence_refs必须包含该候选的评估记录。"
                "程序配对比较的决定不能覆盖；已停止路线若重新启动必须使用新路线并说明新增依据及旧路线。",
                {"round_candidate_ids": round_ids, "review_candidate_ids": review_ids,
                 "candidate_decisions": self.decisions, "route_states": self.routes},
                _object_schema({"analysis": _text_schema("完整研究反馈"),
                    "decisions": _array_schema(DECISION_SCHEMA, minItems=len(review_ids), maxItems=len(review_ids)),
                    "diagnostics": _array_schema(_text_schema("证据缺口和所需诊断")),
                    "proposals": _array_schema(PROPOSAL_SCHEMA)}),
                validate=lambda response: self._check_optimization(response, review_ids)))
            decisions, self.routes, self.proposals = self._check_optimization(optimization, review_ids)
            self.decisions.update(decisions)
            if optimization_id not in saved:
                self.store.append(optimization_id, "optimization", {**optimization, "route_states": self.routes})
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
                      "stop_reason": "optimizer returned no authorized optimization proposals"}
        write_research_report(self.root / "A-report.md", self.spec.run_id, self.spec.purpose, self.store.all(), "A", replace=True)
        path = self.root / "a-complete.json"
        if path.exists():
            require(json.loads(path.read_text()) == completion, "completed A checkpoint differs from saved evidence")
        else:
            write_json(path, completion)
        return completion

    def _check_ideation(self, response: dict[str, Any], pending: dict[str, Any]):
        require(set(response) == {"candidates", "dispositions", "analysis"}, "invalid ideation schema")
        text(response["analysis"], "ideation analysis")
        require(isinstance(response["candidates"], list), "candidates must be a list")
        definitions = [candidate(value) for value in response["candidates"]]
        for definition in definitions:
            _reject_reversed_lead_lag_semantics(
                definition["expression"], definition["name"], definition["hypothesis"],
                definition["meaning"], definition["change_reason"])
            _reject_misstated_lookback(definition["expression"], definition["meaning"])
            _reject_unsupported_oi_cost_basis(
                definition["expression"], definition["name"], definition["hypothesis"],
                definition["meaning"], definition["change_reason"])
            _reject_unsupported_residualization(
                definition["name"], definition["hypothesis"],
                definition["meaning"], definition["change_reason"])
        dispositions = response["dispositions"]
        require(isinstance(dispositions, list) and len(dispositions) == len(pending)
                and {d["proposal_id"] for d in dispositions} == set(pending), "every pending proposal needs one disposition")
        experiments: dict[int, dict[str, Any]] = {}
        routes, proposals = copy.deepcopy(self.routes), copy.deepcopy(self.proposals)
        for disposition in dispositions:
            require(set(disposition) == {"proposal_id", "action", "reason", "candidate_index"}, "invalid disposition")
            text(disposition["reason"], "disposition reason")
            action, index = disposition["action"], disposition["candidate_index"]
            require(action in {"adopt", "abandon"}, "unknown disposition")
            proposal = pending[disposition["proposal_id"]]
            require(self.decisions[proposal["control_id"]]["continue_optimization"],
                    "candidate decision does not authorize optimization")
            if action == "abandon":
                require(index is None, "abandon has no candidate")
            else:
                require(type(index) is int and 0 <= index < len(definitions) and index not in experiments, "invalid adopted candidate index")
                definition = definitions[index]
                require(definition["proposal_id"] == disposition["proposal_id"], "candidate/proposal linkage differs")
                plan = self._proposal_plan(proposal)
                require(definition["parent_id"] == plan["control_id"], "experiment must reference its control")
                require(definition["direction"] == self.candidates[plan["control_id"]]["definition"]["direction"], "controlled modifications keep direction fixed")
                self._register_route(plan, routes)
                route = routes[plan["route_id"]]
                require(route["decision"] == "continue", "route is stopped or paused")
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

    def _check_optimization(self, optimization: dict[str, Any], review_ids: list[str]):
        require(set(optimization) == {"analysis", "decisions", "diagnostics", "proposals"}, "invalid optimizer schema")
        text(optimization["analysis"], "optimization analysis")
        require(isinstance(optimization["diagnostics"], list) and isinstance(optimization["proposals"], list), "optimizer lists are required")
        for diagnostic in optimization["diagnostics"]:
            text(diagnostic, "diagnostic")
        decisions = self._check_decisions(optimization["decisions"], review_ids)
        proposal_controls = set()
        routes, proposals = copy.deepcopy(self.routes), copy.deepcopy(self.proposals)
        for proposal in optimization["proposals"]:
            require(set(proposal) == set(PROPOSAL_SCHEMA["properties"]), "invalid optimization proposal")
            pid = identifier(proposal["proposal_id"])
            require(pid not in proposals, "proposal IDs must be unique")
            plan = self._proposal_plan(proposal)
            task = plan["modification_task"]
            _reject_unsupported_oi_cost_basis(
                f"{task['modification_hypothesis']} {task['change_target']}",
                task["modification_hypothesis"], task["change_target"])
            _reject_unsupported_residualization(
                task["modification_hypothesis"], task["change_target"])
            require(plan["control_id"] in decisions and decisions[plan["control_id"]]["continue_optimization"],
                    "proposal is not authorized by the candidate decision")
            proposal_controls.add(plan["control_id"])
            self._register_route(plan, routes)
            require(routes[plan["route_id"]]["decision"] == "continue", "cannot propose on a stopped/paused route without new evidence")
            proposals[pid] = {**proposal, "status": "pending"}
        require(proposal_controls == {cid for cid, decision in decisions.items() if decision["continue_optimization"]},
                "every continuing candidate requires an experiment proposal")
        return decisions, routes, proposals

    @staticmethod
    def _proposal_plan(value: dict[str, Any]) -> dict[str, Any]:
        return modification_plan({name: value[name] for name in MODIFICATION_PLAN_FIELDS})

    def _check_decisions(self, values: Any, review_ids: list[str]) -> dict[str, dict[str, Any]]:
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
                "evaluation", "calculation", "duplicate", "experiment_result"}
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
            if disposition == "pause":
                text(value["resume_condition"], "pause resume condition")
            else:
                require(value["resume_condition"] is None, "resume condition belongs only to paused candidates")
            decisions[cid] = value
        return decisions

    def _register_route(self, plan: dict[str, Any], routes: dict[str, Any]) -> None:
        plan = modification_plan(plan)
        control = plan["control_id"]
        require(control in self.candidates and "evaluation" in self.candidates[control], "experiment control must have A evidence")
        records = {record["id"] for record in self.store.all()}
        require(all(ref in records for ref in plan["evidence_refs"]), "modification evidence must reference existing records")
        require(f"{control}-evaluation" in plan["evidence_refs"],
                "modification evidence must reference its control evaluation")
        rid = plan["route_id"]
        if rid in routes:
            require(routes[rid]["plan"] == plan, "cannot rewrite an existing route's success criteria")
        else:
            if plan["restart_of"] is not None:
                require(plan["restart_of"] in routes, "restarted route must link an existing route")
                require(routes[plan["restart_of"]]["decision"] != "continue", "only stopped or paused routes can restart")
            routes[rid] = {"plan": plan, "attempts": 0, "decision": "continue"}

    def _finish_uncomputed_route(self, item: dict[str, Any]) -> None:
        if item["experiment"]:
            route = self.routes[item["experiment"]["route_id"]]
            route["decision"] = "pause_insufficient"

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
