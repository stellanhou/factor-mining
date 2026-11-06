"""Calculator role: deterministic formula execution and formula-error repair."""
from __future__ import annotations

import pandas as pd

from crypto_quant.features.factor_expressions import evaluate_expression
from crypto_quant.features.factor_inputs import FactorInputPanel
from .contracts import _object_schema, _text_schema, require
from .research_actions import FormulaBudgetError, archive_factor_values
from .model import ApiCallError
from .records import ModelResponseError


class CalculatorRole:
    def _calculate(self, cid: str, panel: FactorInputPanel) -> pd.Series | None:
        item = self.candidates[cid]
        definition = item["definition"]
        expression = definition["expression"]
        checks = []
        result = None
        status = "calculation_failed"

        for attempt in range(self.spec.max_repairs + 1):
            compiled = None
            try:
                compiled = self._compile(expression)
                result = evaluate_expression(expression, panel)
            except ValueError as exc:
                error = str(exc)
                checks.append({"attempt": attempt, "expression": expression,
                               "actual_steps": compiled.calculation_steps() if compiled else None,
                               "program_error": error})
                if isinstance(exc, FormulaBudgetError) or attempt == self.spec.max_repairs:
                    break
                try:
                    repair = self.gateway.ask(
                        "calculator",
                        "根据程序报错修复因子公式，使其使用允许的字段和算子并可执行。"
                        "只修改公式；无需判断候选的文字解释或金融假设。"
                        "只依据程序报错修复，不因量纲或跨币尺度修改可执行公式。"
                        "保留原计算含义，不得删除归一化分母、输入项或改变窗口以简化公式。",
                        {"candidate_id": cid, "current_expression": expression,
                         "program_error": error, "catalog_record_id": "inputs"},
                        _object_schema({
                            "candidate_id": {"type": "string", "const": cid},
                            "repair_expression": _text_schema("修复后的可执行公式"),
                            "reason": _text_schema("针对程序报错的修复依据"),
                        }),
                        validate=lambda value: require(value["candidate_id"] == cid,
                                                       "formula repair belongs to a different candidate"),
                    )
                except (ModelResponseError, ApiCallError) as exc:
                    if isinstance(exc, ApiCallError) and not exc.retryable:
                        raise
                    checks[-1]["model_error"] = str(exc)
                    status = "formula_repair_failed"
                    break
                expression = repair["repair_expression"]
                continue
            checks.append({"attempt": attempt, "expression": expression,
                           "actual_steps": compiled.calculation_steps(), "program_error": None})
            status = "computed"
            break

        calculation = {"status": status, "original_expression": definition["expression"],
                       "original_meaning": definition["meaning"], "checks": checks,
                       "executed_expression": result.definition if result is not None else None}
        if result is not None:
            calculation["valid_values"] = int(result.values.notna().sum())
            calculation["values_artifact"] = archive_factor_values(
                self.archive_root, self.root, self.spec, cid, definition, result)
        item["calculation"] = calculation
        self.store.append(f"{cid}-calculation", "calculation", {"candidate_id": cid, **calculation})
        return result.values if result is not None else None
