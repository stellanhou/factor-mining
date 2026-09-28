"""Evaluator role: evidence interpretation and report validation."""
from __future__ import annotations

from typing import Any

from .contracts import _object_schema, _text_schema, _array_schema
from .model import ApiCallError
from .records import AgentGateway, ContextBudgetError, ModelResponseError

REPORT_SCHEMA = _object_schema({
    "analysis": _text_schema("完整解释并引用具体证据"), "mechanism": _text_schema("待检验经济机制"),
    **{name: _array_schema(_text_schema(description), minItems=1) for name, description in {
        "conditions": "适用及失效条件", "falsifiers": "可推翻条件",
        "limitations": "不确定性及实际检查的数据覆盖", "next_steps": "下一步研究"}.items()},
})

class EvaluatorRole:
    def _interpret(self, cid: str, gateway: AgentGateway) -> dict[str, Any]:
        return gateway.ask("evaluator",
            "依据程序完整评估和公式执行记录，解释方向、幅度、稳定性和可信性。"
            "只提供证据分析、适用条件、限制和研究建议，不输出保留/优化/暂停/淘汰或创意卡准入决定。"
            "A段候选去向由优化Agent回答四问后决定；B段通过与交付资格由程序按冻结规则计算，"
            "B段解释须参考validation_result记录，不能用模型意见覆盖程序结论。"
            "不能凭综合分数或单个p值判断有效或无效；同时覆盖失败、证据不足和失效条件。"
            "A段若有horizon_comparison，必须并列解读同一因子在1h、4h、24h的Rank IC、分组收益和分阶段稳定性，"
            "说明是否衰减、增强或反向；不得只挑最好期限，或仅凭较长期累计收益较大判断更好。"
            "该对照使用共同有效样本、同一方向和HAC带宽，p值为探索性未校正结果；"
            "它不改变24h主评估、配对修改检验及B准入规则，收益未扣交易成本和资金费。"
            "如数值表分页，按需要读取原文并在limitations中说明实际检查覆盖。",
            {"candidate_id": cid}, REPORT_SCHEMA)

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
