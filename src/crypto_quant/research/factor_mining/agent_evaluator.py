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
            "A段必须读取horizon_comparison中列出的全部期限，解释相同公式与方向在共同样本下的Rank IC、分组收益和分阶段稳定性，"
            "说明期限间信号是否衰减、增强或反向；不得只挑最好期限，也不得根据A结果自行确定保留范围。"
            "若A评估包含rank_displacement，须并列解释Δ=1h、4h、24h各自D的均值、中位数、90分位数、有效时点数和有效比例，"
            "并说明覆盖、并列或常量截面以及不同评价阶段的差异；Δ是两次观测的真实小时距离。"
            "将预测能力与排序变化作为两类证据并列解释，不把D称为实际换手、交易次数、手续费或收益改善；"
            "D不是成本金额，也不能单独支持保留或淘汰。缺失、边界排除及配对覆盖不足应按记录说明，不能解释成稳定。"
            "A期限对照用于开发，p值为探索性未校正结果；收益未扣交易成本和资金费。"
            "B段必须逐项解释冻结的retained_horizons中每个期限，包括有效样本、跨段排除、Rank IC及校正结果、"
            "方向IC、有向价差、各轨道诊断、通过或失败原因；不能遗漏未通过期限或只报告总体结论。"
            "若冻结运行含rank_displacement诊断，说明三个Δ各自的覆盖与可用阶段信息，并指出它是描述性排序代理指标，"
            "不代表实际费用；不得把B段该诊断反馈为A段优化证据。"
            "明确区分逐期限统计判定、A与B期限交集，以及整项eligible_for_idea_pool程序结论；"
            "说明候选与期限组成的BH检验家族，不把多期限通过解释成多张卡，也不宣称卡片级错误发现率受到控制。"
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
