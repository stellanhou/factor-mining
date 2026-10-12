"""Evaluator role: A research decisions, evidence interpretation, and reports."""
from __future__ import annotations

from typing import Any

from .contracts import require, text, _object_schema, _text_schema, _array_schema, _nullable
from .model import ApiCallError
from .agent_optimizer import OptimizerRole
from .records import AgentGateway, ContextBudgetError, ModelResponseError

RESEARCH_QUESTIONS = {
    "research_basis": "已有结果中什么现象支持继续？引用证据，区分观察与推测，不限定现象类型。",
    "modification_hypothesis": "准备改什么，为什么可能改善？",
    "verifiable_improvement": "与谁比较，预期什么变化，什么结果使修改路线停止？",
    "attempt_value": "是否属于无新增信息的重复失败尝试？继续能否带来新的可验证信息？",
}

REPORT_SCHEMA = _object_schema({
    "analysis": _text_schema("完整解释并引用具体证据"), "mechanism": _text_schema("待检验经济机制"),
    **{name: _array_schema(_text_schema(description), minItems=1) for name, description in {
        "conditions": "适用及失效条件", "falsifiers": "可推翻条件",
        "limitations": "不确定性及实际检查的数据覆盖", "next_steps": "下一步研究"}.items()},
})

CANDIDATE_DECISION_SCHEMA = _object_schema({
    "candidate_id": _text_schema("本次评估的候选ID"),
    "disposition": {"type": "string", "enum": ["optimize", "retain", "pause", "discard"]},
    "retained_horizons": _array_schema(
        {"type": "integer", "enum": [1, 4, 24]}, maxItems=3,
        description="retain时选择送B验证的期限；其他去向填写空列表"),
    "continue_optimization": {"type": "boolean"},
    "answers": _object_schema({key: {**_object_schema({
        "supported": {"type": "boolean"}, "reason": _text_schema(question)}), "description": question}
        for key, question in RESEARCH_QUESTIONS.items()}),
    "reason": _text_schema("引用本候选A证据说明去向"),
    "resume_condition": _nullable(_text_schema("pause时的恢复条件，否则为null")),
    "evidence_refs": _array_schema(_text_schema("已有records中的精确记录ID，至少包含本候选证据"), minItems=1),
})

INFORMATION_REQUEST_SCHEMA = _object_schema({
    "purpose": _text_schema("本次补充取证要解决的问题"),
    "questions": _array_schema(_text_schema("需要构想Agent通过获准A证据回答的具体问题"), minItems=1),
    "record_refs": _array_schema(_text_schema("当前A记录或获准历史A cycle记录ID；无特定记录时可为空")),
})

ROUND_REVIEW_SCHEMA = _object_schema({
    "round_action": {"type": "string", "enum": ["continue_ideation", "continue_optimization", "end_direction"]},
    "candidate_decisions": _array_schema(CANDIDATE_DECISION_SCHEMA),
    "information_request": _nullable(INFORMATION_REQUEST_SCHEMA),
    "candidate_reports": _array_schema(_object_schema({
        "candidate_id": _text_schema("已评估候选ID"),
        **REPORT_SCHEMA["properties"],
    })),
    "analysis": _text_schema("本轮证据综合说明"),
})

class EvaluatorRole:
    def _check_candidate_decisions(self, values: Any, review_ids: list[str]) -> dict[str, dict[str, Any]]:
        require(isinstance(values, list) and len(values) == len(review_ids),
                "every reviewed candidate needs one evaluator decision")
        records = {record["id"]: record for record in self.store.all()}
        decisions = {}
        for value in values:
            require(isinstance(value, dict) and set(value) == set(CANDIDATE_DECISION_SCHEMA["properties"]),
                    "invalid evaluator candidate decision schema")
            cid = value["candidate_id"]
            require(isinstance(cid, str) and cid in review_ids and cid not in decisions,
                    "unknown or repeated evaluator candidate decision")
            candidate = self.candidates[cid]
            disposition, continuing = value["disposition"], value["continue_optimization"]
            require(disposition in {"optimize", "retain", "pause", "discard"} and type(continuing) is bool,
                    "invalid evaluator candidate disposition")
            retained_horizons = value["retained_horizons"]
            require(isinstance(retained_horizons, list)
                    and all(type(horizon) is int and horizon in {1, 4, 24} for horizon in retained_horizons)
                    and len(retained_horizons) == len(set(retained_horizons)),
                    "retained_horizons must be a unique list from [1, 4, 24]")
            if disposition == "retain":
                require(bool(retained_horizons), "retain requires at least one retained horizon")
            else:
                require(not retained_horizons, "only retain decisions may specify retained horizons")
            require((disposition != "optimize" or continuing)
                    and (disposition not in {"pause", "discard"} or not continuing),
                    "candidate disposition contradicts optimization decision")
            text(value["reason"], "candidate decision reason")
            answers = value["answers"]
            require(isinstance(answers, dict) and set(answers) == set(RESEARCH_QUESTIONS),
                    "answer all four research questions")
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
                require("evaluation" in candidate
                        and any(record["kind"] == "evaluation" for record in own_results),
                        "retention or optimization requires evaluated candidate evidence")
            if disposition == "retain":
                evaluation_records = [record for record in own_results if record["kind"] == "evaluation"]
                require(len(evaluation_records) == 1
                        and evaluation_records[0]["id"] == f"{cid}-evaluation"
                        and evaluation_records[0]["data"]["segment"] == "A",
                        "retained candidate requires its A evaluation record")
                comparison = evaluation_records[0]["data"]["horizon_comparison"]
                require(isinstance(comparison, dict) and isinstance(comparison["horizons"], dict),
                        "retained candidate requires A horizon comparison evidence")
                for horizon in retained_horizons:
                    require(str(horizon) in comparison["horizons"],
                            f"retained {horizon}h A horizon evidence is missing")
                    OptimizerRole._check_retained_horizon_evidence(
                        comparison["horizons"][str(horizon)], horizon,
                        self.candidates[cid]["definition"]["direction"])
            if disposition == "discard":
                require("evaluation" in candidate or "duplicate_of" in candidate,
                        "calculation failure alone cannot discard a candidate; pause for diagnosis")
            if continuing:
                require(all(answer["supported"] for answer in answers.values()),
                        "optimization requires support for all four questions")
            if disposition == "pause":
                text(value["resume_condition"], "pause resume condition")
            else:
                require(value["resume_condition"] is None, "resume condition belongs only to paused candidates")
            decisions[cid] = value
        return decisions

    def _allowed_information_refs(self) -> set[str]:
        records = self.store.all()
        allowed = {record["id"] for record in records}
        for record in records:
            if record["kind"] != "goal_context":
                continue
            prior = record["data"]["prior_A_research"]
            allowed.update(item["source_record_id"] for item in prior["tasks"])
            allowed.update(item["source_record_id"] for item in prior["cycles"])
        return allowed | self.allowed_history_record_ids

    def _check_round_review(self, response: dict[str, Any], decision_ids: list[str],
                            report_ids: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        require(set(response) == set(ROUND_REVIEW_SCHEMA["properties"]), "invalid research review schema")
        text(response["analysis"], "round review analysis")
        require(isinstance(report_ids, list) and len(report_ids) == len(set(report_ids))
                and set(report_ids) <= set(decision_ids),
                "A report candidate IDs must be unique reviewed candidates")
        require(all(cid in self.candidates and "evaluation" in self.candidates[cid]
                    for cid in report_ids),
                "A reports require evaluated candidates")
        decisions = self._check_candidate_decisions(response["candidate_decisions"], decision_ids)
        action = response["round_action"]
        require(action in {"continue_ideation", "continue_optimization", "end_direction"},
                "invalid research review round_action")
        continuing = {cid for cid, decision in decisions.items() if decision["continue_optimization"]}
        if action == "continue_optimization":
            require(bool(continuing), "continue_optimization requires evaluator-authorized candidates")
        elif action == "continue_ideation":
            require(not continuing, "continue_ideation cannot bypass authorized optimization tasks")
        else:
            require(not continuing, "end_direction cannot leave optimization authorized")
        information_request = response["information_request"]
        if information_request is not None:
            require(isinstance(information_request, dict)
                    and set(information_request) == {"purpose", "questions", "record_refs"},
                    "invalid evaluator information_request schema")
            text(information_request["purpose"], "information request purpose")
            questions = information_request["questions"]
            require(isinstance(questions, list) and questions,
                    "information_request requires a concrete question")
            for question in questions:
                text(question, "information request question")
            refs = information_request["record_refs"]
            require(isinstance(refs, list) and all(isinstance(ref, str) for ref in refs)
                    and set(refs) <= self._allowed_information_refs(),
                    "information_request references an unknown or unauthorized record")
        require(action != "end_direction" or information_request is None,
                "end_direction cannot request more evidence")
        require(action != "continue_ideation" or information_request is not None,
                "continue_ideation requires a concrete next-step request")
        reports = response["candidate_reports"]
        require(isinstance(reports, list) and len(reports) == len(report_ids),
                "every pending A report requires one evaluator report")
        report_map = {}
        for report in reports:
            cid = report["candidate_id"]
            require(cid in report_ids and cid not in report_map, "unknown or repeated A candidate report")
            report_map[cid] = {key: value for key, value in report.items() if key != "candidate_id"}
        require(set(report_map) == set(report_ids), "A candidate reports are incomplete")
        return decisions, report_map

    def _review_round(self, round_no: int, round_ids: list[str], review_ids: list[str],
                      report_ids: list[str], gateway: AgentGateway,
                      *, evaluator_review_ref: dict[str, str] | None = None) -> dict[str, Any]:
        task = (
            "你是评估Agent，审阅本轮A段构想、计算状态与数值证据，决定下一步路线。"
            "逐个review_candidate_ids给出candidate_decisions；candidate_reports只为report_candidate_ids返回解释。"
            "disposition=retain可同时保留送B期限并继续优化；pause须写resume_condition；discard须有实际评估或重复证据，"
            "单纯计算失败只能暂停并诊断。候选是否进入B及continue_optimization都由你决定；Optimizer不再作这些裁决。"
            "round_action=continue_ideation表示继续构想/补充取证，continue_optimization表示把已授权候选交给Optimizer设计修改任务，"
            "end_direction表示结束当前A研究方向（可保留候选送B），不代表Goal完成。"
            "若candidate_ids为空、Ideator返回空候选、资料缺口或公式无法计算，必须依据本轮任务和A证据评估，"
            "不能把空候选视为Ideator终止命令；若continue_ideation，information_request必须列明具体purpose、questions和可选A record_refs，"
            "不能要求Ideator为了填列表制造公式，也不能用空request让它盲目空转。"
            "candidate_decisions必须回答research_basis、modification_hypothesis、verifiable_improvement、attempt_value四问；"
            "continue_optimization须四问全部supported，原有去向判断门槛不得因职责迁移而放宽。"
            "四问用于判断当前研究依据、待检验的修改假设、可验证的改善方向和继续尝试价值；"
            "不要求尚未执行的proposal或配对结果预先得到证明，也不能把下一次实验的预期结果当成已有证据。"
            "去向判断须遵守既有准则：A未做全批次FDR、未做正交化或A样本非独立，均不能单独作为拒绝固定送B的理由；"
            "B由程序按冻结规则执行全批次校正和固定样本验证。A上综合固定方向Rank IC的幅度与不确定性、ICIR和分阶段表现、"
            "分组收益形态及金融假设；有重复迹象的微弱关系仍可retain，不要求显著。"
            "不得仅因单因子价差不显著、五组不严格单调或尚未做策略端中性化而拒绝送B；"
            "ICIR只作稳定性诊断，不设脱离本批历史分布的硬阈值，稍正的IC点估计也不能单独证明有效。"
            "retain可同时继续优化；证据不足、IC接近零或修改路线受阻且没有明确方向时pause并给恢复条件。"
            "只有充分否定证据或无增量的重复失败才discard，不能仅因一个统计量未达完美而淘汰。"
            "候选direction、公式和程序数值由记录提供；不得生成公式、修改公式、修改评估规则或覆盖程序route_decision。"
            "每个candidate_report必须检查同一公式和方向在horizon_comparison全部期限的Rank IC、分组收益和分阶段差异；"
            "不得只挑最好期限。若有rank_displacement，说明Δ=1、4、24的D均值、中位数、P90、覆盖和分阶段结果。"
            "D是排序变化代理，不是成交或成本；A p值未作多重校正，收益未扣成本与资金费。"
            "保留期限须有完整A数值证据；不要求统计显著，但不能把缺失或未读数据说成已核验。"
            "route_decision是程序对配对检验的唯一状态，不能被模型结论覆盖。A只是开发证据，B准入仍由程序冻结规则完成。"
        )
        payload: dict[str, Any] = {"round_no": round_no, "round_candidate_ids": round_ids,
                                   "review_candidate_ids": review_ids, "report_candidate_ids": report_ids}
        if evaluator_review_ref is not None:
            payload["evaluator_review_ref"] = evaluator_review_ref
        schema = _object_schema({**ROUND_REVIEW_SCHEMA["properties"],
            "candidate_decisions": _array_schema(CANDIDATE_DECISION_SCHEMA,
                minItems=len(review_ids), maxItems=len(review_ids)),
            "candidate_reports": _array_schema(_object_schema({
                "candidate_id": _text_schema("已评估候选ID"), **REPORT_SCHEMA["properties"]}),
                minItems=len(report_ids), maxItems=len(report_ids))})
        result = gateway.ask("evaluator", task, payload, schema,
            validate=lambda response: self._check_round_review(response, review_ids, report_ids))
        return result

    def _interpret(self, cid: str, gateway: AgentGateway) -> dict[str, Any]:
        return gateway.ask("evaluator",
            "依据程序完整评估和公式执行记录，解释方向、幅度、稳定性和可信性。"
            "本请求只补写已保存报告的叙述，不能改变此前Evaluator在research_review记录中的A段候选去向。"
            "B段通过与交付资格由程序按冻结规则计算；B段解释须参考validation_result记录，不能用模型意见覆盖程序结论。"
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
