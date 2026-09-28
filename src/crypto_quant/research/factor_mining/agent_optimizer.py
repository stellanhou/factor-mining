"""Optimizer role: model request, candidate decisions, and route checks."""
from __future__ import annotations

import copy
from typing import Any

from .contracts import identifier, modification_plan, require, text, _object_schema, _text_schema, _array_schema, _nullable

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

class OptimizerRole:
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

    def _ask_optimizer(self, round_ids: list[str], review_ids: list[str]) -> dict[str, Any]:
        return self.gateway.ask("optimizer",
            "先阅读本轮所有候选、失败及完整评估，再判断哪里有值得继续的具体依据。"
            "评估Agent只提供证据解释，不作候选去向决定；由你独立综合证据回答四问并决定去向。"
            "先逐个review_candidate_ids回答四个研究问题并决定去向，decisions必须完整且不重复。"
            "研究依据由你根据证据分析，不限于预设现象；引用真实record ID并说明观察与推测。"
            "区分当前版本与修改路线：retain可同时优化；修改路线停止不自动淘汰原版。"
            "去向决策准则：全批次按研究合同指定方法进行FDR多重校正，样本外独立检验是B段程序的专职职责，"
            "不得将‘A段未做全批次FDR、未做正交化或A段非独立样本’作为拒绝送B的理由。"
            "A段判断当前候选是否值得固定送B时，综合预设方向的Rank IC幅度与不确定性、"
            "ICIR和分阶段结果、分组收益形态及金融假设；微弱但有重复迹象的预测关系可以retain。"
            "不得仅因单因子价差未显著、五组不严格单调或尚未做策略端中性化而拒绝送B。"
            "ICIR只是稳定性诊断，不设脱离本批历史分布的硬门槛；IC点估计稍正也不能单独证明有效。"
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
            "判断条件须与experiment_design.metric、最低改善和允许IC损失一致。"
            "control_id必须是获准继续的已评估候选；evidence_refs必须包含该候选的评估记录。"
            "程序配对比较的决定不能覆盖；已停止路线若重新启动必须使用新路线并说明新增依据及旧路线。",
            {"round_candidate_ids": round_ids, "review_candidate_ids": review_ids,
             "candidate_decisions": self.decisions, "route_states": self.routes},
            _object_schema({"analysis": _text_schema("完整研究反馈"),
                "decisions": _array_schema(DECISION_SCHEMA, minItems=len(review_ids), maxItems=len(review_ids)),
                "diagnostics": _array_schema(_text_schema("证据缺口和所需诊断")),
                "proposals": _array_schema(PROPOSAL_SCHEMA)}),
            validate=lambda response: self._check_optimization(response, review_ids))
