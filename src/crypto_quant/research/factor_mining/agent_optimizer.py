"""Optimizer role: model request, candidate decisions, and route checks."""
from __future__ import annotations

import copy
from typing import Any

from .contracts import (identifier, modification_plan, number, require, text,
                        _object_schema, _text_schema, _array_schema, _nullable)

MODIFICATION_TASK_SCHEMA = _object_schema({
    "core_hypothesis": _text_schema("原候选的核心金融假设"),
    "observed_problem": _text_schema("引用已有证据说明当前版本的具体问题"),
    "modification_hypothesis": _text_schema("为什么指定改动可能改善或更好检验核心假设"),
    "change_target": _text_schema("允许Ideator实现的语义改动目标；不得包含最终可执行公式"),
    "fixed_components": _text_schema("除被检验改动外必须保持不变的定义部分"),
})

_EXPERIMENT_DESIGN_FIELDS = {
    "question": _text_schema("新候选与原候选的配对检验问题"),
    "min_improvement": {"type": "number", "exclusiveMinimum": 0,
                        "description": "配对改善最低要求；rank_displacement时为D_control−D_trial的绝对下降量"},
    "max_ic_loss": {"type": "number", "minimum": 0, "description": "允许的有向IC损失"},
    "expected_outcome": _text_schema("修改假设得到支持时预期出现的可观测变化"),
    "stop_condition": _text_schema("预定改善或IC损失区间上限不足时的停止条件"),
    "pause_condition": _text_schema("证据不精确或缺数据时的暂停条件"),
}

EXPERIMENT_DESIGN_SCHEMA = {"anyOf": [
    _object_schema({**_EXPERIMENT_DESIGN_FIELDS,
        "metric": {"type": "string", "enum": ["rank_ic", "directional_spread"]}}),
    _object_schema({**_EXPERIMENT_DESIGN_FIELDS,
        "metric": {"type": "string", "enum": ["rank_displacement"]},
        "horizon_hours": {"type": "integer", "enum": [1, 4, 24],
                           "description": "配对Rank IC使用的预测期限H，必须显式选择"},
        "displacement_hours": {"type": "integer", "enum": [1, 4, 24],
                               "description": "待降低排名变化D的间隔Δ，必须显式选择"}}),
]}

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
    "retained_horizons": _array_schema(
        {"type": "integer", "enum": [1, 4, 24]}, maxItems=3,
        description="retain时选择有完整A数值证据的送B期限；其他去向填写空列表"),
    "continue_optimization": {"type": "boolean"},
    "answers": _object_schema({key: {**_object_schema({
        "supported": {"type": "boolean"}, "reason": _text_schema(question)}), "description": question}
        for key, question in RESEARCH_QUESTIONS.items()}),
    "evidence_refs": _array_schema(_text_schema("已有records中的精确记录ID，至少包含当前候选的程序结果"), minItems=1),
    "reason": _text_schema("解释当前版本的保留价值、证据是否充分及去向；不能把单个不显著结果当作无效"),
    "resume_condition": _nullable(_text_schema("pause时写明恢复研究所需的新数据或诊断；其他去向为null")),
})

class OptimizerRole:
    @staticmethod
    def _check_retained_horizon_evidence(report: Any, horizon: int, direction: int) -> None:
        require(isinstance(report, dict) and report.get("segment") == "A"
                and type(report.get("horizon_hours")) is int and report["horizon_hours"] == horizon
                and type(report.get("direction")) is int and report["direction"] == direction,
                f"retained {horizon}h requires matching A horizon evidence")
        required_report = {"summary", "coverage", "periods", "stages", "per_symbol"}
        require(required_report <= report.keys(), f"retained {horizon}h A evidence is incomplete")
        summary = report["summary"]
        require(isinstance(summary, dict), f"retained {horizon}h A summary is incomplete")
        summary_fields = {"rank_ic", "directional_spread", "raw_high_low_spread_mean", "ic_direction_share",
                          "spread_direction_share", "group_means", "positive_stage_share", "valid_stages"}
        require(summary_fields <= summary.keys(), f"retained {horizon}h A summary is incomplete")
        inference_fields = {"mean", "std", "mean_std_ratio", "n", "grid_periods", "method", "lags",
                            "confidence", "alternative", "se", "ci", "p_value", "status"}
        for name in ("rank_ic", "directional_spread"):
            metric = summary.get(name)
            require(isinstance(metric, dict) and inference_fields <= metric.keys(),
                    f"retained {horizon}h A {name} evidence is incomplete")
        coverage_fields = {"eligible_observations", "purged_observations", "purged_hours", "status_counts"}
        require(isinstance(report["coverage"], dict) and coverage_fields <= report["coverage"].keys()
                and isinstance(report["periods"], list) and isinstance(report["stages"], list)
                and isinstance(report["per_symbol"], list),
                f"retained {horizon}h A detail evidence is incomplete")

    @staticmethod
    def _check_rank_displacement_evidence(candidate: dict[str, Any], experiment: dict[str, Any]) -> None:
        evaluation = candidate.get("evaluation")
        require(isinstance(evaluation, dict) and evaluation.get("segment") == "A",
                "rank-displacement route requires control A evaluation evidence")
        comparison = evaluation.get("horizon_comparison")
        require(isinstance(comparison, dict) and isinstance(comparison.get("horizons"), dict),
                "rank-displacement route requires A horizon-comparison evidence")
        horizon = experiment["horizon_hours"]
        horizon_report = comparison["horizons"].get(str(horizon))
        OptimizerRole._check_retained_horizon_evidence(
            horizon_report, horizon, candidate["definition"]["direction"])

        displacement = candidate.get("rank_displacement") or evaluation.get("rank_displacement")
        require(isinstance(displacement, dict) and displacement.get("segment") == "A"
                and isinstance(displacement.get("definition_version"), str)
                and bool(displacement["definition_version"])
                and isinstance(displacement.get("deltas"), dict)
                and set(displacement["deltas"]) == {"1", "4", "24"},
                "rank-displacement route requires complete A displacement diagnostics")
        selected = displacement["deltas"][str(experiment["displacement_hours"])]
        require(isinstance(selected, dict)
                and isinstance(selected.get("summary"), dict)
                and isinstance(selected.get("coverage"), dict)
                and isinstance(selected.get("periods"), list)
                and isinstance(selected.get("stages"), list),
                "rank-displacement route requires selected-interval A coverage and stage evidence")
        summary = selected["summary"]
        summary_fields = {"mean", "median", "p90", "valid_periods", "expected_periods", "valid_period_share"}
        require(summary_fields <= summary.keys()
                and type(summary["valid_periods"]) is int and summary["valid_periods"] > 0
                and type(summary["expected_periods"]) is int
                and summary["expected_periods"] >= summary["valid_periods"],
                "rank-displacement route requires available A interval observations")
        for field in ("mean", "median", "p90"):
            number(summary[field], f"rank-displacement {field}")
        share = number(summary["valid_period_share"], "rank-displacement valid_period_share")
        require(0 < share <= 1, "rank-displacement A valid-period share must lie in (0, 1]")

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
            retained_horizons = value["retained_horizons"]
            require(isinstance(retained_horizons, list)
                    and all(type(horizon) is int and horizon in {1, 4, 24} for horizon in retained_horizons)
                    and len(retained_horizons) == len(set(retained_horizons)),
                    "retained_horizons must be a unique list from [1, 4, 24]")
            if disposition == "retain":
                require(bool(retained_horizons), "retain requires at least one retained horizon")
            else:
                require(not retained_horizons, "only retain decisions may specify retained horizons")
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
            if disposition == "retain":
                evaluation_records = [record for record in own_results if record["kind"] == "evaluation"]
                require(len(evaluation_records) == 1
                        and evaluation_records[0]["id"] == f"{cid}-evaluation"
                        and evaluation_records[0]["data"].get("segment") == "A",
                        "retained candidate requires its A evaluation record")
                comparison = evaluation_records[0]["data"].get("horizon_comparison")
                require(isinstance(comparison, dict) and isinstance(comparison.get("horizons"), dict),
                        "retained candidate requires A horizon comparison evidence")
                horizons = comparison["horizons"]
                definition = self.candidates[cid]["definition"]
                for horizon in retained_horizons:
                    self._check_retained_horizon_evidence(horizons.get(str(horizon)), horizon,
                                                          definition["direction"])
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
        if plan["experiment_design"]["metric"] == "rank_displacement":
            self._check_rank_displacement_evidence(
                self.candidates[control], plan["experiment_design"])
            goal_context = next((record["data"] for record in self.store.all()
                                 if record["kind"] == "goal_context"), None)
            target = goal_context["goal"].get("quality_target") if goal_context else None
            if target is not None:
                design = plan["experiment_design"]
                require(design["horizon_hours"] == target["horizon_hours"]
                        and design["displacement_hours"] == target["displacement_hours"]
                        and design["min_improvement"] >= target["min_improvement"]
                        and design["max_ic_loss"] <= target["max_ic_loss"],
                        "rank-displacement proposal must respect the frozen Goal quality target")
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
            "quality_target.horizon_hours是预测期限H，displacement_hours是排名变化间隔Δ；合同默认24h标签不是本Goal期限的证明。"
            "读取本轮和历史A evidence中的horizon_comparison，按每个报告自己的horizon_hours与label核对实际口径。"
            "单版A的horizon_comparison使用共同24h边界、末端剔除25h；rank_displacement配对按所选H标签剔除末端H+1h，D两端限定A内。"
            "配对不支持自定义末端掩码；proposal的fixed_components等文字必须与程序规则一致，不得把25h剔除作为1h/4h配对条件。"
            "历史candidate_ref只能指明公式和方向来源，不能作为当前run的control_id；需要跨轮配对时，"
            "先要求当前run重算该公式/方向作为baseline，再只对已评估的当前候选提出正式proposal。"
            "各历史cycle的route_states彼此独立且属于其run；不得因重算baseline改写历史route状态或沿用其attempt。"
            "restart_of仅可引用当前run已有的暂停/停止route_id，不能引用历史cycle的route_id。"
            "不要仅凭重算baseline继续已暂停/停止的历史修改方向；应选择有当前A新增证据支持的独立假设和新路线。"
            "区分当前版本与修改路线：retain可同时优化；修改路线停止不自动淘汰原版。"
            "去向决策准则：全批次按研究合同指定方法进行FDR多重校正，样本外独立检验是B段程序的专职职责，"
            "不得将‘A段未做全批次FDR、未做正交化或A段非独立样本’作为拒绝送B的理由。"
            "A段判断当前候选是否值得固定送B时，综合预设方向的Rank IC幅度与不确定性、"
            "ICIR和分阶段结果、分组收益形态及金融假设；微弱但有重复迹象的预测关系可以retain。"
            "retain必须输出非空retained_horizons，只能从[1,4,24]选择且每个选择都须有完整的A数值证据；"
            "不要求统计显著，保留微弱但有重复迹象的期限。optimize、pause、discard的retained_horizons必须为空列表。"
            "不得仅因单因子价差未显著、五组不严格单调或尚未做策略端中性化而拒绝送B。"
            "ICIR只是稳定性诊断，不设脱离本批历史分布的硬门槛；IC点估计稍正也不能单独证明有效。"
            "若认为当前候选仍有改进空间，支持在判定retain的同时将continue_optimization设为true继续优化。"
            "证据不足、IC接近零或修改路线受阻缺乏明确方向时pause并写恢复条件；"
            "有充分否定证据或无增量的重复才discard；不得仅因单一统计量未达完美而淘汰。"
            "continue_optimization为true须四个回答均获支持，"
            "并为该候选提供至少一条proposal；false时禁止proposal。"
            "依据不足先给diagnostics，不强行生成修改任务。不得输出新候选定义、名称、最终公式、"
            "公式含义或方向；这些由下一轮Ideator首次生成。"
            "四问中的修改假设与尝试价值要求已有A观察、可能改善的理由及可证伪检验，不要求新修改在首次实验前已被证明有效。"
            "历史diagnostics里的未执行建议不自动成为后续实验的准入前提，也不能要求尚未由Ideator生成的定义提前具有执行结果。"
            "如果A评估包含rank_displacement证据，阅读其中1h、4h、24h的D、覆盖和分阶段结果，"
            "据此判断是否有具体的排序变化问题；D只表示排名变化，不是成交、手续费或实际换手金额。"
            "只有有明确A证据支持的修改任务才可选择metric=rank_displacement；不得默认平滑所有因子。"
            "rank_displacement任务必须显式指定horizon_hours和displacement_hours，二者分别表示预测期限H与排名变化间隔Δ，"
            "各自只能是1、4或24小时；不得从最优结果倒选。该路线min_improvement是D_control−D_trial的绝对下降量，"
            "没有Goal预声明要求时，必须依据A证据给出正值，不得套用示例或统一默认值。"
            "Goal若含quality_target，rank_displacement路线沿用其H、Δ，最低改善不得降低，允许IC损失不得放宽；"
            "研究目标已预声明数值不代表数据能达到，缺乏修改依据仍可暂停，不强造提案。"
            "max_ic_loss保持现有的允许有向IC损失含义。"
            "明确说明预测证据与D的两目标取舍：预测能力保持且D下降、预测增强且D未增、两者同时上升、"
            "稳定但预测证据弱、或覆盖缩小后表面改善；需要覆盖不同时先要求共同样本配对比较。"
            "每条proposal只输出：精确证据引用、原核心假设、已观察问题、修改假设、语义改动目标和固定部分；"
            "以及与原候选的配对检验问题、主指标、最低改善、允许IC损失、预期、停止和暂停条件。"
            "同一假设下可在change_target中指定多个相互关联的语义改动，但不写可执行表达式。"
            "判断条件须与experiment_design.metric、最低改善和允许IC损失一致。"
            "control_id必须是获准继续的已评估候选；evidence_refs必须包含该候选的评估记录。"
            "程序配对比较的decision与route_decision是唯一的路线状态，不可在分析或建议中覆盖、改写或当作模型判定；"
            "已停止路线若重新启动必须使用新路线并说明新增依据及旧路线。",
            {"round_candidate_ids": round_ids, "review_candidate_ids": review_ids,
             "candidate_decisions": self.decisions, "route_states": self.routes},
            _object_schema({"analysis": _text_schema("完整研究反馈"),
                "decisions": _array_schema(DECISION_SCHEMA, minItems=len(review_ids), maxItems=len(review_ids)),
                "diagnostics": _array_schema(_text_schema("证据缺口和所需诊断")),
                "proposals": _array_schema(PROPOSAL_SCHEMA)}),
            validate=lambda response: self._check_optimization(response, review_ids))
