"""Optimizer role: propose controlled modifications for evaluator-authorized candidates."""
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

    def _check_proposals(self, optimization: dict[str, Any], authorized_ids: list[str]):
        require(set(optimization) == {"proposals"}, "optimizer may return modification proposals only")
        require(isinstance(optimization["proposals"], list), "optimizer proposals must be a list")
        proposal_controls = set()
        routes, proposals = copy.deepcopy(self.routes), copy.deepcopy(self.proposals)
        for proposal in optimization["proposals"]:
            require(set(proposal) == set(PROPOSAL_SCHEMA["properties"]), "invalid optimization proposal")
            pid = identifier(proposal["proposal_id"])
            require(pid not in proposals, "proposal IDs must be unique")
            plan = self._proposal_plan(proposal)
            require(plan["control_id"] in authorized_ids,
                    "proposal control is outside evaluator authorization")
            require(self.decisions[plan["control_id"]]["continue_optimization"] is True,
                    "proposal is not authorized by an evaluator candidate decision")
            proposal_controls.add(plan["control_id"])
            self._register_route(plan, routes)
            require(routes[plan["route_id"]]["decision"] == "continue", "cannot propose on a stopped/paused route without new evidence")
            proposals[pid] = {**proposal, "status": "pending"}
        require(proposal_controls == set(authorized_ids),
                "every evaluator-authorized candidate requires a modification proposal")
        return routes, proposals

    @staticmethod
    def _proposal_plan(value: dict[str, Any]) -> dict[str, Any]:
        return modification_plan({name: value[name] for name in MODIFICATION_PLAN_FIELDS})

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

    def _ask_optimizer(self, authorized_ids: list[str]) -> dict[str, Any]:
        return self.gateway.ask("optimizer",
            "只为Evaluator已授权继续优化的control提出修改任务和配对实验计划。"
            "不评判候选是否保留、暂停、淘汰或继续，不改变retained_horizons和Evaluator的candidate_decisions。"
            "不得输出新公式、候选名称、公式含义或方向；这些由Ideator生成。"
            "每个authorized_control_id须恰好有至少一条proposal，control_id必须精确匹配授权候选并引用其A evaluation。"
            "proposal只描述核心假设、观察到的问题、修改假设、语义change_target与fixed_components，以及配对实验设计；"
            "设计须与已有A证据和Goal一致，不得篡改数值指标、降低最低改善要求或放宽允许损失门槛。"
            "rank_displacement须明确写预测期限H和间隔Δ；配对按所选H的下一开盘标签边界剔除末端H+1小时，D两端均须在A内。"
            "单版A期限对照统一剔除末端25小时；不得把该25小时边界套用于1h/4h配对，也不得要求自定义末端掩码。"
            "D只是排名变化代理指标，不是实际换手或成本。历史candidate_ref仅作公式和方向来源；control_id必须是当前run已评估候选。"
            "不得将历史diagnostics中的未执行建议升级成proposal前置条件；可说明当前A证据缺口并设计本次可执行的配对检验。"
            "终止、暂停与保留由Evaluator发出；程序的route_decision保持唯一，不得覆盖或改写。",
            {"authorized_control_ids": authorized_ids,
             "candidate_decisions": {cid: self.decisions[cid] for cid in authorized_ids},
             "route_states": self.routes},
            _object_schema({"proposals": _array_schema(PROPOSAL_SCHEMA, minItems=len(authorized_ids))}),
            validate=lambda response: self._check_proposals(response, authorized_ids))
