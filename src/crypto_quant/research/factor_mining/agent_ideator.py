"""Ideator role: candidate schema, model request, and proposal adoption checks."""
from __future__ import annotations

import copy
from typing import Any

from .contracts import candidate, require, text, _object_schema, _text_schema, _array_schema, _nullable

CANDIDATE_SCHEMA = _object_schema({
    "meaning": {"type": "string", "description": "候选的计算含义；允许留空"},
    **{name: _text_schema(description) for name, description in {
        "name": "因子名称", "expression": "白名单公式",
        "hypothesis": "待检验的金融解释",
        "change_reason": "生成理由；采用修改任务时说明公式如何实现change_target并遵守fixed_components"}.items()},
    "direction": {"type": "integer", "enum": [-1, 1], "description": "预先固定预测方向"},
    "parent_id": _nullable(_text_schema("原候选ID；无父候选时为null")),
    "proposal_id": _nullable(_text_schema("采用的优化建议ID；自主构想时为null")),
})

class IdeatorRole:
    def _check_ideation(self, response: dict[str, Any], pending: dict[str, Any]):
        require(set(response) == {"candidates", "dispositions", "analysis"}, "invalid ideation schema")
        text(response["analysis"], "ideation analysis")
        require(isinstance(response["candidates"], list), "candidates must be a list")
        definitions = [candidate(value) for value in response["candidates"]]
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

    def _ask_ideator(self, pending: dict[str, Any], *, ideation_id: str,
                     factor_landscape_ref: dict[str, str],
                     allowed_record_ids: set[str]) -> dict[str, Any]:
        task = (
            "构造当前有研究依据的一组可证伪候选公式，写清计算含义，不为凑数量生成候选。"
            "表达式须满足合同的节点和历史长度上限；预算超限会记录计算失败，计算Agent不会删项或缩短窗口替你改写研究假设。"
            "阅读全部历史及优化任务，逐条记录采用或放弃。"
            "读取payload.factor_landscape_ref对应的本轮A段因子树和分支证据；树用于提示相似度与研究覆盖，拥挤分支不是禁入条件。"
            "对每个新增候选，在change_reason或analysis中说明与相似已有公式的差异，以及新增的可证伪价值。"
            "尚未尝试的方向须结合字段库存和A研究记录判断；树只描述已有信号，不能凭分支数量推断经济机制空白。"
            "新公式计算前的相似性判断只是字段与表达式层面的假设；跨分支组合的实际相关性须在A段计算后确认，不保证低相关或收益改善。"
            "树不能替代历史记录中的失败原因、A段评价或既有优化任务。"
            "先读取Goal上下文中的prediction_horizon：goal_horizon_hours是预测期限H，displacement_hours是排名变化间隔Δ。"
            "合同默认标签perp_next_open_24h不代表本Goal只评价24h；历史和当前A结果须以horizon_comparison中实际horizon_hours与label为准。"
            "单版A多期限比较共同剔除末端25h；rank_displacement配对使用所选H标签、末端剔除H+1h且D两端在A内。"
            "这是程序固定口径，生成公式不能改变执行掩码；不要把单版25h边界声明为1h/4h配对条件。"
            "若research_task要求以历史候选作配对基准，先把历史表达式和方向原样作为一个当前run的新基线候选生成，parent_id与proposal_id都为null；"
            "程序会在本run重新计算其A证据。旧candidate_ref不能作为当前run的parent_id或control_id。"
            "历史task或diagnostics中的未执行建议不等于已冻结的具体定义；不能要求新定义在首次生成前已有自己的执行证据。"
            "Optimizer随后必须基于这个已评估的本轮候选正式声明proposal，之后才可生成配对child；不能要求程序预先绑定旧父因子或提供尚不存在的配对证明。"
            "历史cycle里的route_states属于各自run，只作决策依据；重算基线不会恢复或改写旧cycle路线。"
            "历史route_id不能填写为本run proposal.restart_of；该字段只引用当前run已停止或暂停的路线。"
            "若历史修改路线已暂停或停止，不要靠重算control沿用该修改方向；改做有新增A证据的独立研究方向。"
            "Optimizer只给出修改任务和配对检验合同；最终候选定义和可执行公式必须由你首次生成。"
            "采用任务时，候选parent_id必须引用control_id，proposal_id必须引用当前任务，"
            "预测方向保持不变；change_reason须具体说明公式如何实现change_target并遵守fixed_components。"
            "不得改写Optimizer预先声明的修改任务或判断标准；无法一致实现时放弃，由Optimizer另立新任务。"
            "对于metric=rank_displacement的pending任务，只生成该明确任务要求的对应版本；不得将平滑自动应用于其他因子，"
            "也不得额外生成未获准的同类窗口变体。其horizon_hours和displacement_hours只是配对评价的H与Δ，"
            "不代表持有期或公式变换，不得擅自改动。"
            "遵守candidate_decisions：暂停或淘汰的候选不自动重启；保留但未获准优化的版本不修改。"
            "不可重复已停止路线且不给新增依据。"
            "说明候选公式的跨币比较含义；可考虑收益率、比值或相对变化，程序不按量纲拦截。"
        )
        payload = {"catalog_record_id": "inputs", "pending_proposals": pending,
                   "candidate_decisions": self.decisions, "ideation_id": ideation_id,
                   "factor_landscape_ref": factor_landscape_ref}
        return self.gateway.ask("ideator",
            task, payload,
            _object_schema({"candidates": _array_schema(CANDIDATE_SCHEMA),
                "dispositions": _array_schema(_object_schema({
                    "proposal_id": _text_schema("每条pending建议的ID"),
                    "action": {"type": "string", "enum": ["adopt", "abandon"]},
                    "reason": _text_schema("采用时说明候选如何落实任务；放弃时说明不能一致实现的原因"),
                    "candidate_index": _nullable({"type": "integer", "minimum": 0})}),
                    minItems=len(pending), maxItems=len(pending)),
                "analysis": _text_schema("本轮构想依据")}),
            validate=lambda response: self._check_ideation(response, pending),
            allowed_record_ids=allowed_record_ids)
