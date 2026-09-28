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

    def _ask_ideator(self, pending: dict[str, Any]) -> dict[str, Any]:
        return self.gateway.ask("ideator",
            "构造当前有研究依据的一组可证伪候选公式，写清计算含义，不为凑数量生成候选。"
            "阅读全部历史及优化任务，逐条记录采用或放弃。"
            "Optimizer只给出修改任务和配对检验合同；最终候选定义和可执行公式必须由你首次生成。"
            "采用任务时，候选parent_id必须引用control_id，proposal_id必须引用当前任务，"
            "预测方向保持不变；change_reason须具体说明公式如何实现change_target并遵守fixed_components。"
            "不得改写Optimizer预先声明的修改任务或判断标准；无法一致实现时放弃，由Optimizer另立新任务。"
            "遵守candidate_decisions：暂停或淘汰的候选不自动重启；保留但未获准优化的版本不修改。"
            "不可重复已停止路线且不给新增依据。"
            "说明候选公式的跨币比较含义；可考虑收益率、比值或相对变化，程序不按量纲拦截。",
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
