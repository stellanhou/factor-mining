from __future__ import annotations

import copy
import json

from crypto_quant.research.factor_mining.contracts import dumps
from crypto_quant.research.factor_mining.model import ModelReply
from crypto_quant.research.factor_mining.workflow import FactorMiner
from test_factor_mining import definition, input_panel, narrative, research_decision, specification


_INFORMATION_REQUEST = {
    "purpose": "补足跨市场相对价格公式的A段历史依据，再决定下一轮构想方向",
    "questions": [
        "已授权A历史中有哪些使用perp_close与spot_close相对变化的候选？它们的公式和A段观察结果是什么？"
    ],
    "record_refs": [],
}


class EmptyCandidateReviewModel:
    """Offline role script for the empty-candidate/evaluator permission path."""

    def __init__(self, *, first_empty_action: str):
        self.first_empty_action = first_empty_action
        self.requests: list[dict] = []
        self.ideations = 0
        self.optimizations = 0

    def complete(self, messages, *, max_output_tokens, session_id):
        request = json.loads(messages[1]["content"])
        self.requests.append(copy.deepcopy(request))
        role = request["role"]

        if role == "ideator":
            self.ideations += 1
            if self.ideations == 1:
                result = {
                    "candidates": [],
                    "dispositions": [],
                    "analysis": "当前公式相似路线缺少A历史依据，先请求具体资料再决定是否构想。",
                }
            else:
                result = {
                    "candidates": [definition("div(perp_close,ts_mean(perp_close))")],
                    "dispositions": [],
                    "analysis": "依据Evaluator要求补充的历史信息提出一个带三小时均值定义的候选。",
                }
        elif role == "calculator":
            payload = request["payload"]
            result = {
                "candidate_id": payload["candidate_id"],
                "repair_expression": "div(perp_close,ts_mean(perp_close,3))",
                "reason": "程序错误显示均值算子缺少窗口参数；按候选含义补齐三小时窗口。",
            }
        elif role == "evaluator":
            payload = request["payload"]
            if not payload["round_candidate_ids"]:
                if self.first_empty_action == "continue_ideation":
                    result = {
                        "round_action": "continue_ideation",
                        "candidate_decisions": [],
                        "information_request": copy.deepcopy(_INFORMATION_REQUEST),
                        "candidate_reports": [],
                        "analysis": "空候选和资料缺口需要补充A历史后再作方向判断。",
                    }
                else:
                    result = {
                        "round_action": "end_direction",
                        "candidate_decisions": [],
                        "information_request": None,
                        "candidate_reports": [],
                        "analysis": "现有A历史不足以形成候选，Evaluator据此结束当前研究方向。",
                    }
            else:
                candidate_ids = payload["review_candidate_ids"]
                result = {
                    "round_action": "end_direction",
                    "candidate_decisions": [research_decision(cid, "retain") for cid in candidate_ids],
                    "information_request": None,
                    "candidate_reports": [
                        {"candidate_id": cid, **narrative()}
                        for cid in payload["report_candidate_ids"]
                    ],
                    "analysis": "候选已完成程序计算和A段评估；保留候选并结束当前方向。",
                }
        elif role == "optimizer":
            self.optimizations += 1
            raise AssertionError("optimizer must not run without evaluator authorization")
        else:
            raise AssertionError(f"unexpected role: {role}")

        return ModelReply(
            dumps({"result": result, "read_records": []}),
            {"fixture": True},
            "offline-factor-ideator-review-flow",
        )


def _roles(model: EmptyCandidateReviewModel) -> list[str]:
    return [request["role"] for request in model.requests]


def test_empty_ideation_requests_evaluator_review_before_retrying_with_readable_evidence(tmp_path):
    spec = specification()
    model = EmptyCandidateReviewModel(first_empty_action="continue_ideation")
    miner = FactorMiner(spec, model, tmp_path / "run")

    result = miner.explore(input_panel(spec))

    assert _roles(model) == ["ideator", "evaluator", "ideator", "calculator", "evaluator"]
    assert model.ideations == 2
    assert model.optimizations == 0

    first_ideation = model.requests[0]
    assert first_ideation["payload"]["ideation_id"] == "round-001-ideation"
    assert "evaluator_review_ref" not in first_ideation["payload"]
    first_review = model.requests[1]["payload"]
    assert first_review["round_candidate_ids"] == []

    second_ideation = model.requests[2]
    expected_ref = {"record_id": "round-001-review", "pointer": "/information_request"}
    assert second_ideation["payload"]["evaluator_review_ref"] == expected_ref
    readable_review = next(
        record for record in second_ideation["records"] if record["id"] == "round-001-review"
    )
    assert readable_review["kind"] == "research_review"
    assert readable_review["data"]["information_request"] == _INFORMATION_REQUEST

    records = {record["id"]: record["data"] for record in miner.store.all()}
    assert records["round-001-review"]["round_action"] == "continue_ideation"
    assert records["round-001-review"]["information_request"] == _INFORMATION_REQUEST
    assert records["candidate-0001-calculation"]["status"] == "computed"
    assert records["candidate-0001-calculation"]["executed_expression"]["expression"] == (
        "div(perp_close,ts_mean(perp_close,3))"
    )
    final_review = records["round-002-review"]
    assert final_review["round_action"] == "end_direction"
    assert final_review["candidate_decisions"][0]["disposition"] == "retain"
    assert final_review["candidate_decisions"][0]["continue_optimization"] is False
    assert result["completed_rounds"] == 2
    assert result["evaluated_ids"] == ["candidate-0001"]
    assert result["retained_ids"] == ["candidate-0001"]
    assert result["stop_reason"] == "evaluator ended the current A research direction"


def test_empty_ideation_ends_only_after_evaluator_returns_end_direction(tmp_path):
    spec = specification()
    model = EmptyCandidateReviewModel(first_empty_action="end_direction")
    miner = FactorMiner(spec, model, tmp_path / "run")

    result = miner.explore(input_panel(spec))

    assert _roles(model) == ["ideator", "evaluator"]
    assert model.ideations == 1
    assert model.optimizations == 0
    review = next(record for record in miner.store.all() if record["id"] == "round-001-review")
    assert review["data"]["round_candidate_ids"] == []
    assert review["data"]["round_action"] == "end_direction"
    assert review["data"]["information_request"] is None
    assert result["completed_rounds"] == 1
    assert result["candidate_ids"] == []
    assert result["stop_reason"] == "evaluator ended the current A research direction"
