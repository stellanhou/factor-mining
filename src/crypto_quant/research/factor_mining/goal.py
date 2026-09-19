"""Durable outcome goals, owned by the existing optimizer role.

Research decisions see A evidence only. Completion review gets A definitions and
program-issued admission receipts, never B measurements or rejection feedback.
"""
from __future__ import annotations

import fcntl
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS
from .contracts import ResearchSpec, digest, identifier, require, text
from .model import JsonModel
from .records import AgentGateway, RecordStore, write_json
from .workflow import (FactorMiner, _array_schema, _code_fingerprint, _object_schema,
                       _panel_fingerprint, _reject_unsupported_oi_cost_basis,
                       _text_schema, validate_panel)


@dataclass(frozen=True)
class GoalSpec:
    goal_id: str
    objective: str
    target_ideas: int

    def __post_init__(self):
        identifier(self.goal_id)
        text(self.objective, "goal objective")
        require(type(self.target_ideas) is int and self.target_ideas > 0,
                "target_ideas must be an explicitly supplied positive integer")


TASK_SCHEMA = _object_schema({
    "action": {"type": "string", "enum": ["explore_new", "wait_data"]},
    "task": _text_schema("下一项研究任务；只给研究问题和方向，不给最终公式"),
    "reason": _text_schema("相对已有研究的新增价值；等待时说明为什么现有数据无法推进其他方向"),
    "evidence_refs": _array_schema(_text_schema("当前records中的精确记录ID"), minItems=1),
    "dependencies": _array_schema(_object_schema({
        "field": {"type": "string", "enum": list(INPUT_COLUMNS)},
        "min_valid_rows": {"type": "integer", "minimum": 1},
        "reason": _text_schema("所需A段字段及最低有效资产小时观测数的研究依据"),
    })),
})
MATCH_SCHEMA = _object_schema({
    "matches": _array_schema(_object_schema({
        "idea_id": _text_schema("程序提供的合格创意卡ID"),
        "matches_goal": {"type": "boolean"},
        "reason": _text_schema("逐项对照目标文本与A段定义、证据，说明是否满足目标要求"),
    })),
})


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class GoalRunner:
    def __init__(self, root: Path, model: JsonModel):
        self.root, self.model = Path(root), model
        saved = _read(self.root / "goal.json")
        expected = saved.pop("sha256")
        require(digest(saved) == expected, "goal contract changed")
        self.goal = GoalSpec(**saved["goal"])
        self.spec = ResearchSpec.from_dict(saved["research"])
        self.inputs = saved["inputs"]
        self.model_settings = saved["model_settings"]
        require(saved["code"] == _code_fingerprint(), "goal code changed; inspect before resuming")
        self.events = RecordStore(self.root / "events")
        self.research = RecordStore(self.root / "research_records")
        self.receipts = RecordStore(self.root / "completion_records")
        self.state = self.status(self.root)

    @classmethod
    def create(cls, goal: GoalSpec, spec: ResearchSpec, model: JsonModel, output_root: Path,
               *, inputs: dict[str, Any], model_settings: dict[str, Any]) -> GoalRunner:
        root = Path(output_root) / goal.goal_id
        root.mkdir(parents=True, exist_ok=False)
        contract = {"goal": asdict(goal), "research": spec.as_dict(), "inputs": inputs,
                    "model_settings": model_settings, "code": _code_fingerprint()}
        write_json(root / "goal.json", {**contract, "sha256": digest(contract)})
        RecordStore(root / "events").append("event-00000001", "goal_state", {
            "goal_id": goal.goal_id, "status": "active", "phase": "select_task", "cycle": 0,
            "current_run": None, "task": None, "qualified_ideas": [], "error": None})
        return cls(root, model)

    @staticmethod
    def status(root: Path) -> dict[str, Any]:
        require((Path(root) / "events").is_dir(), "goal state directory does not exist")
        records = RecordStore(Path(root) / "events").all()
        require(bool(records), "goal has no saved state")
        return records[-1]["data"]

    def _save(self, **changes) -> None:
        state = {**self.state, **changes}
        sequence = len(self.events.all()) + 1
        self.events.append(f"event-{sequence:08d}", "goal_state", state)
        self.state = state

    def _gateway(self, store: RecordStore) -> AgentGateway:
        return AgentGateway(self.model, replace(self.spec, run_id=self.goal.goal_id),
                            store, self.root / "model_calls")

    def _counts(self, panel: FactorInputPanel) -> dict[str, int]:
        validate_panel(panel, self.spec, "A")
        start, end = self.spec.bounds("A")
        hours = panel.values.index.get_level_values("timestamp")
        rows = panel.values.loc[(hours >= start) & (hours < end) & panel.universe]
        return {field: int(rows[field].notna().sum()) for field in INPUT_COLUMNS}

    def _ready(self, dependencies: list[dict[str, Any]], panel: FactorInputPanel) -> bool:
        counts = self._counts(panel)
        return all(counts[d["field"]] >= d["min_valid_rows"] for d in dependencies)

    def _check_task(self, value: dict[str, Any], panel: FactorInputPanel) -> None:
        require(value["action"] in {"explore_new", "wait_data"}, "unknown goal action")
        for key in ("task", "reason"):
            text(value[key], key)
        _reject_unsupported_oi_cost_basis(
            f"{value['task']} {value['reason']}", value["task"], value["reason"])
        refs = value["evidence_refs"]
        records = {r["id"] for r in self.research.all()}
        require(bool(refs) and len(refs) == len(set(refs)) and all(r in records for r in refs),
                "goal task evidence_refs must use only top-level Goal record IDs from "
                f"allowed_evidence_refs: {sorted(records)}; nested candidate/run record IDs are not valid here")
        dependencies = value["dependencies"]
        require(len({d["field"] for d in dependencies}) == len(dependencies), "duplicate data dependency")
        for dependency in dependencies:
            require(dependency["field"] in INPUT_COLUMNS, "unsupported dependency field")
            require(type(dependency["min_valid_rows"]) is int and dependency["min_valid_rows"] > 0,
                    "dependency needs a positive observation count")
            text(dependency["reason"], "dependency reason")
            start, end = self.spec.bounds("A")
            hours = panel.universe.index.get_level_values("timestamp")
            capacity = int(panel.universe.loc[(hours >= start) & (hours < end)].sum())
            require(dependency["min_valid_rows"] <= capacity,
                    "dependency exceeds the fixed A universe capacity; cannot wait for an impossible count")
        if value["action"] == "wait_data":
            require(bool(dependencies) and not self._ready(dependencies, panel),
                    "waiting requires an unmet, program-checkable data dependency")
        else:
            require(not dependencies,
                    "explore_new requires dependencies=[]; do not list fields that are already available")

    def _select(self, panel: FactorInputPanel) -> None:
        index = len(self.research.all()) + 1
        catalog = {
            "fingerprint": _panel_fingerprint(panel), "catalog": panel.ideation_context(),
            "valid_A_rows": self._counts(panel)}
        matching = [record for record in self.research.all()
                    if record["kind"] == "data_provenance"
                    and record["data"]["fingerprint"] == catalog["fingerprint"]]
        if matching:
            catalog_id = matching[-1]["id"]
        else:
            catalog_id = f"inputs-{index:08d}"
            self.research.append(catalog_id, "data_provenance", catalog)
        decision = self._gateway(self.research).ask("optimizer",
            "你继续担任优化Agent，现在负责Goal层的下一项研究决策，不新增角色。"
            "对照goal和全部A研究记录，选择新的有依据的研究方向，或等待明确缺失的数据。"
            "已有候选的优化仍由候选Loop按四问和配对合同执行。当前候选无修改提案只结束该候选Loop，"
            "不代表Goal完成。尚未想到新方向时继续梳理未回答问题，不能输出结束或完成。"
            "explore_new须说明新增研究价值，不重复已经检验的相同公式；不输出候选定义或最终公式，"
            "交给构想Agent生成。新数据可以支持重查旧问题，但需说明新增依据。"
            "领先-滞后研究必须保持时间对齐语义：corr(当期x, delay(y,n))表示y领先x，"
            "不得把当期x称为领先变量。"
            "严格按catalog字段口径提出研究问题；open_interest_value是当前未平仓名义价值，"
            "与open_interest_base的比值只是当前聚合名义价值与基础币数量之比；不得称为平均名义价格，"
            "也不提供平均开仓成本、平均入场价、持仓成本或浮动盈亏。OI变化只能表示当前聚合未平仓"
            "数量/名义价值变化，不能推断杠杆率、加杠杆/去杠杆、开平仓方向或由哪类参与者造成。"
            "只有缺少具体A段字段观测且无法推进其他方向时才wait_data，列出字段、最低有效观测数及依据。"
            "action=explore_new时dependencies必须严格为空数组[]，不得列出已有或覆盖充足的字段；"
            "dependencies只在action=wait_data且字段当前确实不足时填写。"
            "evidence_refs只能从payload.allowed_evidence_refs原样选择Goal顶层记录ID；"
            "research_cycle内部的candidate、evaluation、calculation或round记录ID不能直接放入evidence_refs。"
            "不要将单个候选暂停当成整个Goal必须等待。B验收规则由程序执行，不能修改或请求B/C结果。",
            {"goal_phase": "select_task", "goal": asdict(self.goal), "catalog_record_id": catalog_id,
             "allowed_evidence_refs": sorted(r["id"] for r in self.research.all())},
            TASK_SCHEMA, validate=lambda value: self._check_task(value, panel))
        self.research.append(f"task-{index:08d}", "goal_task", decision)
        if decision["action"] == "wait_data":
            self._save(status="waiting", phase="wait_data", task=decision)
        else:
            archived_cycles = []
            for record in self.research.all():
                if record["kind"] != "research_cycle":
                    continue
                prefix, separator, sequence = record["id"].partition("-")
                require(prefix == "cycle" and separator and len(sequence) == 8 and sequence.isdigit(),
                        "research cycle record has an invalid identifier")
                archived_cycles.append(int(sequence))
            cycle = max([self.state["cycle"], *archived_cycles]) + 1
            # Keep identifiers independent of the length of the user's goal ID.
            run_id = f"goal-{digest(asdict(self.goal))[:12]}-{cycle:06d}"
            self._save(status="active", phase="explore", cycle=cycle, current_run=run_id, task=decision)

    @staticmethod
    def _cycle_context(record_id: str, data: dict[str, Any]) -> dict[str, Any]:
        source = {record["id"]: record for record in data["records"]}
        definitions = {record["data"]["id"]: record["data"]["definition"]
                       for record in source.values() if record["kind"] == "candidate"}
        decisions = {}
        for record in source.values():
            if record["kind"] == "optimization":
                decisions.update({decision["candidate_id"]: decision
                                  for decision in record["data"]["decisions"]})
        candidates, previous = [], []
        for cid, definition in definitions.items():
            calculation = source.get(f"{cid}-calculation")
            evaluation = source.get(f"{cid}-evaluation")
            comparison = source.get(f"{cid}-comparison")
            executed = calculation["data"]["executed_expression"] if calculation else None
            if executed:
                previous.append({"expression": executed["expanded_expression"],
                                 "direction": definition["direction"],
                                 "candidate_ref": f"{data['run_id']}/{cid}"})
            decision = decisions.get(cid)
            candidates.append({
                "candidate_ref": f"{data['run_id']}/{cid}",
                "definition": definition,
                "calculation": None if calculation is None else {
                    "status": calculation["data"]["status"],
                    "executed_expression": executed,
                },
                "A_evaluation": None if evaluation is None else {
                    "summary": evaluation["data"]["summary"],
                    "coverage": evaluation["data"]["coverage"],
                },
                "final_decision": None if decision is None else {
                    key: decision[key] for key in (
                        "disposition", "continue_optimization", "reason", "resume_condition")
                },
                "paired_comparison": None if comparison is None else {
                    key: comparison["data"][key] for key in (
                        "route_decision", "reason", "paired_ic_change", "paired_improvement")
                },
            })
        return {"source_record_id": record_id, "run_id": data["run_id"],
                "candidates": candidates, "previous_expressions": previous}

    def _context(self, panel: FactorInputPanel) -> dict[str, Any]:
        history = self.research.all()
        fingerprint = _panel_fingerprint(panel)
        previous = []
        cycles = []
        for record in history:
            if record["kind"] != "research_cycle":
                continue
            data = record["data"]
            source = {r["id"]: r for r in data["records"]}
            # With changed A evidence, a new run may re-examine the same formula.
            if source["inputs"]["data"]["fingerprint"] != fingerprint:
                continue
            context = data.get("context") or self._cycle_context(record["id"], data)
            previous.extend(context["previous_expressions"])
            cycles.append({key: value for key, value in context.items() if key != "previous_expressions"})
        tasks = [{"source_record_id": record["id"], **record["data"]}
                 for record in history if record["kind"] == "goal_task"]
        return {"goal": asdict(self.goal), "research_task": self.state["task"],
                "prior_A_research": {"tasks": tasks, "cycles": cycles},
                "previous_expressions": previous}

    def _miner(self) -> FactorMiner:
        miner = FactorMiner.open(self.root / "runs" / self.state["current_run"], self.model)
        require(miner.spec == replace(self.spec, run_id=self.state["current_run"]),
                "research run differs from the saved Goal contract")
        return miner

    def _archive_A(self, miner: FactorMiner) -> None:
        record_id = f"cycle-{self.state['cycle']:08d}"
        data = {"run_id": miner.spec.run_id,
                "records": [r for r in miner.store.all() if r["kind"] != "goal_context"]}
        data["context"] = self._cycle_context(record_id, data)
        existing = {r["id"]: r["data"] for r in self.research.all()}
        if record_id in existing:
            require(existing[record_id] == data, "archived A evidence changed")
        else:
            self.research.append(record_id, "research_cycle", data)

    def _review_completion(self, miner: FactorMiner, validation: dict[str, Any]) -> None:
        ideas = []
        frozen, _ = miner._checked_frozen()
        for cid, decision in validation["decisions"].items():
            if decision["eligible_for_idea_pool"] and "idea_card" in decision:
                card = _read(Path(decision["idea_card"]))
                require(card["b_validation_status"] == "passed" and card["source"]["run_id"] == miner.spec.run_id,
                        "goal admission receipt differs from the idea card")
                ideas.append({"idea_id": card["id"], "definition": frozen["candidates"][cid]["definition"],
                              "a_evaluation": frozen["candidates"][cid]["a_evaluation"],
                              "a_model_report": frozen["candidates"][cid]["a_model_report"],
                              "admitted_by_program": True})
        if not ideas:
            return
        receipt_id = f"admission-{self.state['cycle']:08d}"
        records = {r["id"]: r["data"] for r in self.receipts.all()}
        if receipt_id not in records:
            self.receipts.append(receipt_id, "program_admission", {"ideas": ideas})
        else:
            require(records[receipt_id] == {"ideas": ideas}, "goal admission evidence changed")
        match_id = f"match-{self.state['cycle']:08d}"
        ids = {item["idea_id"] for item in ideas}

        def check(value):
            matches = value["matches"]
            require(len(matches) == len(ids) and {m["idea_id"] for m in matches} == ids,
                    "review each admitted idea exactly once")
            for match in matches:
                require(type(match["matches_goal"]) is bool, "goal match must be boolean")
                text(match["reason"], "goal match reason")

        if match_id in records:
            result = records[match_id]
            check(result)
        else:
            result = self._gateway(self.receipts).ask("optimizer",
                "你是优化Agent，当前只核验已获程序准入的成果是否符合Goal文本。"
                "逐项核对目标要求与候选定义、A证据，输出匹配结论及理由。不能更改Goal或B标准。"
                "程序已检查B准入；这里没有B数值或失败反馈，不得推测这些结果。"
                "此上下文只作成果核验，不能生成或修改后续研究任务。程序按匹配的合格创意数量判断完成。",
                {"goal_phase": "verify_completion", "goal": asdict(self.goal), "idea_ids": sorted(ids)},
                MATCH_SCHEMA, validate=check)
            self.receipts.append(match_id, "goal_match", result)
        qualified = list(dict.fromkeys(self.state["qualified_ideas"] +
                                      [m["idea_id"] for m in result["matches"] if m["matches_goal"]]))
        self._save(qualified_ideas=qualified)

    def _step(self, load_A: Callable[[], FactorInputPanel], load_B: Callable[[], FactorInputPanel],
              load_B_membership: Callable[[], pd.Series], idea_pool: Path) -> None:
        phase = self.state["phase"]
        if phase == "select_task":
            self._select(load_A())
            return
        if phase == "wait_data":
            if self._ready(self.state["task"]["dependencies"], load_A()):
                self._save(status="active", phase="select_task")
            return
        if phase == "explore":
            panel = load_A()
            run_dir = self.root / "runs" / self.state["current_run"]
            if not run_dir.exists():
                miner = FactorMiner(replace(self.spec, run_id=self.state["current_run"]), self.model,
                                    self.root / "runs")
                write_json(miner.root / "model-settings.json", self.model_settings)
            else:
                miner = self._miner()
            if not (miner.root / "a-complete.json").exists():
                if miner.store.all():
                    records = {r["id"] for r in miner.store.all()}
                    if "goal-context" not in records:
                        require(records == {"inputs"}, "partial Goal run is missing its research instructions")
                        require(miner.store.read("inputs", "/fingerprint", 0, 1) == _panel_fingerprint(panel),
                                "A inputs changed during Goal initialization")
                        miner.store.append("goal-context", "goal_context", self._context(panel))
                    miner.resume_explore(panel)
                else:
                    miner.explore(panel, goal_context=self._context(panel))
            self._save(phase="prepare_validation")
            return
        miner = self._miner()
        if phase == "prepare_validation":
            completion = _read(miner.root / "a-complete.json")
            if not completion["retained_ids"]:
                self._save(phase="finish_cycle")
                return
            if not (miner.root / "frozen_batch.json").exists():
                records = miner.store.all()
                reports = {r["data"]["candidate_id"] for r in records if r["kind"] == "model_report"}
                if set(completion["retained_ids"]) - reports:
                    result = miner.complete_reports("A", idea_pool)
                    require(not set(result["pending_report_ids"]) & set(completion["retained_ids"]),
                            "retained A reports remain pending; resume after the reporting error is resolved")
                miner.freeze(completion["retained_ids"], load_B_membership())
            self._save(phase="validate")
        elif phase == "validate":
            if (miner.root / "b-numerical-complete.json").exists():
                validation = miner.complete_reports("B", idea_pool)
            else:
                require(not (miner.root / "b-access-started.json").exists(),
                        "B was accessed without a complete numerical checkpoint; inspect evidence, do not reread B")
                validation = miner.validate(load_B, idea_pool)
            require(validation["status"] == "complete", "B reports remain pending; resume to complete saved reports")
            self._save(phase="finish_cycle")
        elif phase == "finish_cycle":
            self._archive_A(miner)
            if (miner.root / "validation.json").exists():
                # Recheck frozen evidence and card contents before issuing receipts.
                validation = miner.complete_reports("B", idea_pool)
                self._review_completion(miner, validation)
            if len(self.state["qualified_ideas"]) >= self.goal.target_ideas:
                self._save(status="complete", phase="complete", error=None)
            else:
                self._save(status="active", phase="select_task", current_run=None, error=None)
        else:
            raise ValueError(f"unknown goal phase: {phase}")

    def run(self, load_A: Callable[[], FactorInputPanel], load_B: Callable[[], FactorInputPanel],
            load_B_membership: Callable[[], pd.Series], idea_pool: Path, *, poll_seconds: float,
            sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
        require(poll_seconds > 0, "poll_seconds must be positive")
        with (self.root / "runner.lock").open("a") as lock:
            # OS releases this lock on process exit; a second runner must fail.
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state = self.status(self.root)
            if self.state["status"] == "complete":
                return self.state
            self._save(status="waiting" if self.state["phase"] == "wait_data" else "active", error=None)
            try:
                while self.state["status"] != "complete":
                    self._step(load_A, load_B, load_B_membership, Path(idea_pool))
                    if self.state["status"] == "waiting":
                        sleep(poll_seconds)
            except KeyboardInterrupt:
                self._save(status="paused", error=None)
                raise
            except Exception as exc:
                self._save(status="error", error={"type": type(exc).__name__, "reason": str(exc)})
                raise
            return self.state
