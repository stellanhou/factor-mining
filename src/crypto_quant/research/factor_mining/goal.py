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
from crypto_quant.research.progress import ProgressLog
from .contracts import (ResearchSpec, digest, identifier, number, require, text, without_hash_metadata,
                        _array_schema, _object_schema, _text_schema)
from .runtime import GRAPH_CONFIG
from langgraph.graph import StateGraph, START, END
from .model import JsonModel
from .records import (AgentGateway, RecordStore, compact_record, load_record_reference,
                      read_pointer, record_reference, write_json)
from .workflow import FactorMiner, _panel_metadata, validate_panel


@dataclass(frozen=True)
class GoalSpec:
    goal_id: str
    objective: str
    target_ideas: int | None = None
    quality_target: dict[str, Any] | None = None
    max_cycles: int | None = None

    def __post_init__(self):
        identifier(self.goal_id)
        text(self.objective, "goal objective")
        require((self.target_ideas is None) != (self.quality_target is None),
                "declare either target_ideas or quality_target")
        if self.target_ideas is not None:
            require(type(self.target_ideas) is int and self.target_ideas > 0,
                    "target_ideas must be an explicitly supplied positive integer")
            require(self.max_cycles is None, "max_cycles belongs to quality goals")
        else:
            target = self.quality_target
            require(isinstance(target, dict) and set(target) == {
                "metric", "horizon_hours", "displacement_hours", "min_improvement", "max_ic_loss"},
                "quality_target must declare the complete rank-displacement comparison")
            require(target["metric"] == "rank_displacement",
                    "quality goals use paired rank-displacement evidence")
            for key in ("horizon_hours", "displacement_hours"):
                require(type(target[key]) is int and target[key] in {1, 4, 24},
                        f"quality_target {key} must be 1, 4 or 24")
            require(number(target["min_improvement"], "quality minimum improvement") > 0,
                    "quality minimum improvement must be positive")
            require(number(target["max_ic_loss"], "quality maximum IC loss") >= 0,
                    "quality maximum IC loss must be nonnegative")
            require(self.max_cycles is None or (type(self.max_cycles) is int and self.max_cycles > 0),
                    "quality max_cycles must be a positive integer or null for no cycle limit")

    def as_dict(self):
        return {key: value for key, value in asdict(self).items() if value is not None}


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


class GoalResearchStore(RecordStore):
    """Read saved cycle handoffs without loading their archived numeric records."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.index = self.root / "context_index"

    def append(self, record_id: str, kind: str, data: Any) -> str:
        result = super().append(record_id, kind, data)
        if kind == "research_cycle":
            record = {"id": record_id, "kind": kind, "data": data, "sha256": digest(data)}
            write_json(self.index / f"{record_id}.json", compact_record(record))
        return result

    def all(self) -> list[dict[str, Any]]:
        records = []
        for path in sorted(self.root.glob("*.json")):
            if path.name.startswith("cycle-"):
                summary = self.index / path.name
                require(summary.is_file(), f"missing saved cycle context index: {summary}")
                records.append(self._load(summary))
            else:
                records.append(self._load(path))
        return records

    def _source_run(self, data: dict[str, Any]) -> Path:
        run_path = Path(data["records"]["run_path"])
        require(not run_path.is_absolute() and run_path.parts
                and all(part not in {".", ".."} for part in run_path.parts),
                "saved research run path is invalid")
        run = self.root.parent.parent / run_path
        require(run.is_dir() and run.name == data["run_id"], "saved research run is missing")
        return run

    def source_records(self, record_id: str) -> list[dict[str, Any]]:
        cycle = self._load(self.root / f"{identifier(record_id)}.json")["data"]
        run = self._source_run(cycle)
        records = []
        for item in cycle["records"]["items"]:
            record = load_record_reference(run / "a_records", item["ref"])
            require(record["kind"] == item["kind"], "saved research record kind changed")
            records.append(record)
        return records

    def read(self, record_id: str, pointer: str, offset: int, limit: int) -> Any:
        if record_id.startswith("cycle-") and pointer.startswith("/context"):
            return RecordStore(self.index).read(record_id, pointer, offset, limit)
        if record_id.startswith("cycle-") and pointer.startswith("/records"):
            cycle = self._load(self.root / f"{identifier(record_id)}.json")["data"]
            items = cycle["records"]["items"]
            if pointer == "/records":
                return read_pointer(items, "", offset, limit)
            parts = pointer.split("/")
            require(len(parts) >= 3 and parts[2].isdigit(), "use a saved research record index")
            index = int(parts[2])
            require(index < len(items), "saved research record index is out of range")
            item = items[index]
            source_run = self._source_run(cycle)
            record_id = identifier(item["ref"]["record_id"])
            source_record = _read(source_run / "a_records" / f"{record_id}.json")
            require(source_record["id"] == record_id,
                    "referenced research record identifier differs")
            require(source_record["kind"] == item["kind"],
                    "saved research record kind changed")
            if len(parts) >= 4 and parts[3] == "data":
                nested_pointer = "/" + "/".join(parts[4:]) if len(parts) > 4 else ""
                return RecordStore(source_run / "a_records").read(
                    record_id, nested_pointer, offset, limit)
            return read_pointer(source_record, "/" + "/".join(parts[3:]) if len(parts) > 3 else "",
                                offset, limit)
        return super().read(record_id, pointer, offset, limit)


class GoalRunner:
    def __init__(self, root: Path, model: JsonModel):
        self.root, self.model = Path(root), model
        saved = _read(self.root / "goal.json")
        self.spec = ResearchSpec.from_dict(saved["research"])
        self.inputs = saved["inputs"]
        self.events = RecordStore(self.root / "events")
        self.research = GoalResearchStore(self.root / "research_records")
        self.receipts = RecordStore(self.root / "completion_records")
        self.state = self.status(self.root)
        goal = dict(saved["goal"])
        if "max_cycles" in self.state:
            goal["max_cycles"] = self.state["max_cycles"]
        self.goal = GoalSpec(**goal)
        self.model_settings = self.state.get("model_settings", saved["model_settings"])
        self.progress = ProgressLog.for_run(self.root)

    @classmethod
    def create(cls, goal: GoalSpec, spec: ResearchSpec, model: JsonModel, output_root: Path,
               *, inputs: dict[str, Any], model_settings: dict[str, Any]) -> GoalRunner:
        root = Path(output_root) / goal.goal_id
        root.mkdir(parents=True, exist_ok=False)
        contract = {"goal": goal.as_dict(), "research": spec.as_dict(), "inputs": inputs,
                    "model_settings": model_settings}
        write_json(root / "goal.json", contract)
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
                            store, self.root / "model_calls", progress=self.progress)

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
        records = self.research.all()
        index = len(records) + 1
        catalog = {"fingerprint": _panel_metadata(panel),
                   "catalog": panel.ideation_context(), "valid_A_rows": self._counts(panel)}
        catalog_id = f"inputs-{index:08d}"
        self.research.append(catalog_id, "data_provenance", catalog)
        decision = self._gateway(self.research).ask("optimizer",
            "你继续担任优化Agent，现在负责Goal层的下一项研究决策，不新增角色。"
            "对照goal和全部A研究记录，选择新的有依据的研究方向，或等待明确缺失的数据。"
            "已有候选的优化仍由候选Loop按四问和配对合同执行。当前候选无修改提案只结束该候选Loop，"
            "不代表Goal完成。尚未想到新方向时继续梳理未回答问题，不能输出结束或完成。"
            "explore_new须说明新增研究价值，不重复已经检验的相同公式；不输出候选定义或最终公式，"
            "交给构想Agent生成。新数据可以支持重查旧问题，但需说明新增依据。"
            "只有缺少具体A段字段观测且无法推进其他方向时才wait_data，列出字段、最低有效观测数及依据。"
            "action=explore_new时dependencies必须严格为空数组[]，不得列出已有或覆盖充足的字段；"
            "dependencies只在action=wait_data且字段当前确实不足时填写。"
            "evidence_refs只能从payload.allowed_evidence_refs原样选择Goal顶层记录ID；"
            "research_cycle内部的candidate、evaluation、calculation或round记录ID不能直接放入evidence_refs。"
            "不要将单个候选暂停当成整个Goal必须等待。B验收规则由程序执行，不能修改或请求B/C结果。",
            {"goal_phase": "select_task", "goal": self.goal.as_dict(), "catalog_record_id": catalog_id,
             "allowed_evidence_refs": sorted({r["id"] for r in records} | {catalog_id})},
            TASK_SCHEMA, validate=lambda value: self._check_task(value, panel))
        self.research.append(f"task-{index:08d}", "goal_task", decision)
        if decision["action"] == "wait_data":
            self._save(status="waiting", phase="wait_data", task=decision)
        else:
            archived_cycles = []
            for record in records:
                if record["kind"] != "research_cycle":
                    continue
                prefix, separator, sequence = record["id"].partition("-")
                require(prefix == "cycle" and separator and len(sequence) == 8 and sequence.isdigit(),
                        "research cycle record has an invalid identifier")
                archived_cycles.append(int(sequence))
            cycle = max([self.state["cycle"], *archived_cycles]) + 1
            # Keep identifiers independent of the length of the user's goal ID.
            run_id = f"goal-{digest(self.goal.as_dict())[:12]}-{cycle:06d}"
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
            a_evaluation = None
            if evaluation is not None:
                require(evaluation["data"].get("segment") == "A",
                        "Goal cycle context can include A evaluation evidence only")
                displacement = evaluation["data"].get("rank_displacement")
                if displacement is None:
                    displacement_context = {"status": "not_calculated"}
                else:
                    require(isinstance(displacement, dict)
                            and displacement.get("segment") == "A"
                            and isinstance(displacement.get("definition_version"), str)
                            and isinstance(displacement.get("deltas"), dict)
                            and set(displacement["deltas"]) == {"1", "4", "24"},
                            "Goal cycle rank-displacement evidence is incomplete or not A-only")
                    displacement_context = {
                        "status": "available",
                        "definition_version": displacement["definition_version"],
                        "deltas": {delta: {
                            "summary": displacement["deltas"][delta]["summary"],
                            "coverage": displacement["deltas"][delta]["coverage"],
                        } for delta in ("1", "4", "24")},
                    }
                a_evaluation = {
                    "summary": evaluation["data"]["summary"],
                    "coverage": evaluation["data"]["coverage"],
                    "rank_displacement": displacement_context,
                }
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
                "A_evaluation": a_evaluation,
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
        cycles = []
        for record in history:
            if record["kind"] != "research_cycle":
                continue
            context = record["data"]["context"]
            cycles.append({key: value for key, value in context.items() if key != "previous_expressions"})
        tasks = [{"source_record_id": record["id"], **record["data"]}
                 for record in history if record["kind"] == "goal_task"]
        return {"goal": self.goal.as_dict(), "research_task": self.state["task"],
                "prior_A_research": {"tasks": tasks, "cycles": cycles},
                "previous_expressions": []}

    def _miner(self) -> FactorMiner:
        miner = FactorMiner.open(self.root / "runs" / self.state["current_run"], self.model)
        require(miner.spec == replace(self.spec, run_id=self.state["current_run"]),
                "research run differs from the saved Goal contract")
        return miner

    def _archive_A(self, miner: FactorMiner) -> None:
        record_id = f"cycle-{self.state['cycle']:08d}"
        if record_id in {record["id"] for record in self.research.all()}:
            return
        records = [_read(path) for path in sorted(miner.store.root.glob("*.json"))]
        records = [record for record in records if record["kind"] != "goal_context"]
        context = self._cycle_context(record_id, {"run_id": miner.spec.run_id, "records": records})
        run_path = miner.root.resolve().relative_to(self.root.parent.resolve())
        items = []
        for index, record in enumerate(records):
            data = record["data"]
            candidate_id = data.get("candidate_id")
            if record["kind"] == "candidate":
                candidate_id = data["id"]
            item = {"record_id": record["id"], "kind": record["kind"],
                    "candidate_id": candidate_id, "ref": record_reference(record),
                    "pointer": f"/records/{index}"}
            if record["kind"] == "evaluation":
                item["factor_archive_ref"] = data["factor_archive"]
            items.append(item)
        data = {"run_id": miner.spec.run_id, "context": context,
                "candidate_ids": [record["data"]["id"] for record in records
                                  if record["kind"] == "candidate"],
                "records": {"run_path": str(run_path), "items": items}}
        self.research.append(record_id, "research_cycle", data)

    def _quality_evidence(self, miner: FactorMiner, cid: str, retained_horizons: list[int]) -> dict[str, Any]:
        target = self.goal.quality_target
        require(target is not None, "quality evidence requires a quality goal")
        result = {"passed": False, "target": target, "segment": "A"}
        candidate = miner.candidates[cid]
        if candidate["experiment"] is None:
            return {**result, "reason": "candidate has no predeclared paired experiment"}
        ref = {"record_id": f"{cid}-comparison"}
        record = load_record_reference(miner.store.root, ref)
        comparison = record["data"]
        require(record["kind"] == "experiment_result" and comparison["candidate_id"] == cid
                and comparison["plan"] == candidate["experiment"],
                "quality comparison differs from its candidate's predeclared experiment")
        design = comparison["plan"]["experiment_design"]
        if design["metric"] != target["metric"]:
            return {**result, "comparison_ref": ref, "reason": "paired metric differs from quality target"}
        require(comparison["horizon_hours"] == design["horizon_hours"]
                and comparison["displacement_hours"] == design["displacement_hours"],
                "quality comparison intervals differ from its plan")
        result.update(comparison_ref=ref, control_id=comparison["plan"]["control_id"],
                      experiment_design=design, paired_improvement=comparison["paired_improvement"],
                      paired_ic_change=comparison["paired_ic_change"])
        if (design["horizon_hours"] != target["horizon_hours"]
                or design["displacement_hours"] != target["displacement_hours"]
                or target["horizon_hours"] not in retained_horizons
                or design["min_improvement"] < target["min_improvement"]
                or design["max_ic_loss"] > target["max_ic_loss"]):
            return {**result, "reason": "predeclared experiment does not meet the frozen quality target"}
        improvement, ic = comparison["paired_improvement"], comparison["paired_ic_change"]
        if improvement["status"] != "estimated" or ic["status"] != "estimated":
            return {**result, "reason": "paired uncertainty is unavailable"}
        bounds = []
        for metric in (improvement, ic):
            require(isinstance(metric["ci"], list) and len(metric["ci"]) == 2,
                    "estimated quality evidence requires two confidence bounds")
            lower, upper = (number(value, "quality confidence bound") for value in metric["ci"])
            require(lower <= upper, "quality confidence bounds are reversed")
            bounds.append(lower)
        passed = (comparison["route_decision"] == "continue"
                  and bounds[0] >= target["min_improvement"] and bounds[1] >= -target["max_ic_loss"])
        return {**result, "passed": passed,
                "reason": "paired lower bounds meet quality target" if passed
                          else "paired lower bounds do not meet quality target"}

    def _review_completion(self, miner: FactorMiner, validation: dict[str, Any]) -> None:
        require(validation["status"] == "complete", "Goal admission requires complete B reports")
        ideas = []
        delivery_ideas = []
        quality_eligible_ids = set()
        frozen = miner._checked_frozen()
        receipt_id = f"admission-{self.state['cycle']:08d}"
        records = {r["id"]: r["data"] for r in self.receipts.all()}
        b_store = RecordStore(miner.root / "b_records", run_root=miner.root.resolve(),
                              archive_root=miner.archive_root)
        b_records = {record["id"]: record for record in b_store.all()}
        for cid, decision in validation["decisions"].items():
            if not decision["eligible_for_idea_pool"]:
                require("idea_card" not in decision,
                        "ineligible B verdict cannot have an idea card")
                continue
            require("idea_card" in decision, "eligible B verdict is missing its idea card")
            card = _read(Path(decision["idea_card"]))
            item = frozen["candidates"][cid]
            b_evaluation = b_records[f"{cid}-evaluation"]
            b_report = b_records[f"{cid}-report"]
            b_validation = b_records[f"{cid}-validation"]
            require(b_evaluation["kind"] == "evaluation" and b_report["kind"] == "model_report"
                    and b_validation["kind"] == "validation_result",
                    "Goal admission B evidence has unexpected record kinds")
            evaluation = b_evaluation["data"]
            require(b_report["data"]["candidate_id"] == cid,
                    "Goal admission model report belongs to another candidate")
            program = {key: value for key, value in decision.items() if key != "idea_card"}
            saved_program = {key: value for key, value in b_validation["data"].items()
                             if key != "candidate_id"}
            require(b_validation["data"]["candidate_id"] == cid and program == saved_program,
                    "Goal admission verdict differs from the saved B program result")
            require(program["decision_source"] == "program"
                    and program["admission_scheme"] == miner.spec.admission_scheme,
                    "Goal admission verdict has an unexpected source or scheme")
            retained_horizons = item["a_decision"]["retained_horizons"]
            require(retained_horizons == decision["retained_horizons"]
                    and len(retained_horizons) == len(set(retained_horizons))
                    and all(type(h) is int and h in miner.spec.b_horizons for h in retained_horizons),
                    "Goal admission horizons differ from the frozen A decision")
            passed_horizons = decision["passed_horizons"]
            require(len(passed_horizons) == len(set(passed_horizons))
                    and all(type(h) is int and h in retained_horizons for h in passed_horizons)
                    and set(passed_horizons) & set(retained_horizons)
                    and decision["validation_status"] == "passed"
                    and decision["eligible_for_idea_pool"],
                    "Goal admission has no passed A/B horizon intersection")
            require(evaluation["candidate_id"] == cid and evaluation["segment"] == "B"
                    and evaluation["direction"] == item["definition"]["direction"]
                    and evaluation["retained_horizons"] == retained_horizons,
                    "Goal admission evaluation differs from the frozen candidate")
            horizons = evaluation["horizons"]
            require(set(horizons) == {str(h) for h in retained_horizons},
                    "Goal admission evaluations differ from the frozen horizon set")
            tests_by_horizon = {test["horizon_hours"]: test for test in decision["tests"]}
            require(len(tests_by_horizon) == len(decision["tests"])
                    and set(tests_by_horizon) == set(retained_horizons)
                    and all(test["candidate_id"] == cid for test in tests_by_horizon.values()),
                    "Goal admission BH tests differ from the frozen candidate horizons")
            horizon_results = decision["horizon_results"]
            require(set(horizon_results) == {str(h) for h in retained_horizons},
                    "Goal admission verdict omits a frozen horizon")
            require(passed_horizons == [h for h in retained_horizons
                                        if horizon_results[str(h)]["validation_status"] == "passed"],
                    "Goal admission passed horizons differ from per-horizon verdicts")
            factor_identity = miner._factor_identity(
                item["executed"]["expanded_expression"], item["definition"]["direction"]).as_dict()
            factor_archives = {}
            horizon_evidence = {}
            for horizon in retained_horizons:
                result = horizons[str(horizon)]
                locator = result["factor_archive"]
                require(result["segment"] == "B" and result["direction"] == item["definition"]["direction"]
                        and result["horizon_hours"] == horizon
                        and locator["identity"] == factor_identity
                        and locator["evaluation_key"] == miner._evaluation_key("B", horizon).as_dict(),
                        "Goal admission factor archive differs from formula, direction, or horizon")
                horizon_verdict = horizon_results[str(horizon)]
                require(horizon_verdict["tests"] == [tests_by_horizon[horizon]],
                        "Goal admission horizon tests differ from the batch correction")
                factor_archives[str(horizon)] = locator
                horizon_evidence[str(horizon)] = {
                    "summary": result["summary"], "coverage": result["coverage"],
                    "verdict": horizon_verdict, "factor_archive": locator}
            a_evaluation_record_id = identifier(item["a_evaluation_ref"]["record_id"])
            a_evaluation = load_record_reference(miner.store.root,
                                                 {"record_id": a_evaluation_record_id})
            require(a_evaluation["kind"] == "evaluation"
                    and a_evaluation["data"]["candidate_id"] == cid
                    and a_evaluation["data"]["factor_archive"] == item["a_evaluation_archive"],
                    "Goal admission A evaluation reference differs from candidate")
            a_horizons = a_evaluation["data"]["horizon_comparison"]["horizons"]
            require(set(map(str, retained_horizons)) <= set(a_horizons),
                    "Goal admission A evaluation omits a retained horizon")
            a_horizon_evidence = {str(h): a_horizons[str(h)]["summary"]
                                  for h in retained_horizons}
            expected_card = miner._idea_card(cid, item, evaluation, b_report["data"],
                                             program, b_report["created_at"])
            require(without_hash_metadata(card) == without_hash_metadata(expected_card),
                    "Goal admission card differs from its candidate, evaluation, verdict, or model report")
            ideas.append({"idea_id": card["id"], "candidate_id": cid,
                          "run_id": miner.spec.run_id,
                          "run_path": str(miner.root.resolve().relative_to(self.root.resolve())),
                          "definition": item["definition"],
                          "a_evaluation_ref": {"record_id": a_evaluation_record_id},
                          "factor_archive_ref": item["a_evaluation_archive"],
                          "retained_horizons": retained_horizons,
                          "a_horizon_evidence": a_horizon_evidence,
                          "a_model_report": item["a_model_report"],
                          "admitted_by_program": True})
            delivery_ideas.append({"idea_id": card["id"], "candidate_id": cid,
                                   "run_id": miner.spec.run_id,
                                   "card_path": str(Path(decision["idea_card"]).resolve()),
                                   "source": card["source"],
                                   "executed_formula": item["executed"],
                                   "direction": item["definition"]["direction"],
                                   "retained_horizons": retained_horizons,
                                   "passed_horizons": passed_horizons,
                                   "a_evaluation_ref": {"record_id": a_evaluation_record_id},
                                   "factor_archive_ref": item["a_evaluation_archive"],
                                   "b_evaluation_ref": {"record_id": b_evaluation["id"]},
                                   "b_validation_ref": {"record_id": b_validation["id"]},
                                   "horizon_evidence": horizon_evidence,
                                   "admission_evidence": program})
            if self.goal.quality_target is not None:
                quality = self._quality_evidence(miner, cid, retained_horizons)
                ideas[-1]["quality_evidence"] = quality
                delivery_ideas[-1]["quality_evidence"] = quality
                if quality["passed"] and self.goal.quality_target["horizon_hours"] in passed_horizons:
                    quality_eligible_ids.add(card["id"])
        if not ideas:
            return
        require(len({idea["idea_id"] for idea in ideas}) == len(ideas),
                "Goal admission contains duplicate idea cards")
        delivery_path = self.root / "delivery_evidence" / f"admission-{self.state['cycle']:08d}.json"
        delivery_record = {"ideas": delivery_ideas}
        if delivery_path.exists():
            require(_read(delivery_path) == delivery_record,
                    "saved Goal delivery evidence differs from the program admission")
        else:
            write_json(delivery_path, delivery_record)
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
                "只根据Goal中的研究目标，逐卡核对候选定义和A段研究证据是否研究了目标所述问题；不把运行流程、批次数量或收据状态当作研究内容要求。"
                "GoalId标识外层Goal，run_id标识它runs目录中的一次研究运行，二者无需相同；程序已核对运行归属。"
                "数量Goal由程序累计target_ideas；质量Goal由程序核对预声明配对目标与同期限B准入。"
                "不以单张卡或单批数量判断研究内容匹配。质量Goal同时阅读quality_evidence，"
                "不得把只通过B或模型评分高当作质量目标达标。"
                "程序已经核验准入、完整报告和卡片落盘；本请求正在生成Goal匹配收据，不得要求匹配收据预先存在。"
                "不能仅以A段统计不显著判为不匹配；仍须依据目标、候选定义和A证据判断研究内容是否符合目标。"
                "研究内容确实不符合Goal时应判false并说明具体差异；不得无条件判true。不能更改Goal或B标准。"
                "程序已检查B准入；这里没有B数值或失败反馈，不得推测这些结果。"
                "此上下文只作成果核验，不能生成或修改后续研究任务。最终完成条件由程序判断。",
                {"goal_phase": "verify_completion", "goal": self.goal.as_dict(), "idea_ids": sorted(ids)},
                MATCH_SCHEMA, validate=check)
            self.receipts.append(match_id, "goal_match", result)
        qualified = list(dict.fromkeys(self.state["qualified_ideas"] +
                                      [m["idea_id"] for m in result["matches"] if m["matches_goal"]
                                       and (self.goal.quality_target is None
                                            or m["idea_id"] in quality_eligible_ids)]))
        self._save(qualified_ideas=qualified)

    def _phase_actions(self, load_A, load_B, load_B_membership, idea_pool):
        def select_task():
            self._select(load_A())
            return

        def wait_data():
            if self._ready(self.state["task"]["dependencies"], load_A()):
                self._save(status="active", phase="select_task")
            return

        def explore():
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
                        require((miner.root / "A-universe.csv").read_text() == panel.universe.rename("eligible").to_csv(),
                                "A membership changed during Goal initialization")
                        miner.store.append("goal-context", "goal_context", self._context(panel))
                    miner.resume_explore(panel)
                else:
                    miner.explore(panel, goal_context=self._context(panel))
            self._save(phase="prepare_validation")
            return
        def prepare_validation():
            miner = self._miner()
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

        def validate():
            miner = self._miner()
            if (miner.root / "b-numerical-complete.json").exists():
                validation = miner.complete_reports("B", idea_pool)
            else:
                require(not (miner.root / "b-access-started.json").exists(),
                        "B was accessed without a complete numerical checkpoint; inspect evidence, do not reread B")
                validation = miner.validate(load_B, idea_pool)
            require(validation["status"] == "complete", "B reports remain pending; resume to complete saved reports")
            self._save(phase="finish_cycle")

        def finish_cycle():
            miner = self._miner()
            self._archive_A(miner)
            if (miner.root / "validation.json").exists():
                # Recheck frozen evidence and card contents before issuing receipts.
                validation = miner.complete_reports("B", idea_pool)
                self._review_completion(miner, validation)
            achieved = (bool(self.state["qualified_ideas"]) if self.goal.quality_target is not None
                        else len(self.state["qualified_ideas"]) >= self.goal.target_ideas)
            if achieved:
                self._save(status="complete", phase="complete", error=None)
            elif (self.goal.quality_target is not None and self.goal.max_cycles is not None
                  and self.state["cycle"] >= self.goal.max_cycles):
                self._save(status="budget_exhausted", phase="budget_exhausted", error=None)
            else:
                self._save(status="active", phase="select_task", current_run=None, error=None)

        return {"select_task": select_task, "wait_data": wait_data, "explore": explore, "prepare_validation": prepare_validation, "validate": validate, "finish_cycle": finish_cycle}

    def _step(self, load_A, load_B, load_B_membership, idea_pool):
        """Execute one persisted phase (also used for checkpoint recovery checks)."""
        self._phase_actions(load_A, load_B, load_B_membership, idea_pool)[self.state["phase"]]()

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
            if self.state["status"] == "budget_exhausted":
                require(self.goal.quality_target is not None, "budget_exhausted requires a quality Goal")
                if self.goal.max_cycles is not None:
                    return self.state
                self._save(status="active", phase="select_task", current_run=None, error=None)
            self._save(status="waiting" if self.state["phase"] == "wait_data" else "active", error=None)
            try:
                graph = StateGraph(dict)
                actions = self._phase_actions(load_A, load_B, load_B_membership, Path(idea_pool))

                def phase_node(name, action):
                    def execute(state):
                        require(self.state["phase"] == name, "graph phase differs from persisted Goal state")
                        action()
                        return dict(self.state)
                    return execute

                def route(state):
                    if state["status"] in {"complete", "budget_exhausted"}:
                        return END
                    return "poll" if state["status"] == "waiting" else state["phase"]

                def poll(state):
                    sleep(poll_seconds)
                    return dict(self.state)

                for name, action in actions.items():
                    graph.add_node(name, self.progress.track(f"goal.{name}", phase_node(name, action)))
                    graph.add_conditional_edges(name, route, [*actions, "poll", END])
                graph.add_node("poll", self.progress.track("goal.poll", poll))
                graph.add_edge("poll", "wait_data")
                graph.add_conditional_edges(START, lambda state: state["phase"], list(actions))
                self.graph = graph.compile()
                self.graph.invoke(dict(self.state), config=GRAPH_CONFIG)
            except KeyboardInterrupt:
                self._save(status="paused", error=None)
                raise
            except Exception as exc:
                self._save(status="error", error={"type": type(exc).__name__, "reason": str(exc)})
                raise
            return self.state
