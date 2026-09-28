"""Ideate -> compute -> evaluate -> decide -> design an experiment.

Exploration and fixed-batch validation have separate record stores. The latter
cannot repair a formula or send feedback into the exploration loop.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from crypto_quant.features.factor_expressions import compile_expression, evaluate_expression
from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS, validate_universe
from crypto_quant.research.progress import ProgressLog
from .agent_calculator import CalculatorRole
from .agent_evaluator import EvaluatorRole
from .agent_ideator import IdeatorRole
from .agent_optimizer import OptimizerRole
from .contracts import ResearchSpec, digest, require
from .evaluation import build_labels, compare_experiment, correct_batch, evaluate_factor, evaluate_horizon_comparison
from .model import JsonModel
from .records import AgentGateway, RecordStore, write_json
from .runtime import GRAPH_CONFIG
from langgraph.graph import StateGraph, START, END
from .reporting import write_group_plot, write_research_report


def _panel_fingerprint(panel: FactorInputPanel) -> dict[str, Any]:
    values = pd.util.hash_pandas_object(panel.values, index=True).to_numpy().tobytes()
    members = pd.util.hash_pandas_object(panel.universe, index=True).to_numpy().tobytes()
    return {"values_sha256": hashlib.sha256(values).hexdigest(), "membership_sha256": hashlib.sha256(members).hexdigest(),
            "columns": list(panel.values.columns), "rows": len(panel.values),
            "dtypes": {k: str(v) for k, v in panel.values.dtypes.items()}}


def validate_panel(panel: FactorInputPanel, spec: ResearchSpec, stage: str) -> None:
    start, end = spec.bounds(stage)
    members = validate_universe(panel.universe)
    require(members.index.equals(panel.values.index) and members.equals(panel.universe), "panel must have canonical explicit membership")
    hours = members.index.get_level_values("timestamp")
    require(hours.min() <= start - pd.Timedelta(hours=spec.max_lookback_hours), "panel is missing declared expression warm-up hours")
    require(hours.max() == end - pd.Timedelta(hours=1), "panel must stop at the end of its allowed data segment")
    require("perp_open" in panel.values, "24h target requires perpetual opening prices")
    require(set(panel.values.columns) == set(INPUT_COLUMNS), "mining uses exactly the shared input-field contract")


PLAN3_TRACKS = {
    "payout": (1.5, 0.40),
    "stability": (0.3, 0.60),
    "balanced": (1.0, 0.50),
}
TRACK_LABELS = {"payout": "赔率轨", "stability": "胜率轨", "balanced": "均衡轨"}
TAG_LABELS = {"payout": "赔率型", "stability": "稳定型", "balanced": "均衡型"}


def b_admission_reasons(report: dict[str, Any], tests: list[dict[str, Any]],
                        spec: ResearchSpec) -> tuple[list[str], list[str]]:
    require(len(tests) == 1 and tests[0]["metric"] == "rank_ic", "expected one corrected Rank IC test")
    summary = report["summary"]
    reasons = []
    if not tests[0]["rejected"]:
        reasons.append("Rank IC lacks batch-corrected support")
    ic = summary["rank_ic"]["mean"]
    if ic is None or ic * report["direction"] < spec.min_abs_ic:
        reasons.append("directional IC is below the predeclared minimum")
    spread = summary["directional_spread"]["mean"]
    share = summary["positive_stage_share"]
    tracks = []
    if spec.admission_scheme == "plan3":
        if (not reasons or not spec.plan3_tracks_gate) and spread is not None and share is not None:
            spread_bp = spread * 10_000
            tracks = [name for name, (min_spread, min_share) in PLAN3_TRACKS.items()
                      if spread_bp >= min_spread and share >= min_share]
        if spec.plan3_tracks_gate and not tracks:
            reasons.append("no Plan 3 track meets spread and stage-share thresholds")
        if not spec.plan3_tracks_gate and (spread is None or spread <= 0):
            reasons.append("directional spread must be positive")
    else:
        if spread is None or spread < spec.min_directional_spread:
            reasons.append("directional return spread is below the predeclared minimum")
        if (
            summary["valid_stages"] is None or summary["valid_stages"] < 2
            or share is None or share < spec.min_stage_share
        ):
            reasons.append("stage repetition is insufficient")
    return reasons, tracks


class FactorMiner(IdeatorRole, CalculatorRole, EvaluatorRole, OptimizerRole):
    def __init__(self, spec: ResearchSpec, model: JsonModel, output_root: Path):
        self.spec, self.model = spec, model
        self.root = Path(output_root) / spec.run_id
        self.root.mkdir(parents=True, exist_ok=False)
        write_json(self.root / "contract.json", spec.as_dict())
        self._attach()

    @classmethod
    def open(cls, run_directory: Path, model: JsonModel) -> FactorMiner:
        self = cls.__new__(cls)
        self.root, self.model = Path(run_directory), model
        self.spec = ResearchSpec.from_dict(json.loads((self.root / "contract.json").read_text()))
        self._attach()
        return self

    def _attach(self) -> None:
        self.progress = ProgressLog.for_run(self.root)
        self.store = RecordStore(self.root / "a_records")
        self.gateway = AgentGateway(self.model, self.spec, self.store, self.root / "model_calls",
                                    progress=self.progress)
        self.candidates: dict[str, dict[str, Any]] = {}
        self.routes: dict[str, dict[str, Any]] = {}
        self.proposals: dict[str, dict[str, Any]] = {}
        self.decisions: dict[str, dict[str, Any]] = {}

    def _validate_panel(self, panel: FactorInputPanel, stage: str) -> None:
        validate_panel(panel, self.spec, stage)

    def _compile(self, expression: str):
        require(len(expression) <= self.spec.max_formula_nodes * 100, "formula exceeds complexity budget")
        compiled = compile_expression(expression)
        require(sum(1 for _ in ast.walk(compiled.tree)) <= self.spec.max_formula_nodes, "formula exceeds node budget")
        require(compiled.lookback_hours <= self.spec.max_lookback_hours, "formula exceeds declared history budget")
        return compiled


    def explore(self, panel: FactorInputPanel, *, goal_context: dict[str, Any] | None = None) -> dict[str, Any]:
        require(not self.store.all() and not (self.root / "frozen_batch.json").exists(), "exploration can start only in a new research run")
        self._validate_panel(panel, "A")
        universe_path = self.root / "A-universe.csv"
        csv = panel.universe.rename("eligible").to_csv()
        if universe_path.exists():
            require(universe_path.read_text() == csv, "saved A universe differs during initialization recovery")
        else:
            with universe_path.open("x") as handle:
                handle.write(csv)
        catalog = panel.ideation_context()
        catalog["formula_rule"] = ("程序只检查允许的字段、算子、参数、复杂度和数据时间边界，"
                                   "不根据量纲拒绝公式；研究解释应说明跨币比较的经济含义。")
        self.store.append("inputs", "data_provenance", {"fingerprint": _panel_fingerprint(panel), "catalog": catalog})
        if goal_context is not None:
            self.store.append("goal-context", "goal_context", goal_context)
        return self._run_exploration(panel)

    def resume_explore(self, panel: FactorInputPanel) -> dict[str, Any]:
        """Replay committed A checkpoints; never reopen a run after B was frozen."""
        require(not (self.root / "frozen_batch.json").exists(), "cannot resume A after freezing B")
        self._validate_panel(panel, "A")
        records = {r["id"]: r for r in self.store.all()}
        require("inputs" in records, "A input record is missing")
        require((self.root / "A-universe.csv").read_text() == panel.universe.rename("eligible").to_csv(),
                "A membership changed; cannot resume the saved research run")
        self.candidates, self.decisions, self.routes, self.proposals = {}, {}, {}, {}
        return self._run_exploration(panel)

    def _run_exploration(self, panel: FactorInputPanel) -> dict[str, Any]:
        labels = build_labels(panel, self.spec, "A")
        try:
            return self._explore_rounds(panel, labels)
        except Exception as exc:
            # API retries happen inside the gateway. An exhausted or invalid run stops here.
            path = self.root / "exploration-stopped.json"
            if not path.exists():
                write_json(path, {"error_type": type(exc).__name__, "reason": str(exc)})
            raise

    def _explore_rounds(self, panel: FactorInputPanel, labels: pd.DataFrame) -> dict[str, Any]:
        labels_by_horizon = {h: build_labels(panel, self.spec, "A", horizon_hours=h) for h in (1, 4)}
        labels_by_horizon[24] = labels
        saved = {r["id"]: r["data"] for r in self.store.all()}
        context = saved.get("goal-context", {})
        seen: dict[tuple[str, int], str] = {
            (item["expression"], item["direction"]): item["candidate_ref"]
            for item in context.get("previous_expressions", [])}
        def ideate(state):
            round_no = state["round_no"] + 1
            pending = {pid: p for pid, p in self.proposals.items() if p["status"] == "pending"}
            ideation_id = f"round-{round_no:03d}-ideation"
            response = saved[ideation_id] if ideation_id in saved else self._ask_ideator(pending)
            definitions, experiments, self.routes, self.proposals = self._check_ideation(response, pending)
            if ideation_id not in saved:
                self.store.append(ideation_id, "ideation", response)
            return {"round_no": round_no, "definitions": definitions, "experiments": experiments,
                    "round_ids": [], "index": 0}

        def calculate(state):
            index, round_no = state["index"], state["round_no"]
            definition, experiments = state["definitions"][index], state["experiments"]
            cid = f"candidate-{len(self.candidates) + 1:04d}"
            item: dict[str, Any] = {"id": cid, "round": round_no, "definition": definition,
                                   "experiment": experiments.get(index)}
            self.candidates[cid] = item
            state["round_ids"].append(cid)
            if f"{cid}-definition" in saved:
                require(saved[f"{cid}-definition"] == item, "saved candidate differs from accepted ideation")
            else:
                self.store.append(f"{cid}-definition", "candidate", item)
            try:
                key = (self._compile(definition["expression"]).expanded_expression, definition["direction"])
            except ValueError:
                key = None  # Invalid formulas still enter formula diagnosis and repair.
            if key in seen:
                item["duplicate_of"] = seen[key]
                if f"{cid}-duplicate" not in saved:
                    self.store.append(f"{cid}-duplicate", "duplicate", {"candidate_id": cid, "duplicate_of": seen[key]})
                self._finish_uncomputed_route(item)
                return {**state, "index": index + 1, "computed": False}
            if f"{cid}-calculation" in saved:
                item["calculation"] = saved[f"{cid}-calculation"]
                executed = item["calculation"]["executed_expression"]
                values = evaluate_expression(executed["expression"], panel).values if executed else None
            else:
                values = self._calculate(cid, panel)
            if values is None:
                self._finish_uncomputed_route(item)
                return {**state, "index": index + 1, "computed": False}
            expression = item["calculation"]["executed_expression"]["expanded_expression"]
            if (expression, definition["direction"]) in seen:
                item["duplicate_of"] = seen[(expression, definition["direction"])]
                if f"{cid}-duplicate" not in saved:
                    self.store.append(f"{cid}-duplicate", "duplicate", {"candidate_id": cid, "duplicate_of": item["duplicate_of"]})
                self._finish_uncomputed_route(item)
                return {**state, "index": index + 1, "computed": False}
            seen[(expression, definition["direction"])] = cid
            return {**state, "cid": cid, "values": values, "computed": True}

        def evaluate_candidate(state):
            cid, values = state["cid"], state["values"]
            item = self.candidates[cid]
            definition = item["definition"]
            report = (saved[f"{cid}-evaluation"] if f"{cid}-evaluation" in saved else
                      evaluate_factor(values, labels, self.spec, "A", definition["direction"]))
            if f"{cid}-evaluation" not in saved:
                report["horizon_comparison"] = evaluate_horizon_comparison(
                    values, labels_by_horizon, self.spec, definition["direction"])
            item["evaluation"] = report
            if f"{cid}-evaluation" not in saved:
                self.store.append(f"{cid}-evaluation", "evaluation", {"candidate_id": cid, **report})
            plot = self.root / "plots" / f"{cid}-A.svg"
            if not plot.exists():
                write_group_plot(plot, report)
            if item["experiment"]:
                if f"{cid}-comparison" in saved:
                    item["comparison"] = saved[f"{cid}-comparison"]
                    self.routes[item["experiment"]["route_id"]]["decision"] = item["comparison"]["route_decision"]
                else:
                    self._compare(cid, values, panel, labels)
            narrative = saved[f"{cid}-report"] if f"{cid}-report" in saved else self._try_report(cid, self.gateway)
            if narrative is not None:
                item["model_report"] = narrative
            return {**state, "index": state["index"] + 1, "values": None}

        def optimize(state):
            round_ids, round_no = state["round_ids"], state["round_no"]
            review_ids = list(dict.fromkeys(round_ids + [cid for cid, decision in self.decisions.items()
                                                        if decision["continue_optimization"]]))
            optimization_id = f"round-{round_no:03d}-optimization"
            optimization = ({k: v for k, v in saved[optimization_id].items() if k != "route_states"}
                            if optimization_id in saved else self._ask_optimizer(round_ids, review_ids))
            decisions, self.routes, self.proposals = self._check_optimization(optimization, review_ids)
            self.decisions.update(decisions)
            if optimization_id not in saved:
                self.store.append(optimization_id, "optimization", {**optimization, "route_states": self.routes})
            return {**state, "continue_research": bool(optimization["proposals"])}

        def next_candidate(state):
            return "calculate" if state["index"] < len(state["definitions"]) else "optimize"

        graph = StateGraph(dict)
        graph.add_node("ideate", self.progress.track("factor.A.ideate", ideate))
        graph.add_node("calculate", self.progress.track("factor.A.calculate", calculate))
        graph.add_node("evaluate", self.progress.track("factor.A.evaluate", evaluate_candidate))
        graph.add_node("optimize", self.progress.track("factor.A.optimize", optimize))
        graph.add_edge(START, "ideate")
        graph.add_conditional_edges("ideate", next_candidate, ["calculate", "optimize"])
        graph.add_conditional_edges("calculate", lambda state: "evaluate" if state["computed"] else next_candidate(state),
                                    ["evaluate", "calculate", "optimize"])
        graph.add_conditional_edges("evaluate", next_candidate, ["calculate", "optimize"])
        graph.add_conditional_edges("optimize", lambda state: "ideate" if state["continue_research"] else END,
                                    ["ideate", END])
        self.graph = graph.compile()
        final = self.graph.invoke({"round_no": 0}, config=GRAPH_CONFIG)
        completed_rounds = final["round_no"]
        completion = {"completed_rounds": completed_rounds, "candidate_ids": list(self.candidates),
                      "evaluated_ids": [cid for cid, item in self.candidates.items() if "evaluation" in item],
                      "retained_ids": [cid for cid, decision in self.decisions.items() if decision["disposition"] == "retain"],
                      "pending_report_ids": [cid for cid, item in self.candidates.items()
                                             if "evaluation" in item and "model_report" not in item],
                      "candidate_decisions": self.decisions,
                      "routes": self.routes, "proposals": self.proposals,
                      "stop_reason": "optimizer returned no authorized optimization proposals"}
        write_research_report(self.root / "A-report.md", self.spec.run_id, self.spec.purpose, self.store.all(), "A", replace=True)
        path = self.root / "a-complete.json"
        if path.exists():
            require(json.loads(path.read_text()) == completion, "completed A checkpoint differs from saved evidence")
        else:
            write_json(path, completion)
        return completion


    def _compare(self, cid: str, values: pd.Series, panel: FactorInputPanel, labels: pd.DataFrame) -> None:
        item = self.candidates[cid]
        plan = item["experiment"]
        control = self.candidates[plan["control_id"]]
        baseline = evaluate_expression(control["calculation"]["executed_expression"]["expression"], panel).values
        common = values.notna() & baseline.notna()
        # Both candidates are ranked/grouped again on the identical valid assets.
        trial = evaluate_factor(values.where(common), labels, self.spec, "A", item["definition"]["direction"])
        base = evaluate_factor(baseline.where(common), labels, self.spec, "A", control["definition"]["direction"])
        comparison = compare_experiment(trial, base, plan, self.spec)
        comparison["common_asset_observations"] = int(common.sum())
        comparison["trial_on_common_assets"] = trial
        comparison["control_on_common_assets"] = base
        route = self.routes[plan["route_id"]]
        route["decision"] = comparison["decision"]
        comparison["route_decision"] = route["decision"]
        item["comparison"] = comparison
        self.store.append(f"{cid}-comparison", "experiment_result", {"candidate_id": cid, **comparison})

    def _load_candidates(self) -> None:
        self.decisions = {}
        for record in self.store.all():
            data = record["data"]
            if record["kind"] == "candidate":
                self.candidates[data["id"]] = data
        for record in self.store.all():
            if record["kind"] in {"calculation", "evaluation", "model_report"}:
                data = record["data"]
                key = "model_report" if record["kind"] == "model_report" else record["kind"]
                self.candidates[data["candidate_id"]][key] = data
            elif record["kind"] == "optimization":
                self.decisions.update({decision["candidate_id"]: decision for decision in record["data"]["decisions"]})

    def freeze(self, candidate_ids: list[str], validation_universe: pd.Series) -> dict[str, Any]:
        require((self.root / "a-complete.json").exists(), "complete A exploration before freezing B")
        require(candidate_ids and len(set(candidate_ids)) == len(candidate_ids), "freeze a nonempty unique candidate batch")
        validation_universe = validate_universe(validation_universe)
        start, end = self.spec.bounds("B")
        times = validation_universe.index.get_level_values("timestamp")
        require(times.min() == start - pd.Timedelta(hours=self.spec.max_lookback_hours)
                and times.max() == end - pd.Timedelta(hours=1), "freeze the complete B membership and warm-up")
        old = pd.read_csv(self.root / "A-universe.csv")
        old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
        old_members = old.set_index(["timestamp", "symbol"])["eligible"]
        overlap = validation_universe.index[validation_universe.index.get_level_values("timestamp") < start]
        require(set(overlap) <= set(old_members.index), "B warm-up universe contains undeclared historical assets")
        require(np.array_equal(old_members.loc[overlap].to_numpy(), validation_universe.loc[overlap].to_numpy()),
                "B warm-up membership differs from its recorded A history")
        frozen_membership = self.root / "B-membership-frozen.csv"
        membership_csv = validation_universe.rename("eligible").to_csv()
        if frozen_membership.exists():
            require(frozen_membership.read_text() == membership_csv, "B membership changed during freeze recovery")
        else:
            with frozen_membership.open("x") as handle:
                handle.write(membership_csv)
        self._load_candidates()
        definitions = {}
        for cid in candidate_ids:
            require(cid in self.candidates and "evaluation" in self.candidates[cid], "only computed and evaluated candidates can be frozen")
            require(cid in self.decisions and self.decisions[cid]["disposition"] == "retain",
                    "only explicitly retained candidates can be frozen for B")
            item = self.candidates[cid]
            require("model_report" in item, "complete the retained candidate report before freezing B")
            definitions[cid] = {"definition": item["definition"], "executed": item["calculation"]["executed_expression"],
                                "calculation_sha256": digest(item["calculation"]), "a_evaluation": item["evaluation"],
                                "a_model_report": item["model_report"], "a_decision": self.decisions[cid]}
        frozen = {"contract": self.spec.as_dict(), "candidates": definitions,
                  "a_records": {r["id"]: r["sha256"] for r in self.store.all()},
                  "frozen_at": datetime.now(timezone.utc).isoformat()}
        frozen["sha256"] = digest(frozen)
        write_json(self.root / "frozen_batch.json", frozen)
        return {"candidate_ids": candidate_ids, "sha256": frozen["sha256"]}

    def validate(self, load_panel: Callable[[], FactorInputPanel], idea_pool: Path) -> dict[str, Any]:
        already_accessed = (self.root / "b-access-started.json").exists()
        try:
            return self._validate_once(load_panel, idea_pool)
        except Exception as exc:
            if not already_accessed and (self.root / "b-access-started.json").exists():
                write_json(self.root / "validation-stopped.json", {"error_type": type(exc).__name__, "reason": str(exc),
                           "B_status": "access was attempted; inspect existing evidence before any subsequent research"})
            raise

    def _checked_frozen(self) -> tuple[dict[str, Any], str]:
        frozen = json.loads((self.root / "frozen_batch.json").read_text())
        expected_hash = frozen.pop("sha256")
        require(frozen["contract"] == json.loads((self.root / "contract.json").read_text()), "research contract changed")
        require(frozen["contract"] == self.spec.as_dict(), "loaded research contract changed")
        return frozen, expected_hash

    def _validate_once(self, load_panel: Callable[[], FactorInputPanel], idea_pool: Path) -> dict[str, Any]:
        frozen, expected_hash = self._checked_frozen()
        # This marker is committed before even invoking the B data loader.
        write_json(self.root / "b-access-started.json", {"frozen_sha256": expected_hash,
                   "time": datetime.now(timezone.utc).isoformat(), "use": "fixed_batch_validation"})
        panel = load_panel()
        self._validate_panel(panel, "B")
        frozen_membership = self.root / "B-membership-frozen.csv"
        if frozen_membership.exists():
            require(frozen_membership.read_text() == panel.universe.rename("eligible").to_csv(),
                    "B membership changed after the batch was frozen")
        with (self.root / "B-universe.csv").open("x") as handle:
            panel.universe.rename("eligible").to_csv(handle)
        labels = build_labels(panel, self.spec, "B")
        store = RecordStore(self.root / "b_records")
        store.append("inputs", "data_provenance", {"fingerprint": _panel_fingerprint(panel), "catalog": panel.ideation_context()})
        reports = {}
        for cid, item in frozen["candidates"].items():
            store.append(f"{cid}-frozen", "frozen_definition_and_A_evidence", item)
            result = evaluate_expression(item["executed"]["expression"], panel)
            expected = {key: value for key, value in item["executed"].items() if key != "unit"}
            require(result.definition == expected, "B execution differs from frozen formula")
            reports[cid] = evaluate_factor(result.values, labels, self.spec, "B", item["definition"]["direction"])
            store.append(f"{cid}-evaluation", "evaluation", {"candidate_id": cid, **reports[cid]})
            write_group_plot(self.root / "plots" / f"{cid}-B.svg", reports[cid])
        correction = correct_batch(reports, self.spec)
        store.append("batch-correction", "multiple_testing", correction)
        decisions = {}
        for cid, report in reports.items():
            tests = [t for t in correction["tests"] if t["candidate_id"] == cid]
            reasons, tracks = b_admission_reasons(report, tests, self.spec)
            validation_status = "not_passed" if reasons else "passed"
            if self.spec.purpose != "research":
                reasons.append("engineering checks cannot enter the idea pool")
            decisions[cid] = {"validation_status": validation_status, "decision_source": "program",
                              "eligible_for_idea_pool": not reasons, "reasons": reasons, "tests": tests,
                              "admission_scheme": self.spec.admission_scheme, "tracks": tracks}
            store.append(f"{cid}-validation", "validation_result", {"candidate_id": cid, **decisions[cid]})
        checkpoint = {"frozen_sha256": expected_hash, "b_records": {r["id"]: r["sha256"] for r in store.all()}}
        write_json(self.root / "b-numerical-complete.json", {**checkpoint, "sha256": digest(checkpoint)})
        return self._complete_b_reports(idea_pool)

    def complete_reports(self, stage: str, idea_pool: Path) -> dict[str, Any]:
        """Fill missing narratives from saved evidence only; never reload A/B prices."""
        require(stage in {"A", "B"}, "report stage must be A or B")
        require(json.loads((self.root / "contract.json").read_text()) == self.spec.as_dict(),
                "research contract changed")
        if stage == "B":
            return self._complete_b_reports(idea_pool)
        require(not (self.root / "frozen_batch.json").exists(), "A evidence cannot change after freezing B")
        records = self.store.all()
        evaluated = [r["data"]["candidate_id"] for r in records if r["kind"] == "evaluation"]
        require(bool(evaluated), "no saved A evaluations to explain")
        reports = {r["data"]["candidate_id"] for r in records if r["kind"] == "model_report"}
        for cid in evaluated:
            if cid not in reports:
                self._try_report(cid, self.gateway)
        records = self.store.all()
        reports = {r["data"]["candidate_id"] for r in records if r["kind"] == "model_report"}
        pending = [cid for cid in evaluated if cid not in reports]
        write_research_report(self.root / "A-report.md", self.spec.run_id, self.spec.purpose, records, "A", replace=True)
        return {"stage": "A", "status": "reports_pending" if pending else "complete", "pending_report_ids": pending,
                "exploration_complete": (self.root / "a-complete.json").exists()}

    def _complete_b_reports(self, idea_pool: Path) -> dict[str, Any]:
        frozen, expected_hash = self._checked_frozen()
        require((self.root / "b-numerical-complete.json").exists(), "complete B numerical validation before completing reports")
        require((self.root / "b-access-started.json").exists(), "B access marker is missing")
        store = RecordStore(self.root / "b_records")
        records = store.all()
        reports = {r["data"]["candidate_id"]: r["data"] for r in records if r["kind"] == "evaluation"}
        decisions = {r["data"]["candidate_id"]: {k: v for k, v in r["data"].items() if k != "candidate_id"}
                     for r in records if r["kind"] == "validation_result"}
        require(set(reports) == set(decisions) == set(frozen["candidates"]), "B numerical batch is incomplete")
        correction = next(r["data"] for r in records if r["kind"] == "multiple_testing")
        family = correction["family"]
        if family == "one two-sided Rank IC mean test per frozen candidate, including unavailable tests":
            require(correction == correct_batch(reports, self.spec), "B batch correction differs from saved evaluations")
            for cid, report in reports.items():
                tests = [test for test in correction["tests"] if test["candidate_id"] == cid]
                reasons, tracks = b_admission_reasons(report, tests, self.spec)
                status = "not_passed" if reasons else "passed"
                if self.spec.purpose != "research":
                    reasons.append("engineering checks cannot enter the idea pool")
                expected = {"validation_status": status, "decision_source": "program",
                            "eligible_for_idea_pool": not reasons, "reasons": reasons, "tests": tests,
                            "admission_scheme": self.spec.admission_scheme, "tracks": tracks}
                require(decisions[cid] == expected, "B program decision differs from saved evaluation")
        else:
            require(family == "two primary two-sided mean tests per frozen candidate, including unavailable tests",
                    "unknown B correction family")
        completed_path = self.root / "validation.json"
        if completed_path.exists():
            completed = json.loads(completed_path.read_text())
            prior_decisions = {cid: {key: value for key, value in decision.items() if key != "idea_card"}
                               for cid, decision in completed["decisions"].items()}
            require(prior_decisions == decisions and completed["batch_correction"] == correction,
                    "completed validation differs from saved evidence")
        narratives = {r["data"]["candidate_id"]: r for r in records if r["kind"] == "model_report"}
        gateway = AgentGateway(self.model, self.spec, store, self.root / "model_calls", stage="B",
                               progress=self.progress)
        for cid in reports:
            if cid not in narratives:
                self._try_report(cid, gateway)
        records = store.all()
        narratives = {r["data"]["candidate_id"]: r for r in records if r["kind"] == "model_report"}
        pending = [cid for cid in reports if cid not in narratives]
        for cid, decision in decisions.items():
            if decision["eligible_for_idea_pool"] and cid in narratives:
                narrative = narratives[cid]
                card = self._idea_card(cid, frozen["candidates"][cid], reports[cid], narrative["data"], decision,
                                       expected_hash, narrative["created_at"])
                path = Path(idea_pool) / f"{self.spec.run_id}--{cid}.json"
                if path.exists():
                    require(json.loads(path.read_text()) == card, "existing idea card differs from saved research evidence")
                else:
                    write_json(path, card)
                decision["idea_card"] = str(path)
        output = {"frozen_sha256": expected_hash, "batch_correction": correction, "decisions": decisions,
                  "status": "reports_pending" if pending else "complete", "pending_report_ids": pending,
                  "B_use": "used for fixed-candidate selection; not untouched data for subsequent strategy development",
                  "C_use": "reserved; never loaded by factor mining"}
        write_research_report(self.root / "B-report.md", self.spec.run_id, self.spec.purpose, records, "B", replace=True)
        if not pending:
            path = self.root / "validation.json"
            if path.exists():
                require(json.loads(path.read_text()) == output, "completed validation differs from saved evidence")
            else:
                write_json(path, output)
        return output

    def _idea_card(self, cid: str, item: dict[str, Any], report: dict[str, Any], narrative: dict[str, Any],
                   decision: dict[str, Any], frozen_hash: str, report_created_at: str) -> dict[str, Any]:
        definition = item["definition"]
        card = {"id": f"{self.spec.run_id}--{cid}", "source_type": "factor_mining", "status": "research_idea",
                "source": {"run_id": self.spec.run_id, "candidate_id": cid, "date": report_created_at,
                           "research_directory": str(self.root.resolve()), "frozen_batch_sha256": frozen_hash},
                "original_claim": {"formula": item["executed"], "meaning": definition["meaning"],
                                   "hypothesis": definition["hypothesis"], "initial_findings": report["summary"]},
                "economic_mechanism": narrative["mechanism"],
                "market_and_horizon": {"venue": "Binance", "market": "USD-M perpetual", "inputs": "1h", "target": "24h"},
                "data_and_coverage": {"fields": item["executed"]["fields"], "B": report["coverage"],
                                      "universe_provenance": self.spec.universe_provenance,
                                      "data_usage_review": self.spec.data_usage_review},
                "falsification_conditions": narrative["falsifiers"],
                "unverified_assumptions": narrative["limitations"] + ["因子参数未经最优性证明；完整选币、调仓、仓位、成本及风险规则待研究"],
                "next_research_plan": narrative["next_steps"], "applicability": narrative["conditions"],
                "admission_evidence": decision, "multiple_testing": {"method": self.spec.fdr_method,
                                                                    "alpha": self.spec.fdr_alpha},
                "a_research_decision": item["a_decision"],
                "b_validation_status": decision["validation_status"],
                "strategy_validation_status": "not_started"}
        if self.spec.admission_scheme == "plan3":
            names = TRACK_LABELS if self.spec.plan3_tracks_gate else TAG_LABELS
            labels = [names[name] for name in decision["tracks"]]
            if not self.spec.plan3_tracks_gate and not labels:
                labels = ["基础型"]
            card["title"] = definition["name"] + " · " + "、".join(labels)
            card["classification"] = {"scheme": self.spec.admission_scheme, "tracks": decision["tracks"],
                                      "track_labels": labels}
            if not self.spec.plan3_tracks_gate:
                card["classification"]["tracks_gate"] = False
            card["unverified_assumptions"].append(
                "准入规则在既有B区间被使用后修订；该B结果不能充当规则选择后的全新独立验证。")
        return card
