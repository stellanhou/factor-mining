"""Ideate -> compute -> evaluate -> decide -> design an experiment.

Exploration and fixed-batch validation have separate record stores. The latter
cannot repair a formula or send feedback into the exploration loop.
"""

from __future__ import annotations

import ast
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from crypto_quant.features.factor_expressions import compile_expression, evaluate_expression
from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS, validate_universe
from crypto_quant.research.progress import ProgressLog
from .agent_calculator import CalculatorRole, FormulaBudgetError
from .agent_evaluator import EvaluatorRole
from .agent_ideator import IdeatorRole
from .agent_optimizer import OptimizerRole
from .contracts import ResearchSpec, dumps, require, without_hash_metadata
from .evaluation import (build_labels, compare_experiment, compare_rank_displacement_experiment,
                         correct_batch, evaluate_factor, evaluate_horizon_comparison,
                         RANK_DISPLACEMENT_VERSION,
                         evaluate_rank_displacement)
from .factor_archive import EvaluationKey, FactorArchive, FactorIdentity
from .model import JsonModel
from .records import AgentGateway, RecordStore, load_record_reference, record_reference, write_json
from .runtime import GRAPH_CONFIG
from langgraph.graph import StateGraph, START, END
from .reporting import write_group_plot, write_research_report


def _panel_metadata(panel: FactorInputPanel) -> dict[str, Any]:
    return {"columns": list(panel.values.columns), "rows": len(panel.values),
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
    if spread is None or spread <= 0:
        reasons.append("directional spread must be positive")
    if spread is not None and share is not None:
        spread_bp = spread * 10_000
        tracks = [name for name, (min_spread, min_share) in PLAN3_TRACKS.items()
                  if spread_bp >= min_spread and share >= min_share]
    return reasons, tracks


def b_candidate_decision(candidate_id: str, report: dict[str, Any], correction: dict[str, Any],
                         spec: ResearchSpec) -> dict[str, Any]:
    candidate_tests = [test for test in correction["tests"]
                       if test["candidate_id"] == candidate_id]
    horizon_results = {}
    passed_horizons, tracks = [], set()
    for horizon in report["retained_horizons"]:
        horizon_report = report["horizons"][str(horizon)]
        horizon_tests = [test for test in candidate_tests if test["horizon_hours"] == horizon]
        reasons, horizon_tracks = b_admission_reasons(horizon_report, horizon_tests, spec)
        status = "not_passed" if reasons else "passed"
        horizon_results[str(horizon)] = {"validation_status": status, "reasons": reasons,
                                         "tests": horizon_tests, "tracks": horizon_tracks}
        if status == "passed":
            passed_horizons.append(horizon)
            tracks.update(horizon_tracks)
    retained_horizons = report["retained_horizons"]
    matched_horizons = [horizon for horizon in retained_horizons if horizon in passed_horizons]
    reasons = [] if matched_horizons else [
        "B-passed horizons do not intersect the A-retained horizons"]
    validation_status = "passed" if matched_horizons else "not_passed"
    if spec.purpose != "research":
        reasons.append("engineering checks cannot enter the idea pool")
    return {"validation_status": validation_status, "decision_source": "program",
            "eligible_for_idea_pool": not reasons, "reasons": reasons,
            "tests": candidate_tests, "admission_scheme": spec.admission_scheme,
            "tracks": [name for name in PLAN3_TRACKS if name in tracks],
            "retained_horizons": retained_horizons,
            "passed_horizons": passed_horizons, "horizon_results": horizon_results}


class FactorMiner(IdeatorRole, CalculatorRole, EvaluatorRole, OptimizerRole):
    def __init__(self, spec: ResearchSpec, model: JsonModel, output_root: Path):
        self.spec, self.model = spec, model
        output_root = Path(output_root).resolve()
        self.root = output_root / spec.run_id
        self.root.mkdir(parents=True, exist_ok=False)
        write_json(self.root / "contract.json", spec.as_dict())
        archive_root = RecordStore._expected_archive_root(self.root.resolve())
        marker = {"format": "one-factor-one-file-v2",
                  "archive_root": os.path.relpath(archive_root, self.root.resolve()),
                  "rank_displacement_version": RANK_DISPLACEMENT_VERSION}
        write_json(self.root / "factor_archive.json", marker)
        self._attach()

    @classmethod
    def open(cls, run_directory: Path, model: JsonModel) -> FactorMiner:
        self = cls.__new__(cls)
        self.root, self.model = Path(run_directory), model
        self.spec = ResearchSpec.from_dict(json.loads((self.root / "contract.json").read_text()))
        self._attach()
        return self

    def _attach(self) -> None:
        resolved_root = self.root.resolve()
        self.archive_root = RecordStore._marked_archive_root(resolved_root)
        require(self.archive_root is not None, "FM-v6 requires a current factor archive marker")
        marker = json.loads((resolved_root / "factor_archive.json").read_text(encoding="utf-8"))
        self.rank_displacement_version = marker.get("rank_displacement_version")
        require(self.rank_displacement_version in {None, RANK_DISPLACEMENT_VERSION},
                "run uses an unsupported rank-displacement definition")
        self.progress = ProgressLog.for_run(self.root)
        self.store = RecordStore(self.root / "a_records", run_root=resolved_root,
                                 archive_root=self.archive_root)
        self.gateway = AgentGateway(self.model, self.spec, self.store, self.root / "model_calls",
                                    progress=self.progress)
        self.candidates: dict[str, dict[str, Any]] = {}
        self.routes: dict[str, dict[str, Any]] = {}
        self.proposals: dict[str, dict[str, Any]] = {}
        self.decisions: dict[str, dict[str, Any]] = {}

    def _factor_identity(self, expression: str, direction: Any) -> FactorIdentity:
        return FactorIdentity(expression, str(direction), "expr-v1")

    def _evaluation_key(self, stage: str, horizon_hours: int) -> EvaluationKey:
        return EvaluationKey(f"{self.spec.run_id}/{stage}", dumps(self.spec.as_dict()),
                             "factor-eval-v1", stage, f"{horizon_hours}h")

    @staticmethod
    def _rank_displacement_summary(report: dict[str, Any]) -> dict[str, Any]:
        require(isinstance(report, dict)
                and report.get("definition_version") == RANK_DISPLACEMENT_VERSION
                and report.get("segment") in {"A", "B"}
                and isinstance(report.get("deltas"), dict)
                and set(report["deltas"]) == {"1", "4", "24"},
                "rank-displacement report is incomplete or uses another definition")
        return {"definition_version": report["definition_version"],
                "segment": report["segment"],
                "deltas": {delta: {"summary": report["deltas"][delta]["summary"],
                                    "coverage": report["deltas"][delta]["coverage"]}
                           for delta in ("1", "4", "24")}}

    def _write_rank_displacement(self, candidate_id: str, stage: str,
                                 report: dict[str, Any], identity: FactorIdentity,
                                 *, source: dict[str, Any]) -> dict[str, Any]:
        require(self.rank_displacement_version == RANK_DISPLACEMENT_VERSION
                and report.get("segment") == stage,
                "rank-displacement report does not match its run definition or segment")
        key = EvaluationKey(f"{self.spec.run_id}/{stage}", dumps(self.spec.as_dict()),
                            self.rank_displacement_version, stage, "rank-displacement")
        archive = FactorArchive.open_for(self.archive_root, identity)
        result = archive.append_evaluation(
            key, report,
            provenance={"run_id": self.spec.run_id, "candidate_id": candidate_id,
                        "segment": stage, "definition_version": report["definition_version"],
                        "expanded_expression": identity.expanded_expression,
                        "direction": identity.direction, "source": source})
        return {"root": os.path.relpath(self.archive_root.resolve(), self.root.resolve()),
                "identity": identity.as_dict(), "evaluation_key": key.as_dict(),
                "evaluation_id": result["evaluation_id"], "value_set_id": result["value_set_id"],
                "factor_value_count": 0}

    def _write_evaluation(self, candidate_id: str, stage: str,
                          report: dict[str, Any], identity: FactorIdentity,
                          value_set_id: str | None = None,
                          factor_value_count: int = 0,
                          definition: dict[str, Any] | None = None) -> dict[str, Any]:
        key = self._evaluation_key(stage, int(report["horizon_hours"]))
        archive = FactorArchive.open_for(self.archive_root, identity)
        result = archive.append_evaluation(
            key, report, value_set_id=value_set_id,
            provenance={"run_id": self.spec.run_id, "candidate_id": candidate_id,
                        "segment": stage, "definition": definition,
                        "expanded_expression": identity.expanded_expression,
                        "direction": identity.direction})
        return {"root": os.path.relpath(self.archive_root.resolve(), self.root.resolve()),
                "identity": identity.as_dict(), "evaluation_key": key.as_dict(),
                "evaluation_id": result["evaluation_id"], "value_set_id": result["value_set_id"],
                "factor_value_count": factor_value_count}

    @staticmethod
    def _evaluation_record_data(candidate_id: str, report: dict[str, Any],
                                locator: dict[str, Any],
                                rank_displacement: dict[str, Any] | None = None,
                                rank_displacement_locator: dict[str, Any] | None = None) -> dict[str, Any]:
        data = {"candidate_id": candidate_id, "segment": report["segment"],
                "direction": report["direction"], "horizon_hours": report["horizon_hours"],
                "summary": report["summary"], "coverage": report["coverage"],
                "factor_archive": locator}
        if rank_displacement is not None:
            require(rank_displacement_locator is not None,
                    "rank-displacement evidence requires an archive locator")
            data["rank_displacement"] = FactorMiner._rank_displacement_summary(rank_displacement)
            data["rank_displacement_archive"] = rank_displacement_locator
        return data

    @staticmethod
    def _multi_horizon_evaluation_record_data(
            candidate_id: str, reports: dict[int, dict[str, Any]],
            locators: dict[int, dict[str, Any]], retained_horizons: list[int],
            rank_displacement: dict[str, Any] | None = None,
            rank_displacement_locator: dict[str, Any] | None = None) -> dict[str, Any]:
        require(bool(reports) and set(reports) == set(locators),
                "each B horizon report requires one archive locator")
        require(bool(retained_horizons) and all(type(horizon) is int for horizon in retained_horizons)
                and len(retained_horizons) == len(set(retained_horizons))
                and set(reports) == set(retained_horizons),
                "B reports and archive locators must match the frozen horizon set")
        first = reports[retained_horizons[0]]
        require(all(report["segment"] == first["segment"] == "B"
                    and report["direction"] == first["direction"]
                    and report["horizon_hours"] == horizon
                    and locators[horizon]["evaluation_key"]["horizon"] == f"{horizon}h"
                    for horizon, report in reports.items()),
                "B horizon reports must share a segment and frozen direction")
        data = {"candidate_id": candidate_id, "segment": "B", "direction": first["direction"],
                "retained_horizons": list(retained_horizons),
                "horizons": {str(horizon): {"summary": reports[horizon]["summary"],
                                             "coverage": reports[horizon]["coverage"],
                                             "factor_archive": locators[horizon]}
                             for horizon in retained_horizons}}
        if rank_displacement is not None:
            require(rank_displacement_locator is not None,
                    "rank-displacement evidence requires an archive locator")
            data["rank_displacement"] = FactorMiner._rank_displacement_summary(rank_displacement)
            data["rank_displacement_archive"] = rank_displacement_locator
        return data

    @staticmethod
    def _landscape_evaluation_context(evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
        if evaluation is None:
            return None
        require(isinstance(evaluation, dict) and evaluation.get("segment") == "A",
                "factor landscape can include A evaluation evidence only")
        result = {"summary": evaluation["summary"], "coverage": evaluation["coverage"]}
        displacement = evaluation.get("rank_displacement")
        if displacement is not None:
            require(isinstance(displacement, dict) and displacement.get("segment") == "A",
                    "factor landscape can include A rank-displacement evidence only")
            result["rank_displacement"] = FactorMiner._rank_displacement_summary(displacement)
        return result

    def _landscape_members(self, round_no: int, saved: dict[str, Any]) -> list[dict[str, Any]]:
        """Collect only candidates reached before this ideation in the A replay."""
        members = []
        record_ids = {record["id"] for record in self.store.all()}
        for cid, item in self.candidates.items():
            if item["round"] >= round_no:
                continue
            require(f"{cid}-definition" in record_ids,
                    "factor landscape source candidate definition record is missing")
            calculation = item.get("calculation")
            if calculation is not None:
                require(f"{cid}-calculation" in record_ids,
                        "factor landscape source calculation record is missing")
            if item.get("evaluation") is not None:
                require(f"{cid}-evaluation" in record_ids,
                        "factor landscape source A evaluation record is missing")
            if "duplicate_of" in item:
                require(f"{cid}-duplicate" in record_ids,
                        "factor landscape source duplicate record is missing")
            if calculation is None:
                calculation_context = {
                    "status": "duplicate" if "duplicate_of" in item else "not_calculated",
                    "executed_expression": None,
                    **({"duplicate_of": item["duplicate_of"]} if "duplicate_of" in item else {}),
                }
            else:
                calculation_context = {
                    "status": calculation["status"],
                    "executed_expression": calculation.get("executed_expression"),
                }
                if calculation.get("status") != "computed" and calculation.get("checks"):
                    calculation_context["checks"] = calculation["checks"]
            decision = self.decisions.get(cid)
            evidence_refs = [f"{cid}-definition"]
            evidence_refs.extend(record_id for record_id in (
                f"{cid}-calculation", f"{cid}-evaluation", f"{cid}-duplicate",
                f"round-{item['round']:03d}-optimization") if record_id in record_ids)
            members.append({
                "candidate_ref": f"{self.spec.run_id}/{cid}",
                "definition": item["definition"],
                "calculation": calculation_context,
                "A_evaluation": self._landscape_evaluation_context(item.get("evaluation")),
                "final_decision": None if decision is None else {
                    key: decision[key] for key in (
                        "disposition", "continue_optimization", "reason", "resume_condition")
                },
                "evidence_refs": evidence_refs,
            })

        goal_context = saved.get("goal-context", {})
        require(isinstance(goal_context, dict), "saved Goal context must be an object")
        prior = goal_context.get("prior_A_research", {})
        require(isinstance(prior, dict), "saved prior A research context must be an object")
        cycles = prior.get("cycles", [])
        require(isinstance(cycles, list), "prior A research cycles must be a complete saved list")
        for cycle_index, cycle in enumerate(cycles):
            require(isinstance(cycle, dict) and "candidates" in cycle
                    and isinstance(cycle["candidates"], list),
                    "prior A research cycle candidates are invalid")
            for candidate_index, candidate in enumerate(cycle["candidates"]):
                require(isinstance(candidate, dict) and isinstance(candidate.get("candidate_ref"), str),
                        "prior A candidate reference is invalid")
                require("definition" in candidate and "calculation" in candidate
                        and "A_evaluation" in candidate and "final_decision" in candidate,
                        "prior A candidate summary is incomplete for the factor landscape")
                member = {"candidate_ref": candidate["candidate_ref"],
                          "definition": candidate["definition"],
                          "calculation": candidate["calculation"],
                          "A_evaluation": candidate["A_evaluation"],
                          "final_decision": candidate["final_decision"],
                          "evidence_refs": [
                              f"goal-context#/prior_A_research/cycles/{cycle_index}/candidates/{candidate_index}",
                              candidate["candidate_ref"],
                          ]}
                if "paired_comparison" in candidate:
                    member["paired_comparison"] = candidate["paired_comparison"]
                members.append(member)
        return members

    def _has_ideator_transcript(self, ideation_id: str) -> bool:
        for path in sorted((self.root / "model_calls").glob("*-request.json")):
            request = json.loads(path.read_text(encoding="utf-8"))
            for message in request["messages"]:
                if message["role"] != "user":
                    continue
                content = json.loads(message["content"])
                if (content.get("role") == "ideator"
                        and content.get("payload", {}).get("ideation_id") == ideation_id):
                    return True
        return False

    def _landscape_snapshot(self, panel: FactorInputPanel, round_no: int,
                            ideation_id: str, saved: dict[str, Any]) -> tuple[dict[str, str], set[str]]:
        record_id = f"round-{round_no:03d}-factor-landscape"
        members = self._landscape_members(round_no, saved)
        source_refs = [member["candidate_ref"] for member in members]
        existing = next((record for record in self.store.all() if record["id"] == record_id), None)
        if existing is not None:
            data = existing["data"]
            require(data.get("ideation_id") == ideation_id and data.get("round_no") == round_no,
                    "saved factor landscape belongs to a different ideation round")
            require(data.get("source_candidate_refs") == source_refs,
                    "saved factor landscape source candidates differ from the replayed A prefix")
            require(isinstance(data.get("context_record_ids"), list)
                    and len(data["context_record_ids"]) == len(set(data["context_record_ids"])),
                    "saved factor landscape context record list is invalid")
            return {"record_id": record_id, "pointer": "/snapshot"}, set(data["context_record_ids"])

        require(not self._has_ideator_transcript(ideation_id),
                "ideator transcript exists without its pre-request factor landscape snapshot")
        context_record_ids = [record["id"] for record in self.store.all()]
        from .landscape import build_factor_landscape
        snapshot = build_factor_landscape(panel, self.spec, members)
        data = {"round_no": round_no, "ideation_id": ideation_id,
                "source_segment": "A", "source_candidate_refs": source_refs,
                "context_record_ids": context_record_ids, "snapshot": snapshot}
        self.store.append(record_id, "factor_landscape", data)
        saved[record_id] = data
        return {"record_id": record_id, "pointer": "/snapshot"}, set(context_record_ids)

    def _ideator_visible_record_ids(self, landscape_record_id: str,
                                    context_record_ids: set[str]) -> set[str]:
        """Use the A record prefix frozen before this landscape was saved."""
        visible = set(context_record_ids) | {landscape_record_id}
        known = {record["id"] for record in self.store.all()}
        require(visible <= known, "factor landscape references missing A context records")
        return visible

    def _validate_panel(self, panel: FactorInputPanel, stage: str) -> None:
        validate_panel(panel, self.spec, stage)

    def _compile(self, expression: str):
        if len(expression) > self.spec.max_formula_nodes * 100:
            raise FormulaBudgetError("formula exceeds complexity budget")
        compiled = compile_expression(expression)
        if sum(1 for _ in ast.walk(compiled.tree)) > self.spec.max_formula_nodes:
            raise FormulaBudgetError("formula exceeds node budget")
        if compiled.lookback_hours > self.spec.max_lookback_hours:
            raise FormulaBudgetError("formula exceeds declared history budget")
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
        self.store.append("inputs", "data_provenance", {"panel": _panel_metadata(panel), "catalog": catalog})
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
            if ideation_id in saved:
                response = saved[ideation_id]
                landscape_record_id = f"round-{round_no:03d}-factor-landscape"
                if landscape_record_id in saved:
                    self._landscape_snapshot(panel, round_no, ideation_id, saved)
                else:
                    require(not self._has_ideator_transcript(ideation_id),
                            "ideator transcript exists without its pre-request factor landscape snapshot")
            else:
                landscape_ref, context_record_ids = self._landscape_snapshot(
                    panel, round_no, ideation_id, saved)
                visible_ids = self._ideator_visible_record_ids(landscape_ref["record_id"], context_record_ids)
                response = self._ask_ideator(
                    pending, ideation_id=ideation_id,
                    factor_landscape_ref=landscape_ref,
                    allowed_record_ids=visible_ids)
            definitions, experiments, self.routes, self.proposals = self._check_ideation(response, pending)
            if ideation_id not in saved:
                self.store.append(ideation_id, "ideation", response)
                saved[ideation_id] = response
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
                expression = item["calculation"]["executed_expression"]["expanded_expression"]
                identity = self._factor_identity(expression, definition["direction"])
                values_artifact = item["calculation"]["values_artifact"]
                displacement = None
                displacement_locator = None
                if self.rank_displacement_version is not None:
                    displacement = evaluate_rank_displacement(values, panel.universe, self.spec, "A")
                    require(displacement["definition_version"] == self.rank_displacement_version,
                            "rank-displacement implementation differs from the run marker")
                    displacement_locator = self._write_rank_displacement(
                        cid, "A", displacement, identity,
                        source={"calculation_artifact": values_artifact})
                    item["rank_displacement"] = displacement
                    item["rank_displacement_archive"] = displacement_locator
                locator = self._write_evaluation(
                    cid, "A", report, identity, value_set_id=values_artifact["value_set_id"],
                    factor_value_count=values_artifact["rows"], definition=definition)
                self.store.append(f"{cid}-evaluation", "evaluation",
                                  self._evaluation_record_data(
                                      cid, report, locator, displacement, displacement_locator))
            elif self.rank_displacement_version is not None:
                require("rank_displacement" in report and "rank_displacement_archive" in report,
                        "new-definition A evaluation is missing saved rank-displacement evidence")
                require(report["rank_displacement"]["definition_version"] == self.rank_displacement_version,
                        "saved A rank-displacement evidence differs from the run marker")
                item["rank_displacement"] = report["rank_displacement"]
                item["rank_displacement_archive"] = report["rank_displacement_archive"]
            for horizon, horizon_report in report["horizon_comparison"]["horizons"].items():
                plot = self.root / "plots" / f"{cid}-A-{horizon}h.svg"
                if not plot.exists():
                    write_group_plot(plot, horizon_report)
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
        design = plan["experiment_design"]
        if design["metric"] == "rank_displacement":
            require(self.rank_displacement_version == RANK_DISPLACEMENT_VERSION,
                    "rank-displacement experiments require the declared evaluator definition")
            require(item["definition"]["direction"] == control["definition"]["direction"],
                    "rank-displacement comparisons keep the frozen direction")
            comparison_labels = build_labels(
                panel, self.spec, "A", horizon_hours=design["horizon_hours"])
            comparison = compare_rank_displacement_experiment(
                values, baseline, comparison_labels, self.spec,
                item["definition"]["direction"], plan, universe=panel.universe)
        else:
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
            evaluation = self.store._load(self.store.root / f"{cid}-evaluation.json")
            require(evaluation["data"] == {"candidate_id": cid, **item["evaluation"]},
                    "saved A evaluation differs from the frozen candidate")
            retained_horizons = self.decisions[cid]["retained_horizons"]
            require(isinstance(retained_horizons, list) and bool(retained_horizons)
                    and all(type(horizon) is int for horizon in retained_horizons)
                    and len(retained_horizons) == len(set(retained_horizons))
                    and set(retained_horizons) <= set(self.spec.b_horizons),
                    "A retained horizons must be a nonempty unique subset of the B contract")
            a_horizons = evaluation["data"]["horizon_comparison"]["horizons"]
            require(set(a_horizons) == {str(horizon) for horizon in self.spec.b_horizons},
                    "complete A horizon comparison evidence before freezing B")
            a_horizon_evidence = {}
            for horizon in retained_horizons:
                report = a_horizons[str(horizon)]
                require(report["horizon_hours"] == horizon and
                        {"summary", "coverage", "periods", "stages", "per_symbol"} <= report.keys(),
                        "retained A horizon evidence is incomplete")
                fields = ("horizon_hours", "label", "direction", "summary", "coverage",
                          "sample_hours", "grouping")
                a_horizon_evidence[str(horizon)] = {field: report[field] for field in fields}
            frozen_candidate = {"definition": item["definition"],
                                "executed": item["calculation"]["executed_expression"],
                                "a_evaluation_ref": record_reference(evaluation),
                                "a_evaluation_archive": evaluation["data"]["factor_archive"],
                                "a_evaluation_summary": {"summary": evaluation["data"]["summary"],
                                                          "coverage": evaluation["data"]["coverage"]},
                                "retained_horizons": retained_horizons,
                                "a_horizon_evidence": a_horizon_evidence,
                                "a_model_report": item["model_report"], "a_decision": self.decisions[cid]}
            if self.rank_displacement_version is not None:
                displacement = evaluation["data"].get("rank_displacement")
                displacement_locator = evaluation["data"].get("rank_displacement_archive")
                require(displacement is not None and displacement_locator is not None,
                        "freeze requires the saved A rank-displacement evidence")
                frozen_candidate.update({
                    "rank_displacement_version": self.rank_displacement_version,
                    "a_rank_displacement": self._rank_displacement_summary(displacement),
                    "a_rank_displacement_archive": displacement_locator,
                })
            definitions[cid] = frozen_candidate
        frozen = {"contract": self.spec.as_dict(), "candidates": definitions,
                  "frozen_at": datetime.now(timezone.utc).isoformat()}
        write_json(self.root / "frozen_batch.json", frozen)
        return {"candidate_ids": candidate_ids}

    def validate(self, load_panel: Callable[[], FactorInputPanel], idea_pool: Path) -> dict[str, Any]:
        already_accessed = (self.root / "b-access-started.json").exists()
        try:
            return self._validate_once(load_panel, idea_pool)
        except Exception as exc:
            if not already_accessed and (self.root / "b-access-started.json").exists():
                write_json(self.root / "validation-stopped.json", {"error_type": type(exc).__name__, "reason": str(exc),
                           "B_status": "access was attempted; inspect existing evidence before any subsequent research"})
            raise

    def _checked_frozen(self) -> dict[str, Any]:
        frozen = json.loads((self.root / "frozen_batch.json").read_text())
        require(frozen["contract"] == json.loads((self.root / "contract.json").read_text()), "research contract changed")
        require(frozen["contract"] == self.spec.as_dict(), "loaded research contract changed")
        self._load_candidates()
        for cid, item in frozen["candidates"].items():
            require(cid in self.candidates and "calculation" in self.candidates[cid],
                    "frozen candidate is missing its saved A calculation")
            candidate = self.candidates[cid]
            require(item["definition"] == candidate["definition"],
                    "frozen candidate definition differs from its A record")
            require(item["executed"] == candidate["calculation"]["executed_expression"],
                    "frozen formula differs from its A calculation")
            require(item["a_model_report"] == candidate.get("model_report"),
                    "frozen A report differs from its A record")
            require(item["a_decision"] == self.decisions.get(cid),
                    "frozen A decision differs from its A record")
            retained_horizons = item["retained_horizons"]
            require(retained_horizons == item["a_decision"]["retained_horizons"]
                    and isinstance(retained_horizons, list) and bool(retained_horizons)
                    and all(type(horizon) is int for horizon in retained_horizons)
                    and len(retained_horizons) == len(set(retained_horizons))
                    and set(retained_horizons) <= set(self.spec.b_horizons),
                    "frozen horizons differ from the A decision or B contract")
            require("a_evaluation_ref" in item and item.get("a_evaluation_archive") is not None,
                    "FM-v6 frozen batches require A archive references")
            record = load_record_reference(self.store.root, item["a_evaluation_ref"])
            require(record["id"] == f"{cid}-evaluation" and record["data"]["candidate_id"] == cid,
                    "frozen A evaluation reference differs from candidate")
            locator = item["a_evaluation_archive"]
            identity = locator["identity"]
            key = locator["evaluation_key"]
            require(identity["expanded_expression"] == item["executed"]["expanded_expression"]
                    and identity["direction"] == str(item["definition"]["direction"]),
                    "frozen factor identity differs from its formula")
            require(identity == self._factor_identity(
                item["executed"]["expanded_expression"], item["definition"]["direction"]).as_dict()
                    and key == self._evaluation_key("A", 24).as_dict(),
                    "frozen A evaluation version differs from its run")
            require(record["data"]["factor_archive"] == locator,
                    "frozen A archive IDs differ from the saved evaluation")
            require({"summary": record["data"]["summary"], "coverage": record["data"]["coverage"]}
                    == item["a_evaluation_summary"],
                    "frozen A summary differs from the saved evaluation")
            if item.get("rank_displacement_version") is not None:
                displacement = record["data"].get("rank_displacement")
                displacement_locator = record["data"].get("rank_displacement_archive")
                require(item["rank_displacement_version"] == RANK_DISPLACEMENT_VERSION
                        and displacement is not None and displacement_locator is not None
                        and item["a_rank_displacement_archive"] == displacement_locator
                        and displacement_locator["identity"] == locator["identity"]
                        and item["a_rank_displacement"] == self._rank_displacement_summary(displacement)
                        and displacement.get("definition_version") == item["rank_displacement_version"],
                        "frozen A rank-displacement evidence differs from its archived evaluation")
                expected_displacement_key = EvaluationKey(
                    f"{self.spec.run_id}/A", dumps(self.spec.as_dict()),
                    item["rank_displacement_version"], "A", "rank-displacement").as_dict()
                require(displacement_locator["evaluation_key"] == expected_displacement_key,
                        "frozen A rank-displacement archive version differs from its run")
            else:
                require(self.rank_displacement_version is None
                        and "rank_displacement" not in record["data"],
                        "legacy frozen A evidence cannot be backfilled")
            a_horizons = record["data"]["horizon_comparison"]["horizons"]
            require(set(a_horizons) == {str(horizon) for horizon in self.spec.b_horizons}
                    and set(item["a_horizon_evidence"]) == {str(horizon) for horizon in retained_horizons},
                    "frozen A horizon evidence is incomplete or changed")
            for horizon in retained_horizons:
                report = a_horizons[str(horizon)]
                fields = ("horizon_hours", "label", "direction", "summary", "coverage",
                          "sample_hours", "grouping")
                snapshot = {field: report[field] for field in fields}
                require(snapshot == item["a_horizon_evidence"][str(horizon)]
                        and report["horizon_hours"] == horizon
                        and {"summary", "coverage", "periods", "stages", "per_symbol"} <= report.keys(),
                        "frozen A horizon evidence differs from the archived report")
        return frozen

    def _validate_once(self, load_panel: Callable[[], FactorInputPanel], idea_pool: Path) -> dict[str, Any]:
        frozen = self._checked_frozen()
        # This marker is committed before even invoking the B data loader.
        access_marker = {"candidate_ids": list(frozen["candidates"]),
                         "time": datetime.now(timezone.utc).isoformat(),
                         "use": "fixed_batch_validation"}
        write_json(self.root / "b-access-started.json", access_marker)
        panel = load_panel()
        self._validate_panel(panel, "B")
        frozen_membership = self.root / "B-membership-frozen.csv"
        if frozen_membership.exists():
            require(frozen_membership.read_text() == panel.universe.rename("eligible").to_csv(),
                    "B membership changed after the batch was frozen")
        with (self.root / "B-universe.csv").open("x") as handle:
            panel.universe.rename("eligible").to_csv(handle)
        requested_horizons = {horizon for item in frozen["candidates"].values()
                              for horizon in item["retained_horizons"]}
        labels_by_horizon = {horizon: build_labels(
            panel, self.spec, "B", horizon_hours=horizon) for horizon in sorted(requested_horizons)}
        store = RecordStore(self.root / "b_records", run_root=self.root.resolve(),
                            archive_root=self.archive_root)
        store.append("inputs", "data_provenance", {
            "panel": _panel_metadata(panel), "catalog": panel.ideation_context()})
        reports = {}
        for cid, item in frozen["candidates"].items():
            store.append(f"{cid}-frozen", "frozen_definition_and_A_evidence", item)
            result = evaluate_expression(item["executed"]["expression"], panel)
            require(result.definition == item["executed"], "B execution differs from frozen formula")
            identity = self._factor_identity(result.definition["expanded_expression"],
                                             item["definition"]["direction"])
            displacement = None
            if item.get("rank_displacement_version") is not None:
                require(item["rank_displacement_version"] == self.rank_displacement_version
                        == RANK_DISPLACEMENT_VERSION,
                        "frozen rank-displacement definition differs from this run")
                displacement = evaluate_rank_displacement(
                    result.values, panel.universe, self.spec, "B")
                require(displacement["definition_version"] == item["rank_displacement_version"],
                        "B rank-displacement implementation differs from frozen A")
            horizon_reports, locators = {}, {}
            for horizon in item["retained_horizons"]:
                horizon_report = evaluate_factor(
                    result.values, labels_by_horizon[horizon], self.spec, "B",
                    item["definition"]["direction"], horizon_hours=horizon)
                horizon_reports[horizon] = horizon_report
                locators[horizon] = self._write_evaluation(
                    cid, "B", horizon_report, identity, definition=item["definition"])
                write_group_plot(self.root / "plots" / f"{cid}-B-{horizon}h.svg", horizon_report)
            displacement_locator = None
            if displacement is not None:
                displacement_locator = self._write_rank_displacement(
                    cid, "B", displacement, identity,
                    source={"frozen_A_rank_displacement_archive": item["a_rank_displacement_archive"],
                            "B_factor_evaluations": locators})
            reports[cid] = {"segment": "B", "direction": item["definition"]["direction"],
                            "retained_horizons": list(item["retained_horizons"]),
                            "horizons": {str(horizon): horizon_reports[horizon]
                                         for horizon in item["retained_horizons"]}}
            if displacement is not None:
                reports[cid]["rank_displacement"] = displacement
                reports[cid]["rank_displacement_archive"] = displacement_locator
            store.append(f"{cid}-evaluation", "evaluation",
                         self._multi_horizon_evaluation_record_data(
                             cid, horizon_reports, locators, item["retained_horizons"],
                             displacement, displacement_locator))
        correction = correct_batch(reports, self.spec)
        store.append("batch-correction", "multiple_testing", correction)
        decisions = {}
        for cid, report in reports.items():
            decisions[cid] = b_candidate_decision(cid, report, correction, self.spec)
            store.append(f"{cid}-validation", "validation_result", {"candidate_id": cid, **decisions[cid]})
        checkpoint = {"candidate_ids": list(reports),
                      "retained_horizons": {cid: report["retained_horizons"]
                                            for cid, report in reports.items()}}
        write_json(self.root / "b-numerical-complete.json", checkpoint)
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
        frozen = self._checked_frozen()
        require((self.root / "b-numerical-complete.json").exists(), "complete B numerical validation before completing reports")
        require((self.root / "b-access-started.json").exists(), "B access marker is missing")
        store = RecordStore(self.root / "b_records", run_root=self.root.resolve(),
                            archive_root=self.archive_root)
        records = store.all()
        evaluation_records = [record for record in records if record["kind"] == "evaluation"]
        validation_records = [record for record in records if record["kind"] == "validation_result"]
        correction_records = [record for record in records if record["kind"] == "multiple_testing"]
        require(len(evaluation_records) == len(validation_records) == len(frozen["candidates"])
                and len(correction_records) == 1, "B numerical records are incomplete or repeated")
        reports = {record["data"]["candidate_id"]: {
            key: value for key, value in record["data"].items() if key != "candidate_id"}
            for record in evaluation_records}
        decisions = {record["data"]["candidate_id"]: {
            key: value for key, value in record["data"].items() if key != "candidate_id"}
            for record in validation_records}
        require(set(reports) == set(decisions) == set(frozen["candidates"]), "B numerical batch is incomplete")
        expected_checkpoint = {"candidate_ids": list(frozen["candidates"]),
                               "retained_horizons": {cid: item["retained_horizons"]
                                                     for cid, item in frozen["candidates"].items()}}
        require(json.loads((self.root / "b-numerical-complete.json").read_text()) == expected_checkpoint,
                "B numerical checkpoint differs from the frozen batch")
        for cid, report in reports.items():
            frozen_item = frozen["candidates"][cid]
            require(report["segment"] == "B"
                    and report["direction"] == frozen["candidates"][cid]["definition"]["direction"]
                    and report["retained_horizons"] == frozen["candidates"][cid]["retained_horizons"]
                    and set(report["horizons"]) == {str(horizon) for horizon in report["retained_horizons"]},
                    "B evaluation horizons differ from the frozen batch")
            if frozen_item.get("rank_displacement_version") is not None:
                displacement = report.get("rank_displacement")
                displacement_locator = report.get("rank_displacement_archive")
                expected_displacement_key = EvaluationKey(
                    f"{self.spec.run_id}/B", dumps(self.spec.as_dict()),
                    frozen_item["rank_displacement_version"], "B", "rank-displacement").as_dict()
                require(frozen_item["rank_displacement_version"] == self.rank_displacement_version
                        == RANK_DISPLACEMENT_VERSION
                        and displacement is not None and displacement_locator is not None
                        and displacement.get("definition_version") == frozen_item["rank_displacement_version"]
                        and displacement.get("segment") == "B"
                        and displacement_locator["identity"]
                        == report["horizons"][str(report["retained_horizons"][0])]["factor_archive"]["identity"]
                        and displacement_locator["evaluation_key"] == expected_displacement_key,
                        "B rank-displacement diagnostic differs from frozen A definition")
            else:
                require("rank_displacement" not in report and "rank_displacement_archive" not in report
                        and self.rank_displacement_version is None,
                        "legacy frozen B cannot receive new rank-displacement diagnostics")
            identity = self._factor_identity(
                frozen["candidates"][cid]["executed"]["expanded_expression"], report["direction"]).as_dict()
            for horizon in report["retained_horizons"]:
                horizon_report = report["horizons"][str(horizon)]
                locator = horizon_report["factor_archive"]
                require(horizon_report["segment"] == "B"
                        and horizon_report["direction"] == frozen["candidates"][cid]["definition"]["direction"]
                        and horizon_report["horizon_hours"] == horizon
                        and locator["identity"] == identity
                        and locator["evaluation_key"] == self._evaluation_key("B", horizon).as_dict(),
                        "B archive identity differs from its frozen formula, direction or horizon")
        correction = correction_records[0]["data"]
        require(correction == correct_batch(reports, self.spec), "B batch correction differs from saved evaluations")
        for cid, report in reports.items():
            expected = b_candidate_decision(cid, report, correction, self.spec)
            require(decisions[cid] == expected, "B program decision differs from saved evaluation")
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
                                       narrative["created_at"])
                path = Path(idea_pool) / f"{self.spec.run_id}--{cid}.json"
                if path.exists():
                    existing_card = json.loads(path.read_text())
                    require(without_hash_metadata(existing_card) == without_hash_metadata(card),
                            "existing idea card differs from saved research evidence")
                else:
                    write_json(path, card)
                decision["idea_card"] = str(path)
        output = {"batch_correction": correction, "decisions": decisions,
                  "status": "reports_pending" if pending else "complete", "pending_report_ids": pending,
                  "B_use": "used for fixed-candidate selection; not untouched data for subsequent strategy development",
                  "C_use": "reserved; never loaded by factor mining"}
        write_research_report(self.root / "B-report.md", self.spec.run_id, self.spec.purpose, records, "B", replace=True)
        if not pending:
            path = self.root / "validation.json"
            if path.exists():
                prior = json.loads(path.read_text())
                require(prior == output, "completed validation differs from saved evidence")
            else:
                write_json(path, output)
        return output

    def _idea_card(self, cid: str, item: dict[str, Any], report: dict[str, Any], narrative: dict[str, Any],
                   decision: dict[str, Any], report_created_at: str) -> dict[str, Any]:
        definition = item["definition"]
        horizon_summaries = {str(horizon): report["horizons"][str(horizon)]["summary"]
                             for horizon in report["retained_horizons"]}
        horizon_coverage = {str(horizon): report["horizons"][str(horizon)]["coverage"]
                            for horizon in report["retained_horizons"]}
        source = {"run_id": self.spec.run_id, "candidate_id": cid, "date": report_created_at,
                  "research_directory": str(self.root.resolve())}
        card = {"id": f"{self.spec.run_id}--{cid}", "source_type": "factor_mining", "status": "research_idea",
                "source": source,
                "original_claim": {"formula": item["executed"], "meaning": definition["meaning"],
                                   "direction": definition["direction"],
                                   "hypothesis": definition["hypothesis"],
                                   "initial_findings": horizon_summaries},
                "economic_mechanism": narrative["mechanism"],
                "market_and_horizon": {"venue": "Binance", "market": "USD-M perpetual", "inputs": "1h",
                                        "target": decision["passed_horizons"],
                                        "retained_horizons": decision["retained_horizons"],
                                        "passed_horizons": decision["passed_horizons"]},
                "data_and_coverage": {"fields": item["executed"]["fields"], "B": horizon_coverage,
                                      "universe_provenance": self.spec.universe_provenance,
                                      "data_usage_review": self.spec.data_usage_review},
                "falsification_conditions": narrative["falsifiers"],
                "unverified_assumptions": narrative["limitations"] + ["因子参数未经最优性证明；完整选币、调仓、仓位、成本及风险规则待研究"],
                "next_research_plan": narrative["next_steps"], "applicability": narrative["conditions"],
                "admission_evidence": decision, "multiple_testing": {"method": self.spec.fdr_method,
                                                                    "alpha": self.spec.fdr_alpha},
                "evidence_references": {
                    "A": {"record_id": item["a_evaluation_ref"]["record_id"],
                          **({"rank_displacement_archive": item["a_rank_displacement_archive"]}
                             if "a_rank_displacement_archive" in item else {})},
                    "B": {"record_id": f"{cid}-evaluation",
                          "factor_archives": {str(horizon): report["horizons"][str(horizon)]["factor_archive"]
                                              for horizon in report["retained_horizons"]},
                          **({"rank_displacement_archive": report["rank_displacement_archive"]}
                             if "rank_displacement_archive" in report else {})}},
                "a_research_decision": item["a_decision"],
                "b_validation_status": decision["validation_status"],
                "strategy_validation_status": "not_started"}
        if "rank_displacement" in report:
            card["signal_persistence"] = {
                "A": item["a_rank_displacement"],
                "B": self._rank_displacement_summary(report["rank_displacement"]),
            }
        labels = [TAG_LABELS[name] for name in decision["tracks"]] or ["基础型"]
        card["title"] = definition["name"] + " · " + "、".join(labels)
        card["classification"] = {"scheme": self.spec.admission_scheme, "tracks": decision["tracks"],
                                  "track_labels": labels, "tracks_gate": False}
        card["unverified_assumptions"].append(
            "准入规则在既有B区间被使用后修订；该B结果不能充当规则选择后的全新独立验证。")
        return card
