"""Scripted goal integration: actual A/B computation, no network or live claims."""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from test_factor_mining import ScenarioModel, definition, input_panel, specification
from crypto_quant.research.factor_mining.contracts import dumps
from crypto_quant.research.factor_mining.evaluation import evaluate_factor
from crypto_quant.research.factor_mining.factor_archive import EvaluationKey, FactorArchive, FactorIdentity
from crypto_quant.research.factor_mining.goal import GoalRunner, GoalSpec
from crypto_quant.research.factor_mining.model import ModelReply
from crypto_quant.research.factor_mining.workflow import FactorMiner


from crypto_quant.research.factor_mining.codex_model import CodexModel

SETTINGS = CodexModel("scripted-fixture", reasoning_effort="low", timeout_seconds=180).settings()


class GoalModel:
    def __init__(self, *, wait=False, optimize=False, duplicate=False, matches=True):
        self.requests = []
        self.miners = {}
        self.wait, self.optimize, self.duplicate, self.matches = wait, optimize, duplicate, matches
        self.selections = 0
        self.interrupt_role = None
        self.selection_override = None

    def settings(self):
        return SETTINGS.copy()

    def complete(self, messages, *, max_output_tokens, session_id):
        request = json.loads(messages[1]["content"])
        self.requests.append(copy.deepcopy(request))
        role, payload = request["role"], request["payload"]
        if self.interrupt_role == role and "goal_phase" not in payload:
            self.interrupt_role = None
            raise KeyboardInterrupt()
        if payload.get("goal_phase") == "select_task":
            self.selections += 1
            if self.selections > 5:
                raise AssertionError("fixture should have completed or paused before six tasks")
            wait = self.wait and self.selections == 1
            result = {"action": "wait_data" if wait else "explore_new",
                      "task": f"脚本研究方向{self.selections}，由构想生成公式", "reason": "脚本模拟未研究方向",
                      "evidence_refs": [payload["catalog_record_id"]],
                      "dependencies": [{"field": "premium_index", "min_valid_rows": 100,
                                        "reason": "脚本缺失字段依赖"}] if wait else []}
            if self.selection_override:
                self.selection_override(result)
        elif payload.get("goal_phase") == "verify_completion":
            result = {"matches": [{"idea_id": value, "matches_goal": self.matches,
                                    "reason": "脚本核对Goal与A定义"} for value in payload["idea_ids"]]}
        else:
            run_id = request["contract"]["run_id"]
            if run_id not in self.miners:
                window = 3 if self.duplicate else 2 + len(self.miners)
                def transform(req, value):
                    if req["role"] == "ideator" and not req["payload"]["pending_proposals"]:
                        value["candidates"] = [definition("div(perp_close,ts_mean(perp_close))")]
                    if req["role"] == "calculator":
                        value["repair_expression"] = f"div(perp_close,ts_mean(perp_close,{window}))"
                    if req["role"] == "optimizer":
                        ids = {r["id"] for r in req["records"]}
                        for decision in value["decisions"]:
                            cid = decision["candidate_id"]
                            if f"{cid}-duplicate" in ids:
                                decision.update(disposition="discard", retained_horizons=[], continue_optimization=False,
                                                evidence_refs=[f"{cid}-duplicate"], resume_condition=None)
                self.miners[run_id] = ScenarioModel(transform=transform, propose_once=self.optimize)
            return self.miners[run_id].complete(messages, max_output_tokens=max_output_tokens, session_id=session_id)
        return ModelReply(dumps({"result": result, "read_records": []}), {"fixture": True}, "scripted-fixture")


class GoalTests(unittest.TestCase):
    def create(self, directory, model=None, *, target=1, purpose="research"):
        spec = specification(purpose=purpose)
        runner = GoalRunner.create(GoalSpec("outcome-fixture", "研究价格偏离关系的合格成果", target),
                                   spec, model or GoalModel(), Path(directory),
                                   inputs={}, model_settings=SETTINGS)
        return runner, input_panel(spec), input_panel(spec, "B")

    def test_saved_goal_does_not_bind_code_version(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, _, _ = self.create(directory)
            saved = json.loads((runner.root / "goal.json").read_text())
            self.assertNotIn("code", saved)
            self.assertNotIn("sha256", saved)
            saved["code"] = {"workflow.py": "different-code-version"}
            (runner.root / "goal.json").write_text(json.dumps(saved))
            self.assertEqual(GoalRunner(runner.root, runner.model).state["phase"], "select_task")

    def test_model_change_uses_goal_state_without_rewriting_original_goal(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, _, _ = self.create(directory)
            original = (runner.root / "goal.json").read_bytes()
            updated = CodexModel("scripted-next", reasoning_effort="low", timeout_seconds=180).settings()
            runner._save(model_settings=updated)
            self.assertEqual(GoalRunner(runner.root, runner.model).model_settings, updated)
            self.assertEqual((runner.root / "goal.json").read_bytes(), original)

    def run_goal(self, runner, a, b, *, sleep=lambda _: None):
        return runner.run(lambda: a, lambda: b, lambda: b.universe, runner.root / "ideas",
                          poll_seconds=1, sleep=sleep)

    @staticmethod
    def passing_B(values, labels, spec, stage, direction, *, horizon_hours=24):
        report = evaluate_factor(values, labels, spec, stage, direction, horizon_hours=horizon_hours)
        if stage == "B":
            report["summary"]["rank_ic"].update(mean=direction * 0.2, p_value=0.000001)
            report["summary"]["directional_spread"].update(mean=0.003, p_value=0.000001)
            report["summary"].update(valid_stages=2, positive_stage_share=1.0)
        return report

    def test_empty_proposals_start_next_direction_until_verified_success(self):
        with tempfile.TemporaryDirectory() as directory:
            model = GoalModel(optimize=True)
            runner, a, b = self.create(directory, model)
            b_runs = set()
            def evaluation(values, labels, spec, stage, direction, *, horizon_hours=24):
                if stage == "B":
                    b_runs.add(spec.run_id)
                if stage == "B" and len(b_runs) == 1:
                    return evaluate_factor(values, labels, spec, stage, direction, horizon_hours=horizon_hours)
                return self.passing_B(values, labels, spec, stage, direction, horizon_hours=horizon_hours)
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=evaluation):
                result = self.run_goal(runner, a, b)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["cycle"], 2)
            self.assertGreaterEqual(len(result["qualified_ideas"]), 1)
            runs = sorted(path for path in (runner.root / "runs").iterdir()
                          if (path / "contract.json").is_file())
            self.assertEqual(len(runs), 2)
            first = json.loads((runs[0] / "a-complete.json").read_text())
            self.assertEqual(first["completed_rounds"], 2)  # existing optimization route still works
            self.assertFalse(any(d["eligible_for_idea_pool"] for d in json.loads(
                (runs[0] / "validation.json").read_text())["decisions"].values()))
            selections = [r for r in model.requests if r["payload"].get("goal_phase") == "select_task"]
            self.assertEqual(len(selections), 2)
            self.assertTrue(any(r["kind"] == "research_cycle" for r in selections[1]["records"]))
            self.assertIn("cycle-00000001", selections[1]["payload"]["allowed_evidence_refs"])
            self.assertNotIn("candidate-0001-evaluation", selections[1]["payload"]["allowed_evidence_refs"])
            self.assertEqual(sum(record["kind"] == "data_provenance" for record in runner.research.all()), 2)
            archived = next(record for record in runner.research.all() if record["kind"] == "research_cycle")
            self.assertTrue(archived["data"]["context"]["candidates"])
            cycle = json.loads((runner.research.root / f"{archived['id']}.json").read_text())["data"]
            self.assertIsInstance(cycle["records"], dict)
            self.assertEqual(cycle["candidate_ids"], ["candidate-0001", "candidate-0002"])
            items = runner.research.read(archived["id"], "/records", 0, 100)["items"]
            evaluation_index = next(i for i, item in enumerate(items)
                                    if item["ref"]["record_id"] == "candidate-0001-evaluation")
            evaluation_item = items[evaluation_index]
            self.assertEqual(evaluation_item["candidate_id"], "candidate-0001")
            self.assertIn("evaluation_id", evaluation_item["factor_archive_ref"])
            page = runner.research.read(archived["id"],
                                        f"/records/{evaluation_index}/data/periods", 0, 2)
            self.assertEqual(len(page["items"]), 2)
            originals = runner.research.source_records(archived["id"])
            self.assertEqual(page["items"], originals[evaluation_index]["data"]["periods"][:2])
            locator = evaluation_item["factor_archive_ref"]
            source_run = runner.research._source_run(archived["data"])
            factor_archive = FactorArchive.open_existing(
                source_run / locator["root"], FactorIdentity(**locator["identity"]))
            expected_values = factor_archive.page_factor_values(
                EvaluationKey(**locator["evaluation_key"]), offset=0, limit=2)
            values_page = runner.research.read(
                archived["id"], f"/records/{evaluation_index}/data/factor_values", 0, 2)
            self.assertEqual(values_page, expected_values)
            self.assertEqual(evaluation_item["record_id"], "candidate-0001-evaluation")
            self.assertNotIn('"periods"', dumps(cycle))
            admissions = [r for r in runner.receipts.all() if r["kind"] == "program_admission"]
            self.assertTrue(admissions)
            for receipt in admissions:
                for idea in receipt["data"]["ideas"]:
                    self.assertEqual(idea["a_evaluation_ref"], {
                        "record_id": f"{idea['candidate_id']}-evaluation"})
                    self.assertEqual(idea["run_id"], (runner.root / idea["run_path"]).name)
                    source_evaluation = json.loads((runner.root / idea["run_path"] / "a_records" /
                        f"{idea['candidate_id']}-evaluation.json").read_text())
                    self.assertEqual(idea["factor_archive_ref"], source_evaluation["data"]["factor_archive"])
                    self.assertNotIn("a_evaluation", idea)
            for request in model.requests:
                if request["role"] in {"ideator", "optimizer"}:
                    encoded = dumps(request)
                    self.assertNotIn('"batch_correction"', encoded)
                    self.assertNotIn('"validation_status"', encoded)
                    self.assertNotIn('"segment":"B"', encoded)
            for request in [r for r in model.requests if r["role"] == "ideator"]:
                self.assertTrue(any(r["kind"] == "goal_context" for r in request["records"]))
            archived_cycle = next(r["data"] for r in selections[1]["records"] if r["kind"] == "research_cycle")
            self.assertTrue(archived_cycle["records"])
            second_ideation = next(r for r in model.requests
                                   if r["role"] == "ideator" and r["contract"]["run_id"] == runs[1].name)
            goal_context = next(r["data"] for r in second_ideation["records"] if r["kind"] == "goal_context")
            self.assertNotIn("A_history", goal_context)
            self.assertEqual(len(goal_context["prior_A_research"]["cycles"]), 1)
            self.assertTrue(goal_context["prior_A_research"]["cycles"][0]["candidates"])

    def test_goal_context_handoff_keeps_summaries_without_copying_numeric_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, _ = self.create(directory)
            runner.state["task"] = {"action": "explore_new", "task": "next", "reason": "new",
                                    "evidence_refs": ["inputs-00000001"], "dependencies": []}
            self.assertEqual(runner._context(a)["prior_A_research"]["cycles"], [])
            miner = FactorMiner(replace(runner.spec, run_id="prior-run"), ScenarioModel(),
                                runner.root / "runs")
            miner.explore(a)
            runner.state["cycle"] = 1
            runner._archive_A(miner)
            evaluation = miner.store._load(miner.store.root / "candidate-0001-evaluation.json")["data"]
            context = runner._context(a)
            encoded = dumps(context)
            self.assertLess(len(encoded), 10000)
            self.assertNotIn('"periods"', encoded)
            candidate = context["prior_A_research"]["cycles"][0]["candidates"][0]
            self.assertEqual(candidate["A_evaluation"]["summary"], evaluation["summary"])
            self.assertEqual(candidate["final_decision"]["disposition"], "retain")
            self.assertEqual(context["previous_expressions"], [])
            cycle = json.loads((runner.research.root / "cycle-00000001.json").read_text())["data"]
            evaluation_index = next(i for i, item in enumerate(cycle["records"]["items"])
                                    if item["ref"]["record_id"] == "candidate-0001-evaluation")
            original = runner.research.read("cycle-00000001",
                f"/records/{evaluation_index}/data/periods", 5, 1)
            self.assertEqual(original["items"], evaluation["periods"][5:6])

            miner = FactorMiner(specification(), ScenarioModel(), Path(directory) / "retest")
            result = miner.explore(a, goal_context=context)
            self.assertIn("candidate-0001", result["evaluated_ids"])
            self.assertFalse((miner.root / "a_records" / "candidate-0001-duplicate.json").exists())

    def test_goal_context_uses_saved_cycle_context_without_hash_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, _ = self.create(directory)
            runner.state["task"] = {"action": "explore_new", "task": "next", "reason": "new",
                                    "evidence_refs": ["inputs-00000001"], "dependencies": []}
            run_id = "prior-run"
            context = {"source_record_id": "cycle-00000001", "run_id": run_id,
                       "candidates": [{"candidate_ref": "prior-run/candidate-0001"}],
                       "previous_expressions": [{"expression": "cross_rank(funding_24h_sum)"}]}
            runner.research.append("cycle-00000001", "research_cycle", {
                "run_id": run_id,
                "records": {"run_path": f"{runner.goal.goal_id}/runs/{run_id}", "items": []},
                "context": context,
            })
            summary = next(record for record in runner.research.all() if record["kind"] == "research_cycle")
            self.assertEqual(summary["data"]["records"]["items"], [])
            self.assertEqual(runner._context(a)["prior_A_research"]["cycles"][0]["candidates"],
                             context["candidates"])
            self.assertEqual(runner._context(a)["previous_expressions"], [])

    def test_goal_selection_does_not_deduplicate_catalog_by_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, _ = self.create(directory)

            runner._select(a)
            first_id = runner.model.requests[-1]["payload"]["catalog_record_id"]
            runner._select(a)
            second_id = runner.model.requests[-1]["payload"]["catalog_record_id"]

            catalog_ids = [record["id"] for record in runner.research.all()
                           if record["kind"] == "data_provenance"]
            self.assertEqual(catalog_ids, [first_id, second_id])
            self.assertNotEqual(first_id, second_id)
            fingerprints = [record["data"]["fingerprint"] for record in runner.research.all()
                            if record["kind"] == "data_provenance"]
            self.assertEqual(fingerprints[0], fingerprints[1])

    def test_successor_goal_continues_after_imported_cycle_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, _ = self.create(directory)
            runner.research.append("cycle-00000020", "research_cycle", {
                "run_id": "prior-run",
                "records": {"run_path": f"{runner.goal.goal_id}/runs/prior-run", "items": []},
                "context": {"source_record_id": "cycle-00000020", "run_id": "prior-run",
                            "candidates": [], "previous_expressions": []},
            })

            runner._select(a)

            self.assertEqual(runner.state["cycle"], 21)
            self.assertTrue(runner.state["current_run"].endswith("-000021"))

    def test_rearchive_preserves_saved_A_cycle_when_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, _, _ = self.create(directory)
            runner.state["cycle"] = 1
            miner = FactorMiner(specification(run_id="archive-fixture"), GoalModel(), Path(directory) / "runs")
            miner.store.append("inputs", "data_provenance", {"fingerprint": {"first": True}})
            runner._archive_A(miner)
            archive = runner.research.root / "cycle-00000001.json"
            original = archive.read_bytes()

            miner.store.append("new-evidence", "data_provenance", {"changed": True})
            runner._archive_A(miner)

            self.assertEqual(archive.read_bytes(), original)
            self.assertEqual(len([record for record in runner.research.all()
                                  if record["kind"] == "research_cycle"]), 1)

    def test_v6_admission_receipt_recovery_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, b = self.create(directory)
            save = runner._save
            def interrupt_completion(**changes):
                if changes.get("status") == "complete":
                    raise KeyboardInterrupt()
                save(**changes)
            with patch.object(runner, "_save", side_effect=interrupt_completion), patch(
                    "crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=self.passing_B):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_goal(runner, a, b)
            self.assertEqual(runner.state["phase"], "finish_cycle")
            saved_receipt = (runner.receipts.root / "admission-00000001.json").read_bytes()
            saved_match = (runner.receipts.root / "match-00000001.json").read_bytes()
            saved_cycle = (runner.research.root / "cycle-00000001.json").read_bytes()
            idea = json.loads(saved_receipt)["data"]["ideas"][0]
            self.assertIn("a_evaluation_ref", idea)
            self.assertIn("factor_archive_ref", idea)
            self.assertNotIn("a_evaluation", idea)
            saved_cards = {p.name: p.read_bytes() for p in (runner.root / "ideas").glob("*.json")}
            requests = len(runner.model.requests)
            def forbidden():
                raise AssertionError("completion recovery must not load A or B data")
            restarted = GoalRunner(runner.root, runner.model)
            result = restarted.run(forbidden, forbidden, forbidden, runner.root / "ideas", poll_seconds=1)
            self.assertEqual(result["status"], "complete")
            self.assertEqual((runner.receipts.root / "admission-00000001.json").read_bytes(), saved_receipt)
            self.assertEqual((runner.receipts.root / "match-00000001.json").read_bytes(), saved_match)
            self.assertEqual((runner.research.root / "cycle-00000001.json").read_bytes(), saved_cycle)
            self.assertEqual({p.name: p.read_bytes() for p in (runner.root / "ideas").glob("*.json")}, saved_cards)
            self.assertEqual(result["qualified_ideas"], [idea["idea_id"]])
            self.assertEqual(len(runner.model.requests), requests)

    def test_goal_and_ideation_do_not_block_interpretation_text(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, panel, _ = self.create(directory)
            task = {
                "action": "explore_new",
                "task": "用 open_interest_value 和 open_interest_base 讨论平均持仓价格与浮盈亏",
                "reason": "研究杠杆与持仓成本解释",
                "evidence_refs": ["inputs-00000001"],
                "dependencies": [],
            }
            runner.research.append("inputs-00000001", "data_provenance", {"fixture": True})
            runner._check_task(task, panel)

            miner = FactorMiner(specification(), GoalModel(), Path(directory))
            definition_with_interpretation = definition("div(open_interest_value,open_interest_base)")
            definition_with_interpretation.update(
                name="oi_average_entry_price",
                meaning="用名义价值除以数量讨论平均开仓价格与正交化残差",
                hypothesis="平均持仓价与未来收益可能有关",
                change_reason="测试持仓成本偏离",
            )
            response = {"candidates": [definition_with_interpretation],
                        "dispositions": [], "analysis": "测试"}
            definitions, _, _, _ = miner._check_ideation(response, {})
            self.assertEqual(definitions[0]["name"], "oi_average_entry_price")

    def test_wait_polls_without_model_calls_and_automatically_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            model = GoalModel(wait=True)
            runner, a, b = self.create(directory, model)
            a.values["premium_index"] = float("nan")
            sleeps = []
            def sleep(_):
                sleeps.append(len(model.requests))
                self.assertEqual(runner.status(runner.root)["status"], "waiting")
                if len(sleeps) == 2:
                    a.values["premium_index"] = 0.01
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=self.passing_B):
                result = self.run_goal(runner, a, b, sleep=sleep)
            self.assertEqual(sleeps, [1, 1])
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["cycle"], 1)
            self.assertTrue(any(r["data"]["status"] == "waiting" for r in runner.events.all()))

    def test_restart_after_interrupt_does_not_repeat_accepted_ideation(self):
        with tempfile.TemporaryDirectory() as directory:
            model = GoalModel()
            model.interrupt_role = "calculator"
            runner, a, b = self.create(directory, model)
            with self.assertRaises(KeyboardInterrupt):
                self.run_goal(runner, a, b)
            self.assertEqual(GoalRunner.status(runner.root)["status"], "paused")
            restarted = GoalRunner(runner.root, model)
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=self.passing_B):
                result = self.run_goal(restarted, a, b)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(sum(r["role"] == "ideator" for r in model.requests), 1)
            self.assertEqual(sum(r["payload"].get("goal_phase") == "select_task" for r in model.requests), 1)

    def test_B_report_restart_uses_saved_numbers_without_loading_B_again(self):
        with tempfile.TemporaryDirectory() as directory:
            model = GoalModel()
            runner, a, b = self.create(directory, model)
            for _ in range(3):
                runner._step(lambda: a, lambda: b, lambda: b.universe, runner.root / "ideas")
            self.assertEqual(runner.state["phase"], "validate")
            model.interrupt_role = "evaluator"
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=self.passing_B):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_goal(runner, a, b)
            restarted = GoalRunner(runner.root, model)
            def forbidden():
                raise AssertionError("B loader must not be called during report recovery")
            result = restarted.run(lambda: a, forbidden, forbidden, runner.root / "ideas", poll_seconds=1)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(list((runner.root / "runs").glob("*/b-access-started.json"))), 1)

    def test_resume_after_computation_and_after_completed_round_reuses_A_checkpoints(self):
        for checkpoint in ("optimizer", "second_ideation"):
            with self.subTest(checkpoint=checkpoint), tempfile.TemporaryDirectory() as directory:
                model = GoalModel(optimize=True)
                runner, a, b = self.create(directory, model)
                original = model.complete
                intercepted = []
                def interrupt(messages, **kwargs):
                    request = json.loads(messages[1]["content"])
                    role, payload = request["role"], request["payload"]
                    stop = ((checkpoint == "optimizer" and role == "optimizer" and "goal_phase" not in payload)
                            or (checkpoint == "second_ideation" and role == "ideator" and payload["pending_proposals"]))
                    if stop and not intercepted:
                        intercepted.append(True)
                        raise KeyboardInterrupt()
                    return original(messages, **kwargs)
                model.complete = interrupt
                with self.assertRaises(KeyboardInterrupt):
                    self.run_goal(runner, a, b)
                calculators = sum(r["role"] == "calculator" for r in model.requests)
                restarted = GoalRunner(runner.root, model)
                with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=self.passing_B):
                    result = self.run_goal(restarted, a, b)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(sum(r["role"] == "calculator" for r in model.requests), calculators)
                self.assertEqual(sum(r["role"] == "ideator" for r in model.requests), 2)

    def test_concurrent_runner_fails_and_completed_goal_makes_no_calls(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            runner, a, b = self.create(directory)
            with (runner.root / "runner.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    self.run_goal(runner, a, b)
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=self.passing_B):
                self.run_goal(runner, a, b)
            count = len(runner.model.requests)
            def forbidden():
                raise AssertionError("completed Goal must not load data")
            resumed = GoalRunner(runner.root, runner.model)
            self.assertEqual(resumed.run(forbidden, forbidden, forbidden, runner.root / "ideas", poll_seconds=1)["status"],
                             "complete")
            self.assertEqual(len(runner.model.requests), count)

    def test_wait_survives_process_restart_and_user_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, b = self.create(directory, GoalModel(wait=True))
            a.values["premium_index"] = float("nan")
            def stop(_):
                raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                self.run_goal(runner, a, b, sleep=stop)
            saved = GoalRunner.status(runner.root)
            self.assertEqual(saved["phase"], "wait_data")
            self.assertEqual(saved["task"]["dependencies"][0]["field"], "premium_index")
            restarted = GoalRunner(runner.root, runner.model)
            a.values["premium_index"] = 0.1
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=self.passing_B):
                self.assertEqual(self.run_goal(restarted, a, b)["status"], "complete")

    def test_optimizer_cannot_complete_or_wait_for_already_present_data(self):
        for invalid in ["complete", "wait_data"]:
            with self.subTest(action=invalid), tempfile.TemporaryDirectory() as directory:
                model = GoalModel()
                def override(result):
                    result["action"] = invalid
                    result["dependencies"] = [{"field": "premium_index", "min_valid_rows": 1, "reason": "invalid"}]
                model.selection_override = override
                runner, a, b = self.create(directory, model)
                with self.assertRaisesRegex(AssertionError, "fixture should"):
                    self.run_goal(runner, a, b)
                self.assertEqual(runner.state["status"], "error")
                self.assertEqual(runner.state["cycle"], 0)
                self.assertEqual(len([r for r in runner.research.all() if r["kind"] == "invalid_model_response"]), 5)

    def test_goal_rejects_wrong_data_segment_before_first_model_request(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, b = self.create(directory)
            with self.assertRaisesRegex(ValueError, "warm-up|allowed data segment"):
                self.run_goal(runner, b, b)
            self.assertEqual(runner.model.requests, [])
            self.assertEqual(runner.state["status"], "error")

    def test_target_is_runtime_config_and_engineering_or_nonmatching_ideas_do_not_complete(self):
        for purpose, matches in [("engineering_check", True), ("research", False)]:
            with self.subTest(purpose=purpose), tempfile.TemporaryDirectory() as directory:
                model = GoalModel(matches=matches)
                runner, a, b = self.create(directory, model, purpose=purpose)
                with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=self.passing_B):
                    for _ in range(5):
                        runner._step(lambda: a, lambda: b, lambda: b.universe, runner.root / "ideas")
                self.assertEqual(runner.state["phase"], "select_task")
                self.assertEqual(runner.state["qualified_ideas"], [])
        with tempfile.TemporaryDirectory() as directory:
            runner, a, b = self.create(directory, target=2)
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=self.passing_B):
                result = self.run_goal(runner, a, b)
            self.assertEqual(result["cycle"], 2)
            self.assertEqual(len(result["qualified_ideas"]), 2)

    def test_same_formula_without_data_version_is_researched_under_new_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            model = GoalModel(duplicate=True)
            runner, a, b = self.create(directory, model, target=2)
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=self.passing_B):
                for _ in range(5):
                    runner._step(lambda: a, lambda: b, lambda: b.universe, runner.root / "ideas")
                for _ in range(5):
                    runner._step(lambda: a, lambda: b, lambda: b.universe, runner.root / "ideas")
            self.assertEqual(runner.state["cycle"], 2)
            self.assertEqual(runner.state["phase"], "complete")
            self.assertEqual(len(runner.state["qualified_ideas"]), 2)
            second = sorted((runner.root / "runs").iterdir())[1]
            self.assertTrue((second / "a_records/candidate-0001-evaluation.json").exists())
            self.assertTrue((second / "b-access-started.json").exists())

    def test_changed_A_cannot_resume_partial_run_and_incomplete_B_cannot_reread(self):
        with tempfile.TemporaryDirectory() as directory:
            model = GoalModel()
            model.interrupt_role = "calculator"
            runner, a, b = self.create(directory, model)
            with self.assertRaises(KeyboardInterrupt):
                self.run_goal(runner, a, b)
            a.universe.iloc[0] = not a.universe.iloc[0]
            with self.assertRaisesRegex(ValueError, "A membership changed"):
                self.run_goal(GoalRunner(runner.root, model), a, b)
            self.assertEqual(GoalRunner.status(runner.root)["status"], "error")
        with tempfile.TemporaryDirectory() as directory:
            runner, a, b = self.create(directory)
            for _ in range(3):
                runner._step(lambda: a, lambda: b, lambda: b.universe, runner.root / "ideas")
            miner = runner._miner()
            (miner.root / "b-access-started.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "B was accessed"):
                self.run_goal(runner, a, b)

    def test_cli_status_is_read_only_and_resume_uses_saved_settings(self):
        import argparse
        from crypto_quant.research.factor_mining.cli import add_arguments, execute
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        with tempfile.TemporaryDirectory() as directory:
            runner, _, _ = self.create(directory)
            args = parser.parse_args(["goal-status", "--goal-dir", str(runner.root)])
            with patch("crypto_quant.research.factor_mining.model_config.CodexModel") as model:
                result = execute(args)
                model.assert_not_called()
            self.assertEqual(result["phase"], "select_task")
            resume = parser.parse_args(["goal-resume", "--goal-dir", str(runner.root)])
            with patch("crypto_quant.research.factor_mining.model_config.CodexModel") as model, patch(
                    "crypto_quant.research.factor_mining.cli._run_goal") as run:
                model.return_value.settings.return_value = SETTINGS
                execute(resume)
                model.assert_called_once_with("scripted-fixture", timeout_seconds=180, reasoning_effort="low")
                self.assertEqual(run.call_args.args[0].goal, runner.goal)

    def test_cli_start_drives_goal_to_completion_with_saved_inputs(self):
        import argparse
        from crypto_quant.research.factor_mining.cli import add_arguments, execute
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = specification(purpose="research")
            a, b = input_panel(spec), input_panel(spec, "B")
            (root / "contract.json").write_text(dumps(spec.as_dict()))
            (root / "goal.json").write_text(dumps({"goal_id": "cli-goal", "objective": "价格偏离研究", "target_ideas": 1}))
            (root / "universe.csv").write_text("fixture")
            (root / "fixture.sqlite").touch()
            args = parser.parse_args(["goal-start", "--goal", str(root / "goal.json"),
                "--contract", str(root / "contract.json"), "--universe", str(root / "universe.csv"),
                "--db", str(root / "fixture.sqlite"),
                "--output-root", str(root / "goals"), "--idea-pool", str(root / "ideas"),
                "--model", "scripted-fixture", "--reasoning-effort", "low", "--timeout-seconds", "180"])
            def load(db, universe, contract, stage, **kwargs):
                self.assertIn(stage, {"A", "B"})
                self.assertEqual(contract, spec)
                return a if stage == "A" else b
            with patch("crypto_quant.research.factor_mining.cli.model_from_args", return_value=GoalModel()), patch(
                    "crypto_quant.research.factor_mining.cli.load_stage", side_effect=load), patch(
                    "crypto_quant.research.factor_mining.cli.load_membership", return_value=b.universe), patch(
                    "crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=self.passing_B):
                result = execute(args)
            self.assertEqual(result["status"], "complete")
            saved = json.loads((root / "goals/cli-goal/goal.json").read_text())
            self.assertEqual(saved["inputs"]["universe"], str((root / "universe.csv").resolve()))
            self.assertEqual(len(list((root / "ideas").glob("*.json"))), 1)

    def test_goal_reuses_A_panel_until_input_file_changes(self):
        from crypto_quant.research.factor_mining.cli import _run_goal
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db, universe = root / "fixture.sqlite", root / "universe.csv"
            db.write_bytes(b"a")
            universe.write_bytes(b"a")
            panels = [object(), object(), object()]
            runner = Mock()
            runner.inputs = {"db": str(db), "universe": str(universe),
                             "idea_pool": str(root / "ideas"), "include_liquidations": False,
                             "poll_seconds": 1}
            runner.spec, runner.root = specification(), root / "goal"

            def run(load_a, *_args, **_kwargs):
                self.assertIs(load_a(), panels[0])
                self.assertIs(load_a(), panels[0])
                universe.write_bytes(b"new-universe")
                self.assertIs(load_a(), panels[1])
                db.write_bytes(b"new-db")
                self.assertIs(load_a(), panels[2])
                return {"status": "complete"}

            runner.run.side_effect = run
            with patch("crypto_quant.research.factor_mining.cli.load_stage", side_effect=panels) as loader:
                self.assertEqual(_run_goal(runner)["status"], "complete")
            self.assertEqual(loader.call_count, 3)

    def test_checkpoint_write_is_atomic_and_never_overwrites_evidence(self):
        from crypto_quant.research.factor_mining.records import write_json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            with patch("crypto_quant.research.factor_mining.records.os.link", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    write_json(path, {"unfinished": True})
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
            write_json(path, {"accepted": True})
            with self.assertRaises(FileExistsError):
                write_json(path, {"accepted": False})
            self.assertEqual(json.loads(path.read_text()), {"accepted": True})


if __name__ == "__main__":
    unittest.main()
