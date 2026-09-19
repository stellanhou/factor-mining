"""Scripted goal integration: actual A/B computation, no network or live claims."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_factor_mining import ScenarioModel, definition, input_panel, specification
from crypto_quant.research.factor_mining.contracts import dumps
from crypto_quant.research.factor_mining.evaluation import evaluate_factor
from crypto_quant.research.factor_mining.goal import GoalRunner, GoalSpec
from crypto_quant.research.factor_mining.model import ModelReply
from crypto_quant.research.factor_mining.records import ModelResponseError
from crypto_quant.research.factor_mining.workflow import (
    FactorMiner, _panel_fingerprint, _reject_unsupported_oi_cost_basis,
    _reject_unsupported_residualization,
)


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
                        value["candidates"] = [definition(f"div(perp_close,ts_mean(perp_close,{window}))")]
                    if req["role"] == "optimizer":
                        ids = {r["id"] for r in req["records"]}
                        for decision in value["decisions"]:
                            cid = decision["candidate_id"]
                            if f"{cid}-duplicate" in ids:
                                decision.update(disposition="discard", continue_optimization=False,
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

    def run_goal(self, runner, a, b, *, sleep=lambda _: None):
        return runner.run(lambda: a, lambda: b, lambda: b.universe, runner.root / "ideas",
                          poll_seconds=1, sleep=sleep)

    @staticmethod
    def passing_B(values, labels, spec, stage, direction):
        report = evaluate_factor(values, labels, spec, stage, direction)
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
            def evaluation(values, labels, spec, stage, direction):
                if stage == "B":
                    b_runs.add(spec.run_id)
                if stage == "B" and len(b_runs) == 1:
                    return evaluate_factor(values, labels, spec, stage, direction)
                return self.passing_B(values, labels, spec, stage, direction)
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=evaluation):
                result = self.run_goal(runner, a, b)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["cycle"], 2)
            self.assertGreaterEqual(len(result["qualified_ideas"]), 1)
            runs = sorted((runner.root / "runs").iterdir())
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
            self.assertEqual(sum(record["kind"] == "data_provenance" for record in runner.research.all()), 1)
            archived = next(record for record in runner.research.all() if record["kind"] == "research_cycle")
            self.assertTrue(archived["data"]["context"]["candidates"])
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
            fingerprint = runner._context(a)["prior_A_research"]["cycles"]
            self.assertEqual(fingerprint, [])
            runner.research.append("cycle-00000001", "research_cycle", {"run_id": "prior-run", "records": [
                {"id": "inputs", "kind": "data_provenance",
                 "data": {"fingerprint": _panel_fingerprint(a)}},
                {"id": "candidate-0001-definition", "kind": "candidate", "data": {
                    "id": "candidate-0001", "definition": definition("ts_mean(perp_close,24)")}},
                {"id": "candidate-0001-calculation", "kind": "calculation", "data": {
                    "candidate_id": "candidate-0001", "status": "computed",
                    "executed_expression": {"expanded_expression": "ts_mean(perp_close,24)"}}},
                {"id": "candidate-0001-evaluation", "kind": "evaluation", "data": {
                    "candidate_id": "candidate-0001", "summary": {"rank_ic": {"mean": 0.01}},
                    "coverage": {"computed": 10}, "periods": [{"rank_ic": 0.01}] * 10000}},
                {"id": "round-001-optimization", "kind": "optimization", "data": {"decisions": [{
                    "candidate_id": "candidate-0001", "disposition": "pause",
                    "continue_optimization": False, "reason": "weak", "resume_condition": "new evidence"}]}}
            ]})
            context = runner._context(a)
            encoded = dumps(context)
            self.assertLess(len(encoded), 10000)
            self.assertNotIn('"periods"', encoded)
            candidate = context["prior_A_research"]["cycles"][0]["candidates"][0]
            self.assertEqual(candidate["A_evaluation"]["summary"]["rank_ic"]["mean"], 0.01)
            self.assertEqual(candidate["final_decision"]["disposition"], "pause")

    def test_successor_goal_continues_after_imported_cycle_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, _ = self.create(directory)
            runner.research.append("cycle-00000020", "research_cycle", {
                "run_id": "prior-run",
                "records": [],
                "context": {"source_record_id": "cycle-00000020", "run_id": "prior-run",
                            "candidates": [], "previous_expressions": []},
            })

            runner._select(a)

            self.assertEqual(runner.state["cycle"], 21)
            self.assertTrue(runner.state["current_run"].endswith("-000021"))

    def test_goal_and_ideation_reject_oi_notional_as_entry_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, a, _ = self.create(directory)
            invalid_task = {
                "action": "explore_new",
                "task": "用 open_interest_value 和 open_interest_base 推断平均持仓价格与浮盈亏",
                "reason": "检验持仓成本偏离",
                "evidence_refs": ["inputs-00000001"],
                "dependencies": [],
            }
            with self.assertRaisesRegex(ValueError, "does not provide entry price"):
                runner._check_task(invalid_task, a)

            invalid_synonym = {**invalid_task,
                "task": "用 open_interest_value/open_interest_base 估计未平仓平均名义价格",
                "reason": "衡量持仓者平均入场价位与现价偏离",
            }
            with self.assertRaisesRegex(ValueError, "does not provide entry price"):
                runner._check_task(invalid_synonym, a)

            miner = FactorMiner(specification(), GoalModel(), Path(directory))
            invalid_definition = definition("div(open_interest_value,open_interest_base)")
            invalid_definition.update(
                name="oi_average_entry_price",
                meaning="用持仓名义价值除以数量推断平均开仓价格",
                hypothesis="平均持仓价可以表示全市场浮动盈亏",
                change_reason="测试持仓成本偏离",
            )
            response = {"candidates": [invalid_definition], "dispositions": [], "analysis": "测试"}
            with self.assertRaisesRegex(ValueError, "does not provide entry price"):
                miner._check_ideation(response, {})

            leverage_task = {**invalid_task,
                "task": "用 open_interest_base 上升推断大户继续加杠杆",
                "reason": "检验新杠杆头寸的拥挤",
            }
            with self.assertRaisesRegex(ValueError, "does not identify leverage ratios"):
                runner._check_task(leverage_task, a)

            leverage_definition = definition("cross_rank(ts_return(open_interest_base,1))")
            leverage_definition.update(
                name="oi_leverage_up_rank",
                meaning="open_interest_base 上升表示大户继续加杠杆",
                hypothesis="新杠杆头寸会提高去杠杆风险",
                change_reason="用 OI 变化识别杠杆率上升",
            )
            response = {"candidates": [leverage_definition], "dispositions": [], "analysis": "测试"}
            with self.assertRaisesRegex(ValueError, "does not identify leverage ratios"):
                miner._check_ideation(response, {})

            direction_task = {**invalid_task,
                "task": "用 open_interest_base 上升识别方向性头寸建立阶段",
                "reason": "检验新建多头头寸的延续",
            }
            with self.assertRaisesRegex(ValueError, "open/close and long/short position direction"):
                runner._check_task(direction_task, a)

            direction_definition = definition("cross_rank(ts_return(open_interest_base,1))")
            direction_definition.update(
                name="oi_directional_position_building",
                meaning="open_interest_base 只表示当前聚合未平仓基础币数量变化",
                hypothesis="open_interest_base 上升可能是方向性头寸建立阶段",
                change_reason="用 OI 变化识别新建头寸",
            )
            response = {"candidates": [direction_definition], "dispositions": [], "analysis": "测试"}
            with self.assertRaisesRegex(ValueError, "open/close and long/short position direction"):
                miner._check_ideation(response, {})

            allowed_quantity_definition = definition("cross_rank(ts_return(open_interest_base,1))")
            allowed_quantity_definition.update(
                name="oi_quantity_change_rank",
                meaning=("open_interest_base 仅表示当前聚合未平仓基础币数量变化；"
                         "不用于推断杠杆率、开平仓或多空方向"),
                hypothesis="仅测试 OI 数量变化与未来收益的统计关系",
                change_reason="只使用当前 OI 数量变化状态",
            )
            miner._check_ideation(
                {"candidates": [allowed_quantity_definition], "dispositions": [], "analysis": "测试"}, {})

            bounded_definition = definition("cross_rank(ts_return(open_interest_base,1))")
            bounded_definition.update(
                name="oi_quantity_change_rank",
                meaning="open_interest_base 是当前聚合未平仓基础币数量，只计算其相对变化",
                hypothesis="仅测试聚合数量变化的条件关系，不用于推断杠杆率、开平仓、多空方向或参与者身份",
                change_reason="按字段目录限定语义",
            )
            miner._check_ideation(
                {"candidates": [bounded_definition], "dispositions": [], "analysis": "测试"}, {})

            for disclaimer in (
                "open_interest_base 变化仅能表示当前聚合未平仓数量的变化，"
                "不能据此推断杠杆率、加杠杆或去杠杆、开平仓方向。",
                "OI 变化状态只解释为聚合未平仓基础币数量变化；"
                "不应把该状态解释为杠杆率、加杠杆或去杠杆、开平仓方向。",
            ):
                with self.subTest(disclaimer=disclaimer):
                    _reject_unsupported_oi_cost_basis(
                        "open_interest_base", disclaimer)

    def test_rejects_unsupported_residualization_claims(self):
        with self.assertRaisesRegex(ValueError, "no regression/residualization operator"):
            _reject_unsupported_residualization(
                "subtract cross-ranked controls and call the result orthogonalized")

        for disclaimer in (
            "当前白名单没有回归或残差算子，不能把 rank/z-score 算术称为正交化。",
            "Residualization is not supported and must not be claimed.",
            "在不使用回归、残差或中性化算子的前提下，只做分层比较。",
        ):
            with self.subTest(disclaimer=disclaimer):
                _reject_unsupported_residualization(disclaimer)

        with tempfile.TemporaryDirectory() as directory:
            miner = FactorMiner(specification(), GoalModel(), Path(directory))
            invalid_definition = definition("sub(cross_rank(perp_close),cross_zscore(premium_index))")
            invalid_definition.update(
                name="orthogonalized_perp_close",
                meaning="固定系数相减后得到正交化残差",
                hypothesis="剔除共线后仍有预测力",
                change_reason="用 rank 与 z-score 加减实现中性化",
            )
            response = {"candidates": [invalid_definition], "dispositions": [], "analysis": "测试"}
            with self.assertRaisesRegex(ValueError, "no regression/residualization operator"):
                miner._check_ideation(response, {})

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
                self.assertEqual(sum(r["role"] == "calculator" for r in model.requests), calculators + 1)
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
                with self.assertRaises(ModelResponseError):
                    self.run_goal(runner, a, b)
                self.assertEqual(runner.state["status"], "error")
                self.assertEqual(runner.state["cycle"], 0)

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

    def test_same_formula_on_same_A_is_not_retested_under_new_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            model = GoalModel(duplicate=True)
            runner, a, b = self.create(directory, model, target=2)
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=self.passing_B):
                for _ in range(5):
                    runner._step(lambda: a, lambda: b, lambda: b.universe, runner.root / "ideas")
                for _ in range(4):
                    runner._step(lambda: a, lambda: b, lambda: b.universe, runner.root / "ideas")
            self.assertEqual(runner.state["cycle"], 2)
            self.assertEqual(runner.state["phase"], "select_task")
            self.assertEqual(len(runner.state["qualified_ideas"]), 1)
            second = sorted((runner.root / "runs").iterdir())[1]
            self.assertTrue((second / "a_records/candidate-0001-duplicate.json").exists())
            self.assertFalse((second / "b-access-started.json").exists())

    def test_changed_A_cannot_resume_partial_run_and_incomplete_B_cannot_reread(self):
        with tempfile.TemporaryDirectory() as directory:
            model = GoalModel()
            model.interrupt_role = "calculator"
            runner, a, b = self.create(directory, model)
            with self.assertRaises(KeyboardInterrupt):
                self.run_goal(runner, a, b)
            a.values.iloc[0, 0] += 1
            with self.assertRaisesRegex(ValueError, "A inputs changed"):
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
            with patch("crypto_quant.research.factor_mining.cli.CodexModel") as model:
                result = execute(args)
                model.assert_not_called()
            self.assertEqual(result["phase"], "select_task")
            resume = parser.parse_args(["goal-resume", "--goal-dir", str(runner.root)])
            with patch("crypto_quant.research.factor_mining.cli.CodexModel") as model, patch(
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
            args = parser.parse_args(["goal-start", "--goal", str(root / "goal.json"),
                "--contract", str(root / "contract.json"), "--universe", str(root / "universe.csv"),
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
