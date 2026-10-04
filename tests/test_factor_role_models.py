"""Four roles keep their selected transports through corrections and Goal recovery."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_factor_goal import GoalModel, SETTINGS
from test_factor_mining import input_panel, specification
from crypto_quant.research.factor_mining.cli import add_arguments
from crypto_quant.research.factor_mining.goal import GoalRunner, GoalSpec
from crypto_quant.research.factor_mining.model import FACTOR_ROLES, RoleModels
from crypto_quant.research.factor_mining.model_config import model_from_args, model_from_settings


class RecordingModel:
    def __init__(self, name, shared):
        self.name, self.shared, self.roles = name, shared, []

    def settings(self):
        return {**SETTINGS, "model": self.name}

    def complete(self, messages, **kwargs):
        self.roles.append(json.loads(messages[1]["content"])["role"])
        return self.shared.complete(messages, **kwargs)


class RoleModelTests(unittest.TestCase):
    def test_all_four_roles_use_their_models_and_save_complete_configuration(self):
        from test_factor_goal import GoalTests
        shared = GoalModel()
        roles = RoleModels({role: RecordingModel(f"scripted-{role}", shared) for role in FACTOR_ROLES})
        spec = specification(purpose="research")
        a, b = input_panel(spec), input_panel(spec, "B")
        with tempfile.TemporaryDirectory() as directory:
            runner = GoalRunner.create(GoalSpec("roles", "价格偏离研究", 1), spec, roles, Path(directory),
                                       inputs={}, model_settings=roles.settings())
            with patch("crypto_quant.research.factor_mining.workflow.evaluate_factor", side_effect=GoalTests.passing_B):
                result = runner.run(lambda: a, lambda: b, lambda: b.universe, runner.root / "ideas", poll_seconds=1)
            self.assertEqual(result["status"], "complete")
            for role, model in roles.models.items():
                self.assertTrue(model.roles)
                self.assertEqual(set(model.roles), {role})
            saved = GoalRunner(runner.root, model=None).model_settings
            self.assertEqual(saved, roles.settings())
            self.assertEqual(model_from_settings(saved).settings(), saved)
            changed = copy.deepcopy(saved)
            changed["roles"]["optimizer"]["sdk_version"] = "different"
            with self.assertRaises(ValueError):
                model_from_settings(changed)

    def test_role_profile_has_no_implicit_role_or_shared_model(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "roles.json"
            profile.write_text(json.dumps({role: {"model": f"scripted-{role}", "reasoning_effort": "max"}
                                           for role in FACTOR_ROLES}))
            flags = ["goal-start", "--goal", "g.json", "--contract", "c.json", "--universe", "u.csv",
                     "--provider", "codex", "--role-models", str(profile)]
            model = model_from_args(parser.parse_args(flags))
            self.assertIsInstance(model, RoleModels)
            self.assertEqual(set(model.models), FACTOR_ROLES)
            with self.assertRaises(ValueError):
                model_from_args(parser.parse_args(flags + ["--model", "scripted-shared"]))
            profile.write_text(json.dumps({"ideator": {"model": "scripted", "reasoning_effort": "max"}}))
            with self.assertRaises(ValueError):
                model_from_args(parser.parse_args(flags))

    def test_correction_and_evidence_requests_keep_original_role(self):
        shared = GoalModel()
        models = RoleModels({role: RecordingModel(role, shared) for role in FACTOR_ROLES})
        messages = [{"role": "system", "content": "fixture"},
                    {"role": "user", "content": json.dumps({"role": "unknown"})}]
        with self.assertRaises(ValueError):
            models.complete(messages, max_output_tokens=None, session_id="fixture")
        for role in FACTOR_ROLES:
            mock = models.models[role]
            with patch.object(mock, "complete", return_value="reply") as call:
                messages[1]["content"] = json.dumps({"role": role})
                messages.append({"role": "user", "content": json.dumps({"response_error": "correct schema"})})
                self.assertEqual(models.complete(messages, max_output_tokens=None, session_id="fixture"), "reply")
                call.assert_called_once_with(messages, max_output_tokens=None, session_id="fixture")

    def test_run_profile_enables_fast_only_for_sol_and_restores_it(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        profile = Path(__file__).resolve().parents[1] / "examples/factor_mining/role-models.json"
        model = model_from_args(parser.parse_args([
            "goal-start", "--goal", "g.json", "--contract", "c.json", "--universe", "u.csv",
            "--provider", "codex", "--role-models", str(profile)]))
        self.assertEqual({role for role, item in model.models.items() if item.service_tier is not None},
                         {"evaluator", "optimizer"})
        self.assertEqual(model.models["evaluator"].reasoning_effort, "high")
        self.assertEqual(model.models["optimizer"].reasoning_effort, "max")
        self.assertEqual(model_from_settings(model.settings()).settings(), model.settings())
