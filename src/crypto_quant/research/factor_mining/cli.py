"""Explicit run/freeze/validate commands for the factor-mining idea source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from crypto_quant.data_access.market_data import MarketDataStore
from crypto_quant.features.factor_inputs import load_factor_inputs, validate_universe
from .contracts import ResearchSpec, dumps, require
from .goal import GoalRunner, GoalSpec
from .codex_model import CodexModel, PROVIDER, add_model_arguments, model_from_args
from .records import write_json
from .workflow import FactorMiner


def add_arguments(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="mining_action", required=True)
    models = actions.add_parser("models", help="查询当前ChatGPT订阅可用的Codex模型")
    add_model_arguments(models)
    explore = actions.add_parser("explore", help="仅在A段运行因子挖掘")
    explore.add_argument("--contract", type=Path, required=True)
    explore.add_argument("--output-root", type=Path, default=Path("experiments/factor_mining"))
    freeze = actions.add_parser("freeze", help="读取B数据前冻结候选批次")
    freeze.add_argument("--run-dir", type=Path, required=True)
    freeze.add_argument("--candidate-ids", required=True, help="从a-complete.json的retained_ids中选择，以逗号分隔")
    freeze.add_argument("--universe", type=Path, required=True, help="冻结B段成员资格；此操作不读取收益数据")
    validate = actions.add_parser("validate", help="对冻结批次进行一次B段验证并生成符合条件的创意卡")
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.add_argument("--idea-pool", type=Path, default=Path("experiments/idea_pool"))
    reports = actions.add_parser("complete-reports", help="只根据已保存的数值结果补齐模型解释，不重读行情")
    reports.add_argument("--run-dir", type=Path, required=True)
    reports.add_argument("--stage", choices=("A", "B"), required=True)
    reports.add_argument("--idea-pool", type=Path, default=Path("experiments/idea_pool"))
    goal = actions.add_parser("goal-start", help="优化Agent负责成果目标，持续研究直至完成；Ctrl+C暂停")
    goal.add_argument("--goal", type=Path, required=True, help="goal_id、objective、target_ideas的JSON文件")
    goal.add_argument("--contract", type=Path, required=True)
    goal.add_argument("--output-root", type=Path, default=Path("experiments/factor_mining/goals"))
    goal.add_argument("--idea-pool", type=Path, default=Path("experiments/idea_pool"))
    goal.add_argument("--poll-seconds", type=float, default=60, help="等待缺失A数据时的检查间隔")
    resume = actions.add_parser("goal-resume", help="按保存的目标、输入路径和模型设置恢复Goal")
    resume.add_argument("--goal-dir", type=Path, required=True)
    status = actions.add_parser("goal-status", help="读取Goal进度，不调用模型或行情")
    status.add_argument("--goal-dir", type=Path, required=True)
    for action in (explore, validate, goal):
        action.add_argument("--db", type=Path, default=Path("market_data/crypto_quant.sqlite"))
        action.add_argument("--universe", type=Path, required=True, help="timestamp,symbol,eligible的完整小时CSV")
        action.add_argument("--include-liquidations", action="store_true")
    for action in (explore, validate, reports, goal):
        add_model_arguments(action)


def load_membership(universe_path: Path, spec: ResearchSpec, stage: str) -> pd.Series:
    start, end = spec.bounds(stage)
    rows = pd.read_csv(universe_path, dtype={"symbol": str, "eligible": str})
    require(set(rows.columns) == {"timestamp", "symbol", "eligible"}, "universe CSV requires timestamp,symbol,eligible")
    require(all(pd.Timestamp(value).tzinfo is not None for value in rows["timestamp"].unique()),
            "universe CSV timestamps must include an explicit timezone")
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="raise")
    require(rows["eligible"].isin(["true", "false", "True", "False", "1", "0"]).all(), "eligible must explicitly be boolean")
    rows["eligible"] = rows["eligible"].isin(["true", "True", "1"])
    rows = rows.loc[(rows["timestamp"] >= start - pd.Timedelta(hours=spec.max_lookback_hours)) & (rows["timestamp"] < end)]
    members = validate_universe(rows.set_index(["timestamp", "symbol"])["eligible"])
    hours = members.index.get_level_values("timestamp")
    require(hours.min() == start - pd.Timedelta(hours=spec.max_lookback_hours) and hours.max() == end - pd.Timedelta(hours=1),
            "universe CSV must cover the entire allowed stage and warm-up")
    return members


def load_stage(db: Path, universe_path: Path, spec: ResearchSpec, stage: str, *, include_liquidations: bool):
    return load_factor_inputs(MarketDataStore(db), load_membership(universe_path, spec, stage),
                             include_liquidations=include_liquidations)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.mining_action == "models":
        return model_from_args(args).available_models()
    if args.mining_action == "goal-status":
        return GoalRunner.status(args.goal_dir)
    if args.mining_action == "goal-resume":
        # Read and validate the saved contract before constructing a provider client.
        runner = GoalRunner(args.goal_dir, model=None)
        settings = runner.model_settings
        require(settings["provider"] == PROVIDER,
                "old Goal uses another provider; create a new Codex Goal without rewriting historical evidence")
        runner.model = CodexModel(settings["model"], reasoning_effort=settings["reasoning_effort"],
                                  timeout_seconds=settings["timeout_seconds"])
        require(runner.model.settings() == settings, "Codex SDK/runtime changed; create a new Goal")
        return _run_goal(runner)
    if args.mining_action == "freeze":
        # Freezing is deterministic and needs neither a model selection nor a key.
        miner = FactorMiner.open(args.run_dir, model=None)
        return miner.freeze(args.candidate_ids.split(","), load_membership(args.universe, miner.spec, "B"))
    model = model_from_args(args)
    model_settings = model.settings()
    if args.mining_action == "goal-start":
        goal = GoalSpec(**json.loads(args.goal.read_text(encoding="utf-8")))
        spec = ResearchSpec.from_dict(json.loads(args.contract.read_text(encoding="utf-8")))
        require(args.poll_seconds > 0, "poll_seconds must be positive")
        runner = GoalRunner.create(goal, spec, model, args.output_root, model_settings=model_settings, inputs={
            "db": str(args.db.resolve()), "universe": str(args.universe.resolve()),
            "idea_pool": str(args.idea_pool.resolve()), "include_liquidations": args.include_liquidations,
            "poll_seconds": args.poll_seconds})
        return _run_goal(runner)
    if args.mining_action == "explore":
        spec = ResearchSpec.from_dict(json.loads(args.contract.read_text(encoding="utf-8")))
        panel = load_stage(args.db, args.universe, spec, "A", include_liquidations=args.include_liquidations)
        miner = FactorMiner(spec, model, args.output_root)
        write_json(miner.root / "model-settings.json", model_settings)
        return {"run_directory": str(miner.root), **miner.explore(panel)}
    miner = FactorMiner.open(args.run_dir, model)
    if args.mining_action == "complete-reports":
        return miner.complete_reports(args.stage, args.idea_pool)
    write_json(miner.root / "validation-model-settings.json", model_settings)
    return miner.validate(lambda: load_stage(args.db, args.universe, miner.spec, "B",
                                            include_liquidations=args.include_liquidations), args.idea_pool)


def _run_goal(runner: GoalRunner) -> dict[str, Any]:
    inputs, spec = runner.inputs, runner.spec
    db, universe = Path(inputs["db"]), Path(inputs["universe"])
    def stage(name):
        return load_stage(db, universe, spec, name, include_liquidations=inputs["include_liquidations"])
    result = runner.run(lambda: stage("A"), lambda: stage("B"),
                        lambda: load_membership(universe, spec, "B"), Path(inputs["idea_pool"]),
                        poll_seconds=inputs["poll_seconds"])
    return {"goal_directory": str(runner.root), **result}


def main() -> None:
    parser = argparse.ArgumentParser(description="加密货币因子挖掘 → 创意库")
    add_arguments(parser)
    print(dumps(execute(parser.parse_args())))
