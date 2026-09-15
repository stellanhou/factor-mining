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
from .model import OpenCodeGoModel, available_go_models
from .records import write_json
from .workflow import FactorMiner


def add_arguments(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="mining_action", required=True)
    actions.add_parser("models", help="查询OpenCode Go公开模型清单，不使用密钥")
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
    for action in (explore, validate):
        action.add_argument("--db", type=Path, default=Path("market_data/crypto_quant.sqlite"))
        action.add_argument("--universe", type=Path, required=True, help="timestamp,symbol,eligible的完整小时CSV")
        action.add_argument("--include-liquidations", action="store_true")
    for action in (explore, validate, reports):
        action.add_argument("--model", required=True, help="Go裸模型ID，例如kimi-k2.6")
        action.add_argument("--protocol", choices=("chat", "messages"), required=True)
        action.add_argument("--reasoning-effort", choices=("low", "high", "max"),
                            help="显式推理强度，仅用于支持该参数的chat模型；省略则使用提供商默认值")
        action.add_argument("--api-key-env", default="OPENCODE_GO_API_KEY")
        action.add_argument("--timeout-seconds", type=int, default=180)


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
        return available_go_models()
    if args.mining_action == "freeze":
        # Freezing is deterministic and needs neither a model selection nor a key.
        miner = FactorMiner.open(args.run_dir, model=None)
        return miner.freeze(args.candidate_ids.split(","), load_membership(args.universe, miner.spec, "B"))
    model = OpenCodeGoModel(args.model, args.protocol, api_key_env=args.api_key_env, timeout_seconds=args.timeout_seconds,
                            reasoning_effort=args.reasoning_effort)
    model_settings = {"provider": "opencode-go", "model": args.model, "protocol": args.protocol,
                      "reasoning_effort": args.reasoning_effort, "timeout_seconds": args.timeout_seconds}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="加密货币因子挖掘 → 创意库")
    add_arguments(parser)
    print(dumps(execute(parser.parse_args())))
