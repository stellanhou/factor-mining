"""Prepare an isolated, synthetic purpose=research run for DSH Goal acceptance."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_quant.research.factor_mining.contracts import ResearchSpec, dumps
from crypto_quant.research.factor_mining.goal import GoalSpec
from crypto_quant.research.factor_mining.native_chats import load_native_adapter
from crypto_quant.research.factor_mining.workflow import FactorMiner


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "experiments/factor_mining/native_goal_acceptance_20261017"
RUN_ID = "native-goal-acceptance-synthetic-20261017"
OBJECTIVE = (
    "只用本地确定性合成数据验证因子挖掘合同、Coordinator 到 A 的合法交接、"
    "完整 A/B 证据核验与原生 DSH Goal 完成门禁；目标为1个达标候选。"
    "本运行是合成工程验收，不构成真实市场研究、实盘因子或投资结论。"
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT")
CHAT_BINDINGS = {
    "coordinator": {"thread_id": "session-e76e1822-3b09-4ef0-9e39-44f2d09d9f5e", "host_id": "local"},
    "A": {"thread_id": "session-00149626-78ac-4061-8e4b-0cdec703d8f0", "host_id": "local"},
    "B": {"thread_id": "session-68f8400e-3771-4af2-829f-176b6889abaf", "host_id": "local"},
}


class _NoModelCalls:
    def complete(self, *args, **kwargs):
        raise AssertionError("synthetic acceptance preparation must not call a model")


def _spec() -> ResearchSpec:
    return ResearchSpec(
        run_id=RUN_ID,
        objective=OBJECTIVE,
        purpose="research",
        a_start="2022-08-01T00:00:00Z",
        b_start="2024-08-01T00:00:00Z",
        c_start="2025-08-01T00:00:00Z",
        c_end="2026-08-01T00:00:00Z",
        universe_provenance="six-asset deterministic synthetic fixture; engineering acceptance only",
        data_usage_review="all price bars are locally generated synthetic values; no real market observations",
        label="perp_next_open_24h",
        sample_hours=1,
        groups=3,
        min_symbols=6,
        min_periods=26,
        hac_lags=23,
        confidence=0.95,
        stage_hours=48,
        rolling_periods=26,
        fdr_method="BH",
        fdr_alpha=0.05,
        min_abs_ic=0.02,
        min_directional_spread=0.0001,
        min_stage_share=0.5,
        max_repairs=1,
        max_formula_nodes=80,
        max_lookback_hours=3,
        context_tokens=2_000_000,
        output_tokens=4000,
        b_horizons=(1, 4, 24),
    )


def _create_market_database(path: Path, spec: ResearchSpec) -> None:
    start, _ = spec.bounds("A")
    _, end = spec.bounds("B")
    first = start - pd.Timedelta(days=30)
    hours = pd.date_range(first, end, freq="h", inclusive="left")
    rng = np.random.default_rng(20261017)
    drifts = np.linspace(0.00003, 0.00009, len(SYMBOLS))
    returns = drifts[None, :] + rng.normal(0.0, 0.00007, (len(hours), len(SYMBOLS)))
    base = np.power(1.7, np.arange(1, len(SYMBOLS) + 1, dtype=float)) * 100.0
    prices = base[None, :] * np.exp(np.cumsum(returns, axis=0))
    close_noise = rng.normal(0.0, 0.0001, prices.shape)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE futures_price_bars (
                data_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                close_time INTEGER NOT NULL,
                quote_volume REAL NOT NULL,
                trades INTEGER NOT NULL,
                taker_buy_base_volume REAL NOT NULL,
                taker_buy_quote_volume REAL NOT NULL,
                source_file_id INTEGER NOT NULL
            )"""
        )
        rows = []
        for row_index, timestamp in enumerate(hours):
            open_time = int(timestamp.value // 1_000_000)
            close_time = open_time + 3_599_999
            for symbol_index, symbol in enumerate(SYMBOLS):
                open_price = float(prices[row_index, symbol_index])
                close_price = float(open_price * np.exp(close_noise[row_index, symbol_index]))
                rows.append(("klines", symbol, "1h", open_time,
                             open_price, max(open_price, close_price),
                             min(open_price, close_price), close_price, 100.0,
                             close_time, close_price * 100.0, 10, 50.0,
                             close_price * 50.0, 1))
        connection.executemany(
            "INSERT INTO futures_price_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
        )


def _create_universe(path: Path, spec: ResearchSpec) -> None:
    start = pd.Timestamp(spec.a_start) - pd.Timedelta(hours=spec.max_lookback_hours)
    end = pd.Timestamp(spec.c_start)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    frame = pd.DataFrame({
        "timestamp": np.repeat(hours.astype(str), len(SYMBOLS)),
        "symbol": list(SYMBOLS) * len(hours),
        "eligible": ["true"] * (len(hours) * len(SYMBOLS)),
    })
    frame.to_csv(path, index=False)


def prepare() -> tuple[Path, dict[str, object]]:
    binding_path = ROOT / ".codex/factor_research_goal_acceptance_binding.json"
    existing = [path for path in (FIXTURE_ROOT, binding_path) if path.exists()]
    if existing:
        raise FileExistsError("synthetic Goal acceptance paths already exist: "
                              + ", ".join(str(path) for path in existing))

    FIXTURE_ROOT.mkdir(parents=True)
    runs = FIXTURE_ROOT / "runs"
    runs.mkdir()
    database = FIXTURE_ROOT / "market_data.sqlite"
    universe = FIXTURE_ROOT / "universe.csv"
    idea_pool = FIXTURE_ROOT / "idea_pool"
    idea_pool.mkdir()

    spec = _spec()
    _create_market_database(database, spec)
    _create_universe(universe, spec)
    goal = GoalSpec(goal_id="native-goal-acceptance-domain", objective=OBJECTIVE,
                    target_ideas=1)
    miner = FactorMiner(spec, _NoModelCalls(), runs)
    miner.store.append("goal-spec", "goal_contract", goal.as_dict())
    run_root = miner.root.resolve()
    contract_sha256 = hashlib.sha256((run_root / "contract.json").read_bytes()).hexdigest()

    binding = {
        "format_version": 1,
        "research_id": RUN_ID,
        "target": {"version": 1, "objective": OBJECTIVE},
        "goal": goal.as_dict(),
        "run_root": f"../experiments/factor_mining/native_goal_acceptance_20261017/runs/{RUN_ID}",
        "contract_sha256": contract_sha256,
        "inputs": {
            "market_db": "../experiments/factor_mining/native_goal_acceptance_20261017/market_data.sqlite",
            "universe_csv": "../experiments/factor_mining/native_goal_acceptance_20261017/universe.csv",
            "include_liquidations": False,
            "idea_pool": "../experiments/factor_mining/native_goal_acceptance_20261017/idea_pool",
        },
        "chats": CHAT_BINDINGS,
    }
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    binding_path.write_text(dumps(binding) + "\n", encoding="utf-8")
    ticket = load_native_adapter(binding_path, "coordinator").prepare_a_task()
    return binding_path, {"research_id": RUN_ID, "purpose": spec.purpose, "task_id": ticket["task_id"]}


if __name__ == "__main__":
    binding, info = prepare()
    print(json.dumps({"binding": str(binding), **info}, ensure_ascii=False))
