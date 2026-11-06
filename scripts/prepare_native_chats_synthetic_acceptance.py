"""Create the fixed synthetic data/run binding used by native chat acceptance."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_quant.research.factor_mining.contracts import ResearchSpec, dumps
from crypto_quant.research.factor_mining.goal import GoalSpec
from crypto_quant.research.factor_mining.workflow import FactorMiner


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "experiments/factor_mining/native_chats/synthetic_acceptance"
RUN_ID = "native-synthetic-acceptance-20261015"
OBJECTIVE = (
    "验证原生A研究、完整清单封存、一次B数值检查与检查点解释恢复的合成工程闭环；"
    "合格因子目标为1，未达到时如实记录objective_met=false。流程完成不代表因子目标达成或正式入池。"
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT")
CHAT_BINDINGS = {
    "coordinator": {"thread_id": "01a13336-246e-7b01-8c61-4440dbe662df", "host_id": "local"},
    "A": {"thread_id": "01a13336-3017-7df2-bfe1-625c4ebf6fe5", "host_id": "local"},
    "B": {"thread_id": "01a13336-16f3-7060-9ad1-afcc655861a0", "host_id": "local"},
}


class _NoModelCalls:
    def complete(self, *args, **kwargs):
        raise AssertionError("synthetic acceptance preparation must not call a model")


def _spec() -> ResearchSpec:
    return ResearchSpec(
        run_id=RUN_ID,
        objective=OBJECTIVE,
        purpose="engineering_check",
        a_start="2026-09-01T00:00:00Z",
        b_start="2026-09-05T00:00:00Z",
        c_start="2026-09-09T00:00:00Z",
        c_end="2026-09-13T00:00:00Z",
        universe_provenance="six-asset fixed synthetic engineering acceptance fixture",
        data_usage_review="deterministic synthetic price bars; no real market observations",
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
    rng = np.random.default_rng(20261015)
    drifts = np.linspace(0.00003, 0.00009, len(SYMBOLS))
    returns = drifts[None, :] + rng.normal(0.0, 0.00007, (len(hours), len(SYMBOLS)))
    base = np.power(1.7, np.arange(1, len(SYMBOLS) + 1, dtype=float)) * 100.0
    prices = base[None, :] * np.exp(np.cumsum(returns, axis=0))
    close_noise = rng.normal(0.0, 0.0001, prices.shape)

    connection = sqlite3.connect(path)
    try:
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
        connection.commit()
    finally:
        connection.close()


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


def prepare() -> Path:
    require_paths_absent = [
        FIXTURE_ROOT,
        ROOT / ".codex/factor_research_synthetic_binding.json",
    ]
    existing = [path for path in require_paths_absent if path.exists()]
    if existing:
        raise FileExistsError("synthetic acceptance paths already exist: "
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
    miner = FactorMiner(spec, _NoModelCalls(), runs)
    goal = GoalSpec(goal_id="native-synthetic-acceptance", objective=OBJECTIVE,
                    target_ideas=1)
    miner.store.append("goal-spec", "goal_contract", goal.as_dict())
    run_root = miner.root
    contract_sha256 = hashlib.sha256((run_root / "contract.json").read_bytes()).hexdigest()

    binding_path = ROOT / ".codex/factor_research_synthetic_binding.json"
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    binding = {
        "format_version": 1,
        "research_id": RUN_ID,
        "target": {"version": 1, "objective": OBJECTIVE},
        "goal": goal.as_dict(),
        "run_root": f"../experiments/factor_mining/native_chats/synthetic_acceptance/runs/{RUN_ID}",
        "contract_sha256": contract_sha256,
        "inputs": {
            "market_db": "../experiments/factor_mining/native_chats/synthetic_acceptance/market_data.sqlite",
            "universe_csv": "../experiments/factor_mining/native_chats/synthetic_acceptance/universe.csv",
            "include_liquidations": False,
            "idea_pool": "../experiments/factor_mining/native_chats/synthetic_acceptance/idea_pool",
        },
        "chats": CHAT_BINDINGS,
    }
    binding_path.write_text(dumps(binding) + "\n", encoding="utf-8")
    return binding_path


if __name__ == "__main__":
    print(prepare())
