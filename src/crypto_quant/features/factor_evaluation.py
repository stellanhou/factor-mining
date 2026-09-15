"""Exploratory cross-sectional evaluation for causal local factors."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from crypto_quant.data_access.data import interval_to_ms
from crypto_quant.features.factors import FACTOR_COLUMNS, FactorEngine
from crypto_quant.data_access.market_data import MarketDataStore, USD_M_PERPETUAL, _to_utc_timestamp, resolve_market_symbols
from crypto_quant.research.provenance import build_source_manifest
from crypto_quant.research.reporting import _sanitize_json


DEFAULT_FACTOR_UNIVERSE = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "SOLUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT",
    "LINKUSDT", "BCHUSDT", "ETCUSDT", "ATOMUSDT", "UNIUSDT",
    "FILUSDT", "AAVEUSDT", "TRXUSDT", "XLMUSDT", "NEARUSDT",
)

# Raw contract/base-asset open interest is not comparable across symbols, while
# observation ages are diagnostics rather than alpha signals. Keep all three in
# FactorEngine for time-series research, but exclude them from cross-sectional
# ranking.
NON_CROSS_SECTIONAL_FACTORS = frozenset(
    {"open_interest", "funding_age_hours", "metrics_age_minutes"}
)
CROSS_SECTIONAL_FACTOR_COLUMNS = tuple(
    factor for factor in FACTOR_COLUMNS if factor not in NON_CROSS_SECTIONAL_FACTORS
)


@dataclass(frozen=True)
class FactorEvaluationConfig:
    symbols: tuple[str, ...]
    interval: str = "1h"
    start: str = "2023-01-01"
    test_start: str = "2025-01-01"
    end: str = "2026-07-31"
    base_market: str = USD_M_PERPETUAL
    cost_bps: float = 10.0
    quantile: float = 0.20
    min_cross_section: int = 5

    def validate(self) -> None:
        interval_to_ms(self.interval)
        canonical = [resolve_market_symbols(symbol).perpetual for symbol in self.symbols]
        if len(set(canonical)) != len(canonical):
            raise ValueError("symbol universe contains duplicate assets or market aliases")
        if len(set(self.symbols)) < self.min_cross_section:
            raise ValueError("symbol universe is smaller than min_cross_section")
        if not 0.0 < self.quantile < 0.5:
            raise ValueError("quantile must be in (0, 0.5)")
        if self.cost_bps < 0.0:
            raise ValueError("cost_bps cannot be negative")
        if self.min_cross_section < 3:
            raise ValueError("min_cross_section must be at least 3")
        start = _to_utc_timestamp(self.start)
        test = _to_utc_timestamp(self.test_start)
        end = _to_utc_timestamp(self.end)
        if not start < test <= end:
            raise ValueError("dates must satisfy start < test_start <= end")


def _spearman(left: pd.Series, right: pd.Series) -> float:
    joined = pd.concat([left, right], axis=1).dropna()
    if len(joined) < 3 or joined.iloc[:, 0].nunique() < 2 or joined.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman"))


def build_factor_panel(
    engine: FactorEngine,
    config: FactorEvaluationConfig,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """Build a signal-time panel and next-bar open-to-close target return."""
    config.validate()
    end = _to_utc_timestamp(config.end)
    extended_end = end + pd.Timedelta(milliseconds=interval_to_ms(config.interval))
    frames: List[pd.DataFrame] = []
    excluded: List[Dict[str, str]] = []
    for raw_symbol in config.symbols:
        symbol = str(raw_symbol).upper()
        try:
            factors = engine.load(
                symbol,
                interval=config.interval,
                start=config.start,
                end=extended_end,
                base_market=config.base_market,
            )
        except (FileNotFoundError, ValueError) as exc:
            excluded.append({"symbol": symbol, "reason": str(exc)})
            continue
        panel = factors[list(FACTOR_COLUMNS)].copy()
        panel["forward_return"] = (
            factors["close"].shift(-1) / factors["open"].shift(-1) - 1.0
        )
        panel["available_at"] = factors["available_at"]
        panel = panel.loc[
            (panel.index >= _to_utc_timestamp(config.start))
            & (panel.index <= end)
        ]
        panel["symbol"] = symbol
        panel.index.name = "timestamp"
        frames.append(panel.reset_index())
    if not frames:
        raise ValueError("no symbols produced factor panel rows")
    output = pd.concat(frames, ignore_index=True)
    output = output.dropna(subset=["forward_return"])
    output = output.set_index(["timestamp", "symbol"]).sort_index()
    if output.empty:
        raise ValueError("factor panel has no rows with a forward return")
    return output, excluded


def _cross_sectional_ic(
    frame: pd.DataFrame,
    factor: str,
    min_cross_section: int,
) -> pd.Series:
    rows: Dict[pd.Timestamp, float] = {}
    for timestamp, group in frame[[factor, "forward_return"]].groupby(level="timestamp"):
        clean = group.dropna()
        if len(clean) < min_cross_section:
            continue
        value = _spearman(clean[factor], clean["forward_return"])
        if math.isfinite(value):
            rows[pd.Timestamp(timestamp)] = value
    return pd.Series(rows, name="ic", dtype=float).sort_index()


def _portfolio_path(
    frame: pd.DataFrame,
    factor: str,
    direction: float,
    config: FactorEvaluationConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    previous: Dict[str, float] = {}
    returns: List[Dict[str, Any]] = []
    weights: List[Dict[str, Any]] = []
    for timestamp, group in frame[[factor, "forward_return"]].groupby(level="timestamp"):
        clean = group.dropna().reset_index()
        if len(clean) < config.min_cross_section:
            continue
        clean["score"] = clean[factor] * direction
        if clean["score"].nunique() < 2:
            continue
        clean["rank_pct"] = clean["score"].rank(method="average", pct=True)
        short_symbols = clean.loc[
            clean["rank_pct"] <= config.quantile, "symbol"
        ].tolist()
        long_symbols = clean.loc[
            clean["rank_pct"] > 1.0 - config.quantile, "symbol"
        ].tolist()
        if not short_symbols or not long_symbols:
            continue
        current = {
            symbol: -1.0 / len(short_symbols) for symbol in short_symbols
        }
        current.update(
            {symbol: 1.0 / len(long_symbols) for symbol in long_symbols}
        )
        all_symbols = set(previous) | set(current)
        turnover = 0.5 * sum(
            abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0))
            for symbol in all_symbols
        )
        by_symbol = clean.set_index("symbol")["forward_return"]
        gross_return = sum(current[symbol] * float(by_symbol.loc[symbol]) for symbol in current)
        cost = turnover * config.cost_bps / 10_000.0
        net_return = gross_return - cost
        returns.append(
            {
                "timestamp": timestamp,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "cost": cost,
                "eligible_symbols": int(len(clean)),
                "long_leg_symbols": len(long_symbols),
                "short_leg_symbols": len(short_symbols),
            }
        )
        weights.extend(
            {"timestamp": timestamp, "symbol": symbol, "weight": weight}
            for symbol, weight in sorted(current.items())
        )
        previous = current
    path = pd.DataFrame(returns)
    if path.empty:
        return path, pd.DataFrame(weights)
    path = path.set_index("timestamp").sort_index()
    path["equity"] = (1.0 + path["net_return"]).cumprod()
    return path, pd.DataFrame(weights)


def _path_metrics(path: pd.DataFrame, interval: str) -> Dict[str, Any]:
    if path.empty or len(path) < 2:
        return {
            "active_bars": int(len(path)), "total_return": 0.0,
            "annualized_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
            "mean_turnover": 0.0, "annualized_turnover": 0.0, "total_cost": 0.0,
        }
    annualization = (365.25 * interval_to_ms("1d")) / interval_to_ms(interval)
    values = path["net_return"].astype(float)
    volatility = float(values.std(ddof=0))
    sharpe = float(values.mean() / volatility * math.sqrt(annualization)) if volatility > 0 else 0.0
    total_return = float(path["equity"].iloc[-1] - 1.0)
    years = len(path) / annualization
    annualized_return = float(path["equity"].iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and path["equity"].iloc[-1] > 0 else -1.0
    drawdown = path["equity"] / path["equity"].cummax() - 1.0
    return {
        "active_bars": int(len(path)),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "mean_turnover": float(path["turnover"].mean()),
        "annualized_turnover": float(path["turnover"].mean() * annualization),
        "total_cost": float(path["cost"].sum()),
    }


def _period_stability(ic: pd.Series) -> Tuple[pd.DataFrame, float, float]:
    if ic.empty:
        return pd.DataFrame(columns=["period", "mean_ic", "observations"]), 0.0, 0.0
    values = ic.to_frame("ic")
    period_index = values.index
    if period_index.tz is not None:
        period_index = period_index.tz_convert("UTC").tz_localize(None)
    values["period"] = period_index.to_period("Q").astype(str)
    grouped = values.groupby("period")["ic"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["period", "mean_ic", "observations"]
    return grouped, float((grouped["mean_ic"] > 0).mean()), float(grouped["mean_ic"].min())


def evaluate_factor_panel(
    panel: pd.DataFrame,
    config: FactorEvaluationConfig,
    factors: Sequence[str] = CROSS_SECTIONAL_FACTOR_COLUMNS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate factors with training-fixed direction and untouched test calculations."""
    config.validate()
    test_start = _to_utc_timestamp(config.test_start)
    train = panel.loc[panel.index.get_level_values("timestamp") < test_start]
    test = panel.loc[panel.index.get_level_values("timestamp") >= test_start]
    if train.empty or test.empty:
        raise ValueError("factor panel must contain both training and test rows")
    results: List[Dict[str, Any]] = []
    ic_rows: List[pd.DataFrame] = []
    period_rows: List[pd.DataFrame] = []
    path_rows: List[pd.DataFrame] = []
    for factor in factors:
        if factor not in panel.columns:
            continue
        train_ic = _cross_sectional_ic(train, factor, config.min_cross_section)
        raw_test_ic = _cross_sectional_ic(test, factor, config.min_cross_section)
        train_mean = float(train_ic.mean()) if not train_ic.empty else 0.0
        direction = 1.0 if train_mean >= 0.0 else -1.0
        test_ic = raw_test_ic * direction
        path, _ = _portfolio_path(test, factor, direction, config)
        metrics = _path_metrics(path, config.interval)
        symbol_ics: List[float] = []
        for _, group in test[[factor, "forward_return"]].groupby(level="symbol"):
            value = _spearman(group[factor] * direction, group["forward_return"])
            if math.isfinite(value):
                symbol_ics.append(value)
        periods, positive_period_share, worst_period_ic = _period_stability(test_ic)
        if not periods.empty:
            periods.insert(0, "factor", factor)
            period_rows.append(periods)
        if not test_ic.empty:
            table = test_ic.rename("oriented_ic").to_frame().reset_index()
            table.insert(0, "factor", factor)
            ic_rows.append(table)
        if not path.empty:
            table = path.reset_index()
            table.insert(0, "factor", factor)
            path_rows.append(table)
        test_std = float(test_ic.std(ddof=0)) if len(test_ic) else 0.0
        results.append(
            {
                "factor": factor,
                "direction": int(direction),
                "training_ic_mean": train_mean,
                "training_ic_observations": int(len(train_ic)),
                "test_ic_mean": float(test_ic.mean()) if len(test_ic) else 0.0,
                "test_ic_ir": float(test_ic.mean() / test_std) if test_std > 0 else 0.0,
                "test_ic_positive_share": float((test_ic > 0).mean()) if len(test_ic) else 0.0,
                "test_ic_observations": int(len(test_ic)),
                "symbol_positive_share": float(np.mean(np.asarray(symbol_ics) > 0.0)) if symbol_ics else 0.0,
                "symbol_median_ic": float(np.median(symbol_ics)) if symbol_ics else 0.0,
                "positive_quarter_share": positive_period_share,
                "worst_quarter_ic": worst_period_ic,
                **metrics,
            }
        )
    scores = pd.DataFrame(results)
    if scores.empty:
        raise ValueError("no factor had enough observations for evaluation")
    for source, target, ascending in (
        ("test_ic_mean", "ic_rank", True),
        ("sharpe", "sharpe_rank", True),
        ("annualized_turnover", "low_turnover_rank", False),
    ):
        scores[target] = scores[source].rank(pct=True, ascending=ascending)
    scores["selection_score"] = 100.0 * (
        0.30 * scores["ic_rank"]
        + 0.35 * scores["sharpe_rank"]
        + 0.15 * scores["symbol_positive_share"]
        + 0.10 * scores["positive_quarter_share"]
        + 0.10 * scores["low_turnover_rank"]
    )
    scores["eligible_research_lead"] = (
        (scores["training_ic_observations"] >= 100)
        & (scores["test_ic_observations"] >= 100)
        & (scores["active_bars"] >= 100)
        & (scores["test_ic_mean"] > 0.0)
        & (scores["sharpe"] > 0.0)
        & (scores["total_return"] > 0.0)
        & (scores["symbol_positive_share"] >= 0.50)
        & (scores["positive_quarter_share"] >= 0.50)
    )

    def failures(row: pd.Series) -> str:
        output: List[str] = []
        if int(row["training_ic_observations"]) < 100 or int(row["test_ic_observations"]) < 100:
            output.append("insufficient_ic_observations")
        if int(row["active_bars"]) < 100:
            output.append("insufficient_portfolio_bars")
        if float(row["test_ic_mean"]) <= 0.0:
            output.append("nonpositive_test_ic")
        if float(row["sharpe"]) <= 0.0 or float(row["total_return"]) <= 0.0:
            output.append("nonpositive_post_cost_performance")
        if float(row["symbol_positive_share"]) < 0.50:
            output.append("cross_symbol_instability")
        if float(row["positive_quarter_share"]) < 0.50:
            output.append("quarterly_instability")
        return ",".join(output)

    scores["lead_gate_failures"] = scores.apply(failures, axis=1)
    scores = scores.sort_values(
        ["eligible_research_lead", "selection_score", "factor"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return (
        scores,
        pd.concat(ic_rows, ignore_index=True) if ic_rows else pd.DataFrame(),
        pd.concat(period_rows, ignore_index=True) if period_rows else pd.DataFrame(),
        pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame(),
    )


def _report(payload: Dict[str, Any], scores: pd.DataFrame) -> str:
    columns = [
        "factor", "direction", "eligible_research_lead", "selection_score", "test_ic_mean", "test_ic_ir",
        "sharpe", "total_return", "max_drawdown", "annualized_turnover",
        "symbol_positive_share", "positive_quarter_share",
    ]
    return "\n".join(
        [
            "# Causal Cross-Sectional Factor Evaluation",
            "",
            f"Run: `{payload['run_id']}`",
            f"Decision: **`{payload['decision']}`**",
            "",
            "The training sample fixes factor direction. The test sample ranks research leads;",
            "because it is used for selection, it is not untouched confirmation for a resulting strategy.",
            "Signals are known at each base bar close and target the next bar open-to-close return.",
            "The declared major-symbol universe is fixed and may contain survivorship/listing bias;",
            "this first batch is exploratory rather than a full historical-universe simulation.",
            "Raw base-unit open interest and data-age diagnostics are excluded from cross-sectional ranking.",
            "",
            f"Universe requested: {len(payload['config']['symbols'])}; loaded: {payload['loaded_symbol_count']}; excluded: {len(payload['excluded_symbols'])}.",
            f"Panel rows: {payload['panel_rows']}; timestamps: {payload['panel_timestamps']}.",
            "",
            "## Factor Ranking",
            "",
            scores[columns].to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Selection Caveat",
            "",
            "Any strategy created from this ranking must receive a new forward or later-period validation window before promotion.",
            "",
        ]
    )


def run_factor_evaluation(
    db_path: Path,
    output_root: Path,
    config: FactorEvaluationConfig,
) -> Path:
    config.validate()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "_causal_factor_evaluation"
    run_dir = Path(output_root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    engine = FactorEngine(MarketDataStore(db_path))
    panel, excluded = build_factor_panel(engine, config)
    scores, ic, periods, paths = evaluate_factor_panel(panel, config)
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "study": "causal_cross_sectional_factor_evaluation",
        "created_at": now.isoformat(),
        "status": "complete",
        "decision": "preliminary_only",
        "config": asdict(config),
        "loaded_symbol_count": int(panel.index.get_level_values("symbol").nunique()),
        "excluded_symbols": excluded,
        "panel_rows": int(len(panel)),
        "panel_timestamps": int(panel.index.get_level_values("timestamp").nunique()),
        "factor_count": int(len(scores)),
        "excluded_from_cross_section": sorted(NON_CROSS_SECTIONAL_FACTORS),
        "eligible_factor_count": int(scores["eligible_research_lead"].sum()),
        "top_factors": scores.loc[scores["eligible_research_lead"]].head(10).to_dict(orient="records"),
        "selection_notice": "test sample was used to rank factors and cannot confirm a selected strategy",
        "universe_notice": "fixed major-symbol universe may contain survivorship and listing-coverage bias",
        "source_provenance": build_source_manifest(),
        "research_evidence_source": str(run_dir / "results.json"),
        "report_path": str(run_dir / "report.md"),
    }
    (run_dir / "results.json").write_text(
        json.dumps(_sanitize_json(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    scores.to_csv(run_dir / "factor_scores.csv", index=False)
    ic.to_csv(run_dir / "cross_sectional_ic.csv", index=False)
    periods.to_csv(run_dir / "quarterly_ic.csv", index=False)
    paths.to_csv(run_dir / "portfolio_paths.csv", index=False)
    (run_dir / "report.md").write_text(_report(payload, scores), encoding="utf-8")
    ledger_record = {
        "run_id": run_id,
        "study": payload["study"],
        "created_at": payload["created_at"],
        "status": payload["status"],
        "decision": payload["decision"],
        "factor_count": payload["factor_count"],
        "loaded_symbol_count": payload["loaded_symbol_count"],
        "top_factors": [item["factor"] for item in payload["top_factors"]],
        "source_provenance": payload["source_provenance"],
        "report_path": payload["report_path"],
    }
    ledger_path = Path(output_root) / "ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_sanitize_json(ledger_record), allow_nan=False) + "\n")
    return run_dir
