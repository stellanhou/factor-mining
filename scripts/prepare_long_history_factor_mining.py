"""Prepare and verify the 2022-2026 mining dataset without starting model calls.

Reuse the existing fixed-cohort design, selecting before A's warm-up. Never
filter membership by future survival, field coverage, factor performance or B.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_quant.data_access.market_data import resolve_market_symbols
from crypto_quant.features.factor_expressions import evaluate_expression
from crypto_quant.research.factor_mining.cli import load_membership, load_stage
from crypto_quant.research.factor_mining.contracts import ResearchSpec
from crypto_quant.research.factor_mining.evaluation import build_labels
from crypto_quant.research.factor_mining.workflow import validate_panel


HOUR = 3_600_000
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments/factor_mining/v6_inputs"
DEFAULT_CONTRACT = ROOT / "examples/factor_mining/research.contract.json"
EXCLUDED = set("USDC USDP TUSD BUSD FDUSD DAI USDE USD1 USDS XUSD U AEUR EURI EUR GBP AUD BRL TRY RUB BIDR IDRT BVND UST USTC PAX PAXG XAUT".split())


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def millis(value: pd.Timestamp) -> int:
    return int(value.timestamp() * 1000)


def bars(conn, symbol: str, start: pd.Timestamp, end: pd.Timestamp, *, perpetual: bool):
    table = "futures_price_bars" if perpetual else "klines"
    extra = " AND data_type='klines'" if perpetual else ""
    return conn.execute(
        f"SELECT open_time,open,high,low,close,volume,quote_volume,close_time "
        f"FROM {table} WHERE symbol=? AND interval='1h'{extra} "
        "AND open_time>=? AND open_time<? ORDER BY open_time",
        (symbol, millis(start), millis(end)),
    ).fetchall()


def valid_bars(rows) -> np.ndarray:
    if not rows:
        return np.zeros(0, dtype=bool)
    data = np.asarray(rows, dtype=float)
    _, open_, high, low, close, volume, quote, close_time = data.T
    return (np.isfinite(data).all(axis=1) & (data[:, 1:5].min(axis=1) > 0)
            & (volume >= 0) & (quote >= 0) & (low <= np.minimum(open_, close))
            & (np.maximum(open_, close) <= high) & (close_time == data[:, 0] + HOUR - 1))


def select_cohort(conn, selection: pd.Timestamp) -> list[dict]:
    # Inventory identifies names only. No last-observation/end-of-window filter.
    symbols = {r[0] for r in conn.execute("SELECT DISTINCT symbol FROM klines")}
    perpetuals = {r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM futures_archive_files WHERE category='klines' AND interval='1h'")}
    start = selection - pd.Timedelta(days=30)
    expected = np.arange(millis(start), millis(selection), HOUR)
    ranking = []
    for symbol in sorted(symbols):
        if not symbol.endswith("USDT"):
            continue
        base = symbol[:-4]
        leveraged = any(base.endswith(suffix) and base[:-len(suffix)] + "USDT" in symbols
                        for suffix in ("UP", "DOWN", "BULL", "BEAR"))
        if base in EXCLUDED or base in {"BULL", "BEAR"} or leveraged:
            continue
        perpetual = resolve_market_symbols(symbol).perpetual
        if perpetual not in perpetuals:
            continue
        spot = bars(conn, symbol, start, selection, perpetual=False)
        if len(spot) != 720 or not valid_bars(spot).all():
            continue
        perp = bars(conn, perpetual, start, selection, perpetual=True)
        if len(perp) != 720 or not valid_bars(perp).all():
            continue
        if not all(np.array_equal(np.array(r)[:, 0], expected) for r in (spot, perp)):
            continue
        daily_quote = np.array(spot)[:, 6].reshape(30, 24).sum(axis=1)
        ranking.append({"symbol": symbol, "perpetual": perpetual,
                        "median_daily_quote_volume": float(np.median(daily_quote))})
    ranking.sort(key=lambda row: (-row["median_daily_quote_volume"], row["symbol"]))
    if len(ranking) < 30:
        raise ValueError(f"only {len(ranking)} historical candidates; need 30")
    return ranking


def prepare(db: Path, output: Path, contract: Path = DEFAULT_CONTRACT) -> None:
    spec = ResearchSpec.from_dict(json.loads(contract.read_text()))
    output.mkdir(parents=True, exist_ok=False)
    start = pd.Timestamp(spec.a_start) - pd.Timedelta(hours=spec.max_lookback_hours)
    end = pd.Timestamp(spec.c_start)
    conn = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    ranking = select_cohort(conn, start)
    save(output / "historical-ranking.json", {"selection_time": start.isoformat(),
         "ranking_start": (start - pd.Timedelta(days=30)).isoformat(),
         "ranking_end_exclusive": start.isoformat(), "excluded_bases": sorted(EXCLUDED),
         "ranking": ranking, "selected": ranking[:30]})
    print("Historical cohort:", [row["symbol"] for row in ranking[:30]], flush=True)
    times = pd.date_range(start, end, freq="h", inclusive="left")
    parts, coverage = [], []
    for item in ranking[:30]:
        masks = []
        for perpetual, name in [(False, item["symbol"]), (True, item["perpetual"])]:
            rows = bars(conn, name, start, end, perpetual=perpetual)
            good = valid_bars(rows)
            valid_times = pd.to_datetime([row[0] for row, ok in zip(rows, good) if ok], unit="ms", utc=True)
            mask = times.isin(valid_times)
            masks.append(mask)
            coverage.append({"symbol": item["symbol"], "market": "perpetual" if perpetual else "spot",
                             "rows": len(rows), "invalid_rows": int((~good).sum()),
                             "missing_or_invalid_hours": int((~mask).sum())})
        parts.append(pd.DataFrame({"timestamp": times, "symbol": item["symbol"],
                                   "eligible": masks[0] & masks[1]}))
    conn.close()
    universe = pd.concat(parts).sort_values(["timestamp", "symbol"])
    universe.to_csv(output / "universe.csv", index=False)
    hourly = universe.groupby("timestamp")["eligible"].sum()
    segments = {}
    for stage in ("A", "B"):
        begin, finish = spec.bounds(stage)
        counts = hourly.loc[(hourly.index >= begin) & (hourly.index < finish)]
        segments[stage] = {"hours": len(counts), "eligible_asset_hours": int(counts.sum()),
                           "min_assets": int(counts.min()), "max_assets": int(counts.max()),
                           "hours_below_min_symbols": int((counts < spec.min_symbols).sum()),
                           "below_min_symbols_by_day": {str(k.date()): int(v) for k, v in
                               (counts < spec.min_symbols).resample("D").sum().items() if v}}
        if not (counts >= spec.min_symbols).any():
            raise ValueError(f"{stage} has no usable cross sections")
    save(output / "contract.json", spec.as_dict())
    save(output / "universe-summary.json", {"start": start.isoformat(), "end_exclusive": end.isoformat(),
         "rows": len(universe), "segments": segments, "coverage": coverage})
    save(output / "data-usage.json", {"A": [spec.a_start, spec.b_start], "B": [spec.b_start, spec.c_start],
         "nominal_C": [spec.c_start, spec.c_end], "C_is_untouched": False,
         "C_reason": "2026-04 through 2026-07 previously used for factor development and selection",
         "C_loaded_by_this_setup": False, "include_liquidations": False,
         "database": str(db.resolve()), "database_size": db.stat().st_size,
         "database_mtime_ns": db.stat().st_mtime_ns,
         "contract_source": str(contract.resolve())})
    print(json.dumps(segments, ensure_ascii=False), flush=True)


def verify(db: Path, output: Path) -> None:
    spec = ResearchSpec.from_dict(json.loads((output / "contract.json").read_text()))
    a = load_membership(output / "universe.csv", spec, "A")
    b = load_membership(output / "universe.csv", spec, "B")
    overlap = b.index.get_level_values("timestamp") < pd.Timestamp(spec.b_start)
    assert a.reindex(b.loc[overlap].index).equals(b.loc[overlap]), "A/B warm-up membership mismatch"
    del a, b
    for stage in ("A", "B"):
        print(f"Loading full {stage} through production input loader", flush=True)
        panel = load_stage(db, output / "universe.csv", spec, stage, include_liquidations=False)
        validate_panel(panel, spec, stage)
        begin, end = spec.bounds(stage)
        hours = panel.values.index.get_level_values("timestamp")
        in_stage = (hours >= begin) & (hours < end)
        eligible = panel.universe & in_stage
        assert not np.isinf(panel.values.to_numpy()).any()
        fields = {}
        for field in panel.values:
            good = panel.values[field].notna() & eligible
            counts = good.loc[in_stage].groupby(level="timestamp").sum()
            fields[field] = {"eligible_valid_rows": int(good.sum()),
                             "eligible_coverage": float(good.sum() / eligible.sum()),
                             "hours_at_least_min_symbols": int((counts >= spec.min_symbols).sum())}
        # A deterministic input smoke check; no IC, p-values or B-based selection.
        result = evaluate_expression("cross_rank(div(perp_close, ts_delay(perp_close, 24)))", panel)
        labels = build_labels(panel, spec, stage)
        matched = result.values.notna() & labels["forward_return"].notna() & eligible
        counts = matched.loc[in_stage].groupby(level="timestamp").sum()
        assert (counts >= spec.min_symbols).any(), f"no executable {stage} label/factor pairs"
        assert (labels.loc[labels["forward_return"].notna(), "label_end"] < end).all()
        summary = {"stage": stage, "start": begin.isoformat(), "end_exclusive": end.isoformat(),
                   "panel_rows_with_warmup": len(panel.values), "eligible_asset_hours": int(eligible.sum()),
                   "symbols": len(panel.values.index.get_level_values("symbol").unique()),
                   "fields": fields, "input_smoke_expression": result.definition,
                   "smoke_usable_hours": int((counts >= spec.min_symbols).sum()),
                   "label_boundary_checked": True, "infinite_values": 0,
                   "no_factor_performance_tests_run": True}
        save(output / f"{stage}-input-verification.json", summary)
        save(output / f"{stage}-input-diagnostics.json", panel.diagnostics)
        print(stage, "usable hours:", summary["smoke_usable_hours"], flush=True)
        del panel, result, labels, matched, counts
        gc.collect()
    usage = json.loads((output / "data-usage.json").read_text())
    assert db.stat().st_size == usage["database_size"] and db.stat().st_mtime_ns == usage["database_mtime_ns"], "database changed during preparation"
    save(output / "ready.json", {"status": "inputs_verified", "model_run_started": False,
         "C_is_untouched": False, "C_loaded": False,
         "sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                    for name in ("contract.json", "universe.csv", "historical-ranking.json",
                                 "A-input-verification.json", "B-input-verification.json", "data-usage.json")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--db", type=Path, default=ROOT / "market_data/crypto_quant.sqlite")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT,
                        help="FM-v6 research contract used by prepare")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.db, args.output_dir, args.contract)
    else:
        verify(args.db, args.output_dir)
