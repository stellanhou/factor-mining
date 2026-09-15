"""Regression checks for crypto symbol units, missing fields and liquidation timing."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_quant.data_access.market_data import (
    LIQUIDATION_VALUE_COLUMNS, METRICS_COLUMNS, MarketDataStore,
    resolve_market_symbols,
)
from crypto_quant.features.factors import FactorEngine
from crypto_quant.features.factor_evaluation import FactorEvaluationConfig
from crypto_quant.data_access.liquidation_data import build_liquidation_universe


class CryptoInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "crypto_quant.sqlite"
        self.start = pd.Timestamp("2026-01-01", tz="UTC")
        self.hours = pd.date_range(self.start, periods=32, freq="h", name="timestamp")
        self.conn = sqlite3.connect(self.db)
        self.addCleanup(self.conn.close)
        columns = """symbol TEXT, interval TEXT, open_time INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            close_time INTEGER, quote_volume REAL, trades INTEGER,
            taker_buy_base_volume REAL, taker_buy_quote_volume REAL"""
        self.conn.execute(f"CREATE TABLE klines ({columns})")
        self.conn.execute(f"CREATE TABLE futures_price_bars (data_type TEXT, {columns}, source_file_id INTEGER)")
        self.conn.execute("CREATE TABLE futures_funding_rates (symbol TEXT, funding_time INTEGER, funding_interval_hours INTEGER, funding_rate REAL, source_file_id INTEGER)")
        self.conn.execute("CREATE TABLE futures_metrics (symbol TEXT, open_time INTEGER, " + ",".join(f"{c} REAL" for c in METRICS_COLUMNS) + ", source_file_id INTEGER)")
        for timestamp in self.hours:
            ms = timestamp.value // 1_000_000
            def bar(symbol, price):
                return (symbol, "1h", ms, price, price, price, price, 100,
                        ms + 3_599_999, price * 100, 10, 50, price * 50)
            self.conn.execute("INSERT INTO klines VALUES (" + ",".join("?" * 13) + ")", bar("PEPEUSDT", 0.002))
            for category, price in [("klines", 2.2), ("markPriceKlines", 2.15), ("indexPriceKlines", 2.1), ("premiumIndexKlines", 0.04)]:
                self.conn.execute("INSERT INTO futures_price_bars VALUES (" + ",".join("?" * 15) + ")", (category, *bar("1000PEPEUSDT", price), 1))
            if timestamp.hour % 8 == 0:
                self.conn.execute("INSERT INTO futures_funding_rates VALUES (?,?,?,?,?)", ("1000PEPEUSDT", ms, 8, 0.0001, 1))
            for minute in range(0, 60, 5):
                self.conn.execute("INSERT INTO futures_metrics VALUES (?,?,?,?,?,?,?,?,?)", ("1000PEPEUSDT", ms + minute * 60000, 100, 220, 2, 3, 1, 1, 1))
        self.conn.commit()
        self.store = MarketDataStore(self.db)
        self.engine = FactorEngine(self.store)

    def factors(self, **kwargs):
        return self.engine.load("PEPEUSDT", start=self.hours[0], end=self.hours[-1], warmup_days=7, **kwargs)

    def test_symbol_alias_and_price_units(self):
        frame = self.factors()
        other = self.engine.load("1000PEPEUSDT", start=self.hours[0], end=self.hours[-1], warmup_days=7)
        pd.testing.assert_frame_equal(frame, other, check_flags=False)
        self.assertAlmostEqual(frame["basis_trade_spot"].iloc[0], 0.1)
        self.assertAlmostEqual(frame["perp_close"].iloc[0], 0.0022)
        self.assertEqual(frame["close"].iloc[0], 2.2)  # native contract OHLCV
        self.assertEqual(frame["volume"].iloc[0], 100)
        self.assertAlmostEqual(frame["perp_quote_volume"].iloc[0], 220)
        self.assertEqual(frame["premium_index"].iloc[0], 0.04)
        self.assertAlmostEqual(frame["basis_mark_index"].iloc[0], 2.15 / 2.1 - 1)

    def test_spot_base_uses_same_comparable_prices(self):
        frame = self.factors(base_market="spot")
        self.assertEqual(frame["close"].iloc[0], 0.002)
        self.assertAlmostEqual(frame["basis_trade_spot"].iloc[0], 0.1)
        self.assertEqual(frame["funding_rate"].iloc[0], 0.0001)

    def test_native_numeric_symbols_are_not_rescaled(self):
        for symbol in ["1000SATSUSDT", "1000CATUSDT", "1000CHEEMSUSDT", "1MBABYDOGEUSDT", "1INCHUSDT"]:
            result = resolve_market_symbols(symbol)
            self.assertEqual(result.spot, symbol)
            self.assertEqual(result.perpetual_multiplier, 1)

    def test_duplicate_aliases_cannot_enter_cross_section_twice(self):
        config = FactorEvaluationConfig(symbols=("PEPEUSDT", "1000PEPEUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"))
        with self.assertRaisesRegex(ValueError, "duplicate assets"):
            config.validate()
        self.assertEqual(config.interval, "1h")

    def test_partial_metrics_preserve_other_fields_and_latest_null(self):
        timestamp = self.hours[4] + pd.Timedelta(minutes=55)
        self.conn.execute("UPDATE futures_metrics SET count_toptrader_long_short_ratio=NULL WHERE open_time=?", (timestamp.value // 1_000_000,))
        self.conn.commit()
        frame = self.factors()
        row = frame.loc[self.hours[4]]
        self.assertTrue(pd.isna(row["toptrader_account_net"]))
        self.assertEqual(row["open_interest"], 100)
        self.assertEqual(row["toptrader_position_net"], 0.5)
        self.assertEqual(row["metrics_status"], "partial")
        self.assertEqual(frame.attrs["field_coverage"]["toptrader_account_net"]["unavailable_rows"], 1)

    def test_invalid_metrics_are_masked_without_changing_database(self):
        timestamp = (self.hours[4] + pd.Timedelta(minutes=55)).value // 1_000_000
        self.conn.execute("UPDATE futures_metrics SET count_toptrader_long_short_ratio=-2, sum_taker_long_short_vol_ratio=? WHERE open_time=?", (float("inf"), timestamp))
        self.conn.commit()
        frame = self.factors()
        self.assertTrue(pd.isna(frame.loc[self.hours[4], "toptrader_account_net"]))
        self.assertTrue(pd.isna(frame.loc[self.hours[4], "taker_flow_net"]))
        self.assertEqual(frame.attrs["metrics_source_quality"]["count_toptrader_long_short_ratio"]["invalid_rows"], 1)
        self.assertEqual(self.conn.execute("SELECT count_toptrader_long_short_ratio FROM futures_metrics WHERE open_time=?", (timestamp,)).fetchone()[0], -2)

    def test_missing_native_observation_expires_at_next_due_time(self):
        timestamp = (self.hours[4] + pd.Timedelta(minutes=55)).value // 1_000_000
        self.conn.execute("DELETE FROM futures_metrics WHERE open_time=?", (timestamp,))
        self.conn.commit()
        frame = self.factors()
        self.assertEqual(frame.loc[self.hours[4], "metrics_status"], "stale")
        self.assertTrue(pd.isna(frame.loc[self.hours[4], "open_interest"]))
        self.assertEqual(frame.loc[self.hours[5], "metrics_status"], "valid")

    def test_missing_ratio_does_not_become_zero_imbalance(self):
        self.conn.execute("UPDATE futures_metrics SET count_long_short_ratio=NULL")
        self.conn.commit()
        frame = self.factors()
        self.assertTrue(frame["global_account_net"].isna().all())
        self.assertTrue(frame["open_interest"].notna().all())
        json.dumps(self.engine.snapshot(frame, tail=0), allow_nan=False)

    def test_zero_oi_does_not_create_infinite_change(self):
        timestamp = (self.hours[4] + pd.Timedelta(minutes=55)).value // 1_000_000
        self.conn.execute("UPDATE futures_metrics SET sum_open_interest=0 WHERE open_time=?", (timestamp,))
        self.conn.commit()
        frame = self.factors()
        self.assertEqual(frame.loc[self.hours[4], "open_interest"], 0)
        self.assertTrue(pd.isna(frame.loc[self.hours[5], "open_interest_change_1bar"]))
        self.assertTrue(pd.isna(frame.loc[self.hours[28], "open_interest_change_24h"]))
        self.assertFalse(np.isinf(frame["open_interest_change_1bar"]).any())

    def liquidation_fixture(self):
        index = self.hours[:6]
        frame = pd.DataFrame(index=index)
        frame["symbol"] = "1000PEPEUSDT"
        frame["available_at"] = index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
        frame["liquidation_available_at"] = frame["available_at"]
        frame["liquidation_received_at_max"] = index + pd.Timedelta(minutes=30)
        frame["liquidation_coverage_status"] = ["valid_zero", "valid_event", "source_unknown", "valid_event", "valid_event", "valid_event"]
        frame["invalid_notional_count"] = [0, 0, np.nan, 0, 3, 0]
        frame["liquidation_partition_mismatch"] = [False, False, False, True, False, True]
        frame["liquidation_event_count"] = [0, 1, np.nan, 1, 3, 1]
        frame["long_liquidation_count"] = frame["liquidation_event_count"]
        frame["short_liquidation_count"] = [0, 0, np.nan, 0, 0, 0]
        frame["long_liquidation_notional_usdt"] = [0, 10, np.nan, 10, 12, 10]
        frame["short_liquidation_notional_usdt"] = [0, 0, np.nan, 0, 0, 0]
        frame.loc[index[0], "liquidation_received_at_max"] = pd.NaT
        frame.loc[index[3], "liquidation_available_at"] += pd.Timedelta(milliseconds=2)
        frame.loc[index[3], "liquidation_received_at_max"] = frame.loc[index[3], "liquidation_available_at"]
        path = self.db.parent / "derived/liquidation_alignment/symbol=1000PEPEUSDT/aligned_1h.parquet"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path)
        return frame, path

    def test_liquidation_zero_unknown_late_and_invalid_amount(self):
        original, path = self.liquidation_fixture()
        before = path.read_bytes()
        frame = self.store.load_liquidations("PEPEUSDT")
        self.assertTrue(frame.iloc[0]["liquidation_notional_usable"])
        self.assertEqual(frame.iloc[0]["long_liquidation_notional_usdt"], 0)
        for position, reason in [(2, "source_unknown"), (3, "late")]:
            self.assertEqual(frame.iloc[position]["liquidation_status"], reason)
            self.assertTrue(frame.iloc[position][list(LIQUIDATION_VALUE_COLUMNS)].isna().all())
        self.assertEqual(frame.iloc[4]["liquidation_event_count"], 3)
        self.assertTrue(pd.isna(frame.iloc[4]["long_liquidation_notional_usdt"]))
        self.assertEqual(frame.iloc[4]["liquidation_status"], "invalid_notional")
        self.assertTrue(frame.iloc[5]["liquidation_notional_usable"])
        self.assertTrue(frame.iloc[5]["liquidation_partition_mismatch"])
        self.assertEqual(path.read_bytes(), before)
        pd.testing.assert_frame_equal(pd.read_parquet(path), original, check_freq=False)

    def test_liquidation_flags_reach_factor_sample(self):
        self.liquidation_fixture()
        frame = self.factors(include_liquidations=True)
        self.assertTrue(pd.isna(frame.loc[self.hours[3], "long_liquidation_notional_usdt"]))
        self.assertEqual(frame.loc[self.hours[6], "liquidation_status"], "missing")
        self.assertFalse(frame.loc[self.hours[6], "liquidation_count_usable"])
        sample = self.engine.snapshot(frame, tail=0)
        self.assertEqual(sample["liquidation_status_counts"]["late"], 1)
        self.assertEqual(sample["field_coverage"]["long_liquidation_notional_usdt"]["valid_rows"], 3)

    def test_liquidation_missing_file_does_not_fall_back(self):
        with self.assertRaises(FileNotFoundError):
            self.factors(include_liquidations=True)
        with self.assertRaisesRegex(ValueError, "require 1h"):
            self.factors(include_liquidations=True, interval="4h")

    def test_liquidation_invalid_availability_fails(self):
        frame, path = self.liquidation_fixture()
        frame.loc[self.hours[1], "liquidation_received_at_max"] = frame.loc[self.hours[1], "available_at"] + pd.Timedelta(seconds=10)
        frame.to_parquet(path)
        with self.assertRaisesRegex(ValueError, "precedes receipt"):
            self.store.load_liquidations("PEPEUSDT")

    def test_mapped_asset_enters_liquidation_universe(self):
        class Inventory:
            def spot_snapshot(inner):
                return [{"symbol": "PEPEUSDT", "interval": "1h", "start": "2025-07-01", "end": "2026-07-31"}]
            def funding_snapshot(inner):
                return [{"symbol": "1000PEPEUSDT", "start": "2025-07-01", "end": "2026-07-31"}]
            def open_interest_snapshot(inner):
                return [{**inner.funding_snapshot()[0], "period": "5m"}]
            def futures_price_snapshot(inner, price_type):
                return [{**inner.funding_snapshot()[0], "interval": "1h"}]
        result = build_liquidation_universe(Inventory(), ["1000PEPEUSDT"])
        self.assertEqual(result["eligible_count"], 1)
        self.assertEqual(result["eligible"][0]["market_symbols"]["spot"], "PEPEUSDT")


if __name__ == "__main__":
    unittest.main()
