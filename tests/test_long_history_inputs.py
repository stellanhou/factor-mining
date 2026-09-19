"""Historical universe selection must not learn from subsequent observations."""
import importlib.util
import sqlite3
import unittest
from pathlib import Path

import pandas as pd


path = Path(__file__).resolve().parents[1] / "scripts/prepare_long_history_factor_mining.py"
module_spec = importlib.util.spec_from_file_location("long_history_inputs", path)
prep = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(prep)


class HistoricalCohortTests(unittest.TestCase):
    def test_prior_complete_days_median_and_no_future_survival_filter(self):
        conn = sqlite3.connect(":memory:")
        columns = "symbol TEXT, interval TEXT, open_time INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL, quote_volume REAL, close_time INTEGER"
        conn.execute(f"CREATE TABLE klines ({columns})")
        conn.execute(f"CREATE TABLE futures_price_bars ({columns}, data_type TEXT)")
        conn.execute("CREATE TABLE futures_archive_files (symbol TEXT, category TEXT, interval TEXT)")
        selection = pd.Timestamp("2022-07-25T00:00:00Z")
        start = prep.millis(selection - pd.Timedelta(days=30))
        names = [f"C{i:02d}USDT" for i in range(31)] + ["USDCUSDT", "C00UPUSDT", "NEWUSDT", "GAPUSDT"]
        for name in names:
            conn.execute("INSERT INTO futures_archive_files VALUES (?, 'klines', '1h')", (name,))
            rows = []
            for i in range(720):
                if name == "NEWUSDT" or (name == "GAPUSDT" and i == 100):
                    continue
                ts = start + i * prep.HOUR
                quote = 1_000 if name in {"USDCUSDT", "C00UPUSDT"} else 10
                if name == "C00USDT":
                    quote = 1_000_000 if i < 24 else 1  # High total, low median.
                rows.append((name, "1h", ts, 10, 11, 9, 10, 1, quote, ts + prep.HOUR - 1))
            conn.executemany("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            conn.executemany("INSERT INTO futures_price_bars VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                             [(*row, "klines") for row in rows])
        before = prep.select_cohort(conn, selection)
        self.assertEqual(len(before), 31)
        self.assertEqual(before[-1]["symbol"], "C00USDT")
        self.assertEqual(before[-1]["median_daily_quote_volume"], 24)
        self.assertEqual(before[0]["symbol"], "C01USDT")
        # Only one coin survives in the new future data, with an extreme volume.
        # The remaining historical coins (no later records) must stay eligible.
        ts = prep.millis(selection)
        future = ("C30USDT", "1h", ts, 10, 11, 9, 10, 1, 1e12, ts + prep.HOUR - 1)
        conn.execute("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?,?)", future)
        conn.execute("INSERT INTO futures_price_bars VALUES (?,?,?,?,?,?,?,?,?,?,?)", (*future, "klines"))
        self.assertEqual(before, prep.select_cohort(conn, selection))
        conn.close()


if __name__ == "__main__":
    unittest.main()
