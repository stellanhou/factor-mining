"""Contract tests for field units, causal windows, expressions and Agent context."""

import json
import unittest

import numpy as np
import pandas as pd

import test_crypto_data_inputs as fixtures
from crypto_quant.features.factor_inputs import (
    INPUT_COLUMNS, FactorInputPanel, load_factor_inputs, validate_universe,
)
from crypto_quant.features.factor_expressions import (
    compile_expression, evaluate_expression, operator_catalog, template_catalog,
)
from crypto_quant.features.factors import FactorEngine


def panel_from_columns(columns, symbols=("BTCUSDT",)):
    rows = len(next(iter(columns.values()))) // len(symbols)
    index = pd.MultiIndex.from_product([
        pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"), symbols,
    ], names=["timestamp", "symbol"])
    values = pd.DataFrame(columns, index=index)
    return FactorInputPanel(values, pd.Series(True, index=index), {})


class FactorExpressionTests(unittest.TestCase):
    def calculate(self, expression, columns):
        return evaluate_expression(expression, panel_from_columns(columns)).values.to_numpy()

    def test_scalar_math_and_units(self):
        columns = {"spot_quote_volume": [2., 4.], "perp_quote_volume": [3., 8.]}
        for expression, expected in (
            ("add(spot_quote_volume,perp_quote_volume)", [5, 12]),
            ("sub(spot_quote_volume,perp_quote_volume)", [-1, -4]),
            ("mul(spot_quote_volume,perp_quote_volume)", [6, 32]),
            ("div(spot_quote_volume,perp_quote_volume)", [2/3, .5]),
            ("min(spot_quote_volume,perp_quote_volume)", [2, 4]),
            ("max(spot_quote_volume,perp_quote_volume)", [3, 8]),
            ("neg(spot_quote_volume)", [-2, -4]),
            ("abs(neg(spot_quote_volume))", [2, 4]),
            ("sign(neg(spot_quote_volume))", [-1, -1]),
        ):
            with self.subTest(expression=expression):
                np.testing.assert_allclose(self.calculate(expression, columns), expected)
        self.assertEqual(compile_expression("mul(perp_close,perp_volume)").unit.label(), "USDT")
        self.assertEqual(compile_expression("power(sqrt(spot_quote_volume),2)").unit.label(), "USDT")
        for expression in ("add(perp_close,perp_volume)", "max(spot_quote_volume,perp_close)", "cross_rank(open_interest_base)"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                compile_expression(expression)

    def test_invalid_domain_and_zero_denominator_propagate_nan(self):
        columns = {"premium_index": [-1, 0, 4, np.inf, np.nan]}
        for expression, expected in (
            ("log(premium_index)", [np.nan, np.nan, np.log(4), np.nan, np.nan]),
            ("sqrt(premium_index)", [np.nan, 0, 2, np.nan, np.nan]),
            ("power(premium_index,0.5)", [np.nan, 0, 2, np.nan, np.nan]),
            ("div(1,premium_index)", [-1, np.nan, .25, np.nan, np.nan]),
            ("power(premium_index,1000)", [1, 0, np.nan, np.nan, np.nan]),
        ):
            with self.subTest(expression=expression):
                np.testing.assert_allclose(self.calculate(expression, columns), expected, equal_nan=True)

    def test_complete_windows_and_exact_lags(self):
        columns = {"open_interest_value": [1., 2., np.nan, 4., 8., 16.]}
        for expression, expected in (
            ("ts_mean(open_interest_value,3)", [np.nan]*5 + [28/3]),
            ("ts_sum(open_interest_value,3)", [np.nan]*5 + [28]),
            ("ts_min(open_interest_value,3)", [np.nan]*5 + [4]),
            ("ts_max(open_interest_value,3)", [np.nan]*5 + [16]),
            ("ts_delay(open_interest_value,2)", [np.nan, np.nan, 1, 2, np.nan, 4]),
            ("ts_delta(open_interest_value,2)", [np.nan, np.nan, np.nan, 2, np.nan, 12]),
            ("ts_return(open_interest_value,2)", [np.nan, np.nan, np.nan, 1, np.nan, 3]),
        ):
            with self.subTest(expression=expression):
                np.testing.assert_allclose(self.calculate(expression, columns), expected, equal_nan=True)
        self.assertEqual(compile_expression("ts_mean(ts_return(perp_close,24),24)").lookback_hours, 47)

    def test_population_statistics_and_pairwise_completeness(self):
        columns = {"spot_quote_volume": [1., 2., 3., 3.], "perp_quote_volume": [3., 2., 1., np.nan]}
        self.assertAlmostEqual(self.calculate("ts_std(spot_quote_volume,3)", columns)[2], np.sqrt(2/3))
        cov = self.calculate("ts_cov(spot_quote_volume,perp_quote_volume,3)", columns)
        corr = self.calculate("ts_corr(spot_quote_volume,perp_quote_volume,3)", columns)
        self.assertAlmostEqual(cov[2], -2/3)
        self.assertAlmostEqual(corr[2], -1)
        self.assertTrue(np.isnan(cov[3]) and np.isnan(corr[3]))
        constant = {"spot_quote_volume": [2., 2., 2.], "perp_quote_volume": [1., 2., 3.]}
        self.assertEqual(self.calculate("ts_std(spot_quote_volume,3)", constant)[2], 0)
        self.assertTrue(np.isnan(self.calculate("ts_corr(spot_quote_volume,perp_quote_volume,3)", constant)[2]))

    def test_time_rank_ties(self):
        result = self.calculate("ts_rank(open_interest_value,3)", {"open_interest_value": [1., 3., 3., 3.]})
        np.testing.assert_allclose(result, [np.nan, np.nan, .75, .5], equal_nan=True)
        self.assertTrue(np.isnan(self.calculate("ts_rank(open_interest_value,1)", {"open_interest_value": [1.]})).all())

    def test_cross_section_rank_uses_historical_membership_and_reports_counts(self):
        panel = panel_from_columns({"open_interest_value": [2., 2., 8., 2., 5., 99., 2., 5., 99.]},
                                   symbols=("BTCUSDT", "ETHUSDT", "PEPEUSDT"))
        panel.universe.iloc[5] = False
        panel.universe.iloc[7:] = False
        result = evaluate_expression("cross_rank(open_interest_value)", panel)
        np.testing.assert_allclose(result.values, [.25, .25, 1, 0, 1, np.nan, np.nan, np.nan, np.nan], equal_nan=True)
        self.assertEqual(result.cross_section_counts.iloc[:, 0].tolist(), [3, 2, 1])
        pct = evaluate_expression("cross_pct(open_interest_value)", panel)
        np.testing.assert_allclose(pct.values, result.values * 100, equal_nan=True)
        zscore = evaluate_expression("cross_zscore(open_interest_value)", panel)
        np.testing.assert_allclose(zscore.values.iloc[3:6], [-1, 1, np.nan], equal_nan=True)
        panel.values.iloc[:] = 2
        self.assertTrue(evaluate_expression("cross_zscore(open_interest_value)", panel).values.isna().all())

    def test_time_windows_do_not_mix_assets_or_drop_nonmember_history(self):
        panel = panel_from_columns({"open_interest_value": [1., 10., 2., 20., 3., 30.]}, symbols=("BTCUSDT", "ETHUSDT"))
        panel.universe.iloc[0] = False
        result = evaluate_expression("ts_mean(open_interest_value,3)", panel)
        np.testing.assert_allclose(result.values.iloc[-2:], [2, 20])

    def test_future_values_cannot_change_past_results(self):
        panel = panel_from_columns({"open_interest_value": [1., 10., 2., 20., 3., 30., 4., 40.]}, symbols=("BTCUSDT", "ETHUSDT"))
        expression = "ts_mean(cross_rank(ts_delta(open_interest_value,1)),2)"
        original = evaluate_expression(expression, panel).values
        panel.values.iloc[-2:] = [1000]
        changed = evaluate_expression(expression, panel).values
        pd.testing.assert_series_equal(original.iloc[:-2], changed.iloc[:-2])

    def test_unknown_fields_and_python_execution_are_rejected(self):
        invalid = [
            "future_return", "close", "metrics_age_minutes", "taker_flow_net",
            "taker_long_short_volume_ratio", "perp_close.shift(-1)", "perp_close[0]",
            "__import__('os').system('false')", "lambda: 1", "perp_close + 1",
            "ts_mean(perp_close)", "ts_mean(perp_close,0)", "ts_mean(perp_close,-1)",
            "ts_mean(perp_close,2.0)", "ts_mean(perp_close,True)",
            "ts_mean(perp_close,n=2)", "power(perp_close,perp_close)", "1e999",
        ]
        for expression in invalid:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                compile_expression(expression)

    def test_universe_requires_explicit_grid_and_unique_assets(self):
        panel = panel_from_columns({"open_interest_value": [1, 2, 3]})
        with self.assertRaisesRegex(ValueError, "every hourly"):
            validate_universe(panel.universe.iloc[[0, 2]])
        duplicate = panel_from_columns({"open_interest_value": [1, 2]}, symbols=("PEPEUSDT", "1000PEPEUSDT"))
        with self.assertRaisesRegex(ValueError, "duplicate asset"):
            validate_universe(duplicate.universe)
        fractional = panel.universe.astype(float)
        with self.assertRaisesRegex(ValueError, "boolean"):
            validate_universe(fractional)

    def test_templates_are_executable_and_expanded(self):
        self.assertEqual(len(operator_catalog()), 26)
        self.assertEqual(len(template_catalog()), 18)
        definition = compile_expression("basis_change_1bar")
        self.assertEqual(definition.fields, ("perp_close", "spot_close"))
        self.assertNotIn("basis_trade_spot", definition.expanded_expression)
        result = self.calculate("basis_trade_spot", {"perp_close": [100.2], "spot_close": [100.]})
        self.assertAlmostEqual(result[0], .002)


class FactorInputInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CryptoInputTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.index = pd.MultiIndex.from_product([self.fixture.hours, ["PEPEUSDT"]], names=["timestamp", "symbol"])
        self.universe = pd.Series(True, index=self.index)

    def test_entirely_ineligible_future_asset_needs_no_source_read(self):
        index = pd.MultiIndex.from_product([self.fixture.hours, ["PEPEUSDT", "FUTUREUSDT"]], names=["timestamp", "symbol"])
        members = pd.Series(index.get_level_values("symbol") == "PEPEUSDT", index=index)
        panel = load_factor_inputs(self.fixture.store, members)
        self.assertTrue(panel.values.xs("FUTUREUSDT", level="symbol").isna().all().all())
        info = panel.diagnostics["symbols"]["FUTUREUSDT"]
        self.assertFalse(info["data_read"])
        self.assertEqual(len(info["coverage"]), 35)

    def test_normalized_fields_and_actual_agent_message(self):
        panel = load_factor_inputs(self.fixture.store, self.universe)
        self.assertEqual(set(panel.values), set(INPUT_COLUMNS))
        self.assertEqual(len(panel.values.columns), 35)
        row = panel.values.iloc[0]
        for column in ["open", "high", "low", "close"]:
            self.assertAlmostEqual(row[f"perp_{column}"], .0022)
        self.assertEqual(row["perp_volume"], 100000)
        self.assertEqual(row["perp_taker_buy_base_volume"], 50000)
        self.assertEqual(row["open_interest_base"], 100000)
        self.assertAlmostEqual(row["perp_quote_volume"], 220)
        self.assertEqual(row["perp_trades"], 10)
        context = json.loads(panel.ideation_message()["content"])
        self.assertEqual(len(context["fields"]), 35)
        self.assertEqual(len(context["operators"]), 26)
        self.assertIn("expanded_expression", context["templates"][0])
        coverage = context["data"]["symbols"]["PEPEUSDT"]["coverage"]
        self.assertEqual(coverage["long_liquidation_count"]["missing_reasons"], {"not_requested": 32})
        self.assertEqual(coverage["perp_volume"]["coverage_ratio"], 1)
        self.assertEqual(coverage["long_liquidation_count"]["coverage_ratio"], 0)
        self.assertNotIn("close", panel.values)
        self.assertNotIn("taker_flow_net", panel.values)
        result = evaluate_expression("basis_trade_spot", panel)
        np.testing.assert_allclose(result.values, .1)

    def test_missing_hour_remains_in_place_in_reader_and_expression(self):
        stamp = self.fixture.hours[2].value // 1000000
        self.fixture.conn.execute("DELETE FROM futures_price_bars WHERE data_type='klines' AND open_time=?", (stamp,))
        self.fixture.conn.commit()
        frame = self.fixture.factors()
        self.assertEqual(len(frame), 32)
        self.assertFalse(frame.loc[self.fixture.hours[2], "base_bar_present"])
        self.assertTrue(pd.isna(frame.loc[self.fixture.hours[2], "perp_close"]))
        self.assertTrue(pd.isna(frame.loc[self.fixture.hours[3], "perp_log_return_1bar"]))
        self.assertTrue(pd.isna(frame.loc[self.fixture.hours[24], "perp_quote_volume_24h"]))
        panel = load_factor_inputs(self.fixture.store, self.universe)
        result = evaluate_expression("ts_delay(perp_close,1)", panel)
        self.assertTrue(pd.isna(result.values.loc[(self.fixture.hours[3], "PEPEUSDT")]))

    def test_liquidation_quality_reaches_input_and_formula(self):
        self.fixture.liquidation_fixture()
        panel = load_factor_inputs(self.fixture.store, self.universe, include_liquidations=True)
        values = panel.values.xs("PEPEUSDT", level="symbol")
        self.assertEqual(values["long_liquidation_count"].iloc[4], 3)
        self.assertTrue(pd.isna(values["long_liquidation_notional_usdt"].iloc[4]))
        context = panel.ideation_context()["data"]["symbols"]["PEPEUSDT"]["coverage"]
        self.assertEqual(context["long_liquidation_notional_usdt"]["missing_reasons"]["late"], 1)
        self.assertEqual(context["long_liquidation_notional_usdt"]["missing_reasons"]["invalid_notional"], 1)


class FundingWindowTests(unittest.TestCase):
    def funding(self):
        times = pd.date_range("2025-12-29", "2026-01-04", freq="8h", tz="UTC", name="timestamp")
        return pd.DataFrame({"funding_rate": .0001, "funding_interval_hours": 8}, index=times)

    def query(self, *times):
        return pd.Series(pd.to_datetime(list(times), utc=True))

    def test_events_count_once_and_exact_lower_boundary_is_excluded(self):
        values, status = FactorEngine._funding_window(self.funding(), self.query("2026-01-01T00:00Z", "2026-01-01T07:00Z"), 24)
        np.testing.assert_allclose(values, [.0003, .0003])
        self.assertEqual(status.tolist(), ["valid", "valid"])

    def test_invalid_or_missing_settlement_does_not_become_zero(self):
        funding = self.funding().drop(pd.Timestamp("2026-01-01T08:00Z"))
        values, status = FactorEngine._funding_window(funding, self.query("2026-01-01T09:00Z", "2026-01-01T17:00Z"), 24)
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(status.tolist(), ["missing_settlement"] * 2)
        funding = self.funding()
        funding.loc[pd.Timestamp("2026-01-01T08:00Z"), "funding_rate"] = np.nan
        values, status = FactorEngine._funding_window(funding, self.query("2026-01-01T09:00Z"), 24)
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(status[0], "invalid_event")

    def test_gap_expires_when_missing_slot_leaves_window(self):
        funding = self.funding().drop(pd.Timestamp("2026-01-01T08:00Z"))
        values, status = FactorEngine._funding_window(funding, self.query("2026-01-02T07:59:59.999Z", "2026-01-02T08:00:00.000Z"), 24)
        self.assertTrue(np.isnan(values[0]))
        self.assertAlmostEqual(values[1], .0003)
        self.assertEqual(status.tolist(), ["missing_settlement", "valid"])

    def test_invalid_latest_interval_cannot_produce_zero_for_empty_window(self):
        funding = self.funding()
        funding.iloc[-1, funding.columns.get_loc("funding_interval_hours")] = np.nan
        values, status = FactorEngine._funding_window(funding, self.query("2026-01-06T00:00Z"), 24)
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(status[0], "invalid_event")

    def test_window_uses_signal_time_not_last_settlement_time(self):
        funding = self.funding()
        # Switch from eight-hour to hourly settlements at Jan 1 01:00.
        funding = funding.loc[funding.index <= pd.Timestamp("2026-01-01T00:00Z")]
        additional = pd.DataFrame({"funding_rate": .0002, "funding_interval_hours": 1},
                                  index=pd.date_range("2026-01-01T01:00Z", periods=24, freq="h"))
        funding = pd.concat([funding, additional])
        values, status = FactorEngine._funding_window(funding, self.query("2026-01-01T23:59Z", "2026-01-02T00:00Z"), 24)
        np.testing.assert_allclose(values, [.0001 + 23*.0002, 24*.0002])
        self.assertEqual(status.tolist(), ["valid", "valid"])

    def test_future_records_cannot_change_past_coverage(self):
        funding = self.funding().drop(pd.Timestamp("2026-01-01T08:00Z"))
        query = self.query("2026-01-01T07:00Z", "2026-01-01T09:00Z")
        full = FactorEngine._funding_window(funding, query, 24)
        prefix = FactorEngine._funding_window(funding.loc[funding.index <= query.max()], query, 24)
        np.testing.assert_allclose(full[0], prefix[0], equal_nan=True)
        self.assertEqual(full[1].tolist(), prefix[1].tolist())

    def test_millisecond_settlement_jitter_is_not_a_missing_payment(self):
        funding = self.funding()
        funding.index = funding.index + pd.to_timedelta([13 if i % 2 else 0 for i in range(len(funding))], unit="ms")
        values, status = FactorEngine._funding_window(funding, self.query("2026-01-01T07:59:59.999Z", "2026-01-01T15:59:59.999Z"), 24)
        np.testing.assert_allclose(values, [.0003, .0003])
        self.assertEqual(status.tolist(), ["valid", "valid"])
        # Actual event times are retained: the 08:00:00.013 event is not known at 08:00.
        events = funding.index
        delayed = events[(events.hour == 8) & (events.microsecond > 0)][0]
        _, boundary_status = FactorEngine._funding_window(funding, pd.Series([delayed.floor("h")]), 24)
        self.assertEqual(boundary_status[0], "missing_settlement")


if __name__ == "__main__":
    unittest.main()
