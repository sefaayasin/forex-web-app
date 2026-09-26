import unittest

import numpy as np
import pandas as pd

from research_fx_factors import CURRENCIES, excess_returns, last_weekday, net_returns, rank_weights, strategy_weights

MONTHS = pd.date_range("2008-01-31", periods=30, freq="ME")
NAMES = list(CURRENCIES)


def flat_rates(values):
    data = {c: v for c, v in zip(["USD"] + NAMES, values)}
    return pd.DataFrame({c: [data[c]] * len(MONTHS) for c in data}, index=MONTHS, dtype=float)


class FxFactorTests(unittest.TestCase):
    def test_last_weekday_skips_weekends(self):
        self.assertEqual(last_weekday(pd.Timestamp("2024-08-31")), pd.Timestamp("2024-08-30"))
        self.assertEqual(last_weekday(pd.Timestamp("2024-07-31")), pd.Timestamp("2024-07-31"))

    def test_rank_weights_are_dollar_neutral_top_and_bottom_two(self):
        weights = rank_weights(pd.Series([5, 1, 3, 4, 2, 0, 6], index=NAMES))
        self.assertAlmostEqual(weights.sum(), 0.0)
        self.assertEqual(set(weights[weights > 0].index), {"CAD", "EUR"})
        self.assertEqual(set(weights[weights < 0].index), {"CHF", "GBP"})
        self.assertTrue((rank_weights(pd.Series([1, np.nan, 3, 4, 2, 0, 6], index=NAMES)) == 0).all())

    def test_carry_accrues_rate_differential(self):
        prices = pd.DataFrame(1.0, index=MONTHS, columns=NAMES)
        rates = flat_rates([1.0] * 8)
        rates["NZD"] = 13.0
        xr = excess_returns(prices, rates)
        days = (MONTHS[1] - MONTHS[0]).days
        self.assertAlmostEqual(xr.loc[MONTHS[1], "NZD"], 0.12 * days / 365)
        self.assertAlmostEqual(xr.loc[MONTHS[1], "EUR"], 0.0)

    def test_carry_signal_uses_the_previous_months_rate(self):
        rates = flat_rates([1.0] * 8)
        rates.loc[MONTHS[5], "AUD"] = 50.0
        weights = strategy_weights(pd.DataFrame(0.0, index=MONTHS, columns=NAMES), rates)["carry"]
        self.assertLessEqual(weights.loc[MONTHS[5], "AUD"], 0.5)
        self.assertEqual(weights.loc[MONTHS[6], "AUD"], 0.5)

    def test_momentum_12_1_skips_the_latest_month(self):
        xr = pd.DataFrame(0.0, index=MONTHS, columns=NAMES)
        xr.loc[MONTHS[20], "JPY"] = 0.5
        xs = strategy_weights(xr, flat_rates([1.0] * 8))["xs_momentum"]
        self.assertEqual(xs.loc[MONTHS[20], "JPY"], 0.0)
        self.assertEqual(xs.loc[MONTHS[21], "JPY"], 0.5)

    def test_positions_earn_next_month_and_pay_turnover_cost(self):
        index = pd.date_range("2008-11-30", periods=4, freq="ME")
        xr = pd.DataFrame(0.0, index=index, columns=NAMES)
        xr.loc[index[2], "EUR"] = 0.01
        weights = pd.DataFrame(0.0, index=index, columns=NAMES)
        weights.loc[index[1]:, "EUR"] = 1.0
        prices = pd.DataFrame(1.0, index=index, columns=NAMES)
        free = net_returns(weights, xr, prices, {"round_trip_pips": 0.0, "swap_markup_per_year": 0.0})
        self.assertAlmostEqual(free.loc[index[2]], 0.01)
        costly = net_returns(weights, xr, prices, {"round_trip_pips": 2.0, "swap_markup_per_year": 0.0})
        self.assertAlmostEqual(costly.loc[index[2]], 0.01 - 0.0001)
        self.assertAlmostEqual(costly.loc[index[3]], 0.0)


if __name__ == "__main__":
    unittest.main()
