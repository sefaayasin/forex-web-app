import math
import unittest

import pandas as pd

from forex_costs import (
    ECN_COMMISSION_PIPS,
    current_ny_hour,
    cost_share_of_risk,
    estimated_round_trip_cost_pips,
    measured_spread_pips,
)

PROFILE = {"pairs": {"EURUSD": {"median_by_ny_hour": [0.5] * 21 + [2.0, float("nan"), 0.4],
                                "p90_by_ny_hour": [1.0] * 24}}}


class ForexCostsTests(unittest.TestCase):
    def test_measured_spread_by_hour_accepts_yahoo_suffix(self):
        self.assertEqual(measured_spread_pips(PROFILE, "EURUSD=X", 21), 2.0)
        self.assertEqual(measured_spread_pips(PROFILE, "eurusd", 3, "p90"), 1.0)

    def test_missing_pair_hour_or_nan_gives_none(self):
        self.assertIsNone(measured_spread_pips(PROFILE, "GBPUSD", 3))
        self.assertIsNone(measured_spread_pips(PROFILE, "EURUSD", 22))
        self.assertIsNone(measured_spread_pips(PROFILE, "EURUSD", 24))
        self.assertIsNone(measured_spread_pips({}, "EURUSD", 3))

    def test_round_trip_cost_adds_commission(self):
        self.assertAlmostEqual(estimated_round_trip_cost_pips(PROFILE, "EURUSD", 3), round(0.5 + ECN_COMMISSION_PIPS, 1))
        self.assertIsNone(estimated_round_trip_cost_pips(PROFILE, "EURUSD", 22))

    def test_new_york_hour_follows_daylight_saving(self):
        self.assertEqual(current_ny_hour(pd.Timestamp("2026-07-15 21:00", tz="UTC")), 17)
        self.assertEqual(current_ny_hour(pd.Timestamp("2026-01-15 22:00", tz="UTC")), 17)

    def test_cost_share_of_risk(self):
        self.assertAlmostEqual(cost_share_of_risk(1.5, 10.0), 1.5 / 11.5)
        self.assertIsNone(cost_share_of_risk(1.5, 0.0))
        self.assertTrue(math.isclose(cost_share_of_risk(0.0, 10.0), 0.0))


if __name__ == "__main__":
    unittest.main()
