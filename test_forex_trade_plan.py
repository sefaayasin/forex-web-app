import unittest

import pandas as pd

from forex_trade_plan import build_trade_plan, load_plan_stats, measured_outcome, trading_blockers

STATS = {"pairs": {"EURUSD": {"ratios": {
    "0.5": {"ml": {"target_rate": 0.67, "stop_rate": 0.33, "timeout_rate": 0.0, "net_r": -0.13},
            "opposite": {"target_rate": 0.66, "stop_rate": 0.34, "timeout_rate": 0.0, "net_r": -0.15}},
    "1.0": {"ml": {"target_rate": 0.51, "stop_rate": 0.49, "timeout_rate": 0.0, "net_r": -0.126},
            "opposite": {"target_rate": 0.49, "stop_rate": 0.51, "timeout_rate": 0.0, "net_r": -0.16}},
}}}}
WEDNESDAY_NOON_NY = pd.Timestamp("2026-10-07 12:00", tz="America/New_York").tz_convert("UTC")


def plan(**overrides):
    args = dict(pair="EURUSD", price=1.1600, pip=0.0001, atr=0.0006, probability_up=0.58, target_usd=50.0,
                loss_usd=50.0, account_usd=1000.0, pip_value_per_lot=10.0, cost_pips=1.0, stats=STATS, blockers=[])
    args.update(overrides)
    return build_trade_plan(**args)


class TradePlanTests(unittest.TestCase):
    def test_long_plan_levels_lot_and_measured_outcome(self):
        p = plan()
        self.assertEqual((p["action"], p["side"]), ("BEKLE", "LONG"))
        self.assertAlmostEqual(p["stop_pips"], 9.0)
        self.assertAlmostEqual(p["lot"], 50 / (10.0 * 10))  # loss / ((stop + cost) * pip value)
        self.assertAlmostEqual(p["target_pips"], 11.0)  # 50 $ net at 0.5 lot = 10 pip, + 1 pip cost
        self.assertAlmostEqual(p["stop"], 1.1600 - 0.0009)
        self.assertAlmostEqual(p["target"], 1.1600 + 0.0011)
        self.assertEqual(p["measured"]["ratio"], 1.0)  # 11 / 9 = 1.22 -> nearest 1.0
        self.assertAlmostEqual(p["expected_usd"], -0.126 * 9.0 * 0.5 * 10)

    def test_short_side_and_mirrored_levels(self):
        p = plan(probability_up=0.4)
        self.assertEqual(p["side"], "SHORT")
        self.assertAlmostEqual(p["side_probability"], 0.6)
        self.assertGreater(p["stop"], p["entry"])
        self.assertLess(p["target"], p["entry"])

    def test_blockers_turn_the_action_into_wait_but_keep_the_side(self):
        p = plan(blockers=["haber"])
        self.assertEqual((p["action"], p["side"]), ("BEKLE", "LONG"))

    def test_no_ml_view_or_missing_data_waits(self):
        self.assertEqual(plan(probability_up=None)["status"], "no_side")
        self.assertEqual(plan(atr=None)["status"], "no_data")

    def test_risk_and_min_lot_warnings(self):
        self.assertTrue(any("%5" in w for w in plan()["warnings"]))
        self.assertTrue(any("en küçük lot" in w for w in plan(loss_usd=0.5, target_usd=0.5)["warnings"]))

    def test_nearest_measured_ratio(self):
        self.assertEqual(measured_outcome(STATS, "EURUSD", 0.6)["ratio"], 0.5)
        self.assertFalse(measured_outcome(STATS, "EURUSD", 2.0)["exact"])
        self.assertFalse(measured_outcome(STATS, "EURUSD", 0.6)["exact"])

    def test_blockers_for_rollover_weekend_news_and_cost(self):
        self.assertEqual(trading_blockers(WEDNESDAY_NOON_NY), [])
        rollover = pd.Timestamp("2026-10-07 17:10", tz="America/New_York").tz_convert("UTC")
        self.assertIn("Gün sonu", trading_blockers(rollover)[0])
        saturday = pd.Timestamp("2026-10-10 12:00", tz="America/New_York").tz_convert("UTC")
        self.assertIn("Piyasa kapalı", trading_blockers(saturday)[0])
        self.assertIn("30 dk sonra", trading_blockers(WEDNESDAY_NOON_NY, news_minutes=30)[0])
        self.assertIn("10 dk önce", trading_blockers(WEDNESDAY_NOON_NY, news_minutes=-10)[0])
        self.assertEqual(trading_blockers(WEDNESDAY_NOON_NY, news_minutes=90), [])
        self.assertIn("%20", trading_blockers(WEDNESDAY_NOON_NY, cost_share=0.2)[0])

    def test_shipped_stats_cover_every_pair_and_ratio(self):
        stats = load_plan_stats()
        self.assertEqual(len(stats["pairs"]), 28)
        for pair_stats in stats["pairs"].values():
            self.assertEqual(sorted(map(float, pair_stats["ratios"])), stats["ratios"])

    def test_losing_or_unmeasured_plan_cannot_become_an_entry(self):
        self.assertEqual(plan(target_usd=30, loss_usd=100)["action"], "BEKLE")
        self.assertEqual(plan(stats={})["action"], "BEKLE")
        self.assertTrue(plan()["blockers"])

    def test_positive_descriptive_average_does_not_grant_permission(self):
        stats = {"pairs": {"EURUSD": {"ratios": {"1.0": {
            "ml": {"net_r": .1}, "opposite": {"net_r": 0.}}}}}}
        p = plan(stats=stats, target_usd=40)
        self.assertEqual(p["action"], "BEKLE")

    def test_lot_is_rounded_down_and_cash_budget_not_exceeded(self):
        p = plan(loss_usd=100, target_usd=30, atr=.000713)
        self.assertLessEqual(p["planned_loss_usd"], 100)
        self.assertAlmostEqual(p["lot"] * 100, round(p["lot"] * 100))
        self.assertAlmostEqual((p["target_pips"] - 1) * p["lot"] * 10, 30)

    def test_invalid_inputs_cannot_produce_levels(self):
        for override in ({"probability_up": 1.1}, {"probability_up": float("inf")},
                         {"price": float("nan")}, {"cost_pips": -1}, {"pip": 0}):
            self.assertEqual(plan(**override)["action"], "BEKLE")

    def test_partial_exit_at_one_r_when_target_is_farther(self):
        p = plan()  # stop 9, target 11 pips, lot 0.5
        partial = p["partial_exit"]
        self.assertAlmostEqual(partial["price"], 1.1600 + 0.0009)
        self.assertEqual((partial["close_lot"], partial["keep_lot"]), (0.25, 0.25))
        self.assertAlmostEqual(partial["locked_usd"], 0.25 * (9 - 1) * 10)
        self.assertLess(plan(probability_up=0.4)["partial_exit"]["price"], 1.1600)

    def test_no_partial_exit_for_near_target_or_unsplittable_lot(self):
        self.assertIsNone(plan(target_usd=20)["partial_exit"])  # target 5 pips < 1R
        self.assertIsNone(plan(loss_usd=1.5, account_usd=1000)["partial_exit"])  # 0.01 lot cannot be halved


if __name__ == "__main__":
    unittest.main()
