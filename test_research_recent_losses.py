import unittest

import numpy as np

from research_recent_losses import exit_replay


def trade(side="LONG"):
    sign = 1 if side == "LONG" else -1
    return {"journal": {"side": side, "entry": 1.1, "stop": 1.1 - sign * .001,
                         "target": 1.1 + sign * .0003, "atr": .000666, "pip": .0001},
            "open_msc_raw": 1000, "close_msc_raw": 3000,
            "deals": [{"entry": 0, "price": 1.1, "profit": 0, "commission": -.35, "swap": 0, "fee": 0},
                      {"entry": 1, "price": 1.1 - sign * .0005, "profit": -5,
                       "commission": -.35, "swap": 0, "fee": 0}]}


def ticks(bid, ask=None):
    bid = np.array(bid)
    return {"time_msc": np.arange(len(bid)) * 1000 + 1000,
            "bid": bid, "ask": bid + .0002 if ask is None else np.array(ask)}


class ExitReplayTests(unittest.TestCase):
    def test_long_closes_at_bid_and_charges_both_commissions(self):
        result = exit_replay(trade(), ticks([1.1, 1.0994]), .5)
        self.assertEqual(result["reason"], "EARLY_STOP")
        self.assertAlmostEqual(result["net_usd"], -6.7)

    def test_short_stop_uses_ask(self):
        result = exit_replay(trade("SHORT"), ticks([1.0998, 1.1004]), .5)
        self.assertEqual(result["reason"], "EARLY_STOP")
        self.assertAlmostEqual(result["net_usd"], -6.7)

    def test_full_stop_does_not_exit_at_half_stop(self):
        result = exit_replay(trade(), ticks([1.1, 1.0994, 1.0998]), 1.)
        self.assertEqual(result["reason"], "CAPTURE_END_MARK")
        self.assertAlmostEqual(result["net_usd"], -2.7)

    def test_target_and_full_stop_follow_tick_order(self):
        result = exit_replay(trade(), ticks([1.1, 1.1004, 1.098]), 1., True)
        self.assertEqual(result["reason"], "TARGET")
        self.assertAlmostEqual(result["net_usd"], 3.3)
        stopped = exit_replay(trade(), ticks([1.1, 1.098, 1.1004]), 1., True)
        self.assertEqual(stopped["reason"], "STOP")
        self.assertAlmostEqual(stopped["net_usd"], -20.7)

    def test_trailing_cannot_benefit_from_later_future_peak(self):
        result = exit_replay(trade(), ticks([1.1, 1.1004, 1.0997, 1.104]), .5)
        self.assertEqual(result["reason"], "TRAIL")
        self.assertEqual(result["closed_raw_msc"], 3000)
        self.assertAlmostEqual(result["net_usd"], -3.7)

    def test_tick_after_four_hour_horizon_cannot_trigger_target(self):
        data = ticks([1.1, 1.1, 1.1004])
        data["time_msc"] = np.array([1000, 14401000, 14401001])
        result = exit_replay(trade(), data, 1., True)
        self.assertEqual(result["reason"], "CAPTURE_END_MARK")
        self.assertEqual(result["closed_raw_msc"], 14401000)
        self.assertAlmostEqual(result["net_usd"], -.7)


if __name__ == "__main__":
    unittest.main()
