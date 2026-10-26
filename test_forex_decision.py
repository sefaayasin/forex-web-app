import unittest

from forex_decision import final_call, pivot_side, target_check
from forex_trade_plan import build_trade_plan


def sides(**overrides):
    base = {"radar": 0, "firsatlar": 0, "pariteler": 0, "ml": 0}
    base.update(overrides)
    return base


class ForexDecisionTests(unittest.TestCase):
    def test_ml_decides_even_against_all_technical_views(self):
        call = final_call(sides(radar=1, firsatlar=1, pariteler=1, ml=-1), pivot=1, probability_up=0.44)
        self.assertEqual((call["action"], call["pattern"]), ("SHORT", "ml_vs_technical"))
        self.assertIn("%56", call["source"])
        self.assertEqual((call["agree"], call["against"]), (1, 4))
        self.assertIn("%53", call["evidence"])

    def test_ml_agreeing_with_technical_views(self):
        call = final_call(sides(radar=1, firsatlar=1, pariteler=1, ml=1), pivot=-1, probability_up=0.6)
        self.assertEqual((call["action"], call["pattern"], call["agree"]), ("LONG", "ml_with_technical", 4))

    def test_without_ml_technical_majority_then_pivot(self):
        self.assertEqual(final_call(sides(radar=-1, firsatlar=-1, pariteler=1), pivot=1)["action"], "SHORT")
        call = final_call(sides(radar=1, firsatlar=-1), pivot=-1)
        self.assertEqual((call["action"], call["pattern"]), ("SHORT", "pivot"))

    def test_always_a_side_unless_no_data(self):
        self.assertEqual(final_call(sides(), pivot=1)["action"], "LONG")
        self.assertEqual(final_call(sides(), pivot=0)["side"], 0)

    def test_pivot_side(self):
        levels = {"P": 1.1000}
        self.assertEqual(pivot_side(1.1010, levels), 1)
        self.assertEqual(pivot_side(1.0990, levels), -1)
        self.assertEqual(pivot_side(1.1000, levels), 0)
        self.assertEqual(pivot_side(None, levels), 0)
        self.assertEqual(pivot_side(1.1, None), 0)

    def test_trade_plan_uses_given_side_without_ml(self):
        plan = build_trade_plan("EURUSD", 1.1, 0.0001, 0.0010, None, 30, 100, 1000, 10, 1.0, {}, [], side="SHORT")
        self.assertEqual((plan["status"], plan["side"], plan["measured"], plan["side_probability"]),
                         ("ready", "SHORT", None, None))
        self.assertGreater(plan["stop"], 1.1)
        self.assertLess(plan["target"], 1.1)

    def test_target_check_follows_users_thresholds(self):
        stats = {"pairs": {"EURUSD": {"ratios": {
            "0.5": {"ml": {"target_rate": 0.67}, "opposite": {"target_rate": 0.63}},
            "1.0": {"ml": {"target_rate": 0.51}, "opposite": {"target_rate": 0.49}},
            "3.0": {"ml": {"target_rate": 0.24}, "opposite": {"target_rate": 0.22}},
        }}}}
        ok = target_check(stats, "EURUSD", True, target_pips=3.5, stop_pips=7.4, cost_pips=1.0)
        self.assertTrue(ok["ok"])
        self.assertEqual((ok["bound"], round(ok["hit"], 2)), ("≥", 0.67))
        small = target_check(stats, "EURUSD", True, target_pips=1.8, stop_pips=7.4, cost_pips=1.0)
        self.assertFalse(small["ok"])
        self.assertIn("Maliyet", small["reasons"][0])
        far = target_check(stats, "EURUSD", True, target_pips=17.8, stop_pips=7.4, cost_pips=1.0)
        self.assertFalse(far["ok"])
        self.assertIn("%60", far["reasons"][0])
        low, high = far["ok_pips"]
        self.assertAlmostEqual(low, 2.0)
        self.assertAlmostEqual(high / 7.4, 0.5 + 0.5 * (0.67 - 0.60) / (0.67 - 0.51))
        # ML'siz yön: ML ve ters yönün ortalaması (0.65) %60'ı geçer, 1.0 oranında (0.50) geçmez
        self.assertTrue(target_check(stats, "EURUSD", False, 3.7, 7.4, 1.0)["ok"])
        self.assertFalse(target_check(stats, "EURUSD", False, 7.4, 7.4, 1.0)["ok"])
        self.assertFalse(target_check({}, "EURUSD", True, 3.5, 7.4, 1.0)["ok"])

    def test_no_target_reaches_threshold(self):
        stats = {"pooled": {"ratios": {"0.5": {"ml": {"target_rate": 0.55}, "opposite": {"target_rate": 0.5}}}}}
        check = target_check(stats, "XAUUSD", True, 3.0, 10.0, 1.0)
        self.assertFalse(check["ok"])
        self.assertIsNone(check["ok_pips"][1])

    def test_ml_overrides_given_side(self):
        plan = build_trade_plan("EURUSD", 1.1, 0.0001, 0.0010, 0.6, 30, 100, 1000, 10, 1.0, {}, [], side="SHORT")
        self.assertEqual(plan["side"], "LONG")


if __name__ == "__main__":
    unittest.main()
