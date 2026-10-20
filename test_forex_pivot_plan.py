import math
import unittest

from forex_pivot_plan import ladder, pivot_plan

LEVELS = {"S2": 1.0900, "S1": 1.0950, "P": 1.1000, "R1": 1.1050, "R2": 1.1100}


class LadderTests(unittest.TestCase):
    def test_r3_s3_follow_classic_formula(self):
        # H = 2P - S1 = 1.1050, L = 2P - R1 = 1.0950; R3 = H + 2(P - L) = 1.1150, S3 = L - 2(H - P) = 1.0850
        steps = ladder(LEVELS)
        self.assertAlmostEqual(steps["R3"], 1.1150)
        self.assertAlmostEqual(steps["S3"], 1.0850)
        self.assertEqual(list(steps), ["S3", "S2", "S1", "P", "R1", "R2", "R3"])


class PivotPlanTests(unittest.TestCase):
    def test_above_p_is_long_pullback_to_the_level_below(self):
        plan = pivot_plan(1.1070, LEVELS, atr=0.0010)
        self.assertEqual((plan["side"], plan["entry_name"], plan["target_name"]), ("LONG", "R1", "R2"))
        self.assertAlmostEqual(plan["stop"], 1.1040)
        self.assertAlmostEqual(plan["target"], 1.1100)

    def test_below_p_is_short(self):
        plan = pivot_plan(1.0930, LEVELS, atr=0.0010)
        self.assertEqual((plan["side"], plan["entry_name"], plan["target_name"]), ("SHORT", "S1", "S2"))
        self.assertAlmostEqual(plan["stop"], 1.0960)

    def test_above_r3_waits(self):
        plan = pivot_plan(1.1200, LEVELS, atr=0.0010)
        self.assertEqual(plan["side"], "WAIT")
        self.assertEqual(plan["entry_name"], "R3")

    def test_target_closer_than_stop_waits(self):
        plan = pivot_plan(1.1070, LEVELS, atr=0.0060)
        self.assertEqual(plan["side"], "WAIT")
        self.assertIn("stoptan", plan["reason"])

    def test_missing_inputs_wait(self):
        self.assertEqual(pivot_plan(1.1, {}, 0.001)["side"], "WAIT")
        self.assertEqual(pivot_plan(1.1, LEVELS, math.nan)["side"], "WAIT")
        self.assertEqual(pivot_plan(1.1, LEVELS, 0.001)["side"], "WAIT")  # exactly at P


if __name__ == "__main__":
    unittest.main()
