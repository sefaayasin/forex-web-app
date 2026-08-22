import math
import unittest

from forex_decision_core import classify_opportunity_readiness, decide_mtf_signal


class MtfDecisionTests(unittest.TestCase):
    def test_long_signal_uses_threshold(self):
        self.assertEqual(
            decide_mtf_signal(60, 40, 35, 30, "15 Dakika", 60)[0],
            "LONG",
        )
        self.assertEqual(
            decide_mtf_signal(59, 40, 35, 30, "15 Dakika", 60)[0],
            "NONE",
        )

    def test_short_signal_uses_threshold(self):
        self.assertEqual(
            decide_mtf_signal(-65, -40, -35, -30, "15 Dakika", 60)[0],
            "SHORT",
        )

    def test_five_minute_requires_fifteen_minute_confirmation(self):
        self.assertEqual(
            decide_mtf_signal(70, 45, 40, 10, "5 Dakika", 60)[0],
            "NONE",
        )
        self.assertEqual(
            decide_mtf_signal(70, 45, 40, 30, "5 Dakika", 60)[0],
            "LONG",
        )

    def test_higher_timeframes_must_agree(self):
        self.assertEqual(
            decide_mtf_signal(75, 40, -35, 40, "15 Dakika", 60)[0],
            "NONE",
        )

    def test_missing_score_blocks_signal(self):
        self.assertEqual(
            decide_mtf_signal(70, math.nan, 40, 30, "15 Dakika", 60)[0],
            "NONE",
        )


class OpportunityReadinessTests(unittest.TestCase):
    def test_ready_requires_all_execution_conditions(self):
        state, blockers = classify_opportunity_readiness(
            72, "LONG", True, True, False, 0.60, True
        )
        self.assertEqual(state, "READY")
        self.assertEqual(blockers, [])

    def test_high_score_is_not_enough(self):
        state, blockers = classify_opportunity_readiness(
            95, "SHORT", False, True, False, 0.50, True
        )
        self.assertEqual(state, "WATCH")
        self.assertTrue(any("tetiği yok" in item for item in blockers))

    def test_unrealistic_target_stays_watch(self):
        state, blockers = classify_opportunity_readiness(
            80, "LONG", True, True, False, 1.20, True
        )
        self.assertEqual(state, "WATCH")
        self.assertTrue(any("%80" in item for item in blockers))

    def test_low_score_is_neutral(self):
        self.assertEqual(
            classify_opportunity_readiness(35, "LONG", True, True, False, 0.5, True)[0],
            "NEUTRAL",
        )


if __name__ == "__main__":
    unittest.main()
