import math
import unittest

from forex_decision_core import decide_mtf_signal


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


if __name__ == "__main__":
    unittest.main()
