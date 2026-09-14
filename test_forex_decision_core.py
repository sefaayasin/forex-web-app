import math
import unittest

from forex_decision_core import (
    bonferroni_adjust,
    circular_shift_timing_test,
    classify_edge_evidence,
    classify_opportunity_readiness,
    decide_mtf_signal,
    position_level_event,
    stationary_bootstrap_mean_test,
)


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


class PositionLevelTests(unittest.TestCase):
    def test_long_levels_do_not_use_short_inequalities(self):
        self.assertEqual(position_level_event("LONG", 1.1020, 1.0990, 1.1020), "TARGET")
        self.assertEqual(position_level_event("LONG", 1.0990, 1.0990, 1.1020), "STOP")
        self.assertEqual(position_level_event("LONG", 1.1005, 1.0990, 1.1020), "NONE")

    def test_short_levels(self):
        self.assertEqual(position_level_event("SHORT", 1.0980, 1.1030, 1.0980), "TARGET")
        self.assertEqual(position_level_event("SHORT", 1.1030, 1.1030, 1.0980), "STOP")


class EdgeValidationTests(unittest.TestCase):
    def test_stationary_bootstrap_detects_consistent_positive_edge(self):
        result = stationary_bootstrap_mean_test([1.0] * 80, simulations=500, seed=7)
        self.assertEqual(result["status"], "ready")
        self.assertAlmostEqual(result["observed_mean"], 1.0)
        self.assertGreater(result["ci_low"], 0)
        self.assertLess(result["p_value"], 0.01)

    def test_stationary_bootstrap_requires_samples(self):
        result = stationary_bootstrap_mean_test([1.0, -1.0], simulations=100)
        self.assertEqual(result["status"], "insufficient")

    def test_stationary_bootstrap_does_not_confirm_zero_edge(self):
        result = stationary_bootstrap_mean_test([-1.0, 1.0] * 50, simulations=1000, seed=9)
        self.assertAlmostEqual(result["observed_mean"], 0.0)
        self.assertLessEqual(result["ci_low"], 0.0)
        self.assertGreater(result["p_value"], 0.05)

    def test_circular_shift_detects_timing_better_than_other_phases(self):
        block = 40
        increments = []
        for _ in range(30):
            increments.extend([2.0] + [-2.0 / (block - 1)] * (block - 1))
        prices = [100.0]
        for increment in increments:
            prices.append(prices[-1] + increment)
        entries = prices[:-1]
        exits = prices[:-1]
        entry_indices = list(range(0, len(entries) - 1, block))
        result = circular_shift_timing_test(
            entries,
            exits,
            entry_indices,
            [1] * len(entry_indices),
            horizon_bars=1,
            pip_size=1.0,
            simulations=2000,
            seed=11,
        )
        self.assertEqual(result["status"], "ready")
        self.assertGreater(result["observed_mean_pips"], result["null_p95_pips"])
        self.assertLess(result["p_value"], 0.05)

    def test_multiple_testing_adjustment_and_evidence_gate(self):
        self.assertAlmostEqual(bonferroni_adjust(0.01, 6), 0.06)
        label, blockers = classify_edge_evidence(
            trade_count=80,
            average_r=0.2,
            r_ci_low=0.05,
            bootstrap_p_adjusted=0.02,
            oos_trade_count=24,
            oos_average_r=0.12,
            min_trades=60,
        )
        self.assertEqual(label, "DOĞRULANDI")
        self.assertEqual(blockers, [])
        label, blockers = classify_edge_evidence(
            trade_count=30,
            average_r=0.2,
            r_ci_low=0.05,
            bootstrap_p_adjusted=0.02,
            oos_trade_count=9,
            oos_average_r=0.12,
            min_trades=60,
        )
        self.assertEqual(label, "YETERSİZ ÖRNEK")
        self.assertTrue(blockers)

    def test_edge_candidate_and_oos_gate(self):
        candidate, candidate_blockers = classify_edge_evidence(
            trade_count=45,
            average_r=0.15,
            r_ci_low=-0.01,
            bootstrap_p_adjusted=0.08,
            oos_trade_count=14,
            oos_average_r=0.10,
            min_trades=60,
        )
        self.assertEqual(candidate, "ADAY / DEMO")
        self.assertTrue(candidate_blockers)
        rejected, blockers = classify_edge_evidence(
            trade_count=80,
            average_r=0.20,
            r_ci_low=0.05,
            bootstrap_p_adjusted=0.02,
            oos_trade_count=24,
            oos_average_r=-0.05,
            min_trades=60,
        )
        self.assertEqual(rejected, "DOĞRULANMADI")
        self.assertTrue(any("OOS" in blocker for blocker in blockers))


if __name__ == "__main__":
    unittest.main()
