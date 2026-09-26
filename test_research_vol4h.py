import unittest

import numpy as np
import pandas as pd

from research_vol4h import DEPLOYABLE_MODELS, clock_probability, select_config, simple_logistic_probability


def session_pattern(n=2000, noise=0.1, seed=1):
    index = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    y = ((index.hour >= 12) & (index.hour < 16)).astype(int)
    y = np.where(rng.random(n) < noise, 1 - y, y)
    x = pd.DataFrame({"rv_ratio_4": rng.random(n), "rv_ratio_24": rng.random(n)}, index=index)
    return index, x, y


class Vol4hTests(unittest.TestCase):
    def test_selection_uses_a_deployable_four_hour_volatility_candidate(self):
        config = select_config()
        self.assertEqual(config["horizon"], 4)
        self.assertIn(config["model"], DEPLOYABLE_MODELS)
        self.assertNotEqual(config["bundle"], "cot_exploratory")

    def test_clock_baseline_learns_the_session_from_training_rows_only(self):
        index, _, y = session_pattern()
        train, rows = np.arange(0, 1500, 4), np.arange(1500, 2000, 4)
        p = clock_probability(index, y, train, rows)
        self.assertGreater(((p >= .5) == y[rows]).mean(), .85)
        self.assertTrue(np.all(p[index.hour[rows] == 0] < .5))

    def test_clock_baseline_falls_back_to_prevalence_for_unseen_hours(self):
        index, _, y = session_pattern()
        train = np.arange(0, 1500, 4)
        unseen = np.array([1501])
        self.assertAlmostEqual(clock_probability(index, y, train, unseen)[0], y[train].mean())

    def test_simple_logistic_baseline_returns_probabilities(self):
        index, x, y = session_pattern()
        train, rows = np.arange(0, 1500, 4), np.arange(1500, 2000, 4)
        p = simple_logistic_probability(x, y, train, rows)
        self.assertEqual(len(p), len(rows))
        self.assertTrue(((p >= 0) & (p <= 1)).all())


if __name__ == "__main__":
    unittest.main()
