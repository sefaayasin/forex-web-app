import unittest

import numpy as np
import pandas as pd

from research_indicator_zoo import _psar, previous_day, zoo_rules


def bars(n=1500, seed=11):
    rng = np.random.default_rng(seed)
    c = 1.1 + rng.normal(0, .0005, n).cumsum()
    o = np.r_[c[0], c[:-1]]
    spread = np.abs(rng.normal(0, .0004, n))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + spread, "Low": np.minimum(o, c) - spread,
        "Close": c, "Volume": rng.uniform(100, 1000, n)},
        index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))


class IndicatorZooTests(unittest.TestCase):
    def test_every_rule_ignores_future_bars(self):
        original = bars()
        changed = original.copy()
        changed.iloc[1200:, :4] *= 1.05
        changed.iloc[1200:, 4] *= 4
        first, _ = zoo_rules(original, "1h")
        second, _ = zoo_rules(changed, "1h")
        self.assertGreaterEqual(len(first), 30)
        for name in first:
            np.testing.assert_array_equal(first[name][:1200], second[name][:1200], err_msg=name)

    def test_psar_follows_a_clean_trend(self):
        up = np.linspace(1, 2, 200)
        side = _psar(up + .001, up - .001, .02, .2)
        self.assertTrue((side[20:] == 1).all())
        side = _psar(up[::-1] + .001, up[::-1] - .001, .02, .2)
        self.assertTrue((side[20:] == -1).all())

    def test_previous_day_levels_use_the_prior_weekday(self):
        b = bars(24 * 10)
        prev = previous_day(b)
        tuesday = b[b.index.normalize() == pd.Timestamp("2024-01-02", tz="UTC")]
        monday = b[b.index.normalize() == pd.Timestamp("2024-01-01", tz="UTC")]
        self.assertAlmostEqual(prev.loc[tuesday.index[0], "H"], monday.High.max())
        self.assertAlmostEqual(prev.loc[tuesday.index[-1], "C"], monday.Close.iloc[-1])
        self.assertTrue(prev.loc[monday.index].H.isna().all())


if __name__ == "__main__":
    unittest.main()
