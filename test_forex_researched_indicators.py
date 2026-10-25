import unittest

import numpy as np
import pandas as pd

from forex_researched_indicators import (connors_rsi, fisher_transform, frama, kama,
    laguerre_rsi, researched_features, researched_rule_states, roofing_filter, super_smoother, supertrend)
from research_new_indicators import choose_candidate


class NewIndicatorTests(unittest.TestCase):
    def bars(self, n=1200):
        rng = np.random.default_rng(867)
        c = 1.1 + rng.normal(0, .0002, n).cumsum()
        return pd.DataFrame({"Open": c, "High": c+.0003, "Low": c-.0003, "Close": c},
            index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"))

    def test_future_and_truncated_history_do_not_rewrite_any_indicator_or_rule(self):
        bars = self.bars()
        changed = bars.copy()
        changed.iloc[813:] *= 2
        original, modified = researched_features(bars), researched_features(changed)
        pd.testing.assert_frame_equal(original.iloc[:813], modified.iloc[:813])
        pd.testing.assert_frame_equal(original.iloc[:813], researched_features(bars.iloc[:813]))
        rules, altered = researched_rule_states(bars), researched_rule_states(changed)
        for name in rules:
            np.testing.assert_array_equal(rules[name][:813], altered[name][:813], err_msg=name)

    def test_flat_prices_are_stable_and_neutral(self):
        bars = self.bars()
        bars[["Open", "High", "Low", "Close"]] = 1.
        np.testing.assert_allclose(super_smoother(bars.Close), 1.)
        np.testing.assert_allclose(roofing_filter(bars.Close), 0., atol=1e-12)
        np.testing.assert_allclose(fisher_transform(bars.Close).dropna(), 0.)
        np.testing.assert_allclose(frama(bars).dropna(), 1.)
        np.testing.assert_allclose(kama(bars.Close).dropna(), 1.)
        np.testing.assert_allclose(laguerre_rsi(bars.Close).dropna(), .5)

    def test_super_smoother_matches_independent_scalar_recurrence(self):
        values = self.bars().Close
        angle = np.sqrt(2)*np.pi/10
        a = np.exp(-angle)
        feedback, last, previous = 2*a*np.cos(angle), values.iloc[0], values.iloc[0]
        gain = 1-feedback+a*a
        expected = []
        for i, price in enumerate(values):
            before = values.iloc[i-1] if i else price
            current = gain*(price+before)/2+feedback*last-a*a*previous
            expected.append(current)
            previous, last = last, current
        np.testing.assert_allclose(super_smoother(values), expected, atol=1e-12)

    def test_adaptive_average_stays_inside_observed_price_bounds(self):
        bars = self.bars()
        output = kama(bars.Close)
        valid = output.notna()
        self.assertTrue((output[valid] <= bars.Close.cummax()[valid]+1e-12).all())
        self.assertTrue((output[valid] >= bars.Close.cummin()[valid]-1e-12).all())

    def test_connors_strict_percent_rank_excludes_current_return(self):
        # Every return strictly increases; price/streak RSIs also rise -> 100.
        price = pd.Series(np.cumprod(1+np.arange(1,250)/10000))
        self.assertAlmostEqual(connors_rsi(price, 100).iloc[-1], 100.)
        reversed_price = pd.Series(np.cumprod(1-np.arange(1,250)/10000))
        self.assertAlmostEqual(connors_rsi(reversed_price,100).iloc[-1], 0.)

    def test_supertrend_updates_only_after_price_breaks_its_trailing_band(self):
        bars = self.bars()
        st = supertrend(bars)
        for i in np.flatnonzero(st.side.diff().abs().to_numpy()==2):
            if st.side.iloc[i] == 1:
                self.assertGreater(bars.Close.iloc[i], st.line.iloc[i-1])
            else:
                self.assertLess(bars.Close.iloc[i], st.line.iloc[i-1])

    def test_candidate_choice_uses_development_only_and_enforces_sample_size(self):
        rows = [{"rule": "too_small", "audit_mean": 999,
                 "subperiods": [{"trades": 99,"stress_mean_usd":10}, {"trades": 200,"stress_mean_usd":10}]},
                {"rule": "stable", "audit_mean": -999,
                 "subperiods": [{"trades": 200,"stress_mean_usd":1}, {"trades": 200,"stress_mean_usd":2}]},
                {"rule": "unstable", "audit_mean": 999,
                 "subperiods": [{"trades": 200,"stress_mean_usd":5}, {"trades": 200,"stress_mean_usd":0}]}]
        self.assertEqual(choose_candidate(rows)["rule"], "stable")


if __name__ == "__main__":
    unittest.main()
