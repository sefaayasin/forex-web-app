import unittest

import numpy as np
import pandas as pd

from forex_freshness import reversal_warnings, signal_freshness
from research_freshness import freshness_states, structure_with_score, warning_counts


def random_walk_bars(n: int = 900, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0008, n))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.0005, n))
    index = pd.date_range("2026-01-05", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) + spread,
                         "Low": np.minimum(open_, close) - spread, "Close": close}, index=index)


class VectorisedMatchesModuleTests(unittest.TestCase):
    """The research's vectorised rules must agree with forex_freshness at every bar."""

    @classmethod
    def setUpClass(cls):
        cls.frame = structure_with_score(random_walk_bars())
        cls.states = freshness_states(cls.frame, pip=0.0001)

    def test_freshness_state_matches_module(self):
        codes = {"FRESH": 0, "MATURE": 1, "LATE": 2}
        on_side = self.states[(self.states.side != 0) & self.states.typical.notna()]
        self.assertGreater(len(on_side), 100)
        for when, row in on_side.iloc[::7].iterrows():
            side = "LONG" if row.side > 0 else "SHORT"
            upto = self.frame.loc[:when]
            result = signal_freshness(upto, upto["Score"], side, 0.0001, row.typical)
            self.assertEqual(codes[result["state"]], row.state, when)
            self.assertAlmostEqual(result["moved_pips"], row.moved, places=6)

    def test_warning_counts_match_module(self):
        for sign, side in ((1, "LONG"), (-1, "SHORT")):
            counts = warning_counts(self.frame, sign)
            for pos in range(300, len(self.frame), 11):
                expected = len(reversal_warnings(self.frame.iloc[: pos + 1], side))
                self.assertEqual(counts.iloc[pos], expected, (side, self.frame.index[pos]))


if __name__ == "__main__":
    unittest.main()
