import unittest

import numpy as np
import pandas as pd

from research_consensus import ml_at, outcomes, view_sides


def moments(**columns):
    base = {"tf4h": "Alım Yönlü", "tf1h": "Güçlü Alım Yönlü", "tf15": "Alım Yönlü", "tf5": "Alım Yönlü",
            "g15": "Güçlü Alım Yönlü", "g5": "İşlem Yok", "radar_side": "LONG", "radar_score": 77.0, "prob_up": 0.6}
    base.update(columns)
    return pd.DataFrame([base], index=pd.DatetimeIndex([pd.Timestamp("2024-03-04 10:00", tz="UTC")]))


class ResearchConsensusTests(unittest.TestCase):
    def test_all_four_views_long_is_a_long_consensus(self):
        sides = view_sides(moments()).iloc[0]
        self.assertEqual(sides[["radar", "firsatlar", "pariteler", "ml", "technical", "consensus"]].tolist(), [1] * 6)

    def test_ml_disagreeing_keeps_technical_but_drops_consensus(self):
        sides = view_sides(moments(prob_up=0.41)).iloc[0]
        self.assertEqual((sides.technical, sides.ml, sides.consensus), (1, -1, 0))

    def test_radar_below_candidate_threshold_has_no_side(self):
        self.assertEqual(view_sides(moments(radar_score=41.9)).iloc[0].radar, 0)

    def test_one_neutral_timeframe_breaks_pariteler(self):
        self.assertEqual(view_sides(moments(tf5="İşlem Yok")).iloc[0].pariteler, 0)

    def test_conflicting_strong_entries_give_firsatlar_no_side(self):
        self.assertEqual(view_sides(moments(g5="Güçlü Satış Yönlü")).iloc[0].firsatlar, 0)
        self.assertEqual(view_sides(moments(g15="Satış Yönlü", g5="Güçlü Satış Yönlü")).iloc[0].firsatlar, -1)

    def test_missing_ml_probability_has_no_side(self):
        self.assertEqual(view_sides(moments(prob_up=np.nan)).iloc[0].ml, 0)

    def test_outcomes_sign_moves_by_side_and_subtract_cost(self):
        rows = pd.DataFrame({"up_1h": [3.0, 3.0], "up_4h": [-2.0, -2.0], "up_24h": [np.nan, np.nan], "cost": [1.0, 1.0]})
        out = outcomes(rows, pd.Series([1, -1]))
        self.assertEqual(out.net_1h.tolist(), [2.0, -4.0])
        self.assertEqual(out.hit_4h.tolist(), [0.0, 1.0])
        self.assertTrue(out.hit_24h.isna().all())

    def test_ml_reads_the_last_closed_hourly_bar(self):
        probs = pd.Series([0.1, 0.2, 0.3], index=pd.date_range("2024-01-01 08:00", periods=3, freq="h", tz="UTC"))
        self.assertEqual(ml_at(probs, pd.Timestamp("2024-01-01 10:00", tz="UTC")), 0.2)
        self.assertEqual(ml_at(probs, pd.Timestamp("2024-01-01 10:30", tz="UTC")), 0.2)
        self.assertEqual(ml_at(probs, pd.Timestamp("2024-01-01 11:00", tz="UTC")), 0.3)


if __name__ == "__main__":
    unittest.main()
