import itertools
import unittest

import pandas as pd

from forex_overall import overall_reading, view_sides_now
from research_consensus import view_sides

LONG_TFS = {"4 Saat": "Alım Yönlü", "1 Saat": "Güçlü Alım Yönlü", "15 Dakika": "Alım Yönlü", "5 Dakika": "Alım Yönlü"}


def sides(**overrides):
    base = {"radar": 1, "firsatlar": 1, "pariteler": 1, "ml": 1}
    base.update(overrides)
    return base


class ForexOverallTests(unittest.TestCase):
    def test_views_match_the_research_rules(self):
        labels = ["Güçlü Alım Yönlü", "Alım Yönlü", "İşlem Yok", "Satış Yönlü", "Güçlü Satış Yönlü"]
        cases = list(itertools.product(labels[:3], labels[2:], ["LONG", "SHORT", "NONE"], [30.0, 77.0], [0.4, 0.5, 0.6]))
        rows = pd.DataFrame(
            [{"tf4h": "Alım Yönlü", "tf1h": tf1h, "tf15": "Alım Yönlü", "tf5": "Alım Yönlü", "g15": g15, "g5": tf1h,
              "radar_side": radar, "radar_score": score, "prob_up": prob} for g15, tf1h, radar, score, prob in cases])
        expected = view_sides(rows)
        for i, (g15, tf1h, radar, score, prob) in enumerate(cases):
            got = view_sides_now({**LONG_TFS, "1 Saat": tf1h}, (g15, tf1h), radar, score, prob)
            self.assertEqual(got, expected.iloc[i][["radar", "firsatlar", "pariteler", "ml"]].to_dict(), cases[i])

    def test_missing_ml_has_no_side(self):
        self.assertEqual(view_sides_now(LONG_TFS, ("Güçlü Alım Yönlü", "İşlem Yok"), "LONG", 77, None)["ml"], 0)

    def test_all_four_agreeing_is_reported_but_not_a_call(self):
        reading = overall_reading(sides())
        self.assertEqual((reading["pattern"], reading["lean_text"]), ("consensus", "LONG"))
        self.assertEqual(reading["verdict"], "Teknik görüşlere bakarak girme")
        self.assertIn("−2,1 pip", reading["evidence"])

    def test_technical_long_ml_short_is_a_conflict(self):
        reading = overall_reading(sides(ml=-1))
        self.assertEqual(reading["pattern"], "conflict")
        self.assertIn("Teknik görüşler LONG, ML SHORT", reading["headline"])

    def test_mixed_and_empty_views(self):
        self.assertEqual(overall_reading(sides(firsatlar=0, ml=-1))["pattern"], "mixed")
        self.assertEqual(overall_reading(sides(firsatlar=0, ml=-1))["lean"], 1)
        self.assertEqual(overall_reading(sides(radar=0, firsatlar=0, pariteler=0, ml=0))["pattern"], "none")
        self.assertEqual(overall_reading(sides(pariteler=0))["headline"], "3/4 görüş LONG, kalanı yön göstermiyor")
        self.assertIn("çelişiyor (2 LONG · 1 SHORT)", overall_reading(sides(firsatlar=0, ml=-1))["headline"])

    def test_warnings_for_news_cost_and_volatility(self):
        reading = overall_reading(sides(), news_minutes=95, news_title="NZD CPI", cost_share=0.24, volatility_level="high")
        self.assertEqual(len(reading["warnings"]), 3)
        self.assertIn("1 sa 35 dk sonra", reading["warnings"][0])
        self.assertIn("%24", reading["warnings"][1])

    def test_far_news_and_cheap_cost_give_no_warning(self):
        self.assertEqual(overall_reading(sides(), news_minutes=300, cost_share=0.1, volatility_level="normal")["warnings"], [])


if __name__ == "__main__":
    unittest.main()
