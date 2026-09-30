import unittest
from datetime import date

import numpy as np
import pandas as pd

from research_calendar_sources import (
    SOURCES,
    add_scaled_errors,
    best_source,
    fxstreet_value,
    head_to_head,
    holm,
    match_source,
    nasdaq_query_date,
    nasdaq_release_time,
    name_similarity,
    parse_investing_page,
    parse_value,
)

T = pd.Timestamp("2024-10-04 12:30", tz="UTC")


def reference(rows):
    return pd.DataFrame(rows, columns=["id", "series", "name", "currency", "time", "actual"])


class ParsingTests(unittest.TestCase):
    def test_parse_value_handles_suffixes_signs_and_blanks(self):
        self.assertEqual(parse_value("162K"), 162000.0)
        self.assertEqual(parse_value("1,826K"), 1826000.0)
        self.assertEqual(parse_value("-0.3%"), -0.3)
        self.assertEqual(parse_value("−0.5%"), -0.5)
        self.assertEqual(parse_value("¥-0.5T"), -0.5e12)
        self.assertAlmostEqual(parse_value("3.097T"), 3.097e12)
        self.assertIsNone(parse_value("&nbsp;"))
        self.assertIsNone(parse_value(" "))
        self.assertIsNone(parse_value(None))

    def test_fxstreet_value_applies_potency(self):
        self.assertEqual(fxstreet_value(254.0, "K"), 254000.0)
        self.assertEqual(fxstreet_value(0.4, "ZERO"), 0.4)
        self.assertEqual(fxstreet_value(2.4, None), 2.4)
        self.assertIsNone(fxstreet_value(None, "K"))

    def test_nasdaq_time_is_new_york_and_query_date_is_next_day(self):
        self.assertEqual(nasdaq_release_time(date(2024, 10, 4), "08:30"), pd.Timestamp("2024-10-04 12:30", tz="UTC"))
        self.assertEqual(nasdaq_release_time(date(2024, 12, 6), "08:30"), pd.Timestamp("2024-12-06 13:30", tz="UTC"))
        self.assertIsNone(nasdaq_release_time(date(2024, 10, 4), "All Day"))
        self.assertEqual(nasdaq_query_date(date(2024, 10, 4)), date(2024, 10, 5))

    def test_name_similarity_separates_monthly_and_yearly(self):
        self.assertGreater(name_similarity("CPI m/m", "CPI (MoM) (Aug)"), name_similarity("CPI m/m", "CPI (YoY) (Aug)"))

    def test_investing_row_parsing(self):
        page = (
            '<tr id="eventRowId_555500" class="js-event-item" data-event-datetime="2026/09/04 12:30:00">'
            '<td class="left flagCur noWrap"><span title="United States" class="ceFlags">&nbsp;</span> USD</td>'
            '<td class="left textNum sentiment"><i class="grayFullBullishIcon"></i><i class="grayFullBullishIcon"></i>'
            '<i class="grayFullBullishIcon"></i></td>'
            '<td class="left event" title="x"><a href="/x" target="_blank"> Nonfarm Payrolls (Aug)</a> </td>'
            '<td class="bold act" id="eventActual_555500">162K</td>'
            '<td class="fore" id="eventForecast_555500">55K</td>'
            '<td class="prev" id="eventPrevious_555500"><span title="Revised From -23K">21K</span></td></tr>'
        )
        [row] = parse_investing_page(page)
        self.assertEqual(row["currency"], "USD")
        self.assertEqual(row["importance"], 3)
        self.assertEqual(row["name"], "Nonfarm Payrolls (Aug)")
        self.assertEqual((row["actual"], row["forecast"]), (162000.0, 55000.0))
        self.assertEqual(row["time"], pd.Timestamp("2026-09-04 12:30", tz="UTC"))


class MatchingTests(unittest.TestCase):
    def test_match_needs_same_currency_time_and_actual_and_prefers_the_closer_name(self):
        ref = reference([
            (1, 10, "CPI m/m", "USD", T, 0.2),
            (2, 11, "CPI y/y", "USD", T, 2.4),
            (3, 12, "GDP q/q", "EUR", T, 0.2),
        ])
        rows = pd.DataFrame([
            {"name": "CPI (YoY)", "currency": "USD", "time": T, "actual": 2.4, "forecast": 2.3},
            {"name": "Core CPI (MoM)", "currency": "USD", "time": T, "actual": 0.2, "forecast": 0.3},
            {"name": "CPI (MoM)", "currency": "USD", "time": T + pd.Timedelta(minutes=2), "actual": 0.2, "forecast": 0.1},
            {"name": "GDP (QoQ)", "currency": "EUR", "time": T + pd.Timedelta(minutes=30), "actual": 0.2, "forecast": 0.2},
        ])
        matched = match_source(ref, rows).set_index("id")
        self.assertEqual(matched.loc[1, "name"], "CPI (MoM)")
        self.assertEqual(matched.loc[2, "forecast"], 2.3)
        self.assertNotIn(3, matched.index)

    def test_each_source_row_is_used_once(self):
        ref = reference([(1, 10, "Claims", "USD", T, 225e3), (2, 11, "Claims", "USD", T, 225e3)])
        rows = pd.DataFrame([{"name": "Claims", "currency": "USD", "time": T, "actual": 225e3, "forecast": 222e3}])
        self.assertEqual(len(match_source(ref, rows)), 1)


class StatisticsTests(unittest.TestCase):
    def test_scale_falls_back_to_actual_minus_previous_for_short_series(self):
        panel = pd.DataFrame({
            "series": [1, 1], "actual": [10.0, 12.0], "previous": [8.0, 10.0],
            **{f"err_{s}": [1.0, 3.0] for s in SOURCES},
        })
        scaled = add_scaled_errors(panel)
        self.assertEqual(scaled["scale"].tolist(), [2.0, 2.0])
        self.assertEqual(scaled["scaled_investing"].tolist(), [0.5, 1.5])

    def test_head_to_head_counts_ties_and_missing(self):
        a = pd.Series([1.0, 2.0, 3.0, np.nan])
        b = pd.Series([2.0, 2.0, 1.0, 1.0])
        h = head_to_head(a, b)
        self.assertEqual((h["n"], h["a_closer"], h["tie"], h["b_closer"]), (3, 1, 1, 1))

    def test_holm_is_monotone_and_capped(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm([0.9, 0.8]), [1.0, 1.0])

    def test_best_source_requires_beating_every_other_source(self):
        def pair(a, b, a_closer, b_closer, p, low, high):
            return {"a": a, "b": b, "a_closer": a_closer, "b_closer": b_closer, "sign_p_holm": p, "diff_ci_low": low, "diff_ci_high": high}
        winning = [
            pair("investing", "forexfactory", 5, 50, 0.001, 0.1, 0.3),
            pair("investing", "fxstreet", 20, 25, 0.5, -0.1, 0.1),
            pair("investing", "nasdaq", 20, 22, 0.8, -0.1, 0.1),
            pair("forexfactory", "fxstreet", 1, 60, 0.001, 0.2, 0.4),
            pair("forexfactory", "nasdaq", 1, 60, 0.001, 0.2, 0.4),
            pair("fxstreet", "nasdaq", 30, 10, 0.2, -0.2, 0.01),
        ]
        self.assertIsNone(best_source({"pairs": winning}))
        winning[1] = pair("investing", "fxstreet", 10, 40, 0.001, 0.05, 0.2)
        winning[5] = pair("fxstreet", "nasdaq", 40, 10, 0.001, -0.2, -0.05)
        self.assertEqual(best_source({"pairs": winning}), "fxstreet")


if __name__ == "__main__":
    unittest.main()
