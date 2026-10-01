import unittest
from datetime import date
from unittest import mock

import pandas as pd
import requests

import forex_calendar
from forex_calendar import (
    expected_outcome,
    extract_days,
    feed_rows,
    fetch_calendar,
    learn_polarity,
    load_polarity,
    pair_directions,
    parse_value,
    release_outcome,
    rows_from_days,
    week_param,
)

EVENT = {
    "id": 1, "name": "Non-Farm Employment Change", "currency": "USD", "dateline": 1788525000,
    "impactName": "high", "actual": "162K", "forecast": "55K", "previous": "-23K",
    "actualBetterWorse": 1, "timeLabel": "3:30pm", "timeMasked": False,
}
PAGE = 'x window.calendarComponentStates[1] = {\ndays: [{"date":"Fri","events":[' + \
    '{"id":1,"name":"Non-Farm Employment Change","currency":"USD","dateline":1788525000,"impactName":"high",' + \
    '"actual":"162K","forecast":"55K","previous":"-23K","actualBetterWorse":1,"timeLabel":"3:30pm","timeMasked":false}' + \
    ']}], other: [1, [2]] };'


class CalendarTests(unittest.TestCase):
    def test_better_release_strengthens_its_currency(self):
        symbols = ["EURUSD=X", "USDJPY=X", "XAUUSD=X", "EURGBP=X"]
        self.assertEqual(pair_directions("USD", "better", symbols), (["USDJPY"], ["EURUSD", "XAUUSD"]))
        self.assertEqual(pair_directions("USD", "worse", symbols), (["EURUSD", "XAUUSD"], ["USDJPY"]))
        self.assertEqual(pair_directions("GBP", "better", symbols), ([], ["EURGBP"]))

    def test_no_direction_without_a_surprise(self):
        self.assertEqual(pair_directions("USD", None, ["EURUSD=X"]), ([], []))

    def test_week_param(self):
        self.assertEqual(week_param(date(2026, 9, 28)), "sep28.2026")
        self.assertEqual(week_param(date(2024, 10, 7)), "oct7.2024")

    def test_extract_days_reads_the_embedded_array(self):
        days = extract_days(PAGE)
        self.assertEqual(days[0]["events"][0]["forecast"], "55K")

    def test_rows_map_impact_outcome_and_masked_time(self):
        masked = {**EVENT, "id": 2, "impactName": "holiday", "actualBetterWorse": 0, "timeLabel": "All Day", "timeMasked": True}
        worse = {**EVENT, "id": 3, "impactName": "medium", "actualBetterWorse": 2}
        df = rows_from_days([{"events": [EVENT, masked, worse]}]).set_index(pd.Index([1, 2, 3]))
        self.assertEqual(df.loc[1, "time"], pd.Timestamp("2026-09-04 12:30", tz="UTC"))
        self.assertEqual((df.loc[1, "impact"], df.loc[1, "outcome"], df.loc[1, "time_label"]), (3, "better", ""))
        self.assertEqual((df.loc[2, "impact"], df.loc[2, "outcome"], df.loc[2, "time_label"]), (0, None, "All Day"))
        self.assertEqual((df.loc[3, "impact"], df.loc[3, "outcome"]), (2, "worse"))

    def test_empty_days_give_empty_frame_with_columns(self):
        self.assertEqual(list(rows_from_days([]).columns), forex_calendar.COLUMNS)

    def test_feed_rows_have_no_actual(self):
        df = feed_rows([{"title": "CPI m/m", "country": "USD", "date": "2026-10-13T08:30:00-04:00",
                         "impact": "High", "forecast": "0.3%", "previous": "0.2%"}])
        self.assertEqual(df.loc[0, "time"], pd.Timestamp("2026-10-13 12:30", tz="UTC"))
        self.assertEqual((df.loc[0, "impact"], df.loc[0, "actual"]), (3, ""))

    def test_falls_back_to_feed_when_the_page_is_blocked(self):
        feed = mock.Mock(**{"json.return_value": [{"title": "CPI", "country": "USD", "date": "2026-10-13T08:30:00-04:00", "impact": "High"}]})
        with mock.patch.object(forex_calendar, "fetch_week_page", side_effect=requests.HTTPError("403")), \
                mock.patch.object(forex_calendar.requests, "get", return_value=feed):
            df, with_actuals = fetch_calendar(date(2026, 10, 12))
        self.assertFalse(with_actuals)
        self.assertEqual(df["event"].tolist(), ["CPI"])

    def test_page_weeks_are_merged_and_sorted(self):
        early = rows_from_days([{"events": [{**EVENT, "id": 5, "dateline": 1788000000}]}])
        late = rows_from_days([{"events": [EVENT]}])
        with mock.patch.object(forex_calendar, "fetch_week_page", side_effect=[late, early, late]):
            df, with_actuals = fetch_calendar(date(2026, 9, 4))
        self.assertTrue(with_actuals)
        self.assertEqual(len(df), 2)
        self.assertTrue(df["time"].is_monotonic_increasing)

    def test_parse_value_handles_units_and_rejects_text(self):
        self.assertEqual(parse_value("162K"), 162000)
        self.assertEqual(parse_value("-0.3%"), -0.3)
        self.assertEqual(parse_value("<0.1%"), 0.1)
        self.assertIsNone(parse_value("2.92|2.1"))
        self.assertIsNone(parse_value(""))

    def test_polarity_is_learned_from_marked_surprises(self):
        jobs = {**EVENT, "id": 10, "actual": "162K", "forecast": "55K", "actualBetterWorse": 1}
        claims = {**EVENT, "id": 11, "name": "Unemployment Claims", "actual": "230K", "forecast": "220K", "actualBetterWorse": 2}
        unmarked = {**EVENT, "id": 12, "name": "Trade Balance", "actual": "1.0B", "forecast": "2.0B", "actualBetterWorse": 0}
        polarity = learn_polarity(rows_from_days([{"events": [jobs, claims, unmarked]}]))
        self.assertEqual(polarity, {("USD", "Non-Farm Employment Change"): 1, ("USD", "Unemployment Claims"): -1})

    def test_saved_polarity_knows_lower_unemployment_is_good(self):
        polarity = load_polarity()
        self.assertEqual(polarity[("USD", "Unemployment Rate")], -1)
        self.assertEqual(polarity[("USD", "Non-Farm Employment Change")], 1)

    def test_expectation_compares_forecast_with_previous_using_polarity(self):
        polarity = {("USD", "Unemployment Rate"): -1, ("USD", "Non-Farm Employment Change"): 1}
        row = lambda event, forecast, previous: pd.Series(
            {"currency": "USD", "event": event, "forecast": forecast, "previous": previous, "actual": "", "outcome": None}
        )
        self.assertEqual(expected_outcome(row("Unemployment Rate", "4.0%", "4.1%"), polarity), "better")
        self.assertEqual(expected_outcome(row("Non-Farm Employment Change", "89K", "162K"), polarity), "worse")
        self.assertEqual(expected_outcome(row("Non-Farm Employment Change", "89K", "89K"), polarity), "equal")
        self.assertIsNone(expected_outcome(row("FOMC Press Conference", "", ""), polarity))
        self.assertIsNone(expected_outcome(row("Unknown Index", "1.0", "2.0"), polarity))

    def test_unmarked_surprise_is_not_reported_as_equal(self):
        polarity = {("USD", "Unemployment Claims"): -1}
        row = lambda actual, outcome=None: pd.Series(
            {"currency": "USD", "event": "Unemployment Claims", "actual": actual, "forecast": "220K", "outcome": outcome}
        )
        self.assertEqual(release_outcome(row("230K"), polarity), "worse")
        self.assertEqual(release_outcome(row("220K"), polarity), "equal")
        self.assertEqual(release_outcome(row("210K", "better"), polarity), "better")


if __name__ == "__main__":
    unittest.main()
