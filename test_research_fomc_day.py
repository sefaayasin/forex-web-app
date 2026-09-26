import json
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from research_fomc_day import new_york_utc, previous_weekday, price_at, scheduled_fomc_days

PROTOCOL = json.loads((Path(__file__).resolve().parent / "research" / "fomc_day" / "protocol.json").read_text(encoding="utf-8"))


class FomcDayTests(unittest.TestCase):
    def test_new_york_four_pm_follows_daylight_saving(self):
        self.assertEqual(new_york_utc(date(2024, 1, 31), 16), pd.Timestamp("2024-01-31 21:00", tz="UTC"))
        self.assertEqual(new_york_utc(date(2024, 7, 31), 16), pd.Timestamp("2024-07-31 20:00", tz="UTC"))

    def test_price_at_uses_close_of_the_bar_ending_at_that_time(self):
        closes = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-07-31 19:00", "2024-07-31 20:00"], utc=True))
        self.assertEqual(price_at(closes, date(2024, 7, 31), 16), 1.0)

    def test_monday_return_starts_on_friday(self):
        self.assertEqual(previous_weekday(date(2024, 7, 29)), date(2024, 7, 26))
        self.assertEqual(previous_weekday(date(2024, 7, 31)), date(2024, 7, 30))

    def test_scheduled_days_drop_unscheduled_and_add_missing_meeting(self):
        scheduled, excluded = scheduled_fomc_days(PROTOCOL)
        self.assertIn(date(2008, 12, 16), scheduled)
        self.assertIn(date(2024, 11, 7), scheduled)
        for day in (date(2008, 3, 11), date(2010, 5, 9), date(2019, 10, 11), date(2020, 3, 15), date(2025, 8, 22)):
            self.assertNotIn(day, scheduled)
        self.assertIn(date(2020, 3, 16), excluded)
        self.assertEqual(sum(1 for d in scheduled if d.year == 2020), 7)
        self.assertTrue(all(sum(1 for d in scheduled if d.year == y) == 8 for y in range(2008, 2026) if y != 2020))


if __name__ == "__main__":
    unittest.main()
