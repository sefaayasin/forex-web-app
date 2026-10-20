import unittest

import pandas as pd

from forex_pivots import levels_for_day, pivot_levels


def daily_bars(rows, tz="UTC"):
    index = pd.DatetimeIndex([pd.Timestamp(day) for day, *_ in rows])
    index = index.tz_localize(tz) if tz else index
    return pd.DataFrame([dict(High=h, Low=l, Close=c) for _, h, l, c in rows], index=index)


class PivotLevelTests(unittest.TestCase):
    def test_classic_formula(self):
        levels = pivot_levels(daily_bars([("2026-10-14", 1.3100, 1.2900, 1.3060)])).iloc[0]
        pivot = (1.3100 + 1.2900 + 1.3060) / 3
        self.assertAlmostEqual(levels["P"], pivot)
        self.assertAlmostEqual(levels["R1"], 2 * pivot - 1.2900)
        self.assertAlmostEqual(levels["S1"], 2 * pivot - 1.3100)
        self.assertAlmostEqual(levels["R2"], pivot + 0.0200)
        self.assertAlmostEqual(levels["S2"], pivot - 0.0200)
        self.assertAlmostEqual(levels["range"], 0.0200)

    def test_monday_uses_friday_and_weekend_bars_are_dropped(self):
        daily = daily_bars([("2026-10-16", 1.31, 1.29, 1.30),   # Friday
                            ("2026-10-18", 1.40, 1.20, 1.30),   # Sunday stub
                            ("2026-10-19", 1.32, 1.30, 1.31)])  # Monday
        monday = levels_for_day(daily, "2026-10-19 08:00")
        self.assertEqual(monday["source_day"], pd.Timestamp("2026-10-16"))
        self.assertAlmostEqual(monday["range"], 0.02)
        self.assertEqual(levels_for_day(daily, "2026-10-20")["source_day"], pd.Timestamp("2026-10-19"))

    def test_today_bar_is_not_used_and_no_history_is_none(self):
        daily = daily_bars([("2026-10-19", 1.32, 1.30, 1.31), ("2026-10-20", 1.50, 1.10, 1.40)])
        self.assertEqual(levels_for_day(daily, pd.Timestamp("2026-10-20 15:00", tz="UTC"))["source_day"],
                         pd.Timestamp("2026-10-19"))
        self.assertIsNone(levels_for_day(daily, "2026-10-19"))

    def test_wall_date_is_kept_for_london_timestamps(self):
        daily = daily_bars([("2026-10-19", 1.32, 1.30, 1.31)], tz="Europe/London")
        self.assertEqual(pivot_levels(daily).index[0], pd.Timestamp("2026-10-19"))


if __name__ == "__main__":
    unittest.main()
