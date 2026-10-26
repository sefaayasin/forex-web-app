import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

NOW = pd.Timestamp("2026-10-26 17:26:50", tz="UTC")


def offset_fn():
    tree = ast.parse(Path("forex_web_app_streamlit_v14_alert_decision.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_mt5_server_offset")
    namespace = {"pd": pd}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "offset", "exec"), namespace)
    return namespace["_mt5_server_offset"]


def terminal(tick_time):
    tick = None if tick_time is None else SimpleNamespace(time=int(tick_time.timestamp()))
    return SimpleNamespace(symbol_info_tick=lambda symbol: tick)


class Mt5ServerTimeTests(unittest.TestCase):
    def setUp(self):
        self.fn = offset_fn()
        patcher = patch.object(pd.Timestamp, "now", return_value=NOW)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_live_tick_gives_server_offset(self):
        # Pepperstone: UTC 17:26 iken sunucu 20:26 (UTC+3); tick birkaç saniye eski olabilir
        self.assertEqual(self.fn(terminal(NOW + pd.Timedelta(hours=3, seconds=-4)), "EURUSD"), pd.Timedelta(hours=3))
        self.assertEqual(self.fn(terminal(NOW + pd.Timedelta(hours=2, seconds=-90)), "EURUSD"), pd.Timedelta(hours=2))
        self.assertEqual(self.fn(terminal(NOW + pd.Timedelta(hours=5, minutes=30)), "EURUSD"),
                         pd.Timedelta(hours=5, minutes=30))

    def test_stale_or_missing_tick_falls_back_to_new_york_plus_seven(self):
        # 26 Ekim 2026'da New York yaz saatinde (UTC-4) → +3
        stale = NOW - pd.Timedelta(hours=40, minutes=17)
        self.assertEqual(self.fn(terminal(stale), "EURUSD"), pd.Timedelta(hours=3))
        self.assertEqual(self.fn(terminal(None), "EURUSD"), pd.Timedelta(hours=3))


if __name__ == "__main__":
    unittest.main()
