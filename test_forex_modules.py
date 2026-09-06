from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forex_analysis import evaluate_bias, evaluate_market_structure
from forex_config import get_pip_size, normalize_symbol, price_decimals, symbol_pair
from forex_edge import build_edge_validation_report, edge_validation_table
from forex_indicators import add_indicators
import forex_storage


class ConfigModuleTests(unittest.TestCase):
    def test_symbol_and_pip_conventions(self):
        self.assertEqual(normalize_symbol("eur/usd"), "EURUSD=X")
        self.assertEqual(symbol_pair("USDJPY=X"), ("USD", "JPY"))
        self.assertEqual(get_pip_size("USDJPY=X"), 0.01)
        self.assertEqual(price_decimals("USDJPY=X"), 3)
        self.assertEqual(get_pip_size("EURUSD=X"), 0.0001)


class IndicatorModuleTests(unittest.TestCase):
    def test_indicator_contract(self):
        index = pd.date_range("2025-01-01", periods=260, freq="15min", tz="UTC")
        close = pd.Series(np.linspace(1.05, 1.15, len(index)), index=index)
        frame = pd.DataFrame(
            {
                "Open": close.shift(1).fillna(close.iloc[0]),
                "High": close + 0.001,
                "Low": close - 0.001,
                "Close": close,
            },
            index=index,
        )
        result = add_indicators(frame)
        expected = {
            "EMA20", "EMA50", "EMA200", "RSI14", "MACD", "MACDSignal",
            "MACDHist", "BBLow", "BBMid", "BBUp", "ATR14", "Tenkan",
            "Kijun", "SpanA", "SpanB",
        }
        self.assertTrue(expected.issubset(result.columns))
        self.assertGreater(result["ATR14"].dropna().iloc[-1], 0)


class AnalysisModuleTests(unittest.TestCase):
    def test_analysis_contract_on_trending_prices(self):
        index = pd.date_range("2025-01-01", periods=320, freq="15min", tz="UTC")
        trend = np.linspace(1.00, 1.20, len(index))
        pullbacks = 0.004 * np.sin(np.arange(len(index)) * 0.55)
        close = pd.Series(trend + pullbacks, index=index)
        frame = pd.DataFrame(
            {
                "Open": close.shift(1).fillna(close.iloc[0]),
                "High": close + 0.001,
                "Low": close - 0.001,
                "Close": close,
            },
            index=index,
        )
        bias = evaluate_bias(frame)
        structure = evaluate_market_structure(frame)
        self.assertGreater(bias.score, 0)
        self.assertIn(structure.combined_direction, {"LONG", "NEUTRAL"})
        self.assertIsInstance(structure.explanation, str)


class StorageModuleTests(unittest.TestCase):
    def test_journal_and_alert_deduplication(self):
        with TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "test.db"
            with patch.object(forex_storage, "APP_DB_PATH", database):
                forex_storage.add_trade_journal_entry({"Sembol": "EURUSD=X", "Sonuç": "Açık"})
                journal = forex_storage.journal_dataframe()
                self.assertEqual(len(journal), 1)
                self.assertEqual(journal.iloc[0]["Sembol"], "EURUSD=X")
                first = forex_storage.record_alert_once("EURUSD", "LONG", "2025-01-01T00:00", {"x": 1})
                duplicate = forex_storage.record_alert_once("EURUSD", "LONG", "2025-01-01T00:00", {"x": 2})
                self.assertTrue(first)
                self.assertFalse(duplicate)
                self.assertEqual(len(forex_storage.alert_history_dataframe()), 1)


@dataclass
class _BacktestStub:
    trades: pd.DataFrame


class EdgeModuleTests(unittest.TestCase):
    def test_empty_backtest_is_explicitly_insufficient(self):
        report = build_edge_validation_report(
            symbol="EURUSD=X",
            tf_name="15 Dakika",
            period="60d",
            bt=_BacktestStub(pd.DataFrame()),
            cost_pips=1.0,
            fetch_ohlc_fn=lambda *_: pd.DataFrame(),
        )
        self.assertEqual(report["label"], "YETERSİZ ÖRNEK")
        table = edge_validation_table({"TREND": report, "RANGE": report})
        self.assertEqual(list(table["Edge Kanıtı"]), ["YETERSİZ ÖRNEK", "YETERSİZ ÖRNEK"])


if __name__ == "__main__":
    unittest.main()
