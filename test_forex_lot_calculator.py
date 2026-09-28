import math
import unittest

from forex_lot_calculator import (
    ceil_lot,
    floor_lot,
    lot_comparison,
    lot_for_profit,
    lot_for_risk,
    margin_usd,
    pips_for_profit,
    price_levels,
    risk_level,
    trade_outcome,
)

EURUSD_PIP_VALUE = 10.0  # USD per pip for 1 standard lot


class LotCalculatorTests(unittest.TestCase):
    def test_lot_for_risk_includes_cost(self):
        # $1000, 1% = $10 risk, 9 pip stop + 1 pip cost = 10 pips -> 0.10 lot
        self.assertAlmostEqual(lot_for_risk(1000, 1.0, 9.0, 1.0, EURUSD_PIP_VALUE), 0.10)
        self.assertEqual(lot_for_risk(1000, 1.0, 0.0, 0.0, EURUSD_PIP_VALUE), 0.0)
        self.assertEqual(lot_for_risk(0, 1.0, 10.0, 1.0, EURUSD_PIP_VALUE), 0.0)

    def test_floor_lot_rounds_down_to_broker_step(self):
        self.assertAlmostEqual(floor_lot(0.0999), 0.09)
        self.assertAlmostEqual(floor_lot(0.30), 0.30)
        self.assertEqual(floor_lot(0.004), 0.0)
        self.assertEqual(floor_lot(float("nan")), 0.0)

    def test_ceil_lot_rounds_up_so_target_is_reached(self):
        self.assertAlmostEqual(ceil_lot(0.6211), 0.63)
        self.assertAlmostEqual(ceil_lot(0.50), 0.50)
        self.assertEqual(ceil_lot(0.0), 0.0)

    def test_profit_helpers_are_inverse_of_each_other(self):
        lot = lot_for_profit(100, 21.0, 1.0, EURUSD_PIP_VALUE)
        self.assertAlmostEqual(lot, 0.50)
        self.assertAlmostEqual(pips_for_profit(100, lot, 1.0, EURUSD_PIP_VALUE), 21.0)
        self.assertIsNone(lot_for_profit(100, 1.0, 1.0, EURUSD_PIP_VALUE))

    def test_trade_outcome_for_0_30_lot_on_1000_account(self):
        out = trade_outcome(1000, 0.30, 15.0, 30.0, 1.0, EURUSD_PIP_VALUE)
        self.assertAlmostEqual(out["usd_per_pip"], 3.0)
        self.assertAlmostEqual(out["loss_usd"], 48.0)
        self.assertAlmostEqual(out["profit_usd"], 87.0)
        self.assertAlmostEqual(out["loss_pct"], 4.8)
        self.assertEqual(out["risk_level"], "Yüksek")

    def test_risk_level_boundaries(self):
        self.assertEqual(risk_level(0.5), "Makul")
        self.assertEqual(risk_level(1.0), "Makul")
        self.assertEqual(risk_level(1.5), "Dikkat")
        self.assertEqual(risk_level(10.0), "Çok tehlikeli")

    def test_price_levels_by_side(self):
        stop, target = price_levels(1.10000, "LONG", 10.0, 20.0, 0.0001)
        self.assertAlmostEqual(stop, 1.09900)
        self.assertAlmostEqual(target, 1.10200)
        stop, target = price_levels(150.000, "SHORT", 10.0, 20.0, 0.01)
        self.assertAlmostEqual(stop, 150.100)
        self.assertAlmostEqual(target, 149.800)

    def test_margin_uses_usd_notional(self):
        # EURUSD 1.10, 0.10 lot = 11,000 USD notional; 1:100 -> $110
        self.assertAlmostEqual(margin_usd(0.10, 1.10, 0.0001, 10.0, 100), 110.0)
        # USDJPY 150: pip value = 1000/150 USD; 0.10 lot = 10,000 USD notional -> $100
        self.assertAlmostEqual(margin_usd(0.10, 150.0, 0.01, 1000 / 150, 100), 100.0)
        self.assertIsNone(margin_usd(0.10, 1.10, 0.0001, 10.0, 0))

    def test_lot_comparison_adds_suggested_lot_once(self):
        rows = lot_comparison(1000, 15.0, 30.0, 1.0, EURUSD_PIP_VALUE, 100, extra_lots=(0.06, 0.30))
        lots = [row["lot"] for row in rows]
        self.assertEqual(lots, [0.01, 0.05, 0.06, 0.1, 0.3, 0.5, 1.0])
        self.assertTrue(all(math.isfinite(row["pips_for_target"]) for row in rows))


if __name__ == "__main__":
    unittest.main()
