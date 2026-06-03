# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from stock_open_api.simulation.strategy import GridStrategy
from stock_open_api.simulation.account import AccountState


class TestGridStrategy(unittest.TestCase):

    def setUp(self):
        self.config = {
            "grid_spacing": 0.01,
            "grid_levels": 2,
            "lot_size": 100,
            "base_price_source": "open",
        }
        self.strategy = GridStrategy(self.config)

    def _make_bars(self, open_price=10.0):
        """Create a simple day of 5-minute bars."""
        dates = pd.date_range("2025-01-02 09:35", periods=48, freq="5min")
        prices = [open_price + 0.02 * (i % 10) for i in range(48)]
        return pd.DataFrame({
            "open": prices,
            "high": [p + 0.01 for p in prices],
            "low": [p - 0.01 for p in prices],
            "close": prices,
            "volume": 100000,
        }, index=dates)

    def test_day_start_builds_grid(self):
        bars = self._make_bars(10.0)
        self.strategy.on_day_start(bars)
        self.assertIsNotNone(self.strategy._base_price)
        self.assertEqual(self.strategy._base_price, 10.0)
        self.assertEqual(len(self.strategy._buy_levels), 2)
        self.assertEqual(len(self.strategy._sell_levels), 2)

    def test_buy_signal_at_low_price(self):
        bars = self._make_bars(10.0)
        self.strategy.on_day_start(bars)

        acct = AccountState(cash=100_000, base_shares=5000, position=5000)
        # Bar with low=9.85 (below first grid level at 10*0.99=9.90)
        low_bar = pd.Series({"open": 9.88, "high": 9.88, "low": 9.85, "close": 9.88})
        signal = self.strategy.on_bar(low_bar, 0, acct)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "BUY")

    def test_no_signal_in_mid_range(self):
        bars = self._make_bars(10.0)
        self.strategy.on_day_start(bars)

        acct = AccountState(cash=100_000, base_shares=5000, position=5000)
        # Bar staying near open price
        mid_bar = pd.Series({"open": 10.0, "high": 10.02, "low": 9.98, "close": 10.0})
        signal = self.strategy.on_bar(mid_bar, 0, acct)
        self.assertIsNone(signal)

    def test_sell_signal_after_buy(self):
        bars = self._make_bars(10.0)
        self.strategy.on_day_start(bars)

        acct = AccountState(cash=100_000, base_shares=5000, position=5000)

        # First, trigger buy at low
        low_bar = pd.Series({"open": 9.88, "high": 9.88, "low": 9.85, "close": 9.88})
        signal1 = self.strategy.on_bar(low_bar, 0, acct)
        self.assertIsNotNone(signal1)
        self.assertEqual(signal1.side, "BUY")

        # Execute the buy
        acct.execute_buy(signal1.quantity, signal1.limit_price, 0.0003, 5.0)

        # Then, price goes up to sell level (10*1.01=10.10)
        high_bar = pd.Series({"open": 10.12, "high": 10.15, "low": 10.10, "close": 10.12})
        signal2 = self.strategy.on_bar(high_bar, 1, acct)
        self.assertIsNotNone(signal2)
        self.assertEqual(signal2.side, "SELL")


if __name__ == "__main__":
    unittest.main()
