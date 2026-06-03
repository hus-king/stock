# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from stock_open_api.simulation.engine import SimulationEngine
from stock_open_api.simulation.strategy import GridStrategy


class TestSimulationEngine(unittest.TestCase):

    def _make_sample_data(self, open_price=10.0):
        """Create 3 days of simple 5-minute bar data."""
        records = []
        for day_offset, day_str in enumerate(["2025-01-02", "2025-01-03", "2025-01-06"]):
            for minute in range(48):
                t = pd.Timestamp("{} {:02d}:{:02d}:00".format(
                    day_str, 9 + (minute * 5 + 35) // 60,
                    (minute * 5 + 35) % 60))
                # Add some oscillation
                p = open_price + 0.02 * ((minute + day_offset * 7) % 10)
                records.append({
                    "timestamp": t,
                    "open": p,
                    "high": p + 0.01,
                    "low": p - 0.01,
                    "close": p,
                    "volume": 100000,
                    "amount": p * 100000,
                })

        df = pd.DataFrame(records)
        df["date"] = df["timestamp"].dt.date
        return df.set_index("timestamp")

    def test_engine_runs_without_error(self):
        data = self._make_sample_data(10.0)
        strategy = GridStrategy({
            "grid_spacing": 0.005,
            "grid_levels": 3,
            "lot_size": 100,
            "base_price_source": "open",
        })
        engine = SimulationEngine(
            symbol="000001",
            start_cash=100_000,
            base_shares=5000,
        )
        engine.run(data, strategy)

        self.assertGreater(len(engine.daily_stats_list), 0)
        self.assertEqual(engine.daily_stats_list[0]["start_position"], 5000)

    def test_position_returns_to_base(self):
        data = self._make_sample_data(10.0)
        strategy = GridStrategy({
            "grid_spacing": 0.002,
            "grid_levels": 5,
            "lot_size": 100,
            "base_price_source": "open",
        })
        engine = SimulationEngine(
            symbol="000001",
            start_cash=100_000,
            base_shares=5000,
        )
        engine.run(data, strategy)

        # Every day must end at base_shares
        for ds in engine.daily_stats_list:
            self.assertEqual(
                ds["end_position"], engine.base_shares,
                "Day {} ended with {} shares, expected {}".format(
                    ds["date"], ds["end_position"], engine.base_shares))

    def test_no_short_selling(self):
        data = self._make_sample_data(10.0)
        strategy = GridStrategy({
            "grid_spacing": 0.002,
            "grid_levels": 5,
            "lot_size": 100,
            "base_price_source": "open",
        })
        engine = SimulationEngine(
            symbol="000001",
            start_cash=100_000,
            base_shares=1000,  # Small base, limited sell capacity
        )
        engine.run(data, strategy)

        for t in engine.trades:
            if t.side == "SELL":
                self.assertGreaterEqual(t.quantity, 100)
                self.assertEqual(t.quantity % 100, 0)

    def test_lot_size_enforced(self):
        data = self._make_sample_data(10.0)
        strategy = GridStrategy({
            "grid_spacing": 0.002,
            "grid_levels": 5,
            "lot_size": 100,
            "base_price_source": "open",
        })
        engine = SimulationEngine(
            symbol="000001",
            start_cash=100_000,
            base_shares=5000,
        )
        engine.run(data, strategy)

        for t in engine.trades:
            self.assertEqual(t.quantity % 100, 0,
                             "Trade quantity {} is not a multiple of 100".format(t.quantity))

    def test_stats_have_expected_keys(self):
        data = self._make_sample_data(10.0)
        strategy = GridStrategy({
            "grid_spacing": 0.005,
            "grid_levels": 3,
            "lot_size": 100,
            "base_price_source": "open",
        })
        engine = SimulationEngine(
            symbol="000001",
            start_cash=100_000,
            base_shares=5000,
        )
        engine.run(data, strategy)
        stats = engine.get_stats()

        self.assertIsNotNone(stats.total_days)
        self.assertIsNotNone(stats.total_trades)
        self.assertIsNotNone(stats.total_pnl)
        self.assertIsNotNone(stats.day_win_rate)


if __name__ == "__main__":
    unittest.main()
