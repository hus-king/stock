# -*- coding: utf-8 -*-
import unittest

from stock_open_api.simulation.account import AccountState


class TestAccountState(unittest.TestCase):

    def setUp(self):
        self.acct = AccountState(
            cash=100_000.0,
            base_shares=5000,
            position=5000,
        )

    def test_initial_state(self):
        self.assertEqual(self.acct.cash, 100_000.0)
        self.assertEqual(self.acct.position, 5000)
        self.assertEqual(self.acct.base_shares, 5000)
        self.assertEqual(self.acct.available_shares, 0)

    def test_cannot_sell_below_base(self):
        self.assertFalse(self.acct.can_sell(100))

    def test_buy_increases_position(self):
        self.acct.execute_buy(500, 10.0, 0.0003, 5.0)
        self.assertEqual(self.acct.position, 5500)
        self.assertLess(self.acct.cash, 100_000.0)
        self.assertEqual(self.acct.available_shares, 500)

    def test_sell_above_base(self):
        self.acct.execute_buy(500, 10.0, 0.0003, 5.0)
        self.acct.execute_sell(500, 11.0, 0.0003, 5.0, 0.001)
        self.assertEqual(self.acct.position, 5000)
        self.assertGreater(self.acct.realized_pnl, 0)

    def test_cannot_sell_more_than_available(self):
        self.assertFalse(self.acct.can_sell(100))
        self.acct.execute_buy(500, 10.0, 0.0003, 5.0)
        self.assertTrue(self.acct.can_sell(500))
        self.assertFalse(self.acct.can_sell(600))

    def test_cash_constraint_on_buy(self):
        expensive_qty = 20000  # 20000 shares at 10 = 200000, more than cash
        self.assertFalse(self.acct.can_buy(expensive_qty, 10.0, 0.0003, 5.0))


if __name__ == "__main__":
    unittest.main()
