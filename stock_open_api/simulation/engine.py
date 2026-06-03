# -*- coding: utf-8 -*-
"""
@File    : engine.py
@Date    : 2026-06-01
"""
from datetime import datetime

import pandas as pd

from .account import AccountState
from .order import Trade, DailyTrades


LOT_SIZE = 100


class SimulationEngine:
    """Custom simulation engine for intraday T-trading.

    Iterates over minute bars grouped by day, feeding each bar to the
    strategy and executing signals with constraint enforcement.
    At end of each day, forces position back to base_shares (closeout).
    """

    def __init__(self, symbol, start_cash, base_shares,
                 commission_rate=0.0003, min_commission=5.0,
                 stamp_tax=0.001, fill_policy="next_open"):
        self.symbol = symbol
        self.start_cash = start_cash
        self.base_shares = base_shares
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax = stamp_tax
        self.fill_policy = fill_policy  # "next_open" or "this_close"

        self.account = AccountState(
            cash=start_cash,
            base_shares=base_shares,
            position=base_shares,
        )
        self.trades = []
        self.daily_stats_list = []
        self._trade_id = 0

    def run(self, data, strategy):
        """Run simulation over all trading days.

        Args:
            data: DataFrame indexed by timestamp with columns
                  [open, high, low, close, volume, date]
            strategy: BaseStrategy instance
        """
        if "date" not in data.columns:
            data["date"] = data.index.date

        strategy.prepare(data)

        dates = sorted(data["date"].unique())
        all_trades = []

        for day_idx, day in enumerate(dates):
            day_bars = data[data["date"] == day].sort_index()
            if day_bars.empty:
                continue

            strategy.on_day_start(day_bars)

            day_start_value = self._account_value(day_bars.iloc[0]["open"])
            self.account.day_start_value = day_start_value
            day_trades = []
            pending_signal = None
            day_initial_position = self.account.position
            day_max_position = self.account.position

            for i in range(len(day_bars)):
                # Track max intraday position
                if self.account.position > day_max_position:
                    day_max_position = self.account.position
                bar = day_bars.iloc[i]
                bar_time = day_bars.index[i]

                # Execute pending signal at this bar's open
                if pending_signal is not None:
                    fill_price = float(bar["open"])
                    fill_ok = self._fill_signal(
                        pending_signal, fill_price, bar_time, day_trades)
                    if fill_ok:
                        strategy.on_fill(pending_signal.side, pending_signal.quantity, fill_price)
                    pending_signal = None

                # Query strategy
                signal = strategy.on_bar(bar, i, self.account)
                if signal is not None:
                    if self.fill_policy == "this_close":
                        fill_price = float(bar["close"])
                        fill_ok = self._fill_signal(signal, fill_price, bar_time, day_trades)
                        if fill_ok:
                            strategy.on_fill(signal.side, signal.quantity, fill_price)
                    else:
                        pending_signal = signal

            # Fill pending signal at last bar's close if still open
            if pending_signal is not None:
                last_bar = day_bars.iloc[-1]
                fill_price = float(last_bar["close"])
                fill_ok = self._fill_signal(
                    pending_signal, fill_price,
                    day_bars.index[-1], day_trades)
                if fill_ok:
                    strategy.on_fill(pending_signal.side, pending_signal.quantity, fill_price)

            # Closeout: force position back to base_shares only if no pending sells
            last_bar = day_bars.iloc[-1]
            last_time = day_bars.index[-1]
            closeout_price = float(last_bar["close"])
            pending_sell_qty = sum(
                e["qty"] for e in getattr(strategy, "_pending_sells", [])
            )
            if pending_sell_qty == 0:
                self._closeout(closeout_price, last_time, day_trades)

            strategy.on_day_end()

            # Update ML selector history cache with today's bars
            ml_sel = getattr(strategy, "ml_selector", None)
            if ml_sel is not None and hasattr(ml_sel, "update_history"):
                ml_sel.update_history(day_bars)

            # Record daily stats
            day_end_value = self._account_value(closeout_price)
            day_pnl = day_end_value - day_start_value
            daily = {
                "date": day,
                "open_price": float(day_bars.iloc[0]["open"]),
                "close_price": closeout_price,
                "high": float(day_bars["high"].max()),
                "low": float(day_bars["low"].min()),
                "num_trades": len([t for t in day_trades if t.tag != "closeout"]),
                "num_closeout": len([t for t in day_trades if t.tag == "closeout"]),
                "daily_pnl": day_pnl,
                "daily_pnl_pct": day_pnl / day_start_value if day_start_value > 0 else 0,
                "buy_volume": sum(t.quantity for t in day_trades if t.side == "BUY"),
                "sell_volume": sum(t.quantity for t in day_trades if t.side == "SELL"),
                "start_position": day_initial_position,
                "max_position": day_max_position,
                "end_position": self.account.position,
                "total_commission": sum(t.commission for t in day_trades),
            }
            self.daily_stats_list.append(daily)
            all_trades.extend(day_trades)

        self.trades = all_trades

    def _fill_signal(self, signal, fill_price, timestamp, day_trades):
        """Execute a signal, enforcing position and cash constraints."""
        qty = signal.quantity
        qty = qty - (qty % LOT_SIZE)
        if qty < LOT_SIZE:
            return False

        if signal.side == "BUY":
            qty = min(qty, self._max_buyable(fill_price))
            if qty < LOT_SIZE:
                return False
            self._trade_id += 1
            cost = qty * fill_price
            comm = max(cost * self.commission_rate, self.min_commission)
            self.account.execute_buy(qty, fill_price, self.commission_rate,
                                     self.min_commission)
            trade = Trade(
                trade_id=self._trade_id, symbol=self.symbol,
                side="BUY", quantity=qty, price=fill_price,
                commission=comm, timestamp=timestamp)

        elif signal.side == "SELL":
            qty = min(qty, self.account.available_shares)
            qty = qty - (qty % LOT_SIZE)
            if qty < LOT_SIZE:
                return False
            self._trade_id += 1
            proceeds = qty * fill_price
            comm = max(proceeds * self.commission_rate, self.min_commission)
            stamp = proceeds * self.stamp_tax
            self.account.execute_sell(qty, fill_price, self.commission_rate,
                                      self.min_commission, self.stamp_tax)
            trade = Trade(
                trade_id=self._trade_id, symbol=self.symbol,
                side="SELL", quantity=qty, price=fill_price,
                commission=comm + stamp, timestamp=timestamp)

        else:
            return False

        day_trades.append(trade)
        return True

    def _closeout(self, price, timestamp, day_trades):
        """Force position back to base_shares at end of day."""
        delta = self.account.position - self.account.base_shares
        if delta == 0:
            return

        side = "SELL" if delta > 0 else "BUY"
        qty = abs(delta)
        qty = qty - (qty % LOT_SIZE)
        if qty < LOT_SIZE:
            return

        if side == "SELL":
            qty = min(qty, self.account.available_shares)
        else:
            qty = min(qty, self._max_buyable(price))

        qty = qty - (qty % LOT_SIZE)
        if qty < LOT_SIZE:
            return

        self._trade_id += 1
        if side == "BUY":
            self.account.execute_buy(qty, price, self.commission_rate,
                                     self.min_commission)
            comm = max(qty * price * self.commission_rate, self.min_commission)
        else:
            self.account.execute_sell(qty, price, self.commission_rate,
                                      self.min_commission, self.stamp_tax)
            comm = max(qty * price * self.commission_rate, self.min_commission)

        trade = Trade(
            trade_id=self._trade_id, symbol=self.symbol,
            side=side, quantity=qty, price=price,
            commission=comm, timestamp=timestamp, tag="closeout")
        day_trades.append(trade)

    def _max_buyable(self, price):
        cost_per_lot = price * LOT_SIZE * (1 + self.commission_rate)
        if cost_per_lot <= 0:
            return 0
        lots = int(self.account.cash / cost_per_lot)
        return lots * LOT_SIZE

    def _account_value(self, current_price):
        return self.account.cash + self.account.position * current_price

    def get_stats(self):
        """Compute and return SimulationStats."""
        from .metrics import MetricsCalculator
        calc = MetricsCalculator()
        return calc.compute(
            trades=self.trades,
            daily_stats=self.daily_stats_list,
            start_cash=self.start_cash,
            base_shares=self.base_shares,
            start_price=self.daily_stats_list[0]["open_price"] if self.daily_stats_list else 0,
            end_price=self.daily_stats_list[-1]["close_price"] if self.daily_stats_list else 0,
        )

    def get_daily_df(self):
        return pd.DataFrame(self.daily_stats_list)

    def get_trades_df(self):
        return pd.DataFrame([{
            "trade_id": t.trade_id,
            "side": t.side,
            "quantity": t.quantity,
            "price": t.price,
            "commission": t.commission,
            "timestamp": t.timestamp,
            "tag": t.tag,
        } for t in self.trades])
