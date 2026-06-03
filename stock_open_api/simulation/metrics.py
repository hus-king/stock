# -*- coding: utf-8 -*-
"""
@File    : metrics.py
@Date    : 2026-06-01
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SimulationStats:
    total_days: int = 0
    trading_days: int = 0
    total_trades: int = 0
    buy_count: int = 0
    sell_count: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_daily_pnl: float = 0.0
    best_day_pnl: float = 0.0
    worst_day_pnl: float = 0.0
    winning_days: int = 0
    losing_days: int = 0
    day_win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    total_commission: float = 0.0
    buy_hold_pnl: float = 0.0
    buy_hold_pnl_pct: float = 0.0


class MetricsCalculator:
    def compute(self, trades, daily_stats, start_cash, base_shares,
                start_price, end_price):
        stats = SimulationStats()

        stats.total_days = len(daily_stats)
        stats.trading_days = sum(1 for d in daily_stats if d["num_trades"] > 0)

        strategy_trades = [t for t in trades if t.tag != "closeout"]
        stats.total_trades = len(strategy_trades)
        stats.buy_count = sum(1 for t in strategy_trades if t.side == "BUY")
        stats.sell_count = sum(1 for t in strategy_trades if t.side == "SELL")

        daily_pnls = [d["daily_pnl"] for d in daily_stats]
        stats.total_pnl = sum(daily_pnls)
        initial_value = start_cash + base_shares * start_price
        if initial_value > 0:
            stats.total_pnl_pct = stats.total_pnl / initial_value

        stats.avg_daily_pnl = np.mean(daily_pnls) if daily_pnls else 0
        stats.best_day_pnl = max(daily_pnls) if daily_pnls else 0
        stats.worst_day_pnl = min(daily_pnls) if daily_pnls else 0

        stats.winning_days = sum(1 for p in daily_pnls if p > 0)
        stats.losing_days = sum(1 for p in daily_pnls if p < 0)
        if (stats.winning_days + stats.losing_days) > 0:
            stats.day_win_rate = stats.winning_days / (stats.winning_days + stats.losing_days)

        # Max drawdown on cumulative P&L
        cumulative = np.cumsum(daily_pnls)
        if len(cumulative) > 0:
            peak = np.maximum.accumulate(cumulative)
            drawdowns = cumulative - peak
            stats.max_drawdown = abs(float(np.min(drawdowns)))
            # Sharpe (annualized, 244 days)
            if len(daily_pnls) > 1:
                daily_std = np.std(daily_pnls)
                if daily_std > 0:
                    stats.sharpe_ratio = (np.mean(daily_pnls) / daily_std) * np.sqrt(244)

            # Profit factor
            gross_profit = sum(p for p in daily_pnls if p > 0)
            gross_loss = abs(sum(p for p in daily_pnls if p < 0))
            if gross_loss > 0:
                stats.profit_factor = gross_profit / gross_loss
            elif gross_profit > 0:
                stats.profit_factor = float("inf")

        # Avg trade P&L
        if stats.total_trades > 0:
            stats.avg_trade_pnl = stats.total_pnl / stats.total_trades

        # Total commission (all trades including closeouts)
        stats.total_commission = sum(
            d["total_commission"] for d in daily_stats)

        # Buy-and-hold benchmark
        if start_price > 0 and end_price > 0:
            stats.buy_hold_pnl = base_shares * (end_price - start_price)
            stats.buy_hold_pnl_pct = (end_price - start_price) / start_price

        return stats
