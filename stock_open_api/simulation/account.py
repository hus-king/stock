# -*- coding: utf-8 -*-
"""
@File    : account.py
@Date    : 2026-06-01
"""
from dataclasses import dataclass, field


@dataclass
class AccountState:
    """Tracks cash, position, and cost basis during simulation."""
    cash: float
    base_shares: int
    position: int = 0
    avg_cost: float = 0.0
    total_commission: float = 0.0
    realized_pnl: float = 0.0
    trade_count: int = 0
    day_start_value: float = 0.0  # set by engine at start of each day

    def __post_init__(self):
        if self.position == 0:
            self.position = self.base_shares

    @property
    def available_shares(self):
        """Shares that can be sold without going below base_shares."""
        return self.position - self.base_shares

    @property
    def market_value(self):
        return 0.0  # Computed externally with current price

    def can_sell(self, quantity, price=0):
        return quantity <= self.available_shares

    def can_buy(self, quantity, price, commission_rate, min_commission):
        cost = quantity * price
        comm = max(cost * commission_rate, min_commission)
        return cost + comm <= self.cash

    def execute_buy(self, quantity, price, commission_rate, min_commission):
        cost = quantity * price
        comm = max(cost * commission_rate, min_commission)
        total_cost = cost + comm

        old_total = self.position * self.avg_cost
        new_total = old_total + cost
        self.position += quantity
        self.avg_cost = new_total / self.position if self.position > 0 else 0
        self.cash -= total_cost
        self.total_commission += comm
        self.trade_count += 1

    def execute_sell(self, quantity, price, commission_rate, min_commission,
                     stamp_tax=0.0):
        proceeds = quantity * price
        comm = max(proceeds * commission_rate, min_commission)
        stamp = proceeds * stamp_tax
        net_proceeds = proceeds - comm - stamp

        cost_basis = self.avg_cost
        self.position -= quantity
        self.realized_pnl += quantity * (price - cost_basis) - comm - stamp
        self.cash += net_proceeds
        self.total_commission += comm
        self.trade_count += 1

        if self.position == 0:
            self.avg_cost = 0.0

    def reset_day(self):
        """Reset intraday tracking; called at end of each day."""
        pass
