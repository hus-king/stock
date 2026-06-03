# -*- coding: utf-8 -*-
"""
@File    : order.py
@Date    : 2026-06-01
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Signal:
    """Strategy-generated trading signal."""
    side: str  # "BUY" or "SELL"
    quantity: int  # Shares (will be lot-rounded by engine)
    limit_price: Optional[float] = None  # None = market order


@dataclass
class Trade:
    """A filled trade record."""
    trade_id: int
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    tag: str = ""  # "closeout" for forced end-of-day trades


@dataclass
class DailyTrades:
    """Trades grouped by day."""
    date: str
    trades: list = field(default_factory=list)
