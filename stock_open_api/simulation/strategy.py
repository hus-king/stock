# -*- coding: utf-8 -*-
"""
@File    : strategy.py
@Date    : 2026-06-01
"""
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from .order import Signal


class BaseStrategy(ABC):
    """Pluggable intraday T-trading strategy.

    Lifecycle:
      0. prepare(data) — pre-compute indicators over full dataset
      1. on_day_start(bars_df) — initialize day state
      2. on_bar(bar, index, account) -> Optional[Signal]
      3. on_fill(side, quantity, fill_price) — called after each fill
      4. on_day_end() — cleanup
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def on_bar(self, bar, index, account) -> Optional[Signal]:
        ...

    def prepare(self, data):
        pass

    def on_day_start(self, bars_df):
        pass

    def on_fill(self, side: str, quantity: int, fill_price: float):
        pass

    def on_day_end(self):
        pass


# ============================================================
# Grid Strategy (网格交易)
# ============================================================

class GridStrategy(BaseStrategy):
    """Grid trading strategy for intraday T-trading.

    Enhancements:
    - Dynamic lot_size: scales with morning amplitude (high volatility → larger lots)
    - Intraday adaptive grid center: re-anchors buy levels to VWAP every N bars,
      so the grid follows the intraday price center rather than staying fixed at open.

    Config options:
      trend_filter_enabled (bool)
      trend_ma_period (int)
      max_daily_net_buy (int)
      daily_loss_limit (float)
      dynamic_lot_size (bool): enable dynamic lot sizing, default True
      lot_size_min (int): minimum lot size, default lot_size // 2
      lot_size_max (int): maximum lot size, default lot_size * 3
      amp_low (float): amplitude below which lot_size = lot_size_min, default 0.005
      amp_high (float): amplitude above which lot_size = lot_size_max, default 0.025
      adaptive_grid (bool): enable intraday grid re-anchoring, default True
      adaptive_interval (int): re-anchor every N bars, default 12 (60 min)
      adaptive_start (int): start re-anchoring after N bars, default 12
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.grid_spacing  = float(self.config.get("grid_spacing", 0.005))
        self.grid_levels   = int(self.config.get("grid_levels", 5))
        self.lot_size      = int(self.config.get("lot_size", 100))
        self.base_price_source = self.config.get("base_price_source", "open")

        self.trend_filter_enabled = bool(self.config.get("trend_filter_enabled", True))
        self.trend_ma_period      = int(self.config.get("trend_ma_period", 20))
        self.max_daily_net_buy    = int(self.config.get("max_daily_net_buy", 3 * self.lot_size))
        self.daily_loss_limit     = float(self.config.get("daily_loss_limit", -1000))
        self.ml_selector          = self.config.get("ml_selector", None)

        # Dynamic lot size
        self.dynamic_lot_size = bool(self.config.get("dynamic_lot_size", True))
        self.lot_size_min     = int(self.config.get("lot_size_min", max(100, self.lot_size // 2)))
        self.lot_size_max     = int(self.config.get("lot_size_max", self.lot_size * 3))
        self.amp_low          = float(self.config.get("amp_low",  0.005))
        self.amp_high         = float(self.config.get("amp_high", 0.025))

        # Intraday adaptive grid
        self.adaptive_grid     = bool(self.config.get("adaptive_grid", True))
        self.adaptive_interval = int(self.config.get("adaptive_interval", 12))
        self.adaptive_start    = int(self.config.get("adaptive_start", 12))

        self._trend_bearish = {}

        # Per-day state
        self._base_price       = None
        self._active_lot_size  = self.lot_size
        self._buy_levels       = {}
        self._pending_sells    = []
        self._pending_buys     = {}
        self._skip_day         = False
        self._daily_net_bought = 0
        self._day_trade_cash   = 0.0
        self._bars_df          = None
        self._ml_checked       = False
        self._last_anchor_idx  = 0  # last bar index when grid was re-anchored
        self._intraday_prices  = []  # for VWAP calculation
        self._intraday_vols    = []

    # ------------------------------------------------------------------
    # Pre-compute trend signals from full dataset
    # ------------------------------------------------------------------

    def prepare(self, data):
        if not self.trend_filter_enabled:
            return

        if "date" not in data.columns:
            data = data.copy()
            data["date"] = data.index.date

        dates = sorted(data["date"].unique())
        daily_closes = {}
        for d in dates:
            day_bars = data[data["date"] == d]
            daily_closes[d] = float(day_bars.iloc[-1]["close"])

        closes_s = pd.Series(daily_closes)
        ma = closes_s.rolling(window=self.trend_ma_period).mean()

        # Bearish signal for date D: previous day's close < previous day's MA
        prev_close = closes_s.shift(1)
        prev_ma = ma.shift(1)
        bearish = prev_close < prev_ma

        self._trend_bearish = {}
        for d in dates:
            if pd.notna(bearish[d]):
                self._trend_bearish[str(d)] = bool(bearish[d])

    # ------------------------------------------------------------------
    # Day lifecycle
    # ------------------------------------------------------------------

    def on_day_start(self, bars_df):
        first_bar = bars_df.iloc[0]
        day_date = str(first_bar.get("date", bars_df.index[0].date()))
        self._bars_df = bars_df

        if self.base_price_source == "open":
            self._base_price = float(first_bar["open"])
        elif self.base_price_source == "prev_close":
            self._base_price = float(first_bar["close"])
        else:
            self._base_price = float(first_bar["open"])

        if self._base_price <= 0 or np.isnan(self._base_price):
            self._base_price = float(first_bar["close"])
        if self._base_price <= 0 or np.isnan(self._base_price):
            self._base_price = 10.0

        self._buy_levels.clear()
        self._pending_buys.clear()
        self._skip_day         = False
        self._daily_net_bought = 0
        self._day_trade_cash   = 0.0
        self._ml_checked       = False
        self._last_anchor_idx  = 0
        self._intraday_prices  = []
        self._intraday_vols    = []

        # Dynamic lot size: use morning amplitude (first 6 bars)
        if self.dynamic_lot_size and len(bars_df) >= 6:
            morning = bars_df.iloc[:6]
            amp = (float(morning["high"].max()) - float(morning["low"].min())) / self._base_price
            # Linear interpolation between [amp_low, amp_high] → [lot_size_min, lot_size_max]
            t = (amp - self.amp_low) / max(self.amp_high - self.amp_low, 1e-8)
            t = max(0.0, min(1.0, t))
            raw_lot = self.lot_size_min + t * (self.lot_size_max - self.lot_size_min)
            # Round to nearest 100
            self._active_lot_size = max(100, int(round(raw_lot / 100) * 100))
        else:
            self._active_lot_size = self.lot_size

        if self.trend_filter_enabled:
            if self._trend_bearish.get(day_date, False):
                self._skip_day = True
                return

        self._rebuild_buy_levels(self._base_price)

    def _rebuild_buy_levels(self, anchor_price):
        """重建以 anchor_price 为中心的买入网格。"""
        self._buy_levels.clear()
        self._pending_buys.clear()
        for k in range(1, self.grid_levels + 1):
            buy_price = round(anchor_price * (1 - self.grid_spacing * k), 2)
            self._buy_levels[k] = buy_price
            self._pending_buys[k] = self._active_lot_size

    def on_bar(self, bar, index, account):
        if self._skip_day:
            return None

        bar_high  = float(bar["high"])
        bar_low   = float(bar["low"])
        bar_close = float(bar["close"])
        bar_vol   = float(bar.get("volume", 1)) or 1.0

        # Accumulate intraday price/volume for VWAP
        self._intraday_prices.append(bar_close)
        self._intraday_vols.append(bar_vol)

        # ML selector: evaluate after first 6 bars (30 minutes)
        if self.ml_selector is not None and not self._ml_checked and index >= 6:
            self._ml_checked = True
            if hasattr(self, "_bars_df") and self._bars_df is not None:
                morning = self._bars_df.iloc[:6]
                if not self.ml_selector(morning, self._base_price):
                    self._skip_day = True
                    return None

        # Adaptive grid: re-anchor to VWAP every adaptive_interval bars
        if (self.adaptive_grid
                and index >= self.adaptive_start
                and (index - self._last_anchor_idx) >= self.adaptive_interval
                and len(self._intraday_prices) >= 2):
            prices = np.array(self._intraday_prices)
            vols   = np.array(self._intraday_vols)
            vwap   = float(np.sum(prices * vols) / (vols.sum() + 1e-8))
            # Only re-anchor if VWAP has moved enough to justify (> half a grid spacing)
            if abs(vwap - self._base_price) / (self._base_price + 1e-8) > self.grid_spacing * 0.5:
                self._base_price = round(vwap, 2)
                self._rebuild_buy_levels(self._base_price)
                self._last_anchor_idx = index

        # Daily loss limit
        if self._day_trade_cash < self.daily_loss_limit:
            self._skip_day = True
            return None

        # Check pending sells
        for entry in self._pending_sells:
            if entry["qty"] > 0 and bar_high >= entry["sell_price"]:
                qty = min(entry["qty"], account.available_shares)
                qty = qty - (qty % 100)
                if qty >= 100:
                    entry["qty"] -= qty
                    self._daily_net_bought -= qty
                    self._day_trade_cash += qty * entry["sell_price"]
                    return Signal(side="SELL", quantity=qty, limit_price=entry["sell_price"])

        # Check buy levels
        for k in sorted(self._buy_levels.keys()):
            buy_price = self._buy_levels[k]
            if self._pending_buys.get(k, 0) > 0 and bar_low <= buy_price:
                qty = self._pending_buys[k]
                available = self.max_daily_net_buy - self._daily_net_bought
                qty = min(qty, available)
                if qty < 100:
                    return None
                max_buy = int(account.cash / (buy_price * 1.005))
                max_buy = max_buy - (max_buy % 100)
                qty = min(qty, max_buy)
                if qty >= 100:
                    self._pending_buys[k] -= qty
                    self._daily_net_bought += qty
                    return Signal(side="BUY", quantity=qty, limit_price=buy_price)

        return None

    def on_fill(self, side: str, quantity: int, fill_price: float):
        if side == "BUY":
            self._day_trade_cash -= quantity * fill_price
            sell_price = round(fill_price * (1 + self.grid_spacing), 2)
            self._pending_sells.append({"qty": quantity, "sell_price": sell_price})
        elif side == "SELL":
            self._day_trade_cash += quantity * fill_price

    def on_day_end(self):
        self._base_price       = None
        self._active_lot_size  = self.lot_size
        self._buy_levels.clear()
        self._pending_buys.clear()
        self._skip_day         = False
        self._daily_net_bought = 0
        self._day_trade_cash   = 0.0
        self._bars_df          = None
        self._ml_checked       = False
        self._last_anchor_idx  = 0
        self._intraday_prices  = []
        self._intraday_vols    = []
        # _pending_sells intentionally kept — unfilled sell orders carry over to next day


STRATEGY_REGISTRY = {
    "grid": GridStrategy,
}
