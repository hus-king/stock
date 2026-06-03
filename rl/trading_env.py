#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gymnasium 交易环境，用于强化学习做T策略训练。

设计思路：
- 每个 episode = 一只股票的一个交易日
- State: 当前 bar 的技术特征（价格、成交量、均线、波动率等）
- Action: 0=持有, 1=买入一手(lot_size股), 2=卖出一手
- Reward: T交易净现金流（卖出收入 - 买入成本），收盘时对未平仓超额仓位按收盘价结算
- 约束: 不能卖出超过超额仓位（position - base_shares），不能买入超过现金上限和日净买上限
"""
import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

LOT_SIZE = 100
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX = 0.001


class TradingEnv(gym.Env):
    """单日做T强化学习环境。

    每个 episode 随机选取一只股票的一个交易日。
    Agent 在每根5分钟K线上决策买/卖/持有。
    """

    metadata = {"render_modes": []}

    def __init__(self, daily_data: list, base_shares: int = 5000,
                 start_cash: float = 100_000, lot_size: int = LOT_SIZE,
                 max_daily_net_buy: int = 600):
        """
        Args:
            daily_data: list of (symbol, date, bars_df)，每项是一个交易日的数据
            base_shares: 底仓股数
            start_cash: 初始现金
            lot_size: 每手股数
            max_daily_net_buy: 每日最大净买入股数
        """
        super().__init__()
        self.daily_data = daily_data
        self.base_shares = base_shares
        self.start_cash = start_cash
        self.lot_size = lot_size
        self.max_daily_net_buy = max_daily_net_buy

        # Action: 0=持有, 1=买入一手, 2=卖出一手
        self.action_space = spaces.Discrete(3)

        # State: 18维特征向量
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        self._episode_data = None
        self._bar_idx = 0
        self._cash = start_cash
        self._position = base_shares
        self._daily_net_bought = 0
        self._day_trade_cash = 0.0
        self._open_price = 0.0
        self._prev_close = 0.0
        self._avg_buy_price = 0.0   # 超额仓位的平均买入价
        self._excess_cost = 0.0     # 超额仓位的总成本

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 随机选一个交易日
        idx = self.np_random.integers(0, len(self.daily_data))
        symbol, date, bars_df = self.daily_data[idx]

        self._episode_data = bars_df.reset_index(drop=False)
        self._bar_idx = 0
        self._cash = self.start_cash
        self._position = self.base_shares
        self._daily_net_bought = 0
        self._day_trade_cash = 0.0
        self._open_price = float(self._episode_data.iloc[0]["open"])
        self._prev_close = self._open_price
        self._avg_buy_price = 0.0
        self._excess_cost = 0.0

        obs = self._get_obs(0)
        return obs, {}

    def step(self, action):
        bars = self._episode_data
        bar = bars.iloc[self._bar_idx]
        fill_price = float(bar["close"])

        reward = 0.0
        excess_before = self._position - self.base_shares

        if action == 1:  # 买入
            qty = self._calc_buy_qty(fill_price)
            if qty >= self.lot_size:
                cost = qty * fill_price
                comm = max(cost * COMMISSION_RATE, MIN_COMMISSION)
                total_cost = cost + comm
                self._cash -= total_cost
                old_excess = self._position - self.base_shares
                self._excess_cost += total_cost
                self._position += qty
                new_excess = self._position - self.base_shares
                self._avg_buy_price = self._excess_cost / new_excess if new_excess > 0 else 0.0
                self._daily_net_bought += qty
                self._day_trade_cash -= total_cost

        elif action == 2:  # 卖出
            qty = self._calc_sell_qty()
            if qty >= self.lot_size:
                proceeds = qty * fill_price
                comm = max(proceeds * COMMISSION_RATE, MIN_COMMISSION)
                stamp = proceeds * STAMP_TAX
                net = proceeds - comm - stamp
                self._cash += net
                # 中间 reward：本次卖出的实现盈亏
                if self._avg_buy_price > 0:
                    reward = (fill_price - self._avg_buy_price) * qty - comm - stamp
                old_excess = self._position - self.base_shares
                cost_per_share = self._excess_cost / old_excess if old_excess > 0 else fill_price
                self._excess_cost -= cost_per_share * qty
                self._excess_cost = max(0.0, self._excess_cost)
                self._position -= qty
                new_excess = self._position - self.base_shares
                self._avg_buy_price = (self._excess_cost / new_excess
                                       if new_excess > 0 else 0.0)
                self._daily_net_bought -= qty
                self._day_trade_cash += net

        self._bar_idx += 1
        done = self._bar_idx >= len(bars)

        if done:
            excess = self._position - self.base_shares
            if excess > 0:
                close_price = float(bars.iloc[-1]["close"])
                proceeds = excess * close_price
                comm = max(proceeds * COMMISSION_RATE, MIN_COMMISSION)
                stamp = proceeds * STAMP_TAX
                closeout_pnl = (close_price - self._avg_buy_price) * excess - comm - stamp
                reward += closeout_pnl
                self._day_trade_cash += proceeds - comm - stamp
            # 归一化：除以开盘价，让不同价位股票的 reward 可比
            reward = reward / (self._open_price * self.lot_size + 1e-8)
            obs = np.zeros(18, dtype=np.float32)
        else:
            # unrealized PnL 变化（归一化）
            excess = self._position - self.base_shares
            if excess > 0 and self._avg_buy_price > 0:
                next_bar = bars.iloc[self._bar_idx]
                next_close = float(next_bar["close"])
                unreal_delta = (next_close - fill_price) * excess
                reward += unreal_delta / (self._open_price * self.lot_size + 1e-8)
            obs = self._get_obs(self._bar_idx)

        return obs, reward, done, False, {}

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _get_obs(self, idx: int) -> np.ndarray:
        bars = self._episode_data
        bar = bars.iloc[idx]
        n = len(bars)

        close = float(bar["close"])
        open_ = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        volume = float(bar.get("volume", 0)) or 1.0

        # 价格相对开盘价的归一化
        rel_close = (close - self._open_price) / (self._open_price + 1e-8)
        rel_open = (open_ - self._open_price) / (self._open_price + 1e-8)
        rel_high = (high - self._open_price) / (self._open_price + 1e-8)
        rel_low = (low - self._open_price) / (self._open_price + 1e-8)

        # 当根K线振幅
        bar_range = (high - low) / (close + 1e-8)

        # 时间进度 [0, 1]
        time_progress = idx / max(n - 1, 1)

        # 短期均线偏离（用过去5根收盘价）
        start = max(0, idx - 4)
        recent_closes = bars["close"].iloc[start:idx + 1].values.astype(float)
        ma5 = recent_closes.mean()
        ma5_dev = (close - ma5) / (ma5 + 1e-8)

        # 过去5根成交量均值
        recent_vols = bars["volume"].iloc[start:idx + 1].values.astype(float)
        vol_mean = recent_vols.mean() + 1.0
        vol_ratio = volume / vol_mean

        # 账户状态特征
        excess_shares = (self._position - self.base_shares) / self.lot_size  # 超额手数
        cash_ratio = self._cash / (self.start_cash + 1e-8)
        net_buy_ratio = self._daily_net_bought / (self.max_daily_net_buy + 1e-8)
        trade_pnl_ratio = self._day_trade_cash / (self._open_price * self.lot_size + 1e-8)

        # 买入/卖出是否可行
        can_buy = float(self._calc_buy_qty(close) >= self.lot_size)
        can_sell = float(self._calc_sell_qty() >= self.lot_size)

        # 距离收盘的剩余bar数（归一化）
        bars_left = (n - 1 - idx) / max(n - 1, 1)

        # 过去10根收盘价的波动率
        start10 = max(0, idx - 9)
        closes10 = bars["close"].iloc[start10:idx + 1].values.astype(float)
        volatility = closes10.std() / (closes10.mean() + 1e-8) if len(closes10) > 1 else 0.0

        obs = np.array([
            rel_close, rel_open, rel_high, rel_low,
            bar_range, time_progress, ma5_dev, vol_ratio,
            excess_shares, cash_ratio, net_buy_ratio, trade_pnl_ratio,
            can_buy, can_sell, bars_left, volatility,
            float(idx) / max(n - 1, 1),  # 绝对时间位置
            (close - self._prev_close) / (self._prev_close + 1e-8),  # 相对上根收盘涨跌
        ], dtype=np.float32)

        self._prev_close = close
        return obs

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _calc_buy_qty(self, price: float) -> int:
        available_net = self.max_daily_net_buy - self._daily_net_bought
        if available_net < self.lot_size:
            return 0
        cost_per_lot = price * self.lot_size * (1 + COMMISSION_RATE)
        max_lots = int(self._cash / cost_per_lot)
        qty = min(max_lots * self.lot_size, available_net)
        qty = qty - (qty % self.lot_size)
        return qty

    def _calc_sell_qty(self) -> int:
        excess = self._position - self.base_shares
        if excess < self.lot_size:
            return 0
        qty = excess - (excess % self.lot_size)
        return qty


# ------------------------------------------------------------------
# 数据加载工具
# ------------------------------------------------------------------

def load_daily_data(symbols: list, start_date: str, end_date: str,
                    period: str = "5", adjust: str = "qfq",
                    cache_dir: str = "data_cache") -> list:
    """加载多只股票数据，返回 list of (symbol, date, bars_df)。"""
    from stock_open_api.simulation.data import MinuteDataLoader
    loader = MinuteDataLoader(cache_dir=cache_dir)
    daily_data = []

    for sym in symbols:
        try:
            df = loader.load(sym, start_date, end_date, period, adjust)
            if "date" not in df.columns:
                df = df.copy()
                df["date"] = df.index.date
            for date, day_bars in df.groupby("date"):
                if len(day_bars) >= 10:  # 过滤数据太少的天
                    daily_data.append((sym, date, day_bars.reset_index()))
        except Exception as e:
            print("跳过 {}: {}".format(sym, e))

    print("共加载 {} 个交易日样本（来自 {} 只股票）".format(
        len(daily_data), len(symbols)))
    return daily_data
