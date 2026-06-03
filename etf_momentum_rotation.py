# -*- coding: utf-8 -*-
"""
ETF 动量轮动策略回测（改进版）

改进点 vs 原始版:
  1. 动量窗口 20→60 天，过滤短期噪音
  2. 均线方向过滤 — 只买 MA 向上倾斜的 ETF
  3. 换仓门槛 — 动量优势不足时不动，减少无谓换手
  4. 回撤止损 — 组合从高点回撤超过阈值时强制切债券
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import backtrader as bt
import akshare as ak
import pandas as pd


# ============================================================
# 配置参数
# ============================================================
ETF_POOL = {
    "sh510300": "沪深300ETF",
    "sh510500": "中证500ETF",
    "sz159915": "创业板ETF",
    "sh510050": "上证50ETF",
    "sh518880": "黄金ETF",
    "sh511010": "国债ETF",
}

SAFE_HAVEN = "sh511010"
MOMENTUM_WINDOW = 60          # 动量窗口（3个月）
REBALANCE_DAYS = 20           # 调仓间隔
MA_FILTER = 120               # 均线过滤（半年线）
MOMENTUM_THRESHOLD = 0.03     # 换仓门槛: 新标的的动量优势必须 > 3% 才换
DD_STOP = 0.20                # 回撤止损: 组合从高点回撤超 20% 强制切债券
START_CASH = 100_000
COMMISSION = 0.0003
START_DATE = "2015-01-01"
END_DATE = "2026-05-30"


# ============================================================
# 数据下载
# ============================================================
def download_data():
    all_data = {}
    for sym, name in ETF_POOL.items():
        print(f"下载 {name} ({sym}) ...")
        df = ak.fund_etf_hist_sina(symbol=sym)
        df = df.rename(columns={
            "date": "date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        all_data[sym] = df
    return all_data


# ============================================================
# Observer
# ============================================================
class PortfolioValue(bt.Observer):
    lines = ("value",)

    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()


# ============================================================
# V1 原始版策略
# ============================================================
class MomentumRotation(bt.Strategy):
    """原始版: 20天动量 + MA60过滤，无门槛，无止损"""
    params = dict(
        pool=[],
        safe_haven=SAFE_HAVEN,
        momentum_window=20,
        rebalance_days=20,
        ma_filter=60,
    )

    def __init__(self):
        self.day_count = 0
        self.current_holding = None
        self.momentum = {}
        self.ma = {}
        for d in self.datas:
            name = d._name
            self.momentum[name] = bt.indicators.ROC(d.close, period=self.p.momentum_window)
            self.ma[name] = bt.indicators.SMA(d.close, period=self.p.ma_filter)

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f"  {dt} — {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            side = "BUY " if order.isbuy() else "SELL"
            self.log(f"{side} {order.data._name}  size={order.executed.size:.0f}  "
                     f"price={order.executed.price:.3f}  "
                     f"value={order.executed.value:.0f}  "
                     f"comm={order.executed.comm:.1f}")

    def next(self):
        self.day_count += 1
        if self.day_count % self.p.rebalance_days != 0:
            return

        min_len = min(len(d) for d in self.datas)
        if min_len < self.p.ma_filter:
            return

        momentum_scores = {}
        for d in self.datas:
            name = d._name
            if name == self.p.safe_haven:
                continue
            mom = self.momentum[name][0]
            above_ma = d.close[0] > self.ma[name][0]
            momentum_scores[name] = (mom, above_ma)

        active = {k: v for k, v in momentum_scores.items() if v[1]}
        if active:
            target = max(active, key=lambda k: active[k][0])
        else:
            target = self.p.safe_haven

        if target == self.current_holding:
            return

        if self.current_holding:
            self.close(data=self.getdatabyname(self.current_holding))
        target_data = self.getdatabyname(target)
        self.order_target_percent(data=target_data, target=0.98)
        self.current_holding = target


# ============================================================
# V2 改进版策略
# ============================================================
class MomentumRotationV2(bt.Strategy):
    params = dict(
        pool=[],
        safe_haven=SAFE_HAVEN,
        momentum_window=MOMENTUM_WINDOW,
        rebalance_days=REBALANCE_DAYS,
        ma_filter=MA_FILTER,
        momentum_threshold=MOMENTUM_THRESHOLD,
        dd_stop=DD_STOP,
    )

    def __init__(self):
        self.day_count = 0
        self.current_holding = None
        self.entry_value = 0  # 当前持仓周期的入场净值，用于止损失

        self.momentum = {}
        self.ma = {}
        for d in self.datas:
            name = d._name
            self.momentum[name] = bt.indicators.ROC(d.close, period=self.p.momentum_window)
            self.ma[name] = bt.indicators.SMA(d.close, period=self.p.ma_filter)

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f"  {dt} — {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            side = "BUY " if order.isbuy() else "SELL"
            self.log(f"{side} {order.data._name}  size={order.executed.size:.0f}  "
                     f"price={order.executed.price:.3f}  "
                     f"value={order.executed.value:.0f}  "
                     f"comm={order.executed.comm:.1f}")

    def ma_slope(self, name):
        """MA 斜率: 当前 MA vs 5天前 MA，正=向上趋势"""
        if len(self) < self.p.ma_filter + 5:
            return 0
        return self.ma[name][0] - self.ma[name][-5]

    def next(self):
        self.day_count += 1
        current_value = self.broker.getvalue()

        if self.day_count % self.p.rebalance_days != 0:
            return

        min_len = min(len(d) for d in self.datas)
        if min_len < self.p.ma_filter + 5:
            return

        # ---- 回撤止损: 从本次入场净值算，非全局历史峰值 ----
        if self.current_holding and self.current_holding != self.p.safe_haven and self.entry_value > 0:
            dd_from_entry = (self.entry_value - current_value) / self.entry_value
            if dd_from_entry > self.p.dd_stop:
                self.log(f"触发回撤止损 (持仓回撤={dd_from_entry:.1%})")
                self.close(data=self.getdatabyname(self.current_holding))
                target_data = self.getdatabyname(self.p.safe_haven)
                self.order_target_percent(data=target_data, target=0.98)
                self.current_holding = self.p.safe_haven
                self.entry_value = 0
                return

        # ---- 计算每个标的的信号 ----
        momentum_scores = {}
        for d in self.datas:
            name = d._name
            if name == self.p.safe_haven:
                continue
            mom = self.momentum[name][0]
            # 双重过滤: 价格在 MA 上方 + MA 向上倾斜
            above_ma = d.close[0] > self.ma[name][0]
            trend_up = self.ma_slope(name) > 0
            if above_ma and trend_up:
                momentum_scores[name] = mom

        # ---- 选最强标的 ----
        if momentum_scores:
            target = max(momentum_scores, key=momentum_scores.get)
        else:
            target = self.p.safe_haven

        # ---- 换仓门槛: 盯住当前持仓，优势不够大不换 ----
        if self.current_holding and target != self.current_holding:
            # 如果目标是债券，直接切（风险规避优先）
            if target == self.p.safe_haven:
                pass  # 直接走下面的切换逻辑
            else:
                # 新标的动量 vs 当前持有标的动量
                current_mom = self.momentum[self.current_holding][0]
                target_mom = self.momentum[target][0]
                if target_mom - current_mom < self.p.momentum_threshold:
                    return  # 优势不够，不动

        # ---- 执行切换 ----
        if target == self.current_holding:
            return

        if self.current_holding:
            self.close(data=self.getdatabyname(self.current_holding))
        target_data = self.getdatabyname(target)
        self.order_target_percent(data=target_data, target=0.98)
        self.current_holding = target
        # 记录本次入场净值，用于后续止损
        if target != self.p.safe_haven:
            self.entry_value = self.broker.getvalue()


# ============================================================
# 工具函数
# ============================================================
def calc_max_drawdown(values):
    peak = values[0]
    max_dd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def calc_sharpe(daily_returns, risk_free=0.025):
    if len(daily_returns) == 0 or daily_returns.std() == 0:
        return 0
    excess = daily_returns.mean() * 252 - risk_free
    vol = daily_returns.std() * (252 ** 0.5)
    return excess / vol


def run_backtest(strategy_cls, label, all_data, port_values_out=None):
    """运行一次回测，返回关键指标"""
    cerebro = bt.Cerebro()

    for sym, df in all_data.items():
        data = bt.feeds.PandasData(
            dataname=df, name=sym,
            fromdate=pd.Timestamp(START_DATE), todate=pd.Timestamp(END_DATE),
        )
        cerebro.adddata(data)

    pool = [s for s in ETF_POOL if s != SAFE_HAVEN]
    cerebro.addstrategy(strategy_cls, pool=pool)
    cerebro.addobserver(PortfolioValue)
    cerebro.broker.setcash(START_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annual_return")

    results = cerebro.run()
    strat = results[0]

    # 提取净值
    port_value = None
    for obs in strat.getobservers():
        if hasattr(obs.lines, "value"):
            n = len(obs)
            if n > 0:
                values = [obs.lines.value[-i] for i in range(n - 1, -1, -1)]
                port_value = pd.Series(values)
            break

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - START_CASH) / START_CASH

    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get("total", {}).get("total", 0)
    won = trade_analysis.get("won", {}).get("total", 0)
    lost = trade_analysis.get("lost", {}).get("total", 0)
    win_rate = won / total_trades if total_trades > 0 else 0

    daily_ret = port_value.pct_change().dropna() if port_value is not None and len(port_value) > 1 else None
    sharpe = calc_sharpe(daily_ret) if daily_ret is not None else 0
    max_dd = calc_max_drawdown(port_value.values) if port_value is not None else 0

    bh_first = all_data["sh510300"].loc[START_DATE:END_DATE]
    bh_return = (bh_first["close"].iloc[-1] / bh_first["close"].iloc[0]) - 1 if len(bh_first) > 0 else 0

    if port_values_out is not None and port_value is not None:
        port_values_out[label] = port_value

    return {
        "label": label,
        "final_value": final_value,
        "total_return": total_return,
        "total_trades": total_trades,
        "won": won,
        "lost": lost,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "bh_return": bh_return,
    }


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    all_data = download_data()

    # ---- 同时跑两版策略对比 ----
    port_values = {}

    print(f"\n{'='*60}")
    print("【V1 原始版】动量20天 / 均线MA60 / 无门槛 / 无止损")
    print(f"{'='*60}\n")
    r1 = run_backtest(MomentumRotation, "V1 (original)", all_data, port_values)

    print(f"\n{'='*60}")
    print("【V2 改进版】动量60天 / 均线MA120 / 趋势方向过滤 / 3%门槛 / 15%止损")
    print(f"{'='*60}\n")
    r2 = run_backtest(MomentumRotationV2, "V2 (improved)", all_data, port_values)

    # ---- 对比表 ----
    print(f"\n{'='*70}")
    print(f"{'指标':<20} {'V1 原始版':>18} {'V2 改进版':>18} {'买入持有':>18}")
    print(f"{'='*70}")
    for key, label in [
        ("total_return", "总收益率"),
        ("bh_return", "买入持有收益"),
        ("sharpe", "夏普比率"),
        ("max_dd", "最大回撤"),
        ("total_trades", "交易次数"),
        ("win_rate", "胜率"),
    ]:
        v1_val = r1.get(key, 0)
        v2_val = r2.get(key, 0)
        bh_val = r1.get("bh_return", 0)
        if key == "total_return":
            print(f"{label:<20} {v1_val:>18.2%} {v2_val:>18.2%} {bh_val:>18.2%}")
        elif key == "bh_return":
            continue
        elif key == "win_rate":
            print(f"{label:<20} {v1_val:>18.1%} {v2_val:>18.1%} {'—':>18}")
        elif key == "total_trades":
            print(f"{label:<20} {v1_val:>18.0f} {v2_val:>18.0f} {'—':>18}")
        else:
            print(f"{label:<20} {v1_val:>18.2f} {v2_val:>18.2f} {'—':>18}")

    # ---- 画图（两版叠加对比） ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    colors = {"V1 (original)": "lightcoral", "V2 (improved)": "steelblue"}

    ax1 = axes[0]
    for label, pv in port_values.items():
        ax1.plot(pv.index, pv.values / START_CASH, label=label,
                 linewidth=1.2, color=colors.get(label))
    bh_values = all_data["sh510300"]["close"].loc[START_DATE:END_DATE]
    bh_values = bh_values / bh_values.iloc[0]
    ax1.plot(bh_values.index, bh_values.values, label="Buy&Hold 510300",
             linewidth=0.8, alpha=0.5, color="gray")
    ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left")
    ax1.set_title("ETF Momentum Rotation — V1 vs V2")
    ax1.set_ylabel("Net Value")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for label, pv in port_values.items():
        peak = pv.cummax()
        dd = (pv - peak) / peak
        ax2.plot(dd.index, dd.values, label=label,
                 linewidth=0.8, color=colors.get(label))
    ax2.legend(loc="lower left")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "etf_momentum_result.png")
    plt.savefig(output_path, dpi=150)
    print(f"\n图表已保存: {output_path}")
