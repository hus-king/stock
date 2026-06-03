#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日内做T本地模拟回测

基于5分钟K线数据，模拟日内做T策略（默认：网格交易）的收益表现。
支持手动指定股票代码或自动筛选适合做T的标的。

用法:
    python intraday_t_trading.py
    python intraday_t_trading.py --symbol 000001 --strategy grid
    python intraday_t_trading.py --symbol 600519 --start 2024-06-01 --end 2025-06-01

注意：需要使用项目 venv 中的 Python:
    venv/bin/python intraday_t_trading.py
"""
import argparse
import os
import sys

# Redirect to venv Python if running from system Python
_VENV_PYTHON = os.path.join(os.path.dirname(__file__), "venv", "bin", "python")
if sys.executable != _VENV_PYTHON and os.path.exists(_VENV_PYTHON):
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from stock_open_api.simulation.data import MinuteDataLoader, get_stock_pool_from_spot, get_stock_name
from stock_open_api.simulation.engine import SimulationEngine
from stock_open_api.simulation.strategy import STRATEGY_REGISTRY
from stock_open_api.simulation.report import ReportGenerator
from stock_open_api.simulation.visualize import TradingChart


# ============================================================
# ML 择日
# ============================================================
DEFAULT_ML_MODEL = os.path.join(os.path.dirname(__file__), "rl", "models", "ml_selector.pkl")

def load_ml_selector(model_path=None):
    """加载 ML 择日模型，返回 selector 函数或 None。

    selector 内部维护滑动窗口缓存历史 bars，自动计算跨日特征。
    接口：selector(morning_bars, open_price) -> bool
    """
    path = model_path or DEFAULT_ML_MODEL
    if not os.path.exists(path):
        return None
    try:
        import joblib
        import numpy as np
        bundle = joblib.load(path)
        model     = bundle["model"]
        scaler    = bundle["scaler"]
        feat_cols = bundle["features"]  # 18维特征名列表

        # 滑动窗口：保存最近5天的日线数据（每天全量bars）
        history_days = []  # list of day_bars DataFrame

        def selector(morning_bars, open_price, prev_days=None):
            """
            Args:
                morning_bars: 当天前6根K线
                open_price:   当天开盘价
                prev_days:    可选，前N天的bars列表（从远到近），用于跨日特征
            """
            nonlocal history_days
            if len(morning_bars) < 6 or open_price <= 0:
                return True

            closes = morning_bars["close"].values.astype(float)
            vols   = morning_bars["volume"].values.astype(float)
            high   = float(morning_bars["high"].max())
            low    = float(morning_bars["low"].min())
            close  = float(morning_bars.iloc[-1]["close"])
            x      = np.arange(len(closes))
            slope  = np.polyfit(x, closes, 1)[0] / open_price if len(closes) > 1 else 0.0
            rets   = np.diff(closes) / (closes[:-1] + 1e-8)
            vol_mean_per_bar = float(morning_bars["volume"].mean()) + 1e-8
            vwap   = np.sum(closes * vols) / (vols.sum() + 1e-8)

            feat_dict = {
                "m_amplitude":  (high - low) / open_price,
                "m_return":     (close - open_price) / open_price,
                "m_range":      (high - low) / (close + 1e-8),
                "m_vol_ratio":  float(vols.sum()) / (vol_mean_per_bar * 6),
                "m_slope":      slope,
                "m_volatility": float(np.std(rets)) if len(rets) > 0 else 0.0,
                "m_first_gap":  (float(morning_bars.iloc[0]["close"]) - open_price) / open_price,
                "m_high_pos":   morning_bars["high"].values.argmax() / 5.0,
                "m_low_pos":    morning_bars["low"].values.argmin() / 5.0,
                "m_vwap_dev":   (vwap - open_price) / open_price,
            }

            # 跨日特征：使用传入的 prev_days 或内部缓存
            prev_list = prev_days if prev_days is not None else history_days
            if prev_list:
                prev = prev_list[-1]
                p_open  = float(prev.iloc[0]["open"])
                p_close = float(prev.iloc[-1]["close"])
                p_high  = float(prev["high"].max())
                p_low   = float(prev["low"].min())
                feat_dict.update({
                    "prev_amplitude": (p_high - p_low) / (p_open + 1e-8),
                    "prev_return":    (p_close - p_open) / (p_open + 1e-8),
                    "prev_vol_norm":  float(prev["volume"].mean()) / (vol_mean_per_bar + 1e-8),
                    "gap_vs_prev":    (open_price - p_close) / (p_close + 1e-8),
                })
            else:
                feat_dict.update({"prev_amplitude": 0.0, "prev_return": 0.0,
                                   "prev_vol_norm": 1.0,  "gap_vs_prev": 0.0})

            n_hist = min(5, len(prev_list))
            if n_hist >= 2:
                hist = prev_list[-n_hist:]
                amps  = [(float(p["high"].max()) - float(p["low"].min())) /
                         (float(p.iloc[0]["open"]) + 1e-8) for p in hist]
                rets_h = [(float(p.iloc[-1]["close"]) - float(p.iloc[0]["open"])) /
                          (float(p.iloc[0]["open"]) + 1e-8) for p in hist]
                hvols = [float(p["volume"].mean()) for p in hist]
                feat_dict.update({
                    "hist_amp_mean":  float(np.mean(amps)),
                    "hist_amp_std":   float(np.std(amps)),
                    "hist_ret_mean":  float(np.mean(rets_h)),
                    "hist_vol_trend": float(np.polyfit(range(len(hvols)), hvols, 1)[0])
                                     / (np.mean(hvols) + 1e-8),
                })
            else:
                feat_dict.update({"hist_amp_mean": feat_dict["m_amplitude"],
                                   "hist_amp_std": 0.0, "hist_ret_mean": 0.0,
                                   "hist_vol_trend": 0.0})

            feats = np.array([[feat_dict.get(c, 0.0) for c in feat_cols]])
            pred  = model.predict(scaler.transform(feats))[0]
            return bool(pred == 1)

        def update_history(day_bars):
            """每天结束后调用，更新历史缓存。"""
            history_days.append(day_bars)
            if len(history_days) > 5:
                history_days.pop(0)

        selector.update_history = update_history
        return selector
    except Exception as e:
        print("ML模型加载失败: {}，将不使用ML择日".format(e))
        return None


# ============================================================
# 默认配置参数
# ============================================================
STOCK_SYMBOL = "000001"          # 股票代码，None 表示自动筛选
STRATEGY_NAME = "grid"            # 策略名称: grid
STRATEGY_CONFIG = {
    "grid_spacing": 0.015,        # 网格间距 1.5%
    "grid_levels": 3,             # 上下各3档
    "lot_size": 300,              # 每档300股
    "base_price_source": "open",  # 以开盘价为网格中心
    "trend_filter_enabled": True, # 趋势过滤：下跌趋势时跳过
    "trend_ma_period": 20,        # 趋势判断MA周期
    "max_daily_net_buy": 600,     # 每日最大净买入股数
}

START_DATE = "2026-01-01"
END_DATE = "2026-05-30"
PERIOD = "5"                      # 5分钟K线
ADJUST = "qfq"                    # 前复权

START_CASH = 100_000              # 初始现金
BASE_SHARES = 5000                # 底仓（已持有股数）
COMMISSION_RATE = 0.0003          # 佣金万三
MIN_COMMISSION = 5.0              # 最低佣金
STAMP_TAX = 0.001                 # 印花税 0.1%（仅卖出）

FILL_POLICY = "next_open"         # 成交方式: next_open / this_close

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "intraday_t")
OUTPUT_CHART = os.path.join(OUTPUT_DIR, "intraday_t_{symbol}_result.png")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "intraday_t_{symbol}_daily.csv")


# ============================================================
# 命令行参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="日内做T本地模拟回测")
    parser.add_argument("--symbol", default=STOCK_SYMBOL,
                        help="股票代码 (默认: {})".format(STOCK_SYMBOL))
    parser.add_argument("--strategy", default=STRATEGY_NAME,
                        choices=list(STRATEGY_REGISTRY.keys()),
                        help="策略名称 (默认: {})".format(STRATEGY_NAME))
    parser.add_argument("--start", default=START_DATE, help="开始日期")
    parser.add_argument("--end", default=END_DATE, help="结束日期")
    parser.add_argument("--period", default=PERIOD, help="K线周期(分钟)")
    parser.add_argument("--base-shares", type=int, default=BASE_SHARES,
                        help="底仓股数 (默认: {})".format(BASE_SHARES))
    parser.add_argument("--cash", type=float, default=START_CASH,
                        help="初始现金 (默认: {})".format(START_CASH))
    parser.add_argument("--grid-spacing", type=float,
                        default=STRATEGY_CONFIG["grid_spacing"],
                        help="网格间距 (默认: {:.1%})".format(STRATEGY_CONFIG["grid_spacing"]))
    parser.add_argument("--grid-levels", type=int,
                        default=STRATEGY_CONFIG["grid_levels"],
                        help="网格档数 (默认: {})".format(STRATEGY_CONFIG["grid_levels"]))
    parser.add_argument("--lot-size", type=int,
                        default=STRATEGY_CONFIG["lot_size"],
                        help="每档股数 (默认: {})".format(STRATEGY_CONFIG["lot_size"]))
    parser.add_argument("--fill", default=FILL_POLICY,
                        choices=["next_open", "this_close"],
                        help="成交方式 (默认: {})".format(FILL_POLICY))
    parser.add_argument("--no-trend-filter", action="store_true",
                        help="关闭趋势过滤")
    parser.add_argument("--trend-ma", type=int,
                        default=STRATEGY_CONFIG["trend_ma_period"],
                        help="趋势MA周期 (默认: {})".format(
                            STRATEGY_CONFIG["trend_ma_period"]))
    parser.add_argument("--max-net-buy", type=int,
                        default=STRATEGY_CONFIG["max_daily_net_buy"],
                        help="每日最大净买入股数 (默认: {})".format(
                            STRATEGY_CONFIG["max_daily_net_buy"]))
    parser.add_argument("--daily-loss-limit", type=float, default=-1000,
                        help="日内止损上限，元（负数，默认: -1000）")
    parser.add_argument("--ml-selector", default=None,
                        help="ML择日模型路径（默认: rl/models/ml_selector.pkl）, "
                             "设为 'auto' 自动使用默认路径, 'none' 禁用")
    parser.add_argument("--chart", default=None, help="图表输出路径")
    parser.add_argument("--csv", default=None, help="CSV输出路径")
    parser.add_argument("--screen", action="store_true",
                        help="自动筛选股票池（忽略 --symbol）")
    return parser.parse_args()


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()

    # 1. 确定股票
    if args.screen:
        print("正在筛选适合做T的股票...")
        pool = get_stock_pool_from_spot()
        if not pool:
            print("未筛选到符合条件的股票")
            return
        print("筛选到 {} 只股票: {}".format(len(pool), pool[:10]))
        symbol = pool[0]  # 默认取第一只
        print("使用: {}\n".format(symbol))
    else:
        symbol = args.symbol

    strategy_config = {
        "grid_spacing": args.grid_spacing,
        "grid_levels": args.grid_levels,
        "lot_size": args.lot_size,
        "base_price_source": "open",
        "trend_filter_enabled": not args.no_trend_filter,
        "trend_ma_period": args.trend_ma,
        "max_daily_net_buy": args.max_net_buy,
        "daily_loss_limit": args.daily_loss_limit,
    }

    # 加载 ML 择日模型
    ml_model_arg = args.ml_selector
    if ml_model_arg == "none":
        ml_selector = None
        print("ML择日: 已禁用")
    elif ml_model_arg == "auto" or ml_model_arg is None:
        ml_selector = load_ml_selector()
        if ml_selector:
            print("ML择日: 已启用 ({})".format(DEFAULT_ML_MODEL))
            # ML 替代趋势过滤，避免双重过滤
            strategy_config["trend_filter_enabled"] = False
        else:
            print("ML择日: 模型文件不存在，已跳过")
    else:
        ml_selector = load_ml_selector(ml_model_arg)
        if ml_selector:
            print("ML择日: 已启用 ({})".format(ml_model_arg))

    if ml_selector is not None:
        strategy_config["ml_selector"] = ml_selector

    # 解析输出路径（含股票代码）
    chart_path = args.chart or OUTPUT_CHART.format(symbol=symbol)
    csv_path = args.csv or OUTPUT_CSV.format(symbol=symbol)
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)

    # 2. 下载数据
    print("正在下载 {} {}~{} {}分钟K线数据...".format(
        symbol, args.start, args.end, args.period))
    loader = MinuteDataLoader()
    data = loader.load(symbol, args.start, args.end, args.period, ADJUST)
    print("数据量: {} 条\n".format(len(data)))

    # 3. 创建策略
    # 获取股票名称
    stock_name = get_stock_name(symbol)

    strategy_cls = STRATEGY_REGISTRY[args.strategy]
    strategy = strategy_cls(strategy_config)
    parts = [
        "{} ({}) 策略: {} (网格间距={:.1%}, 档数={}, 每档={}股)".format(
            stock_name, symbol, args.strategy,
            args.grid_spacing, args.grid_levels, args.lot_size),
    ]
    if strategy_config["trend_filter_enabled"]:
        parts.append(
            "趋势过滤=MA{}".format(strategy_config["trend_ma_period"]))
    parts.append("日净买上限={}股".format(strategy_config["max_daily_net_buy"]))
    print(", ".join(parts))

    # 4. 运行模拟
    print("初始现金: {:,.0f}, 底仓: {}股, 佣金: {:.1%}\n".format(
        args.cash, args.base_shares, COMMISSION_RATE))
    engine = SimulationEngine(
        symbol=symbol,
        start_cash=args.cash,
        base_shares=args.base_shares,
        commission_rate=COMMISSION_RATE,
        min_commission=MIN_COMMISSION,
        stamp_tax=STAMP_TAX,
        fill_policy=args.fill,
    )
    engine.run(data, strategy)

    # 5. 报表
    stats = engine.get_stats()
    ReportGenerator.print_summary(stats)
    ReportGenerator.print_daily_table(engine.daily_stats_list, tail=20)

    # 6. 图表
    chart = TradingChart()
    chart.plot(stats, engine.daily_stats_list, engine.trades, data, chart_path)

    # 7. 导出
    ReportGenerator.export_csv(engine.daily_stats_list, csv_path, engine.trades)


if __name__ == "__main__":
    main()
