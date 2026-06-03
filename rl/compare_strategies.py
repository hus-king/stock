#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对本地所有缓存股票批量回测三种策略并对比：
1. 买入持有（不做T）
2. 全天网格做T（无ML择日）
3. 网格做T + ML择日

用法:
    venv/bin/python rl/compare_strategies.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_VENV_PYTHON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "venv", "bin", "python")
if sys.executable != _VENV_PYTHON and os.path.exists(_VENV_PYTHON):
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from stock_open_api.simulation.data import MinuteDataLoader
from stock_open_api.simulation.engine import SimulationEngine
from stock_open_api.simulation.strategy import GridStrategy

START_DATE = "2026-04-01"
END_DATE   = "2026-06-01"
BASE_SHARES    = 5000
START_CASH     = 100_000
COMMISSION     = 0.0003
MIN_COMM       = 5.0
STAMP_TAX      = 0.001

STRATEGY_CONFIG = {
    "grid_spacing": 0.015,
    "grid_levels": 3,
    "lot_size": 300,
    "base_price_source": "open",
    "trend_filter_enabled": True,
    "trend_ma_period": 20,
    "max_daily_net_buy": 600,
    "daily_loss_limit": -1000,
}

ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ml_selector.pkl")


def load_ml_selector():
    """复用 intraday_t_trading.py 里的完整实现（含跨日特征）。"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from intraday_t_trading import load_ml_selector as _load
        sel = _load(ML_MODEL_PATH)
        return sel
    except Exception as e:
        print(f"ML模型加载失败: {e}")
        return None
        return None


def run_grid(symbol, data, ml_selector=None):
    cfg = dict(STRATEGY_CONFIG)
    if ml_selector is not None:
        cfg["ml_selector"] = ml_selector
        cfg["trend_filter_enabled"] = False  # ML 替代趋势过滤
    strategy = GridStrategy(cfg)
    engine = SimulationEngine(
        symbol=symbol, start_cash=START_CASH, base_shares=BASE_SHARES,
        commission_rate=COMMISSION, min_commission=MIN_COMM,
        stamp_tax=STAMP_TAX, fill_policy="next_open",
    )
    engine.run(data, strategy)
    stats = engine.get_stats()
    daily = engine.daily_stats_list
    trade_days = sum(1 for d in daily if d["num_trades"] > 0)
    total_comm = sum(d["total_commission"] for d in daily)
    buy_cnt  = sum(1 for t in engine.trades if t.side == "BUY"  and t.tag != "closeout")
    sell_cnt = sum(1 for t in engine.trades if t.side == "SELL" and t.tag != "closeout")
    t_pnl = stats.total_pnl - stats.buy_hold_pnl
    return {
        "t_pnl":       t_pnl,
        "total_pnl":   stats.total_pnl,
        "bah_pnl":     stats.buy_hold_pnl,
        "trade_days":  trade_days,
        "total_days":  len(daily),
        "commission":  total_comm,
        "buy_cnt":     buy_cnt,
        "sell_cnt":    sell_cnt,
        "sharpe":      stats.sharpe_ratio,
        "win_rate":    stats.day_win_rate,
        "max_dd":      stats.max_drawdown,
    }


def main():
    loader = MinuteDataLoader(cache_dir="data_cache")
    ml_selector = load_ml_selector()
    if ml_selector:
        print("ML择日模型已加载\n")
    else:
        print("未找到ML模型，将跳过策略3\n")

    files = glob.glob("data_cache/min_*.parquet")
    symbols = sorted(set(os.path.basename(f).split("_")[1] for f in files))
    print(f"共 {len(symbols)} 只股票，日期范围 {START_DATE} ~ {END_DATE}\n")

    rows = []
    for i, sym in enumerate(symbols):
        try:
            data = loader.load(sym, START_DATE, END_DATE, "5", "qfq")
        except Exception as e:
            print(f"  [{i+1}/{len(symbols)}] {sym} 数据加载失败: {e}")
            continue

        try:
            r_no  = run_grid(sym, data, ml_selector=None)
            r_ml  = run_grid(sym, data, ml_selector=ml_selector) if ml_selector else None

            # 买入持有收益直接从回测结果里取
            bah = r_no["bah_pnl"]

            rows.append({
                "symbol":         sym,
                "总天数":          r_no["total_days"],
                # 策略1：买入持有
                "持有_收益":       round(bah, 0),
                # 策略2：无ML做T
                "无ML_T净贡献":    round(r_no["t_pnl"], 0),
                "无ML_总收益":     round(r_no["total_pnl"], 0),
                "无ML_交易天":     r_no["trade_days"],
                "无ML_买卖比":     f"{r_no['buy_cnt']}/{r_no['sell_cnt']}",
                "无ML_夏普":       round(r_no["sharpe"], 2),
                # 策略3：ML做T
                "ML_T净贡献":      round(r_ml["t_pnl"], 0) if r_ml else "-",
                "ML_总收益":       round(r_ml["total_pnl"], 0) if r_ml else "-",
                "ML_交易天":       r_ml["trade_days"] if r_ml else "-",
                "ML_买卖比":       f"{r_ml['buy_cnt']}/{r_ml['sell_cnt']}" if r_ml else "-",
                "ML_夏普":         round(r_ml["sharpe"], 2) if r_ml else "-",
            })
            ml_t = round(r_ml["t_pnl"], 0) if r_ml else "N/A"
            print(f"  [{i+1:2d}/{len(symbols)}] {sym}  持有={bah:>8.0f}  无ML_T={r_no['t_pnl']:>8.0f}  ML_T={ml_t}")
        except Exception as e:
            print(f"  [{i+1}/{len(symbols)}] {sym} 回测失败: {e}")

    if not rows:
        print("无数据")
        return

    df = pd.DataFrame(rows)

    # 汇总
    numeric_cols = ["持有_收益", "无ML_T净贡献", "无ML_总收益", "ML_T净贡献", "ML_总收益"]
    totals = {}
    for c in numeric_cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        totals[c] = vals.sum()

    win_no_ml = (pd.to_numeric(df["无ML_T净贡献"], errors="coerce") > 0).sum()
    win_ml    = (pd.to_numeric(df["ML_T净贡献"],   errors="coerce") > 0).sum() if ml_selector else 0
    n = len(df)

    print("\n" + "="*70)
    print("汇总对比（全部 {} 只股票）".format(n))
    print("="*70)
    print(f"{'指标':30s} {'买入持有':>12s} {'无ML做T':>12s} {'ML做T':>12s}")
    print("-"*70)
    print(f"{'T净贡献合计(元)':30s} {'—':>12s} {totals['无ML_T净贡献']:>12,.0f} {totals['ML_T净贡献']:>12,.0f}")
    print(f"{'总收益合计(元)':30s} {totals['持有_收益']:>12,.0f} {totals['无ML_总收益']:>12,.0f} {totals['ML_总收益']:>12,.0f}")
    print(f"{'T净贡献为正股票数':30s} {'—':>12s} {win_no_ml:>12d} {win_ml:>12d}")
    print(f"{'T净贡献为正比例':30s} {'—':>12s} {win_no_ml/n:>12.1%} {win_ml/n:>12.1%}")
    avg_trade_no = df["无ML_交易天"].mean()
    avg_trade_ml = pd.to_numeric(df["ML_交易天"], errors="coerce").mean()
    print(f"{'平均交易天数':30s} {'—':>12s} {avg_trade_no:>12.1f} {avg_trade_ml:>12.1f}")
    print("="*70)

    # 保存明细
    out_path = "output/strategy_comparison.csv"
    os.makedirs("output", exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n明细已保存到 {out_path}")

    # 打印明细表
    print("\n明细（按无ML_T净贡献降序）:")
    show_cols = ["symbol", "持有_收益", "无ML_T净贡献", "ML_T净贡献", "无ML_交易天", "ML_交易天", "无ML_夏普", "ML_夏普"]
    df_show = df[show_cols].copy()
    for c in ["持有_收益", "无ML_T净贡献", "ML_T净贡献"]:
        df_show[c] = pd.to_numeric(df_show[c], errors="coerce")
    df_show = df_show.sort_values("无ML_T净贡献", ascending=False)
    print(df_show.to_string(index=False))


if __name__ == "__main__":
    main()
