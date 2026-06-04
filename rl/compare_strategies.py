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

import glob
import hashlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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


def load_baostock_cache(symbol, start_date, end_date, cache_dir="data_cache"):
    """直接从 baostock parquet 缓存加载5分钟数据，避免触发网络请求。"""
    for s, e in [("2024-01-01", "2026-06-01"), ("2026-01-01", "2026-06-01")]:
        key = "{}|{}|{}|5|baostock".format(symbol, s, e)
        h = hashlib.md5(key.encode()).hexdigest()[:8]
        path = os.path.join(cache_dir, "min_{}_{}.parquet".format(symbol, h))
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if "date" not in df.columns:
                df["date"] = df.index.date
            start_dt = pd.to_datetime(start_date).date()
            end_dt   = pd.to_datetime(end_date).date()
            df = df[df["date"].between(start_dt, end_dt)]
            if len(df) > 0:
                return df
    # fallback: 尝试其他缓存
    from stock_open_api.simulation.data import MinuteDataLoader
    try:
        loader = MinuteDataLoader(cache_dir=cache_dir)
        return loader.load(symbol, start_date, end_date, "5", "qfq")
    except Exception:
        return None


def load_ml_selector():
    try:
        import joblib
        bundle = joblib.load(ML_MODEL_PATH)
        model     = bundle["model"]
        scaler    = bundle["scaler"]
        feat_cols = bundle["features"]

        # 直接复用 ml_selector.py 里的 selector 逻辑
        from rl.ml_selector import extract_features

        history_days = []

        def selector(morning_bars, open_price, prev_days=None):
            import numpy as np
            prev_list = prev_days if prev_days is not None else history_days
            daily_df = None  # compare_strategies 不传日K线，用默认值
            feats = extract_features(
                morning_bars,
                prev_list,
                open_price,
                daily_df=daily_df,
                target_date=morning_bars.iloc[0].get("date", None) if len(morning_bars) > 0 else None,
            )
            if feats is None:
                return True
            import numpy as np
            f_arr = np.array([[feats.get(c, 0.0) for c in feat_cols]])
            pred = model.predict(scaler.transform(f_arr))[0]
            return bool(pred == 1)

        def update_history(day_bars):
            history_days.append(day_bars)
            if len(history_days) > 5:
                history_days.pop(0)

        selector.update_history = update_history
        return selector
    except Exception as e:
        print(f"ML模型加载失败: {e}")
        return None


def run_grid(symbol, data, ml_selector=None):
    cfg = dict(STRATEGY_CONFIG)
    if ml_selector is not None:
        cfg["ml_selector"] = ml_selector
        cfg["trend_filter_enabled"] = False
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
        data = load_baostock_cache(sym, START_DATE, END_DATE)
        if data is None or len(data) == 0:
            print(f"  [{i+1:3d}/{len(symbols)}] {sym} 数据加载失败")
            continue

        try:
            r_no = run_grid(sym, data, ml_selector=None)
            r_ml = run_grid(sym, data, ml_selector=ml_selector) if ml_selector else None
            bah  = r_no["bah_pnl"]

            rows.append({
                "symbol":       sym,
                "总天数":        r_no["total_days"],
                "持有_收益":     round(bah, 0),
                "无ML_T净贡献":  round(r_no["t_pnl"], 0),
                "无ML_总收益":   round(r_no["total_pnl"], 0),
                "无ML_交易天":   r_no["trade_days"],
                "无ML_夏普":     round(r_no["sharpe"], 2),
                "ML_T净贡献":    round(r_ml["t_pnl"], 0) if r_ml else "-",
                "ML_总收益":     round(r_ml["total_pnl"], 0) if r_ml else "-",
                "ML_交易天":     r_ml["trade_days"] if r_ml else "-",
                "ML_夏普":       round(r_ml["sharpe"], 2) if r_ml else "-",
            })
            ml_t   = f"{r_ml['t_pnl']:>8.0f}" if r_ml else "     N/A"
            ml_tot = f"{r_ml['total_pnl']:>8.0f}" if r_ml else "     N/A"
            print(f"  [{i+1:3d}/{len(symbols)}] {sym}"
                  f"  持有={bah:>8.0f}"
                  f"  无ML_T={r_no['t_pnl']:>8.0f}  无ML_总={r_no['total_pnl']:>8.0f}"
                  f"  ML_T={ml_t}  ML_总={ml_tot}")
        except Exception as e:
            print(f"  [{i+1:3d}/{len(symbols)}] {sym} 回测失败: {e}")

    if not rows:
        print("无数据")
        return

    df = pd.DataFrame(rows)

    # 数值化
    for c in ["持有_收益", "无ML_T净贡献", "无ML_总收益", "ML_T净贡献", "ML_总收益"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n = len(df)
    bah_total    = df["持有_收益"].sum()
    no_ml_t      = df["无ML_T净贡献"].sum()
    no_ml_total  = df["无ML_总收益"].sum()
    ml_t         = df["ML_T净贡献"].sum()
    ml_total     = df["ML_总收益"].sum()

    win_bah   = (df["持有_收益"]    > 0).sum()
    win_no_ml = (df["无ML_T净贡献"] > 0).sum()
    win_ml    = (df["ML_T净贡献"]   > 0).sum()
    win_no_ml_total = (df["无ML_总收益"] > 0).sum()
    win_ml_total    = (df["ML_总收益"]   > 0).sum()

    avg_trade_no = df["无ML_交易天"].mean()
    avg_trade_ml = pd.to_numeric(df["ML_交易天"], errors="coerce").mean()

    print("\n" + "="*75)
    print(f"汇总对比（{n} 只股票，{START_DATE} ~ {END_DATE}）")
    print("="*75)
    print(f"{'指标':28s} {'买入持有':>14s} {'无ML做T':>14s} {'ML择日做T':>14s}")
    print("-"*75)
    print(f"{'总收益合计(元)':28s} {bah_total:>14,.0f} {no_ml_total:>14,.0f} {ml_total:>14,.0f}")
    print(f"{'T净贡献合计(元)':28s} {'—':>14s} {no_ml_t:>14,.0f} {ml_t:>14,.0f}")
    print(f"{'总收益相对持有增益(元)':28s} {'—':>14s} {no_ml_total-bah_total:>14,.0f} {ml_total-bah_total:>14,.0f}")
    print(f"{'总收益为正股票数':28s} {win_bah:>14d} {win_no_ml_total:>14d} {win_ml_total:>14d}")
    print(f"{'总收益为正比例':28s} {win_bah/n:>14.1%} {win_no_ml_total/n:>14.1%} {win_ml_total/n:>14.1%}")
    print(f"{'T净贡献为正股票数':28s} {'—':>14s} {win_no_ml:>14d} {win_ml:>14d}")
    print(f"{'T净贡献为正比例':28s} {'—':>14s} {win_no_ml/n:>14.1%} {win_ml/n:>14.1%}")
    print(f"{'平均交易天数/只':28s} {'—':>14s} {avg_trade_no:>14.1f} {avg_trade_ml:>14.1f}")
    print("="*75)

    # 保存明细
    out_path = "output/strategy_comparison.csv"
    os.makedirs("output", exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n明细已保存到 {out_path}")

    # 打印明细表
    print("\n明细（按ML_T净贡献降序）:")
    show = ["symbol", "持有_收益", "无ML_T净贡献", "ML_T净贡献",
            "无ML_总收益", "ML_总收益", "无ML_交易天", "ML_交易天"]
    df_show = df[show].sort_values("ML_T净贡献", ascending=False)
    print(df_show.to_string(index=False))


if __name__ == "__main__":
    main()
