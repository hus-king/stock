#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按月批量跑三策略对比（2024-01 ~ 2026-06），每月汇总买入持有 / 无ML做T / ML择日做T。
每月完成后立即追加保存，支持断点续跑。

修复：selector 传入完整日K线特征（daily_df + target_date），与训练时特征一致。
并行：每只股票作为独立任务，进程池处理。

用法:
    venv/bin/python rl/compare_monthly.py
    venv/bin/python rl/compare_monthly.py --start 2024-01 --end 2026-06
"""
import os, sys, glob, hashlib, argparse, calendar, warnings, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

import multiprocessing as mp
import pandas as pd
import numpy as np

from stock_open_api.simulation.engine import SimulationEngine
from stock_open_api.simulation.strategy import GridStrategy
from rl.ml_selector import load_daily, extract_features

ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ml_selector.pkl")
CACHE_DIR     = "data_cache"
OUTPUT_CSV    = "output/monthly_comparison.csv"

BASE_SHARES = 5000
START_CASH  = 100_000
COMMISSION  = 0.0003
MIN_COMM    = 5.0
STAMP_TAX   = 0.001

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


# ============================================================
# 工具函数
# ============================================================

def gen_months(start_ym, end_ym):
    months = []
    y, m = int(start_ym[:4]), int(start_ym[5:])
    ey, em = int(end_ym[:4]), int(end_ym[5:])
    while (y, m) <= (ey, em):
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1; y += 1
    return months


def load_minute_data(symbol, start_date, end_date):
    """优先读 baostock 缓存。"""
    for s, e in [("2024-01-01", "2026-06-01"), ("2024-01-15", "2026-06-01"),
                 ("2026-01-01", "2026-06-01")]:
        key = "{}|{}|{}|5|baostock".format(symbol, s, e)
        h = hashlib.md5(key.encode()).hexdigest()[:8]
        path = os.path.join(CACHE_DIR, "min_{}_{}.parquet".format(symbol, h))
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if "date" not in df.columns:
                df["date"] = df.index.date
            start_dt = pd.to_datetime(start_date).date()
            end_dt   = pd.to_datetime(end_date).date()
            df = df[df["date"].between(start_dt, end_dt)]
            if len(df) > 0:
                return df
    return None


def make_ml_selector(model, scaler, feat_cols, daily_df, threshold=0.5):
    """
    构造一个带完整日K线特征的 selector 闭包。
    threshold: 做T的概率阈值，默认0.5，可调高减少假正例。
    """
    history_days = []

    def selector(morning_bars, open_price, prev_days=None):
        prev_list = prev_days if prev_days is not None else history_days
        target_date = None
        if len(morning_bars) > 0:
            try:
                target_date = pd.Timestamp(morning_bars.index[0]).date()
            except Exception:
                pass
            if target_date is None and "date" in morning_bars.columns:
                target_date = morning_bars.iloc[0]["date"]

        feats = extract_features(
            morning_bars, prev_list, open_price,
            daily_df=daily_df,
            target_date=target_date,
        )
        if feats is None:
            return True
        f_arr = np.array([[feats.get(c, 0.0) for c in feat_cols]])
        prob = float(model.predict_proba(scaler.transform(f_arr))[0][1])
        return prob >= threshold

    def update_history(day_bars):
        history_days.append(day_bars)
        if len(history_days) > 5:
            history_days.pop(0)

    selector.update_history = update_history
    return selector


# ============================================================
# 进程池全局变量（每个 worker 只初始化一次）
# ============================================================
_g_model     = None
_g_scaler    = None
_g_feat_cols = None


def _worker_init():
    global _g_model, _g_scaler, _g_feat_cols
    try:
        import joblib
        bundle = joblib.load(ML_MODEL_PATH)
        _g_model     = bundle["model"]
        _g_scaler    = bundle["scaler"]
        _g_feat_cols = bundle["features"]
    except Exception as e:
        _g_model = None


def run_one(symbol, data, ml_selector=None):
    cfg = dict(STRATEGY_CONFIG)
    if ml_selector is not None:
        cfg["ml_selector"] = ml_selector
        cfg["trend_filter_enabled"] = False
    engine = SimulationEngine(
        symbol=symbol, start_cash=START_CASH, base_shares=BASE_SHARES,
        commission_rate=COMMISSION, min_commission=MIN_COMM,
        stamp_tax=STAMP_TAX, fill_policy="next_open",
    )
    engine.run(data, GridStrategy(cfg))
    stats = engine.get_stats()
    # T净贡献 = 每日(daily_pnl - (close-open)*base_shares)之和
    # 即：今天账户收益 - 只持有底仓不操作的盘中收益
    # 不用 total_pnl - bah_pnl，因为 bah_pnl 包含隔夜跳空，与T操作无关
    t_pnl = sum(d["t_pnl"] for d in engine.daily_stats_list)
    bah_pnl = sum(d["bah_pnl"] for d in engine.daily_stats_list)
    return {
        "bah_pnl":    bah_pnl,
        "total_pnl":  stats.total_pnl,
        "t_pnl":      t_pnl,
        "days":       len(engine.daily_stats_list),
        "trade_days": sum(1 for d in engine.daily_stats_list if d["num_trades"] > 0),
    }


def _run_symbol_month(args):
    """进程池工作函数：单只股票单月三策略回测。"""
    sym, start, end, threshold = args
    data = load_minute_data(sym, start, end)
    if data is None or len(data) == 0:
        return None

    # 构造带完整日K线特征的 selector
    ml_selector = None
    if _g_model is not None:
        daily_df = load_daily(sym)
        ml_selector = make_ml_selector(_g_model, _g_scaler, _g_feat_cols, daily_df,
                                       threshold=threshold)

    try:
        r_no = run_one(sym, data, ml_selector=None)
        r_ml = run_one(sym, data, ml_selector=ml_selector) if ml_selector else None
        return {
            "symbol":   sym,
            "bah":      r_no["bah_pnl"],
            "no_t":     r_no["t_pnl"],
            "no_total": r_no["total_pnl"],
            "ml_t":     r_ml["t_pnl"]      if r_ml else 0.0,
            "ml_total": r_ml["total_pnl"]  if r_ml else 0.0,
            "no_trade": r_no["trade_days"],
            "ml_trade": r_ml["trade_days"] if r_ml else 0,
        }
    except Exception:
        return None
    finally:
        gc.collect()


# ============================================================
# 月度汇总保存
# ============================================================

def save_month(rows, ym):
    if not rows:
        return
    os.makedirs("output", exist_ok=True)
    sub = pd.DataFrame(rows)
    n = len(sub)
    new_row = {
        "月份":          ym,
        "股票数":         n,
        "持有_总收益":    int(sub["bah"].sum()),
        "无ML_T净贡献":   int(sub["no_t"].sum()),
        "无ML_总收益":    int(sub["no_total"].sum()),
        "ML_T净贡献":     int(sub["ml_t"].sum()),
        "ML_总收益":      int(sub["ml_total"].sum()),
        "持有正收益占比": f"{(sub['bah'] > 0).sum()/n:.0%}",
        "无ML_T正占比":   f"{(sub['no_t'] > 0).sum()/n:.0%}",
        "ML_T正占比":     f"{(sub['ml_t'] > 0).sum()/n:.0%}",
        "无ML均交易天":   round(float(sub["no_trade"].mean()), 1),
        "ML均交易天":     round(float(sub["ml_trade"].mean()), 1),
    }
    df_new = pd.DataFrame([new_row])
    if os.path.exists(OUTPUT_CSV):
        df_exist = pd.read_csv(OUTPUT_CSV)
        df_exist = df_exist[df_exist["月份"] != ym]
        df_out = pd.concat([df_exist, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.sort_values("月份").reset_index(drop=True).to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8-sig")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   default="2024-01")
    parser.add_argument("--end",     default="2026-06")
    parser.add_argument("--rerun",   action="store_true",
                        help="强制重跑所有月份（忽略已有结果）")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="ML预测做T的概率阈值（默认0.5）")
    args = parser.parse_args()

    # 按阈值区分输出文件
    global OUTPUT_CSV
    thresh_str = f"{args.threshold:.2f}".replace(".", "")
    OUTPUT_CSV = f"output/monthly_comparison_t{thresh_str}.csv"

    months = gen_months(args.start, args.end)
    print(f"月份范围: {months[0]} ~ {months[-1]}  共 {len(months)} 个月")

    done_months = set()
    if not args.rerun and os.path.exists(OUTPUT_CSV):
        try:
            done_months = set(pd.read_csv(OUTPUT_CSV)["月份"].tolist())
            if done_months:
                print(f"已完成 {len(done_months)} 个月（断点续跑），剩余 "
                      f"{len(months)-len(done_months)} 个月")
        except Exception:
            pass

    months_todo = [m for m in months if m not in done_months]
    if not months_todo:
        print("所有月份已完成，直接打印汇总。")
    else:
        min_files = glob.glob(f"{CACHE_DIR}/min_*.parquet")
        symbols   = sorted(set(os.path.basename(f).split("_")[1] for f in min_files))
        print(f"股票数量: {len(symbols)} 只  并行进程: {args.workers}  阈值: {args.threshold}\n")

        for ym in months_todo:
            y, mo  = int(ym[:4]), int(ym[5:])
            last   = calendar.monthrange(y, mo)[1]
            start  = f"{y}-{mo:02d}-01"
            end    = f"{y}-{mo:02d}-{last}"

            tasks = [(sym, start, end, args.threshold) for sym in symbols]
            rows  = []

            with mp.Pool(processes=args.workers,
                         initializer=_worker_init) as pool:
                for r in pool.imap_unordered(_run_symbol_month, tasks,
                                             chunksize=5):
                    if r is not None:
                        rows.append(r)

            if rows:
                bah_s  = sum(r["bah"]  for r in rows)
                no_t_s = sum(r["no_t"] for r in rows)
                ml_t_s = sum(r["ml_t"] for r in rows)
                print(f"  {ym}  n={len(rows):3d}  "
                      f"持有={bah_s:>12,.0f}  "
                      f"无ML_T={no_t_s:>10,.0f}  "
                      f"ML_T={ml_t_s:>10,.0f}", flush=True)
                save_month(rows, ym)
            gc.collect()

    # 打印最终汇总
    if not os.path.exists(OUTPUT_CSV):
        print("无结果文件")
        return

    df = pd.read_csv(OUTPUT_CSV).sort_values("月份").reset_index(drop=True)
    for c in ["持有_总收益","无ML_T净贡献","无ML_总收益","ML_T净贡献","ML_总收益"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print("\n" + "="*105)
    print(f"{'月份':8s} {'n':4s} {'持有_总收益':>13s} {'无ML_T净贡献':>13s} "
          f"{'ML_T净贡献':>13s} {'无ML_总收益':>13s} {'ML_总收益':>13s} "
          f"{'持有正%':>7s} {'ML_T正%':>7s}")
    print("-"*105)
    for _, r in df.iterrows():
        print(f"{r['月份']:8s} {int(r['股票数']):4d} "
              f"{r['持有_总收益']:>13,.0f} {r['无ML_T净贡献']:>13,.0f} "
              f"{r['ML_T净贡献']:>13,.0f} {r['无ML_总收益']:>13,.0f} "
              f"{r['ML_总收益']:>13,.0f} "
              f"{str(r['持有正收益占比']):>7s} {str(r['ML_T正占比']):>7s}")
    print("-"*105)
    totals = df[["持有_总收益","无ML_T净贡献","ML_T净贡献","无ML_总收益","ML_总收益"]].sum()
    print(f"{'合计':8s} {'':4s} "
          f"{totals['持有_总收益']:>13,.0f} {totals['无ML_T净贡献']:>13,.0f} "
          f"{totals['ML_T净贡献']:>13,.0f} {totals['无ML_总收益']:>13,.0f} "
          f"{totals['ML_总收益']:>13,.0f}")
    print("="*105)
    print(f"\n结果已保存: {OUTPUT_CSV}")
    return df


if __name__ == "__main__":
    main()
