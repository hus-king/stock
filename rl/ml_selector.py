#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 择日分类器：用早盘前6根K线（含跨日）特征预测当天是否适合做T。

用法：
    venv/bin/python rl/ml_selector.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_VENV_PYTHON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "venv", "bin", "python")
if sys.executable != _VENV_PYTHON and os.path.exists(_VENV_PYTHON):
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from stock_open_api.simulation.data import MinuteDataLoader
from stock_open_api.simulation.engine import SimulationEngine
from stock_open_api.simulation.strategy import GridStrategy

START_DATE = "2026-04-01"
END_DATE   = "2026-06-01"
EARLY_BARS = 6

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
    "trend_filter_enabled": False,
    "max_daily_net_buy": 600,
    "daily_loss_limit": -9999,
}


# ============================================================
# 回测标签生成
# ============================================================

def run_backtest(symbol, data):
    strategy = GridStrategy(STRATEGY_CONFIG)
    engine = SimulationEngine(
        symbol=symbol, start_cash=START_CASH, base_shares=BASE_SHARES,
        commission_rate=COMMISSION, min_commission=MIN_COMM,
        stamp_tax=STAMP_TAX, fill_policy="next_open",
    )
    engine.run(data, strategy)
    daily_t = {}
    for d in engine.daily_stats_list:
        date = d["date"]
        day_trades = [t for t in engine.trades
                      if hasattr(t.timestamp, "date")
                      and t.timestamp.date() == date
                      and t.tag != "closeout"]
        t_cash = sum(
            t.quantity * t.price * (-1 if t.side == "BUY" else 1) - t.commission
            for t in day_trades
        )
        daily_t[date] = t_cash
    return daily_t


# ============================================================
# 特征提取（早盘 + 昨日 + 近5日）
# ============================================================

def extract_features(day_bars, prev_days_list, open_price):
    if len(day_bars) < EARLY_BARS or open_price <= 0:
        return None

    morning = day_bars.iloc[:EARLY_BARS]
    closes  = morning["close"].values.astype(float)
    vols    = morning["volume"].values.astype(float)

    m_high = float(morning["high"].max())
    m_low  = float(morning["low"].min())
    m_close = float(morning.iloc[-1]["close"])
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0] / open_price if len(closes) > 1 else 0.0
    rets  = np.diff(closes) / (closes[:-1] + 1e-8)
    vol_mean_per_bar = float(day_bars["volume"].mean()) + 1e-8
    vwap  = np.sum(closes * vols) / (vols.sum() + 1e-8)

    feats = {
        "m_amplitude":  (m_high - m_low) / open_price,
        "m_return":     (m_close - open_price) / open_price,
        "m_range":      (m_high - m_low) / (m_close + 1e-8),
        "m_vol_ratio":  float(vols.sum()) / (vol_mean_per_bar * EARLY_BARS),
        "m_slope":      slope,
        "m_volatility": float(np.std(rets)) if len(rets) > 0 else 0.0,
        "m_first_gap":  (float(morning.iloc[0]["close"]) - open_price) / open_price,
        "m_high_pos":   morning["high"].values.argmax() / max(EARLY_BARS - 1, 1),
        "m_low_pos":    morning["low"].values.argmin()  / max(EARLY_BARS - 1, 1),
        "m_vwap_dev":   (vwap - open_price) / open_price,
    }

    if prev_days_list:
        prev = prev_days_list[-1]
        p_open  = float(prev.iloc[0]["open"])
        p_close = float(prev.iloc[-1]["close"])
        p_high  = float(prev["high"].max())
        p_low   = float(prev["low"].min())
        feats.update({
            "prev_amplitude": (p_high - p_low) / (p_open + 1e-8),
            "prev_return":    (p_close - p_open) / (p_open + 1e-8),
            "prev_vol_norm":  float(prev["volume"].mean()) / vol_mean_per_bar,
            "gap_vs_prev":    (open_price - p_close) / (p_close + 1e-8),
        })
    else:
        feats.update({"prev_amplitude": 0.0, "prev_return": 0.0,
                      "prev_vol_norm": 1.0,  "gap_vs_prev": 0.0})

    n_hist = min(5, len(prev_days_list))
    if n_hist >= 2:
        hist = prev_days_list[-n_hist:]
        amps = [(float(p["high"].max()) - float(p["low"].min())) /
                (float(p.iloc[0]["open"]) + 1e-8) for p in hist]
        rets_h = [(float(p.iloc[-1]["close"]) - float(p.iloc[0]["open"])) /
                  (float(p.iloc[0]["open"]) + 1e-8) for p in hist]
        hvols = [float(p["volume"].mean()) for p in hist]
        feats.update({
            "hist_amp_mean":  float(np.mean(amps)),
            "hist_amp_std":   float(np.std(amps)),
            "hist_ret_mean":  float(np.mean(rets_h)),
            "hist_vol_trend": float(np.polyfit(range(len(hvols)), hvols, 1)[0])
                              / (np.mean(hvols) + 1e-8),
        })
    else:
        feats.update({"hist_amp_mean": feats["m_amplitude"], "hist_amp_std": 0.0,
                      "hist_ret_mean": 0.0, "hist_vol_trend": 0.0})
    return feats


# ============================================================
# 构建数据集
# ============================================================

def build_dataset(symbols, start_date, end_date):
    loader = MinuteDataLoader(cache_dir="data_cache")
    all_feats, labels, meta = [], [], []

    for sym in symbols:
        try:
            data = loader.load(sym, start_date, end_date, "5", "qfq")
        except Exception as e:
            print(f"  跳过 {sym}: {e}")
            continue
        if "date" not in data.columns:
            data = data.copy()
            data["date"] = data.index.date

        t_pnl = run_backtest(sym, data)
        dates = sorted(data["date"].unique())
        day_bars_list = [data[data["date"] == d].reset_index() for d in dates]

        for i, (date, day_bars) in enumerate(zip(dates, day_bars_list)):
            open_price = float(day_bars.iloc[0]["open"]) if len(day_bars) > 0 else 0
            feats = extract_features(day_bars, day_bars_list[max(0, i-5):i], open_price)
            if feats is None or date not in t_pnl:
                continue
            all_feats.append(feats)
            labels.append(1 if t_pnl[date] > 0 else 0)
            meta.append({"symbol": sym, "date": date, "t_pnl": t_pnl[date]})

    return pd.DataFrame(all_feats), np.array(labels), pd.DataFrame(meta)


# ============================================================
# 训练与评估
# ============================================================

def train(df_feat, labels, df_meta, train_ratio=0.8):
    dates   = df_meta["date"].values
    cutoff  = sorted(set(dates))[int(len(set(dates)) * train_ratio)]
    tr, te  = dates < cutoff, dates >= cutoff

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(df_feat[tr].values)
    X_te_s  = scaler.transform(df_feat[te].values)
    y_tr, y_te = labels[tr], labels[te]

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    model.fit(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)], verbose=False)

    y_pred = model.predict(X_te_s)
    y_prob = model.predict_proba(X_te_s)[:, 1]

    print(f"\n训练集: {tr.sum()} 天  测试集: {te.sum()} 天")
    print(f"正样本比例 - 训练: {y_tr.mean():.1%}  测试: {y_te.mean():.1%}")
    print("\n分类报告:")
    print(classification_report(y_te, y_pred, target_names=["不做T", "做T"]))
    try:
        print(f"AUC: {roc_auc_score(y_te, y_prob):.4f}")
    except Exception:
        pass

    feat_imp = sorted(zip(df_feat.columns, model.feature_importances_), key=lambda x: -x[1])
    print("\n特征重要性:")
    for name, imp in feat_imp:
        print(f"  {name:25s}: {imp:.4f}")

    # 择日效果对比
    test_meta = df_meta[te].copy()
    test_meta["pred"] = y_pred
    all_t = test_meta["t_pnl"].sum()
    ml_t  = test_meta[test_meta["pred"] == 1]["t_pnl"].sum()
    n_reduced = te.sum() - (test_meta["pred"] == 1).sum()
    print(f"\n[回测对比] 全天做T: {all_t:,.0f}元  ML择日: {ml_t:,.0f}元  "
          f"减少交易: {n_reduced}/{te.sum()} 天")

    return model, scaler


# ============================================================
# Main
# ============================================================

def main():
    files   = glob.glob("data_cache/min_*.parquet")
    symbols = sorted(set(os.path.basename(f).split("_")[1] for f in files))
    print(f"共 {len(symbols)} 只股票，{START_DATE} ~ {END_DATE}\n")

    df_feat, labels, df_meta = build_dataset(symbols, START_DATE, END_DATE)
    print(f"总样本: {len(labels)}  正样本: {labels.sum()} ({labels.mean():.1%})")

    model, scaler = train(df_feat, labels, df_meta)

    import joblib
    os.makedirs("rl/models", exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler,
                 "features": df_feat.columns.tolist()},
                "rl/models/ml_selector.pkl")
    print("\n模型已保存到 rl/models/ml_selector.pkl")


if __name__ == "__main__":
    main()
