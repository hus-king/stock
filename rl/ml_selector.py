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

# 强制行缓冲，确保管道输出实时可见
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

_VENV_PYTHON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "venv", "bin", "python")
if sys.executable != _VENV_PYTHON and os.path.exists(_VENV_PYTHON):
    # 保留 -u 标志，避免 exec 后输出缓冲
    args = [_VENV_PYTHON]
    if "-u" in sys.argv or "-u" in sys.orig_argv if hasattr(sys, "orig_argv") else False:
        args.append("-u")
    args.extend(sys.argv)
    os.execv(_VENV_PYTHON, args)

import glob
import multiprocessing as mp
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

START_DATE = "2024-01-15"
END_DATE   = "2026-06-01"
EARLY_BARS = 6
LABEL_THRESHOLD = 50   # t_pnl > 50 元才是正样本；|t_pnl| <= 50 的排除（排除噪音）
CV_FOLDS = 5           # 时间序列交叉验证折数

BASE_SHARES = 5000
START_CASH  = 100_000
COMMISSION  = 0.0003
MIN_COMM    = 5.0
STAMP_TAX   = 0.001

# 预筛阈值：近60个交易日
SCREEN_AMP_MIN    = 0.02   # 日均振幅 > 2%
SCREEN_TURN_MIN   = 0.01   # 日均换手率 > 1%
SCREEN_LOOKBACK   = 60     # 用最近60个交易日统计

STRATEGY_CONFIG = {
    "grid_spacing": 0.015,
    "grid_levels": 3,
    "lot_size": 300,
    "base_price_source": "open",
    "trend_filter_enabled": False,  # 关闭：让ML自己学趋势过滤，避免双重过滤污染标签
    "trend_ma_period": 20,
    "max_daily_net_buy": 600,
    "daily_loss_limit": -1000,      # 与推理环境一致
}


# ============================================================
# 日K线加载与预筛
# ============================================================

def load_daily(symbol):
    """加载日K线，返回按日期排序的 DataFrame，index 为 date。"""
    path = os.path.join("data_cache", "daily", f"daily_{symbol}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df = df.set_index("date").sort_index()
    # 去掉股票代码前缀 sz./sh.
    if "code" in df.columns:
        df = df.drop(columns=["code"])
    return df


def screen_symbols_by_daily(all_daily_symbols, lookback=SCREEN_LOOKBACK):
    """用日K线筛选活跃标的（振幅+换手），排除ST和长期停牌，返回通过筛选的股票代码列表。"""
    passed = []
    for sym in all_daily_symbols:
        df = load_daily(sym)
        if df is None or len(df) < lookback:
            continue
        # 排除ST股
        if "isST" in df.columns and df["isST"].iloc[-1] == 1:
            continue
        recent = df.iloc[-lookback:]
        # 排除近期有大量停牌的股票（停牌天数超过20%）
        if "tradestatus" in df.columns:
            trade_ratio = (recent["tradestatus"] == 1).mean()
            if trade_ratio < 0.8:
                continue
        avg_amp  = ((recent["high"] - recent["low"]) / recent["open"]).mean()
        avg_turn = recent["turn"].mean() / 100.0
        if avg_amp >= SCREEN_AMP_MIN and avg_turn >= SCREEN_TURN_MIN:
            passed.append(sym)
    return passed


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
    # 直接使用引擎记录的每日T净贡献（daily_pnl - 底仓浮盈）
    daily_t = {d["date"]: d["t_pnl"] for d in engine.daily_stats_list}
    return daily_t


# ============================================================
# 日K线衍生特征（开盘前已知，无未来泄露）
# ============================================================

def extract_daily_features(daily_df, target_date):
    """
    基于日K线计算截止昨日的衍生特征。
    target_date: 当天日期（datetime.date）
    返回 dict 或 None。
    """
    ts = pd.Timestamp(target_date)
    hist = daily_df[daily_df.index < ts]

    # 过滤停牌日
    if "tradestatus" in hist.columns:
        hist = hist[hist["tradestatus"] == 1]

    if len(hist) < 20:
        return None

    recent5  = hist.iloc[-5:]
    recent10 = hist.iloc[-10:]
    recent20 = hist.iloc[-20:]

    closes  = recent20["close"].values.astype(float)
    opens20 = recent20["open"].values.astype(float)
    highs20 = recent20["high"].values.astype(float)
    lows20  = recent20["low"].values.astype(float)
    vols20  = recent20["volume"].values.astype(float)

    # 近5/20日日均振幅
    amp20 = ((highs20 - lows20) / (opens20 + 1e-8)).mean()
    amp5  = ((recent5["high"].values - recent5["low"].values) /
             (recent5["open"].values + 1e-8)).mean()

    # 昨日成交量相对近20日 z-score
    vol_mean = vols20.mean()
    vol_std  = vols20.std() + 1e-8
    vol_zscore = (float(hist.iloc[-1]["volume"]) - vol_mean) / vol_std

    # 近10日收盘价线性斜率（归一化）
    x = np.arange(len(recent10))
    c10 = recent10["close"].values.astype(float)
    trend_slope = np.polyfit(x, c10, 1)[0] / (c10.mean() + 1e-8)

    # 连续同向天数（连涨为正，连跌为负）
    rets = np.diff(closes)
    consec = 0
    if len(rets) > 0:
        last_dir = np.sign(rets[-1])
        for r in reversed(rets):
            if np.sign(r) == last_dir and r != 0:
                consec += int(last_dir)
            else:
                break

    # 近5日换手率均值
    turn_ma5 = float(recent5["turn"].mean()) / 100.0

    # 距近20日最高价的距离（负值=在最高价下方）
    high20 = highs20.max()
    last_close = float(hist.iloc[-1]["close"])
    high_dist = (last_close - high20) / (high20 + 1e-8)

    # PE分位（滚动近2年窗口，避免历史长度不一致导致分位漂移）
    ROLL_WINDOW = 500  # 约2年交易日
    pe_rank = 0.5
    if "peTTM" in hist.columns:
        pe_series = hist["peTTM"].dropna()
        if len(pe_series) > 10:
            roll = pe_series.iloc[-ROLL_WINDOW:]
            cur_pe = float(hist.iloc[-1].get("peTTM", np.nan))
            if not np.isnan(cur_pe):
                pe_rank = float((roll < cur_pe).mean())

    # PB分位（同上，滚动近2年窗口）
    pb_rank = 0.5
    if "pbMRQ" in hist.columns:
        pb_series = hist["pbMRQ"].dropna()
        if len(pb_series) > 10:
            roll = pb_series.iloc[-ROLL_WINDOW:]
            cur_pb = float(hist.iloc[-1].get("pbMRQ", np.nan))
            if not np.isnan(cur_pb):
                pb_rank = float((roll < cur_pb).mean())

    # 用 preclose 计算精确跳空幅度（比5分钟数据更准）
    gap_preclose = 0.0
    if "preclose" in hist.columns:
        preclose = float(hist.iloc[-1]["preclose"])
        if preclose > 0:
            gap_preclose = (last_close - preclose) / preclose

    return {
        "daily_amp_ma5":      float(amp5),
        "daily_amp_ma20":     float(amp20),
        "daily_vol_zscore":   float(vol_zscore),
        "daily_trend_slope":  float(trend_slope),
        "daily_consec_dir":   float(consec),
        "daily_turn_ma5":     float(turn_ma5),
        "daily_high_dist":    float(high_dist),
        "daily_pe_rank":      float(pe_rank),
        "daily_pb_rank":      float(pb_rank),
        "daily_gap_preclose": float(gap_preclose),
    }


# ============================================================
# 早盘特征提取（原有18维）
# ============================================================

def extract_features(day_bars, prev_days_list, open_price, daily_df=None, target_date=None):
    if len(day_bars) < EARLY_BARS or open_price <= 0:
        return None

    morning = day_bars.iloc[:EARLY_BARS]
    closes  = morning["close"].values.astype(float)
    vols    = morning["volume"].values.astype(float)

    m_high  = float(morning["high"].max())
    m_low   = float(morning["low"].min())
    m_close = float(morning.iloc[-1]["close"])
    x       = np.arange(len(closes))
    slope   = np.polyfit(x, closes, 1)[0] / open_price if len(closes) > 1 else 0.0
    rets    = np.diff(closes) / (closes[:-1] + 1e-8)
    vol_mean_per_bar = float(morning["volume"].mean()) + 1e-8
    vwap    = np.sum(closes * vols) / (vols.sum() + 1e-8)

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

    # 原有跨日特征（基于5分钟缓存）
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
        amps   = [(float(p["high"].max()) - float(p["low"].min())) /
                  (float(p.iloc[0]["open"]) + 1e-8) for p in hist]
        rets_h = [(float(p.iloc[-1]["close"]) - float(p.iloc[0]["open"])) /
                  (float(p.iloc[0]["open"]) + 1e-8) for p in hist]
        hvols  = [float(p["volume"].mean()) for p in hist]
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

    # 新增：日K线衍生特征
    if daily_df is not None and target_date is not None:
        df_feats = extract_daily_features(daily_df, target_date)
        if df_feats:
            feats.update(df_feats)
        else:
            feats.update({
                "daily_amp_ma5": feats["m_amplitude"], "daily_amp_ma20": feats["m_amplitude"],
                "daily_vol_zscore": 0.0, "daily_trend_slope": 0.0,
                "daily_consec_dir": 0.0, "daily_turn_ma5": 0.0,
                "daily_high_dist": 0.0, "daily_pe_rank": 0.5,
                "daily_pb_rank": 0.5, "daily_gap_preclose": 0.0,
            })
    else:
        feats.update({
            "daily_amp_ma5": feats["m_amplitude"], "daily_amp_ma20": feats["m_amplitude"],
            "daily_vol_zscore": 0.0, "daily_trend_slope": 0.0,
            "daily_consec_dir": 0.0, "daily_turn_ma5": 0.0,
            "daily_high_dist": 0.0, "daily_pe_rank": 0.5,
            "daily_pb_rank": 0.5, "daily_gap_preclose": 0.0,
        })

    return feats


# ============================================================
# 构建数据集（单股处理函数，供进程池调用）
# ============================================================

def _has_minute_cache(symbol, cache_dir="data_cache"):
    """检查是否有该股票的任意5分钟缓存文件（不触发网络请求）。"""
    import glob as _glob
    pattern = os.path.join(cache_dir, "min_{}_*.parquet".format(symbol))
    return len(_glob.glob(pattern)) > 0


def _load_baostock_cache(symbol, start_date, end_date, cache_dir="data_cache"):
    """直接从 baostock parquet 缓存加载5分钟数据，过滤日期范围。"""
    import hashlib
    for s, e in [("2024-01-01", "2026-06-01"), ("2026-01-01", "2026-06-01")]:
        key = "{}|{}|{}|5|baostock".format(symbol, s, e)
        h = hashlib.md5(key.encode()).hexdigest()[:8]
        path = os.path.join(cache_dir, "min_{}_{}.parquet".format(symbol, h))
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if "date" not in df.columns:
                df["date"] = df.index.date
            # 过滤到请求的日期范围
            start_dt = pd.to_datetime(start_date).date()
            end_dt   = pd.to_datetime(end_date).date()
            df = df[df["date"].between(start_dt, end_dt)]
            if len(df) > 0:
                return df
    return None


def _process_symbol(args):
    """单只股票的回测+特征提取，返回 (feats_list, labels_list, meta_list)。"""
    sym, start_date, end_date = args

    # 只使用本地已有缓存，跳过无缓存股票（避免触发网络下载）
    if not _has_minute_cache(sym):
        return [], [], []

    # 优先直接读 baostock 缓存（避免 MinuteDataLoader 触发网络请求）
    data = _load_baostock_cache(sym, start_date, end_date)
    if data is None:
        try:
            loader = MinuteDataLoader(cache_dir="data_cache")
            data = loader.load(sym, start_date, end_date, "5", "qfq")
        except Exception:
            return [], [], []
    if data is None or len(data) == 0:
        return [], [], []

    if "date" not in data.columns:
        data = data.copy()
        data["date"] = data.index.date

    daily_df = load_daily(sym)

    try:
        t_pnl = run_backtest(sym, data)
    except Exception:
        return [], [], []

    dates = sorted(data["date"].unique())
    day_bars_list = [data[data["date"] == d].reset_index() for d in dates]

    feats_list, labels_list, meta_list = [], [], []
    for i, (date, day_bars) in enumerate(zip(dates, day_bars_list)):
        open_price = float(day_bars.iloc[0]["open"]) if len(day_bars) > 0 else 0
        feats = extract_features(
            day_bars,
            day_bars_list[max(0, i - 5):i],
            open_price,
            daily_df=daily_df,
            target_date=date,
        )
        if feats is None or date not in t_pnl:
            continue
        pnl = t_pnl[date]
        if abs(pnl) <= LABEL_THRESHOLD:
            continue  # 排除噪音样本：|t_pnl| <= 50 元
        feats_list.append(feats)
        labels_list.append(1 if pnl > 0 else 0)
        meta_list.append({"symbol": sym, "date": date, "t_pnl": pnl})

    return feats_list, labels_list, meta_list


def build_dataset(symbols, start_date, end_date, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 2)

    args = [(sym, start_date, end_date) for sym in symbols]
    all_feats, labels, meta = [], [], []

    print(f"并行处理 {len(symbols)} 只股票，使用 {n_jobs} 个进程...")
    with mp.Pool(processes=n_jobs) as pool:
        for i, (f_list, l_list, m_list) in enumerate(
            pool.imap_unordered(_process_symbol, args), 1
        ):
            all_feats.extend(f_list)
            labels.extend(l_list)
            meta.extend(m_list)
            if i % 10 == 0 or i == len(symbols):
                print(f"  {i}/{len(symbols)} 只完成，当前样本数: {len(labels)}")

    return pd.DataFrame(all_feats), np.array(labels), pd.DataFrame(meta)


# ============================================================
# 训练与评估
# ============================================================

def _train_one_model(X_tr, y_tr, X_te, y_te, n_jobs):
    """训练单个 XGBoost 模型，返回 (model, y_prob)。"""
    scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42, eval_metric="logloss", verbosity=0,
        nthread=n_jobs,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    y_prob = model.predict_proba(X_te)[:, 1]
    return model, y_prob


def train(df_feat, labels, df_meta, n_folds=None):
    """时间序列交叉验证 + 全量数据训练最终模型。"""
    if n_folds is None:
        n_folds = CV_FOLDS

    # 按日期排序
    dates = df_meta["date"].values
    sort_idx = np.argsort(dates)
    X_all = df_feat.values[sort_idx]
    y_all = labels[sort_idx]
    d_all = dates[sort_idx]

    unique_dates = sorted(set(d_all))
    n_dates = len(unique_dates)

    # 将日期等分为 n_folds+1 段，每折测试集为一段
    fold_size = n_dates // (n_folds + 1)
    if fold_size < 1:
        print(f"错误：日期数 {n_dates} 不足以做 {n_folds} 折交叉验证")
        return None, None

    n_jobs = max(1, mp.cpu_count() - 2)

    print(f"\n{'='*60}")
    print(f"时间序列交叉验证 ({n_folds}折)  |  标签阈值: t_pnl > {LABEL_THRESHOLD}")
    print(f"{'='*60}")

    cv_results = []
    for fold in range(n_folds):
        # 测试集：倒数第 (n_folds - fold) 段
        test_start_rank = n_dates - (n_folds - fold) * fold_size
        test_end_rank   = min(n_dates, test_start_rank + fold_size)
        test_start_date = unique_dates[test_start_rank]
        test_end_date   = unique_dates[test_end_rank - 1]

        te_mask = (d_all >= test_start_date) & (d_all < test_end_date)
        tr_mask = d_all < test_start_date

        if te_mask.sum() == 0 or tr_mask.sum() == 0:
            continue

        X_tr_raw, X_te_raw = X_all[tr_mask], X_all[te_mask]
        y_tr, y_te = y_all[tr_mask], y_all[te_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_raw)
        X_te_s = scaler.transform(X_te_raw)

        model, y_prob = _train_one_model(X_tr_s, y_tr, X_te_s, y_te, n_jobs)
        y_pred = (y_prob >= 0.5).astype(int)

        acc  = (y_pred == y_te).mean()
        prec = (y_pred[y_pred == 1] == y_te[y_pred == 1]).mean() if y_pred.sum() > 0 else 0
        rec  = (y_te[y_te == 1] == y_pred[y_te == 1]).mean() if y_te.sum() > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        try:
            auc = roc_auc_score(y_te, y_prob)
        except Exception:
            auc = float("nan")

        print(f"\n--- Fold {fold+1}/{n_folds}: "
              f"训练 {unique_dates[0]}~{unique_dates[test_start_rank-1]}, "
              f"测试 {test_start_date}~{test_end_date} ---")
        print(f"  训练样本: {tr_mask.sum():,}  测试样本: {te_mask.sum():,}  "
              f"正样本比例: {y_tr.mean():.1%} / {y_te.mean():.1%}")
        print(f"  accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  "
              f"f1={f1:.4f}  auc={auc:.4f}")

        cv_results.append({
            "fold": fold + 1,
            "test_dates": f"{test_start_date}~{test_end_date}",
            "n_train": tr_mask.sum(),
            "n_test": te_mask.sum(),
            "train_pos": y_tr.mean(),
            "test_pos": y_te.mean(),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
        })

    # 汇总
    if not cv_results:
        print("无有效折，跳过CV")
        return None, None

    aucs  = [r["auc"] for r in cv_results]
    precs = [r["precision"] for r in cv_results]
    recs  = [r["recall"] for r in cv_results]
    f1s   = [r["f1"] for r in cv_results]

    print(f"\n{'='*60}")
    print(f"CV 汇总 (mean ± std)")
    print(f"{'='*60}")
    print(f"  AUC:       {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"  Recall:    {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"  F1:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # === 全量数据训练最终模型 ===
    print(f"\n{'='*60}")
    print("最终模型（全量数据训练）")
    print(f"{'='*60}")
    print(f"总样本: {len(y_all):,}  正样本: {y_all.sum():,} ({y_all.mean():.1%})")
    print(f"特征维度: {df_feat.shape[1]}")

    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)

    model, y_prob_all = _train_one_model(X_all_s, y_all, X_all_s, y_all, n_jobs)

    # 全量数据的训练集分类报告（in-sample，仅供参考）
    y_pred_all = (y_prob_all >= 0.5).astype(int)
    print("\n全量样本分类报告 (in-sample，仅供参考):")
    print(classification_report(y_all, y_pred_all, target_names=["不做T", "做T"]))
    try:
        print(f"AUC: {roc_auc_score(y_all, y_prob_all):.4f}")
    except Exception:
        pass

    feat_imp = sorted(zip(df_feat.columns, model.feature_importances_), key=lambda x: -x[1])
    print("\n特征重要性 (top 15):")
    for name, imp in feat_imp[:15]:
        print(f"  {name:28s}: {imp:.4f}")

    return model, scaler


# ============================================================
# Main
# ============================================================

def main():
    # 1. 从日K线里找出所有300只股票代码
    daily_files = glob.glob("data_cache/daily/daily_*.parquet")
    all_daily_syms = sorted(
        os.path.basename(f).replace("daily_", "").replace(".parquet", "")
        for f in daily_files
    )
    print(f"日K线覆盖: {len(all_daily_syms)} 只股票")

    # 2. 预筛：振幅+换手率
    print(f"预筛条件：近{SCREEN_LOOKBACK}日均振幅>{SCREEN_AMP_MIN:.0%}  "
          f"均换手率>{SCREEN_TURN_MIN:.0%} ...")
    active_syms = screen_symbols_by_daily(all_daily_syms)
    print(f"预筛通过: {len(active_syms)} 只\n")

    # 3. 和已有5分钟缓存取并集（缓存里有的即使不在预筛里也保留）
    min_files = glob.glob("data_cache/min_*.parquet")
    cached_syms = sorted(set(os.path.basename(f).split("_")[1] for f in min_files))
    symbols = sorted(set(active_syms) | set(cached_syms))
    in_both = len(set(active_syms) & set(cached_syms))
    print(f"5分钟缓存: {len(cached_syms)} 只  预筛通过: {len(active_syms)} 只  "
          f"两者交集: {in_both} 只")
    print(f"训练股票池: {len(symbols)} 只（仅使用有本地缓存的，跳过网络下载）")
    print(f"日期: {START_DATE} ~ {END_DATE}\n")

    # 4. 构建数据集
    df_feat, labels, df_meta = build_dataset(symbols, START_DATE, END_DATE)
    if len(labels) == 0:
        print("没有有效样本，退出。")
        return
    print(f"\n总样本: {len(labels)}  正样本: {labels.sum()} ({labels.mean():.1%})")
    print(f"标签阈值: t_pnl > {LABEL_THRESHOLD} (|t_pnl| <= {LABEL_THRESHOLD} 已排除)")
    print(f"特征维度: {df_feat.shape[1]}  特征: {df_feat.columns.tolist()}")

    # 5. 训练
    model, scaler = train(df_feat, labels, df_meta)

    # 6. 保存
    import joblib
    os.makedirs("rl/models", exist_ok=True)
    joblib.dump(
        {"model": model, "scaler": scaler, "features": df_feat.columns.tolist()},
        "rl/models/ml_selector.pkl",
    )
    print("\n模型已保存到 rl/models/ml_selector.pkl")


if __name__ == "__main__":
    main()
