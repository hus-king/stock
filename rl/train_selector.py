#!/usr/bin/env python3
"""精简训练入口：只用已有5分钟缓存的55只股票，跳过预筛，直接并行回测+训练。"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import multiprocessing as mp

from rl.ml_selector import (
    _process_symbol, build_dataset, train, load_daily,
    START_DATE, END_DATE,
)

if __name__ == "__main__":
    import hashlib

    def _has_real_cache(sym, start, end, cache_dir="data_cache"):
        """检查股票是否有真实行情缓存（非合成数据）。"""
        # 合成数据的 hash
        synth_key = "{}|{}|{}|5|synth".format(sym, start, end)
        synth_hash = hashlib.md5(synth_key.encode()).hexdigest()[:8]
        synth_fname = "min_{}_{}.parquet".format(sym, synth_hash)

        all_files = glob.glob(os.path.join(cache_dir, "min_{}_*.parquet".format(sym)))
        # 只要有任何一个非synth的缓存文件就算真实数据
        real_files = [f for f in all_files
                      if os.path.basename(f) != synth_fname]
        return len(real_files) > 0

    min_files = glob.glob("data_cache/min_*.parquet")
    all_cached = sorted(set(os.path.basename(f).split("_")[1] for f in min_files))

    symbols = [sym for sym in all_cached
               if _has_real_cache(sym, START_DATE, END_DATE)]
    excluded = len(all_cached) - len(symbols)
    print(f"缓存股票: {len(all_cached)} 只  排除合成数据: {excluded} 只  "
          f"真实行情: {len(symbols)} 只", flush=True)
    print(f"日期: {START_DATE} ~ {END_DATE}", flush=True)

    n_jobs = max(1, mp.cpu_count() - 2)
    print(f"并行进程数: {n_jobs}", flush=True)

    df_feat, labels, df_meta = build_dataset(symbols, START_DATE, END_DATE, n_jobs=n_jobs)

    if len(labels) == 0:
        print("没有有效样本，退出。")
        sys.exit(1)

    print(f"\n总样本: {len(labels)}  正样本: {labels.sum()} ({labels.mean():.1%})", flush=True)
    print(f"特征维度: {df_feat.shape[1]}", flush=True)

    model, scaler = train(df_feat, labels, df_meta)

    os.makedirs("rl/models", exist_ok=True)
    joblib.dump(
        {"model": model, "scaler": scaler, "features": df_feat.columns.tolist()},
        "rl/models/ml_selector.pkl",
    )
    print("\n模型已保存到 rl/models/ml_selector.pkl", flush=True)
