#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 BaoStock 下载沪深300全部成分股近3年日K线数据，保存到 data_cache/daily/ 目录。

用法:
    venv/bin/python rl/download_daily_baostock.py
    venv/bin/python rl/download_daily_baostock.py --start 2022-01-01 --end 2026-06-01
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import baostock as bs
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache", "daily")
START_DATE = "2023-01-01"
END_DATE   = "2026-06-01"


def get_hs300_symbols():
    """获取当前沪深300成分股列表，返回 baostock 格式代码列表（如 sh.600000）。"""
    rs = bs.query_hs300_stocks()
    df = rs.get_data()
    return df["code"].tolist()


def bs_code_to_plain(code):
    """sh.600000 -> 600000"""
    return code.split(".")[1]


def download_daily(code, start_date, end_date, timeout=60):
    """下载单只股票日K线，返回 DataFrame。timeout秒无响应则放弃。"""
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("baostock query timeout after {}s".format(timeout))

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,preclose,volume,amount,turn,"
            "tradestatus,pctChg,peTTM,pbMRQ,isST",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",  # 前复权
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
    finally:
        signal.alarm(0)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=rs.fields)
    # 转换数值类型
    for col in ["open", "high", "low", "close", "preclose", "volume", "amount",
                "turn", "pctChg", "peTTM", "pbMRQ"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["tradestatus", "isST"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] > 0]
    return df.sort_values("date").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="BaoStock 下载沪深300日K线")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end",   default=END_DATE)
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    print("登录 BaoStock...")
    lg = bs.login()
    if lg.error_code != "0":
        print(f"登录失败: {lg.error_msg}")
        return

    print("获取沪深300成分股列表...")
    symbols = get_hs300_symbols()
    print(f"共 {len(symbols)} 只，日期范围 {args.start} ~ {args.end}\n")

    success, failed = [], []
    for i, code in enumerate(symbols):
        plain = bs_code_to_plain(code)
        cache_path = os.path.join(CACHE_DIR, f"daily_{plain}.parquet")

        print(f"  [{i+1:3d}/{len(symbols)}] {plain} 下载中...", end=" ", flush=True)
        try:
            df = download_daily(code, args.start, args.end)
            if df is None or df.empty:
                print("空数据")
                failed.append(plain)
                continue
            df.to_parquet(cache_path, index=False)
            print(f"OK ({len(df)} 天)")
            success.append(plain)
        except TimeoutError:
            print("TIMEOUT (跳过)")
            failed.append(plain)
        except Exception as e:
            print(f"FAIL: {e}")
            failed.append(plain)
        time.sleep(0.05)  # 避免请求过快

    bs.logout()
    print(f"\n完成: {len(success)} 成功, {len(failed)} 失败")
    if failed:
        print(f"失败列表: {failed[:20]}")

    # 打印数据概况
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".parquet")]
    print(f"\n本地日K线缓存: {len(files)} 只股票，路径: {CACHE_DIR}")


if __name__ == "__main__":
    main()
