#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 BaoStock 下载沪深300成分股5分钟K线数据，保存到 data_cache/ 目录。

BaoStock 5分钟数据范围：2020-01-03 至今（近5年）

用法:
    venv/bin/python rl/download_min_baostock.py
    venv/bin/python rl/download_min_baostock.py --start 2024-01-01 --end 2026-06-01
    venv/bin/python rl/download_min_baostock.py --start 2024-01-01 --end 2026-06-01 --symbols 000001,000002
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import baostock as bs
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")
START_DATE = "2024-01-01"
END_DATE   = "2026-06-01"


def get_hs300_symbols():
    rs = bs.query_hs300_stocks()
    df = rs.get_data()
    return df["code"].tolist()


def bs_code_to_plain(code):
    """sh.600000 -> 600000"""
    return code.split(".")[1]


def plain_to_bs_code(symbol):
    """600000 -> sh.600000,  000001 -> sz.000001"""
    if symbol.startswith(("6", "9")):
        return "sh.{}".format(symbol)
    return "sz.{}".format(symbol)


def cache_path_for(symbol, start_date, end_date):
    """生成缓存文件路径，格式与 MinuteDataLoader 兼容。"""
    import hashlib
    key = "{}|{}|{}|5|baostock".format(symbol, start_date, end_date)
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return os.path.join(CACHE_DIR, "min_{}_{}.parquet".format(symbol, h))


def download_minute(bs_code, start_date, end_date, timeout=60):
    """下载单只股票5分钟K线，返回标准格式 DataFrame。timeout秒无响应则放弃。"""
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("baostock query timeout after {}s".format(timeout))

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,time,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="5",
            adjustflag="2",  # 前复权
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
    finally:
        signal.alarm(0)  # 取消定时器
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=rs.fields)

    # 合并 date + time -> timestamp
    # baostock time 格式: "20260101093500000" -> 取第8~12位 "0935"
    hhmm = df["time"].str[8:12]  # "0935"
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + hhmm.str[:2] + ":" + hhmm.str[2:],
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"])

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] > 0]
    df["date"] = df["timestamp"].dt.date
    df = df[["timestamp", "open", "high", "low", "close", "volume", "amount", "date"]]
    df = df.set_index("timestamp").sort_index()
    return df


def main():
    parser = argparse.ArgumentParser(description="BaoStock 下载沪深300五分钟K线")
    parser.add_argument("--start", default=START_DATE, help="开始日期 (默认 {})".format(START_DATE))
    parser.add_argument("--end",   default=END_DATE,   help="结束日期 (默认 {})".format(END_DATE))
    parser.add_argument("--symbols", default=None,
                        help="指定股票代码，逗号分隔（默认下载全部沪深300）")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="跳过已有缓存文件（默认开启）")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    print("登录 BaoStock...")
    lg = bs.login()
    if lg.error_code != "0":
        print("登录失败: {}".format(lg.error_msg))
        return
    print("登录成功\n")

    # 确定股票列表
    if args.symbols:
        plain_symbols = [s.strip().zfill(6) for s in args.symbols.split(",")]
        print("指定股票: {} 只".format(len(plain_symbols)))
    else:
        print("获取沪深300成分股列表...")
        bs_symbols = get_hs300_symbols()
        plain_symbols = [bs_code_to_plain(c) for c in bs_symbols]
        print("共 {} 只".format(len(plain_symbols)))

    print("日期范围: {} ~ {}  周期: 5分钟\n".format(args.start, args.end))

    success, skipped, failed = [], [], []

    for i, sym in enumerate(plain_symbols):
        path = cache_path_for(sym, args.start, args.end)

        if args.skip_existing and os.path.exists(path):
            print("  [{:3d}/{}] {} 已缓存，跳过".format(i + 1, len(plain_symbols), sym))
            skipped.append(sym)
            continue

        print("  [{:3d}/{}] {} 下载中...".format(i + 1, len(plain_symbols), sym),
              end=" ", flush=True)
        try:
            bs_code = plain_to_bs_code(sym)
            df = download_minute(bs_code, args.start, args.end)
            if df is None or df.empty:
                print("空数据")
                failed.append(sym)
                continue
            df.to_parquet(path)
            days = df["date"].nunique()
            print("OK ({} 天, {} 条)".format(days, len(df)))
            success.append(sym)
        except TimeoutError:
            print("TIMEOUT (跳过)")
            failed.append(sym)
        except Exception as e:
            print("FAIL: {}".format(e))
            failed.append(sym)

        time.sleep(0.1)  # 避免请求过快

    bs.logout()

    print("\n" + "=" * 50)
    print("完成: {} 新下载, {} 已跳过, {} 失败".format(
        len(success), len(skipped), len(failed)))
    if failed:
        print("失败列表: {}".format(failed[:20]))

    # 统计缓存总量
    import glob
    all_min = glob.glob(os.path.join(CACHE_DIR, "min_*.parquet"))
    all_syms = sorted(set(os.path.basename(f).split("_")[1] for f in all_min))
    print("本地5分钟缓存总计: {} 只股票，路径: {}".format(len(all_syms), CACHE_DIR))


if __name__ == "__main__":
    main()
