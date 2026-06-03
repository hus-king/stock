#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量下载沪深300成分股前50只的5分钟K线数据并缓存到 data_cache/

用法:
    venv/bin/python rl/download_data.py
    venv/bin/python rl/download_data.py --count 50 --start 2026-01-01 --end 2026-05-30
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stock_open_api.simulation.data import MinuteDataLoader

START_DATE = "2026-01-01"
END_DATE = "2026-05-30"
PERIOD = "5"
ADJUST = "qfq"
DEFAULT_COUNT = 50


def get_hs300_symbols():
    """获取沪深300成分股列表。"""
    try:
        import akshare as ak
        df = ak.index_stock_cons_weight_csindex(symbol="000300")
        if df is not None and not df.empty:
            col = "成分券代码" if "成分券代码" in df.columns else df.columns[0]
            symbols = df[col].astype(str).str.zfill(6).tolist()
            print("从 csindex 获取到 {} 只沪深300成分股".format(len(symbols)))
            return symbols
    except Exception as e:
        print("csindex 获取失败: {}，尝试备用接口...".format(e))

    try:
        import akshare as ak
        df = ak.index_stock_cons(symbol="000300")
        if df is not None and not df.empty:
            col = "品种代码" if "品种代码" in df.columns else df.columns[0]
            symbols = df[col].astype(str).str.zfill(6).tolist()
            print("从备用接口获取到 {} 只沪深300成分股".format(len(symbols)))
            return symbols
    except Exception as e:
        print("备用接口也失败: {}".format(e))

    # 硬编码部分常见沪深300成分股作为兜底
    fallback = [
        "600519", "000858", "601318", "600036", "000333",
        "600900", "601166", "000651", "600276", "601888",
        "000725", "600926", "000768", "600028", "600000",
        "601398", "601288", "601939", "601988", "600016",
        "000001", "002415", "600887", "601012", "000002",
        "600031", "601601", "000063", "002594", "600309",
        "601628", "600048", "000776", "601919", "600690",
        "002714", "601668", "000100", "600585", "601186",
        "600104", "000538", "601088", "600050", "002304",
        "601766", "600196", "000播", "601390", "600030",
    ]
    # 过滤掉非法代码
    fallback = [s for s in fallback if s.isdigit() and len(s) == 6]
    print("使用内置兜底列表，共 {} 只".format(len(fallback)))
    return fallback


def download_all(symbols, start_date, end_date, period, count):
    loader = MinuteDataLoader(cache_dir="data_cache")
    symbols = symbols[:count]
    success, failed = [], []

    for i, sym in enumerate(symbols):
        print("[{}/{}] 下载 {} ...".format(i + 1, len(symbols), sym), end=" ", flush=True)
        try:
            df = loader.load(sym, start_date, end_date, period, ADJUST)
            days = len(df["date"].unique()) if "date" in df.columns else "?"
            print("OK ({} 天, {} 条)".format(days, len(df)))
            success.append(sym)
        except Exception as e:
            print("FAIL: {}".format(e))
            failed.append(sym)
        # 避免请求过快被限流
        time.sleep(0.5)

    print("\n完成: {} 成功, {} 失败".format(len(success), len(failed)))
    if failed:
        print("失败列表:", failed)
    return success, failed


def parse_args():
    parser = argparse.ArgumentParser(description="批量下载沪深300数据")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help="下载前N只 (默认: {})".format(DEFAULT_COUNT))
    parser.add_argument("--start", default=START_DATE, help="开始日期")
    parser.add_argument("--end", default=END_DATE, help="结束日期")
    parser.add_argument("--period", default=PERIOD, help="K线周期(分钟)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("获取沪深300成分股列表...")
    symbols = get_hs300_symbols()
    print("准备下载前 {} 只，日期范围 {} ~ {}，{}分钟K线\n".format(
        args.count, args.start, args.end, args.period))
    download_all(symbols, args.start, args.end, args.period, args.count)
