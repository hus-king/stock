# -*- coding: utf-8 -*-
"""
@File    : data.py
@Date    : 2026-06-01
"""
import os
import hashlib
import time

import numpy as np
import pandas as pd
import requests

COLUMN_MAP_EM = {
    "时间": "timestamp",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover_rate",
}

COLUMN_MAP_SINA = {
    "day": "timestamp",
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "amount": "amount",
}

EM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
}


def _retry(func, retries=3, delay=2.0):
    for attempt in range(retries):
        try:
            return func()
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise


class MinuteDataLoader:
    """Downloads minute-level K-line data with local cache.

    Tries sources in order:
    1. Eastmoney minute K-line (preferred, but may be blocked)
    2. Sina minute data (recent data only)
    3. Falls back to generating synthetic intraday bars from daily data
    """

    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir

    def _load_from_cache(self, cache_path):
        """Load cached parquet, handling both index-as-timestamp and column formats."""
        if not os.path.exists(cache_path):
            return None
        df = pd.read_parquet(cache_path)
        if df.index.name == "timestamp" or (
                "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex)):
            df = df.reset_index()
            if "index" in df.columns and "timestamp" not in df.columns:
                df = df.rename(columns={"index": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "date" not in df.columns:
            df["date"] = df["timestamp"].dt.date
        return df.set_index("timestamp")

    def _cache_path(self, symbol, start_date, end_date, period, source):
        key = "{}|{}|{}|{}|{}".format(symbol, start_date, end_date, period, source)
        h = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, "min_{}_{}.parquet".format(symbol, h[:8]))

    def _expected_trading_days(self, start_date, end_date):
        """Rough estimate of trading days in a date range."""
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        return max(1, int(days * 5 / 7))

    def _has_sufficient_coverage(self, df, start_date, end_date, min_ratio=0.6):
        """Check whether data covers enough of the requested date range."""
        dates = df["date"] if "date" in df.columns else df.index.normalize()
        unique_days = len(pd.Index(dates).unique())
        expected = self._expected_trading_days(start_date, end_date)
        return unique_days >= expected * min_ratio

    def load(self, symbol, start_date, end_date, period="5",
             adjust="qfq") -> pd.DataFrame:
        """Download minute bars, falling back to synthetic generation.

        Returns DataFrame indexed by timestamp with columns:
        open, high, low, close, volume, date
        """
        period_int = int(period)
        os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Try Eastmoney trends2 (real 1-min bars, ~5 recent days)
        #    Only usable when the requested range fits within its short window.
        trends2_ok = False
        if self._expected_trading_days(start_date, end_date) <= 7:
            cache_path = self._cache_path(symbol, start_date, end_date, period, "trends2")
            result = self._load_from_cache(cache_path)
            if result is not None:
                return result

            df = self._fetch_eastmoney_trends2(symbol, period_int)
            if df is not None and len(df) > 0:
                df = df[df["date"].between(
                    pd.to_datetime(start_date).date(),
                    pd.to_datetime(end_date).date())]
                if len(df) > 0 and self._has_sufficient_coverage(df, start_date, end_date):
                    df = df.set_index("timestamp").sort_index()
                    df.to_parquet(cache_path)
                    return df
                trends2_ok = False

        # 2. Try Eastmoney kline (historical minute data)
        cache_path = self._cache_path(symbol, start_date, end_date, period, "em")
        result = self._load_from_cache(cache_path)
        if result is not None:
            return result

        df = self._fetch_eastmoney_minute(symbol, start_date, end_date, period_int, adjust)
        if df is not None and len(df) > 0:
            df["date"] = df["timestamp"].dt.date
            df = df.set_index("timestamp")
            df.to_parquet(cache_path)
            return df

        # 3. Try Tencent minute data (recent ~8 trading days, good for short ranges)
        cache_path = self._cache_path(symbol, start_date, end_date, period, "tencent")
        result = self._load_from_cache(cache_path)
        if result is not None:
            return result

        if self._expected_trading_days(start_date, end_date) <= 10:
            df = self._fetch_tencent_minute(symbol, period_int)
            if df is not None and len(df) > 0:
                df = df[df["date"].between(
                    pd.to_datetime(start_date).date(),
                    pd.to_datetime(end_date).date())]
                if len(df) > 0 and self._has_sufficient_coverage(df, start_date, end_date):
                    df = df.set_index("timestamp").sort_index()
                    df.to_parquet(cache_path)
                    return df

        # 4. Try Sina minute data (recent ~2-3 months)
        cache_path = self._cache_path(symbol, start_date, end_date, period, "sina")
        result = self._load_from_cache(cache_path)
        if result is not None:
            return result

        df = self._fetch_sina_minute(symbol, period_int)
        if df is not None and len(df) > 0:
            df = df[df["date"].between(
                pd.to_datetime(start_date).date(),
                pd.to_datetime(end_date).date())]
            if len(df) > 0 and self._has_sufficient_coverage(df, start_date, end_date):
                df = df.set_index("timestamp").sort_index()
                df.to_parquet(cache_path)
                return df

        # 5. Fall back to synthetic from daily data
        cache_path = self._cache_path(symbol, start_date, end_date, period, "synth")
        result = self._load_from_cache(cache_path)
        if result is not None:
            return result

        print("Falling back to synthetic intraday bars from daily data...")
        daily_df = self._fetch_daily(symbol, start_date, end_date, adjust)
        if daily_df is None or daily_df.empty:
            raise RuntimeError(
                "No data for symbol={}, range={}~{}".format(symbol, start_date, end_date))
        df = self._synthesize_intraday(daily_df, period_int)
        df["date"] = df["timestamp"].dt.date
        df = df.set_index("timestamp")
        df.to_parquet(cache_path)
        return df

    # ============================================================
    # Eastmoney minute data
    # ============================================================

    def _fetch_eastmoney_minute(self, symbol, start_date, end_date, period, adjust):
        """Try akshare's Eastmoney minute function."""
        try:
            import akshare as ak
            raw = _retry(lambda: ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                start_date="{} 09:30:00".format(start_date),
                end_date="{} 15:00:00".format(end_date),
                period=str(period),
                adjust=adjust,
            ), retries=1, delay=1.0)
            if raw is None or raw.empty:
                return None
            df = raw.rename(columns=COLUMN_MAP_EM)
            keep = ["timestamp", "open", "high", "low", "close", "volume", "amount"]
            df = df[[c for c in keep if c in df.columns]]
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df.sort_values("timestamp")
        except Exception:
            return None

    # ============================================================
    # Eastmoney trends2 (real 1-min bars, last ~5 trading days)
    # ============================================================

    def _fetch_eastmoney_trends2(self, symbol, target_period):
        """Download 1-minute bars via Eastmoney trends2 API.

        This endpoint is more accessible than the kline API and returns
        real intraday 1-minute data for the last few trading days.
        Resamples to target_period if needed.
        """
        market = 0 if symbol.startswith(("0", "3")) else 1
        secid = "{}.{}".format(market, symbol)

        params = {
            "secid": secid,
            "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
            "iscr": "0",
            "ndays": "5",  # Max reliable range
        }

        def _get():
            r = requests.get(
                "https://push2his.eastmoney.com/api/qt/stock/trends2/get",
                headers=EM_HEADERS, params=params, timeout=30)
            r.raise_for_status()
            return r.json()

        try:
            data = _retry(_get, retries=3, delay=2.0)
        except Exception:
            return None

        trends = data.get("data", {}).get("trends", [])
        if not trends:
            return None

        records = []
        for line in trends:
            parts = line.split(",")
            if len(parts) < 8:
                continue
            # f51=time, f52=price, f53=avg, f54=price, f55=price, f56=vol, f57=amount
            ts = pd.to_datetime(parts[0])
            price = float(parts[1])
            vol = float(parts[5]) if len(parts) > 5 else 0
            amt = float(parts[6]) if len(parts) > 6 else 0
            records.append({
                "timestamp": ts,
                "date": ts.date(),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": vol,
                "amount": amt,
            })

        if not records:
            return None

        df = pd.DataFrame(records).sort_values("timestamp")

        if target_period > 1:
            df = df.set_index("timestamp")
            df = df.resample("{}min".format(target_period)).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
                "date": "first",
            }).dropna(subset=["open"])
            df = df.reset_index()

        return df

    # ============================================================
    # Sina minute data (recent 2-3 months only)
    # ============================================================

    def _to_sina_symbol(self, symbol):
        """Convert 6-digit code to Sina format: sz000001 / sh600519."""
        if symbol.startswith(("6", "9")):
            return "sh{}".format(symbol)
        return "sz{}".format(symbol)

    def _fetch_tencent_minute(self, symbol, period):
        """Download recent minute data from Tencent Finance API.

        Returns up to ~320 bars (about 7 trading days) of 5-min data.
        Format: [timestamp, open, close, high, low, volume, {}, turnover]
        """
        sina_sym = self._to_sina_symbol(symbol)
        period_key = "m{}".format(period)
        url = (
            "https://ifzq.gtimg.cn/appstock/app/kline/mkline"
            "?param={sym},{key},,320&_var={key}_data".format(
                sym=sina_sym, key=period_key)
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://gu.qq.com",
        }
        try:
            def _get():
                r = requests.get(url, headers=headers, timeout=15)
                r.raise_for_status()
                import json
                raw = json.loads(r.text.split("=", 1)[1])
                return raw["data"][sina_sym].get(period_key, [])

            records = _retry(_get, retries=2, delay=2.0)
            if not records:
                return None

            rows = []
            for item in records:
                # [时间戳, 开, 收, 高, 低, 量, {}, 换手率]
                ts_str = str(item[0])  # "202605221425"
                ts = pd.to_datetime(ts_str, format="%Y%m%d%H%M")
                rows.append({
                    "timestamp": ts,
                    "date": ts.date(),
                    "open": float(item[1]),
                    "close": float(item[2]),
                    "high": float(item[3]),
                    "low": float(item[4]),
                    "volume": float(item[5]),
                    "amount": float(item[5]) * float(item[2]),
                })

            df = pd.DataFrame(rows)
            df = df.dropna(subset=["open", "high", "low", "close"])
            df = df[df["close"] > 0]
            return df.sort_values("timestamp")
        except Exception:
            return None

    def _fetch_sina_minute(self, symbol, period):
        """Download recent minute data directly from Sina API."""
        sina_sym = self._to_sina_symbol(symbol)
        url = (
            "https://quotes.sina.cn/cn/api/jsonp_v2.php/"
            "var%20_{sym}_{period}_1=/"
            "CN_MarketDataService.getKLineData"
            "?symbol={sym}&scale={period}&ma=no&datalen=1023"
        ).format(sym=sina_sym, period=period)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://finance.sina.com.cn",
        }
        try:
            def _get():
                r = requests.get(url, headers=headers, timeout=15)
                r.raise_for_status()
                text = r.text
                # Format: var _sz000001_5_1=([...]);
                start = text.index("=[") + 1
                end = text.rindex("]") + 1
                import json
                return json.loads(text[start:end])

            records = _retry(_get, retries=2, delay=3.0)
            if not records:
                return None

            rows = []
            for item in records:
                ts = pd.to_datetime(item.get("d") or item.get("day"))
                rows.append({
                    "timestamp": ts,
                    "date": ts.date(),
                    "open": float(item.get("o") or item.get("open") or 0),
                    "high": float(item.get("h") or item.get("high") or 0),
                    "low": float(item.get("l") or item.get("low") or 0),
                    "close": float(item.get("c") or item.get("close") or 0),
                    "volume": float(item.get("v") or item.get("volume") or 0),
                })

            df = pd.DataFrame(rows)
            df = df.dropna(subset=["open", "high", "low", "close"])
            df = df[(df["close"] > 0)]
            df["amount"] = df["volume"] * df["close"]
            return df.sort_values("timestamp")
        except Exception:
            return None

    # ============================================================
    # Daily data for synthetic generation
    # ============================================================

    def _fetch_daily(self, symbol, start_date, end_date, adjust):
        """Download daily K-line, trying multiple sources."""
        import akshare as ak

        # 1. Try Sina daily (uses sz/sh prefix)
        sina_sym = self._to_sina_symbol(symbol)
        try:
            def _get_sina():
                return ak.stock_zh_a_daily(
                    symbol=sina_sym,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust=adjust,
                )
            raw = _retry(_get_sina, retries=3, delay=2.0)
            if raw is not None and not raw.empty and "date" in raw.columns:
                keep = ["date", "open", "high", "low", "close", "volume", "amount"]
                df = raw[[c for c in keep if c in raw.columns]].copy()
                df["date"] = pd.to_datetime(df["date"])
                for c in ["open", "close", "high", "low", "volume", "amount"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["open", "high", "low", "close"])
                if len(df) > 0:
                    return df.sort_values("date")
        except Exception:
            pass

        # 2. Try Eastmoney daily via akshare
        try:
            def _get_em():
                return ak.stock_zh_a_hist(
                    symbol=symbol, period="daily",
                    start_date=start_date, end_date=end_date, adjust=adjust,
                )
            raw = _retry(_get_em, retries=3, delay=2.0)
            if raw is not None and not raw.empty:
                col_map = {
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low",
                    "成交量": "volume", "成交额": "amount",
                }
                df = raw.rename(columns=col_map)
                keep = ["date", "open", "close", "high", "low", "volume", "amount"]
                df = df[[c for c in keep if c in df.columns]]
                df["date"] = pd.to_datetime(df["date"])
                for c in ["open", "close", "high", "low", "volume", "amount"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["open", "high", "low", "close"])
                if len(df) > 0:
                    return df.sort_values("date")
        except Exception:
            pass

        return None

    # ============================================================
    # Synthetic intraday bar generation
    # ============================================================

    def _synthesize_intraday(self, daily_df, period_minutes):
        """Generate synthetic minute bars from daily OHLCV.

        Uses oscillatory price paths (sine-based with noise) that respect
        daily OHLC, producing realistic intraday swings for grid trading.
        """
        bars_per_day = 48  # 5-min bars for 4-hour session
        all_bars = []
        rng = np.random.RandomState(42)

        for _, row in daily_df.iterrows():
            day = row["date"]
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            vol = float(row.get("volume", 0)) if pd.notna(row.get("volume", 0)) else 0

            if o <= 0 or h <= l:
                continue

            midpoint = (h + l) / 2.0
            amplitude = (h - l) / 2.0
            if amplitude <= 0:
                amplitude = o * 0.005

            # Multi-frequency oscillation for realistic intraday patterns
            # Main oscillation: 2-3 full cycles during the session
            # Add harmonic + noise for natural look
            times = []
            prices = []
            for i in range(bars_per_day):
                minutes = 9 * 60 + 30 + i * period_minutes + period_minutes
                hh, mm = divmod(minutes, 60)
                ts = pd.Timestamp("{} {:02d}:{:02d}:00".format(
                    day.strftime("%Y-%m-%d"), int(hh), int(mm)))
                times.append(ts)

                t = i / (bars_per_day - 1)  # 0 to 1

                # Trend: open -> close over the day
                trend = o + (c - o) * t

                # Oscillation around midpoint: 3 cycles during the session
                # Phase offset varies per day for variety
                phase = rng.random() * 2 * np.pi
                freq = 3.0 + rng.random()  # 3-4 cycles per day
                oscillation = amplitude * np.sin(2 * np.pi * freq * t + phase)

                # Add minor high-frequency component
                hf = amplitude * 0.15 * np.sin(2 * np.pi * 12 * t + phase * 1.7)

                # Noise
                noise_scale = amplitude * 0.08
                noise = noise_scale * rng.randn()

                raw_price = trend + oscillation * 0.7 + hf + noise
                raw_price = max(l * 0.998, min(h * 1.002, raw_price))

                prices.append(raw_price)

            # Ensure path touches daily high and low at least once
            prices_array = np.array(prices)
            if prices_array.max() < h * 0.995:
                boost_idx = len(prices) // 3 + rng.randint(0, len(prices) // 3)
                prices_array[boost_idx] = h
            if prices_array.min() > l * 1.005:
                dip_idx = 2 * len(prices) // 3 + rng.randint(0, len(prices) // 3)
                prices_array[dip_idx] = l

            # Force close to match
            prices_array[-1] = c

            day_vol_per_bar = vol / bars_per_day if vol > 0 else 1e6

            for i, ts in enumerate(times):
                p_open = prices_array[i]
                if i < len(prices_array) - 1:
                    p_close = prices_array[i + 1]
                else:
                    p_close = p_open
                seg = prices_array[max(0, i - 1):min(i + 2, len(prices_array))]
                bar_high = max(seg)
                bar_low = min(seg)
                bar_vol = day_vol_per_bar * (0.5 + rng.random())
                all_bars.append({
                    "timestamp": ts,
                    "open": round(p_open, 2),
                    "high": round(bar_high, 2),
                    "low": round(bar_low, 2),
                    "close": round(p_close, 2),
                    "volume": bar_vol,
                    "amount": bar_vol * (p_open + p_close) / 2,
                })

        df = pd.DataFrame(all_bars)
        return df.sort_values("timestamp")


def get_stock_pool_from_spot(min_price=5, max_price=200, min_turnover=1.0):
    """Screen A-share stocks suitable for intraday T-trading."""
    import akshare as ak
    raw = _retry(lambda: ak.stock_zh_a_spot_em(), retries=1, delay=1.0)
    if raw is None or raw.empty:
        return []

    df = raw.copy()
    df = df[~df["名称"].str.contains("ST|PT|退", na=False)]

    price_col = "最新价" if "最新价" in df.columns else "最新"
    turnover_col = "换手率" if "换手率" in df.columns else "换手"

    if price_col not in df.columns or turnover_col not in df.columns:
        return []

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[turnover_col] = pd.to_numeric(df[turnover_col], errors="coerce")

    mask = (
        (df[price_col] >= min_price) &
        (df[price_col] <= max_price) &
        (df[turnover_col] >= min_turnover)
    )
    df = df[mask].sort_values(turnover_col, ascending=False)
    return df["代码"].tolist()


_STOCK_NAME_CACHE = {}


def get_stock_name(symbol):
    """Look up the stock name for a given 6-digit symbol via Sina."""
    if symbol in _STOCK_NAME_CACHE:
        return _STOCK_NAME_CACHE[symbol]

    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    sina_sym = "{}{}".format(prefix, symbol)

    try:
        r = requests.get(
            "https://hq.sinajs.cn/list={}".format(sina_sym),
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://finance.sina.com.cn",
            },
            timeout=10,
        )
        r.encoding = "gbk"
        text = r.text
        # Format: var hq_str_sh600028="name,...";
        if '"' in text:
            name = text.split('"')[1].split(",")[0]
            if name and name != symbol:
                _STOCK_NAME_CACHE[symbol] = name
                return name
    except Exception:
        pass

    _STOCK_NAME_CACHE[symbol] = symbol
    return symbol
