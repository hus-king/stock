# -*- coding: utf-8 -*-
"""
@File    : visualize.py
@Date    : 2026-06-01
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


# Chinese font setup
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class TradingChart:

    def plot(self, stats, daily_stats, trades, price_data, output_path,
             show_days=60):
        """Generate 4-panel report chart.

        Panel 1: Price + trade markers (last show_days)
        Panel 2: Daily intraday position (last show_days)
        Panel 3: Daily P&L bar chart (full period)
        Panel 4: Cumulative P&L + drawdown (full period)
        """
        if not daily_stats:
            print("No data to plot.")
            return

        daily_df = pd.DataFrame(daily_stats)
        daily_df["date"] = pd.to_datetime(daily_df["date"])

        trades_df = None
        if trades:
            trades_df = pd.DataFrame([{
                "timestamp": pd.to_datetime(t.timestamp),
                "side": t.side,
                "price": t.price,
                "quantity": t.quantity,
                "tag": t.tag,
            } for t in trades])

        fig, axes = plt.subplots(4, 1, figsize=(16, 18),
                                 gridspec_kw={"height_ratios": [2.5, 1.5, 1.5, 2]})
        fig.suptitle("Intraday T-Trading Simulation Report", fontsize=16, fontweight="bold")

        # Panel 1: Price + trades (zoom to last show_days)
        self._plot_price_trades(axes[0], price_data, trades_df, daily_df, show_days)

        # Panel 2: Position (last show_days)
        self._plot_position(axes[1], daily_df, show_days)

        # Panel 3: Daily P&L (full)
        self._plot_daily_pnl(axes[2], daily_df)

        # Panel 4: Cumulative P&L + drawdown (full)
        self._plot_cumulative(axes[3], daily_df, stats)

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Chart saved to {}".format(output_path))

    def _plot_price_trades(self, ax, price_data, trades_df, daily_df, show_days):
        # Get price data for last show_days
        last_dates = daily_df["date"].iloc[-show_days:]
        if "date" not in price_data.columns:
            price_data["date"] = price_data.index.date
        price_data["date"] = pd.to_datetime(price_data["date"])

        mask = price_data["date"].isin(last_dates)
        plot_data = price_data[mask]
        if plot_data.empty:
            ax.set_title("Price & Trades (no data)")
            return

        ax.plot(plot_data.index, plot_data["close"], color="steelblue",
                linewidth=0.8, alpha=0.8, label="Close")

        if trades_df is not None and not trades_df.empty:
            plot_start = plot_data.index.min()
            plot_end = plot_data.index.max()
            t_mask = (
                (trades_df["timestamp"] >= plot_start) &
                (trades_df["timestamp"] <= plot_end)
            )
            t_plot = trades_df[t_mask]

            buys = t_plot[t_plot["side"] == "BUY"]
            sells = t_plot[t_plot["side"] == "SELL"]
            closeouts = t_plot[t_plot["tag"] == "closeout"]

            if not buys.empty:
                ax.scatter(buys["timestamp"], buys["price"], marker="^",
                           color="green", s=60, alpha=0.8, zorder=5, label="Buy")
            if not sells.empty:
                ax.scatter(sells["timestamp"], sells["price"], marker="v",
                           color="red", s=60, alpha=0.8, zorder=5, label="Sell")
            if not closeouts.empty:
                ax.scatter(closeouts["timestamp"], closeouts["price"], marker=".",
                           color="gray", s=30, alpha=0.5, zorder=4, label="Closeout")

        ax.set_ylabel("Price (yuan)")
        ax.set_title("Price & Trade Markers (last {} days)".format(show_days))
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_position(self, ax, daily_df, show_days):
        last = daily_df.iloc[-show_days:]
        if last.empty:
            return

        ax.bar(last["date"], last["buy_volume"], color="green", alpha=0.6,
               label="Buy Volume")
        ax.bar(last["date"], [-v for v in last["sell_volume"]], color="red",
               alpha=0.6, label="Sell Volume")
        ax.set_ylabel("Volume (shares)")
        ax.set_title("Daily Buy/Sell Volume (last {} days)".format(show_days))
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)

    def _plot_daily_pnl(self, ax, daily_df):
        colors = ["green" if p > 0 else "red" for p in daily_df["daily_pnl"]]
        ax.bar(daily_df["date"], daily_df["daily_pnl"], color=colors, alpha=0.7,
               width=1.0)
        ax.set_ylabel("P&L (yuan)")
        ax.set_title("Daily P&L")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3)

    def _plot_cumulative(self, ax, daily_df, stats):
        cumulative = daily_df["daily_pnl"].cumsum()
        ax.plot(daily_df["date"], cumulative, color="steelblue",
                linewidth=1.5, label="Cumulative P&L")

        # Drawdown shading
        if len(cumulative) > 0:
            peak = np.maximum.accumulate(cumulative.values)
            drawdown = cumulative.values - peak
            ax.fill_between(daily_df["date"], cumulative, peak,
                            where=(cumulative < peak), color="red",
                            alpha=0.15, label="Drawdown")

        # Buy-hold comparison line
        bh_start = stats.buy_hold_pnl_pct * 0  # reference zero
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        ax.axhline(y=stats.buy_hold_pnl, color="orange", linewidth=0.8,
                   linestyle="--", label="Buy&Hold P&L ({:,.0f})".format(stats.buy_hold_pnl))

        ax.set_ylabel("P&L (yuan)")
        ax.set_title("Cumulative P&L & Drawdown")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
