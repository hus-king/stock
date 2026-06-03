# -*- coding: utf-8 -*-
"""
@File    : report.py
@Date    : 2026-06-01
"""
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.box import SQUARE, SIMPLE


console = Console()


def _fmt(n):
    """Format a number with commas and 2 decimal places."""
    return "{:,.2f}".format(n)


def _fmt_pct(v):
    """Format a float as percentage."""
    return "{:.2%}".format(v)


class ReportGenerator:

    @staticmethod
    def print_summary(stats):
        """Print a formatted summary table."""
        table = Table(title=None, box=SQUARE, show_header=True,
                      header_style="bold", padding=(0, 1))
        table.add_column("指标", justify="left", no_wrap=True)
        table.add_column("数值", justify="right", no_wrap=True)

        table.add_row("总收益(元)", _fmt(stats.total_pnl))
        table.add_row("总收益率", _fmt_pct(stats.total_pnl_pct))
        table.add_row("T交易净贡献(元)", _fmt(stats.total_pnl - stats.buy_hold_pnl))
        table.add_row("买入持有收益(元)", _fmt(stats.buy_hold_pnl))
        table.add_row("买入持有收益率", _fmt_pct(stats.buy_hold_pnl_pct))
        table.add_row("交易天数", "{} / {}".format(stats.trading_days, stats.total_days))
        table.add_row("总成交笔数", "{}".format(stats.total_trades))
        table.add_row("  买入 / 卖出", "{} / {}".format(stats.buy_count, stats.sell_count))
        table.add_row("胜率(按日)", _fmt_pct(stats.day_win_rate))
        table.add_row("盈利天数 / 亏损天数", "{} / {}".format(stats.winning_days, stats.losing_days))
        table.add_row("平均日盈利(元)", _fmt(stats.avg_daily_pnl))
        table.add_row("最大日盈利(元)", _fmt(stats.best_day_pnl))
        table.add_row("最大日亏损(元)", _fmt(stats.worst_day_pnl))
        table.add_row("最大回撤(元)", _fmt(stats.max_drawdown))
        table.add_row("夏普比率", "{:.2f}".format(stats.sharpe_ratio))
        pf = "{:.2f}".format(stats.profit_factor) if stats.profit_factor != float("inf") else "N/A"
        table.add_row("盈亏比", pf)
        table.add_row("总佣金(元)", _fmt(stats.total_commission))
        ratio = stats.total_commission / stats.total_pnl if stats.total_pnl != 0 else 0
        table.add_row("佣金占收益比", _fmt_pct(ratio))

        console.print()
        console.print(table)

    @staticmethod
    def print_daily_table(daily_stats, tail=20):
        """Print recent daily P&L."""
        if not daily_stats:
            return
        df = pd.DataFrame(daily_stats[-tail:])

        table = Table(title=None, box=SIMPLE, show_header=True,
                      header_style="bold", padding=(0, 1))
        table.add_column("日期", justify="left", no_wrap=True)
        table.add_column("笔数", justify="right", no_wrap=True)
        table.add_column("日盈亏", justify="right", no_wrap=True)
        table.add_column("累计盈亏", justify="right", no_wrap=True)
        table.add_column("最高持仓", justify="right", no_wrap=True)

        # Start cumulative from all days before the visible tail window
        head = daily_stats[:-tail] if len(daily_stats) > tail else []
        cumulative = sum(d["daily_pnl"] for d in head)
        for _, row in df.iterrows():
            cumulative += row["daily_pnl"]
            max_pos = int(row.get("max_position", row["end_position"]))
            table.add_row(
                str(row["date"]),
                str(int(row["num_trades"])),
                _fmt(row["daily_pnl"]),
                _fmt(cumulative),
                "{:,}".format(max_pos),
            )

        console.print()
        console.print(table)

    @staticmethod
    def export_csv(daily_stats, path, trades=None):
        """Export daily stats (and optionally trades) to CSV."""
        df = pd.DataFrame(daily_stats)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print("Daily stats exported to {}".format(path))

        if trades:
            trades_df = pd.DataFrame([{
                "trade_id": t.trade_id,
                "side": t.side,
                "quantity": t.quantity,
                "price": t.price,
                "commission": t.commission,
                "timestamp": t.timestamp,
                "tag": t.tag,
            } for t in trades])
            trades_path = path.replace(".csv", "_trades.csv")
            trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
            print("Trades exported to {}".format(trades_path))
