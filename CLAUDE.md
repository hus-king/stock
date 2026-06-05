# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All scripts must run with the project venv. The entry-point scripts auto-exec into venv if invoked from system Python, but always prefer:

```bash
venv/bin/python <script>
```

Install dependencies:

```bash
python3 -m venv venv
venv/bin/pip install akshare baostock xgboost scikit-learn \
    gymnasium stable-baselines3 joblib matplotlib pandas pyarrow backtrader
```

## Common Commands

```bash
# Single-stock backtest (auto-loads ML selector if model exists)
venv/bin/python intraday_t_trading.py --symbol 000858 --start 2026-04-01 --end 2026-06-01

# Disable ML selector
venv/bin/python intraday_t_trading.py --symbol 000858 --ml-selector none

# Batch 3-strategy comparison over all cached stocks
venv/bin/python rl/compare_strategies.py

# Download 5-min data for top-N HS300 stocks
venv/bin/python rl/download_data.py --count 50 --start 2026-04-01 --end 2026-06-01

# Download daily K-lines (BaoStock, full HS300, ~3 years)
venv/bin/python rl/download_daily_baostock.py

# Train ML day-selection model (fast: uses cached stocks only)
venv/bin/python rl/train_selector.py

# Train ML model (full pipeline: screening + training, 300 stocks)
venv/bin/python rl/ml_selector.py

# Monthly 3-strategy comparison (all 300 stocks × 30 months)
venv/bin/python rl/compare_monthly.py
venv/bin/python rl/compare_monthly.py --rerun --threshold 0.5

# ETF momentum rotation backtest (V1 vs V2 comparison)
venv/bin/python etf_momentum_rotation.py

# Run tests
venv/bin/python -m pytest tests/
venv/bin/python -m pytest tests/test_hello.py
```

## Architecture

### Intraday T-Trading System (`intraday_t_trading.py` + `stock_open_api/simulation/`)

The core backtest pipeline has four layers:

1. **Data** (`simulation/data.py`) — `MinuteDataLoader` fetches 5-min K-lines, trying Eastmoney first then Sina as fallback, with Parquet cache in `data_cache/`. Cache filenames encode symbol + date-range hash.

2. **Strategy** (`simulation/strategy.py`) — `BaseStrategy` ABC with lifecycle hooks: `prepare(data)` → `on_day_start` → `on_bar` → `on_fill` → `on_day_end`. `GridStrategy` is the only concrete implementation; registered in `STRATEGY_REGISTRY`. The strategy emits `Signal` objects (from `order.py`).

3. **Engine** (`simulation/engine.py`) — `SimulationEngine` iterates bars grouped by day, executes signals via `AccountState`, enforces constraints (daily loss limit, net-buy cap), and force-closes back to `base_shares` at end of each day.

4. **Reporting** (`simulation/report.py`, `simulation/visualize.py`, `simulation/metrics.py`) — `ReportGenerator` prints summary + daily table and exports CSV; `TradingChart` saves PNG to `output/intraday_t/`.

**ML day-selection** (`rl/ml_selector.py` + `rl/train_selector.py`): an XGBoost classifier with 28 features — 10 morning-bar features (first 6×5min bars), 8 cross-day features (prior 5 days), 10 daily K-line features (amplitude, turnover, PE/PB rank, etc.). Labels are binary: `t_pnl > 50` (positive, "trade") or `t_pnl < -50` (negative, "skip"), with `|t_pnl| <= 50` excluded as noise (~77% of days). Training uses 5-fold expanding-window time-series CV. The bundle (`model`, `scaler`, `features`) lives at `rl/models/ml_selector.pkl`. When active, it replaces the trend-filter.

**T-net contribution (t_pnl)**: defined as `daily_pnl - (close - open) × base_shares` summed per day. This isolates intraday T-trading skill from overnight gaps — the period-level `total_pnl - buy_hold_pnl` would include overnight gaps which T-trading cannot capture.

**Batch comparison** (`rl/compare_strategies.py`): single-period 3-strategy comparison over all cached stocks. `rl/compare_monthly.py` does the same but month-by-month with checkpoint saving, enabling trend analysis across 30 months.

### ETF Momentum Rotation (`etf_momentum_rotation.py`)

Standalone backtrader script. Downloads ETF daily data via akshare, runs V1 (original) and V2 (improved) strategies back-to-back, prints a comparison table, and saves a chart to `etf_momentum_result.png`. V2 adds MA slope filter, rebalance threshold (3%), and drawdown stop-loss vs V1.

### Data Flow

```
download_data.py / download_daily_baostock.py
        ↓
data_cache/min_*.parquet   data_cache/daily/
        ↓
MinuteDataLoader (with cache)
        ↓
SimulationEngine.run(data, strategy)
        ↓
output/intraday_t/  (PNG + CSV)
```

### Key Configuration Constants

Default backtest parameters live at the top of `intraday_t_trading.py` (grid spacing 1.5%, 3 levels, 300 shares/level, 5000 base shares, ¥100k cash, commission 万三, stamp tax 0.1%). All are overridable via CLI flags.
