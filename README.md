# stock

A股日内做T量化回测系统，支持网格策略、ML择日和参数自适应。

## 功能

- **网格做T回测**：基于5分钟K线，模拟日内网格交易策略
- **ML择日**：用早盘30分钟特征（含跨日）训练 XGBoost 分类器，预测当天是否适合做T
- **盘中自适应**：网格中心跟随日内VWAP动态移动
- **动态仓位**：根据早盘振幅自动调整每档买卖手数
- **策略对比**：批量对比买入持有 / 无ML做T / ML择日做T 三种策略

## 数据来源

| 数据 | 来源 | 说明 |
|------|------|------|
| 5分钟K线 | 新浪财经 | 近60天真实数据 |
| 日K线 | BaoStock | 近3年，沪深300全量 |
| 沪深300成分股 | BaoStock / csindex | 实时获取 |

## 目录结构

```
stock/
├── intraday_t_trading.py      # 回测主程序
├── etf_momentum_rotation.py   # ETF动量轮动策略
├── stock_open_api/
│   ├── simulation/            # 回测引擎
│   │   ├── strategy.py        # 网格策略（含ML择日、自适应）
│   │   ├── engine.py          # 模拟引擎
│   │   ├── data.py            # 数据加载（多源，含缓存）
│   │   ├── account.py         # 账户状态
│   │   ├── metrics.py         # 指标计算
│   │   ├── report.py          # 报表输出
│   │   └── visualize.py       # 图表
│   └── api/                   # 原始股票数据接口
├── rl/
│   ├── ml_selector.py         # ML择日模型训练
│   ├── compare_strategies.py  # 三策略批量对比
│   ├── download_data.py       # 新浪5分钟数据下载
│   ├── download_daily_baostock.py  # BaoStock日K线下载
│   ├── trading_env.py         # Gymnasium强化学习环境
│   ├── train.py               # PPO训练脚本
│   └── models/
│       └── ml_selector.pkl    # 已训练的择日模型
├── data_cache/
│   ├── daily/                 # 日K线缓存（parquet）
│   └── min_*.parquet          # 分钟K线缓存
└── output/                    # 回测输出（图表、CSV）
```

## 安装

```bash
git clone https://github.com/your-username/stock.git
cd stock
python3 -m venv venv
venv/bin/pip install akshare baostock xgboost scikit-learn \
    gymnasium stable-baselines3 joblib matplotlib pandas pyarrow
```

## 快速开始

### 单只股票回测

```bash
# 基本回测（自动加载ML择日模型）
venv/bin/python intraday_t_trading.py --symbol 600926 --start 2026-04-01 --end 2026-06-01

# 关闭ML择日
venv/bin/python intraday_t_trading.py --symbol 600926 --ml-selector none

# 调整网格参数
venv/bin/python intraday_t_trading.py --symbol 000858 --grid-spacing 0.02 --lot-size 200
```

### 下载数据

```bash
# 下载沪深300前50只近60天5分钟数据
venv/bin/python rl/download_data.py --count 50 --start 2026-04-01 --end 2026-06-01

# 下载沪深300全量近3年日K线
venv/bin/python rl/download_daily_baostock.py
```

### 训练ML择日模型

```bash
venv/bin/python rl/ml_selector.py
```

### 三策略批量对比

```bash
venv/bin/python rl/compare_strategies.py
```

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbol` | 000001 | 股票代码 |
| `--start` / `--end` | 2026-01-01 / 2026-05-30 | 回测区间 |
| `--grid-spacing` | 1.5% | 网格间距 |
| `--grid-levels` | 3 | 上下档数 |
| `--lot-size` | 300 | 每档股数 |
| `--base-shares` | 5000 | 底仓股数 |
| `--daily-loss-limit` | -1000 | 日内止损（元） |
| `--ml-selector` | auto | ML择日：auto / none / 路径 |

## 回测结果示例（55只沪深300，2026-04-01~06-01）

| 策略 | T净贡献合计 | 为正股票数 |
|------|------------|-----------|
| 买入持有 | -116.5万 | — |
| 全天做T | +96.6万 | 28/55 |
| ML择日做T | +92.4万 | 31/55 |

> T净贡献 = 做T策略相比纯持仓的额外收益。
