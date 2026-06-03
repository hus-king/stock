#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 Stable-Baselines3 PPO 训练做T强化学习 agent。

用法:
    venv/bin/python rl/train.py
    venv/bin/python rl/train.py --timesteps 500000 --symbols 000725 600926 000768 600900

流程:
    1. 加载已缓存的5分钟K线数据
    2. 按时间切分训练集(前80%)和测试集(后20%)
    3. 用 PPO 训练，每隔一段时间在测试集上评估
    4. 保存最优模型到 rl/models/
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Redirect to venv Python if needed
_VENV_PYTHON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "venv", "bin", "python")
if sys.executable != _VENV_PYTHON and os.path.exists(_VENV_PYTHON):
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from rl.trading_env import TradingEnv, load_daily_data

# 默认使用已缓存的股票
DEFAULT_SYMBOLS = None  # None = auto-detect from data_cache

START_DATE = "2026-04-01"
END_DATE = "2026-06-01"
PERIOD = "5"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


class RewardLogCallback(BaseCallback):
    """每隔 N 步打印平均 episode reward。"""

    def __init__(self, log_interval=10000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
        if self.num_timesteps % self.log_interval == 0 and self._episode_rewards:
            mean_r = np.mean(self._episode_rewards[-100:])
            print("  step={:,}  mean_ep_reward(last100)={:.2f}".format(
                self.num_timesteps, mean_r))
        return True


def split_by_date(daily_data, train_ratio=0.8):
    """按日期切分训练/测试集（时间序列不能随机打乱）。"""
    dates = sorted(set(d for _, d, _ in daily_data))
    cutoff_idx = int(len(dates) * train_ratio)
    cutoff_date = dates[cutoff_idx]
    train = [(s, d, b) for s, d, b in daily_data if d < cutoff_date]
    test = [(s, d, b) for s, d, b in daily_data if d >= cutoff_date]
    return train, test


def evaluate(model, test_data, n_episodes=200):
    """在测试集上评估模型，返回平均 episode reward。"""
    env = TradingEnv(test_data)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(int(action))
            ep_reward += reward
        rewards.append(ep_reward)
    return np.mean(rewards), np.std(rewards)


def parse_args():
    parser = argparse.ArgumentParser(description="PPO 做T策略训练")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="股票代码列表")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="训练总步数 (默认: 200000)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="并行环境数 (默认: 4)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="学习率 (默认: 3e-4)")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. 加载数据：自动检测 data_cache 里所有股票
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")
    if args.symbols is None:
        import glob
        parquet_files = glob.glob(os.path.join(cache_dir, "min_*.parquet"))
        symbols = list(set(os.path.basename(f).split("_")[1] for f in parquet_files))
        symbols.sort()
    else:
        symbols = args.symbols
    print("加载数据: {} 只股票，{} ~ {}".format(len(symbols), args.start, args.end))
    daily_data = load_daily_data(symbols, args.start, args.end, PERIOD, cache_dir=cache_dir)

    if len(daily_data) < 20:
        print("数据不足（{}个交易日），请先运行 rl/download_data.py".format(len(daily_data)))
        return

    # 2. 切分训练/测试集
    train_data, test_data = split_by_date(daily_data, train_ratio=0.8)
    print("训练集: {} 天，测试集: {} 天".format(len(train_data), len(test_data)))

    # 3. 创建向量化环境
    def make_env():
        env = TradingEnv(train_data)
        return Monitor(env)

    vec_env = make_vec_env(make_env, n_envs=args.n_envs)

    # 4. 初始化 PPO
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,       # 增大探索系数，防止过早收敛到躺平
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=0,
    )

    # 5. 训练
    print("\n开始训练，总步数: {:,}，并行环境: {}\n".format(
        args.timesteps, args.n_envs))

    callbacks = [RewardLogCallback(log_interval=20000)]

    # 测试集评估回调
    if len(test_data) >= 10:
        eval_env = Monitor(TradingEnv(test_data))
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=MODEL_DIR,
            log_path=MODEL_DIR,
            eval_freq=max(args.timesteps // 10, 10000),
            n_eval_episodes=100,
            deterministic=True,
            verbose=1,
        )
        callbacks.append(eval_cb)

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    # 6. 保存最终模型
    final_path = os.path.join(MODEL_DIR, "ppo_trading_final")
    model.save(final_path)
    print("\n模型已保存到: {}.zip".format(final_path))

    # 7. 测试集评估
    if len(test_data) >= 10:
        print("\n测试集评估（n=200 episodes）...")
        mean_r, std_r = evaluate(model, test_data, n_episodes=200)
        print("平均 episode reward: {:.2f} ± {:.2f} 元".format(mean_r, std_r))

        # 对比：随机策略基线
        class RandomModel:
            def predict(self, obs, deterministic=False):
                return np.random.randint(0, 3), None

        rand_mean, rand_std = evaluate(RandomModel(), test_data, n_episodes=200)
        print("随机策略基线:        {:.2f} ± {:.2f} 元".format(rand_mean, rand_std))
        print("相对提升:            {:.2f} 元".format(mean_r - rand_mean))


if __name__ == "__main__":
    main()
