# -*- coding:utf-8 -*-
# @Time : 2022/2/19 5:39 下午
# @Author : huichuan LI
# @File : bandit_test.py
# @Software: PyCharm
from Bandit.policy import EpsilonGreedy, UCB1
from Bandit.bandit import BernoulliBandit
import numpy as np
from Bandit.trainer import BanditTrainer


def mse(bandit, policy):
    """
    Computes the mean squared error between a policy's estimates of the
    expected arm payouts and the true expected payouts.
    """
    if not hasattr(policy, "ev_estimates") or len(policy.ev_estimates) == 0:
        return np.nan

    se = []
    evs = bandit.arm_evs
    ests = sorted(policy.ev_estimates.items(), key=lambda x: x[0])
    for ix, (est, ev) in enumerate(zip(ests, evs)):
        se.append((est[1] - ev) ** 2)
    return np.mean(se)


# 定义 T = 1000 个用户，即总共进行1000次实现
T = 1000
# 定义 N = 10 个标签，即 N 个 物品
N = 10

# 保证结果可复现，设置随机数种子
np.random.seed(888)
# 每个物品的累积点击率（理论概率）
true_rewards = np.random.uniform(low=0, high=1, size=N)
# true_rewards = np.array([0.5] * N)
# 每个物品的当前点击率
now_rewards = np.zeros(N)
# 每个物品的点击次数
chosen_count = np.zeros(N)

total_reward = 0

bb = BernoulliBandit(true_rewards)

eg = EpsilonGreedy()
ucb1 = UCB1()

BanditTrainer().compare([eg, ucb1], bb, 10000, 5)
