# -*- coding:utf-8 -*-
# @Time : 2022/2/19 5:39 下午
# @Author : huichuan LI
# @File : bandit_test.py
# @Software: PyCharm
from Bandit.policy import EpsilonGreedy, UCB1, ThompsonSamplingBetaBinomial,LinUCB
from Bandit.bandit import BernoulliBandit,ContextualLinearBandit
import numpy as np
from Bandit.trainer import BanditTrainer


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
tom = ThompsonSamplingBetaBinomial()

BanditTrainer().compare([eg, ucb1, tom], bb, 10000, 5)



lb = LinUCB()
ct = ContextualLinearBandit(T,N)
BanditTrainer().compare([lb], ct, 10000, 5)
