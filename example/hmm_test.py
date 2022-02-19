# -*- coding:utf-8 -*-
# @Time : 2022/2/19 12:15 上午
# @Author : huichuan LI
# @File : hmm_test.py
# @Software: PyCharm
import numpy as np
from prob.HMM import MultinomialHMM

pi = np.array([0.25, 0.25, 0.25, 0.25])
# 状态转移概率矩阵
A = np.array([
    [0, 1, 0, 0],
    [0.4, 0, 0.6, 0],
    [0, 0.4, 0, 0.6],
    [0, 0, 0.5, 0.5]])
# 观测概率矩阵
B = np.array([
    [0.5, 0.5],
    [0.6, 0.4],
    [0.2, 0.8],
    [0.3, 0.7]])
# 可能的状态数和观测数
N = 4
M = 2

hmm = MultinomialHMM(A=A, B=B, pi=pi, eps=0)

print(hmm.generate(5, [3, 2, 1, 0], [0, 1]))

print(hmm.log_likelihood(np.array([1, 0, 1, 0, 0])))

# 给定观测序列
O = np.array([1, 0, 1, 1, 0])
# 输出最可能的隐状态序列
print(hmm.decode(O))
