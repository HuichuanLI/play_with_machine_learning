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

hmm = MultinomialHMM(A, B, pi, eps=1e-9)
