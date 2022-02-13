# -*- coding:utf-8 -*-
# @Time : 2022/2/14 12:11 上午
# @Author : huichuan LI
# @File : MCMC.py
# @Software: PyCharm

### M-H采样
# 导入相关库
import random
from scipy.stats import norm
import matplotlib.pyplot as plt


# 定义平稳分布为正态分布
def smooth_dist(theta):
    '''
    输入：
    thetas：数组
    输出：
    y：正态分布概率密度函数
    '''
    y = norm.pdf(theta, loc=3, scale=2)
    return y


# 定义M-H采样函数
def MH_sample(T, sigma):
    '''
    输入：
    T：采样序列长度
    sigma：生成随机序列的尺度参数
    输出：
    pi：经M-H采样后的序列
    '''
    # 初始分布
    pi = [0 for i in range(T)]
    t = 0
    while t < T - 1:
        t = t + 1
        # 状态转移进行随机抽样
        pi_star = norm.rvs(loc=pi[t - 1], scale=sigma, size=1, random_state=None)
        alpha = min(1, (smooth_dist(pi_star[0]) / smooth_dist(pi[t - 1])))
        # 从均匀分布中随机抽取一个数u
        u = random.uniform(0, 1)
        # 拒绝-接受采样
        if u < alpha:
            pi[t] = pi_star[0]
        else:
            pi[t] = pi[t - 1]
    return pi


# 执行MH采样
pi = MH_sample(10000, 1)

### 绘制采样分布
# 绘制目标分布散点图
plt.scatter(pi, norm.pdf(pi, loc=3, scale=2), label='Target Distribution')
# 绘制采样分布直方图
plt.hist(pi,
         100,
         normed=1,
         facecolor='red',
         alpha=0.6,
         label='Samples Distribution')
plt.legend()
plt.show();