# -*- coding:utf-8 -*-
# @Time : 2022/2/15 11:35 下午
# @Author : huichuan LI
# @File : EM.py
# @Software: PyCharm
# 导入numpy库
import numpy as np


### EM算法过程函数定义
def em(data, thetas, max_iter=30, eps=1e-3):
    '''
    输入：
    data：观测数据
    thetas：初始化的估计参数值
    max_iter：最大迭代次数
    eps：收敛阈值
    输出：
    thetas：估计参数
    '''
    # 初始化似然函数值
    ll_old = -np.infty
    for i in range(max_iter):
        ### E步：求隐变量分布
        # 对数似然
        log_like = np.array([np.sum(data * np.log(theta), axis=1) for theta in thetas])
        # 似然
        like = np.exp(log_like)
        # 求隐变量分布
        ws = like / like.sum(0)
        # 概率加权
        vs = np.array([w[:, None] * data for w in ws])
        ### M步：更新参数值
        thetas = np.array([v.sum(0) / v.sum() for v in vs])
        # 更新似然函数
        ll_new = np.sum([w * l for w, l in zip(ws, log_like)])
        print("Iteration: %d" % (i + 1))
        print("theta_B = %.2f, theta_C = %.2f, ll = %.2f"
              % (thetas[0, 0], thetas[1, 0], ll_new))
        # 满足迭代条件即退出迭代
        if np.abs(ll_new - ll_old) < eps:
            break
        ll_old = ll_new
    return thetas


# 观测数据，5次独立试验，每次试验10次抛掷的正反次数
# 比如第一次试验为5次正面5次反面
observed_data = np.array([(5, 5), (9, 1), (8, 2), (4, 6), (7, 3)])
# 初始化参数值，即硬币B的正面概率为0.6，硬币C的正面概率为0.5
thetas = np.array([[0.6, 0.4], [0.5, 0.5]])
thetas = em(observed_data, thetas, max_iter=30, eps=1e-3)

print(thetas)