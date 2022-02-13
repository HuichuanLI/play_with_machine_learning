# -*- coding:utf-8 -*-
# @Time : 2022/2/13 10:37 下午
# @Author : huichuan LI
# @File : max_entropy.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from collections import defaultdict


class MaxEnt:
    def __init__(self, max_iter=100):
        # 训练输入
        self.X_ = None
        # 训练标签
        self.y_ = None
        # 标签类别数量
        self.m = None
        # 特征数量
        self.n = None
        # 训练样本量
        self.N = None
        # 常数特征取值
        self.M = None
        # 权重系数
        self.w = None
        # 标签名称
        self.labels = defaultdict(int)
        # 特征名称
        self.features = defaultdict(int)
        # 最大迭代次数
        self.max_iter = max_iter

    ### 计算特征函数关于经验联合分布P(X,Y)的期望
    def _EP_hat_f(self, x, y):
        self.Pxy = np.zeros((self.m, self.n))
        self.Px = np.zeros(self.n)
        for x_, y_ in zip(x, y):
            # 遍历每个样本
            for x__ in set(x_):
                self.Pxy[self.labels[y_], self.features[x__]] += 1
                self.Px[self.features[x__]] += 1
        self.EP_hat_f = self.Pxy / self.N

    ### 计算特征函数关于模型P(Y|X)与经验分布P(X)的期望
    def _EP_f(self):
        self.EPf = np.zeros((self.m, self.n))
        for X in self.X_:
            pw = self._pw(X)
            pw = pw.reshape(self.m, 1)
            px = self.Px.reshape(1, self.n)
            self.EPf += pw * px / self.N

    ### 最大熵模型P(y|x)
    def _pw(self, x):
        mask = np.zeros(self.n + 1)
        for ix in x:
            mask[self.features[ix]] = 1
        tmp = self.w * mask[1:]
        pw = np.exp(np.sum(tmp, axis=1))
        Z = np.sum(pw)
        pw = pw / Z
        return pw

    ### 熵模型拟合
    ### 基于改进的迭代尺度方法IIS
    def fit(self, x, y):
        # 训练输入
        self.X_ = x
        # 训练输出
        self.y_ = list(set(y))
        # 输入数据展平后集合
        tmp = set(self.X_.flatten())
        # 特征命名
        self.features = defaultdict(int, zip(tmp, range(1, len(tmp) + 1)))
        # 标签命名
        self.labels = dict(zip(self.y_, range(len(self.y_))))
        # 特征数
        self.n = len(self.features) + 1
        # 标签类别数量
        self.m = len(self.labels)
        # 训练样本量
        self.N = len(x)
        # 计算EP_hat_f
        self._EP_hat_f(x, y)
        # 初始化系数矩阵
        self.w = np.zeros((self.m, self.n))
        # 循环迭代
        i = 0
        while i <= self.max_iter:
            # 计算EPf
            self._EP_f()
            # 令常数特征函数为M
            self.M = 100
            # IIS算法步骤(3)
            tmp = np.true_divide(self.EP_hat_f, self.EPf)
            tmp[tmp == np.inf] = 0
            tmp = np.nan_to_num(tmp)
            sigma = np.where(tmp != 0, 1 / self.M * np.log(tmp), 0)
            # 更新系数:IIS步骤(4)
            self.w = self.w + sigma
            i += 1
        print('training done.')
        return self

    # 定义最大熵模型预测函数
    def predict(self, x):
        res = np.zeros(len(x), dtype=np.int64)
        for ix, x_ in enumerate(x):
            tmp = self._pw(x_)
            print(tmp, np.argmax(tmp), self.labels)
            res[ix] = self.labels[self.y_[np.argmax(tmp)]]
        return np.array([self.y_[ix] for ix in res])
