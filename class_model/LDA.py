# -*- coding:utf-8 -*-
# @Time : 2022/2/12 1:21 下午
# @Author : huichuan LI
# @File : LDA.py
# @Software: PyCharm
import numpy as np


class LDA():
    def __init__(self):
        # 初始化权重矩阵
        self.w = None

    # 计算协方差矩阵
    def calc_cov(self, X, Y=None):
        m = X.shape[0]
        # 数据标准化
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        Y = X if Y == None else (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
        return 1 / m * np.matmul(X.T, Y)

    # 对数据进行投影
    def project(self, X, y):
        self.fit(X, y)
        X_projection = X.dot(self.w)
        return X_projection

    # LDA拟合过程
    def fit(self, X, y):
        # 按类分组
        X0 = X[y == 0]
        X1 = X[y == 1]

        # 分别计算两类数据自变量的协方差矩阵
        sigma0 = self.calc_cov(X0)
        sigma1 = self.calc_cov(X1)
        # 计算类内散度矩阵
        Sw = sigma0 + sigma1

        # 分别计算两类数据自变量的均值和差
        u0, u1 = np.mean(X0, axis=0), np.mean(X1, axis=0)
        mean_diff = np.atleast_1d(u0 - u1)

        # 对类内散度矩阵进行奇异值分解
        U, S, V = np.linalg.svd(Sw)
        # 计算类内散度矩阵的逆
        Sw_ = np.dot(np.dot(V.T, np.linalg.pinv(np.diag(S))), U.T)
        # 计算w
        self.w = Sw_.dot(mean_diff)

    # LDA分类预测
    def predict(self, X):
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred
