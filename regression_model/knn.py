# -*- coding:utf-8 -*-
# @Time : 2022/2/12 4:04 下午
# @Author : huichuan LI
# @File : knn.py
# @Software: PyCharm
import numpy as np

from collections import Counter


class KNN:
    ### 定义欧氏距离
    def compute_distances(self, X, X_train):
        '''
        输入：
        X：测试样本实例矩阵
        X_train：训练样本实例矩阵
        输出：
        dists：欧式距离
        '''
        # 测试实例样本量
        num_test = X.shape[0]
        # 训练实例样本量
        num_train = X_train.shape[0]
        # 基于训练和测试维度的欧氏距离初始化
        dists = np.zeros((num_test, num_train))
        # 测试样本与训练样本的矩阵点乘
        M = np.dot(X, X_train.T)
        # 测试样本矩阵平方
        te = np.square(X).sum(axis=1)
        # 训练样本矩阵平方
        tr = np.square(X_train).sum(axis=1)
        # 计算欧式距离
        dists = np.sqrt(-2 * M + tr + np.matrix(te).T)
        return dists

    ### 定义预测函数
    def predict_labels(self, y_train, dists, k=1):
        '''
        输入：
        y_train：训练集标签
        dists：测试集与训练集之间的欧氏距离矩阵
        k：k值
        输出：
        y_pred：测试集预测结果
        '''
        # 测试样本量
        num_test = dists.shape[0]
        # 初始化测试集预测结果
        y_pred = np.zeros(num_test)
        # 遍历
        for i in range(num_test):
            # 初始化最近邻列表
            closest_y = []
            # 按欧氏距离矩阵排序后取索引，并用训练集标签按排序后的索引取值
            # 最后拉平列表
            # 注意np.argsort函数的用法
            labels = y_train[np.argsort(dists[i, :])].flatten()
            # 取最近的k个值
            closest_y = labels[0:k]
            # 对最近的k个值进行计数统计
            # 这里注意collections模块中的计数器Counter的用法
            c = Counter(closest_y)
            # 取计数最多的那一个类别
            y_pred[i] = c.most_common(1)[0][0]
        return y_pred
