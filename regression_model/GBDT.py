# -*- coding:utf-8 -*-
# @Time : 2022/2/13 7:03 下午
# @Author : huichuan LI
# @File : GBDT.py
# @Software: PyCharm
### GBDT定义
import numpy as np
from regression_model.CART import TreeNode, BinaryDecisionTree, RegressionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


### 定义回归树的平方损失
class SquareLoss():
    # 定义平方损失
    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    # 定义平方损失的梯度
    def gradient(self, y, y_pred):
        return -(y - y_pred)


class GBDT(object):
    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_gini_impurity, max_depth, regression):
        ### 常用超参数
        # 树的棵树
        self.n_estimators = n_estimators
        # 学习率
        self.learning_rate = learning_rate
        # 结点最小分裂样本数
        self.min_samples_split = min_samples_split
        # 结点最小基尼不纯度
        self.min_gini_impurity = min_gini_impurity
        # 最大深度
        self.max_depth = max_depth
        # 默认为回归树
        self.regression = regression
        # 损失为平方损失
        self.loss = SquareLoss()
        # 如果是分类树，需要定义分类树损失函数
        # 这里省略，如需使用，需自定义分类损失函数
        if not self.regression:
            self.loss = None
        # 多棵树叠加
        self.estimators = []
        for i in range(self.n_estimators):
            self.estimators.append(RegressionTree(min_samples_split=self.min_samples_split,
                                                  min_gini_impurity=self.min_gini_impurity,
                                                  max_depth=self.max_depth))

    # 拟合方法
    def fit(self, X, y):
        # 前向分步模型初始化，第一棵树
        self.estimators[0].fit(X, y)
        # 第一棵树的预测结果
        y_pred = self.estimators[0].predict(X)
        # 前向分步迭代训练
        for i in range(1, self.n_estimators):
            gradient = self.loss.gradient(y, y_pred)
            self.estimators[i].fit(X, gradient)
            y_pred -= np.multiply(self.learning_rate, self.estimators[i].predict(X))

    # 预测方法
    def predict(self, X):
        # 回归树预测
        y_pred = self.estimators[0].predict(X)
        for i in range(1, self.n_estimators):
            y_pred -= np.multiply(self.learning_rate, self.estimators[i].predict(X))
        # 分类树预测
        if not self.regression:
            # 将预测值转化为概率
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            # 转化为预测标签
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred


### GBDT回归树
class GBDTRegressor(GBDT):
    def __init__(self, n_estimators=300, learning_rate=0.1, min_samples_split=2,
                 min_var_reduction=1e-6, max_depth=3):
        super(GBDTRegressor, self).__init__(n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            min_samples_split=min_samples_split,
                                            min_gini_impurity=min_var_reduction,
                                            max_depth=max_depth,
                                            regression=True)
