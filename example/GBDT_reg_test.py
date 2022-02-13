# -*- coding:utf-8 -*-
# @Time : 2022/2/13 7:11 下午
# @Author : huichuan LI
# @File : GBDT_reg_test.py
# @Software: PyCharm
### GBDT分类树
# 导入sklearn数据集模块
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from regression_model.GBDT import GBDTRegressor


def data_shuffle(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


# 导入波士顿房价数据集
boston = datasets.load_boston()
# 打乱数据集
X, y = data_shuffle(boston.data, boston.target, seed=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 创建GBRT实例
model = GBDTRegressor()
# 模型训练
model.fit(X_train, y_train)
# 模型预测
y_pred = model.predict(X_test)
# 计算模型预测的均方误差
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error of NumPy GBRT:", mse)
