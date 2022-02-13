# -*- coding:utf-8 -*-
# @Time : 2022/2/13 6:46 下午
# @Author : huichuan LI
# @File : adaboost_test.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from class_model.adaboost import Adaboost

from sklearn.model_selection import train_test_split
# 导入sklearn模拟二分类数据生成模块
from sklearn.datasets.samples_generator import make_blobs

# 生成模拟二分类数据集
X, y = make_blobs(n_samples=150, n_features=2, centers=2,
                  cluster_std=1.2, random_state=40)
# 将标签转换为1/-1
y_ = y.copy()
y_[y_ == 0] = -1
y_ = y_.astype(float)
# 训练/测试数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y_,
                                                    test_size=0.3, random_state=43)
# 设置颜色参数
colors = {0: 'r', 1: 'g'}
# 绘制二分类数据集的散点图
plt.scatter(X[:, 0], X[:, 1], marker='o', c=pd.Series(y).map(colors))
plt.show()

