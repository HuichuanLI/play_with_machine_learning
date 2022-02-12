# -*- coding:utf-8 -*-
# @Time : 2022/2/13 12:03 上午
# @Author : huichuan LI
# @File : NN_test.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from class_model.NN import NN

def create_dataset():
    np.random.seed(1)
    m = 400  # 数据量
    N = int(m / 2)  # 每个标签的实例数
    D = 2  # 数据维度
    X = np.zeros((m, D))  # 数据矩阵
    Y = np.zeros((m, 1), dtype='uint8')  # 标签维度
    a = 4

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


X, Y = create_dataset()
plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral);

nn = NN()
parameters = nn.nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
