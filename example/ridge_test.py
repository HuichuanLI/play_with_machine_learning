# -*- coding:utf-8 -*-
# @Time : 2022/2/12 12:30 下午
# @Author : huichuan LI
# @File : ridge_test.py
# @Software: PyCharm
from class_model import lasso, logestic_regression, ridge
import numpy as np
# 导入matplotlib绘图库
import matplotlib.pyplot as plt
# 导入生成分类数据函数
from sklearn.datasets.samples_generator import make_classification

data = np.genfromtxt("../data/example.data", delimiter=',')
# 选择特征与标签
x = data[:, 0:100]
y = data[:, 100].reshape(-1, 1)
# 加一列
X = np.column_stack((np.ones((x.shape[0], 1)), x))

# 划分训练集与测试集
X_train, y_train = X[:70], y[:70]
X_test, y_test = X[70:], y[70:]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

ridge_model = ridge.Ridge()
loss, loss_list, params, grads = ridge_model.ridge_train(X_train, y_train, 0.01, 1000)

# 简单绘图
import matplotlib.pyplot as plt

f = X_test.dot(params['w']) + params['b']

plt.scatter(range(X_test.shape[0]), y_test)
plt.plot(f, color='darkorange')
plt.xlabel('X')
plt.ylabel('y')
plt.show();

# 训练过程中的损失下降
plt.plot(loss_list, color='blue')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
