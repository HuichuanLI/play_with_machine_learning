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

plt.show()

nn = NN()
parameters = nn.nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn.nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: nn.predict(parameters, x.T), X, Y[0])
    predictions = nn.predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
plt.show()