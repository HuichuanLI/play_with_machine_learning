# -*- coding:utf-8 -*-
# @Time : 2022/2/12 11:55 下午
# @Author : huichuan LI
# @File : NN.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import sklearn


class NN:

    def initialize_parameters(self, n_x, n_h, n_y):
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def layer_sizes(self, X, Y):
        n_x = X.shape[0]  # 输入层大小
        n_h = 4  # 隐藏层大小
        n_y = Y.shape[0]  # 输出层大小
        return (n_x, n_h, n_y)

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def forward_propagation(self, X, parameters):
        # 获取各参数初始值
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        # 执行前向计算
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)
        assert (A2.shape == (1, X.shape[1]))

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}

        return A2, cache

    def compute_cost(self, A2, Y, parameters):
        # 训练样本量
        m = Y.shape[1]
        # 计算交叉熵损失
        logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
        cost = -1 / m * np.sum(logprobs)
        # 维度压缩
        cost = np.squeeze(cost)

        assert (isinstance(cost, float))
        return cost

    def backward_propagation(self, parameters, cache, X, Y):
        m = X.shape[1]
        # 获取W1和W2
        W1 = parameters['W1']
        W2 = parameters['W2']
        # 获取A1和A2
        A1 = cache['A1']
        A2 = cache['A2']
        # 执行反向传播
        dZ2 = A2 - Y
        dW2 = 1 / m * np.dot(dZ2, A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = 1 / m * np.dot(dZ1, X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
        return grads

    def update_parameters(self, parameters, grads, learning_rate=1.2):
        # 获取参数
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        # 获取梯度
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']
        # 参数更新
        W1 -= dW1 * learning_rate
        b1 -= db1 * learning_rate
        W2 -= dW2 * learning_rate
        b2 -= db2 * learning_rate

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        return parameters

    def nn_model(self, X, Y, n_h, num_iterations=10000, print_cost=False):
        np.random.seed(3)
        n_x = self.layer_sizes(X, Y)[0]
        n_y = self.layer_sizes(X, Y)[2]
        # 初始化模型参数
        parameters = self.initialize_parameters(n_x, n_h, n_y)
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        # 梯度下降和参数更新循环
        for i in range(0, num_iterations):
            # 前向传播计算
            A2, cache = self.forward_propagation(X, parameters)
            # 计算当前损失
            cost = self.compute_cost(A2, Y, parameters)
            # 反向传播
            grads = self.backward_propagation(parameters, cache, X, Y)
            # 参数更新
            parameters = self.update_parameters(parameters, grads, learning_rate=1.2)
            # 打印损失
            if print_cost and i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters

    def predict(self, parameters, X):
        A2, cache = self.forward_propagation(X, parameters)
        predictions = (A2 > 0.5)
        return predictions
