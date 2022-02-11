# -*- coding:utf-8 -*-
# @Time : 2022/2/11 11:55 下午
# @Author : huichuan LI
# @File : lasso.py
# @Software: PyCharm
import numpy as np
import pandas as pd


class Lasso:
    # 定义参数初始化函数
    def initialize(self, dims):
        w = np.zeros((dims, 1))
        b = 0
        return w, b

    # 定义符号函数
    def sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    # 定义lasso损失函数
    def l1_loss(self, X, y, w, b, alpha):
        num_train = X.shape[0]
        num_feature = X.shape[1]
        y_hat = np.dot(X, w) + b
        loss = np.sum((y_hat - y) ** 2) / num_train + np.sum(alpha * abs(w))

        dw = np.dot(X.T, (y_hat - y)) / num_train + alpha * np.vectorize(self.sign)(w)
        db = np.sum((y_hat - y)) / num_train
        return y_hat, loss, dw, db

    # 定义训练过程
    def lasso_train(self, X, y, learning_rate=0.01, epochs=300):
        loss_list = []
        w, b = self.initialize(X.shape[1])
        for i in range(1, epochs):
            y_hat, loss, dw, db = self.l1_loss(X, y, w, b, 0.1)
            w += -learning_rate * dw
            b += -learning_rate * db
            loss_list.append(loss)

            if i % 300 == 0:
                print('epoch %d loss %f' % (i, loss))
            params = {
                'w': w,
                'b': b
            }
            grads = {
                'dw': dw,
                'db': db
            }
        return loss, loss_list, params, grads

    # 定义预测函数
    def predict(self, X, params):
        w = params['w']
        b = params['b']

        y_pred = np.dot(X, w) + b
        return y_pred


if __name__ == "__main__":
    # 导入matplotlib绘图库
    import matplotlib.pyplot as plt
    # 导入生成分类数据函数
    from sklearn.datasets.samples_generator import make_classification

    # 生成100*2的模拟二分类数据集
    X, labels = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=2)
    # 设置随机数种子
    rng = np.random.RandomState(2)
    # 对生成的特征数据添加一组均匀分布噪声
    X += 2 * rng.uniform(size=X.shape)
    # 标签类别数
    unique_lables = set(labels)
    # 根据标签类别数设置颜色
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_lables)))
    # 绘制模拟数据的散点图
    for k, col in zip(unique_lables, colors):
        x_k = X[labels == k]
        plt.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="k",
                 markersize=14)
    plt.title('Simulated binary data set')
    plt.show();

    labels = labels.reshape((-1, 1))
    data = np.concatenate((X, labels), axis=1)
    print(data.shape)

    # 训练集与测试集的简单划分
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], labels[:offset]
    X_test, y_test = X[offset:], labels[offset:]
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    print('X_train=', X_train.shape)
    print('X_test=', X_test.shape)
    print('y_train=', y_train.shape)
    print('y_test=', y_test.shape)
    lasso = Lasso()
    loss, loss_list, params, grads = lasso.lasso_train(X_train, y_train, 0.01, 3000)
    print(params)

    y_pred = lasso.predict(X_test, params)
    print(y_pred)

    # 简单绘图
    import matplotlib.pyplot as plt

    f = X_test.dot(params['w']) + params['b']

    plt.scatter(range(X_test.shape[0]), y_test)
    plt.plot(f, color='darkorange')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

    # 训练过程中的损失下降
    plt.plot(loss_list, color='blue')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
