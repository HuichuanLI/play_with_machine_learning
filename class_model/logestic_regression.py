# -*- coding:utf-8 -*-
# @Time : 2022/2/11 11:15 下午
# @Author : huichuan LI
# @File : logestic_regression.py
# @Software: PyCharm
import numpy as np
from metric.metric import accuracy


class logestic_regression:

    def initialize_params(self, dims):
        W = np.zeros((dims, 1))
        b = 0
        return W, b

    def sigmoid(self, x):
        z = 1 / (1 + np.exp(-x))
        return z

    ### 定义逻辑回归模型主体
    def logistic(self, X, y, W, b):
        '''
        输入：
        X: 输入特征矩阵
        y: 输出标签向量
        W: 权值参数
        b: 偏置参数
        输出：
        a: 逻辑回归模型输出
        cost: 损失
        dW: 权值梯度
        db: 偏置梯度
        '''
        # 训练样本量
        num_train = X.shape[0]
        # 训练特征数
        num_feature = X.shape[1]
        # 逻辑回归模型输出
        a = self.sigmoid(np.dot(X, W) + b)
        # 交叉熵损失
        cost = -1 / num_train * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
        # 权值梯度
        dW = np.dot(X.T, (a - y)) / num_train
        # 偏置梯度
        db = np.sum(a - y) / num_train
        # 压缩损失数组维度
        cost = np.squeeze(cost)
        return a, cost, dW, db

    ### 定义逻辑回归模型训练过程
    def logistic_train(self, X, y, learning_rate, epochs):
        '''
        输入：
        X: 输入特征矩阵
        y: 输出标签向量
        learning_rate: 学习率
        epochs: 训练轮数
        输出：
        cost_list: 损失列表
        params: 模型参数
        grads: 参数梯度
        '''
        # 初始化模型参数
        W, b = self.initialize_params(X.shape[1])
        # 初始化损失列表
        cost_list = []

        # 迭代训练
        for i in range(epochs):
            # 计算当前次的模型计算结果、损失和参数梯度
            a, cost, dW, db = self.logistic(X, y, W, b)
            # 参数更新
            W = W - learning_rate * dW
            b = b - learning_rate * db
            # 记录损失
            if i % 100 == 0:
                cost_list.append(cost)
                # 打印训练过程中的损失
            if i % 100 == 0:
                print('epoch %d cost %f' % (i, cost))

                # 保存参数
        params = {
            'W': W,
            'b': b
        }

        # 保存梯度
        grads = {
            'dW': dW,
            'db': db
        }
        return cost_list, params, grads

    ### 定义预测函数
    def predict(self, X, params):
        '''
        输入：
        X: 输入特征矩阵
        params: 训练好的模型参数
        输出：
        y_prediction: 转换后的模型预测值
        '''
        # 模型预测值
        y_prediction = self.sigmoid(np.dot(X, params['W']) + params['b'])
        # 基于分类阈值对概率预测值进行类别转换
        for i in range(len(y_prediction)):
            if y_prediction[i] > 0.5:
                y_prediction[i] = 1
            else:
                y_prediction[i] = 0

        return y_prediction


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

    lr = logestic_regression()
    cost_list, params, grads = lr.logistic_train(X_train, y_train, 0.01, 1000)

    from sklearn.metrics import accuracy_score, classification_report

    y_pred = lr.predict(X_test, params)

    # print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    accuracy_score_test = accuracy(y_test, y_pred)
    print(accuracy_score_test)
