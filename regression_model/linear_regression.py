# -*- coding:utf-8 -*-
# @Time : 2022/2/11 12:03 上午
# @Author : huichuan LI
# @File : linear_regression.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from metric.metric import r2_score

class Linear_regression:
    ### 初始化模型参数
    def initialize_params(self, dims):
        '''
        输入：
        dims：训练数据变量维度
        输出：
        w：初始化权重参数值
        b：初始化偏差参数值
        '''
        # 初始化权重参数为零矩阵
        w = np.zeros((dims, 1))
        # 初始化偏差参数为零
        b = 0
        return w, b

    ### 包括线性回归公式、均方损失和参数偏导三部分
    def linear_loss(self, X, y, w, b):
        '''
        输入:
        X：输入变量矩阵
        y：输出标签向量
        w：变量参数权重矩阵
        b：偏差项
        输出：
        y_hat：线性模型预测输出
        loss：均方损失值
        dw：权重参数一阶偏导
        db：偏差项一阶偏导
        '''
        # 训练样本数量
        num_train = X.shape[0]
        # 训练特征数量
        num_feature = X.shape[1]
        # 线性回归预测输出
        y_hat = np.dot(X, w) + b
        # 计算预测输出与实际标签之间的均方损失
        loss = np.sum((y_hat - y) ** 2) / num_train
        # 基于均方损失对权重参数的一阶偏导数
        dw = np.dot(X.T, (y_hat - y)) / num_train
        # 基于均方损失对偏差项的一阶偏导数
        db = np.sum((y_hat - y)) / num_train
        return y_hat, loss, dw, db

    ### 定义线性回归模型训练过程
    def linear_train(self, X, y, learning_rate=0.01, epochs=10000):
        '''
        输入：
        X：输入变量矩阵
        y：输出标签向量
        learning_rate：学习率
        epochs：训练迭代次数
        输出：
        loss_his：每次迭代的均方损失
        params：优化后的参数字典
        grads：优化后的参数梯度字典
        '''
        # 记录训练损失的空列表
        loss_his = []
        # 初始化模型参数
        w, b = self.initialize_params(X.shape[1])
        # 迭代训练
        for i in range(1, epochs):
            # 计算当前迭代的预测值、损失和梯度
            y_hat, loss, dw, db = self.linear_loss(X, y, w, b)
            # 基于梯度下降的参数更新
            w += -learning_rate * dw
            b += -learning_rate * db
            # 记录当前迭代的损失
            loss_his.append(loss)
            # 每1000次迭代打印当前损失信息
            if i % 10000 == 0:
                print('epoch %d loss %f' % (i, loss))
            # 将当前迭代步优化后的参数保存到字典
            params = {
                'w': w,
                'b': b
            }
            # 将当前迭代步的梯度保存到字典
            grads = {
                'dw': dw,
                'db': db
            }
        return loss_his, params, grads

    ### 定义线性回归预测函数
    def predict(self, X, params):
        '''
        输入：
        X：测试数据集
        params：模型训练参数
        输出：
        y_pred：模型预测结果
        '''
        # 获取模型参数
        w = params['w']
        b = params['b']
        # 预测
        y_pred = np.dot(X, w) + b
        return y_pred


if __name__ == "__main__":
    # 导入sklearn diabetes数据接口
    from sklearn.datasets import load_diabetes
    # 导入sklearn打乱数据函数
    from sklearn.utils import shuffle

    # 获取diabetes数据集
    diabetes = load_diabetes()
    # 获取输入和标签
    data, target = diabetes.data, diabetes.target
    # 打乱数据集
    X, y = shuffle(data, target, random_state=13)
    # 按照8/2划分训练集和测试集
    offset = int(X.shape[0] * 0.8)
    # 训练集
    X_train, y_train = X[:offset], y[:offset]
    # 测试集
    X_test, y_test = X[offset:], y[offset:]
    # 将训练集改为列向量的形式
    y_train = y_train.reshape((-1, 1))
    # 将验证集改为列向量的形式
    y_test = y_test.reshape((-1, 1))
    # 打印训练集和测试集维度
    print("X_train's shape: ", X_train.shape)
    print("X_test's shape: ", X_test.shape)
    print("y_train's shape: ", y_train.shape)
    print("y_test's shape: ", y_test.shape)

    lr = Linear_regression()
    loss_his, params, grads = lr.linear_train(X_train, y_train, 0.01, 200000)
    print(params)

    y_pred = lr.predict(X_test, params)

    print(r2_score(y_test,y_pred))

    import matplotlib.pyplot as plt

    f = X_test.dot(params['w']) + params['b']

    plt.scatter(range(X_test.shape[0]), y_test)
    plt.plot(f, color='darkorange')
    plt.xlabel('X_test')
    plt.ylabel('y_test')
    plt.show();

    plt.plot(loss_his, color='blue')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()