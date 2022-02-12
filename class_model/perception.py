# -*- coding:utf-8 -*-
# @Time : 2022/2/12 7:46 下午
# @Author : huichuan LI
# @File : perception.py
# @Software: PyCharm
import pandas as pd
import numpy as np


class perception:
    def initialize_parameters(self, dim):
        w = np.zeros(dim, dtype=np.float32)
        b = 0.0
        return w, b

    # 定义sign符号函数
    def sign(self, x, w, b):
        return np.dot(x, w) + b

    # 定义感知机训练函数
    def train(self, X_train, y_train, learning_rate):
        # 参数初始化
        w, b = self.initialize_parameters(X_train.shape[1])
        # 初始化误分类
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]
                # 如果存在误分类点
                # 更新参数
                # 直到没有误分类点
                if y * self.sign(X, w, b) <= 0:
                    w = w + learning_rate * np.dot(y, X)
                    b = b + learning_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
                print('There is no missclassification!')

            # 保存更新后的参数
            params = {
                'w': w,
                'b': b
            }
        return params
