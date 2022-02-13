# -*- coding:utf-8 -*-
# @Time : 2022/2/13 7:29 下午
# @Author : huichuan LI
# @File : nb.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from collections import Counter


class NB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_count = Counter(y)
        self.class_prior = {key: value / len(y) for key, value in self.class_count.items()}
        print(self.class_prior)
        self.prior = dict()
        for col in range(X.shape[1]):
            for j in self.classes:
                p_x_y = Counter(X[y == j, col])
                for i in p_x_y.keys():
                    self.prior[(col, i, j)] = p_x_y[i] / self.class_count[j]
        return

    def predict(self, X_test):
        choice_res = []
        for elem in X_test:
            res = []
            for c in self.classes:
                p_y = self.class_prior[c]
                p_x_y = 1
                for i in range(X_test.shape[1]):
                    if tuple([i, elem[i], c]) in self.prior:
                        p_x_y *= self.prior[tuple([i, elem[i], c])]
                    else:
                        p_x_y *= 1e-9
                res.append(p_y * p_x_y)
            choice_res.append(self.classes[np.argmax(res)])
        return choice_res


