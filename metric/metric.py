# -*- coding:utf-8 -*-
# @Time : 2022/2/11 12:10 上午
# @Author : huichuan LI
# @File : metric.py
# @Software: PyCharm


import numpy as np


### 定义R2系数函数
def r2_score(y_test, y_pred):
    '''
    输入：
    y_test：测试集标签值
    y_pred：测试集预测值
    输出：
    r2：R2系数
    '''
    # 测试标签均值
    y_avg = np.mean(y_test)
    # 总离差平方和
    ss_tot = np.sum((y_test - y_avg) ** 2)
    # 残差平方和
    ss_res = np.sum((y_test - y_pred) ** 2)
    # R2计算
    r2 = 1 - (ss_res / ss_tot)
    return r2


def accuracy(y_test, y_pred):
    correct_count = 0
    for i in range(len(y_test)):
        for j in range(len(y_pred)):
            if y_test[i] == y_pred[j] and i == j:
                correct_count += 1

    accuracy_score = correct_count / len(y_test)
    return accuracy_score


