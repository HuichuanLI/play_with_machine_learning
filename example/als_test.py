# -*- coding:utf-8 -*-
# @Time : 2022/2/17 12:23 上午
# @Author : huichuan LI
# @File : als_test.py
# @Software: PyCharm
import numpy as np
from factorization.als import VanillaALS
from preprocessing.standaradize import Standardizer

data = np.genfromtxt("../data/example.data", delimiter=',')
# 选择特征与标签
x = data[:, 0:100]
y = data[:, 100].reshape(-1, 1)
std = Standardizer()
std.fit(x)
x = std.transform(x)
print(x)
als = VanillaALS(3)
als.fit(x)
print("W")
print(als.W)
print("H")
print(als.H)
