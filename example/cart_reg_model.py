# -*- coding:utf-8 -*-
# @Time : 2022/2/12 5:33 下午
# @Author : huichuan LI
# @File : cart_cls_model.py
# @Software: PyCharm
from sklearn.datasets import load_boston
from regression_model.CART import RegressionTree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

y_train = [[elem_y] for elem_y in y_train]
model = RegressionTree()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
