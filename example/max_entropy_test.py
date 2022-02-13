# -*- coding:utf-8 -*-
# @Time : 2022/2/13 10:39 下午
# @Author : huichuan LI
# @File : max_entropy_test.py
# @Software: PyCharm
from prob.max_entropy import MaxEnt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

raw_data = load_iris()
X, labels = raw_data.data, raw_data.target
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=43)
print(X_train.shape, y_train.shape)
from sklearn.metrics import accuracy_score

maxent = MaxEnt()
maxent.fit(X_train, y_train)
y_pred = maxent.predict(X_test)
print(accuracy_score(y_test, y_pred))
