# -*- coding:utf-8 -*-
# @Time : 2022/2/12 11:36 下午
# @Author : huichuan LI
# @File : C4.py
# @Software: PyCharm
from class_model.C4 import ClassificationTree
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


data = datasets.load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1, 1), test_size=0.3)
clf = ClassificationTree()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
