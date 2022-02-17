# -*- coding:utf-8 -*-
# @Time : 2022/2/13 9:50 下午
# @Author : huichuan LI
# @File : nb_test.py
# @Software: PyCharm
from prob.nb import NB
from prob.GaussianNB import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = NB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print("Accuracy of GaussianNB in iris data test:",
      accuracy_score(y_test, y_pred))


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print("Accuracy of GaussianNB in iris data test:",
      accuracy_score(y_test, y_pred))
