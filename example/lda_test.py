# -*- coding:utf-8 -*-
# @Time : 2022/2/12 2:01 下午
# @Author : huichuan LI
# @File : lda_test.py
# @Software: PyCharm
# -*- coding:utf-8 -*-
# @Time : 2022/2/12 12:30 下午
# @Author : huichuan LI
# @File : ridge_test.py
# @Software: PyCharm
from class_model import  LDA
import numpy as np
# 导入matplotlib绘图库
import matplotlib.pyplot as plt
# 导入生成分类数据函数
from sklearn.datasets.samples_generator import make_classification

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = datasets.load_iris()
X = data.data
y = data.target


X = X[y != 2]
y = y[y != 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

lda_model = LDA.LDA()
lda_model.fit(X_train, y_train)

y_pred = lda_model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

