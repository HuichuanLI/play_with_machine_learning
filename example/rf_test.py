# -*- coding:utf-8 -*-
# @Time : 2022/2/12 10:17 下午
# @Author : huichuan LI
# @File : rf_test.py
# @Software: PyCharm

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from class_model.RF import RandomForest
from sklearn.metrics import accuracy_score
import numpy as np

n_estimators = 10
# 列抽样最大特征数
max_features = 15
# 生成模拟二分类数据集
X, y = make_classification(n_samples=1000, n_features=20, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

rf = RandomForest(n_estimators=10, max_features=15)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))
