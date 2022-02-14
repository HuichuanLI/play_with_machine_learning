# -*- coding:utf-8 -*-
# @Time : 2022/2/15 12:09 上午
# @Author : huichuan LI
# @File : xgboost_test.py
# @Software: PyCharm
from sklearn import datasets
from class_model.xgboost import XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 导入鸢尾花数据集
data = datasets.load_iris()
# 获取输入输出
X, y = data.data, data.target
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
# 创建xgboost分类器
clf = XGBoost()
# 模型拟合
clf.fit(X_train, y_train)
# 模型预测
y_pred = clf.predict(X_test)
# 准确率评估
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
