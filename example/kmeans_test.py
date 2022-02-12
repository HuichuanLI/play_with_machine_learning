# -*- coding:utf-8 -*-
# @Time : 2022/2/12 6:49 下午
# @Author : huichuan LI
# @File : kmeans_test.py
# @Software: PyCharm

import numpy as np
from cluster.kmeans import Kmeans

# 测试数据
X = np.array([[0,2],[0,0],[1,0],[5,0],[5,2]])
# 设定聚类类别为2个，最大迭代次数为10次
labels = Kmeans().kmeans(X, 2, 10)
# 打印每个样本所属的类别标签
print(labels)