# -*- coding:utf-8 -*-
# @Time : 2022/2/12 6:06 下午
# @Author : huichuan LI
# @File : kmeans.py
# @Software: PyCharm
import numpy as np


class Kmeans:

    # 定义欧式距离
    def euclidean_distance(self, x1, x2):
        distance = 0
        # 距离的平方项再开根号
        for i in range(len(x1)):
            distance += pow((x1[i] - x2[i]), 2)
        return np.sqrt(distance)

    # 定义中心初始化函数
    def centroids_init(self, k, X):
        m, n = X.shape
        centroids = np.zeros((k, n))
        for i in range(k):
            # 每一次循环随机选择一个类别中心
            centroid = X[np.random.choice(range(m))]
            centroids[i] = centroid
        return centroids

    # 根据上一步聚类结果计算新的中心点
    def calculate_centroids(self, clusters, k, X):
        n = X.shape[1]
        centroids = np.zeros((k, n))
        # 以当前每个类样本的均值为新的中心点
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # 获取每个样本所属的聚类类别
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(X.shape[0])
        for cluster_i, cluster in enumerate(clusters):
            for X_i in cluster:
                y_pred[X_i] = cluster_i
        return y_pred

    # 定义样本的最近质心点所属的类别索引
    def closest_centroid(self, sample, centroids):
        closest_i = 0
        closest_dist = float('inf')
        for i, centroid in enumerate(centroids):
            # 根据欧式距离判断，选择最小距离的中心点所属类别
            distance = self.euclidean_distance(sample, centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i

    # 根据上述各流程定义kmeans算法流程
    def kmeans(self, X, k, max_iterations):
        # 1.初始化中心点
        centroids = self.centroids_init(k, X)
        # 遍历迭代求解
        for _ in range(max_iterations):
            # 2.根据当前中心点进行聚类
            clusters = self.build_clusters(centroids, k, X)
            # 保存当前中心点
            prev_centroids = centroids
            # 3.根据聚类结果计算新的中心点
            centroids = self.calculate_centroids(clusters, k, X)
            # 4.设定收敛条件为中心点是否发生变化
            diff = centroids - prev_centroids
            if not diff.any():
                break
        # 返回最终的聚类标签
        return self.get_cluster_labels(clusters, X)

    # 定义构建类别过程
    def build_clusters(self, centroids, k, X):
        clusters = [[] for _ in range(k)]
        for x_i, x in enumerate(X):
            # 将样本划分到最近的类别区域
            centroid_i = self.closest_centroid(x, centroids)
            clusters[centroid_i].append(x_i)
        return clusters
