# -*- coding:utf-8 -*-
# @Time : 2022/2/12 7:58 下午
# @Author : huichuan LI
# @File : RF.py
# @Software: PyCharm
from class_model.CART import *


class RandomForest():
    def __init__(self, n_estimators=100, min_samples_split=2, min_gain=999,
                 max_depth=float("inf"), max_features=None):
        # 树的棵树
        self.n_estimators = n_estimators
        # 树最小分裂样本数
        self.min_samples_split = min_samples_split
        # 最小增益
        self.min_gain = min_gain
        # 树最大深度
        self.max_depth = max_depth
        # 所使用最大特征数
        self.max_features = max_features

        self.trees = []
        # 基于决策树构建森林
        for _ in range(self.n_estimators):
            tree = ClassificationTree(min_samples_split=self.min_samples_split, min_gini_impurity=self.min_gain,
                                      max_depth=self.max_depth)
            self.trees.append(tree)

    # 自助抽样
    def bootstrap_sampling(self, X, y):
        X_y = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        np.random.shuffle(X_y)
        n_samples = X.shape[0]
        sampling_subsets = []

        for _ in range(self.n_estimators):
            # 第一个随机性，行抽样
            idx1 = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_Xy = X_y[idx1, :]
            bootstrap_X = bootstrap_Xy[:, :-1]
            bootstrap_y = bootstrap_Xy[:, -1]
            sampling_subsets.append([bootstrap_X, bootstrap_y])
        return sampling_subsets

    # 随机森林训练
    def fit(self, X, y):
        # 对森林中每棵树训练一个双随机抽样子集
        sub_sets = self.bootstrap_sampling(X, y)
        n_features = X.shape[1]
        # 设置max_feature
        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))

        for i in range(self.n_estimators):
            # 第二个随机性，列抽样
            sub_X, sub_y = sub_sets[i]
            idx2 = np.random.choice(n_features, self.max_features, replace=True)
            sub_X = sub_X[:, idx2]
            self.trees[i].fit(sub_X, sub_y)
            # 保存每次列抽样的列索引，方便预测时每棵树调用
            self.trees[i].feature_indices = idx2
            print('The {}th tree is trained done...'.format(i + 1))

    # 随机森林预测
    def predict(self, X):
        y_preds = []
        for i in range(self.n_estimators):
            idx = self.trees[i].feature_indices
            sub_X = X[:, idx]
            y_pred = self.trees[i].predict(sub_X)
            y_preds.append(y_pred)

        y_preds = np.array(y_preds).T
        res = []
        for j in y_preds:
            res.append(np.bincount(j.astype('int')).argmax())
        return res
