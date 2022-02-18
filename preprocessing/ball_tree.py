# -*- coding:utf-8 -*-
# @Time : 2022/2/18 10:08 下午
# @Author : huichuan LI
# @File : ball_tree.py
# @Software: PyCharm
import numpy as np

from priorityqueue import PriorityQueue


class BallTreeNode:
    def __init__(self, centroid=None, X=None, y=None):
        self.left = None
        self.right = None
        self.radius = None
        self.is_leaf = False

        self.data = X
        self.targets = y
        self.centroid = centroid

    def __repr__(self):
        fstr = "BallTreeNode(centroid={}, is_leaf={})"
        return fstr.format(self.centroid, self.is_leaf)

    def to_dict(self):
        d = self.__dict__
        d["id"] = "BallTreeNode"
        return d


class BallTree:
    def __init__(self, leaf_size=40, metric=None):
        self.root = None
        self.leaf_size = leaf_size
        self.metric = metric if metric is not None else euclidean

    def fit(self, X, y=None):
        """
        Build a ball tree recursively using the O(M log N) `k`-d construction
        algorithm.
        Notes
        -----
        Recursively divides data into nodes defined by a centroid `C` and radius
        `r` such that each point below the node lies within the hyper-sphere
        defined by `C` and `r`.
        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            An array of `N` examples each with `M` features.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, \*)` or None
            An array of target values / labels associated with the entries in
            `X`. Default is None.
        """
        centroid, left_X, left_y, right_X, right_y = self._split(X, y)
        self.root = BallTreeNode(centroid=centroid)
        self.root.radius = np.max([self.metric(centroid, x) for x in X])
        self.root.left = self._build_tree(left_X, left_y)
        self.root.right = self._build_tree(right_X, right_y)

    def _build_tree(self, X, y):
        centroid, left_X, left_y, right_X, right_y = self._split(X, y)

        if X.shape[0] <= self.leaf_size:
            leaf = BallTreeNode(centroid=centroid, X=X, y=y)
            leaf.radius = np.max([self.metric(centroid, x) for x in X])
            leaf.is_leaf = True
            return leaf

        node = BallTreeNode(centroid=centroid)
        node.radius = np.max([self.metric(centroid, x) for x in X])
        node.left = self._build_tree(left_X, left_y)
        node.right = self._build_tree(right_X, right_y)
        return node

    def nearest_neighbors(self, k, x):
        """
        Find the `k` nearest neighbors in the ball tree to a query vector `x`
        using the KNS1 algorithm.
        Parameters
        ----------
        k : int
            The number of closest points in `X` to return
        x : :py:class:`ndarray <numpy.ndarray>` of shape `(1, M)`
            The query vector.
        Returns
        -------
        nearest : list of :class:`PQNode` s of length `k`
            List of the `k` points in `X` to closest to the query vector. The
            ``key`` attribute of each :class:`PQNode` contains the point itself, the
            ``val`` attribute contains its target, and the ``distance``
            attribute contains its distance to the query vector.
        """
        # maintain a max-first priority queue with priority = distance to x
        PQ = PriorityQueue(capacity=k, heap_order="max")
        nearest = self._knn(k, x, PQ, self.root)
        for n in nearest:
            n.distance = self.metric(x, n.key)
        return nearest

    def _knn(self, k, x, PQ, root):
        dist = self.metric
        dist_to_ball = dist(x, root.centroid) - root.radius
        dist_to_farthest_neighbor = dist(x, PQ.peek()["key"]) if len(PQ) > 0 else np.inf

        if dist_to_ball >= dist_to_farthest_neighbor and len(PQ) == k:
            return PQ
        if root.is_leaf:
            targets = [None] * len(root.data) if root.targets is None else root.targets
            for point, target in zip(root.data, targets):
                dist_to_x = dist(x, point)
                if len(PQ) == k and dist_to_x < dist_to_farthest_neighbor:
                    PQ.push(key=point, val=target, priority=dist_to_x)
                else:
                    PQ.push(key=point, val=target, priority=dist_to_x)
        else:
            l_closest = dist(x, root.left.centroid) < dist(x, root.right.centroid)
            PQ = self._knn(k, x, PQ, root.left if l_closest else root.right)
            PQ = self._knn(k, x, PQ, root.right if l_closest else root.left)
        return PQ
