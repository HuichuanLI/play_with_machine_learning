# -*- coding:utf-8 -*-
# @Time : 2022/2/17 12:21 上午
# @Author : huichuan LI
# @File : als.py
# @Software: PyCharm
from copy import deepcopy

import numpy as np


class VanillaALS:
    def __init__(self, K, alpha=1, max_iter=200, tol=1e-4):
        self.K = K
        self.W = None
        self.H = None
        self.tol = tol
        self.alpha = alpha
        self.max_iter = max_iter

    @property
    def parameters(self):
        """Return a dictionary of the current model parameters"""
        return {"W": self.W, "H": self.H}

    @property
    def hyperparameters(self):
        """Return a dictionary of the model hyperparameters"""
        return {
            "id": "ALSFactor",
            "K": self.K,
            "tol": self.tol,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
        }

    def _init_factor_matrices(self, X, W=None, H=None):
        """Randomly initialize the factor matrices"""
        N, M = X.shape
        scale = np.sqrt(X.mean() / self.K)
        self.W = np.random.rand(N, self.K) * scale if W is None else W
        self.H = np.random.rand(self.K, M) * scale if H is None else H

        assert self.W.shape == (N, self.K)
        assert self.H.shape == (self.K, M)

    def _loss(self, X, Xhat):
        """Regularized Frobenius loss"""
        alpha, W, H = self.alpha, self.W, self.H
        sq_fnorm = lambda x: np.sum(x ** 2)  # noqa: E731
        return sq_fnorm(X - Xhat) + alpha * (sq_fnorm(W) + sq_fnorm(H))

    def _update_factor(self, X, A):
        """Perform the ALS update"""
        T1 = np.linalg.inv(A.T @ A + self.alpha * np.eye(self.K))
        return X @ A @ T1

    def fit(self, X, W=None, H=None, n_initializations=10, verbose=False):
        if W is not None and H is not None:
            n_initializations = 1

        best_loss = np.inf
        for f in range(n_initializations):
            if verbose:
                print("\nINITIALIZATION {}".format(f + 1))

            new_W, new_H, loss = self._fit(X, W, H, verbose)

            if loss <= best_loss:
                best_loss = loss
                best_W, best_H = deepcopy(new_W), deepcopy(new_H)

        self.W, self.H = best_W, best_H

        if verbose:
            print("\nFINAL LOSS: {}".format(best_loss))

    def _fit(self, X, W, H, verbose):
        self._init_factor_matrices(X, W, H)
        W, H = self.W, self.H

        for i in range(self.max_iter):
            # 这里计算的通过固定X,计算H.T
            W = self._update_factor(X, H.T)
            # 通过计算
            H = self._update_factor(X.T, W).T

            loss = self._loss(X, W @ H)

            if verbose:
                print("[Iter {}] Loss: {:.8f}".format(i + 1, loss))

            if loss <= self.tol:
                break

        return W, H, loss
