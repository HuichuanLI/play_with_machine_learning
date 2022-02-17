# -*- coding:utf-8 -*-
# @Time : 2022/2/18 12:40 上午
# @Author : huichuan LI
# @File : GaussianNB.py
# @Software: PyCharm
import numpy as np


class GaussianNB:
    def __init__(self, eps=1e-6):
        self.labels = None
        self.hyperparameters = {"eps": eps}
        self.parameters = {
            "mean": None,  # shape: (K, M)
            "sigma": None,  # shape: (K, M)
            "prior": None,  # shape: (K,)
        }

    def fit(self, X, y):
        P = self.parameters
        H = self.hyperparameters
        self.labels = np.unique(y)

        K = len(self.labels)
        N, M = X.shape
        P["mean"] = np.zeros((K, M))
        P["sigma"] = np.zeros((K, M))
        P["prior"] = np.zeros((K,))
        for i, c in enumerate(self.labels):
            X_c = X[y == c, :]

            P["mean"][i, :] = np.mean(X_c, axis=0)
            P["sigma"][i, :] = np.var(X_c, axis=0) + H["eps"]
            P["prior"][i] = X_c.shape[0] / N
        return self

    def predict(self, X):
        return self.labels[self._log_posterior(X).argmax(axis=1)]

    def _log_posterior(self, X):
        K = len(self.labels)
        log_posterior = np.zeros((X.shape[0], K))
        for i in range(K):
            log_posterior[:, i] = self._log_class_posterior(X, i)
        return log_posterior

    def _log_class_posterior(self, X, class_idx):
        P = self.parameters
        mu = P["mean"][class_idx]
        prior = P["prior"][class_idx]
        sigsq = P["sigma"][class_idx]

        # log likelihood = log X | N(mu, sigsq)
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigsq))
        log_likelihood -= 0.5 * np.sum(((X - mu) ** 2) / sigsq, axis=1)
        return log_likelihood + np.log(prior)
