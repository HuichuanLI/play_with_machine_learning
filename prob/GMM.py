# -*- coding:utf-8 -*-
# @Time : 2022/2/15 11:55 下午
# @Author : huichuan LI
# @File : GMM.py
# @Software: PyCharm
import numpy as np


def logsumexp(log_probs, axis=None):
    """
    Redefine scipy.special.logsumexp
    """
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)


def log_gaussian_pdf(x_i, mu, sigma):
    """Compute log N(x_i | mu, sigma)"""
    n = len(mu)
    a = n * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(sigma)

    y = np.linalg.solve(sigma, x_i - mu)
    c = np.dot(x_i - mu, y)
    return -0.5 * (a + b + c)


class GMM(object):
    def __init__(self, C=3, seed=None):
        self.elbo = None
        self.parameters = {}
        self.hyperparameters = {
            "C": C,
            "seed": seed,
        }

        self.is_fit = False

        if seed:
            np.random.seed(seed)

    def _initialize_params(self, X):
        """Randomly initialize the starting GMM parameters."""
        N, d = X.shape
        C = self.hyperparameters["C"]

        rr = np.random.rand(C)

        self.parameters = {
            "pi": rr / rr.sum(),  # cluster priors
            "Q": np.zeros((N, C)),  # variational distribution q(T)
            "mu": np.random.uniform(-5, 10, C * d).reshape(C, d),  # cluster means
            "sigma": np.array([np.eye(d) for _ in range(C)]),  # cluster covariances
        }

        self.elbo = None
        self.is_fit = False

    def likelihood_lower_bound(self, X):
        """Compute the LLB under the current GMM parameters."""
        N = X.shape[0]
        P = self.parameters
        C = self.hyperparameters["C"]
        pi, Q, mu, sigma = P["pi"], P["Q"], P["mu"], P["sigma"]

        eps = np.finfo(float).eps
        expec1, expec2 = 0.0, 0.0
        for i in range(N):
            x_i = X[i]

            for c in range(C):
                pi_k = pi[c]
                z_nk = Q[i, c]
                mu_k = mu[c, :]
                sigma_k = sigma[c, :, :]

                log_pi_k = np.log(pi_k + eps)
                log_p_x_i = log_gaussian_pdf(x_i, mu_k, sigma_k)
                prob = z_nk * (log_p_x_i + log_pi_k)

                expec1 += prob
                expec2 += z_nk * np.log(z_nk + eps)

        loss = expec1 - expec2
        return loss

    def fit(self, X, max_iter=100, tol=1e-3, verbose=False):
        prev_vlb = -np.inf
        self._initialize_params(X)

        for _iter in range(max_iter):
            try:
                # Estep 通过X计算log(x),获取q_i
                self._E_step(X)
                # 通过q_i，获取参数["pi"],P["mu"], P["sigma"]
                self._M_step(X)
                vlb = self.likelihood_lower_bound(X)

                if verbose:
                    print(f"{_iter + 1}. Lower bound: {vlb}")

                converged = _iter > 0 and np.abs(vlb - prev_vlb) <= tol
                if np.isnan(vlb) or converged:
                    break

                prev_vlb = vlb

            except np.linalg.LinAlgError:
                print("Singular matrix: components collapsed")
                return -1

        self.elbo = vlb
        self.is_fit = True
        return 0

    def predict(self, X, soft_labels=True):
        assert self.is_fit, "Must call the `.fit` method before making predictions"

        P = self.parameters
        C = self.hyperparameters["C"]
        mu, sigma = P["mu"], P["sigma"]

        y = []
        for x_i in X:
            cprobs = [log_gaussian_pdf(x_i, mu[c, :], sigma[c, :, :]) for c in range(C)]

            if not soft_labels:
                y.append(np.argmax(cprobs))
            else:
                y.append(cprobs)

        return np.array(y)

    def _E_step(self, X):
        P = self.parameters
        C = self.hyperparameters["C"]
        pi, Q, mu, sigma = P["pi"], P["Q"], P["mu"], P["sigma"]

        for i, x_i in enumerate(X):
            denom_vals = []
            for c in range(C):
                pi_c = pi[c]
                mu_c = mu[c, :]
                sigma_c = sigma[c, :, :]
                # print(pi_c, mu_c, sigma_c)
                log_pi_c = np.log(pi_c)
                log_p_x_i = log_gaussian_pdf(x_i, mu_c, sigma_c)
                # print(log_p_x_i)
                # log N(X_i | mu_c, Sigma_c) + log pi_c
                denom_vals.append(log_p_x_i + log_pi_c)
            # log \sum_c exp{ log N(X_i | mu_c, Sigma_c) + log pi_c } ]
            log_denom = logsumexp(denom_vals)
            q_i = np.exp([num - log_denom for num in denom_vals])
            np.testing.assert_allclose(np.sum(q_i), 1, err_msg="{}".format(np.sum(q_i)))

            Q[i, :] = q_i

    def _M_step(self, X):
        N, d = X.shape
        P = self.parameters
        C = self.hyperparameters["C"]
        pi, Q, mu, sigma = P["pi"], P["Q"], P["mu"], P["sigma"]
        denoms = np.sum(Q, axis=0)
        # update cluster priors
        pi = denoms / N
        # update cluster means
        nums_mu = [np.dot(Q[:, c], X) for c in range(C)]
        for ix, (num, den) in enumerate(zip(nums_mu, denoms)):
            mu[ix, :] = num / den if den > 0 else np.zeros_like(num)

        # update cluster covariances
        for c in range(C):
            mu_c = mu[c, :]
            n_c = denoms[c]

            outer = np.zeros((d, d))
            for i in range(N):
                wic = Q[i, c]
                xi = X[i, :]
                outer += wic * np.outer(xi - mu_c, xi - mu_c)

            outer = outer / n_c if n_c > 0 else outer
            sigma[c, :, :] = outer

        np.testing.assert_allclose(np.sum(pi), 1, err_msg="{}".format(np.sum(pi)))
