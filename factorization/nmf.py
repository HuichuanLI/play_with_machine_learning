# -*- coding:utf-8 -*-
# @Time : 2022/2/17 12:25 上午
# @Author : huichuan LI
# @File : nmf.py
# @Software: PyCharm
import numpy as np
from factorization.als import VanillaALS
class NMF:
    def __init__(self, K, max_iter=200, tol=1e-4):
        self.K = K
        self.W = None
        self.H = None
        self.tol = tol
        self.max_iter = max_iter

    @property
    def parameters(self):
        """Return a dictionary of the current model parameters"""
        return {"W": self.W, "H": self.H}

    @property
    def hyperparameters(self):
        """Return a dictionary of the model hyperparameters"""
        return {
            "id": "NMF",
            "K": self.K,
            "tol": self.tol,
            "max_iter": self.max_iter,
        }

    def _init_factor_matrices(self, X, W, H):
        """Initialize the factor matrices using vanilla ALS"""
        ALS = None
        N, M = X.shape

        # initialize factors using ALS if not already defined
        if W is None:
            ALS = VanillaALS(self.K, alpha=0, max_iter=200)
            ALS.fit(X, verbose=False)
            W = ALS.W / np.linalg.norm(ALS.W, axis=0)

        if H is None:
            H = np.abs(np.random.rand(self.K, M)) if ALS is None else ALS.H

        assert W.shape == (N, self.K)
        assert H.shape == (self.K, M)

        self.H = H
        self.W = W

    def _loss(self, X, Xhat):
        """Return the least-squares reconstruction loss between X and Xhat"""
        return np.sum((X - Xhat) ** 2)

    def _update_H(self, X, W, H):
        """Perform the fast HALS update for H"""
        eps = np.finfo(float).eps
        XtW = X.T @ W  # dim: (M, K)
        WtW = W.T @ W  # dim: (K, K)

        for k in range(self.K):
            H[k, :] += XtW[:, k] - H.T @ WtW[:, k]
            H[k, :] = np.clip(H[k, :], eps, np.inf)  # enforce nonnegativity
        return H

    def _update_W(self, X, W, H):
        """Perform the fast HALS update for W"""
        eps = np.finfo(float).eps
        XHt = X @ H.T  # dim: (N, K)
        HHt = H @ H.T  # dim: (K, K)

        for k in range(self.K):
            W[:, k] = W[:, k] * HHt[k, k] + XHt[:, k] - W @ HHt[:, k]
            W[:, k] = np.clip(W[:, k], eps, np.inf)  # enforce nonnegativity

            # renormalize the new column
            n = np.linalg.norm(W[:, k])
            W[:, k] /= n if n > 0 else 1.0
        return W

    def fit(self, X, W=None, H=None, n_initializations=10, verbose=False):
        r"""
        Factor a data matrix into two nonnegative low rank factor matrices via
        fast HALS.
        Notes
        -----
        This method implements Algorithm 2 from [*]_. In contrast to vanilla
        ALS, HALS proceeds by minimizing a *set* of local cost functions with
        the same global minima. Each cost function is defined on a "residue" of
        the factor matrices **W** and **H**:
        .. math::
           \mathbf{X}^{(j)} :=
                \mathbf{X} - \mathbf{WH}^\top + \mathbf{w}_j \mathbf{h}_j^\top
        where :math:`\mathbf{X}^{(j)}` is the :math:`j^{th}` residue, **X** is
        the input data matrix, and :math:`\mathbf{w}_j` and
        :math:`\mathbf{h}_j` are the :math:`j^{th}` columns of the current
        factor matrices **W** and **H**. HALS proceeds by minimizing the cost
        for each residue, first with respect to :math:`\mathbf{w}_j`, and then
        with respect to :math:`\mathbf{h}_j`. In either case, the cost for
        residue `j`, :math:`\mathcal{L}^{(j)}` is simply:
        .. math::
            \mathcal{L}^{(j)} :=
                || \mathbf{X}^{(j)} - \mathbf{w}_j \mathbf{h}_j^\top ||
        where :math:`||\cdot||` denotes the Frobenius norm. For NMF,
        minimization is performed under the constraint that all elements of
        both **W** and **H** are nonnegative.
        References
        ----------
        .. [*] Cichocki, A., & Phan, A. (2009). Fast local algorithms for
           large scale nonnegative matrix and tensor factorizations. *IEICE
           Transactions on Fundamentals of Electronics, Communications and
           Computer Sciences, 92(3)*, 708-721.
        Parameters
        ----------
        X : numpy array of shape `(N, M)`
            The data matrix to factor.
        W : numpy array of shape `(N, K)` or None
            An initial value for the `W` factor matrix. If None, initialize
            **W** using vanilla ALS. Default is None.
        H : numpy array of shape `(K, M)` or None
            An initial value for the `H` factor matrix. If None, initialize
            **H** using vanilla ALS. Default is None.
        n_initializations : int
            Number of re-initializations of the algorithm to perform before
            taking the answer with the lowest reconstruction error. This value
            is ignored and set to 1 if both `W` and `H` are not None. Default
            is 10.
        verbose : bool
            Whether to print the loss at each iteration. Default is False.
        """
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
            H = self._update_H(X, W, H)
            W = self._update_W(X, W, H)
            loss = self._loss(X, W @ H)

            if verbose:
                print("[Iter {}] Loss: {:.8f}".format(i + 1, loss))

            if loss <= self.tol:
                break
        return W, H, loss