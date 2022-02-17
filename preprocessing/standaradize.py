# -*- coding:utf-8 -*-
# @Time : 2022/2/18 12:03 上午
# @Author : huichuan LI
# @File : standaradize.py
# @Software: PyCharm
import numpy as np


class Standardizer:
    def __init__(self, with_mean=True, with_std=True):
        """
        Feature-wise standardization for vector inputs.
        Notes
        -----
        Due to the sensitivity of empirical mean and standard deviation
        calculations to extreme values, `Standardizer` cannot guarantee
        balanced feature scales in the presence of outliers. In particular,
        note that because outliers for each feature can have different
        magnitudes, the spread of the transformed data on each feature can be
        very different.
        Similar to sklearn, `Standardizer` uses a biased estimator for the
        standard deviation: ``numpy.std(x, ddof=0)``.
        Parameters
        ----------
        with_mean : bool
            Whether to scale samples to have 0 mean during transformation.
            Default is True.
        with_std : bool
            Whether to scale samples to have unit variance during
            transformation. Default is True.
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self._is_fit = False

    @property
    def hyperparameters(self):
        H = {"with_mean": self.with_mean, "with_std": self.with_std}
        return H

    @property
    def parameters(self):
        params = {
            "mean": self._mean if hasattr(self, "mean") else None,
            "std": self._std if hasattr(self, "std") else None,
        }
        return params

    def __call__(self, X):
        return self.transform(X)

    def fit(self, X):
        """
        Store the feature-wise mean and standard deviation across the samples
        in `X` for future scaling.
        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            An array of N samples, each with dimensionality `C`
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.shape[0] < 2:
            raise ValueError("`X` must contain at least 2 samples")

        std = np.ones(X.shape[1])
        mean = np.zeros(X.shape[1])

        if self.with_mean:
            mean = np.mean(X, axis=0)

        if self.with_std:
            std = np.std(X, axis=0, ddof=0)

        self._mean = mean
        self._std = std
        self._is_fit = True

    def transform(self, X):
        """
        Standardize features by removing the mean and scaling to unit variance.
        For a sample `x`, the standardized score is calculated as:
        .. math::
            z = (x - u) / s
        where `u` is the mean of the training samples or zero if `with_mean` is
        False, and `s` is the standard deviation of the training samples or 1
        if `with_std` is False.
        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            An array of N samples, each with dimensionality `C`.
        Returns
        -------
        Z : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            The feature-wise standardized version of `X`.
        """
        if not self._is_fit:
            raise Exception("Must call `fit` before using the `transform` method")
        return (X - self._mean) / self._std

    def inverse_transform(self, Z):
        """
        Convert a collection of standardized features back into the original
        feature space.
        For a standardized sample `z`, the unstandardized score is calculated as:
        .. math::
            x = z s + u
        where `u` is the mean of the training samples or zero if `with_mean` is
        False, and `s` is the standard deviation of the training samples or 1
        if `with_std` is False.
        Parameters
        ----------
        Z : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            An array of `N` standardized samples, each with dimensionality `C`.
        Returns
        -------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            The unstandardixed samples from `Z`.
        """
        assert self._is_fit, "Must fit `Standardizer` before calling inverse_transform"
        P = self.parameters
        mean, std = P["mean"], P["std"]
        return Z * std + mean
