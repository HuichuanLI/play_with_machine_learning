# -*- coding:utf-8 -*-
# @Time : 2022/2/19 4:59 下午
# @Author : huichuan LI
# @File : bandit.py
# @Software: PyCharm


import numpy as np
from abc import ABC, abstractmethod


class Bandit(ABC):
    def __init__(self, rewards, reward_probs, context=None):
        assert len(rewards) == len(reward_probs)
        self.step = 0
        self.n_arms = len(rewards)

        super().__init__()

    def __repr__(self):
        """A string representation for the bandit"""
        HP = self.hyperparameters
        params = ", ".join(["{}={}".format(k, v) for (k, v) in HP.items() if k != "id"])
        return "{}({})".format(HP["id"], params)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {}

    @abstractmethod
    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.
        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            The current context matrix for each of the bandit arms, if
            applicable. Default is None.
        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        """
        pass

    def pull(self, arm_id, context=None):
        """
        "Pull" (i.e., sample from) a given arm's payoff distribution.
        Parameters
        ----------
        arm_id : int
            The integer ID of the arm to sample from
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D,)` or None
            The context vector for the current timestep if this is a contextual
            bandit. Otherwise, this argument is unused and defaults to None.
        Returns
        -------
        reward : float
            The reward sampled from the given arm's payoff distribution
        """
        assert arm_id < self.n_arms

        self.step += 1
        return self._pull(arm_id, context)

    def reset(self):
        """Reset the bandit step and action counters to zero."""
        self.step = 0

    @abstractmethod
    def _pull(self, arm_id):
        pass


# 不同的bandit实现，不同的老虎机
class MultinomialBandit(Bandit):
    def __init__(self, payoffs, payoff_probs):
        """
        A multi-armed bandit where each arm is associated with a different
        multinomial payoff distribution.
        """
        super().__init__(payoffs, payoff_probs)

        for r, rp in zip(payoffs, payoff_probs):
            assert len(r) == len(rp)
            np.testing.assert_almost_equal(sum(rp), 1.0)

        payoffs = np.array([np.array(x) for x in payoffs])
        payoff_probs = np.array([np.array(x) for x in payoff_probs])

        self.payoffs = payoffs
        self.payoff_probs = payoff_probs
        self.arm_evs = np.array([sum(p * v) for p, v in zip(payoff_probs, payoffs)])
        self.best_ev = np.max(self.arm_evs)
        self.best_arm = np.argmax(self.arm_evs)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {
            "id": "MultinomialBandit",
            "payoffs": self.payoffs,
            "payoff_probs": self.payoff_probs,
        }

    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.
        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            Unused. Default is None.
        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        return self.best_ev, self.best_arm

    def _pull(self, arm_id, context):
        return int(np.random.rand() <= self.payoff_probs[arm_id])


class BernoulliBandit(Bandit):
    def __init__(self, payoff_probs):
        """
        A multi-armed bandit where each arm is associated with an independent
        Bernoulli payoff distribution.
        Parameters
        ----------
        payoff_probs : list of length `K`
            A list of the payoff probability for each arm. ``payoff_probs[k]``
            holds the probability of payoff for arm `k`.
        """
        payoffs = [1] * len(payoff_probs)
        super().__init__(payoffs, payoff_probs)

        for p in payoff_probs:
            assert p >= 0 and p <= 1

        self.payoffs = np.array(payoffs)
        self.payoff_probs = np.array(payoff_probs)

        self.arm_evs = self.payoff_probs
        self.best_ev = np.max(self.arm_evs)
        self.best_arm = np.argmax(self.arm_evs)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {
            "id": "BernoulliBandit",
            "payoff_probs": self.payoff_probs,
        }

    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.
        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            Unused. Default is None.
        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        return self.best_ev, self.best_arm

    def _pull(self, arm_id, context):
        return int(np.random.rand() <= self.payoff_probs[arm_id])


class GaussianBandit(Bandit):
    def __init__(self, payoff_dists, payoff_probs):
        """
        A multi-armed bandit that is similar to
        :class:`BernoulliBandit`, but instead of each arm having
        a fixed payout of 1, the payoff values are sampled from independent
        Gaussian RVs.
        """
        super().__init__(payoff_dists, payoff_probs)

        for (mean, var), rp in zip(payoff_dists, payoff_probs):
            assert var > 0
            assert np.testing.assert_almost_equal(sum(rp), 1.0)

        self.payoff_dists = payoff_dists
        self.payoff_probs = payoff_probs
        self.arm_evs = np.array([mu for (mu, var) in payoff_dists])
        self.best_ev = np.max(self.arm_evs)
        self.best_arm = np.argmax(self.arm_evs)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {
            "id": "GaussianBandit",
            "payoff_dists": self.payoff_dists,
            "payoff_probs": self.payoff_probs,
        }

    def _pull(self, arm_id, context):
        mean, var = self.payoff_dists[arm_id]

        reward = 0
        if np.random.rand() < self.payoff_probs[arm_id]:
            reward = np.random.normal(mean, var)

        return reward

    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.
        Parameters
        ----------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        return self.best_ev, self.best_arm

