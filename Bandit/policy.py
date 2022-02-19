# -*- coding:utf-8 -*-
# @Time : 2022/2/19 5:32 下午
# @Author : huichuan LI
# @File : policy.py
# @Software: PyCharm
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class BanditPolicyBase(ABC):
    def __init__(self):
        """A simple base class for multi-armed bandit policies"""
        self.step = 0
        self.ev_estimates = {}
        self.is_initialized = False
        super().__init__()

    def __repr__(self):
        """Return a string representation of the policy"""
        HP = self.hyperparameters
        params = ", ".join(["{}={}".format(k, v) for (k, v) in HP.items() if k != "id"])
        return "{}({})".format(HP["id"], params)

    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        pass

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        pass

    def act(self, bandit, context=None):
        """
        Select an arm and sample from its payoff distribution.
        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D,)` or None
            The context vector for the current timestep if interacting with a
            contextual bandit. Otherwise, this argument is unused. Default is
            None.
        Returns
        -------
        rwd : float
            The reward received after pulling ``arm_id``.
        arm_id : int
            The arm that was pulled to generate ``rwd``.
        """
        if not self.is_initialized:
            self._initialize_params(bandit)

        arm_id = self._select_arm(bandit, context)
        rwd = self._pull_arm(bandit, arm_id, context)
        self._update_params(arm_id, rwd, context)
        return rwd, arm_id

    def reset(self):
        """Reset the policy parameters and counters to their initial states."""
        self.step = 0
        self._reset_params()
        self.is_initialized = False

    def _pull_arm(self, bandit, arm_id, context):
        """Execute a bandit action and return the received reward."""
        self.step += 1
        return bandit.pull(arm_id, context)

    @abstractmethod
    def _select_arm(self, bandit, context):
        """Select an arm based on the current context"""
        pass

    @abstractmethod
    def _update_params(self, bandit, context):
        """Update the policy parameters after an interaction"""
        pass

    @abstractmethod
    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        pass

    @abstractmethod
    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        pass


class EpsilonGreedy(BanditPolicyBase):
    def __init__(self, epsilon=0.05, ev_prior=0.5):
        r"""
        An epsilon-greedy policy for multi-armed bandit problems.
        Notes
        epsilon : float in [0, 1]
            The probability of taking a random action. Default is 0.05.
        ev_prior : float
            The starting expected payoff for each arm before any data has been
            observed. Default is 0.5.
        """
        super().__init__()
        self.epsilon = epsilon
        self.ev_prior = ev_prior
        self.pull_counts = defaultdict(lambda: 0)

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        return {"ev_estimates": self.ev_estimates}

    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "id": "EpsilonGreedy",
            "epsilon": self.epsilon,
            "ev_prior": self.ev_prior,
        }

    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        self.ev_estimates = {i: self.ev_prior for i in range(bandit.n_arms)}
        self.is_initialized = True

    def _select_arm(self, bandit, context=None):
        if np.random.rand() < self.epsilon:
            arm_id = np.random.choice(bandit.n_arms)
        else:
            ests = self.ev_estimates
            (arm_id, _) = max(ests.items(), key=lambda x: x[1])
        return arm_id

    def _update_params(self, arm_id, reward, context=None):
        E, C = self.ev_estimates, self.pull_counts
        C[arm_id] += 1
        E[arm_id] += (reward - E[arm_id]) / (C[arm_id])

    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        self.ev_estimates = {}
        self.pull_counts = defaultdict(lambda: 0)


class UCB1(BanditPolicyBase):
    def __init__(self, C=1, ev_prior=0.5):
        self.C = C
        self.ev_prior = ev_prior
        super().__init__()

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        return {"ev_estimates": self.ev_estimates}

    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "C": self.C,
            "id": "UCB1",
            "ev_prior": self.ev_prior,
        }

    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        self.ev_estimates = {i: self.ev_prior for i in range(bandit.n_arms)}
        self.is_initialized = True

    def _select_arm(self, bandit, context=None):
        # add eps to avoid divide-by-zero errors on the first pull of each arm
        eps = np.finfo(float).eps
        N, T = bandit.n_arms, self.step + 1
        E, C = self.ev_estimates, self.pull_counts
        scores = [E[a] + self.C * np.sqrt(np.log(T) / (C[a] + eps)) for a in range(N)]
        return np.argmax(scores)

    def _update_params(self, arm_id, reward, context=None):
        E, C = self.ev_estimates, self.pull_counts
        C[arm_id] += 1
        E[arm_id] += (reward - E[arm_id]) / (C[arm_id])

    def _reset_params(self):
        self.ev_estimates = {}
        self.pull_counts = defaultdict(lambda: 0)


class ThompsonSamplingBetaBinomial(BanditPolicyBase):
    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.alphas, self.betas = [], []
        self.alpha, self.beta = alpha, beta
        self.is_initialized = False

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        return {
            "ev_estimates": self.ev_estimates,
            "alphas": self.alphas,
            "betas": self.betas,
        }

    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "id": "ThompsonSamplingBetaBinomial",
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def _initialize_params(self, bandit):
        bhp = bandit.hyperparameters
        fstr = "ThompsonSamplingBetaBinomial only defined for BernoulliBandit, got: {}"
        assert bhp["id"] == "BernoulliBandit", fstr.format(bhp["id"])

        # initialize the model prior
        self.alphas = [self.alpha] * bandit.n_arms
        self.betas = [self.beta] * bandit.n_arms
        assert len(self.alphas) == len(self.betas) == bandit.n_arms

        self.ev_estimates = {i: self._map_estimate(i, 1) for i in range(bandit.n_arms)}
        self.is_initialized = True

    def _select_arm(self, bandit, context):
        if not self.is_initialized:
            self._initialize_prior(bandit)

        # draw a sample from the current model posterior
        posterior_sample = np.random.beta(self.alphas, self.betas)

        # greedily select an action based on this sample
        return np.argmax(posterior_sample)

    def _update_params(self, arm_id, rwd, context):
        """
        Compute the parameters of the Beta posterior, P(payoff prob | rwd),
        for arm `arm_id`.
        """
        self.alphas[arm_id] += rwd
        self.betas[arm_id] += 1 - rwd
        self.ev_estimates[arm_id] = self._map_estimate(arm_id, rwd)

    def _map_estimate(self, arm_id, rwd):
        """Compute the current MAP estimate for an arm's payoff probability"""
        A, B = self.alphas, self.betas
        if A[arm_id] > 1 and B[arm_id] > 1:
            map_payoff_prob = (A[arm_id] - 1) / (A[arm_id] + B[arm_id] - 2)
        elif A[arm_id] < 1 and B[arm_id] < 1:
            map_payoff_prob = rwd  # 0 or 1 equally likely, make a guess
        elif A[arm_id] <= 1 and B[arm_id] > 1:
            map_payoff_prob = 0
        elif A[arm_id] > 1 and B[arm_id] <= 1:
            map_payoff_prob = 1
        else:
            map_payoff_prob = 0.5
        return map_payoff_prob

    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        self.alphas, self.betas = [], []
        self.ev_estimates = {}


class LinUCB(BanditPolicyBase):
    def __init__(self, alpha=1):
        super().__init__()

        self.alpha = alpha
        self.A, self.b = [], []
        self.is_initialized = False

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        return {"ev_estimates": self.ev_estimates, "A": self.A, "b": self.b}

    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "id": "LinUCB",
            "alpha": self.alpha,
        }

    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        bhp = bandit.hyperparameters
        fstr = "LinUCB only defined for contextual linear bandits, got: {}"
        assert bhp["id"] == "ContextualLinearBandit", fstr.format(bhp["id"])

        self.A, self.b = [], []
        for _ in range(bandit.n_arms):
            self.A.append(np.eye(bandit.D))
            self.b.append(np.zeros(bandit.D))

        self.is_initialized = True

    def _select_arm(self, bandit, context):
        probs = []
        for a in range(bandit.n_arms):
            C, A, b = context[:, a], self.A[a], self.b[a]
            A_inv = np.linalg.inv(A)
            theta_hat = A_inv @ b
            p = theta_hat @ C + self.alpha * np.sqrt(C.T @ A_inv @ C)

            probs.append(p)
        return np.argmax(probs)

    def _update_params(self, arm_id, rwd, context):
        """Compute the parameters for A and b."""
        self.A[arm_id] += context[:, arm_id] @ context[:, arm_id].T
        self.b[arm_id] += rwd * context[:, arm_id]

    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        self.A, self.b = [], []
        self.ev_estimates = {}
