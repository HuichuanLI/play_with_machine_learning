# -*- coding:utf-8 -*-
# @Time : 2022/2/15 11:56 下午
# @Author : huichuan LI
# @File : HMM.py
# @Software: PyCharm

import numpy as np


class MultinomialHMM:
    def __init__(self, A=None, B=None, pi=None, eps=None):
        r"""
        A simple hidden Markov model with multinomial emission distribution.
        Parameters
        ----------
        A : :py:class:`ndarray <numpy.ndarray>` of shape `(N, N)` or None
            The transition matrix between latent states in the HMM. Index `i`,
            `j` gives the probability of transitioning from latent state `i` to
            latent state `j`. Default is None.
        B : :py:class:`ndarray <numpy.ndarray>` of shape `(N, V)` or None
            The emission matrix. Entry `i`, `j` gives the probability of latent
            state i emitting an observation of type `j`. Default is None.
        pi : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)` or None
            The prior probability of each latent state. If None, use a uniform
            prior over states. Default is None.
        eps : float or None
            Epsilon value to avoid :math:`\log(0)` errors. If None, defaults to
            the machine epsilon. Default is None.

        """
        # prior probability of each latent state
        if pi is not None:
            pi[pi == 0] = eps

        # number of latent state types
        N = None
        if A is not None:
            N = A.shape[0]
            A[A == 0] = eps

        # number of observation types
        V = None
        if B is not None:
            V = B.shape[1]
            B[B == 0] = eps

        self.parameters = {
            "A": A,  # transition matrix
            "B": B,  # emission matrix
            "pi": pi,  # prior probability of each latent state
        }

        self.hyperparameters = {
            "eps": eps,  # epsilon
        }

        self.derived_variables = {
            "N": N,  # number of latent state types
            "V": V,  # number of observation types
        }

    def generate(self, n_steps, latent_state_types, obs_types):
        """
        Sample a sequence from the HMM.
        Parameters
        ----------
        n_steps : int
            The length of the generated sequence
        latent_state_types : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            A collection of labels for the latent states
        obs_types : :py:class:`ndarray <numpy.ndarray>` of shape `(V,)`
            A collection of labels for the observations
        Returns
        -------
        states : :py:class:`ndarray <numpy.ndarray>` of shape `(n_steps,)`
            The sampled latent states.
        emissions : :py:class:`ndarray <numpy.ndarray>` of shape `(n_steps,)`
            The sampled emissions.
        """
        P = self.parameters
        A, B, pi = P["A"], P["B"], P["pi"]

        # sample the initial latent state
        s = np.random.multinomial(1, pi).argmax()
        states = [latent_state_types[s]]

        # generate an emission given latent state
        v = np.random.multinomial(1, B[s, :]).argmax()
        emissions = [obs_types[v]]

        # sample a latent transition, rinse, and repeat
        for i in range(n_steps - 1):
            s = np.random.multinomial(1, A[s, :]).argmax()
            states.append(latent_state_types[s])

            v = np.random.multinomial(1, B[s, :]).argmax()
            emissions.append(obs_types[v])

        return np.array(states), np.array(emissions)


