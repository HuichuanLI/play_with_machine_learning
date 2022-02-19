# -*- coding:utf-8 -*-
# @Time : 2022/2/19 5:51 下午
# @Author : huichuan LI
# @File : trainer.py
# @Software: PyCharm
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

_PLOTTING = True


def mse(bandit, policy):
    if not hasattr(policy, "ev_estimates") or len(policy.ev_estimates) == 0:
        return np.nan

    se = []
    evs = bandit.arm_evs
    ests = sorted(policy.ev_estimates.items(), key=lambda x: x[0])
    for ix, (est, ev) in enumerate(zip(ests, evs)):
        se.append((est[1] - ev) ** 2)
    return np.mean(se)


class BanditTrainer:
    def __init__(self):
        """
        An object to facilitate multi-armed bandit training, comparison, and
        evaluation.
        """
        self.logs = {}

    def compare(
            self,
            policies,
            bandit,
            n_trials,
            n_duplicates,
            plot=True,
            seed=None,
            smooth_weight=0.999,
            out_dir=None,
    ):
        self.init_logs(policies)

        all_axes = [None] * len(policies)
        if plot and _PLOTTING:
            fig, all_axes = plt.subplots(len(policies), 2, sharex=True)
            fig.set_size_inches(10.5, len(policies) * 5.25)

        for policy, axes in zip(policies, all_axes):
            if seed:
                np.random.seed(seed)

            bandit.reset()
            policy.reset()

            self.train(
                policy,
                bandit,
                n_trials,
                n_duplicates,
                axes=axes,
                plot=plot,
                verbose=True,
                out_dir=out_dir,
                smooth_weight=smooth_weight,
            )

    def init_logs(self, policies):
        if not isinstance(policies, list):
            policies = [policies]

        self.logs = {
            str(p): {
                "mse": defaultdict(lambda: []),
                "regret": defaultdict(lambda: []),
                "reward": defaultdict(lambda: []),
                "cregret": defaultdict(lambda: []),
                "optimal_arm": defaultdict(lambda: []),
                "selected_arm": defaultdict(lambda: []),
                "optimal_reward": defaultdict(lambda: []),
            }
            for p in policies
        }

    def train(
            self,
            policy,
            bandit,
            n_trials,
            n_duplicates,
            plot=True,
            axes=None,
            verbose=True,
            print_every=100,
            smooth_weight=0.999,
            out_dir=None,
    ):
        if not str(policy) in self.logs:
            self.init_logs(policy)

        p = str(policy)
        D, L = n_duplicates, self.logs
        for d in range(D):
            if verbose:
                print("\nDUPLICATE {}/{}\n".format(d + 1, D))

            bandit.reset()
            policy.reset()

            avg_oracle_reward, cregret = 0, 0
            for trial_id in range(n_trials):
                rwd, arm, orwd, oarm = self._train_step(bandit, policy)

                loss = mse(bandit, policy)
                regret = orwd - rwd

                avg_oracle_reward += orwd
                cregret += regret

                L[p]["mse"][trial_id + 1].append(loss)
                L[p]["reward"][trial_id + 1].append(rwd)
                L[p]["regret"][trial_id + 1].append(regret)
                L[p]["cregret"][trial_id + 1].append(cregret)
                L[p]["optimal_arm"][trial_id + 1].append(oarm)
                L[p]["selected_arm"][trial_id + 1].append(arm)
                L[p]["optimal_reward"][trial_id + 1].append(orwd)

                if (trial_id + 1) % print_every == 0 and verbose:
                    fstr = "Trial {}/{}, {}/{}, Regret: {:.4f}"
                    print(fstr.format(trial_id + 1, n_trials, d + 1, D, regret))

            avg_oracle_reward /= n_trials

    def _train_step(self, bandit, policy):
        P, B = policy, bandit
        C = B.get_context() if hasattr(B, "get_context") else None
        rwd, arm = P.act(B, C)
        oracle_rwd, oracle_arm = B.oracle_payoff(C)
        return rwd, arm, oracle_rwd, oracle_arm

