"""DQN policy that masks illegal actions using a mask appended to the observation.

Pairs with ``delayed_reward.environments.action_mask.ActionMaskObservationWrapper``,
which appends a 0/1 legal-action bit per action to the end of the flat
observation vector. Masking inside the Q-network's ``forward`` (rather than
only at prediction time) means every downstream use of the Q-values —
greedy action selection, epsilon-greedy's greedy branch, and the Bellman
target computed from ``q_net_target`` during training — automatically
respects it, with no need to subclass ``DQN`` itself.
"""

from __future__ import annotations

import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

_ILLEGAL_Q_VALUE = -1.0e8


class MaskedQNetwork(QNetwork):
    """QNetwork that reads a legal-action mask from the last N obs columns."""

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        q_values = self.q_net(self.extract_features(obs, self.features_extractor))
        n_actions = int(self.action_space.n)
        mask = obs[..., -n_actions:]
        return q_values.masked_fill(mask < 0.5, _ILLEGAL_Q_VALUE)


class MaskedDQNPolicy(DQNPolicy):
    """DQNPolicy whose online and target Q-networks are ``MaskedQNetwork``."""

    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQNetwork(**net_args).to(self.device)
