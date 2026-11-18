"""Double DQN (van Hasselt, Guez & Silver, 2016) on top of Stable-Baselines3's DQN.

Plain DQN builds its TD target as ``r + gamma * max_a Q_target(s', a)``: the
target network both *chooses* and *scores* the next action, so noise in its
estimates is maximised over and the target is biased upward. Double DQN
decouples the two: the online network chooses ``a* = argmax_a Q_online(s', a)``
and the target network scores it, ``r + gamma * Q_target(s', a*)``.

Everything else (replay, Huber loss, gradient clipping, target sync, epsilon
schedule) is SB3's DQN unchanged. With ``MaskedDQNPolicy`` both networks mask
illegal actions in ``forward``, so ``a*`` is always a legal action.
"""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import DQN
from torch.nn import functional as F


def double_dqn_next_q(q_online_next: th.Tensor, q_target_next: th.Tensor) -> th.Tensor:
    """Target-network value of the online network's greedy next action, shape (batch, 1)."""
    next_actions = q_online_next.argmax(dim=1, keepdim=True)
    return th.gather(q_target_next, dim=1, index=next_actions)


class DoubleDQN(DQN):
    """SB3 ``DQN`` whose TD target uses Double DQN action selection."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Mirrors stable_baselines3 DQN.train (2.9.x); only the target differs.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_q_values = double_dqn_next_q(
                    self.q_net(replay_data.next_observations),
                    self.q_net_target(replay_data.next_observations),
                )
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
