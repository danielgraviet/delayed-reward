"""Per-step living-cost reward wrapper.

The project's success rewards are otherwise sparse and binary (see
``binary_reward.py``), which leaves near-zero, near-tied Q-values for
idling/turning versus making progress toward the goal under heavy discounting.
A small per-step penalty breaks that tie so a greedy policy strictly prefers
reaching the goal sooner over spinning in place indefinitely.
"""

from __future__ import annotations

import gymnasium as gym


class StepPenaltyWrapper(gym.RewardWrapper):
    """Subtract a fixed penalty from every step's reward."""

    def __init__(self, env: gym.Env, *, penalty: float) -> None:
        if penalty < 0.0:
            raise ValueError("penalty must be >= 0 (it is subtracted from reward)")
        super().__init__(env)
        self.penalty = penalty

    def reward(self, reward: float) -> float:
        return float(reward) - self.penalty
