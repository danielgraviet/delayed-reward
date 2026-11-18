"""Construct the SB3 DQN learner from a ``DQNTrialConfig``."""

from __future__ import annotations

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.utils import LinearSchedule

from delayed_reward.agents.dqn.double_dqn import DoubleDQN
from delayed_reward.agents.dqn.masked_policy import MaskedDQNPolicy
from delayed_reward.config import DQNParams, DQNTrialConfig, LRSchedule


def make_lr_schedule(params: DQNParams, end_fraction: float = 1.0) -> float | LinearSchedule:
    """Return SB3's ``learning_rate`` argument: a constant, or a linear anneal.

    ``linear`` decays from ``learning_rate`` at the first update to ``lr_final``
    after ``end_fraction`` of training (SB3 passes the remaining progress,
    1 -> 0) and holds ``lr_final`` after that.
    """
    if not 0 < end_fraction <= 1:
        raise ValueError(f"end_fraction must be in (0, 1], got {end_fraction}")
    if params.lr_schedule is LRSchedule.LINEAR:
        return LinearSchedule(params.learning_rate, params.lr_final, end_fraction)
    return params.learning_rate


def build_model(config: DQNTrialConfig, env: gym.Env) -> DQN:
    """Construct the SB3 learner from ``config``; every hyperparameter comes from ``config.params``."""
    params = config.params
    algorithm = DoubleDQN if params.double_dqn else DQN
    return algorithm(
        MaskedDQNPolicy if config.env.is_keydoor else "MlpPolicy",
        env,
        policy_kwargs={"net_arch": list(params.network)},
        learning_rate=make_lr_schedule(params, config.schedule_fraction),
        buffer_size=params.buffer_size,
        learning_starts=params.learning_starts,
        batch_size=params.batch_size,
        gamma=params.gamma,
        train_freq=params.train_freq,
        target_update_interval=params.target_update_interval,
        exploration_fraction=params.exploration_fraction * config.schedule_fraction,
        exploration_initial_eps=params.exploration_initial_eps,
        exploration_final_eps=params.exploration_final_eps,
        max_grad_norm=params.max_grad_norm,
        seed=config.trial_seed,
        device="auto",
        verbose=0,
    )
