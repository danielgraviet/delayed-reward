"""Delayed subgoal reward relabeling on a real SB3 DQN replay buffer."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer

from delayed_reward.agents.dqn.callbacks import EpisodeMetricsCallback
from delayed_reward.agents.dqn.model import build_model
from delayed_reward.config import DQNTrialConfig, EnvKind, EnvSpec, RewardAttribution
from delayed_reward.environments.registry import HALLWAY_KEYDOOR_ENV_ID
from delayed_reward.experiments.dqn import build_dqn_env

PICKUP = {"environment_step": 10}
RELEASE = {
    "environment_step": 60,
    "subgoal_reward_delivered": True,
    "subgoal_reward_origin_step": 10,
    "subgoal_reward_amount": 1.0,
}


def _attached(attribution: RewardAttribution) -> tuple[EpisodeMetricsCallback, DQN, ReplayBuffer]:
    config = DQNTrialConfig(env=EnvSpec(kind=EnvKind.KEYDOOR, env_id=HALLWAY_KEYDOOR_ENV_ID), total_timesteps=1_000)
    model = build_model(config, build_dqn_env(config.env, seed=0))
    callback = EpisodeMetricsCallback(reward_attribution=attribution)
    callback.init_callback(model)
    assert model.replay_buffer is not None
    return callback, model, model.replay_buffer


def _store_and_step(callback: EpisodeMetricsCallback, buffer: ReplayBuffer, info: dict, reward: float) -> None:
    obs = np.zeros((1, *buffer.obs_shape), dtype=np.float32)
    buffer.add(obs, obs, np.zeros((1, 1)), np.array([reward]), np.array([False]), [info])
    callback.update_locals({"infos": [info], "dones": [False]})
    callback.on_step()


def test_origin_attribution_moves_reward_to_source_transition() -> None:
    callback, _model, buffer = _attached(RewardAttribution.ORIGIN)
    _store_and_step(callback, buffer, PICKUP, reward=0.0)
    _store_and_step(callback, buffer, RELEASE, reward=1.0)
    assert buffer.rewards[0, 0] == 1.0
    assert buffer.rewards[1, 0] == 0.0


def test_arrival_attribution_keeps_reward_on_release_transition() -> None:
    callback, _model, buffer = _attached(RewardAttribution.ARRIVAL)
    _store_and_step(callback, buffer, PICKUP, reward=0.0)
    _store_and_step(callback, buffer, RELEASE, reward=1.0)
    assert buffer.rewards[0, 0] == 0.0
    assert buffer.rewards[1, 0] == 1.0
