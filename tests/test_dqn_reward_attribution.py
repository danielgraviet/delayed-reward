from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from delayed_reward.dqn_experiment import EpisodeMetricsCallback


class FakeReplayBuffer:
    buffer_size = 100

    def __init__(self) -> None:
        self.pos = 0
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)


def callback_step(
    callback: EpisodeMetricsCallback,
    buffer: FakeReplayBuffer,
    info: dict,
    reward: float = 0.0,
):
    inserted_index = buffer.pos
    buffer.rewards[inserted_index, 0] = reward
    buffer.pos = (buffer.pos + 1) % buffer.buffer_size
    callback.model = SimpleNamespace(replay_buffer=buffer)
    callback.locals = {"infos": [info], "dones": [False]}
    callback._on_step()


def test_origin_attribution_moves_reward_to_source_transition() -> None:
    buffer = FakeReplayBuffer()
    callback = EpisodeMetricsCallback(reward_attribution="origin")
    callback.model = SimpleNamespace(replay_buffer=buffer)
    callback_step(callback, buffer, {"environment_step": 10})
    source_index = 0
    assert buffer.rewards[source_index, 0] == 0.0
    callback_step(
        callback,
        buffer,
        {
            "environment_step": 60,
            "subgoal_reward_delivered": True,
            "subgoal_reward_origin_step": 10,
            "subgoal_reward_amount": 1.0,
        },
        reward=1.0,
    )
    arrival_index = 1
    assert buffer.rewards[source_index, 0] == 1.0
    assert buffer.rewards[arrival_index, 0] == 0.0


def test_arrival_attribution_keeps_reward_on_release_transition() -> None:
    buffer = FakeReplayBuffer()
    callback = EpisodeMetricsCallback(reward_attribution="arrival")
    callback.model = SimpleNamespace(replay_buffer=buffer)
    callback_step(callback, buffer, {"environment_step": 10})
    callback_step(callback, buffer, {
            "environment_step": 60,
            "subgoal_reward_delivered": True,
            "subgoal_reward_origin_step": 10,
            "subgoal_reward_amount": 1.0,
        }, reward=1.0)
    assert buffer.rewards[1, 0] == 1.0
