"""Tests for SubgoalDelayedRewardWrapper against a real MiniGrid DoorKey env.

The fixed DoorKey-6x6 seed-0 path picks up the key on step index 1 and reaches the
goal on step index 13.
"""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
import pytest
from minigrid.wrappers import FullyObsWrapper, ReseedWrapper

from delayed_reward.environments.binary_reward import BinarySuccessRewardWrapper
from delayed_reward.environments.keydoor_actions import KeyDoorActionWrapper
from delayed_reward.environments.subgoal_delayed_reward import (
    SubgoalDelayedRewardWrapper,
)

LEFT, RIGHT, FORWARD, PICKUP, TOGGLE = 0, 1, 2, 3, 4

PATH_TO_GOAL_6X6 = (
    LEFT,
    PICKUP,
    FORWARD,
    LEFT,
    FORWARD,
    FORWARD,
    RIGHT,
    TOGGLE,
    FORWARD,
    FORWARD,
    RIGHT,
    FORWARD,
    FORWARD,
    FORWARD,
)
PICKUP_STEP_INDEX = 1
GOAL_STEP_INDEX = 13


def make_env(*, delay: int, subgoal_reward: float = 1.0, seed: int = 0) -> gym.Env:
    """Real MiniGrid DoorKey stack with the key-subgoal delay wrapper."""
    env: gym.Env = gym.make("MiniGrid-DoorKey-6x6-v0")
    env = KeyDoorActionWrapper(env)
    env = FullyObsWrapper(env)
    env = BinarySuccessRewardWrapper(env)
    env = SubgoalDelayedRewardWrapper(env, delay=delay, subgoal_reward=subgoal_reward)
    env = ReseedWrapper(env, seeds=(seed,))
    return env


def step_sequence(env: gym.Env, actions: tuple[int, ...]):
    outcomes = []
    for action in actions:
        _obs, reward, terminated, truncated, info = env.step(action)
        outcomes.append((float(reward), bool(terminated), bool(truncated), dict(info)))
        if terminated or truncated:
            break
    return outcomes


def test_negative_delay_rejected() -> None:
    env = gym.make("MiniGrid-DoorKey-6x6-v0")
    with pytest.raises(ValueError, match="delay must be >= 0"):
        SubgoalDelayedRewardWrapper(env, delay=-1)
    env.close()


def test_delay_zero_delivers_subgoal_reward_instantly_at_pickup() -> None:
    env = make_env(delay=0)
    env.reset()
    outcomes = step_sequence(env, PATH_TO_GOAL_6X6)
    env.close()

    assert len(outcomes) == len(PATH_TO_GOAL_6X6)
    reward, terminated, _truncated, info = outcomes[PICKUP_STEP_INDEX]
    assert reward == 1.0
    assert terminated is False
    assert info.get("subgoal_reached") is True
    assert info.get("subgoal_reward_delivered") is True

    goal_reward, goal_terminated, goal_truncated, _goal_info = outcomes[GOAL_STEP_INDEX]
    assert goal_reward == 1.0
    assert goal_terminated is True
    assert goal_truncated is False


def test_delay_holds_subgoal_reward_then_delivers_after_n_steps() -> None:
    """The subgoal reward arrives ``delay`` steps after pickup; the goal later on the path still pays instantly."""
    delay = 2
    env = make_env(delay=delay)
    env.reset()
    outcomes = step_sequence(env, PATH_TO_GOAL_6X6)
    env.close()

    pickup_reward, pickup_terminated, _t, pickup_info = outcomes[PICKUP_STEP_INDEX]
    assert pickup_reward == 0.0
    assert pickup_terminated is False
    assert pickup_info.get("subgoal_reached") is True
    assert "subgoal_reward_delivered" not in pickup_info

    for reward, terminated, _t, _info in outcomes[PICKUP_STEP_INDEX + 1 : PICKUP_STEP_INDEX + delay]:
        assert reward == 0.0
        assert terminated is False

    delivered_reward, delivered_terminated, _t, delivered_info = outcomes[PICKUP_STEP_INDEX + delay]
    assert delivered_reward == 1.0
    assert delivered_terminated is False
    assert delivered_info.get("subgoal_reward_delivered") is True

    goal_reward, goal_terminated, _t, _info = outcomes[GOAL_STEP_INDEX]
    assert goal_reward == 1.0
    assert goal_terminated is True


def test_pending_subgoal_reward_is_flushed_on_the_terminating_step() -> None:
    """When the delay outlives the path, the goal step pays goal + pending subgoal reward.

    Nothing leaks into the next episode.
    """
    env = make_env(delay=20)
    env.reset()
    outcomes = step_sequence(env, PATH_TO_GOAL_6X6)

    assert len(outcomes) == len(PATH_TO_GOAL_6X6)
    reward, terminated, truncated, info = outcomes[-1]
    assert reward == 2.0
    assert terminated is True
    assert truncated is False
    assert info["subgoal_reward_delivered"] is True
    assert info["subgoal_reward_origin_step"] == PICKUP_STEP_INDEX + 1
    assert info["subgoal_reward_amount"] == 1.0

    env.reset()
    for _ in range(30):
        _obs, reward, terminated, truncated, info = env.step(LEFT)
        assert reward == 0.0
        assert "subgoal_reward_delivered" not in info
        if terminated or truncated:
            break
    env.close()


def test_reset_clears_pending_subgoal_state() -> None:
    """A subgoal reward pending at reset is dropped."""
    env = make_env(delay=5)
    env.reset()
    step_sequence(env, PATH_TO_GOAL_6X6[: PICKUP_STEP_INDEX + 1])

    env.reset()
    _obs, reward, terminated, truncated, info = env.step(LEFT)
    env.close()

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert "subgoal_reached" not in info
    assert "subgoal_reward_delivered" not in info


def test_pending_subgoal_reward_is_flushed_on_truncation() -> None:
    """Picking up the key and spinning until truncation, before the 50-step delay, still pays the subgoal."""
    max_steps = 10
    env: gym.Env = gym.make("MiniGrid-DoorKey-6x6-v0", max_steps=max_steps)
    env = KeyDoorActionWrapper(env)
    env = FullyObsWrapper(env)
    env = BinarySuccessRewardWrapper(env)
    env = SubgoalDelayedRewardWrapper(env, delay=50, subgoal_reward=1.0)
    env = ReseedWrapper(env, seeds=(0,))
    env.reset()
    outcomes = step_sequence(env, PATH_TO_GOAL_6X6[: PICKUP_STEP_INDEX + 1] + (LEFT,) * max_steps)
    env.close()

    assert len(outcomes) == max_steps
    *before_end, (reward, terminated, truncated, info) = outcomes
    assert all(r == 0.0 and "subgoal_reward_delivered" not in i for r, _t, _tr, i in before_end)
    assert truncated is True
    assert terminated is False
    assert reward == 1.0
    assert info["subgoal_reward_delivered"] is True
    assert info["subgoal_reward_origin_step"] == PICKUP_STEP_INDEX + 1
