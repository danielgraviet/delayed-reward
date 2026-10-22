"""Tests for SubgoalDelayedRewardWrapper against a real MiniGrid DoorKey env."""

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

# KeyDoorActionWrapper: 0=left, 1=right, 2=forward, 3=pickup, 4=toggle
LEFT, RIGHT, FORWARD, PICKUP, TOGGLE = 0, 1, 2, 3, 4

# DoorKey-6x6 seed 0: pick up the key, cross through the door, reach the goal.
# Pickup happens on step index 1; the goal is reached on step index 13.
PATH_TO_GOAL_6X6 = (
    LEFT, PICKUP, FORWARD, LEFT, FORWARD, FORWARD, RIGHT, TOGGLE,
    FORWARD, FORWARD, RIGHT, FORWARD, FORWARD, FORWARD,
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

    delivered_reward, delivered_terminated, _t, delivered_info = outcomes[
        PICKUP_STEP_INDEX + delay
    ]
    assert delivered_reward == 1.0
    assert delivered_terminated is False
    assert delivered_info.get("subgoal_reward_delivered") is True

    # The goal, reached later on the same fixed path, still pays out instantly.
    goal_reward, goal_terminated, _t, _info = outcomes[GOAL_STEP_INDEX]
    assert goal_reward == 1.0
    assert goal_terminated is True


def test_goal_ends_episode_instantly_even_with_subgoal_delay_still_pending() -> None:
    # Delay outlives the remaining path to the goal (12 steps after pickup):
    # the goal must still end the episode on contact, paying out both rewards.
    env = make_env(delay=20)
    env.reset()
    outcomes = step_sequence(env, PATH_TO_GOAL_6X6)
    env.close()

    assert len(outcomes) == len(PATH_TO_GOAL_6X6)
    reward, terminated, truncated, info = outcomes[-1]
    assert reward == 2.0  # goal (1.0) + the still-pending subgoal reward (1.0)
    assert terminated is True
    assert truncated is False
    assert info.get("subgoal_reward_delivered") is True


def test_reset_clears_pending_subgoal_state() -> None:
    env = make_env(delay=5)
    env.reset()
    step_sequence(env, PATH_TO_GOAL_6X6[: PICKUP_STEP_INDEX + 1])  # picks up the key

    env.reset()
    _obs, reward, terminated, truncated, info = env.step(LEFT)
    env.close()

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert "subgoal_reached" not in info
    assert "subgoal_reward_delivered" not in info
