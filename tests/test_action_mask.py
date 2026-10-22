"""Tests for the keydoor legal-action mask against a real MiniGrid DoorKey env."""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.wrappers import FullyObsWrapper, ReseedWrapper

from delayed_reward.environments.action_mask import (
    ActionMaskObservationWrapper,
    compute_keydoor_action_mask,
)
from delayed_reward.environments.binary_reward import BinarySuccessRewardWrapper
from delayed_reward.environments.factory import DQNObservationWrapper
from delayed_reward.environments.keydoor_actions import KeyDoorActionWrapper

# KeyDoorActionWrapper: 0=left, 1=right, 2=forward, 3=pickup, 4=toggle
LEFT, RIGHT, FORWARD, PICKUP, TOGGLE = 0, 1, 2, 3, 4

# Same fixed DoorKey-6x6 seed-0 path used in test_subgoal_delayed_reward.py:
# pick up the key, cross through the door, reach the goal.
PATH_TO_GOAL_6X6 = (
    LEFT, PICKUP, FORWARD, LEFT, FORWARD, FORWARD, RIGHT, TOGGLE,
    FORWARD, FORWARD, RIGHT, FORWARD, FORWARD, FORWARD,
)


def make_env(*, seed: int = 0) -> gym.Env:
    env: gym.Env = gym.make("MiniGrid-DoorKey-6x6-v0")
    env = KeyDoorActionWrapper(env)
    env = FullyObsWrapper(env)
    env = BinarySuccessRewardWrapper(env)
    env = ReseedWrapper(env, seeds=(seed,))
    return env


def test_mask_shape_and_turns_always_legal() -> None:
    env = make_env()
    env.reset()
    mask = compute_keydoor_action_mask(env.unwrapped)
    env.close()

    assert mask.shape == (5,)
    assert mask.dtype == np.float32
    assert mask[LEFT] == 1.0
    assert mask[RIGHT] == 1.0


def test_pickup_illegal_until_facing_key_then_illegal_again_once_carried() -> None:
    env = make_env()
    env.reset()

    # Right after reset the agent isn't adjacent to/facing the key yet.
    assert compute_keydoor_action_mask(env.unwrapped)[PICKUP] == 0.0

    # First path action (LEFT) turns the agent to face the key.
    env.step(PATH_TO_GOAL_6X6[0])
    assert compute_keydoor_action_mask(env.unwrapped)[PICKUP] == 1.0

    # Second path action (PICKUP) actually picks it up.
    env.step(PATH_TO_GOAL_6X6[1])
    assert env.unwrapped.carrying is not None
    assert compute_keydoor_action_mask(env.unwrapped)[PICKUP] == 0.0
    env.close()


def test_toggle_illegal_without_key_legal_once_carrying_it() -> None:
    env = make_env()
    env.reset()
    for action in PATH_TO_GOAL_6X6[:7]:  # up to, but not including, TOGGLE
        env.step(action)
    mask = compute_keydoor_action_mask(env.unwrapped)
    assert mask[TOGGLE] == 1.0  # carrying the key and facing the locked door

    env.step(PATH_TO_GOAL_6X6[7])
    mask_after_open = compute_keydoor_action_mask(env.unwrapped)
    assert mask_after_open[TOGGLE] == 1.0  # still has an effect (would close it)
    env.close()


def test_action_mask_observation_wrapper_appends_matching_mask() -> None:
    env = make_env()
    env = DQNObservationWrapper(env)
    env = ActionMaskObservationWrapper(env)
    obs, _info = env.reset()

    base_dim = env.observation_space.shape[0] - 5
    appended_mask = obs[base_dim:]
    direct_mask = compute_keydoor_action_mask(env.unwrapped)
    np.testing.assert_array_equal(appended_mask, direct_mask)
    env.close()
