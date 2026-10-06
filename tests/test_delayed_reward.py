"""Tests for DelayedRewardWrapper against a real MiniGrid Empty env."""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
import pytest
from minigrid.wrappers import ReseedWrapper

from delayed_reward.environments.binary_reward import BinarySuccessRewardWrapper
from delayed_reward.environments.delayed_reward import DelayedRewardWrapper
from delayed_reward.environments.nav_actions import NavActionWrapper

# NavActionWrapper: 0=left, 1=right, 2=forward
LEFT, RIGHT, FORWARD = 0, 1, 2

# Empty-5x5: start (1,1) facing right; goal at (3,3).
PATH_TO_GOAL_5X5 = (FORWARD, FORWARD, RIGHT, FORWARD, FORWARD)


def make_env(*, delay: int, max_steps: int | None = None, seed: int = 0) -> gym.Env:
    """Real MiniGrid stack with navigation actions + delayed reward."""
    kwargs: dict[str, object] = {}
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    env: gym.Env = gym.make("MiniGrid-Empty-5x5-v0", **kwargs)
    env = NavActionWrapper(env)
    env = BinarySuccessRewardWrapper(env)
    env = DelayedRewardWrapper(env, delay=delay)
    env = ReseedWrapper(env, seeds=(seed,))
    return env


def step_sequence(env: gym.Env, actions: tuple[int, ...]):
    """Apply actions; return the list of (reward, terminated, truncated, info)."""
    outcomes = []
    for action in actions:
        _obs, reward, terminated, truncated, info = env.step(action)
        outcomes.append((float(reward), bool(terminated), bool(truncated), dict(info)))
        if terminated or truncated:
            break
    return outcomes


def test_negative_delay_rejected() -> None:
    env = gym.make("MiniGrid-Empty-5x5-v0")
    with pytest.raises(ValueError, match="delay must be >= 0"):
        DelayedRewardWrapper(env, delay=-1)
    env.close()


def test_delay_zero_rewards_and_terminates_on_goal() -> None:
    env = make_env(delay=0)
    env.reset()
    outcomes = step_sequence(env, PATH_TO_GOAL_5X5)
    env.close()

    assert len(outcomes) == len(PATH_TO_GOAL_5X5)
    *pre_goal, goal = outcomes
    for reward, terminated, _truncated, _info in pre_goal:
        assert reward == 0.0
        assert terminated is False

    reward, terminated, truncated, info = goal
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert "delayed_reward_delivered" not in info


def test_delay_holds_reward_then_delivers_after_n_steps() -> None:
    delay = 2
    env = make_env(delay=delay)
    env.reset()

    to_goal = step_sequence(env, PATH_TO_GOAL_5X5)
    reward, terminated, truncated, info = to_goal[-1]
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info.get("goal_reached") is True
    assert env.unwrapped.agent_pos[0] == 3 and env.unwrapped.agent_pos[1] == 3

    # Spin in place for the delay countdown (turning does not leave the goal).
    during = step_sequence(env, (LEFT,) * delay)
    env.close()

    assert len(during) == delay
    *waiting, delivered = during
    for reward, terminated, _truncated, _info in waiting:
        assert reward == 0.0
        assert terminated is False

    reward, terminated, truncated, info = delivered
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info.get("delayed_reward_delivered") is True


def test_reset_clears_pending_reward() -> None:
    env = make_env(delay=5)
    env.reset()
    step_sequence(env, PATH_TO_GOAL_5X5)  # starts pending countdown

    env.reset()
    # A turn before any goal contact must stay zero / non-terminal.
    _obs, reward, terminated, truncated, info = env.step(LEFT)
    env.close()

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert "goal_reached" not in info
    assert "delayed_reward_delivered" not in info


def test_reentering_goal_during_delay_does_not_reset_timer() -> None:
    """Walk onto goal, leave, walk back; delivery still happens on original schedule."""
    delay = 3
    env = make_env(delay=delay)
    env.reset()
    step_sequence(env, PATH_TO_GOAL_5X5)

    # Facing down on goal. Turn around and step north off the goal, then back south.
    # After goal path: dir=1 (down). Left -> dir=0 (right); left -> dir=3 (up).
    leave_and_return = (LEFT, LEFT, FORWARD, LEFT, LEFT, FORWARD)
    outcomes = step_sequence(env, leave_and_return)
    env.close()

    # Original schedule: delivery on the 3rd post-goal step, regardless of re-entry.
    assert len(outcomes) == 3
    for reward, terminated, _truncated, _info in outcomes[:2]:
        assert reward == 0.0
        assert terminated is False

    reward, terminated, truncated, info = outcomes[2]
    assert reward == 1.0
    assert terminated is True
    assert info.get("delayed_reward_delivered") is True


def test_truncation_during_delay_delivers_pending_reward() -> None:
    # Path to goal is 5 steps; allow only 2 more before timeout while delay=5.
    env = make_env(delay=5, max_steps=7)
    env.reset()
    step_sequence(env, PATH_TO_GOAL_5X5)

    outcomes = step_sequence(env, (LEFT, LEFT, LEFT, LEFT, LEFT))
    env.close()

    assert len(outcomes) == 2  # steps 6 and 7; truncates on 7
    reward, terminated, truncated, info = outcomes[-1]
    assert reward == 1.0
    assert terminated is True
    assert truncated is True
    assert info.get("delayed_reward_delivered") is True
