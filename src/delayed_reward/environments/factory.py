"""Factory for a fixed, fully observable Empty Minigrid environment."""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid-* envs
from minigrid.wrappers import FullyObsWrapper, ReseedWrapper

from delayed_reward.encoding.pose import PoseEncoder
from delayed_reward.environments.adapter import MinigridAdapter
from delayed_reward.environments.binary_reward import BinarySuccessRewardWrapper
from delayed_reward.environments.delayed_reward import DelayedRewardWrapper
from delayed_reward.environments.nav_actions import NavActionWrapper

DEFAULT_ENV_ID = "MiniGrid-Empty-8x8-v0"


def create_empty_env(
    *,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = 0,
    delay: int = 0,
    render_mode: str | None = None,
) -> MinigridAdapter:
    """Build a fixed Empty Minigrid with optional delayed goal reward.

    Parameters
    ----------
    delay:
        Steps after goal contact before the success reward is delivered and the
        episode terminates. ``0`` restores the default immediate-reward MDP.
    """
    env: gym.Env = gym.make(env_id, render_mode=render_mode)
    env = NavActionWrapper(env)
    env = FullyObsWrapper(env)
    env = BinarySuccessRewardWrapper(env)
    env = DelayedRewardWrapper(env, delay=delay)
    env = ReseedWrapper(env, seeds=(seed,))
    return MinigridAdapter(env, encoder=PoseEncoder())

