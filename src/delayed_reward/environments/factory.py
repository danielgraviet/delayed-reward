"""Factory for a fixed, fully observable Empty-5x5 Minigrid environment."""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid-* envs
from minigrid.wrappers import FullyObsWrapper, ReseedWrapper

from delayed_reward.encoding.pose import PoseEncoder
from delayed_reward.environments.adapter import MinigridAdapter
from delayed_reward.environments.nav_actions import NavActionWrapper


def create_empty_5x5(*, seed: int = 0, render_mode: str | None = None) -> MinigridAdapter:
    """Build MiniGrid-Empty-5x5 with fixed layout/start and navigation actions only."""
    env: gym.Env = gym.make("MiniGrid-Empty-8x8-v0", render_mode=render_mode)
    env = NavActionWrapper(env)
    env = FullyObsWrapper(env)
    env = ReseedWrapper(env, seeds=(seed,))
    return MinigridAdapter(env, encoder=PoseEncoder())
