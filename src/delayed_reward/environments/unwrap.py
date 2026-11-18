"""Typed access to the MiniGrid environment underneath a wrapper stack."""

from __future__ import annotations

import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv


def minigrid_env(env: gym.Env) -> MiniGridEnv:
    """Return ``env.unwrapped`` as a ``MiniGridEnv``; raise ``TypeError`` for any other env."""
    core = env.unwrapped
    if not isinstance(core, MiniGridEnv):
        raise TypeError(f"expected a MiniGrid env underneath the wrappers, got {type(core).__name__}")
    return core
