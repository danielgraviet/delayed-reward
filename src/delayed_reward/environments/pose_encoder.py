"""Pose-based state encoding for fully observable fixed grids."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

import gymnasium as gym

from delayed_reward.environments.unwrap import minigrid_env


class PoseEncoder:
    """Encode agent pose as (x, y, direction); ignore step count and grid pixels."""

    def encode(self, observation: Mapping[str, object], env: gym.Env) -> Hashable:
        """Read the pose from the env itself; under full observability it determines the state."""
        del observation
        core = minigrid_env(env)
        x, y = int(core.agent_pos[0]), int(core.agent_pos[1])
        return (x, y, int(core.agent_dir))
