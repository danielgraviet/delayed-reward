"""Restrict Minigrid actions to left / right / forward / pickup / toggle."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
from minigrid.core.actions import Actions


class KeyDoorActionWrapper(gym.ActionWrapper):
    """Map Discrete(5) onto turn-left, turn-right, forward, pickup, and toggle."""

    _ACTIONS = (
        Actions.left,
        Actions.right,
        Actions.forward,
        Actions.pickup,
        Actions.toggle,
    )

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Discrete(len(self._ACTIONS))

    def action(self, action: int) -> int:
        return int(self._ACTIONS[action])
