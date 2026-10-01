"""Restrict Minigrid actions to left / right / forward."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
from minigrid.core.actions import Actions


class NavActionWrapper(gym.ActionWrapper):
    """Map Discrete(3) onto Minigrid turn-left, turn-right, and move-forward."""

    _NAV_ACTIONS = (Actions.left, Actions.right, Actions.forward)

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Discrete(len(self._NAV_ACTIONS))

    def action(self, action: int) -> int:
        return int(self._NAV_ACTIONS[action])
