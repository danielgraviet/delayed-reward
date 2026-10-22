"""Legal-action mask for ``KeyDoorActionWrapper``'s 5-action space.

The DQN greedy policy was observed to get stuck permanently repeating a
no-op action (``pickup`` far from the key, ``forward`` into a wall) because
nothing in training discourages a network from assigning those states a high
Q-value. Appending a legal/illegal bit per action to the observation, and
masking illegal actions out before the Q-network's argmax (see
``delayed_reward.masked_dqn_policy``), removes that failure mode instead of
hoping the network learns to avoid it.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.core.world_object import Door, Key, WorldObj

# Mirrors KeyDoorActionWrapper._ACTIONS: left, right, forward, pickup, toggle.
N_ACTIONS = 5


def compute_keydoor_action_mask(unwrapped) -> np.ndarray:
    """Return a ``(5,)`` float32 mask of which actions have an effect right now.

    ``left``/``right`` (turning) are always legal. ``forward`` is legal only
    if the cell ahead is empty or overlappable. ``pickup`` is legal only if
    the cell ahead holds a pickup-able object and the agent isn't already
    carrying something. ``toggle`` is legal only if it would actually open a
    door (unlocked, or locked with the matching key in hand).
    """
    front_cell: WorldObj | None = unwrapped.grid.get(*unwrapped.front_pos)

    can_forward = front_cell is None or front_cell.can_overlap()
    can_pickup = (
        front_cell is not None
        and front_cell.can_pickup()
        and unwrapped.carrying is None
    )
    can_toggle = isinstance(front_cell, Door) and (
        not front_cell.is_locked
        or (isinstance(unwrapped.carrying, Key) and unwrapped.carrying.color == front_cell.color)
    )
    return np.array(
        [1.0, 1.0, float(can_forward), float(can_pickup), float(can_toggle)],
        dtype=np.float32,
    )


class ActionMaskObservationWrapper(gym.ObservationWrapper):
    """Append the legal-action mask to a flat Box observation."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        base_space = env.observation_space
        assert isinstance(base_space, spaces.Box) and base_space.shape is not None
        low = np.concatenate([base_space.low, np.zeros(N_ACTIONS, dtype=np.float32)])
        high = np.concatenate([base_space.high, np.ones(N_ACTIONS, dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        mask = compute_keydoor_action_mask(self.env.unwrapped)
        return np.concatenate([observation, mask]).astype(np.float32)
