"""Factory for a fixed, fully observable Empty Minigrid environment."""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid-* envs
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from minigrid.wrappers import FullyObsWrapper, ReseedWrapper

from delayed_reward.encoding.pose import PoseEncoder
from delayed_reward.environments.action_mask import ActionMaskObservationWrapper
from delayed_reward.environments.adapter import MinigridAdapter
from delayed_reward.environments.binary_reward import BinarySuccessRewardWrapper
from delayed_reward.environments.delayed_reward import DelayedRewardWrapper
from delayed_reward.environments.keydoor_actions import KeyDoorActionWrapper
from delayed_reward.environments.nav_actions import NavActionWrapper
from delayed_reward.environments.subgoal_delayed_reward import SubgoalDelayedRewardWrapper

DEFAULT_ENV_ID = "MiniGrid-Empty-10x10-v0"
RECTANGULAR_ENV_ID = "MiniGrid-Empty-20x8-v0"
KEYDOOR_ENV_ID = "MiniGrid-DoorKey-8x8-v0"
KEYDOOR_DISTRACTOR_ENV_ID = "MiniGrid-DoorKey-8x8-Distractors-v0"
HALLWAY_KEYDOOR_ENV_ID = "MiniGrid-DoorKey-Hallway-20x6-v0"

if DEFAULT_ENV_ID not in registry:
    register(
        id=DEFAULT_ENV_ID,
        entry_point="minigrid.envs:EmptyEnv",
        kwargs={"size": 10},
    )

if RECTANGULAR_ENV_ID not in registry:
    register(
        id=RECTANGULAR_ENV_ID,
        entry_point="delayed_reward.environments.rectangular:RectangularEmptyEnv",
        kwargs={"width": 20, "height": 8},
    )

if HALLWAY_KEYDOOR_ENV_ID not in registry:
    register(
        id=HALLWAY_KEYDOOR_ENV_ID,
        entry_point="delayed_reward.environments.rectangular:RectangularDoorKeyEnv",
        kwargs={"width": 20, "height": 6},
    )

# KEYDOOR_ENV_ID is MiniGrid's own built-in DoorKeyEnv, which doesn't accept
# num_distractors. This variant reproduces its 8x8 layout through our
# RectangularDoorKeyEnv so distractor robustness can be checked against the
# same room size the default keydoor training uses.
if KEYDOOR_DISTRACTOR_ENV_ID not in registry:
    register(
        id=KEYDOOR_DISTRACTOR_ENV_ID,
        entry_point="delayed_reward.environments.rectangular:RectangularDoorKeyEnv",
        kwargs={"width": 8, "height": 8},
    )


def create_empty_env(
    *,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = 0,
    delay: int = 0,
    render_mode: str | None = None,
    max_steps: int | None = None,
) -> MinigridAdapter:
    """Build a fixed Empty Minigrid with optional delayed goal reward.

    Parameters
    ----------
    delay:
        Steps after goal contact before the success reward is delivered and the
        episode terminates. ``0`` restores the default immediate-reward MDP.
    max_steps:
        Episode truncation horizon. ``None`` keeps MiniGrid's default
        (``4 * size**2``, 400 on Empty-10x10). A pending delayed reward is
        delivered at truncation, so delays longer than this horizon are cut short.
    """
    kwargs: dict[str, object] = {"render_mode": render_mode}
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        kwargs["max_steps"] = max_steps
    env: gym.Env = gym.make(env_id, **kwargs)
    env = NavActionWrapper(env)
    env = FullyObsWrapper(env)
    env = BinarySuccessRewardWrapper(env)
    env = DelayedRewardWrapper(env, delay=delay)
    env = ReseedWrapper(env, seeds=(seed,))
    return MinigridAdapter(env, encoder=PoseEncoder())


class DQNObservationWrapper(gym.ObservationWrapper):
    """Flatten the fully observed grid and agent pose into a numeric vector."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        image_space = env.observation_space["image"]  # type: ignore[index]
        self._image_shape = image_space.shape
        width, height = int(image_space.shape[0]), int(image_space.shape[1])
        self.observation_space = spaces.Box(
            low=0.0,
            high=255.0,
            shape=(int(np.prod(image_space.shape)) + 6,),
            dtype=np.float32,
        )
        self._width = width
        self._height = height

    def observation(self, observation: dict[str, object]) -> np.ndarray:
        grid = np.asarray(observation["image"], dtype=np.float32).reshape(-1)
        unwrapped = self.env.unwrapped
        x, y = (int(value) for value in unwrapped.agent_pos)
        direction = int(unwrapped.agent_dir)
        pose = np.zeros(6, dtype=np.float32)
        pose[0] = x / max(1, self._width - 1)
        pose[1] = y / max(1, self._height - 1)
        pose[2 + direction] = 1.0
        return np.concatenate((grid, pose))


def create_empty_dqn_env(
    *,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = 0,
    delay: int = 0,
    render_mode: str | None = None,
    max_steps: int | None = None,
) -> gym.Env:
    """Create a Gymnasium MiniGrid env with a flat numeric DQN observation."""
    kwargs: dict[str, object] = {"render_mode": render_mode}
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        kwargs["max_steps"] = max_steps
    env: gym.Env = gym.make(env_id, **kwargs)
    env = NavActionWrapper(env)
    env = FullyObsWrapper(env)
    env = BinarySuccessRewardWrapper(env)
    env = DelayedRewardWrapper(env, delay=delay)
    env = ReseedWrapper(env, seeds=(seed,))
    env = DQNObservationWrapper(env)
    return env


def create_keydoor_dqn_env(
    *,
    env_id: str = KEYDOOR_ENV_ID,
    seed: int = 0,
    delay: int = 0,
    subgoal_reward: float = 1.0,
    render_mode: str | None = None,
    max_steps: int | None = None,
    num_distractors: int = 0,
) -> gym.Env:
    """Create a key-door MiniGrid with a flat numeric DQN observation.

    Only the key-pickup subgoal reward is delayed; reaching the goal always
    terminates and pays its reward instantly. See ``SubgoalDelayedRewardWrapper``.

    ``num_distractors`` scatters that many cosmetic, non-interactive colored
    tiles on the agent's side of the room (see ``environments.distractors``)
    to probe robustness to irrelevant visual clutter. Only ``env_id``s backed
    by ``RectangularDoorKeyEnv`` (e.g. ``KEYDOOR_DISTRACTOR_ENV_ID`` or
    ``HALLWAY_KEYDOOR_ENV_ID``) accept it; passing a nonzero value with
    MiniGrid's own built-in ``KEYDOOR_ENV_ID`` raises a ``TypeError``.
    """
    kwargs: dict[str, object] = {"render_mode": render_mode}
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        kwargs["max_steps"] = max_steps
    if num_distractors:
        kwargs["num_distractors"] = num_distractors
    env: gym.Env = gym.make(env_id, **kwargs)
    env = KeyDoorActionWrapper(env)
    env = FullyObsWrapper(env)
    env = BinarySuccessRewardWrapper(env)
    env = SubgoalDelayedRewardWrapper(env, delay=delay, subgoal_reward=subgoal_reward)
    env = ReseedWrapper(env, seeds=(seed,))
    env = DQNObservationWrapper(env)
    env = ActionMaskObservationWrapper(env)
    return env

