"""Build the wrapped MiniGrid environments the experiments train on."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, ReseedWrapper

from delayed_reward.environments.action_mask import ActionMaskObservationWrapper
from delayed_reward.environments.adapter import MinigridAdapter
from delayed_reward.environments.binary_reward import BinarySuccessRewardWrapper
from delayed_reward.environments.delayed_reward import DelayedRewardWrapper
from delayed_reward.environments.keydoor_actions import KeyDoorActionWrapper
from delayed_reward.environments.layout import NO_RANDOMIZATION, LayoutRandomization, SeedOnceWrapper
from delayed_reward.environments.nav_actions import NavActionWrapper
from delayed_reward.environments.observation import DQNObservationWrapper
from delayed_reward.environments.pose_encoder import PoseEncoder
from delayed_reward.environments.registry import (
    DEFAULT_ENV_ID,
    HALLWAY_KEYDOOR_ENV_ID,
    KEYDOOR_DISTRACTOR_ENV_ID,
    KEYDOOR_ENV_ID,
    supports_layout_randomization,
)
from delayed_reward.environments.step_penalty import StepPenaltyWrapper
from delayed_reward.environments.subgoal_delayed_reward import (
    SubgoalDelayedRewardWrapper,
)


def _make(env_id: str, *, render_mode: str | None, max_steps: int | None, **kwargs: Any) -> gym.Env:
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        kwargs["max_steps"] = max_steps
    return gym.make(env_id, render_mode=render_mode, **kwargs)


def _empty_env(*, env_id: str, seed: int, delay: int, render_mode: str | None, max_steps: int | None) -> gym.Env:
    """Fully observed Empty MiniGrid with the delayed binary goal reward, reseeded every episode."""
    env = _make(env_id, render_mode=render_mode, max_steps=max_steps)
    env = NavActionWrapper(env)
    env = FullyObsWrapper(env)
    env = BinarySuccessRewardWrapper(env)
    env = DelayedRewardWrapper(env, delay=delay)
    return ReseedWrapper(env, seeds=(seed,))


def create_empty_env(
    *,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = 0,
    delay: int = 0,
    render_mode: str | None = None,
    max_steps: int | None = None,
) -> MinigridAdapter:
    """Build a fixed Empty Minigrid with optional delayed goal reward, for the tabular agent.

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
    env = _empty_env(env_id=env_id, seed=seed, delay=delay, render_mode=render_mode, max_steps=max_steps)
    return MinigridAdapter(env, encoder=PoseEncoder())


def create_empty_dqn_env(
    *,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = 0,
    delay: int = 0,
    render_mode: str | None = None,
    max_steps: int | None = None,
) -> gym.Env:
    """Create a Gymnasium MiniGrid env with a flat numeric DQN observation."""
    env = _empty_env(env_id=env_id, seed=seed, delay=delay, render_mode=render_mode, max_steps=max_steps)
    return DQNObservationWrapper(env)


def create_keydoor_dqn_env(
    *,
    env_id: str = KEYDOOR_ENV_ID,
    seed: int = 0,
    delay: int = 0,
    subgoal_reward: float = 1.0,
    render_mode: str | None = None,
    max_steps: int | None = None,
    num_distractors: int = 0,
    step_penalty: float = 0.0,
    randomize: LayoutRandomization = NO_RANDOMIZATION,
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

    ``step_penalty`` subtracts a fixed cost from every step's reward (applied
    after the delayed/subgoal reward, so the living cost itself is never
    delayed and is identical across all ``delay`` settings). See
    ``StepPenaltyWrapper``.

    ``randomize`` re-samples the chosen layout elements (agent start, key,
    door) uniformly every episode; the rest stay at the env's fixed positions.
    Like ``num_distractors`` it needs an env backed by ``RectangularDoorKeyEnv``
    (see ``supports_layout_randomization``); ``KEYDOOR_DISTRACTOR_ENV_ID`` is
    the 8x8 one.
    """
    kwargs: dict[str, Any] = {}
    if num_distractors:
        kwargs["num_distractors"] = num_distractors
    if randomize.any:
        if not supports_layout_randomization(env_id):
            raise ValueError(
                f"{env_id} does not support layout randomization; use an env backed by "
                f"RectangularDoorKeyEnv, e.g. {KEYDOOR_DISTRACTOR_ENV_ID} or {HALLWAY_KEYDOOR_ENV_ID}"
            )
        kwargs["randomize"] = randomize
    env = _make(env_id, render_mode=render_mode, max_steps=max_steps, **kwargs)
    env = KeyDoorActionWrapper(env)
    env = FullyObsWrapper(env)
    env = BinarySuccessRewardWrapper(env)
    env = SubgoalDelayedRewardWrapper(env, delay=delay, subgoal_reward=subgoal_reward)
    if step_penalty:
        env = StepPenaltyWrapper(env, penalty=step_penalty)
    env = SeedOnceWrapper(env, seed) if randomize.any else ReseedWrapper(env, seeds=(seed,))
    env = DQNObservationWrapper(env)
    env = ActionMaskObservationWrapper(env)
    return env
