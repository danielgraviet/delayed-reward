"""Fuzz the real wrapper stacks with random action sequences and check invariants that must always hold."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import example, given, strategies as st

from delayed_reward.environments.factory import create_empty_dqn_env, create_keydoor_dqn_env
from delayed_reward.environments.keydoor_actions import KeyDoorActionWrapper
from delayed_reward.environments.layout import LayoutRandomization
from delayed_reward.environments.registry import HALLWAY_KEYDOOR_ENV_ID, KEYDOOR_DISTRACTOR_ENV_ID
from delayed_reward.environments.unwrap import minigrid_env

N_KEYDOOR_ACTIONS = len(KeyDoorActionWrapper.ACTIONS)
MAX_STEPS = 40


def _core_state(env) -> tuple:
    core = minigrid_env(env)
    carrying = None if core.carrying is None else (core.carrying.type, core.carrying.color)
    return tuple(int(v) for v in core.agent_pos), int(core.agent_dir), carrying, core.grid.encode().tobytes()


@given(
    env_id=st.sampled_from([HALLWAY_KEYDOOR_ENV_ID, KEYDOOR_DISTRACTOR_ENV_ID]),
    delay=st.integers(0, 30),
    subgoal_reward=st.sampled_from([0.0, 0.3, 1.0]),
    step_penalty=st.sampled_from([0.0, 0.001]),
    randomize=st.builds(LayoutRandomization, st.booleans(), st.booleans(), st.booleans()),
    seed=st.integers(0, 2**16),
    actions=st.lists(st.integers(0, N_KEYDOOR_ACTIONS - 1), min_size=1, max_size=120),
)
def test_keydoor_stack_invariants(*, env_id, delay, subgoal_reward, step_penalty, randomize, seed, actions) -> None:
    """Across episodes: valid observations, finite bounded returns, one subgoal payout, masked actions are no-ops."""
    env = create_keydoor_dqn_env(
        env_id=env_id,
        seed=seed,
        delay=delay,
        subgoal_reward=subgoal_reward,
        max_steps=MAX_STEPS,
        step_penalty=step_penalty,
        randomize=randomize,
    )
    obs, _ = env.reset()
    episode_return, episode_steps, subgoal_paid = 0.0, 0, 0
    try:
        for action in actions:
            mask = obs[-N_KEYDOOR_ACTIONS:]
            before = _core_state(env)
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert math.isfinite(float(reward))
            episode_return += float(reward)
            episode_steps += 1
            subgoal_paid += bool(info.get("subgoal_reward_delivered"))
            if mask[action] == 0 and not (terminated or truncated):
                assert _core_state(env) == before
            assert episode_steps <= MAX_STEPS
            assert subgoal_paid <= 1
            assert episode_return <= 1.0 + subgoal_reward + 1e-9
            assert episode_return >= -step_penalty * episode_steps - 1e-9
            if terminated or truncated:
                obs, _ = env.reset()
                episode_return, episode_steps, subgoal_paid = 0.0, 0, 0
    finally:
        env.close()


@given(
    delay=st.integers(0, 15),
    seed=st.integers(0, 2**16),
    actions=st.lists(st.integers(0, 2), min_size=1, max_size=150),
)
@example(delay=0, seed=0, actions=[2, 2, 0, 0, 0, 2, 2])
@example(delay=3, seed=0, actions=[2, 2, 1, 2, 2, 0, 0, 0, 0])
def test_empty_delayed_reward_invariants(delay, seed, actions) -> None:
    """Each episode pays 0 or 1, exactly ``delay`` steps after goal contact unless truncation cuts it short."""
    env = create_empty_dqn_env(env_id="MiniGrid-Empty-5x5-v0", seed=seed, delay=delay, max_steps=MAX_STEPS)
    env.reset()
    episode_return, step, goal_step = 0.0, 0, None
    try:
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            step += 1
            episode_return += float(reward)
            if info.get("goal_reached"):
                assert goal_step is None
                goal_step = step
            if float(reward) > 0:
                assert terminated
                assert goal_step is not None
                assert step == goal_step + delay or truncated
            if terminated or truncated:
                assert episode_return in (0.0, 1.0)
                assert (episode_return == 1.0) == (goal_step is not None)
                env.reset()
                episode_return, step, goal_step = 0.0, 0, None
    finally:
        env.close()


def test_masked_action_check_is_not_vacuous() -> None:
    """The keydoor fuzz only proves something if random play actually hits masked actions."""
    env = create_keydoor_dqn_env(env_id=HALLWAY_KEYDOOR_ENV_ID, seed=0, max_steps=MAX_STEPS)
    obs, _ = env.reset()
    rng = np.random.default_rng(0)
    masked = 0
    for _ in range(200):
        action = int(rng.integers(N_KEYDOOR_ACTIONS))
        masked += obs[-N_KEYDOOR_ACTIONS:][action] == 0
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    assert masked >= 20
