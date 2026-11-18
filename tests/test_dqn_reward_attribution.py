"""Delayed subgoal reward relabeling, checked inside SB3's real ``learn`` loop.

SB3 calls callbacks before it stores each step's transition, so the only honest
test drives ``model.learn`` and then reads the replay buffer it filled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from stable_baselines3.common.callbacks import BaseCallback

from delayed_reward.agents.dqn.callbacks import EpisodeMetricsCallback
from delayed_reward.agents.dqn.model import build_model
from delayed_reward.config import DQNParams, DQNTrialConfig, EnvKind, EnvSpec, RewardAttribution
from delayed_reward.environments.registry import HALLWAY_KEYDOOR_ENV_ID
from delayed_reward.experiments.dqn import build_dqn_env

PENALTY = 0.001
SUBGOAL = 0.3
STEPS = 6_000


@dataclass
class _Step:
    slot: int
    picked_up: bool
    delivered: bool


class _SlotRecorder(BaseCallback):
    """Remember, per step, which replay slot SB3 is about to store that step's transition in."""

    def __init__(self) -> None:
        super().__init__()
        self.steps: list[_Step] = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        buffer = self.model.replay_buffer  # type: ignore[attr-defined]
        self.steps.append(
            _Step(
                slot=buffer.pos, picked_up=bool(info.get("subgoal_reached")), delivered="subgoal_reward_amount" in info
            )
        )
        return True


def _collect(attribution: RewardAttribution, delay: int) -> tuple[np.ndarray, list[_Step]]:
    """Fill a real replay buffer with exploration-only steps (no gradient updates) and return it."""
    config = DQNTrialConfig(
        env=EnvSpec(
            kind=EnvKind.KEYDOOR,
            env_id=HALLWAY_KEYDOOR_ENV_ID,
            max_steps=200,
            delay=delay,
            subgoal_reward=SUBGOAL,
            step_penalty=PENALTY,
        ),
        total_timesteps=STEPS,
        params=DQNParams(learning_starts=STEPS + 1),
        seed=3,
    )
    model = build_model(config, build_dqn_env(config.env, seed=config.trial_seed))
    recorder = _SlotRecorder()
    model.learn(STEPS, callback=[EpisodeMetricsCallback(reward_attribution=attribution), recorder])
    assert model.replay_buffer is not None
    return model.replay_buffer.rewards[:STEPS, 0].copy(), recorder.steps


@pytest.mark.parametrize("delay", [1, 4])
def test_origin_moves_the_delayed_reward_onto_the_pickup_transition(delay: int) -> None:
    rewards, steps = _collect(RewardAttribution.ORIGIN, delay)
    pickups = [s for s in steps if s.picked_up]
    deliveries = [s for s in steps if s.delivered]
    assert len(pickups) >= 3
    for step in pickups:
        assert rewards[step.slot] == pytest.approx(SUBGOAL - PENALTY)
    for step in deliveries:
        assert rewards[step.slot] <= -PENALTY + 1e-6
    others = {s.slot for s in steps} - {s.slot for s in pickups}
    assert not np.any(rewards[sorted(others)] > 0.5)


@pytest.mark.parametrize("delay", [1, 4])
def test_arrival_keeps_the_delayed_reward_where_it_arrived(delay: int) -> None:
    rewards, steps = _collect(RewardAttribution.ARRIVAL, delay)
    pickups = [s for s in steps if s.picked_up]
    deliveries = [s for s in steps if s.delivered]
    assert len(pickups) >= 3
    for step in pickups:
        assert rewards[step.slot] == pytest.approx(-PENALTY)
    for step in deliveries:
        assert rewards[step.slot] >= SUBGOAL - PENALTY - 1e-6


def test_relabeling_conserves_total_reward() -> None:
    """Same seed, same exploration: origin only moves reward between transitions."""
    origin, _ = _collect(RewardAttribution.ORIGIN, 4)
    arrival, _ = _collect(RewardAttribution.ARRIVAL, 4)
    assert origin.sum() == pytest.approx(arrival.sum(), abs=1e-4)
    assert not np.allclose(origin, arrival)
