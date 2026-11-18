"""Double DQN target and its training wiring."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch as th

from delayed_reward.config import DQNParams, DQNTrialConfig, EnvKind, EnvSpec, EvalSpec
from delayed_reward.agents.dqn.double_dqn import DoubleDQN, double_dqn_next_q
from delayed_reward.environments.registry import HALLWAY_KEYDOOR_ENV_ID
from delayed_reward.experiments.dqn import train_trial


def test_online_net_picks_and_target_net_scores():
    # Row 0: online prefers action 1, target prefers action 0.
    # Plain DQN would use max(target) = 5; Double DQN uses target[1] = 2.
    q_online = th.tensor([[0.0, 1.0, -1.0], [3.0, 0.0, 0.0]])
    q_target = th.tensor([[5.0, 2.0, 0.0], [4.0, 9.0, 1.0]])
    assert double_dqn_next_q(q_online, q_target).tolist() == [[2.0], [4.0]]
    # Never larger than the plain-DQN target.
    assert (double_dqn_next_q(q_online, q_target) <= q_target.max(dim=1, keepdim=True).values).all()


def test_masked_actions_are_never_chosen():
    # A masked action carries the policy's -1e8 in both nets, so it can't be the online argmax.
    q_online = th.tensor([[-1.0e8, 0.5, 0.1]])
    q_target = th.tensor([[-1.0e8, 0.3, 7.0]])
    assert double_dqn_next_q(q_online, q_target).item() == pytest.approx(0.3)


def test_train_trial_uses_and_records_double_dqn():
    trial_dir = Path(tempfile.mkdtemp())
    config = DQNTrialConfig(
        env=EnvSpec(kind=EnvKind.KEYDOOR, env_id=HALLWAY_KEYDOOR_ENV_ID, step_penalty=0.001),
        total_timesteps=2_000,
        params=DQNParams(double_dqn=True),
        eval=EvalSpec(every_episodes=None),
    )
    train_trial(config, trial_dir)
    config = json.loads((trial_dir / "summary.json").read_text())["config"]
    assert config["double_dqn"] is True
    assert isinstance(DoubleDQN.load(trial_dir / "model"), DoubleDQN)
