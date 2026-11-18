"""Double DQN target and its CLI / resume wiring."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch as th

from delayed_reward.double_dqn import DoubleDQN, double_dqn_next_q
from delayed_reward.dqn_experiment import train_trial
from delayed_reward.environments.factory import HALLWAY_KEYDOOR_ENV_ID

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import parallel_dqn_sweep  # noqa: E402


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


def _args(tmp_path: Path, *extra: str):
    return parallel_dqn_sweep.build_parser().parse_args(
        ["--trials", "1", "--run-dir", str(tmp_path), "--env-kind", "keydoor", "--env-id", HALLWAY_KEYDOOR_ENV_ID, *extra]
    )


def _write_summary(tmp_path: Path, config: dict) -> None:
    trial_dir = tmp_path / "delay_0" / "trial_0"
    trial_dir.mkdir(parents=True)
    (trial_dir / "summary.json").write_text(json.dumps({"config": config}))


def test_resume_reads_old_summaries_as_plain_dqn(tmp_path):
    args = _args(tmp_path)
    config = parallel_dqn_sweep._expected_config(args, HALLWAY_KEYDOOR_ENV_ID, 0, 0)
    config.pop("double_dqn")
    _write_summary(tmp_path, config)
    assert parallel_dqn_sweep.plan_tasks(args, HALLWAY_KEYDOOR_ENV_ID, tmp_path, [0], [0]) == ([], 1)
    with pytest.raises(SystemExit, match="double_dqn"):
        parallel_dqn_sweep.plan_tasks(_args(tmp_path, "--double-dqn"), HALLWAY_KEYDOOR_ENV_ID, tmp_path, [0], [0])


def test_train_trial_uses_and_records_double_dqn():
    trial_dir = Path(tempfile.mkdtemp())
    train_trial(
        trial=0, env_id=HALLWAY_KEYDOOR_ENV_ID, seed=0, total_timesteps=2_000, delay=0, max_steps=None,
        trial_dir=trial_dir, env_kind="keydoor", step_penalty=0.001, eval_every_episodes=None, double_dqn=True,
    )
    config = json.loads((trial_dir / "summary.json").read_text())["config"]
    assert config["double_dqn"] is True
    assert isinstance(DoubleDQN.load(trial_dir / "model"), DoubleDQN)
