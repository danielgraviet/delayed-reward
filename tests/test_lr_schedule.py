"""Learning-rate schedule option and its resume checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from delayed_reward.config import DQNParams, LRSchedule
from delayed_reward.agents.dqn.model import make_lr_schedule
from delayed_reward.cli import dqn_sweep
from delayed_reward.cli.dqn import dqn_config_from_args
from delayed_reward.environments.registry import HALLWAY_KEYDOOR_ENV_ID
from delayed_reward.sweep.dqn import plan_tasks

BASE = ["--env-kind", "keydoor", "--env-id", HALLWAY_KEYDOOR_ENV_ID]


def _args(tmp_path: Path, *extra: str):
    return dqn_sweep.build_parser().parse_args(["--trials", "1", "--run-dir", str(tmp_path), *BASE, *extra])


def _write_summary(tmp_path: Path, config: dict) -> None:
    trial_dir = tmp_path / "delay_0" / "trial_0"
    trial_dir.mkdir(parents=True)
    (trial_dir / "summary.json").write_text(json.dumps({"config": config}))


def _plan(args, tmp_path):
    return plan_tasks(dqn_config_from_args(args), tmp_path, [0], [0])


def test_constant_schedule_is_a_plain_float():
    assert make_lr_schedule(DQNParams(learning_rate=1e-3)) == 1e-3


def test_linear_schedule_decays_from_initial_to_final():
    schedule = make_lr_schedule(DQNParams(learning_rate=1e-3, lr_schedule=LRSchedule.LINEAR, lr_final=1e-5))
    # SB3 passes progress_remaining: 1.0 at the start, 0.0 at the end.
    assert schedule(1.0) == pytest.approx(1e-3)
    assert schedule(0.5) == pytest.approx((1e-3 + 1e-5) / 2)
    assert schedule(0.0) == pytest.approx(1e-5)


@pytest.mark.parametrize("lr, schedule, final", [(0.0, "constant", 0.0), (1e-3, "linear", 2e-3), (1e-3, "cosine", 0.0)])
def test_invalid_schedules_are_rejected(lr, schedule, final):
    with pytest.raises(ValueError):
        DQNParams(learning_rate=lr, lr_schedule=schedule, lr_final=final)


def test_lr_final_is_only_recorded_for_linear(tmp_path):
    constant = dqn_config_from_args(_args(tmp_path, "--lr-final", "1e-5"))
    linear = dqn_config_from_args(_args(tmp_path, "--lr-schedule", "linear", "--lr-final", "1e-5"))
    assert constant.to_dict()["lr_final"] is None
    assert linear.to_dict()["lr_final"] == 1e-5


def test_sweep_rejects_bad_lr_final(tmp_path):
    with pytest.raises(SystemExit, match="lr_final"):
        dqn_sweep.resolve_sweep(_args(tmp_path, "--lr-schedule", "linear", "--lr-final", "1"))


def test_resume_skips_matching_summary(tmp_path):
    _write_summary(tmp_path, dqn_config_from_args(_args(tmp_path)).to_dict())
    assert _plan(_args(tmp_path), tmp_path) == ([], 1)


def test_resume_refuses_summary_missing_a_setting(tmp_path):
    config = dqn_config_from_args(_args(tmp_path)).to_dict()
    config.pop("lr_schedule")
    _write_summary(tmp_path, config)
    with pytest.raises(SystemExit, match="lr_schedule"):
        _plan(_args(tmp_path), tmp_path)


@pytest.mark.parametrize(
    "changed, key",
    [
        (["--lr-schedule", "linear"], "lr_schedule"),
        (["--learning-rate", "5e-4"], "learning_rate"),
        (["--eval-every-episodes", "5"], "eval_every_episodes"),
        (["--double-dqn"], "double_dqn"),
    ],
)
def test_resume_refuses_changed_training_settings(tmp_path, changed, key):
    _write_summary(tmp_path, dqn_config_from_args(_args(tmp_path)).to_dict())
    with pytest.raises(SystemExit, match=key):
        _plan(_args(tmp_path, *changed), tmp_path)
