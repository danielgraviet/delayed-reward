"""Learning-rate schedule option and its resume checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from delayed_reward.dqn_experiment import lr_config, make_lr_schedule
from delayed_reward.environments.factory import HALLWAY_KEYDOOR_ENV_ID

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import parallel_dqn_sweep  # noqa: E402

BASE = ["--env-kind", "keydoor", "--env-id", HALLWAY_KEYDOOR_ENV_ID]


def _args(tmp_path: Path, *extra: str):
    return parallel_dqn_sweep.build_parser().parse_args(["--trials", "1", "--run-dir", str(tmp_path), *BASE, *extra])


def _write_summary(tmp_path: Path, config: dict) -> None:
    trial_dir = tmp_path / "delay_0" / "trial_0"
    trial_dir.mkdir(parents=True)
    (trial_dir / "summary.json").write_text(json.dumps({"config": config}))


def _plan(args, tmp_path):
    return parallel_dqn_sweep.plan_tasks(args, HALLWAY_KEYDOOR_ENV_ID, tmp_path, [0], [0])


def test_constant_schedule_is_a_plain_float():
    assert make_lr_schedule(1e-3, "constant", 0.0) == 1e-3


def test_linear_schedule_decays_from_initial_to_final():
    schedule = make_lr_schedule(1e-3, "linear", 1e-5)
    # SB3 passes progress_remaining: 1.0 at the start, 0.0 at the end.
    assert schedule(1.0) == pytest.approx(1e-3)
    assert schedule(0.5) == pytest.approx((1e-3 + 1e-5) / 2)
    assert schedule(0.0) == pytest.approx(1e-5)


@pytest.mark.parametrize("lr, schedule, final", [(0.0, "constant", 0.0), (1e-3, "linear", 2e-3), (1e-3, "cosine", 0.0)])
def test_invalid_schedules_are_rejected(lr, schedule, final):
    with pytest.raises(ValueError):
        make_lr_schedule(lr, schedule, final)


def test_lr_final_is_only_recorded_for_linear():
    assert lr_config(1e-3, "constant", 1e-5)["lr_final"] is None
    assert lr_config(1e-3, "linear", 1e-5)["lr_final"] == 1e-5


def test_sweep_rejects_bad_lr_final(tmp_path):
    with pytest.raises(SystemExit, match="lr_final"):
        parallel_dqn_sweep.resolve_sweep(_args(tmp_path, "--lr-schedule", "linear", "--lr-final", "1"))


def test_resume_accepts_summaries_written_before_the_lr_and_eval_keys(tmp_path):
    args = _args(tmp_path)
    config = parallel_dqn_sweep._expected_config(args, HALLWAY_KEYDOOR_ENV_ID, 0, 0)
    for key in ("lr_schedule", "lr_final", "eval_every_episodes"):
        config.pop(key)
    _write_summary(tmp_path, config)
    assert _plan(args, tmp_path) == ([], 1)


@pytest.mark.parametrize(
    "changed, key",
    [
        (["--lr-schedule", "linear"], "lr_schedule"),
        (["--learning-rate", "5e-4"], "learning_rate"),
        (["--eval-every-episodes", "5"], "eval_every_episodes"),
    ],
)
def test_resume_refuses_changed_training_settings(tmp_path, changed, key):
    _write_summary(tmp_path, parallel_dqn_sweep._expected_config(_args(tmp_path), HALLWAY_KEYDOOR_ENV_ID, 0, 0))
    with pytest.raises(SystemExit, match=key):
        _plan(_args(tmp_path, *changed), tmp_path)
