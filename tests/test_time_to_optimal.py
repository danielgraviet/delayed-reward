"""Time-to-optimal criterion, early stop, schedule horizon, and their resume checks."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from delayed_reward.dqn_experiment import GreedyEvalCallback, make_lr_schedule, train_trial
from delayed_reward.environments.factory import HALLWAY_KEYDOOR_ENV_ID

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import parallel_dqn_sweep  # noqa: E402

BASE = ["--env-kind", "keydoor", "--env-id", HALLWAY_KEYDOOR_ENV_ID]


def _callback(**kwargs) -> GreedyEvalCallback:
    return GreedyEvalCallback(eval_env=None, eval_every_episodes=1, max_steps=50, **kwargs)


def _feed(callback: GreedyEvalCallback, flags: list[bool]) -> None:
    for index, flag in enumerate(flags, start=1):
        callback._train_episode_count = index
        callback.num_timesteps = index * 100
        callback._record_optimal(flag)


def test_streak_resets_and_first_and_sustained_are_separate():
    callback = _callback(optimal_steps=25, sustain_evals=3)
    _feed(callback, [False, True, True, False, True, True, True, True])
    assert callback.first_optimal == {"train_episode": 2, "timestep": 200}
    assert callback.sustained_optimal == {"train_episode": 7, "timestep": 700}
    summary = callback.time_to_optimal(total_timesteps=5_000)
    assert summary["censored"] is False
    assert summary["censor_timestep"] is None


def test_never_sustained_is_right_censored_at_the_cap():
    callback = _callback(optimal_steps=25, sustain_evals=3)
    _feed(callback, [True, True, False, True])
    summary = callback.time_to_optimal(total_timesteps=5_000)
    assert summary["first_optimal_timestep"] == 100
    assert summary["sustained_optimal_timestep"] is None
    assert summary["censored"] is True
    assert summary["censor_timestep"] == 5_000


def test_no_criterion_means_no_summary():
    assert _callback().time_to_optimal(total_timesteps=5_000) is None
    with pytest.raises(ValueError):
        _callback(stop_on_sustained=True)


def test_linear_lr_reaches_final_at_end_fraction_then_holds():
    schedule = make_lr_schedule(1e-3, "linear", 1e-4, end_fraction=0.25)
    assert schedule(1.0) == pytest.approx(1e-3)
    assert schedule(0.875) == pytest.approx((1e-3 + 1e-4) / 2)
    assert schedule(0.75) == pytest.approx(1e-4)
    assert schedule(0.0) == pytest.approx(1e-4)


def _short_trial(tmp_path: Path, monkeypatch, run_eval, **kwargs) -> Path:
    monkeypatch.setattr(GreedyEvalCallback, "_run_eval", run_eval)
    trial_dir = tmp_path / "trial_0"
    trial_dir.mkdir()
    train_trial(
        trial=0, env_id=HALLWAY_KEYDOOR_ENV_ID, seed=0, delay=0, max_steps=50,
        trial_dir=trial_dir, env_kind="keydoor", step_penalty=0.001, eval_every_episodes=1,
        **kwargs,
    )
    return trial_dir


def test_trial_stops_at_sustained_optimal_and_records_it(tmp_path, monkeypatch):
    trial_dir = _short_trial(
        tmp_path, monkeypatch, lambda self: (1.0, 25.0, True),
        total_timesteps=50_000, optimal_steps=25, sustain_evals=3, stop_on_sustained=True,
    )
    summary = json.loads((trial_dir / "summary.json").read_text())
    record = summary["time_to_optimal"]
    # Each training episode is truncated at 50 steps; the 3rd eval closes the streak.
    assert record["sustained_optimal_timestep"] == 150
    assert record["trained_timesteps"] == 150
    assert record["stopped_early"] is True and record["censored"] is False
    assert summary["config"]["optimal_steps"] == 25
    with (trial_dir / "eval_metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [row["optimal"] for row in rows] == ["1", "1", "1"]


def test_schedule_horizon_fixes_epsilon_decay_independent_of_cap(tmp_path, monkeypatch):
    seen: list[tuple[int, float]] = []

    def run_eval(self):
        seen.append((self.num_timesteps, self.model.exploration_rate))
        return 0.0, 50.0, False

    trial_dir = _short_trial(
        tmp_path, monkeypatch, run_eval, total_timesteps=2_000, schedule_timesteps=1_000,
    )
    # Epsilon anneals over 0.4 * schedule_timesteps = 400 steps, not 0.4 * 2000
    # (SB3 updates it just after the callback, so step 400 itself still reads 0.0524).
    assert all(eps == pytest.approx(0.05) for step, eps in seen if step > 400)
    assert seen[0][1] > 0.8
    config = json.loads((trial_dir / "summary.json").read_text())["config"]
    assert config["schedule_timesteps"] == 1_000
    assert config["exploration_fraction"] == 0.4


def _args(tmp_path: Path, *extra: str):
    return parallel_dqn_sweep.build_parser().parse_args(
        ["--trials", "1", "--run-dir", str(tmp_path), *BASE, *extra]
    )


def _write_summary(tmp_path: Path, config: dict) -> None:
    trial_dir = tmp_path / "delay_0" / "trial_0"
    trial_dir.mkdir(parents=True)
    (trial_dir / "summary.json").write_text(json.dumps({"config": config}))


def _legacy_config() -> dict:
    return {
        "algorithm": "stable-baselines3-dqn", "env": HALLWAY_KEYDOOR_ENV_ID, "env_kind": "keydoor",
        "delay": 0, "subgoal_reward": 0.3, "reward_attribution": "arrival", "step_penalty": 0.0,
        "checkpoint_window": 100, "trial": 0, "seed": 0, "total_timesteps": 500_000,
        "max_steps": None, "layout_randomization": [], "fixed_layout": "canonical", "learning_rate": 1e-3,
        "eval_episodes": 1,
    }


def test_legacy_summary_resumes_without_criterion(tmp_path):
    _write_summary(tmp_path, _legacy_config())
    args = _args(tmp_path)
    pending, skipped = parallel_dqn_sweep.plan_tasks(args, HALLWAY_KEYDOOR_ENV_ID, tmp_path, [0], [0])
    assert (pending, skipped) == ([], 1)


def test_legacy_summary_refuses_new_criterion(tmp_path):
    _write_summary(tmp_path, _legacy_config())
    args = _args(tmp_path, "--optimal-steps", "25", "--stop-on-sustained")
    with pytest.raises(SystemExit, match="refusing to mix"):
        parallel_dqn_sweep.plan_tasks(args, HALLWAY_KEYDOOR_ENV_ID, tmp_path, [0], [0])


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--stop-on-sustained"], "requires --optimal-steps"),
        (["--schedule-timesteps", "600000"], "schedule-timesteps"),
        (["--optimal-steps", "25", "--eval-every-episodes", "0"], "eval-every-episodes"),
    ],
)
def test_sweep_rejects_bad_criterion_args(tmp_path, extra, message):
    with pytest.raises(SystemExit, match=message):
        parallel_dqn_sweep.resolve_sweep(_args(tmp_path, *extra))


def test_run_summary_counts_censored_trials_without_solve_times(tmp_path):
    records = [
        {"first_optimal_timestep": 100, "sustained_optimal_timestep": 300, "censored": False, "censor_timestep": None},
        {"first_optimal_timestep": 200, "sustained_optimal_timestep": None, "censored": True, "censor_timestep": 900},
        {"first_optimal_timestep": None, "sustained_optimal_timestep": None, "censored": True, "censor_timestep": 900},
    ]
    for trial, record in enumerate(records):
        trial_dir = tmp_path / "delay_0" / f"trial_{trial}"
        trial_dir.mkdir(parents=True)
        (trial_dir / "summary.json").write_text(json.dumps({"config": {}, "time_to_optimal": record}))
    out = parallel_dqn_sweep.summarize_time_to_optimal(tmp_path)
    entry = json.loads(out.read_text())["0"]
    assert (entry["n_sustained"], entry["n_censored"], entry["n_first_optimal"]) == (1, 2, 2)
    assert entry["sustained_timesteps_solved"] == [300]
    # 1 of 3 solved: the median trial is censored, so no median is claimed.
    assert entry["median_sustained_timestep"] is None
