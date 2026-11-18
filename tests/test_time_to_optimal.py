"""Time-to-optimal criterion, early stop, schedule horizon, and their CLI checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from stable_baselines3.common.utils import LinearSchedule

from delayed_reward.agents.dqn.callbacks import GreedyEvalCallback
from delayed_reward.agents.dqn.model import make_lr_schedule
from delayed_reward.cli import dqn_sweep
from delayed_reward.config import DQNParams, DQNTrialConfig, EnvKind, EnvSpec, EvalSpec, LRSchedule
from delayed_reward.environments.registry import HALLWAY_KEYDOOR_ENV_ID
from delayed_reward.experiments.dqn import build_dqn_env, train_trial
from delayed_reward.sweep.dqn import summarize_time_to_optimal

BASE = ["--env-kind", "keydoor", "--env-id", HALLWAY_KEYDOOR_ENV_ID]


def _callback(**kwargs) -> GreedyEvalCallback:
    eval_env = build_dqn_env(EnvSpec(kind=EnvKind.KEYDOOR, env_id=HALLWAY_KEYDOOR_ENV_ID), seed=0)
    return GreedyEvalCallback(eval_env=eval_env, spec=EvalSpec(every_episodes=1, **kwargs), max_steps=50)


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
    assert summary is not None
    assert summary["censored"] is False
    assert summary["censor_timestep"] is None


def test_never_sustained_is_right_censored_at_the_cap():
    callback = _callback(optimal_steps=25, sustain_evals=3)
    _feed(callback, [True, True, False, True])
    summary = callback.time_to_optimal(total_timesteps=5_000)
    assert summary is not None
    assert summary["first_optimal_timestep"] == 100
    assert summary["sustained_optimal_timestep"] is None
    assert summary["censored"] is True
    assert summary["censor_timestep"] == 5_000


def test_no_criterion_means_no_summary():
    assert _callback().time_to_optimal(total_timesteps=5_000) is None
    with pytest.raises(ValueError):
        _callback(stop_on_sustained=True)


def test_linear_lr_reaches_final_at_end_fraction_then_holds():
    params = DQNParams(learning_rate=1e-3, lr_schedule=LRSchedule.LINEAR, lr_final=1e-4)
    schedule = make_lr_schedule(params, end_fraction=0.25)
    assert isinstance(schedule, LinearSchedule)
    assert schedule(1.0) == pytest.approx(1e-3)
    assert schedule(0.875) == pytest.approx((1e-3 + 1e-4) / 2)
    assert schedule(0.75) == pytest.approx(1e-4)
    assert schedule(0.0) == pytest.approx(1e-4)


def _short_trial(
    tmp_path: Path, monkeypatch, run_eval, *, total_timesteps: int, schedule_timesteps: int = 0, **eval_kwargs
) -> Path:
    monkeypatch.setattr(GreedyEvalCallback, "_run_eval", run_eval)
    trial_dir = tmp_path / "trial_0"
    trial_dir.mkdir()
    config = DQNTrialConfig(
        env=EnvSpec(kind=EnvKind.KEYDOOR, env_id=HALLWAY_KEYDOOR_ENV_ID, max_steps=50, step_penalty=0.001),
        total_timesteps=total_timesteps,
        schedule_timesteps=schedule_timesteps,
        eval=EvalSpec(every_episodes=1, **eval_kwargs),
    )
    train_trial(config, trial_dir)
    return trial_dir


def test_trial_stops_at_sustained_optimal_and_records_it(tmp_path, monkeypatch):
    """Each training episode is truncated at 50 steps; the 3rd eval closes the streak."""
    trial_dir = _short_trial(
        tmp_path,
        monkeypatch,
        lambda _self: (1.0, 25.0, True),
        total_timesteps=50_000,
        optimal_steps=25,
        sustain_evals=3,
        stop_on_sustained=True,
    )
    summary = json.loads((trial_dir / "summary.json").read_text())
    record = summary["time_to_optimal"]
    assert record["sustained_optimal_timestep"] == 150
    assert record["trained_timesteps"] == 150
    assert record["stopped_early"] is True and record["censored"] is False
    assert summary["config"]["optimal_steps"] == 25
    with (trial_dir / "eval_metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [row["optimal"] for row in rows] == ["1", "1", "1"]


def test_schedule_horizon_fixes_epsilon_decay_independent_of_cap(tmp_path, monkeypatch):
    """Epsilon anneals over 0.4 * schedule_timesteps = 400 steps, not 0.4 * the 2000-step cap.

    SB3 updates epsilon just after the callback, so step 400 itself still reads 0.0524.
    """
    seen: list[tuple[int, float]] = []

    def run_eval(self):
        seen.append((self.num_timesteps, self.model.exploration_rate))
        return 0.0, 50.0, False

    trial_dir = _short_trial(
        tmp_path,
        monkeypatch,
        run_eval,
        total_timesteps=2_000,
        schedule_timesteps=1_000,
    )
    assert all(eps == pytest.approx(0.05) for step, eps in seen if step > 400)
    assert seen[0][1] > 0.8
    config = json.loads((trial_dir / "summary.json").read_text())["config"]
    assert config["schedule_timesteps"] == 1_000
    assert config["exploration_fraction"] == 0.4


def _args(tmp_path: Path, *extra: str):
    return dqn_sweep.build_parser().parse_args(["--trials", "1", "--run-dir", str(tmp_path), *BASE, *extra])


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--stop-on-sustained"], "stop_on_sustained requires optimal_steps"),
        (["--schedule-timesteps", "600000"], "schedule_timesteps must be in"),
        (["--optimal-steps", "25", "--eval-every-episodes", "0"], "optimal_steps needs periodic evals"),
    ],
)
def test_sweep_rejects_bad_criterion_args(tmp_path, extra, message):
    with pytest.raises(SystemExit, match=message):
        dqn_sweep.resolve_sweep(_args(tmp_path, *extra))


def test_run_summary_counts_censored_trials_without_solve_times(tmp_path):
    """1 of 3 solved: the median trial is censored, so no median is claimed."""
    records = [
        {"first_optimal_timestep": 100, "sustained_optimal_timestep": 300, "censored": False, "censor_timestep": None},
        {"first_optimal_timestep": 200, "sustained_optimal_timestep": None, "censored": True, "censor_timestep": 900},
        {"first_optimal_timestep": None, "sustained_optimal_timestep": None, "censored": True, "censor_timestep": 900},
    ]
    for trial, record in enumerate(records):
        trial_dir = tmp_path / "delay_0" / f"trial_{trial}"
        trial_dir.mkdir(parents=True)
        (trial_dir / "summary.json").write_text(json.dumps({"config": {}, "time_to_optimal": record}))
    out = summarize_time_to_optimal(tmp_path)
    assert out is not None
    entry = json.loads(out.read_text())["0"]
    assert (entry["n_sustained"], entry["n_censored"], entry["n_first_optimal"]) == (1, 2, 2)
    assert entry["sustained_timesteps_solved"] == [300]
    assert entry["median_sustained_timestep"] is None
