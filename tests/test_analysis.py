"""Learning metrics: known answers, properties against a plain reference, and real runs."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, strategies as st

from delayed_reward.config import DQNTrialConfig, EnvKind, EnvSpec, EvalSpec
from delayed_reward.environments.registry import HALLWAY_KEYDOOR_ENV_ID
from delayed_reward.experiments.dqn import train_trial
from delayed_reward.results.analysis import (
    SUSTAIN_EVALS,
    TrialEvals,
    _checkpoint_solved,
    censored_median,
    load_trial_evals,
    solved_at,
    stability,
    success_rate,
    summarize,
    time_to_solve,
)
from delayed_reward.results.learning_charts import plot_run_overview, short_count

REPO_ROOT = Path(__file__).resolve().parents[1]
NINE_DELAY_RUN = REPO_ROOT / "results" / "run_20261102_164946"


def _trial(solved: list[bool], *, every: int = 100, cap: int = 10_000) -> TrialEvals:
    return TrialEvals(
        delay=0,
        trial=0,
        timesteps=np.arange(1, len(solved) + 1) * every,
        solved=np.array(solved, dtype=bool),
        cap=cap,
    )


@pytest.mark.parametrize(
    ("solved", "expected"),
    [
        pytest.param([True, True, True], 300, id="solved-from-the-start"),
        pytest.param([False, True, True, False, True, True, True], 700, id="streak-resets-on-a-miss"),
        pytest.param([True, False, True, False, True, True], None, id="never-three-in-a-row"),
        pytest.param([], None, id="no-checkpoints"),
    ],
)
def test_time_to_solve_known_answers(solved: list[bool], expected: int | None) -> None:
    assert time_to_solve(_trial(solved)) == expected


@pytest.mark.parametrize(
    ("times", "expected"),
    [
        pytest.param([300, 100, 200], 200, id="all-solved"),
        pytest.param([100, None, 200], 200, id="unsolved-ranks-last"),
        pytest.param([100, None, None], None, id="minority-solved-has-no-median"),
        pytest.param([None, None], None, id="none-solved"),
        pytest.param([], None, id="no-trials"),
    ],
)
def test_censored_median_known_answers(times: list[int | None], expected: int | None) -> None:
    assert censored_median(times) == expected


SOLVED_RUNS = st.lists(st.booleans(), min_size=1, max_size=40)


@given(SOLVED_RUNS)
def test_time_to_solve_marks_the_first_completed_streak(solved: list[bool]) -> None:
    trial = _trial(solved)
    found = time_to_solve(trial)
    first = next(
        (i for i in range(SUSTAIN_EVALS - 1, len(solved)) if all(solved[i - SUSTAIN_EVALS + 1 : i + 1])),
        None,
    )
    assert found == (None if first is None else int(trial.timesteps[first]))


@given(st.lists(st.none() | st.integers(0, 10**6), max_size=15))
def test_censored_median_matches_ranking_unsolved_as_infinite(times: list[int | None]) -> None:
    ranked = sorted(np.inf if t is None else t for t in times)
    reference = ranked[(len(times) - 1) // 2] if times else np.inf
    assert censored_median(times) == (None if reference == np.inf else reference)


@given(st.lists(SOLVED_RUNS, min_size=1, max_size=6))
def test_success_rate_is_a_share_bounded_by_the_checkpoints(runs: list[list[bool]]) -> None:
    trials = [_trial(run) for run in runs]
    end = max(int(trial.timesteps[-1]) for trial in trials)
    rate = success_rate(trials, np.array([0, 50, end, end + 10_000]))
    assert np.all((rate >= 0) & (rate <= 1))
    assert rate[0] == 0 and rate[1] == 0
    assert rate[2] == rate[3] == pytest.approx(np.mean([run[-1] for run in runs]))


@given(SOLVED_RUNS)
def test_stability_counts_only_from_the_first_solve(solved: list[bool]) -> None:
    value = stability(_trial(solved))
    if not any(solved):
        assert value is None
        return
    tail = solved[solved.index(True) :]
    assert value is not None
    assert value == pytest.approx(sum(tail) / len(tail))
    assert 0 < value <= 1


def test_solved_at_holds_the_latest_checkpoint_until_the_next() -> None:
    trial = _trial([False, True, False])
    assert solved_at(trial, np.array([99, 100, 200, 250, 300, 10_000])).tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]


def _without_goal_rate(row: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in row.items() if key != "goal_rate"}


@pytest.mark.parametrize(
    "env",
    [
        pytest.param(EnvSpec(kind=EnvKind.EMPTY, env_id="MiniGrid-Empty-5x5-v0", max_steps=30), id="empty-5x5"),
        pytest.param(
            EnvSpec(kind=EnvKind.KEYDOOR, env_id=HALLWAY_KEYDOOR_ENV_ID, max_steps=120, step_penalty=0.001),
            id="hallway-with-step-penalty",
        ),
    ],
)
def test_reward_rule_agrees_with_recorded_goal_rate(tmp_path: Path, env: EnvSpec) -> None:
    """Old runs lack ``goal_rate``; the reward-bookkeeping fallback must give the same answer on real evals."""
    config = DQNTrialConfig(env=env, total_timesteps=6_000, eval=EvalSpec(every_episodes=2, episodes=2))
    train_trial(config, tmp_path)
    settings = config.to_dict()
    with (tmp_path / "eval_metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    for row in rows:
        assert _checkpoint_solved(row, settings) == _checkpoint_solved(_without_goal_rate(row), settings)


def test_nine_delay_run_summary_is_pinned() -> None:
    """The committed 9-delay keydoor run: only delay 0 solves for most trials."""
    summaries = summarize(load_trial_evals(NINE_DELAY_RUN))
    assert [(s.delay, s.n_solved, s.n_trials) for s in summaries] == [
        (0, 7, 10),
        (1, 3, 10),
        (2, 3, 10),
        (5, 2, 10),
        (10, 3, 10),
        (25, 2, 10),
        (50, 4, 10),
        (100, 4, 10),
        (200, 3, 10),
    ]
    assert summaries[0].median_time_to_solve == 290_353
    assert all(s.median_time_to_solve is None for s in summaries[1:])


def test_overview_writes_analysis_and_three_charts(tmp_path: Path) -> None:
    for source in NINE_DELAY_RUN.glob("delay_*/trial_*"):
        target = tmp_path / source.parent.name / source.name
        target.mkdir(parents=True)
        for name in ("summary.json", "eval_metrics.csv", "metrics.csv"):
            shutil.copy(source / name, target / name)
    paths = plot_run_overview(tmp_path)
    assert [path.name for path in paths] == [
        "analysis.json",
        "time_to_solve.png",
        "success_rate.png",
        "trial_timelines.png",
    ]
    assert all(path.stat().st_size > 0 for path in paths)


@pytest.mark.parametrize(
    ("value", "label"),
    [(0, "0"), (950, "950"), (290_353, "290k"), (500_000, "500k"), (1_500_000, "1.5M"), (2_000_000, "2M")],
)
def test_timestep_labels_are_short(value: int, label: str) -> None:
    assert short_count(value) == label
