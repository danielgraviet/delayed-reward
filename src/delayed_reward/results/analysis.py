"""Learning metrics from greedy evaluation checkpoints: did a trial solve the task, when, and did it stay solved.

A checkpoint is *solved* when every greedy (exploration-off) eval episode reached
the goal. A trial's *time to solve* is the training timestep of the checkpoint
that completes ``SUSTAIN_EVALS`` solved checkpoints in a row. Trials that never
get there are censored at their timestep cap: they count as slower than every
trial that solved, and are never given a made-up solve time.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SUSTAIN_EVALS = 3
_DELAY_DIR = re.compile(r"^delay_(\d+)$")
_TRIAL_DIR = re.compile(r"^trial_(\d+)$")
_TOLERANCE = 1e-6


@dataclass(frozen=True, kw_only=True)
class TrialEvals:
    """One trial's greedy-eval checkpoints, in training order."""

    delay: int
    trial: int
    timesteps: np.ndarray
    solved: np.ndarray
    cap: int

    def __post_init__(self) -> None:
        if self.timesteps.shape != self.solved.shape:
            raise ValueError("timesteps and solved must have the same length")
        if np.any(np.diff(self.timesteps) < 0):
            raise ValueError("checkpoint timesteps must be non-decreasing")


@dataclass(frozen=True, kw_only=True)
class DelaySummary:
    """How the trials at one delay learned."""

    delay: int
    n_trials: int
    n_solved: int
    median_time_to_solve: int | None
    solve_times: list[int | None]
    final_success_rate: float
    mean_stability: float | None


def time_to_solve(trial: TrialEvals, *, sustain: int = SUSTAIN_EVALS) -> int | None:
    """Timestep of the checkpoint completing ``sustain`` solved checkpoints in a row, or None if it never does."""
    streak = 0
    for timestep, solved in zip(trial.timesteps, trial.solved, strict=True):
        streak = streak + 1 if solved else 0
        if streak >= sustain:
            return int(timestep)
    return None


def solved_at(trial: TrialEvals, timesteps: np.ndarray) -> np.ndarray:
    """Whether the latest checkpoint at or before each timestep was solved; False before the first checkpoint."""
    latest = np.searchsorted(trial.timesteps, timesteps, side="right") - 1
    return np.where(latest >= 0, trial.solved[np.clip(latest, 0, None)], False)


def success_rate(trials: Sequence[TrialEvals], timesteps: np.ndarray) -> np.ndarray:
    """Share of ``trials`` whose greedy policy was solving the task at each timestep."""
    if not trials:
        raise ValueError("success_rate needs at least one trial")
    return np.mean([solved_at(trial, timesteps) for trial in trials], axis=0)


def stability(trial: TrialEvals) -> float | None:
    """Share of checkpoints, from the first solved one on, that are solved; None if it never solved."""
    hits = np.flatnonzero(trial.solved)
    if hits.size == 0:
        return None
    return float(np.mean(trial.solved[hits[0] :]))


def censored_median(times: Sequence[int | None]) -> int | None:
    """Median time with unsolved trials ranked last; None unless more than half of the trials solved."""
    if not times:
        return None
    rank = (len(times) - 1) // 2
    solved = sorted(t for t in times if t is not None)
    return solved[rank] if rank < len(solved) else None


def summarize(trials_by_delay: Mapping[int, Sequence[TrialEvals]]) -> list[DelaySummary]:
    """One ``DelaySummary`` per delay, in increasing delay order."""
    summaries = []
    for delay in sorted(trials_by_delay):
        trials = trials_by_delay[delay]
        times = [time_to_solve(trial) for trial in trials]
        stabilities = [s for s in (stability(trial) for trial in trials) if s is not None]
        summaries.append(
            DelaySummary(
                delay=delay,
                n_trials=len(trials),
                n_solved=sum(t is not None for t in times),
                median_time_to_solve=censored_median(times),
                solve_times=times,
                final_success_rate=float(np.mean([bool(trial.solved[-1]) for trial in trials])),
                mean_stability=float(np.mean(stabilities)) if stabilities else None,
            )
        )
    return summaries


def write_analysis(run_dir: Path, summaries: Sequence[DelaySummary]) -> Path:
    """Write the per-delay summaries to ``run_dir/analysis.json``."""
    path = run_dir / "analysis.json"
    payload = {"sustain_evals": SUSTAIN_EVALS, "delays": [asdict(summary) for summary in summaries]}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _checkpoint_solved(row: Mapping[str, str], config: Mapping[str, object]) -> bool:
    """Every eval episode reached the goal, from ``goal_rate`` or, without it, from the reward bookkeeping.

    Reaching the goal pays 1.0 plus the subgoal reward on the key-door tasks; any
    step penalty is subtracted per step. Adding the penalty back recovers what
    the goal paid, and this stays exact for means over several eval episodes.
    """
    if row.get("goal_rate") not in (None, ""):
        return float(row["goal_rate"]) >= 1 - _TOLERANCE
    penalty = _number(config, "step_penalty")
    subgoal = _number(config, "subgoal_reward")
    recovered = float(row["return"]) + penalty * float(row["steps"])
    return recovered >= 1.0 + subgoal - _TOLERANCE


def _number(config: Mapping[str, object], key: str) -> float:
    """A numeric config value; a missing or null one (e.g. ``subgoal_reward`` on the empty env) reads as 0."""
    value = config.get(key)
    if value is None:
        return 0.0
    if not isinstance(value, int | float):
        raise TypeError(f"config {key!r} must be a number, got {value!r}")
    return float(value)


def _trial_cap(trial_dir: Path, config: Mapping[str, object]) -> int:
    cap = config.get("total_timesteps")
    if isinstance(cap, int):
        return cap
    with (trial_dir / "metrics.csv").open() as fh:
        return sum(int(row["steps"]) for row in csv.DictReader(fh))


def _numbered(parent: Path, pattern: re.Pattern[str]) -> list[tuple[int, Path]]:
    found = [(int(m.group(1)), path) for path in parent.iterdir() if path.is_dir() and (m := pattern.match(path.name))]
    return sorted(found)


def load_trial_evals(run_dir: Path) -> dict[int, list[TrialEvals]]:
    """Every trial under ``run_dir`` that recorded greedy evals, grouped by delay."""
    by_delay: dict[int, list[TrialEvals]] = {}
    for delay, delay_dir in _numbered(run_dir, _DELAY_DIR):
        for trial, trial_dir in _numbered(delay_dir, _TRIAL_DIR):
            eval_path, summary_path = trial_dir / "eval_metrics.csv", trial_dir / "summary.json"
            if not (eval_path.exists() and summary_path.exists()):
                continue
            config = json.loads(summary_path.read_text())["config"]
            with eval_path.open() as fh:
                rows = list(csv.DictReader(fh))
            if not rows:
                continue
            by_delay.setdefault(delay, []).append(
                TrialEvals(
                    delay=delay,
                    trial=trial,
                    timesteps=np.array([int(row["train_timestep"]) for row in rows]),
                    solved=np.array([_checkpoint_solved(row, config) for row in rows]),
                    cap=_trial_cap(trial_dir, config),
                )
            )
    return by_delay
