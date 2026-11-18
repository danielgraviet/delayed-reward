"""Write run/trial directories and per-trial metrics files."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


def create_run_dir(base: Path | str = "results") -> Path:
    """Create a timestamped results subdirectory and return its path."""
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def create_delay_dir(run_dir: Path, delay: int) -> Path:
    """Create ``run_dir/delay_<n>`` for one delay setting's artifacts."""
    delay_dir = run_dir / f"delay_{delay}"
    delay_dir.mkdir(parents=True, exist_ok=True)
    return delay_dir


def create_trial_dir(delay_dir: Path, trial: int) -> Path:
    """Create ``delay_dir/trial_<k>`` for one seeded replicate."""
    trial_dir = delay_dir / f"trial_{trial}"
    trial_dir.mkdir(parents=True, exist_ok=False)
    return trial_dir


def save_metrics(
    run_dir: Path,
    *,
    rewards: Sequence[float],
    steps: Sequence[int],
    config: Mapping[str, object],
    window: int = 100,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Write per-episode CSV metrics and a JSON summary into ``run_dir``.

    ``extra`` adds top-level keys to ``summary.json`` (e.g. ``time_to_optimal``).
    """
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["episode", "return", "steps"])
        for episode, (reward, n_steps) in enumerate(zip(rewards, steps, strict=True)):
            writer.writerow([episode, reward, n_steps])

    last = list(rewards[-window:]) if rewards else []
    summary = {
        "config": dict(config),
        "n_episodes": len(rewards),
        "final_window": window,
        "final_mean_return": (sum(last) / len(last)) if last else None,
        "final_mean_steps": (
            sum(steps[-window:]) / min(window, len(steps)) if steps else None
        ),
        **(extra or {}),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def save_eval_metrics(
    run_dir: Path,
    *,
    train_episodes: Sequence[int],
    train_timesteps: Sequence[int],
    returns: Sequence[float],
    steps: Sequence[float],
    optimal: Sequence[bool] | None = None,
) -> None:
    """Write periodic exploration-off evaluation results to ``eval_metrics.csv``.

    With ``optimal`` (one flag per checkpoint), an ``optimal`` column is added.

    Unlike ``metrics.csv`` (one row per training episode, taken while the
    agent is still exploring), this is one row per evaluation checkpoint:
    how the greedy policy performed with randomness turned off.
    """
    metrics_path = run_dir / "eval_metrics.csv"
    with metrics_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["train_episode", "train_timestep", "return", "steps"]
        columns = [train_episodes, train_timesteps, returns, steps]
        if optimal is not None:
            header.append("optimal")
            columns.append([int(flag) for flag in optimal])
        writer.writerow(header)
        for row in zip(*columns, strict=True):
            writer.writerow(row)
