"""Compare learning onset across delay settings for a training run."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_DELAY_DIR_RE = re.compile(r"^delay_(\d+)$")
_TRIAL_DIR_RE = re.compile(r"^trial_(\d+)$")
DEFAULT_ONSET_THRESHOLD = 0.5


def latest_run_dir(results_root: Path | str = "results") -> Path:
    """Return the most recently modified ``run_*`` directory."""
    root = Path(results_root)
    runs = sorted(root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run_* directories under {root.resolve()}")
    return runs[0]


def _read_returns(csv_path: Path) -> np.ndarray:
    returns: list[float] = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            returns.append(float(row["return"]))
    return np.asarray(returns, dtype=np.float64)


def load_delay_trial_metrics(run_dir: Path) -> dict[int, list[np.ndarray]]:
    """Load returns for each delay, with one array per trial.

    Supports:
    - ``delay_*/trial_*/metrics.csv`` (multi-trial)
    - ``delay_*/metrics.csv`` (legacy single-trial)
    """
    metrics: dict[int, list[np.ndarray]] = {}
    for child in sorted(run_dir.iterdir()):
        match = _DELAY_DIR_RE.match(child.name)
        if not match:
            continue
        delay = int(match.group(1))
        trial_dirs = sorted(
            (
                p
                for p in child.iterdir()
                if p.is_dir() and _TRIAL_DIR_RE.match(p.name)
            ),
            key=lambda p: int(_TRIAL_DIR_RE.match(p.name).group(1)),  # type: ignore[union-attr]
        )
        series: list[np.ndarray] = []
        if trial_dirs:
            for trial_dir in trial_dirs:
                csv_path = trial_dir / "metrics.csv"
                if csv_path.exists():
                    series.append(_read_returns(csv_path))
        else:
            csv_path = child / "metrics.csv"
            if csv_path.exists():
                series.append(_read_returns(csv_path))
        if series:
            metrics[delay] = series
    if not metrics:
        raise FileNotFoundError(
            f"No delay_*/metrics.csv or delay_*/trial_*/metrics.csv under {run_dir}"
        )
    return metrics


def load_delay_metrics(run_dir: Path) -> dict[int, np.ndarray]:
    """Load a single returns series per delay (first trial / legacy layout)."""
    return {
        delay: series[0] for delay, series in load_delay_trial_metrics(run_dir).items()
    }


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean padded at the start with partial windows."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if values.size == 0:
        return values.copy()
    cumulative = np.cumsum(values, dtype=np.float64)
    out = np.empty_like(cumulative)
    for i in range(values.size):
        start = max(0, i + 1 - window)
        total = cumulative[i] - (cumulative[start - 1] if start > 0 else 0.0)
        out[i] = total / (i - start + 1)
    return out


def learning_onset_episode(
    returns: np.ndarray,
    *,
    window: int,
    threshold: float,
) -> int | None:
    """First episode where the rolling mean return reaches ``threshold``."""
    if returns.size == 0:
        return None
    smoothed = rolling_mean(returns, window)
    hits = np.flatnonzero(smoothed >= threshold)
    if hits.size == 0:
        return None
    return int(hits[0])


def plot_delay_survival(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    smooth_window: int = 100,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
) -> Path:
    """Plot the fraction of trials that learn to solve at each reward delay."""
    metrics = load_delay_trial_metrics(run_dir)
    output = output_path or (run_dir / "delay_survival.png")
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    onset_lists: dict[int, list[int | None]] = {}
    onset_means: dict[int, float | None] = {}
    onset_stds: dict[int, float | None] = {}
    n_trials_by_delay: dict[int, int] = {}
    pass_counts: dict[int, int] = {}
    max_episodes = 0

    for delay, series in sorted(metrics.items()):
        n_trials_by_delay[delay] = len(series)
        max_episodes = max(max_episodes, *(s.size for s in series))

        onsets = [
            learning_onset_episode(s, window=smooth_window, threshold=onset_threshold)
            for s in series
        ]
        onset_lists[delay] = onsets
        achieved = [o for o in onsets if o is not None]
        onset_means[delay] = float(np.mean(achieved)) if achieved else None
        onset_stds[delay] = (
            float(np.std(achieved)) if len(achieved) > 1 else (0.0 if achieved else None)
        )
        n_passed = sum(o is not None for o in onsets)
        pass_counts[delay] = n_passed
        label = f"Delay {delay}: {n_passed}/{len(series)} passed"
        passed_at = sorted(o for o in onsets if o is not None)
        if passed_at:
            episodes = np.arange(max_episodes + 1)
            passed_fraction = np.searchsorted(passed_at, episodes, side="right") / len(series)
            ax.step(episodes, passed_fraction * 100, where="post", label=label, linewidth=2.2)
        else:
            ax.plot([], [], label=label, linewidth=2.2)

    ax.set_title(f"How much reward delay can the learner solve?\n{run_dir.name}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Trials that have passed (%)")
    ax.set_xlim(0, max_episodes)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20), labels=[f"{v}%" for v in range(0, 101, 20)])
    ax.grid(True, alpha=0.3)
    ax.legend(
        title=f"Passed = rolling mean return ≥ {onset_threshold} for {smooth_window} episodes",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )

    summary = {
        "onset_threshold": onset_threshold,
        "smooth_window": smooth_window,
        "n_trials_by_delay": {str(d): n_trials_by_delay[d] for d in sorted(metrics)},
        "onset_episodes_by_delay": {
            str(d): onset_lists[d] for d in sorted(metrics)
        },
        "n_passed_by_delay": {str(d): pass_counts[d] for d in sorted(metrics)},
        "pass_fraction_by_delay": {
            str(d): pass_counts[d] / n_trials_by_delay[d] for d in sorted(metrics)
        },
        "onset_mean_by_delay": {str(d): onset_means[d] for d in sorted(metrics)},
        "onset_std_by_delay": {str(d): onset_stds[d] for d in sorted(metrics)},
    }
    (run_dir / "delay_survival.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_delay_first_rewards(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    smooth_window: int = 100,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
) -> Path:
    return plot_delay_survival(
        run_dir,
        output_path=output_path,
        smooth_window=smooth_window,
        onset_threshold=onset_threshold,
    )
