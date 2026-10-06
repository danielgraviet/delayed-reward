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


def _pad_stack(series: list[np.ndarray]) -> np.ndarray:
    """Stack 1D arrays to shape (n_trials, max_len), padding shorter with nan."""
    max_len = max(s.size for s in series)
    stacked = np.full((len(series), max_len), np.nan, dtype=np.float64)
    for i, s in enumerate(series):
        stacked[i, : s.size] = s
    return stacked


def plot_delay_survival(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    smooth_window: int = 100,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
) -> Path:
    """Plot mean±std smoothed returns and onset stats across trials per delay."""
    metrics = load_delay_trial_metrics(run_dir)
    output = output_path or (run_dir / "delay_survival.png")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    curve_ax, bar_ax = axes

    onset_means: dict[int, float | None] = {}
    onset_stds: dict[int, float | None] = {}
    onset_lists: dict[int, list[int | None]] = {}
    n_trials_by_delay: dict[int, int] = {}

    for delay, series in sorted(metrics.items()):
        n_trials_by_delay[delay] = len(series)
        smoothed_trials = [rolling_mean(s, smooth_window) for s in series]
        stacked = _pad_stack(smoothed_trials)
        mean_curve = np.nanmean(stacked, axis=0)
        std_curve = np.nanstd(stacked, axis=0)
        episodes = np.arange(mean_curve.size)

        onsets = [
            learning_onset_episode(s, window=smooth_window, threshold=onset_threshold)
            for s in series
        ]
        onset_lists[delay] = onsets
        achieved = [o for o in onsets if o is not None]
        if achieved:
            onset_means[delay] = float(np.mean(achieved))
            onset_stds[delay] = float(np.std(achieved)) if len(achieved) > 1 else 0.0
        else:
            onset_means[delay] = None
            onset_stds[delay] = None

        n = len(series)
        if onset_means[delay] is not None:
            label = (
                f"delay={delay} (n={n}, onset "
                f"{onset_means[delay]:.0f}±{onset_stds[delay]:.0f})"
            )
        else:
            label = f"delay={delay} (n={n}, no onset)"

        (line,) = curve_ax.plot(episodes, mean_curve, label=label, linewidth=1.8)
        curve_ax.fill_between(
            episodes,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=line.get_color(),
            alpha=0.2,
        )
        if onset_means[delay] is not None:
            onset_x = onset_means[delay]
            onset_y = float(
                mean_curve[min(int(round(onset_x)), mean_curve.size - 1)]
            )
            curve_ax.axvline(
                onset_x,
                color=line.get_color(),
                linestyle="--",
                alpha=0.55,
                linewidth=1.2,
            )
            curve_ax.scatter([onset_x], [onset_y], color=line.get_color(), s=36, zorder=5)

    curve_ax.axhline(
        onset_threshold,
        color="#333333",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
        label=f"onset threshold={onset_threshold}",
    )
    curve_ax.set_title(f"Delay survival — {run_dir.name}")
    curve_ax.set_xlabel("Episode")
    curve_ax.set_ylabel(f"Return (rolling mean, w={smooth_window})")
    curve_ax.legend(loc="lower right", fontsize=9)
    curve_ax.grid(True, alpha=0.3)

    delays = sorted(onset_means)
    bar_values = [onset_means[d] if onset_means[d] is not None else 0.0 for d in delays]
    bar_errs = [onset_stds[d] if onset_stds[d] is not None else 0.0 for d in delays]
    colors = ["#4C78A8" if onset_means[d] is not None else "#B0B0B0" for d in delays]
    bars = bar_ax.bar(
        [str(d) for d in delays],
        bar_values,
        yerr=bar_errs,
        color=colors,
        capsize=4,
        error_kw={"ecolor": "#333333", "capthick": 1.0},
    )
    for delay, bar in zip(delays, bars, strict=True):
        mean = onset_means[delay]
        std = onset_stds[delay]
        n_ok = sum(o is not None for o in onset_lists[delay])
        n = n_trials_by_delay[delay]
        if mean is None:
            text = f"none\n0/{n}"
        else:
            text = f"{mean:.0f}±{std:.0f}\n{n_ok}/{n}"
        bar_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (std or 0.0),
            text,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    bar_ax.set_xlabel("Delay")
    bar_ax.set_ylabel("Learning onset episode")
    bar_ax.set_title(
        f"Mean±std episodes until rolling mean return ≥ {onset_threshold}"
    )
    bar_ax.grid(True, axis="y", alpha=0.3)

    summary = {
        "onset_threshold": onset_threshold,
        "smooth_window": smooth_window,
        "n_trials_by_delay": {str(d): n_trials_by_delay[d] for d in delays},
        "onset_episodes_by_delay": {
            str(d): onset_lists[d] for d in delays
        },
        "onset_mean_by_delay": {str(d): onset_means[d] for d in delays},
        "onset_std_by_delay": {str(d): onset_stds[d] for d in delays},
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
