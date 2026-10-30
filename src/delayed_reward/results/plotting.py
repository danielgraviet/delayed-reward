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


def _iter_trial_csvs(
    run_dir: Path, *, filename: str = "metrics.csv", required: bool = True
) -> dict[int, list[Path]]:
    """Map each ``delay_*`` subdirectory to its ``<filename>`` paths.

    Supports ``delay_*/trial_*/<filename>`` (multi-trial) and
    ``delay_*/<filename>`` (legacy single-trial). With ``required=False``,
    missing files are silently skipped instead of raising -- useful for
    optional per-trial outputs like ``eval_metrics.csv`` that older runs
    won't have.
    """
    by_delay: dict[int, list[Path]] = {}
    for child in sorted(run_dir.iterdir()):
        match = _DELAY_DIR_RE.match(child.name)
        if not match:
            continue
        delay = int(match.group(1))
        trial_dirs = sorted(
            (p for p in child.iterdir() if p.is_dir() and _TRIAL_DIR_RE.match(p.name)),
            key=lambda p: int(_TRIAL_DIR_RE.match(p.name).group(1)),  # type: ignore[union-attr]
        )
        if trial_dirs:
            paths = [
                p / filename for p in trial_dirs if (p / filename).exists()
            ]
        else:
            csv_path = child / filename
            paths = [csv_path] if csv_path.exists() else []
        if paths:
            by_delay[delay] = paths
    if not by_delay and required:
        raise FileNotFoundError(
            f"No delay_*/{filename} or delay_*/trial_*/{filename} under {run_dir}"
        )
    return by_delay


def _read_returns(csv_path: Path) -> np.ndarray:
    returns: list[float] = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            returns.append(float(row["return"]))
    return np.asarray(returns, dtype=np.float64)


def _read_table(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``(returns, steps)`` columns for one trial's ``metrics.csv``."""
    returns: list[float] = []
    steps: list[float] = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            returns.append(float(row["return"]))
            steps.append(float(row["steps"]))
    return np.asarray(returns, dtype=np.float64), np.asarray(steps, dtype=np.float64)


def load_delay_trial_metrics(run_dir: Path) -> dict[int, list[np.ndarray]]:
    """Load returns for each delay, with one array per trial."""
    return {
        delay: [_read_returns(p) for p in paths]
        for delay, paths in _iter_trial_csvs(run_dir).items()
    }


def load_delay_trial_tables(
    run_dir: Path,
) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """Load ``(returns, steps)`` per trial, grouped by delay."""
    return {
        delay: [_read_table(p) for p in paths]
        for delay, paths in _iter_trial_csvs(run_dir).items()
    }


def _read_eval_table(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``(train_episode, return)`` columns from one trial's ``eval_metrics.csv``."""
    episodes: list[float] = []
    returns: list[float] = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            episodes.append(float(row["train_episode"]))
            returns.append(float(row["return"]))
    return np.asarray(episodes, dtype=np.float64), np.asarray(returns, dtype=np.float64)


def load_delay_trial_eval_tables(
    run_dir: Path,
) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """Load exploration-off eval ``(train_episode, return)`` per trial, by delay.

    Returns an empty dict if no trial has ``eval_metrics.csv`` (e.g. a run
    from before evaluation rollouts were added).
    """
    return {
        delay: [_read_eval_table(p) for p in paths]
        for delay, paths in _iter_trial_csvs(
            run_dir, filename="eval_metrics.csv", required=False
        ).items()
    }


def load_delay_metrics(run_dir: Path) -> dict[int, np.ndarray]:
    """Load a single returns series per delay (first trial / legacy layout)."""
    return {
        delay: series[0] for delay, series in load_delay_trial_metrics(run_dir).items()
    }


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean padded at the start with partial windows.

    Note the padding means ``out[i]`` for ``i < window - 1`` is an average
    of fewer than ``window`` samples (just ``out[0] == values[0]``). That is
    fine for a display curve, but callers deciding pass/fail against a
    threshold should skip indices before ``window - 1`` — see
    ``learning_onset_episode``.
    """
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
    """First episode, once the rolling window is fully populated, where the
    rolling mean return reaches ``threshold``.

    Episodes before the window fills are excluded. With a partial window,
    the "rolling mean" at episode 0 is just that episode's raw return, so a
    single lucky early episode could clear a low threshold and get reported
    as onset=0 regardless of delay or condition -- which is exactly what
    happened once action masking made early episodes easy to win by chance.
    """
    if returns.size == 0:
        return None
    smoothed = rolling_mean(returns, window)
    start = window - 1
    if start >= smoothed.size:
        return None
    hits = np.flatnonzero(smoothed[start:] >= threshold)
    if hits.size == 0:
        return None
    return int(hits[0] + start)


def sustained_onset_episode(
    returns: np.ndarray,
    *,
    window: int,
    threshold: float,
) -> int | None:
    """First episode after which the rolling mean never again drops below
    ``threshold`` for the remainder of the recorded run.

    ``learning_onset_episode`` only reports the first crossing, so a trial
    that clears the threshold early and then regresses later in training is
    still reported as having "passed". This instead finds the last episode
    (once the window is full) where the rolling mean was below threshold,
    and reports the episode right after it -- ``None`` if it never recovers,
    or if it never clears the threshold with a full window at all.
    """
    if returns.size == 0:
        return None
    smoothed = rolling_mean(returns, window)
    start = window - 1
    if start >= smoothed.size:
        return None
    below = np.flatnonzero(smoothed[start:] < threshold) + start
    if below.size == 0:
        hits = np.flatnonzero(smoothed[start:] >= threshold)
        return int(hits[0] + start) if hits.size else None
    last_below = int(below[-1])
    if last_below == smoothed.size - 1:
        return None
    return last_below + 1


def _stack_ragged(series: list[np.ndarray]) -> np.ndarray:
    """Stack 1-D arrays of different lengths into a 2-D array, NaN-padded."""
    max_len = max(s.size for s in series)
    out = np.full((len(series), max_len), np.nan)
    for i, s in enumerate(series):
        out[i, : s.size] = s
    return out


def _aggregate_trials(
    series: list[np.ndarray], window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth each trial, then return (mean, std, n_contributing) per episode.

    Shorter trials drop out of the mean/std once they end (NaN-padded), so
    ``n_contributing`` tells you how many trials actually back each point --
    a band built from 1 trial near the tail looks identical to one built
    from 10 unless you check this.
    """
    smoothed = [rolling_mean(s, window) for s in series]
    stacked = _stack_ragged(smoothed)
    counts = np.sum(~np.isnan(stacked), axis=0)
    mean = np.nanmean(stacked, axis=0)
    with np.errstate(invalid="ignore"):
        std = np.nanstd(stacked, axis=0)
    std = np.where(counts >= 2, std, np.nan)
    return mean, std, counts


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
    sustained_lists: dict[int, list[int | None]] = {}
    onset_means: dict[int, float | None] = {}
    onset_stds: dict[int, float | None] = {}
    sustained_means: dict[int, float | None] = {}
    sustained_stds: dict[int, float | None] = {}
    n_trials_by_delay: dict[int, int] = {}
    pass_counts: dict[int, int] = {}
    sustained_pass_counts: dict[int, int] = {}
    max_episodes = 0

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color, (delay, series) in zip(colors, sorted(metrics.items())):
        n_trials_by_delay[delay] = len(series)
        max_episodes = max(max_episodes, *(s.size for s in series))

        onsets = [
            learning_onset_episode(s, window=smooth_window, threshold=onset_threshold)
            for s in series
        ]
        sustained = [
            sustained_onset_episode(s, window=smooth_window, threshold=onset_threshold)
            for s in series
        ]
        onset_lists[delay] = onsets
        sustained_lists[delay] = sustained

        achieved = [o for o in onsets if o is not None]
        onset_means[delay] = float(np.mean(achieved)) if achieved else None
        onset_stds[delay] = (
            float(np.std(achieved)) if len(achieved) > 1 else (0.0 if achieved else None)
        )
        susts = [o for o in sustained if o is not None]
        sustained_means[delay] = float(np.mean(susts)) if susts else None
        sustained_stds[delay] = (
            float(np.std(susts)) if len(susts) > 1 else (0.0 if susts else None)
        )

        n_passed = sum(o is not None for o in onsets)
        n_sustained_passed = sum(o is not None for o in sustained)
        pass_counts[delay] = n_passed
        sustained_pass_counts[delay] = n_sustained_passed

        label = (
            f"Delay {delay}: {n_passed}/{len(series)} crossed, "
            f"{n_sustained_passed}/{len(series)} sustained"
        )
        episodes = np.arange(max_episodes + 1)

        passed_at = sorted(o for o in onsets if o is not None)
        if passed_at:
            passed_fraction = (
                np.searchsorted(passed_at, episodes, side="right") / len(series)
            )
            ax.step(
                episodes,
                passed_fraction * 100,
                where="post",
                label=label,
                linewidth=2.2,
                color=color,
            )
        else:
            ax.plot([], [], label=label, linewidth=2.2, color=color)

        sustained_at = sorted(o for o in sustained if o is not None)
        if sustained_at:
            sustained_fraction = (
                np.searchsorted(sustained_at, episodes, side="right") / len(series)
            )
            ax.step(
                episodes,
                sustained_fraction * 100,
                where="post",
                linewidth=1.4,
                linestyle="--",
                color=color,
            )

    ax.set_title(f"How much reward delay can the learner solve?\n{run_dir.name}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Trials that have passed (%)")
    ax.set_xlim(0, max_episodes)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20), labels=[f"{v}%" for v in range(0, 101, 20)])
    ax.grid(True, alpha=0.3)
    ax.legend(
        title=(
            f"Solid = rolling mean ≥ {onset_threshold} once {smooth_window} "
            f"episodes are in the window\nDashed = sustained (never drops back "
            f"below {onset_threshold} again)"
        ),
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
        "sustained_onset_episodes_by_delay": {
            str(d): sustained_lists[d] for d in sorted(metrics)
        },
        "n_sustained_passed_by_delay": {
            str(d): sustained_pass_counts[d] for d in sorted(metrics)
        },
        "sustained_pass_fraction_by_delay": {
            str(d): sustained_pass_counts[d] / n_trials_by_delay[d]
            for d in sorted(metrics)
        },
        "sustained_onset_mean_by_delay": {
            str(d): sustained_means[d] for d in sorted(metrics)
        },
        "sustained_onset_std_by_delay": {
            str(d): sustained_stds[d] for d in sorted(metrics)
        },
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


def _plot_aggregate(
    ax: plt.Axes,
    mean: np.ndarray,
    std: np.ndarray,
    counts: np.ndarray,
    n_trials: int,
    *,
    color: str,
    label: str,
    x: np.ndarray | None = None,
) -> None:
    """Plot a mean curve, solid+shaded while every trial is still running,
    dash-dot and unshaded once shorter trials have ended and the mean is
    only backed by a subset. Trial episode counts vary a lot in this
    project (e.g. one DQN trial ran 3200+ episodes while siblings stopped
    under 500), so a flat tail at the far right is often just one trial,
    not a converged average across all of them -- this makes that visible
    instead of silently averaging it away.

    ``x`` defaults to the episode index (0..len-1); pass explicit x-values
    when the samples aren't one-per-episode (e.g. sparse eval checkpoints).
    """
    episodes = x if x is not None else np.arange(mean.size)
    full = counts >= n_trials
    full_mean = np.where(full, mean, np.nan)
    ax.plot(episodes, full_mean, color=color, linewidth=2.0, label=label)
    full_lower = np.where(full & ~np.isnan(std), mean - std, np.nan)
    full_upper = np.where(full & ~np.isnan(std), mean + std, np.nan)
    ax.fill_between(episodes, full_lower, full_upper, color=color, alpha=0.2, linewidth=0)

    if np.any(~full):
        partial_mean = np.where(~full, mean, np.nan)
        ax.plot(episodes, partial_mean, color=color, linewidth=1.2, linestyle="-.", alpha=0.6)


def plot_learning_curves(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    smooth_window: int = 100,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
) -> Path:
    """Plot smoothed return vs. episode, mean +/- std across trials, per delay.

    This is the curve the pass/fail summary in ``plot_delay_survival`` can't
    show you: whether performance is actually rising, flat, or degrading
    over training, not just whether it crossed a threshold once.
    """
    metrics = load_delay_trial_metrics(run_dir)
    output = output_path or (run_dir / "learning_curves.png")
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color, (delay, series) in zip(colors, sorted(metrics.items())):
        mean, std, counts = _aggregate_trials(series, smooth_window)
        _plot_aggregate(ax, mean, std, counts, len(series), color=color, label=f"Delay {delay} (n={len(series)})")

    ax.axhline(
        onset_threshold, color="gray", linestyle=":", linewidth=1.2,
        label=f"onset threshold ({onset_threshold})",
    )
    ax.set_title(f"Smoothed return over training (window={smooth_window})\n{run_dir.name}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Rolling mean return (window={smooth_window})")
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="Shaded band = ±1 std across trials\nDash-dot tail = fewer trials still running (shorter ones ended)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_episode_length(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    smooth_window: int = 100,
) -> Path:
    """Plot smoothed steps-per-episode vs. episode, mean +/- std, per delay.

    Falling episode length alongside rising return means the agent is also
    finding more direct paths, not just eventually stumbling into reward.
    """
    tables = load_delay_trial_tables(run_dir)
    output = output_path or (run_dir / "episode_length.png")
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color, (delay, trials) in zip(colors, sorted(tables.items())):
        steps_series = [steps for _returns, steps in trials]
        mean, std, counts = _aggregate_trials(steps_series, smooth_window)
        _plot_aggregate(ax, mean, std, counts, len(trials), color=color, label=f"Delay {delay} (n={len(trials)})")

    ax.set_title(f"Smoothed episode length over training (window={smooth_window})\n{run_dir.name}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Rolling mean steps/episode (window={smooth_window})")
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="Shaded band = ±1 std across trials\nDash-dot tail = fewer trials still running (shorter ones ended)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_eval_curve(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
) -> Path | None:
    """Plot exploration-off evaluation return vs. training episode, per delay.

    The training-return curves (``plot_learning_curves``) are taken while
    the agent is still exploring (random actions by design, forever --
    see ``exploration_final_eps``), so they understate and add noise to
    what the policy has actually learned. These points instead come from
    periodic rollouts with randomness turned off entirely, so they show
    the policy's real performance at each checkpoint.

    Returns ``None`` and writes nothing if the run has no
    ``eval_metrics.csv`` files (e.g. a run from before evaluation rollouts
    were added, or trained with ``--eval-every-episodes 0``).
    """
    eval_tables = load_delay_trial_eval_tables(run_dir)
    if not eval_tables:
        return None
    output = output_path or (run_dir / "eval_curve.png")
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color, (delay, trials) in zip(colors, sorted(eval_tables.items())):
        returns_series = [r for _episodes, r in trials]
        stacked = _stack_ragged(returns_series)
        counts = np.sum(~np.isnan(stacked), axis=0)
        mean = np.nanmean(stacked, axis=0)
        with np.errstate(invalid="ignore"):
            std = np.nanstd(stacked, axis=0)
        std = np.where(counts >= 2, std, np.nan)
        longest_episodes = max(trials, key=lambda t: t[0].size)[0]
        x = longest_episodes[: mean.size]
        _plot_aggregate(
            ax, mean, std, counts, len(trials), color=color,
            label=f"Delay {delay} (n={len(trials)})", x=x,
        )

    ax.axhline(onset_threshold, color="gray", linestyle=":", linewidth=1.2, label=f"onset threshold ({onset_threshold})")
    ax.set_title(f"Exploration-off evaluation return over training\n{run_dir.name}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Evaluation return (greedy policy, no exploration)")
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="Shaded band = ±1 std across trials\nDash-dot tail = fewer trials still running",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_condition_comparison(
    conditions: dict[str, Path],
    *,
    output_path: Path,
    smooth_window: int = 100,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
) -> Path:
    """Compare conditions trained as separate runs (e.g. baseline vs. two
    reward-attribution schemes under the same delay), each its own run dir.

    ``plot_delay_survival``/``plot_learning_curves`` compare delays *within*
    one run directory; this instead overlays whole run directories -- each
    one's trials (across however many ``delay_*`` subdirs it has, normally
    just one) are pooled under that condition's label.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_return, ax_steps, ax_final) = plt.subplots(
        3, 1, figsize=(12, 16), constrained_layout=True
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    summary: dict[str, dict] = {}
    final_labels: list[str] = []
    final_means: list[float] = []
    final_stds: list[float] = []

    for color, (name, run_dir) in zip(colors, conditions.items()):
        tables = load_delay_trial_tables(run_dir)
        trials = [pair for pairs in tables.values() for pair in pairs]
        returns_series = [r for r, _s in trials]
        steps_series = [s for _r, s in trials]

        mean_r, std_r, counts_r = _aggregate_trials(returns_series, smooth_window)
        _plot_aggregate(ax_return, mean_r, std_r, counts_r, len(trials), color=color, label=f"{name} (n={len(trials)})")

        mean_s, std_s, counts_s = _aggregate_trials(steps_series, smooth_window)
        _plot_aggregate(ax_steps, mean_s, std_s, counts_s, len(trials), color=color, label=f"{name} (n={len(trials)})")

        final_window_means = [
            float(r[-smooth_window:].mean()) if r.size >= smooth_window else float(r.mean())
            for r in returns_series
        ]
        final_labels.append(name)
        final_means.append(float(np.mean(final_window_means)))
        final_stds.append(float(np.std(final_window_means)) if len(final_window_means) > 1 else 0.0)

        onsets = [learning_onset_episode(r, window=smooth_window, threshold=onset_threshold) for r in returns_series]
        sustained = [sustained_onset_episode(r, window=smooth_window, threshold=onset_threshold) for r in returns_series]
        summary[name] = {
            "run_dir": str(run_dir),
            "n_trials": len(trials),
            "n_episodes_per_trial": [int(r.size) for r in returns_series],
            "final_window_mean_return_per_trial": final_window_means,
            "final_window_mean_return": final_means[-1],
            "final_window_std_return": final_stds[-1],
            "n_crossed_threshold": sum(o is not None for o in onsets),
            "n_sustained_above_threshold": sum(o is not None for o in sustained),
            "onset_episodes": onsets,
            "sustained_onset_episodes": sustained,
        }

    ax_return.axhline(onset_threshold, color="gray", linestyle=":", linewidth=1.2)
    ax_return.set_title("Smoothed return over training (dash-dot = fewer trials still running)")
    ax_return.set_xlabel("Training episode")
    ax_return.set_ylabel(f"Rolling mean return (window={smooth_window})")
    ax_return.grid(True, alpha=0.3)
    ax_return.legend(loc="lower right")

    ax_steps.set_title("Smoothed episode length over training")
    ax_steps.set_xlabel("Training episode")
    ax_steps.set_ylabel(f"Rolling mean steps/episode (window={smooth_window})")
    ax_steps.grid(True, alpha=0.3)
    ax_steps.legend(loc="upper right")

    x = np.arange(len(final_labels))
    bar_colors = colors[: len(final_labels)]
    ax_final.bar(x, final_means, yerr=final_stds, capsize=6, color=bar_colors, alpha=0.85)
    for xi, means_for_cond in zip(x, (summary[name]["final_window_mean_return_per_trial"] for name in final_labels)):
        jitter = (np.random.default_rng(0).random(len(means_for_cond)) - 0.5) * 0.15
        ax_final.scatter(np.full(len(means_for_cond), xi) + jitter, means_for_cond, color="black", zorder=3, s=20)
    ax_final.axhline(onset_threshold, color="gray", linestyle=":", linewidth=1.2)
    ax_final.set_xticks(x, labels=final_labels)
    ax_final.set_title(f"Final performance (mean return over last {smooth_window} episodes)")
    ax_final.set_ylabel("Mean return")
    ax_final.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Condition comparison", fontsize=14)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return output_path
