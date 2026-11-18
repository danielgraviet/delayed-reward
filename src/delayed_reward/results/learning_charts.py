"""The three per-run learning charts, drawn from greedy-eval checkpoints (see ``results.analysis``).

- ``time_to_solve.png``: when each trial first solved the task, one row per delay.
- ``success_rate.png``: share of trials solving the task at each point in training.
- ``trial_timelines.png``: every checkpoint of every trial, solved or not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from delayed_reward.results.analysis import (
    SUSTAIN_EVALS,
    DelaySummary,
    TrialEvals,
    load_trial_evals,
    success_rate,
    summarize,
    write_analysis,
)

SOLVED = "#2a78d6"
UNSOLVED = "#f0efec"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SUCCESS_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RATE_BINS = 25


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )


def _timestep_axis(ax: Axes, cap: int) -> None:
    ax.set_xlim(0, cap)
    ax.set_xlabel("Training timesteps")
    ax.xaxis.set_major_formatter(FuncFormatter(short_count))


def short_count(value: float, _position: object = None) -> str:
    """Compact timestep label: 290353 -> "290k", 1500000 -> "1.5M"."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _title(ax: Axes, run_dir: Path, headline: str, explainer: str) -> None:
    ax.set_title(f"{headline}  ({run_dir.name})\n{explainer}", loc="left", color=INK, fontsize=11)


def _save(fig: Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_time_to_solve(run_dir: Path, summaries: Sequence[DelaySummary], cap: int) -> Path:
    """One row per delay, one dot per trial at its time to solve; hollow marks at the cap are unsolved trials."""
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(summaries) + 1.5))
    for row, summary in enumerate(summaries):
        solved = [t for t in summary.solve_times if t is not None]
        unsolved = summary.n_trials - len(solved)
        ax.scatter(
            solved,
            row + np.linspace(-0.18, 0.18, max(len(solved), 1))[: len(solved)],
            s=36,
            color=SOLVED,
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
        )
        if unsolved:
            ax.scatter(
                [cap] * unsolved,
                row + np.linspace(-0.3, 0.3, unsolved),
                s=30,
                facecolor="none",
                edgecolor=MUTED,
                linewidth=1,
                zorder=3,
                clip_on=False,
            )
        median = summary.median_time_to_solve
        if median is not None:
            ax.plot([median, median], [row - 0.34, row + 0.34], color=INK, linewidth=2, zorder=4)
        median_text = f"median {short_count(median)}" if median is not None else "no median"
        ax.text(
            cap * 1.03,
            row,
            f"{summary.n_solved}/{summary.n_trials} solved · {median_text}",
            va="center",
            color=MUTED,
            fontsize=9,
        )
    ax.set_yticks(range(len(summaries)), [f"delay {s.delay}" for s in summaries])
    ax.set_ylim(len(summaries) - 0.5, -0.5)
    _timestep_axis(ax, cap)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    _title(
        ax,
        run_dir,
        "Time to solve, by reward delay",
        f"Dot = a trial's first {SUSTAIN_EVALS} solved greedy evals in a row · bar = median · "
        f"hollow = not solved by {short_count(cap)}",
    )
    return _save(fig, run_dir / "time_to_solve.png")


def plot_success_rate(run_dir: Path, trials_by_delay: Mapping[int, Sequence[TrialEvals]], cap: int) -> Path:
    """Heatmap: rows are delays, columns are training-time bins, color is the share of trials solving the task."""
    delays = sorted(trials_by_delay)
    edges = np.linspace(0, cap, RATE_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    rates = np.array([success_rate(trials_by_delay[delay], centers) for delay in delays])
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(delays) + 1.7))
    image = ax.imshow(
        rates,
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list("success", SUCCESS_RAMP),
        vmin=0,
        vmax=1,
        extent=(0, cap, len(delays) - 0.5, -0.5),
        interpolation="nearest",
    )
    ax.set_yticks(range(len(delays)), [f"delay {delay}" for delay in delays])
    _timestep_axis(ax, cap)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("Trials whose greedy policy reaches the goal", color=MUTED)
    bar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _title(
        ax,
        run_dir,
        "Greedy success rate over training",
        "Each cell: share of a delay's trials whose latest greedy eval reached the goal",
    )
    return _save(fig, run_dir / "success_rate.png")


def plot_trial_timelines(run_dir: Path, trials_by_delay: Mapping[int, Sequence[TrialEvals]], cap: int) -> Path:
    """One row per trial, grouped by delay: blue where the greedy policy was solving the task, gray where not."""
    rows = [trial for delay in sorted(trials_by_delay) for trial in trials_by_delay[delay]]
    fig, ax = plt.subplots(figsize=(9, 0.17 * len(rows) + 1.7))
    for y, trial in enumerate(rows):
        bounds = np.append(trial.timesteps, max(cap, int(trial.timesteps[-1])))
        widths = np.diff(bounds)
        colors = [SOLVED if solved else UNSOLVED for solved in trial.solved]
        ax.barh(np.full(len(widths), y), widths, left=bounds[:-1], height=0.8, color=colors, linewidth=0)
    group_starts = [i for i, trial in enumerate(rows) if i == 0 or trial.delay != rows[i - 1].delay]
    group_ends = [*group_starts[1:], len(rows)]
    ax.set_yticks(
        [(start + end - 1) / 2 for start, end in zip(group_starts, group_ends, strict=True)],
        [f"delay {rows[start].delay}" for start in group_starts],
    )
    for start in group_starts[1:]:
        ax.axhline(start - 0.5, color=MUTED, linewidth=0.8)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    _timestep_axis(ax, cap)
    for side in ax.spines.values():
        side.set_visible(False)
    _title(
        ax,
        run_dir,
        "Every trial's greedy evals",
        "One row per trial · blue = greedy policy reaches the goal · gray = it doesn't",
    )
    return _save(fig, run_dir / "trial_timelines.png")


def plot_run_overview(run_dir: Path) -> list[Path]:
    """Write ``analysis.json`` and the three learning charts; empty if no trial recorded greedy evals."""
    trials_by_delay = load_trial_evals(run_dir)
    if not trials_by_delay:
        return []
    _style()
    cap = max(trial.cap for trials in trials_by_delay.values() for trial in trials)
    summaries = summarize(trials_by_delay)
    return [
        write_analysis(run_dir, summaries),
        plot_time_to_solve(run_dir, summaries, cap),
        plot_success_rate(run_dir, trials_by_delay, cap),
        plot_trial_timelines(run_dir, trials_by_delay, cap),
    ]
