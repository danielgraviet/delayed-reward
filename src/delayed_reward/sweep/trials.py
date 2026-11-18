"""Shared sweep mechanics: trial-directory bookkeeping, local process pool, final plot."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from delayed_reward.config import TrialConfig, config_mismatches
from delayed_reward.results.plotting import plot_delay_survival
from delayed_reward.results.saver import create_delay_dir


def stash_incomplete(run_dir: Path, delay: int, trial: int) -> None:
    """Move an unfinished ``delay_<d>/trial_<t>`` aside to ``run_dir/_incomplete`` so it can rerun cleanly."""
    trial_dir = run_dir / f"delay_{delay}" / f"trial_{trial}"
    if not trial_dir.exists():
        return
    backup_root = run_dir / "_incomplete"
    backup_root.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    trial_dir.rename(backup_root / f"delay_{delay}_trial_{trial}_{stamp}")


def plan_tasks[C: TrialConfig](
    template: C, run_dir: Path, delays: Sequence[int], trial_ids: Sequence[int]
) -> tuple[list[C], int]:
    """Return the pending trial configs and the count of completed ones skipped.

    Completed trials must match the requested settings exactly. Leftover
    directories of incomplete trials are moved under ``run_dir/_incomplete``.
    """
    pending: list[C] = []
    skipped = 0
    for delay in delays:
        delay_dir = create_delay_dir(run_dir, delay)
        for trial in trial_ids:
            config = template.for_trial(delay=delay, trial=trial)
            summary_path = delay_dir / f"trial_{trial}" / "summary.json"
            if summary_path.exists():
                saved = json.loads(summary_path.read_text()).get("config", {})
                mismatches = config_mismatches(saved, config)
                if mismatches:
                    raise SystemExit(f"refusing to mix settings in {summary_path}: {mismatches}")
                skipped += 1
                continue
            stash_incomplete(run_dir, delay, trial)
            pending.append(config)
    return pending, skipped


def run_in_processes[T](
    run: Callable[[T], Path], tasks: Sequence[tuple[int, int, T]], *, workers: int
) -> list[tuple[int, int, BaseException]]:
    """Run ``run(task)`` for each ``(delay, trial, task)`` in a process pool; return the failures."""
    failures: list[tuple[int, int, BaseException]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, task): (delay, trial) for delay, trial, task in tasks}
        for future in as_completed(futures):
            delay, trial = futures[future]
            try:
                trial_dir = future.result()
            except Exception as exc:
                failures.append((delay, trial, exc))
                print(f"FAILED delay={delay} trial={trial}: {exc!r}", flush=True)
            else:
                print(f"DONE delay={delay} trial={trial} {trial_dir}", flush=True)
    return failures


def plot_survival_if_any(run_dir: Path, *, smooth_window: int, onset_threshold: float) -> None:
    """Write the run's delay-survival plot once at least one trial has metrics."""
    if not any(run_dir.glob("delay_*/trial_*/metrics.csv")):
        return
    plot_path = plot_delay_survival(run_dir, smooth_window=smooth_window, onset_threshold=onset_threshold)
    print(f"PLOT {plot_path.resolve()}", flush=True)
