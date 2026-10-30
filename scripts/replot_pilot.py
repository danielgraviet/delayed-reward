#!/usr/bin/env python3
"""Regenerate plots/summaries for an existing run from its saved metrics.csv
files -- no retraining. Useful after changing the plotting/onset logic.

Two modes:
- a single run directory (``delay_*/trial_*/metrics.csv`` inside it):
  regenerates ``delay_survival.{json,png}``, ``learning_curves.png``, and
  ``episode_length.png`` in place.
- a pilot directory containing several condition subdirectories (each its
  own run directory, e.g. ``baseline/``, ``delay50_arrival/``,
  ``delay50_origin/``): does the above for each subdirectory, then also
  writes a cross-condition ``condition_comparison.{png,json}`` at the top.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delayed_reward.results.plotting import (
    DEFAULT_ONSET_THRESHOLD,
    plot_condition_comparison,
    plot_delay_survival,
    plot_episode_length,
    plot_eval_curve,
    plot_learning_curves,
)


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and any(p.name.startswith("delay_") for p in path.iterdir())


def _replot_run_dir(run_dir: Path, *, smooth_window: int, onset_threshold: float) -> None:
    print(f"  delay_survival  -> {plot_delay_survival(run_dir, smooth_window=smooth_window, onset_threshold=onset_threshold).resolve()}")
    print(f"  learning_curves -> {plot_learning_curves(run_dir, smooth_window=smooth_window, onset_threshold=onset_threshold).resolve()}")
    print(f"  episode_length  -> {plot_episode_length(run_dir, smooth_window=smooth_window).resolve()}")
    eval_curve_path = plot_eval_curve(run_dir, onset_threshold=onset_threshold)
    if eval_curve_path is not None:
        print(f"  eval_curve      -> {eval_curve_path.resolve()}")
    else:
        print("  eval_curve      -> skipped (no eval_metrics.csv in this run)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory or pilot directory")
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--onset-threshold", type=float, default=DEFAULT_ONSET_THRESHOLD)
    args = parser.parse_args()

    root = args.run_dir
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    if _is_run_dir(root):
        print(f"Run directory: {root.resolve()}")
        _replot_run_dir(root, smooth_window=args.smooth_window, onset_threshold=args.onset_threshold)
        return

    conditions = {
        p.name: p
        for p in sorted(root.iterdir())
        if _is_run_dir(p)
    }
    if not conditions:
        raise SystemExit(
            f"{root} is neither a run directory (no delay_* subdirs) "
            "nor a pilot directory containing run-directory subfolders"
        )

    print(f"Pilot directory: {root.resolve()}")
    print(f"Conditions: {list(conditions)}")
    for name, run_dir in conditions.items():
        print(f"[{name}]")
        _replot_run_dir(run_dir, smooth_window=args.smooth_window, onset_threshold=args.onset_threshold)

    if len(conditions) > 1:
        comparison_path = plot_condition_comparison(
            conditions,
            output_path=root / "condition_comparison.png",
            smooth_window=args.smooth_window,
            onset_threshold=args.onset_threshold,
        )
        print(f"condition_comparison -> {comparison_path.resolve()}")
        print(f"condition_comparison (json) -> {comparison_path.with_suffix('.json').resolve()}")


if __name__ == "__main__":
    main()
