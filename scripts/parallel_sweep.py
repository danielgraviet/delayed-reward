#!/usr/bin/env python3
"""Run independent delayed-reward trials concurrently in one run directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delayed_reward.__main__ import train_one_trial
from delayed_reward.results.plotting import (
    DEFAULT_ONSET_THRESHOLD,
    plot_delay_survival,
)
from delayed_reward.results.saver import create_delay_dir, create_run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--delay", type=int, nargs="+", default=[0])
    parser.add_argument("--env-id", default="MiniGrid-Empty-10x10-v0")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--onset-threshold", type=float, default=DEFAULT_ONSET_THRESHOLD)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing run directory to resume; completed, matching trials are skipped",
    )
    return parser


def _expected_config(args: argparse.Namespace, delay: int, trial: int) -> dict[str, object]:
    return {
        "episodes": args.episodes,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "env_seed": args.seed,
        "trial": trial,
        "log_every": args.log_every,
        "delay": delay,
        "max_steps": args.max_steps,
        "env": args.env_id,
    }


def _run_trial(task: tuple[argparse.Namespace, Path, int, int]) -> Path:
    args, run_dir, delay, trial = task
    return train_one_trial(
        delay=delay,
        trial=trial,
        env_id=args.env_id,
        seed=args.seed,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        log_every=args.log_every,
        delay_dir=create_delay_dir(run_dir, delay),
        max_steps=args.max_steps,
    )


def main() -> None:
    args = build_parser().parse_args()
    delays = list(dict.fromkeys(args.delay))
    if args.episodes < 1 or args.trials < 1 or args.workers < 1 or args.log_every < 1:
        raise SystemExit("episodes, trials, workers, and log-every must be >= 1")
    if any(delay < 0 for delay in delays):
        raise SystemExit("delay values must be >= 0")
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit("max-steps must be >= 1")
    if not 0.0 <= args.onset_threshold <= 1.0:
        raise SystemExit("onset-threshold must be in [0, 1]")

    if args.run_dir is None:
        run_dir = create_run_dir(args.results_dir)
    else:
        run_dir = args.run_dir
        if not run_dir.is_dir():
            raise SystemExit(f"run directory does not exist: {run_dir}")

    tasks: list[tuple[argparse.Namespace, Path, int, int]] = []
    skipped = 0
    for delay in delays:
        delay_dir = create_delay_dir(run_dir, delay)
        for trial in range(args.trials):
            summary_path = delay_dir / f"trial_{trial}" / "summary.json"
            if summary_path.exists():
                import json

                saved_config = json.loads(summary_path.read_text()).get("config", {})
                expected = _expected_config(args, delay, trial)
                mismatches = {
                    key: (saved_config.get(key), value)
                    for key, value in expected.items()
                    if saved_config.get(key) != value
                }
                if mismatches:
                    raise SystemExit(
                        f"refusing to mix settings in {summary_path}: {mismatches}"
                    )
                skipped += 1
            else:
                trial_dir = delay_dir / f"trial_{trial}"
                if trial_dir.exists():
                    # Keep artifacts from interrupted/failed trials, then let
                    # train_one_trial create a clean directory at the usual path.
                    backup_root = run_dir / "_incomplete"
                    backup_root.mkdir(exist_ok=True)
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                    trial_dir.rename(
                        backup_root / f"delay_{delay}_trial_{trial}_{stamp}"
                    )
                tasks.append((args, run_dir, delay, trial))

    print(f"RUN_DIR {run_dir.resolve()}", flush=True)
    print(
        f"DELAYS {delays} TRIALS {args.trials} WORKERS {args.workers} "
        f"PENDING {len(tasks)} SKIPPED {skipped}",
        flush=True,
    )
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_trial, task): task[2:] for task in tasks}
        for future in as_completed(futures):
            delay, trial = futures[future]
            try:
                trial_dir = future.result()
            except Exception as exc:
                failures.append((delay, trial, exc))
                print(f"FAILED delay={delay} trial={trial}: {exc!r}", flush=True)
            else:
                print(f"DONE delay={delay} trial={trial} {trial_dir}", flush=True)

    has_metrics = any(
        metrics_path.exists()
        for delay in delays
        for metrics_path in (run_dir / f"delay_{delay}").glob("trial_*/metrics.csv")
    )
    if has_metrics:
        plot_path = plot_delay_survival(
            run_dir,
            smooth_window=args.log_every,
            onset_threshold=args.onset_threshold,
        )
        print(f"PLOT {plot_path.resolve()}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed; resume with --run-dir")
    print(f"All delays finished. Results under {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
