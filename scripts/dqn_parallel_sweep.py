#!/usr/bin/env python3
"""Run independent DQN trials concurrently and preserve resumable artifacts."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delayed_reward.dqn_experiment import train_trial
from delayed_reward.environments.factory import DEFAULT_ENV_ID
from delayed_reward.results.plotting import plot_delay_survival
from delayed_reward.results.saver import create_delay_dir, create_run_dir


def _run_trial(task: tuple[argparse.Namespace, Path, int, int]) -> Path:
    args, run_dir, delay, trial = task
    trial_dir = create_delay_dir(run_dir, delay) / f"trial_{trial}"
    trial_dir.mkdir(parents=True, exist_ok=False)
    train_trial(
        trial=trial,
        env_id=args.env_id,
        seed=args.seed,
        total_timesteps=args.timesteps,
        delay=delay,
        max_steps=args.max_steps,
        trial_dir=trial_dir,
    )
    return trial_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--delay", type=int, nargs="+", default=[0])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()
    if min(args.trials, args.workers, args.timesteps) < 1:
        raise SystemExit("trials, workers, and timesteps must be >= 1")
    delays = list(dict.fromkeys(args.delay))
    if any(delay < 0 for delay in delays) or (args.max_steps is not None and args.max_steps < 1):
        raise SystemExit("delay must be >= 0 and max-steps must be >= 1")

    run_dir = args.run_dir or create_run_dir(args.results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[argparse.Namespace, Path, int, int]] = []
    skipped = 0
    for delay in delays:
        delay_dir = create_delay_dir(run_dir, delay)
        for trial in range(args.trials):
            trial_dir = delay_dir / f"trial_{trial}"
            summary = trial_dir / "summary.json"
            if summary.exists():
                config = json.loads(summary.read_text()).get("config", {})
                expected = {
                    "env": args.env_id,
                    "delay": delay,
                    "trial": trial,
                    "seed": args.seed,
                    "total_timesteps": args.timesteps,
                    "max_steps": args.max_steps,
                }
                mismatches = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
                if mismatches:
                    raise SystemExit(f"refusing to mix settings in {summary}: {mismatches}")
                skipped += 1
                continue
            if trial_dir.exists():
                backup_root = run_dir / "_incomplete"
                backup_root.mkdir(exist_ok=True)
                backup = backup_root / f"delay_{delay}_trial_{trial}"
                suffix = 1
                while backup.exists():
                    backup = backup_root / f"delay_{delay}_trial_{trial}_{suffix}"
                    suffix += 1
                trial_dir.rename(backup)
            pending.append((args, run_dir, delay, trial))

    print(
        f"RUN_DIR {run_dir.resolve()} delays={delays} trials={args.trials} "
        f"pending={len(pending)} skipped={skipped} workers={args.workers} "
        f"timesteps={args.timesteps}",
        flush=True,
    )
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_trial, task): task[2:] for task in pending}
        for future in as_completed(futures):
            delay, trial = futures[future]
            try:
                path = future.result()
            except Exception as exc:
                failures.append((trial, exc))
                print(f"FAILED delay={delay} trial={trial}: {exc!r}", flush=True)
            else:
                print(f"DONE delay={delay} trial={trial} {path}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed; rerun with --run-dir")
    plot_path = plot_delay_survival(run_dir, smooth_window=100, onset_threshold=0.5)
    print(f"PLOT {plot_path.resolve()}", flush=True)
    print(f"All DQN trials finished under {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
