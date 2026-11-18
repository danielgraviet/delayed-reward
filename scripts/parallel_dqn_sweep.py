#!/usr/bin/env python3
"""Run independent DQN trials concurrently in one run directory.

Mirrors ``scripts/parallel_sweep.py``'s resume/skip/plot pattern, but drives
``delayed_reward.dqn_experiment.train_trial`` (Stable-Baselines3 DQN on the
keydoor/empty MiniGrid envs) instead of the tabular Q-learning path.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delayed_reward.config import DQNTrialConfig, WandbSettings, config_mismatches
from delayed_reward.dqn_experiment import (
    add_dqn_args,
    dqn_config_from_args,
    train_trial,
    wandb_from_args,
)
from delayed_reward.results.plotting import DEFAULT_ONSET_THRESHOLD, plot_delay_survival
from delayed_reward.results.saver import create_delay_dir, create_run_dir, create_trial_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dqn_args(parser)
    parser.add_argument("--delay", type=int, nargs="+", default=[0])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--onset-threshold", type=float, default=DEFAULT_ONSET_THRESHOLD)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing run directory to resume; completed, matching trials are skipped",
    )
    parser.add_argument(
        "--trial-ids",
        type=int,
        nargs="+",
        default=None,
        help="Run only these trial indices (each < --trials); used by remote launchers",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the survival plot")
    return parser


def _run_trial(task: tuple[DQNTrialConfig, Path, WandbSettings | None]) -> Path:
    config, run_dir, wandb = task
    trial_dir = create_trial_dir(create_delay_dir(run_dir, config.env.delay), config.trial)
    train_trial(config, trial_dir, wandb=wandb)
    return trial_dir


def resolve_sweep(args: argparse.Namespace) -> tuple[DQNTrialConfig, list[int], list[int], Path]:
    """Validate args and return (config template, delays, trial_ids, run_dir)."""
    delays = list(dict.fromkeys(args.delay))
    if args.trials < 1 or args.workers < 1:
        raise SystemExit("trials and workers must be >= 1")
    if any(delay < 0 for delay in delays):
        raise SystemExit("delay values must be >= 0")
    if not 0.0 <= args.onset_threshold <= 1.0:
        raise SystemExit("onset-threshold must be in [0, 1]")
    template = dqn_config_from_args(args)
    trial_ids = list(dict.fromkeys(args.trial_ids)) if args.trial_ids else list(range(args.trials))
    if any(not 0 <= trial < args.trials for trial in trial_ids):
        raise SystemExit("trial-ids must be in [0, trials)")

    if args.run_dir is None:
        run_dir = create_run_dir(args.results_dir)
    else:
        run_dir = args.run_dir
        if not run_dir.is_dir():
            raise SystemExit(f"run directory does not exist: {run_dir}")
    return template, delays, trial_ids, run_dir


def plan_tasks(
    template: DQNTrialConfig,
    run_dir: Path,
    delays: list[int],
    trial_ids: list[int],
) -> tuple[list[DQNTrialConfig], int]:
    """Return the pending trial configs and the count of completed ones skipped.

    Completed trials must match the requested settings. Leftover directories of
    incomplete trials are moved under ``run_dir/_incomplete``.
    """
    pending: list[DQNTrialConfig] = []
    skipped = 0
    for delay in delays:
        delay_dir = create_delay_dir(run_dir, delay)
        for trial in trial_ids:
            config = template.for_trial(delay=delay, trial=trial)
            trial_dir = delay_dir / f"trial_{trial}"
            summary_path = trial_dir / "summary.json"
            if summary_path.exists():
                saved = json.loads(summary_path.read_text()).get("config", {})
                mismatches = config_mismatches(saved, config)
                if mismatches:
                    raise SystemExit(f"refusing to mix settings in {summary_path}: {mismatches}")
                skipped += 1
                continue
            if trial_dir.exists():
                # Keep artifacts from interrupted/failed trials, then let
                # train_trial create a clean directory at the usual path.
                backup_root = run_dir / "_incomplete"
                backup_root.mkdir(exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                trial_dir.rename(backup_root / f"delay_{delay}_trial_{trial}_{stamp}")
            pending.append(config)
    return pending, skipped


def summarize_time_to_optimal(run_dir: Path) -> Path | None:
    """Collect each trial's ``time_to_optimal`` into ``run_dir/time_to_optimal.json``.

    Trials that never reached the sustained criterion are right-censored at
    their cap: they are listed and counted, but their cap is never reported as
    a solve time. Returns None when no trial recorded the criterion.
    """
    by_delay: dict[str, dict[str, object]] = {}
    for summary_path in sorted(run_dir.glob("delay_*/trial_*/summary.json")):
        record = json.loads(summary_path.read_text()).get("time_to_optimal")
        if record is None:
            continue
        delay = summary_path.parent.parent.name.removeprefix("delay_")
        trial = int(summary_path.parent.name.removeprefix("trial_"))
        entry = by_delay.setdefault(delay, {"trials": {}})
        entry["trials"][trial] = {
            key: record[key]
            for key in (
                "first_optimal_timestep",
                "sustained_optimal_timestep",
                "censored",
                "censor_timestep",
            )
        }
    if not by_delay:
        return None
    for entry in by_delay.values():
        trials = dict(sorted(entry["trials"].items()))
        entry["trials"] = trials
        solved = sorted(
            t["sustained_optimal_timestep"] for t in trials.values() if not t["censored"]
        )
        first = sorted(
            t["first_optimal_timestep"]
            for t in trials.values()
            if t["first_optimal_timestep"] is not None
        )
        n = len(trials)
        entry.update(
            {
                "n_trials": n,
                "n_sustained": len(solved),
                "n_censored": n - len(solved),
                "n_first_optimal": len(first),
                # Order statistic over all trials; censored trials rank last, so
                # the median exists only when more than half of them solved.
                "median_sustained_timestep": (
                    solved[(n - 1) // 2] if len(solved) > (n - 1) // 2 else None
                ),
                "sustained_timesteps_solved": solved,
                "first_optimal_timesteps": first,
            }
        )
    out = run_dir / "time_to_optimal.json"
    out.write_text(json.dumps(dict(sorted(by_delay.items(), key=lambda kv: int(kv[0]))), indent=2) + "\n")
    for delay, entry in by_delay.items():
        print(
            f"TIME_TO_OPTIMAL delay={delay} sustained={entry['n_sustained']}/{entry['n_trials']} "
            f"censored={entry['n_censored']} median={entry['median_sustained_timestep']}",
            flush=True,
        )
    return out


def main() -> None:
    args = build_parser().parse_args()
    template, delays, trial_ids, run_dir = resolve_sweep(args)
    pending, skipped = plan_tasks(template, run_dir, delays, trial_ids)
    wandb = wandb_from_args(args, group=run_dir.name)

    print(f"RUN_DIR {run_dir.resolve()}", flush=True)
    print(
        f"ENV {template.env.env_id} DELAYS {delays} TRIALS {args.trials} WORKERS {args.workers} "
        f"RANDOMIZE {args.randomize or 'none'} PENDING {len(pending)} SKIPPED {skipped}",
        flush=True,
    )
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_trial, (config, run_dir, wandb)): config for config in pending}
        for future in as_completed(futures):
            config = futures[future]
            delay, trial = config.env.delay, config.trial
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
    if has_metrics and not args.no_plot:
        plot_path = plot_delay_survival(
            run_dir,
            smooth_window=100,
            onset_threshold=args.onset_threshold,
        )
        print(f"PLOT {plot_path.resolve()}", flush=True)
    summarize_time_to_optimal(run_dir)
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed; resume with --run-dir")
    print(f"All DQN trials finished under {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
