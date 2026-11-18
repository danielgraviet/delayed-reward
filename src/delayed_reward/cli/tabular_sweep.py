"""Run independent tabular Q-learning trials concurrently in one run directory, resuming completed ones."""

from __future__ import annotations

import argparse
from pathlib import Path

from delayed_reward.cli.tabular import add_tabular_args, tabular_config_from_args, wandb_from_args
from delayed_reward.results.saver import create_run_dir
from delayed_reward.sweep.tabular import run_trial_task
from delayed_reward.sweep.trials import plan_tasks, plot_survival_if_any, run_in_processes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_tabular_args(parser)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing run directory to resume; completed, matching trials are skipped",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.trials < 1 or args.workers < 1:
        raise SystemExit("trials and workers must be >= 1")
    template, delays = tabular_config_from_args(args)
    if args.run_dir is None:
        run_dir = create_run_dir(args.results_dir)
    else:
        run_dir = args.run_dir
        if not run_dir.is_dir():
            raise SystemExit(f"run directory does not exist: {run_dir}")

    pending, skipped = plan_tasks(template, run_dir, delays, range(args.trials))
    wandb = wandb_from_args(args, group=run_dir.name)
    print(f"RUN_DIR {run_dir.resolve()}", flush=True)
    print(
        f"DELAYS {delays} TRIALS {args.trials} WORKERS {args.workers} PENDING {len(pending)} SKIPPED {skipped}",
        flush=True,
    )
    tasks = [(config.delay, config.trial, (config, run_dir, wandb)) for config in pending]
    failures = run_in_processes(run_trial_task, tasks, workers=args.workers)
    plot_survival_if_any(run_dir, smooth_window=template.log_every, onset_threshold=args.onset_threshold)
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed; resume with --run-dir")
    print(f"All delays finished. Results under {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
