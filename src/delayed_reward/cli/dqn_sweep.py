"""Run independent DQN trials concurrently in one run directory, resuming completed ones."""

from __future__ import annotations

import argparse
from pathlib import Path

from delayed_reward.cli.dqn import add_dqn_args, dqn_config_from_args, wandb_from_args
from delayed_reward.config import DQNTrialConfig
from delayed_reward.results.saver import create_run_dir
from delayed_reward.sweep.dqn import run_trial_task, summarize_time_to_optimal
from delayed_reward.sweep.trials import plan_tasks, run_in_processes, write_run_overview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dqn_args(parser)
    parser.add_argument("--delay", type=int, nargs="+", default=[0])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
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
    parser.add_argument("--no-plot", action="store_true", help="Skip the learning charts")
    return parser


def resolve_sweep(args: argparse.Namespace) -> tuple[DQNTrialConfig, list[int], list[int], Path]:
    """Validate args and return (config template, delays, trial_ids, run_dir)."""
    delays = list(dict.fromkeys(args.delay))
    if args.trials < 1 or args.workers < 1:
        raise SystemExit("trials and workers must be >= 1")
    if any(delay < 0 for delay in delays):
        raise SystemExit("delay values must be >= 0")
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    template, delays, trial_ids, run_dir = resolve_sweep(args)
    pending, skipped = plan_tasks(template, run_dir, delays, trial_ids)
    wandb = wandb_from_args(args, group=run_dir.name)

    print(f"RUN_DIR {run_dir.resolve()}", flush=True)
    print(
        f"ENV {template.env.env_id} DELAYS {delays} TRIALS {args.trials} WORKERS {args.workers} "
        f"RANDOMIZE {args.randomize or 'none'} PENDING {len(pending)} SKIPPED {skipped}",
        flush=True,
    )
    tasks = [(config.env.delay, config.trial, (config, run_dir, wandb)) for config in pending]
    failures = run_in_processes(run_trial_task, tasks, workers=args.workers)
    if not args.no_plot:
        write_run_overview(run_dir)
    summarize_time_to_optimal(run_dir)
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed; resume with --run-dir")
    print(f"All DQN trials finished under {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
