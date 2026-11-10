#!/usr/bin/env python3
"""Run independent DQN trials concurrently in one run directory.

Mirrors ``scripts/parallel_sweep.py``'s resume/skip/plot pattern, but drives
``delayed_reward.dqn_experiment.train_trial`` (Stable-Baselines3 DQN on the
keydoor/empty MiniGrid envs) instead of the tabular Q-learning path.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delayed_reward.dqn_experiment import train_trial
from delayed_reward.environments.factory import DEFAULT_ENV_ID, KEYDOOR_ENV_ID
from delayed_reward.results.plotting import DEFAULT_ONSET_THRESHOLD, plot_delay_survival
from delayed_reward.results.saver import create_delay_dir, create_run_dir, create_trial_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-kind", choices=("empty", "keydoor"), default="empty")
    parser.add_argument(
        "--env-id",
        default=None,
        help=f"Defaults to {DEFAULT_ENV_ID!r} (empty) or {KEYDOOR_ENV_ID!r} (keydoor)",
    )
    parser.add_argument("--delay", type=int, nargs="+", default=[0])
    parser.add_argument("--subgoal-reward", type=float, default=0.3)
    parser.add_argument(
        "--reward-attribution", choices=("arrival", "origin"), default="arrival"
    )
    parser.add_argument(
        "--step-penalty",
        type=float,
        default=0.0,
        help="Fixed cost subtracted from every step's reward (--env-kind keydoor only)",
    )
    parser.add_argument(
        "--checkpoint-window",
        type=int,
        default=100,
        help="Save/GIF the best deterministic-eval checkpoint from the last N evals",
    )
    parser.add_argument(
        "--eval-every-episodes",
        type=int,
        default=20,
        help="Deterministic eval cadence; pass 0 to disable",
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--onset-threshold", type=float, default=DEFAULT_ONSET_THRESHOLD)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing run directory to resume; completed, matching trials are skipped",
    )
    parser.add_argument("--wandb", action="store_true", help="Log each trial to Weights & Biases")
    parser.add_argument("--wandb-project", default="delayed-reward")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-log-every", type=int, default=1000)
    parser.add_argument(
        "--trial-ids",
        type=int,
        nargs="+",
        default=None,
        help="Run only these trial indices (each < --trials); used by remote launchers",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the survival plot")
    return parser


def _expected_config(args: argparse.Namespace, env_id: str, delay: int, trial: int) -> dict[str, object]:
    return {
        "env_kind": args.env_kind,
        "env": env_id,
        "delay": delay,
        "subgoal_reward": args.subgoal_reward if args.env_kind == "keydoor" else None,
        "reward_attribution": args.reward_attribution,
        "step_penalty": args.step_penalty if args.env_kind == "keydoor" else None,
        "trial": trial,
        "seed": args.seed + trial,
        "total_timesteps": args.timesteps,
        "max_steps": args.max_steps,
    }


def _run_trial(task: tuple[argparse.Namespace, str, Path, int, int]) -> Path:
    args, env_id, run_dir, delay, trial = task
    trial_dir = create_trial_dir(create_delay_dir(run_dir, delay), trial)
    train_trial(
        trial=trial,
        env_id=env_id,
        seed=args.seed,
        total_timesteps=args.timesteps,
        delay=delay,
        max_steps=args.max_steps,
        trial_dir=trial_dir,
        env_kind=args.env_kind,
        subgoal_reward=args.subgoal_reward,
        reward_attribution=args.reward_attribution,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=run_dir.name,
        wandb_log_every=args.wandb_log_every,
        eval_every_episodes=args.eval_every_episodes or None,
        step_penalty=args.step_penalty,
        checkpoint_window=args.checkpoint_window,
    )
    return trial_dir


def resolve_sweep(args: argparse.Namespace) -> tuple[list[int], list[int], str, Path]:
    """Validate args and return (delays, trial_ids, env_id, run_dir)."""
    delays = list(dict.fromkeys(args.delay))
    if args.trials < 1 or args.timesteps < 1 or args.workers < 1:
        raise SystemExit("trials, timesteps, and workers must be >= 1")
    if any(delay < 0 for delay in delays):
        raise SystemExit("delay values must be >= 0")
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit("max-steps must be >= 1")
    if args.step_penalty < 0 or args.checkpoint_window < 1:
        raise SystemExit("step-penalty must be >= 0 and checkpoint-window must be >= 1")
    if not 0.0 <= args.onset_threshold <= 1.0:
        raise SystemExit("onset-threshold must be in [0, 1]")
    trial_ids = list(dict.fromkeys(args.trial_ids)) if args.trial_ids else list(range(args.trials))
    if any(not 0 <= trial < args.trials for trial in trial_ids):
        raise SystemExit("trial-ids must be in [0, trials)")

    env_id = args.env_id or (KEYDOOR_ENV_ID if args.env_kind == "keydoor" else DEFAULT_ENV_ID)

    if args.run_dir is None:
        run_dir = create_run_dir(args.results_dir)
    else:
        run_dir = args.run_dir
        if not run_dir.is_dir():
            raise SystemExit(f"run directory does not exist: {run_dir}")
    return delays, trial_ids, env_id, run_dir


def plan_tasks(
    args: argparse.Namespace,
    env_id: str,
    run_dir: Path,
    delays: list[int],
    trial_ids: list[int],
) -> tuple[list[tuple[int, int]], int]:
    """Return pending (delay, trial) pairs and the count of completed ones skipped.

    Completed trials must match the requested settings. Leftover directories of
    incomplete trials are moved under ``run_dir/_incomplete``.
    """
    pending: list[tuple[int, int]] = []
    skipped = 0
    for delay in delays:
        delay_dir = create_delay_dir(run_dir, delay)
        for trial in trial_ids:
            summary_path = delay_dir / f"trial_{trial}" / "summary.json"
            if summary_path.exists():
                saved_config = json.loads(summary_path.read_text()).get("config", {})
                expected = _expected_config(args, env_id, delay, trial)
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
                    # train_trial create a clean directory at the usual path.
                    backup_root = run_dir / "_incomplete"
                    backup_root.mkdir(exist_ok=True)
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                    trial_dir.rename(
                        backup_root / f"delay_{delay}_trial_{trial}_{stamp}"
                    )
                pending.append((delay, trial))
    return pending, skipped


def main() -> None:
    args = build_parser().parse_args()
    delays, trial_ids, env_id, run_dir = resolve_sweep(args)
    pending, skipped = plan_tasks(args, env_id, run_dir, delays, trial_ids)
    tasks = [(args, env_id, run_dir, delay, trial) for delay, trial in pending]

    print(f"RUN_DIR {run_dir.resolve()}", flush=True)
    print(
        f"ENV {env_id} DELAYS {delays} TRIALS {args.trials} WORKERS {args.workers} "
        f"PENDING {len(tasks)} SKIPPED {skipped}",
        flush=True,
    )
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_trial, task): task[3:] for task in tasks}
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
    if has_metrics and not args.no_plot:
        plot_path = plot_delay_survival(
            run_dir,
            smooth_window=100,
            onset_threshold=args.onset_threshold,
        )
        print(f"PLOT {plot_path.resolve()}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed; resume with --run-dir")
    print(f"All DQN trials finished under {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
