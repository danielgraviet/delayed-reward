"""CLI for training tabular Q-learning on a fixed Empty Minigrid, one trial at a time."""

from __future__ import annotations

import argparse
from pathlib import Path

from delayed_reward.environments.registry import DEFAULT_ENV_ID
from delayed_reward.experiments.tabular import policy_seed_for, train_one_trial
from delayed_reward.results.plotting import (
    DEFAULT_ONSET_THRESHOLD,
    plot_delay_survival,
)
from delayed_reward.results.saver import create_delay_dir, create_run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tabular Q-learning on fixed Empty Minigrid, sweeping reward delays "
            "to find how much delay the learner can still solve."
        ),
    )
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Base seed. Env layout uses this. Trial t uses policy seed "
            "base+t+10003*delay"
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help=(
            "Independent seeded replicates per delay "
            "(e.g. --trials 5 runs each delay five times)"
        ),
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="delayed-reward")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--delay",
        type=int,
        nargs="+",
        default=[0],
        metavar="N",
        help=(
            "One or more reward delays in steps after goal contact "
            "(0 = immediate). Example: --delay 0 2 5 10"
        ),
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=DEFAULT_ONSET_THRESHOLD,
        help="Rolling-mean return that counts as learning onset",
    )
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Episode truncation horizon. Default is MiniGrid's "
            "(400 on Empty-10x10). Delayed rewards still pending at "
            "truncation are delivered immediately."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory under which a timestamped run folder is created",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    delays = list(dict.fromkeys(args.delay))  # preserve order, drop duplicates
    if any(d < 0 for d in delays):
        raise SystemExit("--delay values must be >= 0")
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    if not 0.0 <= args.onset_threshold <= 1.0:
        raise SystemExit("--onset-threshold must be in [0, 1]")

    run_dir = create_run_dir(args.results_dir)
    print(f"Run directory: {run_dir.resolve()}")
    print(f"Delays: {delays}")
    print(f"Trials per delay: {args.trials}")
    print(f"Env seed (shared layout): {args.seed}")
    print("Policy seeds:")
    for delay in delays:
        for trial in range(args.trials):
            print(
                f"  delay={delay} trial={trial} -> "
                f"{policy_seed_for(args.seed, delay, trial)}"
            )

    for delay in delays:
        delay_dir = create_delay_dir(run_dir, delay)
        for trial in range(args.trials):
            train_one_trial(
                delay=delay,
                trial=trial,
                env_id=args.env_id,
                seed=args.seed,
                episodes=args.episodes,
                alpha=args.alpha,
                gamma=args.gamma,
                epsilon=args.epsilon,
                log_every=args.log_every,
                delay_dir=delay_dir,
                max_steps=args.max_steps,
                wandb_enabled=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_group=run_dir.name,
            )

    plot_path = plot_delay_survival(
        run_dir,
        smooth_window=args.log_every,
        onset_threshold=args.onset_threshold,
    )
    print(f"\nAll delays finished. Results under {run_dir.resolve()}")
    print(f"Delay survival plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
