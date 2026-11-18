"""CLI for training tabular Q-learning on a fixed Empty Minigrid, one trial at a time."""

from __future__ import annotations

import argparse
from pathlib import Path

from delayed_reward.config import TabularTrialConfig, WandbSettings
from delayed_reward.environments.registry import DEFAULT_ENV_ID
from delayed_reward.experiments.tabular import train_one_trial
from delayed_reward.results.plotting import DEFAULT_ONSET_THRESHOLD, plot_delay_survival
from delayed_reward.results.saver import create_delay_dir, create_run_dir


def add_tabular_args(parser: argparse.ArgumentParser) -> None:
    """Add the options that define a tabular trial (and its W&B logging), shared by the tabular CLIs."""
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed. Env layout uses this. Trial t uses policy seed base+t+10003*delay",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Episode truncation horizon. Default is MiniGrid's (400 on Empty-10x10). "
            "Delayed rewards still pending at truncation are delivered immediately."
        ),
    )
    parser.add_argument(
        "--delay",
        type=int,
        nargs="+",
        default=[0],
        metavar="N",
        help="One or more reward delays in steps after goal contact (0 = immediate). Example: --delay 0 2 5 10",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=DEFAULT_ONSET_THRESHOLD,
        help="Rolling-mean return that counts as learning onset",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--wandb", action="store_true", help="Log each trial to Weights & Biases")
    parser.add_argument("--wandb-project", default="delayed-reward")
    parser.add_argument("--wandb-entity", default=None)


def tabular_config_from_args(args: argparse.Namespace) -> tuple[TabularTrialConfig, list[int]]:
    """The trial template (delay 0, trial 0) and the deduplicated delays from ``add_tabular_args`` options."""
    delays = list(dict.fromkeys(args.delay))
    if any(delay < 0 for delay in delays):
        raise SystemExit("delay values must be >= 0")
    if not 0.0 <= args.onset_threshold <= 1.0:
        raise SystemExit("onset-threshold must be in [0, 1]")
    try:
        template = TabularTrialConfig(
            env_id=args.env_id,
            max_steps=args.max_steps,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            seed=args.seed,
            log_every=args.log_every,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return template, delays


def wandb_from_args(args: argparse.Namespace, *, group: str) -> WandbSettings | None:
    """W&B settings from ``add_tabular_args`` options, or None without ``--wandb``."""
    if not args.wandb:
        return None
    return WandbSettings(project=args.wandb_project, entity=args.wandb_entity, group=group)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tabular Q-learning on fixed Empty Minigrid, sweeping reward delays "
            "to find how much delay the learner can still solve. Trials run sequentially."
        ),
    )
    add_tabular_args(parser)
    parser.add_argument("--trials", type=int, default=1, help="Independent seeded replicates per delay")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.trials < 1:
        raise SystemExit("trials must be >= 1")
    template, delays = tabular_config_from_args(args)

    run_dir = create_run_dir(args.results_dir)
    wandb = wandb_from_args(args, group=run_dir.name)
    print(f"Run directory: {run_dir.resolve()}")
    print(f"Delays: {delays}")
    print(f"Trials per delay: {args.trials}")
    print(f"Env seed (shared layout): {template.seed}")
    configs = [template.for_trial(delay=delay, trial=trial) for delay in delays for trial in range(args.trials)]
    print("Policy seeds:")
    for config in configs:
        print(f"  delay={config.delay} trial={config.trial} -> {config.policy_seed}")

    for config in configs:
        train_one_trial(config, create_delay_dir(run_dir, config.delay), wandb=wandb)

    plot_path = plot_delay_survival(run_dir, smooth_window=template.log_every, onset_threshold=args.onset_threshold)
    print(f"\nAll delays finished. Results under {run_dir.resolve()}")
    print(f"Delay survival plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
