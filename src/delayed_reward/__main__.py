"""CLI entrypoint for training tabular Q-learning on a fixed Empty Minigrid."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from delayed_reward.agents.tabular_q import TabularQLearningAgent
from delayed_reward.environments.factory import DEFAULT_ENV_ID, create_empty_env
from delayed_reward.policies.epsilon_greedy import EpsilonGreedyPolicy
from delayed_reward.results.plotting import (
    DEFAULT_ONSET_THRESHOLD,
    plot_delay_survival,
)
from delayed_reward.results.saver import (
    create_delay_dir,
    create_run_dir,
    create_trial_dir,
    record_greedy_policy_gif,
    save_metrics,
)
from delayed_reward.training.observers import LoggingObserver
from delayed_reward.training.trainer import Trainer
from delayed_reward.training.wandb_logging import start_wandb_run


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


def policy_seed_for(base_seed: int, delay: int, trial: int) -> int:
    """Independent exploration RNG per (delay, trial); env layout stays fixed."""
    return base_seed + trial + 10_003 * delay


def train_one_trial(
    *,
    delay: int,
    trial: int,
    env_id: str,
    seed: int,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    log_every: int,
    delay_dir: Path,
    max_steps: int | None = None,
    wandb_enabled: bool = False,
    wandb_project: str = "delayed-reward",
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
) -> Path:
    """Train one (delay, trial) replicate and write artifacts under ``delay_dir``."""
    env_seed = seed
    explore_seed = policy_seed_for(seed, delay, trial)
    print(
        f"\n=== delay={delay} trial={trial} "
        f"(env_seed={env_seed}, policy_seed={explore_seed}) ==="
    )

    env = create_empty_env(
        env_id=env_id, seed=env_seed, delay=delay, max_steps=max_steps
    )
    rng = np.random.default_rng(explore_seed)
    policy = EpsilonGreedyPolicy(epsilon=epsilon, rng=rng)
    agent = TabularQLearningAgent(
        n_actions=env.n_actions,
        policy=policy,
        alpha=alpha,
        gamma=gamma,
    )
    config = {
        "episodes": episodes, "alpha": alpha, "gamma": gamma, "epsilon": epsilon,
        "env_seed": env_seed, "policy_seed": explore_seed, "trial": trial,
        "log_every": log_every, "delay": delay, "max_steps": max_steps, "env": env_id,
        "algorithm": "tabular-q-learning",
    }
    wandb_run = start_wandb_run(
        enabled=wandb_enabled, project=wandb_project, entity=wandb_entity,
        group=wandb_group, name=f"delay-{delay}-trial-{trial}", config=config,
    )
    observer = LoggingObserver(log_every=log_every, wandb_run=wandb_run)
    trainer = Trainer(env, agent, observers=[observer])

    try:
        trainer.train(episodes)
    finally:
        env.close()

    trial_dir = create_trial_dir(delay_dir, trial)
    save_metrics(
        trial_dir,
        rewards=observer.rewards,
        steps=observer.steps,
        config=config,
        window=log_every,
    )
    gif_path = record_greedy_policy_gif(
        agent.q_store,
        seed=env_seed,
        delay=delay,
        env_id=env_id,
        output_path=trial_dir / "policy.gif",
        max_steps=max_steps if max_steps is not None else 200,
        episode_max_steps=max_steps,
    )

    if observer.rewards:
        last = observer.rewards[-log_every:]
        print(
            f"delay={delay} trial={trial} final window mean return "
            f"({len(last)} eps): {sum(last) / len(last):.3f}"
        )
    print(f"Saved to {trial_dir.resolve()}")
    print(f"Policy GIF: {gif_path.resolve()}")
    if wandb_run is not None:
        wandb_run.finish()
    return trial_dir


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
