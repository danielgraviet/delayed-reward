"""CLI entrypoint for training tabular Q-learning on a fixed Empty Minigrid."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from delayed_reward.agents.tabular_q import TabularQLearningAgent
from delayed_reward.environments.factory import DEFAULT_ENV_ID, create_empty_env
from delayed_reward.policies.epsilon_greedy import EpsilonGreedyPolicy
from delayed_reward.results.saver import (
    create_delay_dir,
    create_run_dir,
    record_greedy_policy_gif,
    save_metrics,
)
from delayed_reward.training.observers import LoggingObserver
from delayed_reward.training.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tabular Q-learning on a fixed Empty Minigrid environment",
    )
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    parser.add_argument("--seed", type=int, default=0, help="Env reseed + RNG seed")
    parser.add_argument("--log-every", type=int, default=100)
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
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory under which a timestamped run folder is created",
    )
    return parser


def train_for_delay(
    *,
    delay: int,
    env_id: str,
    seed: int,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    log_every: int,
    run_dir: Path,
) -> Path:
    """Train one delay setting and write metrics + policy GIF under ``run_dir``."""
    print(f"\n=== delay={delay} ===")
    env = create_empty_env(env_id=env_id, seed=seed, delay=delay)
    rng = np.random.default_rng(seed)
    policy = EpsilonGreedyPolicy(epsilon=epsilon, rng=rng)
    agent = TabularQLearningAgent(
        n_actions=env.n_actions,
        policy=policy,
        alpha=alpha,
        gamma=gamma,
    )
    observer = LoggingObserver(log_every=log_every)
    trainer = Trainer(env, agent, observers=[observer])

    try:
        trainer.train(episodes)
    finally:
        env.close()

    delay_dir = create_delay_dir(run_dir, delay)
    config = {
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "seed": seed,
        "log_every": log_every,
        "delay": delay,
        "env": env_id,
    }
    save_metrics(
        delay_dir,
        rewards=observer.rewards,
        steps=observer.steps,
        config=config,
        window=log_every,
    )
    gif_path = record_greedy_policy_gif(
        agent.q_store,
        seed=seed,
        delay=delay,
        env_id=env_id,
        output_path=delay_dir / "policy.gif",
    )

    if observer.rewards:
        last = observer.rewards[-log_every:]
        print(
            f"delay={delay} final window mean return "
            f"({len(last)} eps): {sum(last) / len(last):.3f}"
        )
    print(f"Saved to {delay_dir.resolve()}")
    print(f"Policy GIF: {gif_path.resolve()}")
    return delay_dir


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    delays = list(dict.fromkeys(args.delay))  # preserve order, drop duplicates
    if any(d < 0 for d in delays):
        raise SystemExit("--delay values must be >= 0")

    run_dir = create_run_dir(args.results_dir)
    print(f"Run directory: {run_dir.resolve()}")
    print(f"Delays: {delays}")

    for delay in delays:
        train_for_delay(
            delay=delay,
            env_id=args.env_id,
            seed=args.seed,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            log_every=args.log_every,
            run_dir=run_dir,
        )

    print(f"\nAll delays finished. Results under {run_dir.resolve()}")


if __name__ == "__main__":
    main()
