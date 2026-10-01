"""CLI entrypoint for training tabular Q-learning on MiniGrid-Empty-5x5."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from delayed_reward.agents.tabular_q import TabularQLearningAgent
from delayed_reward.environments.factory import create_empty_5x5
from delayed_reward.policies.epsilon_greedy import EpsilonGreedyPolicy
from delayed_reward.results.saver import (
    create_run_dir,
    record_greedy_policy_gif,
    save_metrics,
)
from delayed_reward.training.observers import LoggingObserver
from delayed_reward.training.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tabular Q-learning on a fixed MiniGrid-Empty-5x5 environment",
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    parser.add_argument("--seed", type=int, default=0, help="Env reseed + RNG seed")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory under which a timestamped run folder is created",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    env = create_empty_5x5(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    policy = EpsilonGreedyPolicy(epsilon=args.epsilon, rng=rng)
    agent = TabularQLearningAgent(
        n_actions=env.n_actions,
        policy=policy,
        alpha=args.alpha,
        gamma=args.gamma,
    )
    observer = LoggingObserver(log_every=args.log_every)
    trainer = Trainer(env, agent, observers=[observer])

    try:
        trainer.train(args.episodes)
    finally:
        env.close()

    run_dir = create_run_dir(args.results_dir)
    config = {
        "episodes": args.episodes,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "log_every": args.log_every,
        "env": "MiniGrid-Empty-5x5-v0",
    }
    save_metrics(
        run_dir,
        rewards=observer.rewards,
        steps=observer.steps,
        config=config,
        window=args.log_every,
    )
    gif_path = record_greedy_policy_gif(
        agent.q_store,
        seed=args.seed,
        output_path=run_dir / "policy.gif",
    )

    if observer.rewards:
        last = observer.rewards[-args.log_every :]
        print(
            f"\nDone. Final window mean return "
            f"({len(last)} eps): {sum(last) / len(last):.3f}"
        )
    print(f"Results saved to {run_dir.resolve()}")
    print(f"Policy GIF: {gif_path.resolve()}")


if __name__ == "__main__":
    main()
