#!/usr/bin/env python3
"""Check whether a trained keydoor DQN is still keying off key -> door -> goal.

Loads a saved Stable-Baselines3 DQN model and rolls out deterministic
(greedy) episodes in the clean training layout and in a distractor-populated
variant of the same layout (see ``environments.distractors``), reporting
key-pickup rate, goal-reach rate, and mean return for each. A policy that
actually learned "grab key, go to door, get reward" should see pickup/goal
rates hold up under clutter; a large drop points to the policy having
overfit to incidental, clutter-free cues instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from stable_baselines3 import DQN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delayed_reward.environments.factory import (
    KEYDOOR_DISTRACTOR_ENV_ID,
    KEYDOOR_ENV_ID,
    create_keydoor_dqn_env,
)


def rollout(model: DQN, env, *, max_steps: int) -> tuple[float, bool, bool]:
    """Run one greedy episode; return (total_reward, picked_up_key, reached_goal)."""
    observation, _info = env.reset()
    total_reward = 0.0
    picked_up_key = False
    reached_goal = False
    for _ in range(max_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        picked_up_key = picked_up_key or bool(info.get("subgoal_reached", False))
        if terminated or truncated:
            reached_goal = bool(terminated)
            break
    return total_reward, picked_up_key, reached_goal


def save_rollout_gif(model: DQN, env, *, max_steps: int, path: Path) -> None:
    observation, _info = env.reset()
    frames = [Image.fromarray(np.asarray(env.render()))]
    for _ in range(max_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, _reward, terminated, truncated, _info = env.step(int(action))
        frames.append(Image.fromarray(np.asarray(env.render())))
        if terminated or truncated:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)


def evaluate(
    model: DQN,
    *,
    env_id: str,
    num_distractors: int,
    episodes: int,
    seed: int,
    subgoal_reward: float,
    max_steps: int,
) -> dict[str, float]:
    returns: list[float] = []
    pickups = 0
    goals = 0
    for episode in range(episodes):
        env = create_keydoor_dqn_env(
            env_id=env_id,
            seed=seed + episode,
            subgoal_reward=subgoal_reward,
            num_distractors=num_distractors,
            max_steps=max_steps,
        )
        try:
            total_reward, picked_up_key, reached_goal = rollout(model, env, max_steps=max_steps)
        finally:
            env.close()
        returns.append(total_reward)
        pickups += int(picked_up_key)
        goals += int(reached_goal)
    return {
        "mean_return": float(np.mean(returns)),
        "key_pickup_rate": pickups / episodes,
        "goal_reach_rate": goals / episodes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="Path to a saved SB3 model, e.g. trial dir/model.zip")
    parser.add_argument("--env-id", default=KEYDOOR_ENV_ID, help="Clean (no-distractor) env to compare against")
    parser.add_argument("--distractor-env-id", default=KEYDOOR_DISTRACTOR_ENV_ID)
    parser.add_argument("--num-distractors", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subgoal-reward", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--gif-dir",
        type=Path,
        default=None,
        help="If set, save clean.gif and distractor.gif rollouts (seed episode) here",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    if args.num_distractors < 0:
        raise SystemExit("--num-distractors must be >= 0")

    model = DQN.load(args.model_path)

    variants = (
        ("clean", args.env_id, 0),
        ("distractor", args.distractor_env_id, args.num_distractors),
    )
    for label, env_id, num_distractors in variants:
        metrics = evaluate(
            model,
            env_id=env_id,
            num_distractors=num_distractors,
            episodes=args.episodes,
            seed=args.seed,
            subgoal_reward=args.subgoal_reward,
            max_steps=args.max_steps,
        )
        print(
            f"{label:>10}: key_pickup_rate={metrics['key_pickup_rate']:.2f} "
            f"goal_reach_rate={metrics['goal_reach_rate']:.2f} "
            f"mean_return={metrics['mean_return']:.3f} episodes={args.episodes}"
        )
        if args.gif_dir is not None:
            gif_env = create_keydoor_dqn_env(
                env_id=env_id,
                seed=args.seed,
                subgoal_reward=args.subgoal_reward,
                num_distractors=num_distractors,
                max_steps=args.max_steps,
                render_mode="rgb_array",
            )
            try:
                save_rollout_gif(
                    model, gif_env, max_steps=args.max_steps, path=args.gif_dir / f"{label}.gif"
                )
            finally:
                gif_env.close()


if __name__ == "__main__":
    main()
