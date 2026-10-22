"""Train Stable-Baselines3 DQN on the project's fully observed MiniGrid."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from delayed_reward.environments.factory import (
    DEFAULT_ENV_ID,
    KEYDOOR_ENV_ID,
    create_empty_dqn_env,
    create_keydoor_dqn_env,
)
from delayed_reward.results.saver import (
    create_delay_dir,
    create_run_dir,
    create_trial_dir,
    save_metrics,
)


def _make_env(
    *,
    env_kind: str,
    env_id: str,
    seed: int,
    delay: int,
    subgoal_reward: float,
    render_mode: str | None = None,
    max_steps: int | None = None,
) -> gym.Env:
    if env_kind == "keydoor":
        return create_keydoor_dqn_env(
            env_id=env_id,
            seed=seed,
            delay=delay,
            subgoal_reward=subgoal_reward,
            render_mode=render_mode,
            max_steps=max_steps,
        )
    return create_empty_dqn_env(
        env_id=env_id, seed=seed, delay=delay, render_mode=render_mode, max_steps=max_steps
    )


class EpisodeMetricsCallback(BaseCallback):
    """Collect completed episode returns and lengths from Gymnasium stats."""

    def __init__(self) -> None:
        super().__init__()
        self.rewards: list[float] = []
        self.steps: list[int] = []

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done and "episode" in info:
                self.rewards.append(float(info["episode"]["r"]))
                self.steps.append(int(info["episode"]["l"]))
        return True


def train_trial(
    *,
    trial: int,
    env_id: str,
    seed: int,
    total_timesteps: int,
    delay: int,
    max_steps: int | None,
    trial_dir: Path,
    env_kind: str = "empty",
    subgoal_reward: float = 1.0,
) -> None:
    env = _make_env(
        env_kind=env_kind,
        env_id=env_id,
        seed=seed + trial,
        delay=delay,
        subgoal_reward=subgoal_reward,
        max_steps=max_steps,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    callback = EpisodeMetricsCallback()
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [128, 128]},
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10.0,
        seed=seed + trial,
        device="auto",
        verbose=0,
    )
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    finally:
        env.close()

    config: dict[str, object] = {
        "algorithm": "stable-baselines3-dqn",
        "env": env_id,
        "env_kind": env_kind,
        "delay": delay,
        "subgoal_reward": subgoal_reward if env_kind == "keydoor" else None,
        "trial": trial,
        "seed": seed + trial,
        "total_timesteps": total_timesteps,
        "max_steps": max_steps,
        "observation": "fully-observed encoded grid plus normalized agent pose",
        "network": [128, 128],
        "learning_rate": 1e-3,
        "buffer_size": 50_000,
        "learning_starts": 1_000,
        "batch_size": 64,
        "gamma": 0.99,
        "train_freq": 4,
        "target_update_interval": 1_000,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "device": str(model.device),
    }
    save_metrics(
        trial_dir,
        rewards=callback.rewards,
        steps=callback.steps,
        config=config,
        window=100,
    )
    model.save(trial_dir / "model")
    _save_policy_gif(
        model,
        env_kind=env_kind,
        env_id=env_id,
        seed=seed + trial,
        delay=delay,
        subgoal_reward=subgoal_reward,
        max_steps=max_steps or 200,
        path=trial_dir / "policy.gif",
    )
    recent = callback.rewards[-100:]
    print(
        f"trial={trial} episodes={len(callback.rewards)} "
        f"last_100_mean_return={np.mean(recent) if recent else float('nan'):.3f} "
        f"device={model.device} path={trial_dir}",
        flush=True,
    )


def _save_policy_gif(
    model: DQN,
    *,
    env_kind: str,
    env_id: str,
    seed: int,
    delay: int,
    subgoal_reward: float,
    max_steps: int,
    path: Path,
) -> None:
    env = _make_env(
        env_kind=env_kind,
        env_id=env_id,
        seed=seed,
        delay=delay,
        subgoal_reward=subgoal_reward,
        render_mode="rgb_array",
    )
    frames: list[Image.Image] = []
    try:
        observation, _ = env.reset()
        frames.append(Image.fromarray(np.asarray(env.render())))
        for _ in range(max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, _info = env.step(int(action))
            frames.append(Image.fromarray(np.asarray(env.render())))
            if terminated or truncated:
                break
    finally:
        env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-kind", choices=("empty", "keydoor"), default="empty")
    parser.add_argument(
        "--env-id",
        default=None,
        help=f"Defaults to {DEFAULT_ENV_ID!r} (empty) or {KEYDOOR_ENV_ID!r} (keydoor)",
    )
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument(
        "--subgoal-reward",
        type=float,
        default=1.0,
        help="Reward for picking up the key (--env-kind keydoor only)",
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.trials < 1 or args.timesteps < 1:
        raise SystemExit("trials and timesteps must be >= 1")
    if args.delay < 0 or (args.max_steps is not None and args.max_steps < 1):
        raise SystemExit("delay must be >= 0 and max-steps must be >= 1")
    env_id = args.env_id or (KEYDOOR_ENV_ID if args.env_kind == "keydoor" else DEFAULT_ENV_ID)
    run_dir = args.run_dir or create_run_dir(args.results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    delay_dir = create_delay_dir(run_dir, args.delay)
    print(
        f"RUN_DIR {run_dir.resolve()} trials={args.trials} "
        f"timesteps={args.timesteps} delay={args.delay} env_kind={args.env_kind}",
        flush=True,
    )
    for trial in range(args.trials):
        trial_dir = create_trial_dir(delay_dir, trial)
        train_trial(
            trial=trial,
            env_id=env_id,
            seed=args.seed,
            total_timesteps=args.timesteps,
            delay=args.delay,
            max_steps=args.max_steps,
            trial_dir=trial_dir,
            env_kind=args.env_kind,
            subgoal_reward=args.subgoal_reward,
        )


if __name__ == "__main__":
    main()
