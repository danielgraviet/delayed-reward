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
from delayed_reward.masked_dqn_policy import MaskedDQNPolicy
from delayed_reward.training.wandb_logging import log_metrics, start_wandb_run
from delayed_reward.results.saver import (
    create_delay_dir,
    create_run_dir,
    create_trial_dir,
    save_eval_metrics,
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
    """Collect episode metrics and optionally relabel delayed subgoal rewards."""

    def __init__(self, *, reward_attribution: str = "arrival", wandb_run=None, log_every: int = 1000) -> None:
        super().__init__()
        if reward_attribution not in {"arrival", "origin"}:
            raise ValueError("reward_attribution must be 'arrival' or 'origin'")
        self.reward_attribution = reward_attribution
        self.rewards: list[float] = []
        self.steps: list[int] = []
        self._transition_indices: dict[int, int] = {}
        self.wandb_run = wandb_run
        self.log_every = log_every

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        metrics: dict[str, float] = {}
        buffer = self.model.replay_buffer
        if buffer is not None:
            current_index = (buffer.pos - 1) % buffer.buffer_size
            for info in infos:
                step = info.get("environment_step")
                if step is not None:
                    self._transition_indices[int(step)] = current_index
                    oldest = int(step) - buffer.buffer_size
                    self._transition_indices = {
                        key: value for key, value in self._transition_indices.items() if key > oldest
                    }
                if self.reward_attribution == "origin" and "subgoal_reward_origin_step" in info:
                    origin = int(info["subgoal_reward_origin_step"])
                    origin_index = self._transition_indices.get(origin)
                    if origin_index is None:
                        raise RuntimeError(f"origin transition {origin} is no longer in replay")
                    amount = float(info["subgoal_reward_amount"])
                    buffer.rewards[current_index, 0] -= amount
                    buffer.rewards[origin_index, 0] += amount
        for done, info in zip(self.locals["dones"], infos, strict=True):
            if done and "episode" in info:
                self.rewards.append(float(info["episode"]["r"]))
                self.steps.append(int(info["episode"]["l"]))
                metrics.update({
                    "episode/return": float(info["episode"]["r"]),
                    "episode/steps": float(info["episode"]["l"]),
                    "episode/count": float(len(self.rewards)),
                })
        if self.num_timesteps % self.log_every == 0:
            values = self.model.logger.name_to_value
            metrics.update({
                key.replace("/", "_"): float(value)
                for key, value in values.items()
                if key.startswith(("train/", "rollout/")) and np.isscalar(value)
            })
            metrics["timesteps"] = float(self.num_timesteps)
        if metrics:
            log_metrics(self.wandb_run, metrics, step=int(self.num_timesteps))
        return True


class GreedyEvalCallback(BaseCallback):
    """Periodically run the current policy with exploration turned off.

    Training episodes mix in epsilon-random actions by design (see
    ``exploration_final_eps`` below -- it never reaches 0), so even the
    training return curve is noisier than the policy actually is. This
    callback separately asks "how good is the greedy policy right now" by
    rolling out ``n_eval_episodes`` episodes with ``deterministic=True`` and
    no exploration, every ``eval_every_episodes`` completed training
    episodes.

    The environment is reseeded to the same fixed layout every episode (see
    ``ReseedWrapper`` in ``environments/factory.py``), so a greedy policy's
    rollout is itself deterministic -- there's no maze-layout or start-state
    randomness left once exploration is off. That's why ``n_eval_episodes``
    defaults to 1: running it more times would just repeat the identical
    trajectory. The parameter exists in case a future environment variant
    randomizes the layout per episode, where repeated eval rollouts would
    actually differ and averaging them would matter.
    """

    def __init__(
        self,
        *,
        eval_env: gym.Env,
        eval_every_episodes: int,
        max_steps: int,
        n_eval_episodes: int = 1,
    ) -> None:
        super().__init__()
        if eval_every_episodes < 1:
            raise ValueError("eval_every_episodes must be >= 1")
        if n_eval_episodes < 1:
            raise ValueError("n_eval_episodes must be >= 1")
        self.eval_env = eval_env
        self.eval_every_episodes = eval_every_episodes
        self.max_steps = max_steps
        self.n_eval_episodes = n_eval_episodes
        self._train_episode_count = 0
        self.eval_train_episodes: list[int] = []
        self.eval_timesteps: list[int] = []
        self.eval_returns: list[float] = []
        self.eval_steps: list[float] = []

    def _run_eval(self) -> tuple[float, float]:
        returns: list[float] = []
        lengths: list[int] = []
        for _ in range(self.n_eval_episodes):
            observation, _info = self.eval_env.reset()
            total_reward = 0.0
            length = 0
            for _ in range(self.max_steps):
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _info = self.eval_env.step(
                    int(action)
                )
                total_reward += float(reward)
                length += 1
                if terminated or truncated:
                    break
            returns.append(total_reward)
            lengths.append(length)
        return float(np.mean(returns)), float(np.mean(lengths))

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done and "episode" in info:
                self._train_episode_count += 1
                if self._train_episode_count % self.eval_every_episodes == 0:
                    eval_return, eval_length = self._run_eval()
                    self.eval_train_episodes.append(self._train_episode_count)
                    self.eval_timesteps.append(int(self.num_timesteps))
                    self.eval_returns.append(eval_return)
                    self.eval_steps.append(eval_length)
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
    reward_attribution: str = "arrival",
    wandb_enabled: bool = False,
    wandb_project: str = "delayed-reward",
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_log_every: int = 1000,
    eval_every_episodes: int | None = 20,
) -> None:
    if wandb_log_every < 1:
        raise ValueError("wandb_log_every must be >= 1")
    if eval_every_episodes is not None and eval_every_episodes < 1:
        raise ValueError("eval_every_episodes must be >= 1 or None")
    env = _make_env(
        env_kind=env_kind,
        env_id=env_id,
        seed=seed + trial,
        delay=delay,
        subgoal_reward=subgoal_reward,
        max_steps=max_steps,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    config: dict[str, object] = {
        "algorithm": "stable-baselines3-dqn", "env": env_id, "env_kind": env_kind,
        "delay": delay, "trial": trial, "seed": seed + trial,
        "total_timesteps": total_timesteps, "max_steps": max_steps,
        "subgoal_reward": subgoal_reward if env_kind == "keydoor" else None,
        "reward_attribution": reward_attribution,
    }
    wandb_run = start_wandb_run(
        enabled=wandb_enabled, project=wandb_project, entity=wandb_entity,
        group=wandb_group, name=f"delay-{delay}-trial-{trial}", config=config,
    )
    callback = EpisodeMetricsCallback(
        reward_attribution=reward_attribution, wandb_run=wandb_run, log_every=wandb_log_every
    )
    callbacks: list[BaseCallback] = [callback]
    eval_callback: GreedyEvalCallback | None = None
    if eval_every_episodes is not None:
        eval_env = _make_env(
            env_kind=env_kind,
            env_id=env_id,
            seed=seed + trial,
            delay=delay,
            subgoal_reward=subgoal_reward,
            max_steps=max_steps,
        )
        eval_callback = GreedyEvalCallback(
            eval_env=eval_env,
            eval_every_episodes=eval_every_episodes,
            max_steps=max_steps or 2000,
        )
        callbacks.append(eval_callback)
    policy = MaskedDQNPolicy if env_kind == "keydoor" else "MlpPolicy"
    model = DQN(
        policy,
        env,
        policy_kwargs={"net_arch": [128, 128]},
        learning_rate=1e-3,
        buffer_size=150_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.995,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10.0,
        seed=seed + trial,
        device="auto",
        verbose=0,
    )
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    finally:
        env.close()
        if eval_callback is not None:
            eval_callback.eval_env.close()

    config: dict[str, object] = {
        "algorithm": "stable-baselines3-dqn",
        "env": env_id,
        "env_kind": env_kind,
        "action_masking": env_kind == "keydoor",
        "delay": delay,
        "subgoal_reward": subgoal_reward if env_kind == "keydoor" else None,
        "reward_attribution": reward_attribution,
        "trial": trial,
        "seed": seed + trial,
        "total_timesteps": total_timesteps,
        "max_steps": max_steps,
        "observation": "fully-observed encoded grid plus normalized agent pose",
        "network": [128, 128],
        "learning_rate": 1e-3,
        "buffer_size": 150_000,
        "learning_starts": 1_000,
        "batch_size": 64,
        "gamma": 0.995,
        "train_freq": 4,
        "target_update_interval": 1_000,
        "exploration_fraction": 0.4,
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
    if eval_callback is not None:
        save_eval_metrics(
            trial_dir,
            train_episodes=eval_callback.eval_train_episodes,
            train_timesteps=eval_callback.eval_timesteps,
            returns=eval_callback.eval_returns,
            steps=eval_callback.eval_steps,
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
    if wandb_run is not None:
        wandb_run.finish()


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
        max_steps=max_steps,
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
        default=0.3,
        help="Reward for picking up the key (--env-kind keydoor only)",
    )
    parser.add_argument(
        "--reward-attribution",
        choices=("arrival", "origin"),
        default="arrival",
        help="Store delayed key reward on its arrival or pickup transition",
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=750_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="delayed-reward")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-log-every", type=int, default=1000)
    parser.add_argument(
        "--eval-every-episodes",
        type=int,
        default=20,
        help=(
            "Run a deterministic (exploration-off) evaluation episode every "
            "N completed training episodes. Pass 0 to disable."
        ),
    )
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
            reward_attribution=args.reward_attribution,
            wandb_enabled=args.wandb, wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity, wandb_group=run_dir.name,
            wandb_log_every=args.wandb_log_every,
            eval_every_episodes=args.eval_every_episodes or None,
        )


if __name__ == "__main__":
    main()
