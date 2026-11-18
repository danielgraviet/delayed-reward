"""Run one DQN trial end to end: build env and model, train, evaluate, save artifacts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from delayed_reward.agents.dqn.callbacks import EpisodeMetricsCallback, GreedyEvalCallback
from delayed_reward.agents.dqn.model import build_model
from delayed_reward.config import DQNTrialConfig, EnvSpec, WandbSettings
from delayed_reward.environments.factory import create_empty_dqn_env, create_keydoor_dqn_env
from delayed_reward.results.saver import save_eval_metrics, save_metrics
from delayed_reward.tracking import start_wandb_run


def build_dqn_env(spec: EnvSpec, *, seed: int, render_mode: str | None = None) -> gym.Env:
    """Build the flat-observation DQN env that ``spec`` describes."""
    if spec.is_keydoor:
        return create_keydoor_dqn_env(
            env_id=spec.env_id,
            seed=seed,
            delay=spec.delay,
            subgoal_reward=spec.subgoal_reward,
            render_mode=render_mode,
            max_steps=spec.max_steps,
            step_penalty=spec.step_penalty,
            randomize=spec.randomize,
        )
    return create_empty_dqn_env(
        env_id=spec.env_id,
        seed=seed,
        delay=spec.delay,
        render_mode=render_mode,
        max_steps=spec.max_steps,
    )


def train_trial(
    config: DQNTrialConfig, trial_dir: Path, *, wandb: WandbSettings | None = None
) -> None:
    """Train one DQN trial and write its artifacts into ``trial_dir``.

    With ``config.eval.stop_on_sustained`` the trial ends at its sustained
    optimal solve (see ``GreedyEvalCallback``).
    """
    env = gym.wrappers.RecordEpisodeStatistics(build_dqn_env(config.env, seed=config.trial_seed))
    settings = config.to_dict()
    wandb_run = None
    if wandb is not None:
        wandb_run = start_wandb_run(
            enabled=True, project=wandb.project, entity=wandb.entity, group=wandb.group,
            name=f"delay-{config.env.delay}-trial-{config.trial}", config=settings,
        )
    callback = EpisodeMetricsCallback(
        reward_attribution=config.reward_attribution,
        wandb_run=wandb_run,
        log_every=wandb.log_every if wandb is not None else 1000,
    )
    callbacks: list[BaseCallback] = [callback]
    eval_callback: GreedyEvalCallback | None = None
    if config.eval.every_episodes is not None:
        eval_callback = GreedyEvalCallback(
            eval_env=build_dqn_env(config.env, seed=config.trial_seed),
            spec=config.eval,
            max_steps=config.env.max_steps or 2000,
        )
        callbacks.append(eval_callback)
    model = build_model(config, env)
    try:
        model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
    finally:
        env.close()
        if eval_callback is not None:
            eval_callback.eval_env.close()

    best = eval_callback.best_in_window() if eval_callback is not None else None
    if best is not None:
        model.policy.load_state_dict(best["state_dict"])
        checkpoint_selection: dict[str, object] = {
            "strategy": "best_of_last_n_eval_checkpoints",
            "window": config.eval.checkpoint_window,
            "selected_train_episode": best["train_episode"],
            "selected_timestep": best["timestep"],
            "selected_eval_return": best["return"],
            "selected_eval_steps": best["steps"],
        }
    else:
        checkpoint_selection = {"strategy": "final_weights"}

    time_to_optimal = (
        eval_callback.time_to_optimal(config.total_timesteps) if eval_callback is not None else None
    )
    save_metrics(
        trial_dir,
        rewards=callback.rewards,
        steps=callback.steps,
        config={**settings, "device": str(model.device), "checkpoint_selection": checkpoint_selection},
        window=100,
        extra={"time_to_optimal": time_to_optimal} if time_to_optimal is not None else None,
    )
    if eval_callback is not None:
        save_eval_metrics(
            trial_dir,
            train_episodes=eval_callback.eval_train_episodes,
            train_timesteps=eval_callback.eval_timesteps,
            returns=eval_callback.eval_returns,
            steps=eval_callback.eval_steps,
            optimal=eval_callback.eval_optimal if config.eval.optimal_steps is not None else None,
        )
    model.save(trial_dir / "model")
    _save_policy_gif(model, config.env, seed=config.trial_seed, path=trial_dir / "policy.gif")
    recent = callback.rewards[-100:]
    print(
        f"trial={config.trial} episodes={len(callback.rewards)} "
        f"last_100_mean_return={np.mean(recent) if recent else float('nan'):.3f} "
        f"device={model.device} "
        + (
            f"sustained_optimal_timestep={time_to_optimal['sustained_optimal_timestep']} "
            if time_to_optimal is not None
            else ""
        )
        + f"path={trial_dir}",
        flush=True,
    )
    if wandb_run is not None:
        if time_to_optimal is not None:
            wandb_run.summary.update(
                {f"time_to_optimal/{k}": v for k, v in time_to_optimal.items()}
            )
        wandb_run.finish()


def _save_policy_gif(model: DQN, spec: EnvSpec, *, seed: int, path: Path) -> None:
    max_steps = spec.max_steps or 200
    env = build_dqn_env(replace(spec, max_steps=max_steps), seed=seed, render_mode="rgb_array")
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
