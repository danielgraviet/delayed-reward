"""Run one tabular Q-learning trial end to end and save its artifacts."""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path

import numpy as np
from PIL import Image

from delayed_reward.agents.tabular.observers import GreedyEvalObserver, LoggingObserver
from delayed_reward.agents.tabular.policies import EpsilonGreedyPolicy, GreedyPolicy
from delayed_reward.agents.tabular.q_learning import TabularQLearningAgent
from delayed_reward.agents.tabular.trainer import Trainer
from delayed_reward.config import TabularTrialConfig, WandbSettings
from delayed_reward.environments.factory import create_empty_env
from delayed_reward.environments.registry import DEFAULT_ENV_ID
from delayed_reward.protocols import QValueStore
from delayed_reward.results.saver import create_trial_dir, save_eval_metrics, save_metrics
from delayed_reward.tracking import start_wandb_run

GREEDY_EVAL_CAP = 10_000


def train_one_trial(config: TabularTrialConfig, delay_dir: Path, *, wandb: WandbSettings | None = None) -> Path:
    """Train one (delay, trial) replicate and write artifacts under ``delay_dir``."""
    delay, trial = config.delay, config.trial
    print(f"\n=== delay={delay} trial={trial} (env_seed={config.seed}, policy_seed={config.policy_seed}) ===")

    env = create_empty_env(env_id=config.env_id, seed=config.seed, delay=delay, max_steps=config.max_steps)
    policy = EpsilonGreedyPolicy(epsilon=config.epsilon, rng=np.random.default_rng(config.policy_seed))
    agent = TabularQLearningAgent(n_actions=env.n_actions, policy=policy, alpha=config.alpha, gamma=config.gamma)
    settings = config.to_dict()
    wandb_run = None
    if wandb is not None:
        wandb_run = start_wandb_run(
            enabled=True,
            project=wandb.project,
            entity=wandb.entity,
            group=wandb.group,
            name=f"delay-{delay}-trial-{trial}",
            config=settings,
        )
    observer = LoggingObserver(log_every=config.log_every, wandb_run=wandb_run)
    eval_env = create_empty_env(env_id=config.env_id, seed=config.seed, delay=delay, max_steps=config.max_steps)
    evals = GreedyEvalObserver(
        eval_env=eval_env, q_store=agent.q_store, every_episodes=config.eval_every_episodes, max_steps=GREEDY_EVAL_CAP
    )
    trainer = Trainer(env, agent, observers=[observer, evals])

    try:
        trainer.train(config.episodes)
    finally:
        env.close()
        eval_env.close()

    trial_dir = create_trial_dir(delay_dir, trial)
    save_metrics(trial_dir, rewards=observer.rewards, steps=observer.steps, config=settings, window=config.log_every)
    save_eval_metrics(
        trial_dir,
        train_episodes=evals.train_episodes,
        train_timesteps=evals.train_timesteps,
        returns=evals.returns,
        steps=evals.steps,
        goal_rates=evals.goal_rates,
    )
    gif_path = record_greedy_policy_gif(
        agent.q_store,
        seed=config.seed,
        delay=delay,
        env_id=config.env_id,
        output_path=trial_dir / "policy.gif",
        max_steps=config.max_steps if config.max_steps is not None else 200,
        episode_max_steps=config.max_steps,
    )

    if observer.rewards:
        last = observer.rewards[-config.log_every :]
        print(f"delay={delay} trial={trial} final window mean return ({len(last)} eps): {sum(last) / len(last):.3f}")
    print(f"Saved to {trial_dir.resolve()}")
    print(f"Policy GIF: {gif_path.resolve()}")
    if wandb_run is not None:
        wandb_run.finish()
    return trial_dir


def record_greedy_policy_gif(
    q_store: QValueStore,
    *,
    seed: int,
    output_path: Path,
    delay: int = 0,
    env_id: str = DEFAULT_ENV_ID,
    max_steps: int = 200,
    episode_max_steps: int | None = None,
    frame_duration_ms: int = 200,
) -> Path:
    """Roll out the greedy policy once and save frames as an animated GIF."""
    env = create_empty_env(
        env_id=env_id,
        seed=seed,
        delay=delay,
        render_mode="rgb_array",
        max_steps=episode_max_steps,
    )
    policy = GreedyPolicy()
    frames: list[np.ndarray] = []

    try:
        state: Hashable = env.reset()
        frames.append(env.render())

        for _ in range(max_steps):
            action = policy.select(state, q_store.get(state))
            transition = env.step(action)
            frames.append(env.render())
            state = transition.state
            if transition.terminated or transition.truncated:
                break
    finally:
        env.close()

    if not frames:
        raise RuntimeError("No frames captured for policy GIF")

    images = [Image.fromarray(frame) for frame in frames]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    return output_path
