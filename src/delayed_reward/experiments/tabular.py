"""Run one tabular Q-learning trial end to end and save its artifacts."""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path

import numpy as np
from PIL import Image

from delayed_reward.agents.tabular.observers import LoggingObserver
from delayed_reward.agents.tabular.policies import EpsilonGreedyPolicy, GreedyPolicy
from delayed_reward.agents.tabular.q_learning import TabularQLearningAgent
from delayed_reward.agents.tabular.trainer import Trainer
from delayed_reward.environments.factory import create_empty_env
from delayed_reward.environments.registry import DEFAULT_ENV_ID
from delayed_reward.protocols import QValueStore
from delayed_reward.results.saver import create_trial_dir, save_metrics
from delayed_reward.tracking import start_wandb_run


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
