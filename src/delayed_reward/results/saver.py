"""Persist training metrics and policy rollout visualizations."""

from __future__ import annotations

import csv
import json
from collections.abc import Hashable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from delayed_reward.environments.factory import DEFAULT_ENV_ID, create_empty_env
from delayed_reward.policies.epsilon_greedy import GreedyPolicy
from delayed_reward.protocols import QValueStore


def create_run_dir(base: Path | str = "results") -> Path:
    """Create a timestamped results subdirectory and return its path."""
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def create_delay_dir(run_dir: Path, delay: int) -> Path:
    """Create ``run_dir/delay_<n>`` for one delay setting's artifacts."""
    delay_dir = run_dir / f"delay_{delay}"
    delay_dir.mkdir(parents=True, exist_ok=False)
    return delay_dir


def save_metrics(
    run_dir: Path,
    *,
    rewards: Sequence[float],
    steps: Sequence[int],
    config: Mapping[str, object],
    window: int = 100,
) -> None:
    """Write per-episode CSV metrics and a JSON summary into ``run_dir``."""
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["episode", "return", "steps"])
        for episode, (reward, n_steps) in enumerate(zip(rewards, steps, strict=True)):
            writer.writerow([episode, reward, n_steps])

    last = list(rewards[-window:]) if rewards else []
    summary = {
        "config": dict(config),
        "n_episodes": len(rewards),
        "final_window": window,
        "final_mean_return": (sum(last) / len(last)) if last else None,
        "final_mean_steps": (
            sum(steps[-window:]) / min(window, len(steps)) if steps else None
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def record_greedy_policy_gif(
    q_store: QValueStore,
    *,
    seed: int,
    output_path: Path,
    delay: int = 0,
    env_id: str = DEFAULT_ENV_ID,
    max_steps: int = 200,
    frame_duration_ms: int = 200,
) -> Path:
    """Roll out the greedy policy once and save frames as an animated GIF."""
    env = create_empty_env(
        env_id=env_id, seed=seed, delay=delay, render_mode="rgb_array"
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
