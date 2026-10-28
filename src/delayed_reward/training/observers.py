"""Episode observers for training metrics (Observer pattern)."""

from __future__ import annotations

from collections import deque
from typing import Any

from delayed_reward.training.wandb_logging import log_metrics


class LoggingObserver:
    """Track episode returns/steps and periodically print window averages."""

    def __init__(self, log_every: int = 100, wandb_run: Any | None = None) -> None:
        if log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {log_every}")
        self._log_every = log_every
        self._rewards: list[float] = []
        self._steps: list[int] = []
        self._window: deque[tuple[float, int]] = deque(maxlen=log_every)
        self._wandb_run = wandb_run

    def on_episode_end(self, episode: int, total_reward: float, steps: int) -> None:
        self._rewards.append(total_reward)
        self._steps.append(steps)
        self._window.append((total_reward, steps))

        window_reward = sum(r for r, _ in self._window) / len(self._window)
        window_steps = sum(s for _, s in self._window) / len(self._window)
        log_metrics(
            self._wandb_run,
            {
                "episode/return": total_reward,
                "episode/steps": float(steps),
                "episode/return_mean": window_reward,
                "episode/steps_mean": window_steps,
            },
            step=episode + 1,
        )

        if (episode + 1) % self._log_every == 0:
            print(
                f"episode={episode + 1:5d}  "
                f"mean_return={window_reward:7.3f}  "
                f"mean_steps={window_steps:6.1f}"
            )

    @property
    def rewards(self) -> list[float]:
        return self._rewards

    @property
    def steps(self) -> list[int]:
        return self._steps
