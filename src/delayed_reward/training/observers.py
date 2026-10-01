"""Episode observers for training metrics (Observer pattern)."""

from __future__ import annotations

from collections import deque


class LoggingObserver:
    """Track episode returns/steps and periodically print window averages."""

    def __init__(self, log_every: int = 100) -> None:
        if log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {log_every}")
        self._log_every = log_every
        self._rewards: list[float] = []
        self._steps: list[int] = []
        self._window: deque[tuple[float, int]] = deque(maxlen=log_every)

    def on_episode_end(self, episode: int, total_reward: float, steps: int) -> None:
        self._rewards.append(total_reward)
        self._steps.append(steps)
        self._window.append((total_reward, steps))

        if (episode + 1) % self._log_every == 0:
            mean_reward = sum(r for r, _ in self._window) / len(self._window)
            mean_steps = sum(s for _, s in self._window) / len(self._window)
            print(
                f"episode={episode + 1:5d}  "
                f"mean_return={mean_reward:7.3f}  "
                f"mean_steps={mean_steps:6.1f}"
            )

    @property
    def rewards(self) -> list[float]:
        return self._rewards

    @property
    def steps(self) -> list[int]:
        return self._steps
