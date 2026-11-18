"""Episode observers for training metrics (Observer pattern)."""

from __future__ import annotations

from collections import deque
from typing import Any

from delayed_reward.agents.tabular.policies import GreedyPolicy
from delayed_reward.protocols import Environment, QValueStore
from delayed_reward.tracking import log_metrics


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
            print(f"episode={episode + 1:5d}  mean_return={window_reward:7.3f}  mean_steps={window_steps:6.1f}")

    @property
    def rewards(self) -> list[float]:
        return self._rewards

    @property
    def steps(self) -> list[int]:
        return self._steps


class GreedyEvalObserver:
    """Every ``every_episodes`` training episodes, roll out the greedy policy once on a separate env.

    Records the same per-checkpoint columns as the DQN eval callback, so both
    kinds of trial feed the same learning analysis.
    """

    def __init__(self, *, eval_env: Environment, q_store: QValueStore, every_episodes: int, max_steps: int) -> None:
        if every_episodes < 1:
            raise ValueError(f"every_episodes must be >= 1, got {every_episodes}")
        self._env = eval_env
        self._q_store = q_store
        self._every = every_episodes
        self._max_steps = max_steps
        self._policy = GreedyPolicy()
        self._timesteps = 0
        self.train_episodes: list[int] = []
        self.train_timesteps: list[int] = []
        self.returns: list[float] = []
        self.steps: list[float] = []
        self.goal_rates: list[float] = []

    def on_episode_end(self, episode: int, total_reward: float, steps: int) -> None:
        del total_reward
        self._timesteps += steps
        if (episode + 1) % self._every:
            return
        eval_return, eval_steps, reached_goal = self._rollout()
        self.train_episodes.append(episode + 1)
        self.train_timesteps.append(self._timesteps)
        self.returns.append(eval_return)
        self.steps.append(float(eval_steps))
        self.goal_rates.append(float(reached_goal))

    def _rollout(self) -> tuple[float, int, bool]:
        state = self._env.reset()
        total = 0.0
        for step in range(1, self._max_steps + 1):
            transition = self._env.step(self._policy.select(state, self._q_store.get(state)))
            total += transition.reward
            state = transition.state
            if transition.terminated or transition.truncated:
                return total, step, transition.terminated
        return total, self._max_steps, False
