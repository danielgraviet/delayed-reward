"""SB3 callbacks: per-episode metrics with reward relabeling, and periodic greedy evaluation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from delayed_reward.config import EvalSpec, RewardAttribution
from delayed_reward.tracking import log_metrics


@dataclass(frozen=True, kw_only=True)
class EvalCheckpoint:
    """One greedy-eval checkpoint and a CPU copy of the policy weights at that point."""

    train_episode: int
    timestep: int
    eval_return: float
    eval_steps: float
    state_dict: dict[str, th.Tensor]


class EpisodeMetricsCallback(BaseCallback):
    """Collect episode metrics and optionally relabel delayed subgoal rewards.

    SB3's ``train/`` and ``rollout/`` stats are only gathered when logging to W&B.
    """

    def __init__(
        self,
        *,
        reward_attribution: RewardAttribution = RewardAttribution.ARRIVAL,
        wandb_run=None,
        log_every: int = 1000,
    ) -> None:
        super().__init__()
        self.reward_attribution = RewardAttribution(reward_attribution)
        self.rewards: list[float] = []
        self.steps: list[int] = []
        self._transition_indices: dict[int, int] = {}
        self.wandb_run = wandb_run
        self.log_every = log_every

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        metrics: dict[str, float] = {}
        buffer = cast(OffPolicyAlgorithm, self.model).replay_buffer
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
                if self.reward_attribution is RewardAttribution.ORIGIN and "subgoal_reward_origin_step" in info:
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
                metrics.update(
                    {
                        "episode/return": float(info["episode"]["r"]),
                        "episode/steps": float(info["episode"]["l"]),
                        "episode/count": float(len(self.rewards)),
                    }
                )
        if self.wandb_run is not None and self.num_timesteps % self.log_every == 0:
            values = self.model.logger.name_to_value
            metrics.update(
                {
                    key.replace("/", "_"): float(value)
                    for key, value in values.items()
                    if key.startswith(("train/", "rollout/")) and isinstance(value, int | float | np.number)
                }
            )
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
    rolling out ``spec.episodes`` episodes with ``deterministic=True`` and
    no exploration, every ``spec.every_episodes`` completed training
    episodes.

    By default the environment is reseeded to the same fixed layout every
    episode (see ``ReseedWrapper`` in ``environments/factory.py``), so a greedy
    policy's rollout is itself deterministic -- there's no layout or
    start-state randomness left once exploration is off. That's why
    ``spec.episodes`` defaults to 1: running it more times would just repeat
    the identical trajectory. With layout randomization on (see
    ``LayoutRandomization``) every eval episode draws a new start/key/door, so
    raise ``spec.episodes`` to average over them.

    Online TD updates keep nudging the Q-network after it has already found
    a working greedy policy, and the near-tied Q-values this task's sparse
    reward produces (see ``StepPenaltyWrapper``) mean a late update can flip
    the final argmax into a degenerate loop even after many evals showed a
    clean solve. ``spec.checkpoint_window`` keeps a deep-copied policy snapshot for each
    of the last ``checkpoint_window`` eval checkpoints so ``best_in_window`` can
    hand back whichever of those recent snapshots actually solved the task
    fastest, instead of trusting whatever the raw final-timestep weights do.

    With ``spec.optimal_steps`` set, a checkpoint counts as an *optimal solve* when
    every eval episode reaches the goal (terminates, not truncated) in at most
    ``optimal_steps`` steps. The callback records the first optimal checkpoint
    and the first checkpoint closing a run of ``sustain_evals`` consecutive
    optimal ones (the sustained solve), and with ``stop_on_sustained`` it ends
    training there. Consecutive checkpoints are separated by training, so on a
    fixed layout they are not repeats of one deterministic rollout.
    """

    def __init__(self, *, eval_env: gym.Env, spec: EvalSpec, max_steps: int) -> None:
        super().__init__()
        if spec.every_episodes is None:
            raise ValueError("GreedyEvalCallback needs spec.every_episodes")
        self.eval_env = eval_env
        self.eval_every_episodes = spec.every_episodes
        self.max_steps = max_steps
        self.n_eval_episodes = spec.episodes
        self.optimal_steps = spec.optimal_steps
        self.sustain_evals = spec.sustain_evals
        self.stop_on_sustained = spec.stop_on_sustained
        self._train_episode_count = 0
        self.eval_train_episodes: list[int] = []
        self.eval_timesteps: list[int] = []
        self.eval_returns: list[float] = []
        self.eval_steps: list[float] = []
        self.eval_optimal: list[bool] = []
        self._optimal_streak = 0
        self.first_optimal: dict[str, int] | None = None
        self.sustained_optimal: dict[str, int] | None = None
        self._window: deque[EvalCheckpoint] = deque(maxlen=spec.checkpoint_window)

    def _run_eval(self) -> tuple[float, float, bool]:
        """Return (mean return, mean length, whether every episode was an optimal solve)."""
        returns: list[float] = []
        lengths: list[int] = []
        all_optimal = self.optimal_steps is not None
        for _ in range(self.n_eval_episodes):
            observation, _info = self.eval_env.reset()
            total_reward = 0.0
            length = 0
            reached_goal = False
            for _ in range(self.max_steps):
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _info = self.eval_env.step(int(action))
                total_reward += float(reward)
                length += 1
                if terminated or truncated:
                    reached_goal = bool(terminated)
                    break
            returns.append(total_reward)
            lengths.append(length)
            if self.optimal_steps is not None:
                all_optimal = all_optimal and reached_goal and length <= self.optimal_steps
        return float(np.mean(returns)), float(np.mean(lengths)), all_optimal

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done and "episode" in info:
                self._train_episode_count += 1
                if self._train_episode_count % self.eval_every_episodes == 0:
                    eval_return, eval_length, optimal = self._run_eval()
                    self.eval_train_episodes.append(self._train_episode_count)
                    self.eval_timesteps.append(int(self.num_timesteps))
                    self.eval_returns.append(eval_return)
                    self.eval_steps.append(eval_length)
                    self.eval_optimal.append(optimal)
                    self._record_optimal(optimal)
                    self._window.append(
                        EvalCheckpoint(
                            train_episode=self._train_episode_count,
                            timestep=int(self.num_timesteps),
                            eval_return=eval_return,
                            eval_steps=eval_length,
                            state_dict={
                                key: value.detach().clone().cpu()
                                for key, value in self.model.policy.state_dict().items()
                            },
                        )
                    )
        return not (self.stop_on_sustained and self.sustained_optimal is not None)

    def _record_optimal(self, optimal: bool) -> None:
        if self.optimal_steps is None:
            return
        self._optimal_streak = self._optimal_streak + 1 if optimal else 0
        point = {"train_episode": self._train_episode_count, "timestep": int(self.num_timesteps)}
        if optimal and self.first_optimal is None:
            self.first_optimal = point
        if self._optimal_streak >= self.sustain_evals and self.sustained_optimal is None:
            self.sustained_optimal = point

    def time_to_optimal(self, total_timesteps: int) -> dict[str, object] | None:
        """Summary of first/sustained optimal checkpoints; ``censored`` if never sustained."""
        if self.optimal_steps is None:
            return None
        return {
            "optimal_steps": self.optimal_steps,
            "sustain_evals": self.sustain_evals,
            "first_optimal_timestep": (self.first_optimal or {}).get("timestep"),
            "first_optimal_train_episode": (self.first_optimal or {}).get("train_episode"),
            "sustained_optimal_timestep": (self.sustained_optimal or {}).get("timestep"),
            "sustained_optimal_train_episode": (self.sustained_optimal or {}).get("train_episode"),
            "censored": self.sustained_optimal is None,
            "censor_timestep": total_timesteps if self.sustained_optimal is None else None,
            "trained_timesteps": int(self.num_timesteps),
            "stopped_early": self.stop_on_sustained and self.sustained_optimal is not None,
        }

    def best_in_window(self) -> EvalCheckpoint | None:
        """Return the window checkpoint with the highest return (fewest steps breaks ties)."""
        if not self._window:
            return None
        return min(self._window, key=lambda checkpoint: (-checkpoint.eval_return, checkpoint.eval_steps))
