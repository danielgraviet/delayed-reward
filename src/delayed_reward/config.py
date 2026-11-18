"""Typed trial configuration: the single definition of what a trial runs.

Every consumer reads these objects instead of re-listing settings: the CLIs
build them from arguments, the experiments build env and learner from them,
and ``to_dict`` is the one serialization used for W&B, ``summary.json`` and
resume matching. Constructing an invalid config raises ``ValueError``, so bad
settings fail before any training starts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol, Self

from delayed_reward.environments.layout import NO_RANDOMIZATION, LayoutRandomization
from delayed_reward.environments.registry import (
    DEFAULT_ENV_ID,
    HALLWAY_KEYDOOR_ENV_ID,
    KEYDOOR_DISTRACTOR_ENV_ID,
    KEYDOOR_ENV_ID,
    fixed_layout_source,
    supports_layout_randomization,
)


class EnvKind(StrEnum):
    EMPTY = "empty"
    KEYDOOR = "keydoor"

    @property
    def default_env_id(self) -> str:
        return KEYDOOR_ENV_ID if self is EnvKind.KEYDOOR else DEFAULT_ENV_ID


class RewardAttribution(StrEnum):
    """Which replay transition stores a delayed key reward: its arrival or its pickup."""

    ARRIVAL = "arrival"
    ORIGIN = "origin"


class LRSchedule(StrEnum):
    CONSTANT = "constant"
    LINEAR = "linear"


def _coerce(instance: object, name: str, enum: type[StrEnum]) -> None:
    object.__setattr__(instance, name, enum(getattr(instance, name)))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _in_range(name: str, value: float, low: float, high: float) -> None:
    """``low <= value <= high``; NaN fails every comparison, so it is rejected too."""
    _require(low <= value <= high, f"{name} must be in [{low}, {high}], got {value}")


def _at_least(name: str, value: float, low: float) -> None:
    _require(math.isfinite(value) and value >= low, f"{name} must be a finite number >= {low}, got {value}")


@dataclass(frozen=True, kw_only=True)
class EnvSpec:
    """Which environment a trial trains on and how its reward is shaped.

    ``env_id`` defaults to the kind's standard env. ``subgoal_reward`` and
    ``step_penalty`` apply to the key-door kind only.
    """

    kind: EnvKind = EnvKind.EMPTY
    env_id: str = ""
    delay: int = 0
    max_steps: int | None = None
    subgoal_reward: float = 0.3
    step_penalty: float = 0.0
    randomize: LayoutRandomization = NO_RANDOMIZATION

    def __post_init__(self) -> None:
        _coerce(self, "kind", EnvKind)
        if not self.env_id:
            object.__setattr__(self, "env_id", self.kind.default_env_id)
        if self.delay < 0:
            raise ValueError(f"delay must be >= 0, got {self.delay}")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        _at_least("step_penalty", self.step_penalty, 0)
        _require(math.isfinite(self.subgoal_reward), f"subgoal_reward must be finite, got {self.subgoal_reward}")
        if self.step_penalty and self.kind is not EnvKind.KEYDOOR:
            raise ValueError("step_penalty requires env kind keydoor")
        if self.randomize.any and self.kind is not EnvKind.KEYDOOR:
            raise ValueError("layout randomization requires env kind keydoor")
        if self.randomize.any and not supports_layout_randomization(self.env_id):
            raise ValueError(
                f"{self.env_id} does not support layout randomization; use an env backed by "
                f"RectangularDoorKeyEnv, e.g. {KEYDOOR_DISTRACTOR_ENV_ID} or {HALLWAY_KEYDOOR_ENV_ID}"
            )

    @property
    def is_keydoor(self) -> bool:
        return self.kind is EnvKind.KEYDOOR


@dataclass(frozen=True, kw_only=True)
class DQNParams:
    """Learner hyperparameters, passed to SB3 exactly as recorded."""

    network: tuple[int, ...] = (128, 128)
    learning_rate: float = 1e-3
    lr_schedule: LRSchedule = LRSchedule.CONSTANT
    lr_final: float = 0.0
    double_dqn: bool = False
    buffer_size: int = 150_000
    learning_starts: int = 1_000
    batch_size: int = 64
    gamma: float = 0.995
    train_freq: int = 4
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.4
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    max_grad_norm: float = 10.0

    def __post_init__(self) -> None:
        _coerce(self, "lr_schedule", LRSchedule)
        _require(bool(self.network) and all(width >= 1 for width in self.network), "network widths must be >= 1")
        _require(
            math.isfinite(self.learning_rate) and self.learning_rate > 0,
            f"learning_rate must be a finite number > 0, got {self.learning_rate}",
        )
        if self.lr_schedule is LRSchedule.LINEAR:
            _in_range("lr_final", self.lr_final, 0, self.learning_rate)
        _require(math.isfinite(self.lr_final), f"lr_final must be finite, got {self.lr_final}")
        for name in ("buffer_size", "batch_size", "train_freq", "target_update_interval"):
            _at_least(name, getattr(self, name), 1)
        _at_least("learning_starts", self.learning_starts, 0)
        _in_range("gamma", self.gamma, 0, 1)
        _require(
            0 < self.exploration_fraction <= 1,
            f"exploration_fraction must be in (0, 1], got {self.exploration_fraction}",
        )
        _in_range("exploration_initial_eps", self.exploration_initial_eps, 0, 1)
        _in_range("exploration_final_eps", self.exploration_final_eps, 0, self.exploration_initial_eps)
        _require(
            math.isfinite(self.max_grad_norm) and self.max_grad_norm > 0,
            f"max_grad_norm must be a finite number > 0, got {self.max_grad_norm}",
        )


@dataclass(frozen=True, kw_only=True)
class EvalSpec:
    """Periodic greedy evaluation, checkpoint selection and the optimal-solve criterion.

    ``every_episodes=None`` turns evaluation off. With ``optimal_steps`` set, an
    eval checkpoint is an optimal solve when every episode reaches the goal in
    at most that many steps; ``sustain_evals`` consecutive ones make a sustained
    solve, and ``stop_on_sustained`` ends training there.
    """

    every_episodes: int | None = 20
    episodes: int = 1
    checkpoint_window: int = 100
    optimal_steps: int | None = None
    sustain_evals: int = 3
    stop_on_sustained: bool = False

    def __post_init__(self) -> None:
        if self.every_episodes is not None and self.every_episodes < 1:
            raise ValueError(f"eval every_episodes must be >= 1 or None, got {self.every_episodes}")
        if self.episodes < 1:
            raise ValueError(f"eval episodes must be >= 1, got {self.episodes}")
        if self.checkpoint_window < 1:
            raise ValueError(f"checkpoint_window must be >= 1, got {self.checkpoint_window}")
        if self.optimal_steps is not None and self.optimal_steps < 1:
            raise ValueError(f"optimal_steps must be >= 1, got {self.optimal_steps}")
        if self.sustain_evals < 1:
            raise ValueError(f"sustain_evals must be >= 1, got {self.sustain_evals}")
        if self.optimal_steps is not None and self.every_episodes is None:
            raise ValueError("optimal_steps needs periodic evals (every_episodes)")
        if self.stop_on_sustained and self.optimal_steps is None:
            raise ValueError("stop_on_sustained requires optimal_steps")


@dataclass(frozen=True, kw_only=True)
class DQNTrialConfig:
    """Everything that determines one DQN trial's outcome.

    ``total_timesteps`` is the cap; ``schedule_timesteps`` (default: the cap)
    is the horizon the epsilon and linear-LR schedules anneal over. The trial's
    RNG seed is ``seed + trial``.
    """

    env: EnvSpec
    total_timesteps: int
    params: DQNParams = field(default_factory=DQNParams)
    eval: EvalSpec = field(default_factory=EvalSpec)
    reward_attribution: RewardAttribution = RewardAttribution.ARRIVAL
    seed: int = 0
    trial: int = 0
    schedule_timesteps: int = 0

    def __post_init__(self) -> None:
        _coerce(self, "reward_attribution", RewardAttribution)
        if self.total_timesteps < 1:
            raise ValueError(f"total_timesteps must be >= 1, got {self.total_timesteps}")
        if self.trial < 0:
            raise ValueError(f"trial must be >= 0, got {self.trial}")
        if not self.schedule_timesteps:
            object.__setattr__(self, "schedule_timesteps", self.total_timesteps)
        if not 1 <= self.schedule_timesteps <= self.total_timesteps:
            raise ValueError("schedule_timesteps must be in [1, total_timesteps]")

    @property
    def delay(self) -> int:
        return self.env.delay

    @property
    def trial_seed(self) -> int:
        return self.seed + self.trial

    @property
    def schedule_fraction(self) -> float:
        return self.schedule_timesteps / self.total_timesteps

    def for_trial(self, *, delay: int, trial: int) -> DQNTrialConfig:
        return replace(self, env=replace(self.env, delay=delay), trial=trial)

    def to_dict(self) -> dict[str, object]:
        """JSON-ready settings, as recorded in ``summary.json`` and W&B and matched on resume."""
        env, params, ev = self.env, self.params, self.eval
        keydoor = env.is_keydoor
        linear = params.lr_schedule is LRSchedule.LINEAR
        return {
            "algorithm": "stable-baselines3-dqn",
            "env": env.env_id,
            "env_kind": env.kind.value,
            "action_masking": keydoor,
            "delay": env.delay,
            "subgoal_reward": env.subgoal_reward if keydoor else None,
            "reward_attribution": self.reward_attribution.value,
            "step_penalty": env.step_penalty if keydoor else None,
            "layout_randomization": env.randomize.names() if keydoor else None,
            "fixed_layout": fixed_layout_source(env.env_id) if keydoor else None,
            "trial": self.trial,
            "seed": self.trial_seed,
            "total_timesteps": self.total_timesteps,
            "max_steps": env.max_steps,
            "observation": "fully-observed encoded grid plus normalized agent pose",
            "network": list(params.network),
            "learning_rate": params.learning_rate,
            "lr_schedule": params.lr_schedule.value,
            "lr_final": params.lr_final if linear else None,
            "double_dqn": params.double_dqn,
            "buffer_size": params.buffer_size,
            "learning_starts": params.learning_starts,
            "batch_size": params.batch_size,
            "gamma": params.gamma,
            "train_freq": params.train_freq,
            "target_update_interval": params.target_update_interval,
            "exploration_fraction": params.exploration_fraction,
            "exploration_initial_eps": params.exploration_initial_eps,
            "exploration_final_eps": params.exploration_final_eps,
            "max_grad_norm": params.max_grad_norm,
            "eval_every_episodes": ev.every_episodes,
            "eval_episodes": ev.episodes,
            "checkpoint_window": ev.checkpoint_window,
            "schedule_timesteps": self.schedule_timesteps,
            "optimal_steps": ev.optimal_steps,
            "sustain_evals": ev.sustain_evals if ev.optimal_steps is not None else None,
            "stop_on_sustained": ev.stop_on_sustained,
        }


@dataclass(frozen=True, kw_only=True)
class WandbSettings:
    """Where a trial logs to W&B."""

    project: str = "delayed-reward"
    entity: str | None = None
    group: str | None = None
    log_every: int = 1000

    def __post_init__(self) -> None:
        if self.log_every < 1:
            raise ValueError(f"wandb log_every must be >= 1, got {self.log_every}")


@dataclass(frozen=True, kw_only=True)
class TabularTrialConfig:
    """Everything that determines one tabular Q-learning trial's outcome.

    The env layout is seeded by ``seed`` for every trial; exploration uses an
    independent ``policy_seed`` per (delay, trial).
    """

    env_id: str = DEFAULT_ENV_ID
    delay: int = 0
    max_steps: int | None = None
    episodes: int = 10_000
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    seed: int = 0
    trial: int = 0
    log_every: int = 100
    eval_every_episodes: int = 100

    def __post_init__(self) -> None:
        _at_least("eval_every_episodes", self.eval_every_episodes, 1)
        if self.delay < 0:
            raise ValueError(f"delay must be >= 0, got {self.delay}")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.episodes < 1:
            raise ValueError(f"episodes must be >= 1, got {self.episodes}")
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")
        if self.trial < 0:
            raise ValueError(f"trial must be >= 0, got {self.trial}")
        if not 0 < self.alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0 <= self.epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")

    @property
    def policy_seed(self) -> int:
        return self.seed + self.trial + 10_003 * self.delay

    def for_trial(self, *, delay: int, trial: int) -> TabularTrialConfig:
        return replace(self, delay=delay, trial=trial)

    def to_dict(self) -> dict[str, object]:
        """JSON-ready settings, as recorded in ``summary.json`` and W&B and matched on resume."""
        return {
            "algorithm": "tabular-q-learning",
            "env": self.env_id,
            "delay": self.delay,
            "max_steps": self.max_steps,
            "episodes": self.episodes,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "env_seed": self.seed,
            "policy_seed": self.policy_seed,
            "trial": self.trial,
            "log_every": self.log_every,
            "eval_every_episodes": self.eval_every_episodes,
        }


class TrialConfig(Protocol):
    """What sweep planning needs from any trial config."""

    @property
    def delay(self) -> int: ...

    @property
    def trial(self) -> int: ...

    def for_trial(self, *, delay: int, trial: int) -> Self: ...

    def to_dict(self) -> dict[str, object]: ...


def config_mismatches(saved: Mapping[str, object], config: TrialConfig) -> dict[str, tuple[object, object]]:
    """Settings where a saved ``summary.json`` config differs from ``config``, as (saved, requested).

    A key missing from ``saved`` counts as a mismatch.
    """
    missing = object()
    mismatches: dict[str, tuple[object, object]] = {}
    for key, value in config.to_dict().items():
        stored = saved.get(key, missing)
        if stored != value:
            mismatches[key] = (None if stored is missing else stored, value)
    return mismatches
