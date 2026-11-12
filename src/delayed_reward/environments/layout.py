"""Per-episode layout randomization for the key-door environments.

By default every element of a key-door layout (agent start, key, door) sits
at a fixed position, the same every episode and for every seed.
``LayoutRandomization`` switches individual elements to being re-sampled
uniformly on every ``reset()`` while the rest stay at their fixed positions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Any

import gymnasium as gym


@dataclass(frozen=True)
class LayoutRandomization:
    """Which key-door layout elements are re-sampled every episode.

    start: agent start cell and facing direction, anywhere on the key side of the wall.
    key:   key cell, anywhere on the agent's side of the wall (never behind the door).
    door:  splitting-wall column and the door's row in it.
    """

    start: bool = False
    key: bool = False
    door: bool = False

    @classmethod
    def element_names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    @classmethod
    def from_names(cls, names: Iterable[str]) -> LayoutRandomization:
        """Build from element names, e.g. ``["start", "door"]`` (as given on the CLI)."""
        requested = set(names)
        unknown = requested.difference(cls.element_names())
        if unknown:
            raise ValueError(
                f"unknown layout element(s) {sorted(unknown)}; choose from {list(cls.element_names())}"
            )
        return cls(**{name: True for name in requested})

    def names(self) -> list[str]:
        """Enabled element names in canonical order (stable for configs and resume checks)."""
        return [name for name in self.element_names() if getattr(self, name)]

    @property
    def any(self) -> bool:
        return bool(self.names())


class SeedOnceWrapper(gym.Wrapper):
    """Seed only the first ``reset()`` so the env's RNG advances across episodes.

    The counterpart of MiniGrid's ``ReseedWrapper``, which reseeds every reset
    and therefore makes every episode identical. An explicit ``seed`` passed to
    ``reset`` (e.g. by Stable-Baselines3) still takes precedence.
    """

    def __init__(self, env: gym.Env, seed: int) -> None:
        super().__init__(env)
        self._seed: int | None = seed

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is None:
            seed, self._seed = self._seed, None
        else:
            self._seed = None
        return self.env.reset(seed=seed, options=options)
