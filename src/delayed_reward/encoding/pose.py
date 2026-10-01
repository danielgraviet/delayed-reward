"""Pose-based state encoding for fully observable fixed grids."""

from __future__ import annotations

from collections.abc import Hashable, Mapping


class PoseEncoder:
    """Encode agent pose as (x, y, direction); ignore step count and grid pixels."""

    def encode(self, observation: Mapping[str, object], env: object) -> Hashable:
        del observation  # pose comes from the unwrapped env under full observability
        unwrapped = env.unwrapped  # type: ignore[attr-defined]
        pos = unwrapped.agent_pos
        x, y = int(pos[0]), int(pos[1])
        direction = int(unwrapped.agent_dir)
        return (x, y, direction)
