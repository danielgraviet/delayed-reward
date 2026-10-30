"""Visual-only distractor objects for MiniGrid key-door environments.

Distractors are colored ``Floor`` tiles: visible and walkable but neither
pickup-able nor interactive, so a trained agent walking over one incurs no
consequence. They exist to test whether a learned policy is actually keying
off the task-relevant key/door color, or just reacting to "any colored
object nearby." ``Floor`` is used instead of ``Ball``/extra ``Key`` objects
because the keydoor action space (left/right/forward/pickup/toggle) has no
"drop" action -- picking up a pickup-able distractor would permanently
occupy the agent's hand and softlock the episode.
"""

from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Floor
from minigrid.minigrid_env import MiniGridEnv


def place_distractors(
    env: MiniGridEnv,
    *,
    num_distractors: int,
    top: tuple[int, int],
    size: tuple[int, int],
    exclude_colors: tuple[str, ...] = (),
) -> None:
    """Scatter ``num_distractors`` random-colored Floor tiles in a region.

    Each tile is placed with ``MiniGridEnv.place_obj``, which already
    guarantees no overlap with the agent or any object placed earlier (e.g.
    the key, door, and goal), so callers should place distractors last.
    """
    if num_distractors < 0:
        raise ValueError(f"num_distractors must be >= 0, got {num_distractors}")
    colors = [color for color in COLOR_NAMES if color not in exclude_colors]
    if not colors:
        raise ValueError("no distractor colors left after exclude_colors")
    for _ in range(num_distractors):
        color = colors[int(env.np_random.integers(len(colors)))]
        env.place_obj(Floor(color), top=top, size=size)
