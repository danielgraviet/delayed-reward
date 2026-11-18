"""Environment ids used by the experiments, registered with Gymnasium on import.

The ``RectangularDoorKeyEnv`` ids use fixed (canonical) layouts, identical for
every seed. ``KEYDOOR_ENV_ID`` is MiniGrid's built-in DoorKeyEnv, which can't
take distractors; ``KEYDOOR_DISTRACTOR_ENV_ID`` reproduces its 8x8 room through
``RectangularDoorKeyEnv`` so distractor robustness is measured on the same room size.
"""

from __future__ import annotations

import minigrid  # noqa: F401 — registers MiniGrid-* envs
from gymnasium.envs.registration import register, registry

DEFAULT_ENV_ID = "MiniGrid-Empty-10x10-v0"
RECTANGULAR_ENV_ID = "MiniGrid-Empty-20x8-v0"
KEYDOOR_ENV_ID = "MiniGrid-DoorKey-8x8-v0"
KEYDOOR_DISTRACTOR_ENV_ID = "MiniGrid-DoorKey-8x8-Distractors-v0"
HALLWAY_KEYDOOR_ENV_ID = "MiniGrid-DoorKey-Hallway-20x6-v0"

if DEFAULT_ENV_ID not in registry:
    register(
        id=DEFAULT_ENV_ID,
        entry_point="minigrid.envs:EmptyEnv",
        kwargs={"size": 10},
    )

if RECTANGULAR_ENV_ID not in registry:
    register(
        id=RECTANGULAR_ENV_ID,
        entry_point="delayed_reward.environments.rectangular:RectangularEmptyEnv",
        kwargs={"width": 20, "height": 8},
    )

_RECTANGULAR_DOORKEY_ENTRY_POINT = "delayed_reward.environments.rectangular:RectangularDoorKeyEnv"

if HALLWAY_KEYDOOR_ENV_ID not in registry:
    register(
        id=HALLWAY_KEYDOOR_ENV_ID,
        entry_point=_RECTANGULAR_DOORKEY_ENTRY_POINT,
        kwargs={
            "width": 20,
            "height": 6,
            "agent_start_pos": (1, 3),
            "key_pos": (7, 1),
            "door_pos": (15, 2),
        },
    )

if KEYDOOR_DISTRACTOR_ENV_ID not in registry:
    register(
        id=KEYDOOR_DISTRACTOR_ENV_ID,
        entry_point=_RECTANGULAR_DOORKEY_ENTRY_POINT,
        kwargs={
            "width": 8,
            "height": 8,
            "agent_start_pos": (1, 4),
            "key_pos": (2, 2),
            "door_pos": (5, 4),
        },
    )


def fixed_layout_source(env_id: str) -> str:
    """Where a keydoor env's un-randomized layout comes from: ``"canonical"`` or ``"seed"``.

    Recorded in each trial's config so resumes never mix the two kinds of runs.
    """
    return "canonical" if supports_layout_randomization(env_id) else "seed"


def supports_layout_randomization(env_id: str) -> bool:
    """Whether ``env_id`` is backed by ``RectangularDoorKeyEnv`` (fixed layout + toggles).

    MiniGrid's built-in ``KEYDOOR_ENV_ID`` instead draws its whole layout from
    the seed and can't keep elements fixed or randomize them individually.
    """
    return env_id in registry and registry[env_id].entry_point == _RECTANGULAR_DOORKEY_ENTRY_POINT
