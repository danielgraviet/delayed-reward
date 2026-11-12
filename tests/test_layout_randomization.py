"""Fixed key-door layouts and the per-episode randomization toggles."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pytest
from minigrid.core.world_object import Door, Floor, Key

from delayed_reward.environments.factory import (
    HALLWAY_KEYDOOR_ENV_ID,
    KEYDOOR_DISTRACTOR_ENV_ID,
    KEYDOOR_ENV_ID,
    create_keydoor_dqn_env,
)
from delayed_reward.environments.layout import LayoutRandomization
from delayed_reward.environments.rectangular import RectangularDoorKeyEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import parallel_dqn_sweep  # noqa: E402

EPISODES = 300
HALLWAY_FIXED = {"start": ((1, 3), 0), "key": (7, 1), "door": (15, 2)}
ALL_TOGGLES = [
    LayoutRandomization(**dict(zip(LayoutRandomization.element_names(), flags)))
    for flags in itertools.product([False, True], repeat=3)
]


def _toggle_id(randomize: LayoutRandomization) -> str:
    return "+".join(randomize.names()) or "fixed"


def _layout(env) -> dict[str, object]:
    u = env.unwrapped
    found: dict[str, object] = {"distractors": []}
    for index, obj in enumerate(u.grid.grid):
        pos = (index % u.width, index // u.width)
        if isinstance(obj, Key):
            found["key"] = pos
        elif isinstance(obj, Door):
            found["door"] = pos
        elif isinstance(obj, Floor):
            found["distractors"].append(pos)
    found["start"] = (tuple(int(v) for v in u.agent_pos), int(u.agent_dir))
    return found


def _episodes(randomize=LayoutRandomization(), *, seed=0, env_id=HALLWAY_KEYDOOR_ENV_ID, episodes=EPISODES, **kw):
    env = create_keydoor_dqn_env(env_id=env_id, seed=seed, randomize=randomize, **kw)
    layouts = []
    for _ in range(episodes):
        env.reset()
        layouts.append(_layout(env))
    env.close()
    return layouts


def test_base_case_is_identical_for_every_episode_and_seed():
    layouts = [layout for seed in range(5) for layout in _episodes(seed=seed, episodes=20)]
    for layout in layouts:
        assert {k: layout[k] for k in HALLWAY_FIXED} == HALLWAY_FIXED


@pytest.mark.parametrize("randomize", ALL_TOGGLES, ids=_toggle_id)
def test_only_toggled_elements_move(randomize):
    layouts = _episodes(randomize)
    for element, fixed_value in HALLWAY_FIXED.items():
        values = {repr(layout[element]) for layout in layouts}
        if getattr(randomize, element):
            assert len(values) > 1, f"{element} should vary"
        else:
            assert values == {repr(fixed_value)}, f"{element} should stay fixed"


@pytest.mark.parametrize("randomize", ALL_TOGGLES, ids=_toggle_id)
@pytest.mark.parametrize("env_id", [HALLWAY_KEYDOOR_ENV_ID, KEYDOOR_DISTRACTOR_ENV_ID])
def test_every_layout_is_legal(randomize, env_id):
    for layout in _episodes(randomize, env_id=env_id, num_distractors=3):
        wall_x = layout["door"][0]
        start, start_dir = layout["start"]
        assert 1 <= layout["key"][0] < wall_x, "key must never be behind the door"
        assert 1 <= start[0] < wall_x, "agent must start on the key side of the wall"
        assert start != layout["key"]
        assert start not in layout["distractors"]
        assert start_dir in range(4)


def test_random_key_covers_every_legal_cell():
    layouts = _episodes(LayoutRandomization(key=True), episodes=2000)
    wall_x = HALLWAY_FIXED["door"][0]
    legal = {(x, y) for x in range(1, wall_x) for y in range(1, 5)} - {HALLWAY_FIXED["start"][0]}
    assert {layout["key"] for layout in layouts} == legal


def test_random_door_covers_every_row_and_stays_right_of_fixed_key():
    layouts = _episodes(LayoutRandomization(door=True), episodes=2000)
    assert {layout["door"][1] for layout in layouts} == {1, 2, 3, 4}
    assert {layout["door"][0] for layout in layouts} == set(range(HALLWAY_FIXED["key"][0] + 1, 18))


def test_same_seed_gives_same_episode_sequence():
    randomize = LayoutRandomization(start=True, key=True, door=True)
    assert _episodes(randomize, seed=3, episodes=30) == _episodes(randomize, seed=3, episodes=30)
    assert _episodes(randomize, seed=3, episodes=30) != _episodes(randomize, seed=4, episodes=30)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"key_pos": (16, 1)},  # behind the door
        {"agent_start_pos": (16, 1)},
        {"key_pos": (1, 3)},  # on the agent's start
        {"door_pos": (18, 2)},  # wall on the goal column
        {"door_pos": (15, 5)},  # door in the outer wall
    ],
)
def test_illegal_fixed_layouts_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RectangularDoorKeyEnv(width=20, height=6, **kwargs)


def test_builtin_doorkey_rejects_randomization():
    with pytest.raises(ValueError, match="does not support layout randomization"):
        create_keydoor_dqn_env(env_id=KEYDOOR_ENV_ID, randomize=LayoutRandomization(start=True))


def test_from_names_round_trips_and_rejects_unknown():
    assert LayoutRandomization.from_names(["door", "start"]).names() == ["start", "door"]
    assert not LayoutRandomization.from_names([]).any
    with pytest.raises(ValueError, match="unknown layout element"):
        LayoutRandomization.from_names(["goal"])


# -- resume checks -----------------------------------------------------------

def _sweep_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return parallel_dqn_sweep.build_parser().parse_args(
        ["--trials", "1", "--run-dir", str(tmp_path), *extra]
    )


def _write_legacy_summary(tmp_path: Path, args: argparse.Namespace, env_id: str) -> None:
    """A summary as written before layout randomization existed."""
    config = parallel_dqn_sweep._expected_config(args, env_id, 0, 0)
    config.pop("layout_randomization")
    config.pop("fixed_layout")
    trial_dir = tmp_path / "delay_0" / "trial_0"
    trial_dir.mkdir(parents=True)
    (trial_dir / "summary.json").write_text(json.dumps({"config": config}))


def _plan(args, env_id, tmp_path):
    return parallel_dqn_sweep.plan_tasks(args, env_id, tmp_path, [0], [0])


@pytest.mark.parametrize(
    "extra, env_id",
    [([], "MiniGrid-Empty-10x10-v0"), (["--env-kind", "keydoor"], KEYDOOR_ENV_ID)],
    ids=["empty", "builtin-keydoor"],
)
def test_resume_accepts_legacy_runs_whose_layouts_are_unchanged(tmp_path, extra, env_id):
    args = _sweep_args(tmp_path, *extra)
    _write_legacy_summary(tmp_path, args, env_id)
    assert _plan(args, env_id, tmp_path) == ([], 1)


def test_resume_refuses_legacy_hallway_runs_with_seeded_layouts(tmp_path):
    args = _sweep_args(tmp_path, "--env-kind", "keydoor", "--env-id", HALLWAY_KEYDOOR_ENV_ID)
    _write_legacy_summary(tmp_path, args, HALLWAY_KEYDOOR_ENV_ID)
    with pytest.raises(SystemExit, match="fixed_layout"):
        _plan(args, HALLWAY_KEYDOOR_ENV_ID, tmp_path)


def test_resume_refuses_changed_randomization(tmp_path):
    base = ["--env-kind", "keydoor", "--env-id", HALLWAY_KEYDOOR_ENV_ID]
    args = _sweep_args(tmp_path, *base)
    config = parallel_dqn_sweep._expected_config(args, HALLWAY_KEYDOOR_ENV_ID, 0, 0)
    trial_dir = tmp_path / "delay_0" / "trial_0"
    trial_dir.mkdir(parents=True)
    (trial_dir / "summary.json").write_text(json.dumps({"config": config}))
    assert _plan(args, HALLWAY_KEYDOOR_ENV_ID, tmp_path) == ([], 1)
    randomized = _sweep_args(tmp_path, *base, "--randomize", "start")
    with pytest.raises(SystemExit, match="layout_randomization"):
        _plan(randomized, HALLWAY_KEYDOOR_ENV_ID, tmp_path)


def test_sweep_rejects_randomization_on_unsupported_env(tmp_path):
    args = _sweep_args(tmp_path, "--env-kind", "keydoor", "--randomize", "start")
    with pytest.raises(SystemExit, match="does not support layout randomization"):
        parallel_dqn_sweep.resolve_sweep(args)
