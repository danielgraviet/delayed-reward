"""Fuzz the trial configs and their CLI with arbitrary inputs.

A config either rejects its inputs with ``ValueError`` (the CLI with ``SystemExit``)
or yields settings that survive a JSON round trip and match themselves on resume.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from hypothesis import given, strategies as st

from delayed_reward.cli import dqn_sweep, tabular_sweep
from delayed_reward.cli.dqn import dqn_config_from_args
from delayed_reward.cli.tabular import tabular_config_from_args
from delayed_reward.config import (
    DQNParams,
    DQNTrialConfig,
    EnvKind,
    EnvSpec,
    EvalSpec,
    LRSchedule,
    TabularTrialConfig,
    config_mismatches,
)
from delayed_reward.environments.layout import LayoutRandomization
from delayed_reward.environments.registry import (
    DEFAULT_ENV_ID,
    HALLWAY_KEYDOOR_ENV_ID,
    KEYDOOR_DISTRACTOR_ENV_ID,
    KEYDOOR_ENV_ID,
)

ANY_INT = st.integers(min_value=-5, max_value=3_000_000)
EDGE_FLOATS = st.sampled_from([math.nan, math.inf, -math.inf, 0.0, -0.0, -1.0, 1.0, 1e-12, 1e300])
ANY_FLOAT = EDGE_FLOATS | st.floats(min_value=-2, max_value=2) | st.floats(allow_nan=True, allow_infinity=True)
MAYBE_INT = st.none() | ANY_INT
ENV_IDS = st.sampled_from(["", DEFAULT_ENV_ID, KEYDOOR_ENV_ID, HALLWAY_KEYDOOR_ENV_ID, KEYDOOR_DISTRACTOR_ENV_ID])


FIELD_VALUES = {
    "env.kind": st.sampled_from(EnvKind),
    "env.env_id": ENV_IDS,
    "env.delay": ANY_INT,
    "env.max_steps": MAYBE_INT,
    "env.subgoal_reward": ANY_FLOAT,
    "env.step_penalty": ANY_FLOAT,
    "env.randomize": st.builds(LayoutRandomization, st.booleans(), st.booleans(), st.booleans()),
    "params.learning_rate": ANY_FLOAT,
    "params.lr_schedule": st.sampled_from(LRSchedule),
    "params.lr_final": ANY_FLOAT,
    "params.gamma": ANY_FLOAT,
    "params.buffer_size": ANY_INT,
    "params.batch_size": ANY_INT,
    "params.exploration_fraction": ANY_FLOAT,
    "params.exploration_final_eps": ANY_FLOAT,
    "eval.every_episodes": MAYBE_INT,
    "eval.episodes": ANY_INT,
    "eval.checkpoint_window": ANY_INT,
    "eval.optimal_steps": MAYBE_INT,
    "eval.sustain_evals": ANY_INT,
    "eval.stop_on_sustained": st.booleans(),
    "total_timesteps": ANY_INT,
    "schedule_timesteps": ANY_INT,
    "seed": ANY_INT,
    "trial": ANY_INT,
}
KEYDOOR_BASE = {"env.kind": EnvKind.KEYDOOR, "env.env_id": HALLWAY_KEYDOOR_ENV_ID}


def _dqn_config(overrides: dict[str, Any]) -> DQNTrialConfig | None:
    """A DQN config with ``overrides`` (``part.field`` keys) on top of valid defaults, or None if rejected."""
    parts: dict[str, dict[str, Any]] = {"env": {}, "params": {}, "eval": {}, "": {}}
    for key, value in overrides.items():
        part, _, name = key.rpartition(".")
        parts[part][name] = value
    try:
        return DQNTrialConfig(
            env=EnvSpec(**parts["env"]),
            params=DQNParams(**parts["params"]),
            eval=EvalSpec(**parts["eval"]),
            **{"total_timesteps": 1_000, **parts[""]},
        )
    except ValueError:
        return None


@st.composite
def dqn_overrides(draw):
    base = draw(st.sampled_from([{}, KEYDOOR_BASE]))
    fields = draw(st.lists(st.sampled_from(sorted(FIELD_VALUES)), min_size=1, max_size=3, unique=True))
    return {**base, **{field: draw(FIELD_VALUES[field]) for field in fields}}


TABULAR_VALUES = {
    "delay": ANY_INT,
    "max_steps": MAYBE_INT,
    "episodes": ANY_INT,
    "alpha": ANY_FLOAT,
    "gamma": ANY_FLOAT,
    "epsilon": ANY_FLOAT,
    "seed": ANY_INT,
    "trial": ANY_INT,
    "log_every": ANY_INT,
}


@st.composite
def tabular_overrides(draw):
    fields = draw(st.lists(st.sampled_from(sorted(TABULAR_VALUES)), min_size=1, max_size=3, unique=True))
    return {field: draw(TABULAR_VALUES[field]) for field in fields}


def _assert_sound(config) -> None:
    settings = config.to_dict()
    restored = json.loads(json.dumps(settings, allow_nan=False))
    assert config_mismatches(restored, config) == {}
    for value in settings.values():
        assert not (isinstance(value, float) and not math.isfinite(value))
    moved = config.for_trial(delay=config.delay + 1, trial=config.trial + 1)
    assert config_mismatches(restored, moved).keys() >= {"delay", "trial"}


@given(dqn_overrides())
def test_dqn_config_is_rejected_or_sound(overrides) -> None:
    config = _dqn_config(overrides)
    if config is not None:
        _assert_sound(config)


@given(tabular_overrides())
def test_tabular_config_is_rejected_or_sound(overrides) -> None:
    try:
        config = TabularTrialConfig(**overrides)
    except ValueError:
        return
    _assert_sound(config)


@pytest.mark.parametrize(
    ("strategy", "build"),
    [(dqn_overrides(), _dqn_config), (tabular_overrides(), lambda o: _try(TabularTrialConfig, o))],
    ids=["dqn", "tabular"],
)
def test_fuzzers_mostly_build_valid_configs(strategy, build) -> None:
    """Guard against a vacuous fuzzer: most generated inputs must reach the soundness checks."""
    built = [build(strategy.example()) is not None for _ in range(200)]
    assert sum(built) >= 60


def _try(builder, overrides):
    try:
        return builder(**overrides)
    except ValueError:
        return None


DQN_FLAGS = {
    "--env-kind": st.sampled_from(["empty", "keydoor", "bogus"]),
    "--env-id": ENV_IDS.filter(bool),
    "--subgoal-reward": st.sampled_from(["0.3", "-1", "nan", "inf", "x"]),
    "--step-penalty": st.sampled_from(["0", "0.001", "-0.1", "nan"]),
    "--max-steps": st.sampled_from(["50", "0", "-3", "1.5"]),
    "--timesteps": st.sampled_from(["1000", "0", "-1"]),
    "--schedule-timesteps": st.sampled_from(["500", "0", "5000000"]),
    "--learning-rate": st.sampled_from(["1e-3", "0", "-1", "nan", "inf"]),
    "--lr-schedule": st.sampled_from(["constant", "linear", "cosine"]),
    "--lr-final": st.sampled_from(["0", "1e-5", "1", "nan"]),
    "--eval-every-episodes": st.sampled_from(["0", "20", "-2"]),
    "--eval-episodes": st.sampled_from(["1", "0"]),
    "--checkpoint-window": st.sampled_from(["100", "0"]),
    "--optimal-steps": st.sampled_from(["25", "0"]),
    "--sustain-evals": st.sampled_from(["3", "0"]),
    "--delay": st.sampled_from(["0", "5", "-1"]),
    "--trials": st.sampled_from(["2", "0"]),
}
DQN_SWITCHES = ["--double-dqn", "--stop-on-sustained"]


@given(
    st.dictionaries(st.sampled_from(sorted(DQN_FLAGS)), st.just(None)),
    st.lists(st.sampled_from(DQN_SWITCHES)),
    st.data(),
)
def test_dqn_cli_rejects_cleanly_or_builds_sound_configs(flags, switches, data) -> None:
    argv = [part for flag in flags for part in (flag, data.draw(DQN_FLAGS[flag]))] + switches
    try:
        args = dqn_sweep.build_parser().parse_args(argv)
        template, delays, _trial_ids, _run_dir = _resolve_without_run_dir(args)
    except SystemExit:
        return
    for delay in delays:
        _assert_sound(template.for_trial(delay=delay, trial=0))


def _resolve_without_run_dir(args):
    delays = list(dict.fromkeys(args.delay))
    if args.trials < 1 or any(delay < 0 for delay in delays):
        raise SystemExit("rejected")
    return dqn_config_from_args(args), delays, range(args.trials), None


TABULAR_FLAGS = {
    "--episodes": st.sampled_from(["100", "0", "-5"]),
    "--alpha": st.sampled_from(["0.1", "0", "1.5", "nan"]),
    "--gamma": st.sampled_from(["0.99", "-0.1", "2", "nan"]),
    "--epsilon": st.sampled_from(["0.1", "-1", "1.1", "inf"]),
    "--log-every": st.sampled_from(["100", "0"]),
    "--max-steps": st.sampled_from(["50", "0"]),
    "--delay": st.sampled_from(["0", "-1", "7"]),
    "--onset-threshold": st.sampled_from(["0.5", "2", "-1", "nan"]),
}


@given(st.dictionaries(st.sampled_from(sorted(TABULAR_FLAGS)), st.just(None), max_size=3), st.data())
def test_tabular_cli_rejects_cleanly_or_builds_sound_configs(flags, data) -> None:
    argv = [part for flag in flags for part in (flag, data.draw(TABULAR_FLAGS[flag]))]
    try:
        template, delays = tabular_config_from_args(tabular_sweep.build_parser().parse_args(argv))
    except SystemExit:
        return
    for delay in delays:
        _assert_sound(template.for_trial(delay=delay, trial=0))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_learning_rate_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        DQNParams(learning_rate=float(value))
