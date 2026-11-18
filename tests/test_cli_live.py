"""Run the real CLIs end to end as subprocesses, the way they are used from a shell.

Tiny budgets keep each run to seconds. The Daytona test spends real sandbox time,
so it only runs when selected with ``-m daytona`` (``make smoke-daytona``).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HALLWAY = ["--env-kind", "keydoor", "--env-id", "MiniGrid-DoorKey-Hallway-20x6-v0", "--step-penalty", "0.001"]
TINY_DQN = [*HALLWAY, "--timesteps", "600", "--eval-every-episodes", "2", "--optimal-steps", "40"]


def _cli(module: str, *argv: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", f"delayed_reward.cli.{module}", *argv],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    if check and result.returncode != 0:
        raise AssertionError(f"{module} exited {result.returncode}:\n{result.stdout}\n{result.stderr}")
    return result


def _run_dir(stdout: str) -> Path:
    line = next(line for line in stdout.splitlines() if line.startswith("RUN_DIR "))
    return Path(line.removeprefix("RUN_DIR ").split()[0])


def _trial_files(run_dir: Path, delays: list[int], trials: int) -> None:
    for delay in delays:
        for trial in range(trials):
            trial_dir = run_dir / f"delay_{delay}" / f"trial_{trial}"
            for name in ("summary.json", "metrics.csv", "policy.gif"):
                assert (trial_dir / name).is_file(), trial_dir / name


def test_dqn_sweep_runs_resumes_and_refuses_changed_settings(tmp_path: Path) -> None:
    sweep = [*TINY_DQN, "--delay", "0", "3", "--trials", "2", "--workers", "2"]
    first = _cli("dqn_sweep", *sweep, "--results-dir", str(tmp_path))
    run_dir = _run_dir(first.stdout)
    _trial_files(run_dir, [0, 3], 2)
    assert (run_dir / "delay_survival.png").is_file()
    assert (run_dir / "time_to_optimal.json").is_file()
    summary = json.loads((run_dir / "delay_3" / "trial_1" / "summary.json").read_text())
    assert (summary["config"]["delay"], summary["config"]["seed"]) == (3, 1)

    resumed = _cli("dqn_sweep", *sweep, "--run-dir", str(run_dir))
    assert "PENDING 0 SKIPPED 4" in resumed.stdout

    (run_dir / "delay_3" / "trial_1" / "summary.json").unlink()
    rerun = _cli("dqn_sweep", *sweep, "--run-dir", str(run_dir))
    assert "PENDING 1 SKIPPED 3" in rerun.stdout
    assert len(list((run_dir / "_incomplete").iterdir())) == 1

    refused = _cli("dqn_sweep", *sweep, "--run-dir", str(run_dir), "--learning-rate", "5e-4", check=False)
    assert refused.returncode != 0
    assert "refusing to mix settings" in refused.stderr and "learning_rate" in refused.stderr

    replot = _cli("replot", str(run_dir))
    assert "learning_curves" in replot.stdout
    assert (run_dir / "learning_curves.png").is_file()


def test_sequential_dqn_cli(tmp_path: Path) -> None:
    result = _cli("dqn", *TINY_DQN, "--delay", "2", "--trials", "1", "--results-dir", str(tmp_path))
    _trial_files(_run_dir(result.stdout), [2], 1)


def test_tabular_sweep_and_sequential_cli(tmp_path: Path) -> None:
    tiny = ["--episodes", "60", "--max-steps", "60", "--log-every", "20", "--delay", "0", "4"]
    sweep = _cli("tabular_sweep", *tiny, "--trials", "2", "--workers", "2", "--results-dir", str(tmp_path / "sweep"))
    run_dir = _run_dir(sweep.stdout)
    _trial_files(run_dir, [0, 4], 2)
    assert "PENDING 0 SKIPPED 4" in _cli("tabular_sweep", *tiny, "--trials", "2", "--run-dir", str(run_dir)).stdout

    _cli("tabular", *tiny, "--trials", "1", "--results-dir", str(tmp_path / "seq"))
    (seq_run,) = (tmp_path / "seq").iterdir()
    _trial_files(seq_run, [0, 4], 1)
    with (seq_run / "delay_4" / "trial_0" / "metrics.csv").open() as fh:
        assert len(list(csv.DictReader(fh))) == 60


@pytest.mark.parametrize(
    "argv",
    [
        ["--randomize", "start"],
        ["--env-kind", "keydoor", "--randomize", "start"],
        ["--lr-schedule", "linear", "--lr-final", "1"],
        ["--learning-rate", "nan"],
        ["--timesteps", "0"],
        ["--optimal-steps", "5", "--eval-every-episodes", "0"],
    ],
    ids=" ".join,
)
def test_bad_input_fails_fast_without_creating_a_run(tmp_path: Path, argv: list[str]) -> None:
    result = _cli("dqn_sweep", *argv, "--results-dir", str(tmp_path), check=False)
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def _live_sandboxes() -> int:
    from daytona import Daytona, ListSandboxesQuery  # noqa: PLC0415

    return len(list(Daytona().list(ListSandboxesQuery(labels={"app": "delayed-reward"}))))


@pytest.mark.daytona
def test_daytona_trial_round_trip(tmp_path: Path) -> None:
    """One 2k-step trial on a real sandbox: downloaded artifacts, CPU device, no sandbox left behind."""
    if not os.environ.get("DAYTONA_API_KEY"):
        pytest.skip("needs DAYTONA_API_KEY (source .env first)")
    result = _cli(
        "daytona",
        *TINY_DQN,
        "--timesteps",
        "2000",
        "--delay",
        "0",
        "--trials",
        "1",
        "--max-sandboxes",
        "1",
        "--poll-seconds",
        "10",
        "--results-dir",
        str(tmp_path),
    )
    run_dir = _run_dir(result.stdout)
    _trial_files(run_dir, [0], 1)
    trial_dir = run_dir / "delay_0" / "trial_0"
    assert (trial_dir / "daytona.log").is_file()
    assert json.loads((trial_dir / "summary.json").read_text())["config"]["device"] == "cpu"
    deadline = time.monotonic() + 90
    while _live_sandboxes() and time.monotonic() < deadline:
        time.sleep(5)
    assert _live_sandboxes() == 0
