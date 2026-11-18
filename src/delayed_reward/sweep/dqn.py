"""DQN sweep pieces: the process-pool trial entry and the run-level time-to-optimal summary."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from delayed_reward.config import DQNTrialConfig, WandbSettings
from delayed_reward.experiments.dqn import train_trial
from delayed_reward.results.saver import create_delay_dir, create_trial_dir


def run_trial_task(task: tuple[DQNTrialConfig, Path, WandbSettings | None]) -> Path:
    """Process-pool entry: train one planned config into its ``delay_<d>/trial_<t>`` directory."""
    config, run_dir, wandb = task
    trial_dir = create_trial_dir(create_delay_dir(run_dir, config.env.delay), config.trial)
    train_trial(config, trial_dir, wandb=wandb)
    return trial_dir


@dataclass(frozen=True, kw_only=True)
class TrialSolve:
    """One trial's time-to-optimal outcome, as read from its ``summary.json``."""

    first_optimal_timestep: int | None
    sustained_optimal_timestep: int | None
    censored: bool
    censor_timestep: int | None


def _delay_summary(trials: dict[int, TrialSolve]) -> dict[str, object]:
    """Counts and solve times for one delay; censored trials rank last, so a median needs a solving majority."""
    solved = sorted(
        t.sustained_optimal_timestep
        for t in trials.values()
        if not t.censored and t.sustained_optimal_timestep is not None
    )
    first = sorted(t.first_optimal_timestep for t in trials.values() if t.first_optimal_timestep is not None)
    n = len(trials)
    median_rank = (n - 1) // 2
    return {
        "trials": {trial: asdict(solve) for trial, solve in sorted(trials.items())},
        "n_trials": n,
        "n_sustained": len(solved),
        "n_censored": n - len(solved),
        "n_first_optimal": len(first),
        "median_sustained_timestep": solved[median_rank] if len(solved) > median_rank else None,
        "sustained_timesteps_solved": solved,
        "first_optimal_timesteps": first,
    }


def summarize_time_to_optimal(run_dir: Path) -> Path | None:
    """Collect each trial's ``time_to_optimal`` into ``run_dir/time_to_optimal.json``.

    Trials that never reached the sustained criterion are right-censored at
    their cap: they are listed and counted, but their cap is never reported as
    a solve time. Returns None when no trial recorded the criterion.
    """
    by_delay: dict[int, dict[int, TrialSolve]] = {}
    for summary_path in sorted(run_dir.glob("delay_*/trial_*/summary.json")):
        record = json.loads(summary_path.read_text()).get("time_to_optimal")
        if record is None:
            continue
        delay = int(summary_path.parent.parent.name.removeprefix("delay_"))
        trial = int(summary_path.parent.name.removeprefix("trial_"))
        by_delay.setdefault(delay, {})[trial] = TrialSolve(
            first_optimal_timestep=record["first_optimal_timestep"],
            sustained_optimal_timestep=record["sustained_optimal_timestep"],
            censored=record["censored"],
            censor_timestep=record["censor_timestep"],
        )
    if not by_delay:
        return None
    summary = {str(delay): _delay_summary(trials) for delay, trials in sorted(by_delay.items())}
    out = run_dir / "time_to_optimal.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    for delay, entry in summary.items():
        print(
            f"TIME_TO_OPTIMAL delay={delay} sustained={entry['n_sustained']}/{entry['n_trials']} "
            f"censored={entry['n_censored']} median={entry['median_sustained_timestep']}",
            flush=True,
        )
    return out
