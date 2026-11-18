"""Plan, resume and summarize DQN sweeps: one run directory, many (delay, trial) configs."""

from __future__ import annotations

import json
from pathlib import Path

from delayed_reward.config import DQNTrialConfig, WandbSettings, config_mismatches
from delayed_reward.experiments.dqn import train_trial
from delayed_reward.results.saver import create_delay_dir, create_trial_dir
from delayed_reward.sweep.trials import stash_incomplete


def plan_tasks(
    template: DQNTrialConfig,
    run_dir: Path,
    delays: list[int],
    trial_ids: list[int],
) -> tuple[list[DQNTrialConfig], int]:
    """Return the pending trial configs and the count of completed ones skipped.

    Completed trials must match the requested settings. Leftover directories of
    incomplete trials are moved under ``run_dir/_incomplete``.
    """
    pending: list[DQNTrialConfig] = []
    skipped = 0
    for delay in delays:
        delay_dir = create_delay_dir(run_dir, delay)
        for trial in trial_ids:
            config = template.for_trial(delay=delay, trial=trial)
            summary_path = delay_dir / f"trial_{trial}" / "summary.json"
            if summary_path.exists():
                saved = json.loads(summary_path.read_text()).get("config", {})
                mismatches = config_mismatches(saved, config)
                if mismatches:
                    raise SystemExit(f"refusing to mix settings in {summary_path}: {mismatches}")
                skipped += 1
                continue
            stash_incomplete(run_dir, delay, trial)
            pending.append(config)
    return pending, skipped


def run_trial_task(task: tuple[DQNTrialConfig, Path, WandbSettings | None]) -> Path:
    """Process-pool entry: train one planned config into its ``delay_<d>/trial_<t>`` directory."""
    config, run_dir, wandb = task
    trial_dir = create_trial_dir(create_delay_dir(run_dir, config.env.delay), config.trial)
    train_trial(config, trial_dir, wandb=wandb)
    return trial_dir


def summarize_time_to_optimal(run_dir: Path) -> Path | None:
    """Collect each trial's ``time_to_optimal`` into ``run_dir/time_to_optimal.json``.

    Trials that never reached the sustained criterion are right-censored at
    their cap: they are listed and counted, but their cap is never reported as
    a solve time. Returns None when no trial recorded the criterion.
    """
    by_delay: dict[str, dict[str, object]] = {}
    for summary_path in sorted(run_dir.glob("delay_*/trial_*/summary.json")):
        record = json.loads(summary_path.read_text()).get("time_to_optimal")
        if record is None:
            continue
        delay = summary_path.parent.parent.name.removeprefix("delay_")
        trial = int(summary_path.parent.name.removeprefix("trial_"))
        entry = by_delay.setdefault(delay, {"trials": {}})
        entry["trials"][trial] = {
            key: record[key]
            for key in (
                "first_optimal_timestep",
                "sustained_optimal_timestep",
                "censored",
                "censor_timestep",
            )
        }
    if not by_delay:
        return None
    for entry in by_delay.values():
        trials = dict(sorted(entry["trials"].items()))
        entry["trials"] = trials
        solved = sorted(
            t["sustained_optimal_timestep"] for t in trials.values() if not t["censored"]
        )
        first = sorted(
            t["first_optimal_timestep"]
            for t in trials.values()
            if t["first_optimal_timestep"] is not None
        )
        n = len(trials)
        entry.update(
            {
                "n_trials": n,
                "n_sustained": len(solved),
                "n_censored": n - len(solved),
                "n_first_optimal": len(first),
                # Order statistic over all trials; censored trials rank last, so
                # the median exists only when more than half of them solved.
                "median_sustained_timestep": (
                    solved[(n - 1) // 2] if len(solved) > (n - 1) // 2 else None
                ),
                "sustained_timesteps_solved": solved,
                "first_optimal_timesteps": first,
            }
        )
    out = run_dir / "time_to_optimal.json"
    out.write_text(json.dumps(dict(sorted(by_delay.items(), key=lambda kv: int(kv[0]))), indent=2) + "\n")
    for delay, entry in by_delay.items():
        print(
            f"TIME_TO_OPTIMAL delay={delay} sustained={entry['n_sustained']}/{entry['n_trials']} "
            f"censored={entry['n_censored']} median={entry['median_sustained_timestep']}",
            flush=True,
        )
    return out
