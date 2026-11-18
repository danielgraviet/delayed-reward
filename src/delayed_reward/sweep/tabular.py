"""Tabular sweep pieces: the process-pool trial entry."""

from __future__ import annotations

from pathlib import Path

from delayed_reward.config import TabularTrialConfig, WandbSettings
from delayed_reward.experiments.tabular import train_one_trial
from delayed_reward.results.saver import create_delay_dir


def run_trial_task(task: tuple[TabularTrialConfig, Path, WandbSettings | None]) -> Path:
    """Process-pool entry: train one planned config into its ``delay_<d>/trial_<t>`` directory."""
    config, run_dir, wandb = task
    return train_one_trial(config, create_delay_dir(run_dir, config.delay), wandb=wandb)
