"""Optional Weights & Biases helpers shared by the experiment entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def start_wandb_run(
    *,
    enabled: bool,
    project: str,
    config: Mapping[str, Any],
    name: str,
    group: str | None = None,
    entity: str | None = None,
    step_metric: str | None = None,
) -> Any | None:
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requires the optional 'wandb' package") from exc
    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        config=dict(config),
        reinit="finish_previous",
    )
    if step_metric:
        run.define_metric(step_metric)
        run.define_metric("*", step_metric=step_metric)
    return run


def log_metrics(run: Any | None, metrics: Mapping[str, float], *, step: int) -> None:
    if run is not None:
        run.log(dict(metrics), step=step)
