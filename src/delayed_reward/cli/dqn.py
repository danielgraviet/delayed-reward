"""Train Stable-Baselines3 DQN trials sequentially in one run directory."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from delayed_reward.config import (
    DQNParams,
    DQNTrialConfig,
    EnvKind,
    EnvSpec,
    EvalSpec,
    LRSchedule,
    RewardAttribution,
    WandbSettings,
)
from delayed_reward.environments.layout import LayoutRandomization
from delayed_reward.environments.registry import DEFAULT_ENV_ID, KEYDOOR_ENV_ID
from delayed_reward.experiments.dqn import train_trial
from delayed_reward.results.saver import create_delay_dir, create_run_dir, create_trial_dir


def add_dqn_args(parser: argparse.ArgumentParser) -> None:
    """Add the options that define a DQN trial (and its W&B logging), shared by the DQN CLIs."""
    parser.add_argument("--env-kind", type=EnvKind, choices=list(EnvKind), default=EnvKind.EMPTY)
    parser.add_argument(
        "--env-id",
        default=None,
        help=f"Defaults to {DEFAULT_ENV_ID!r} (empty) or {KEYDOOR_ENV_ID!r} (keydoor)",
    )
    parser.add_argument(
        "--subgoal-reward",
        type=float,
        default=0.3,
        help="Reward for picking up the key (--env-kind keydoor only)",
    )
    parser.add_argument(
        "--reward-attribution",
        type=RewardAttribution,
        choices=list(RewardAttribution),
        default=RewardAttribution.ARRIVAL,
        help="Store delayed key reward on its arrival or pickup transition",
    )
    parser.add_argument(
        "--step-penalty",
        type=float,
        default=0.0,
        help=(
            "Fixed cost subtracted from every step's reward (--env-kind keydoor "
            "only). Breaks the near-tied Q-values that otherwise let a greedy "
            "policy get stuck spinning in place indefinitely; e.g. 0.001."
        ),
    )
    parser.add_argument(
        "--randomize",
        nargs="+",
        choices=LayoutRandomization.element_names(),
        default=[],
        metavar="ELEMENT",
        help=(
            "Re-sample these key-door layout elements uniformly every episode: "
            f"{', '.join(LayoutRandomization.element_names())} (door = wall column + door row). "
            "Others stay at the env's fixed positions. Needs a RectangularDoorKeyEnv env id, "
            "e.g. MiniGrid-DoorKey-Hallway-20x6-v0 (--env-kind keydoor only)."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Episode truncation horizon")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Training timestep cap")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; trial t uses seed + t")
    parser.add_argument(
        "--schedule-timesteps",
        type=int,
        default=None,
        help=(
            "Horizon the epsilon and linear-LR schedules anneal over (default: "
            "--timesteps). Set it to keep schedules fixed while raising --timesteps."
        ),
    )
    parser.add_argument(
        "--double-dqn",
        action="store_true",
        help="Double DQN targets: the online net picks the next action, the target net scores it",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial Adam learning rate")
    parser.add_argument(
        "--lr-schedule",
        type=LRSchedule,
        choices=list(LRSchedule),
        default=LRSchedule.CONSTANT,
        help=(
            "constant, or linear decay from --learning-rate to --lr-final over "
            "--schedule-timesteps (default --timesteps), then held"
        ),
    )
    parser.add_argument("--lr-final", type=float, default=0.0, help="Final learning rate for --lr-schedule linear")
    parser.add_argument(
        "--eval-every-episodes",
        type=int,
        default=20,
        help=(
            "Run a deterministic (exploration-off) evaluation every N completed training episodes. Pass 0 to disable."
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1,
        help="Greedy episodes averaged per eval; raise it when using --randomize",
    )
    parser.add_argument(
        "--checkpoint-window",
        type=int,
        default=100,
        help=(
            "Save and GIF the best deterministic-eval checkpoint from the last "
            "N eval checkpoints (see --eval-every-episodes), instead of the raw "
            "final-timestep weights."
        ),
    )
    parser.add_argument(
        "--optimal-steps",
        type=int,
        default=None,
        help="Greedy eval reaching the goal in <= N steps counts as an optimal solve",
    )
    parser.add_argument(
        "--sustain-evals",
        type=int,
        default=3,
        help="Consecutive optimal eval checkpoints that make a sustained solve",
    )
    parser.add_argument(
        "--stop-on-sustained",
        action="store_true",
        help="End a trial at its sustained optimal solve (needs --optimal-steps)",
    )
    parser.add_argument("--wandb", action="store_true", help="Log each trial to Weights & Biases")
    parser.add_argument("--wandb-project", default="delayed-reward")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-log-every", type=int, default=1000)


def dqn_config_from_args(args: argparse.Namespace) -> DQNTrialConfig:
    """The trial template (delay 0, trial 0) described by ``add_dqn_args`` options."""
    try:
        return DQNTrialConfig(
            env=EnvSpec(
                kind=args.env_kind,
                env_id=args.env_id or "",
                max_steps=args.max_steps,
                subgoal_reward=args.subgoal_reward,
                step_penalty=args.step_penalty,
                randomize=LayoutRandomization.from_names(args.randomize),
            ),
            total_timesteps=args.timesteps,
            schedule_timesteps=args.schedule_timesteps or 0,
            params=DQNParams(
                learning_rate=args.learning_rate,
                lr_schedule=args.lr_schedule,
                lr_final=args.lr_final,
                double_dqn=args.double_dqn,
            ),
            eval=EvalSpec(
                every_episodes=args.eval_every_episodes or None,
                episodes=args.eval_episodes,
                checkpoint_window=args.checkpoint_window,
                optimal_steps=args.optimal_steps,
                sustain_evals=args.sustain_evals,
                stop_on_sustained=args.stop_on_sustained,
            ),
            reward_attribution=args.reward_attribution,
            seed=args.seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def wandb_from_args(args: argparse.Namespace, *, group: str) -> WandbSettings | None:
    """W&B settings from ``add_dqn_args`` options, or None without ``--wandb``."""
    if not args.wandb:
        return None
    try:
        return WandbSettings(
            project=args.wandb_project, entity=args.wandb_entity, group=group, log_every=args.wandb_log_every
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dqn_args(parser)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.trials < 1:
        raise SystemExit("trials must be >= 1")
    template = dqn_config_from_args(args)
    try:
        template = template.for_trial(delay=args.delay, trial=0)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_dir = args.run_dir or create_run_dir(args.results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    delay_dir = create_delay_dir(run_dir, args.delay)
    wandb = wandb_from_args(args, group=run_dir.name)
    print(
        f"RUN_DIR {run_dir.resolve()} trials={args.trials} "
        f"timesteps={template.total_timesteps} delay={args.delay} env_kind={template.env.kind}",
        flush=True,
    )
    for trial in range(args.trials):
        config = replace(template, trial=trial)
        train_trial(config, create_trial_dir(delay_dir, trial), wandb=wandb)


if __name__ == "__main__":
    main()
