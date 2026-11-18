"""Train Stable-Baselines3 DQN on the project's fully observed MiniGrid."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import LinearSchedule

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
from delayed_reward.double_dqn import DoubleDQN
from delayed_reward.environments.factory import (
    DEFAULT_ENV_ID,
    KEYDOOR_ENV_ID,
    create_empty_dqn_env,
    create_keydoor_dqn_env,
)
from delayed_reward.environments.layout import LayoutRandomization
from delayed_reward.masked_dqn_policy import MaskedDQNPolicy
from delayed_reward.results.saver import (
    create_delay_dir,
    create_run_dir,
    create_trial_dir,
    save_eval_metrics,
    save_metrics,
)
from delayed_reward.training.wandb_logging import log_metrics, start_wandb_run


def build_dqn_env(spec: EnvSpec, *, seed: int, render_mode: str | None = None) -> gym.Env:
    """Build the flat-observation DQN env that ``spec`` describes."""
    if spec.is_keydoor:
        return create_keydoor_dqn_env(
            env_id=spec.env_id,
            seed=seed,
            delay=spec.delay,
            subgoal_reward=spec.subgoal_reward,
            render_mode=render_mode,
            max_steps=spec.max_steps,
            step_penalty=spec.step_penalty,
            randomize=spec.randomize,
        )
    return create_empty_dqn_env(
        env_id=spec.env_id,
        seed=seed,
        delay=spec.delay,
        render_mode=render_mode,
        max_steps=spec.max_steps,
    )


class EpisodeMetricsCallback(BaseCallback):
    """Collect episode metrics and optionally relabel delayed subgoal rewards."""

    def __init__(
        self,
        *,
        reward_attribution: RewardAttribution = RewardAttribution.ARRIVAL,
        wandb_run=None,
        log_every: int = 1000,
    ) -> None:
        super().__init__()
        self.reward_attribution = RewardAttribution(reward_attribution)
        self.rewards: list[float] = []
        self.steps: list[int] = []
        self._transition_indices: dict[int, int] = {}
        self.wandb_run = wandb_run
        self.log_every = log_every

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        metrics: dict[str, float] = {}
        buffer = self.model.replay_buffer
        if buffer is not None:
            current_index = (buffer.pos - 1) % buffer.buffer_size
            for info in infos:
                step = info.get("environment_step")
                if step is not None:
                    self._transition_indices[int(step)] = current_index
                    oldest = int(step) - buffer.buffer_size
                    self._transition_indices = {
                        key: value for key, value in self._transition_indices.items() if key > oldest
                    }
                if self.reward_attribution is RewardAttribution.ORIGIN and "subgoal_reward_origin_step" in info:
                    origin = int(info["subgoal_reward_origin_step"])
                    origin_index = self._transition_indices.get(origin)
                    if origin_index is None:
                        raise RuntimeError(f"origin transition {origin} is no longer in replay")
                    amount = float(info["subgoal_reward_amount"])
                    buffer.rewards[current_index, 0] -= amount
                    buffer.rewards[origin_index, 0] += amount
        for done, info in zip(self.locals["dones"], infos, strict=True):
            if done and "episode" in info:
                self.rewards.append(float(info["episode"]["r"]))
                self.steps.append(int(info["episode"]["l"]))
                metrics.update({
                    "episode/return": float(info["episode"]["r"]),
                    "episode/steps": float(info["episode"]["l"]),
                    "episode/count": float(len(self.rewards)),
                })
        # SB3's train/rollout stats are only gathered for W&B; skip them without a run.
        if self.wandb_run is not None and self.num_timesteps % self.log_every == 0:
            values = self.model.logger.name_to_value
            metrics.update({
                key.replace("/", "_"): float(value)
                for key, value in values.items()
                if key.startswith(("train/", "rollout/")) and np.isscalar(value)
            })
            metrics["timesteps"] = float(self.num_timesteps)
        if metrics:
            log_metrics(self.wandb_run, metrics, step=int(self.num_timesteps))
        return True


class GreedyEvalCallback(BaseCallback):
    """Periodically run the current policy with exploration turned off.

    Training episodes mix in epsilon-random actions by design (see
    ``exploration_final_eps`` below -- it never reaches 0), so even the
    training return curve is noisier than the policy actually is. This
    callback separately asks "how good is the greedy policy right now" by
    rolling out ``spec.episodes`` episodes with ``deterministic=True`` and
    no exploration, every ``spec.every_episodes`` completed training
    episodes.

    By default the environment is reseeded to the same fixed layout every
    episode (see ``ReseedWrapper`` in ``environments/factory.py``), so a greedy
    policy's rollout is itself deterministic -- there's no layout or
    start-state randomness left once exploration is off. That's why
    ``spec.episodes`` defaults to 1: running it more times would just repeat
    the identical trajectory. With layout randomization on (see
    ``LayoutRandomization``) every eval episode draws a new start/key/door, so
    raise ``spec.episodes`` to average over them.

    Online TD updates keep nudging the Q-network after it has already found
    a working greedy policy, and the near-tied Q-values this task's sparse
    reward produces (see ``StepPenaltyWrapper``) mean a late update can flip
    the final argmax into a degenerate loop even after many evals showed a
    clean solve. ``spec.checkpoint_window`` keeps a deep-copied policy snapshot for each
    of the last ``checkpoint_window`` eval checkpoints so ``best_in_window`` can
    hand back whichever of those recent snapshots actually solved the task
    fastest, instead of trusting whatever the raw final-timestep weights do.

    With ``spec.optimal_steps`` set, a checkpoint counts as an *optimal solve* when
    every eval episode reaches the goal (terminates, not truncated) in at most
    ``optimal_steps`` steps. The callback records the first optimal checkpoint
    and the first checkpoint closing a run of ``sustain_evals`` consecutive
    optimal ones (the sustained solve), and with ``stop_on_sustained`` it ends
    training there. Consecutive checkpoints are separated by training, so on a
    fixed layout they are not repeats of one deterministic rollout.
    """

    def __init__(self, *, eval_env: gym.Env, spec: EvalSpec, max_steps: int) -> None:
        super().__init__()
        if spec.every_episodes is None:
            raise ValueError("GreedyEvalCallback needs spec.every_episodes")
        self.eval_env = eval_env
        self.eval_every_episodes = spec.every_episodes
        self.max_steps = max_steps
        self.n_eval_episodes = spec.episodes
        self.optimal_steps = spec.optimal_steps
        self.sustain_evals = spec.sustain_evals
        self.stop_on_sustained = spec.stop_on_sustained
        self._train_episode_count = 0
        self.eval_train_episodes: list[int] = []
        self.eval_timesteps: list[int] = []
        self.eval_returns: list[float] = []
        self.eval_steps: list[float] = []
        self.eval_optimal: list[bool] = []
        self._optimal_streak = 0
        self.first_optimal: dict[str, int] | None = None
        self.sustained_optimal: dict[str, int] | None = None
        self._window: deque[dict[str, object]] = deque(maxlen=spec.checkpoint_window)

    def _run_eval(self) -> tuple[float, float, bool]:
        """Return (mean return, mean length, whether every episode was an optimal solve)."""
        returns: list[float] = []
        lengths: list[int] = []
        all_optimal = self.optimal_steps is not None
        for _ in range(self.n_eval_episodes):
            observation, _info = self.eval_env.reset()
            total_reward = 0.0
            length = 0
            reached_goal = False
            for _ in range(self.max_steps):
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _info = self.eval_env.step(
                    int(action)
                )
                total_reward += float(reward)
                length += 1
                if terminated or truncated:
                    reached_goal = bool(terminated)
                    break
            returns.append(total_reward)
            lengths.append(length)
            if self.optimal_steps is not None:
                all_optimal = all_optimal and reached_goal and length <= self.optimal_steps
        return float(np.mean(returns)), float(np.mean(lengths)), all_optimal

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done and "episode" in info:
                self._train_episode_count += 1
                if self._train_episode_count % self.eval_every_episodes == 0:
                    eval_return, eval_length, optimal = self._run_eval()
                    self.eval_train_episodes.append(self._train_episode_count)
                    self.eval_timesteps.append(int(self.num_timesteps))
                    self.eval_returns.append(eval_return)
                    self.eval_steps.append(eval_length)
                    self.eval_optimal.append(optimal)
                    self._record_optimal(optimal)
                    self._window.append(
                        {
                            "train_episode": self._train_episode_count,
                            "timestep": int(self.num_timesteps),
                            "return": eval_return,
                            "steps": eval_length,
                            "state_dict": {
                                key: value.detach().clone().cpu()
                                for key, value in self.model.policy.state_dict().items()
                            },
                        }
                    )
        return not (self.stop_on_sustained and self.sustained_optimal is not None)

    def _record_optimal(self, optimal: bool) -> None:
        if self.optimal_steps is None:
            return
        self._optimal_streak = self._optimal_streak + 1 if optimal else 0
        point = {"train_episode": self._train_episode_count, "timestep": int(self.num_timesteps)}
        if optimal and self.first_optimal is None:
            self.first_optimal = point
        if self._optimal_streak >= self.sustain_evals and self.sustained_optimal is None:
            self.sustained_optimal = point

    def time_to_optimal(self, total_timesteps: int) -> dict[str, object] | None:
        """Summary of first/sustained optimal checkpoints; ``censored`` if never sustained."""
        if self.optimal_steps is None:
            return None
        return {
            "optimal_steps": self.optimal_steps,
            "sustain_evals": self.sustain_evals,
            "first_optimal_timestep": (self.first_optimal or {}).get("timestep"),
            "first_optimal_train_episode": (self.first_optimal or {}).get("train_episode"),
            "sustained_optimal_timestep": (self.sustained_optimal or {}).get("timestep"),
            "sustained_optimal_train_episode": (self.sustained_optimal or {}).get("train_episode"),
            "censored": self.sustained_optimal is None,
            "censor_timestep": total_timesteps if self.sustained_optimal is None else None,
            "trained_timesteps": int(self.num_timesteps),
            "stopped_early": self.stop_on_sustained and self.sustained_optimal is not None,
        }

    def best_in_window(self) -> dict[str, object] | None:
        """Return the window entry with the highest return (fewest steps breaks ties)."""
        if not self._window:
            return None
        return min(self._window, key=lambda entry: (-entry["return"], entry["steps"]))



def make_lr_schedule(params: DQNParams, end_fraction: float = 1.0) -> float | LinearSchedule:
    """Return SB3's ``learning_rate`` argument: a constant, or a linear anneal.

    ``linear`` decays from ``learning_rate`` at the first update to ``lr_final``
    after ``end_fraction`` of training (SB3 passes the remaining progress,
    1 -> 0) and holds ``lr_final`` after that.
    """
    if not 0 < end_fraction <= 1:
        raise ValueError(f"end_fraction must be in (0, 1], got {end_fraction}")
    if params.lr_schedule is LRSchedule.LINEAR:
        return LinearSchedule(params.learning_rate, params.lr_final, end_fraction)
    return params.learning_rate


def build_model(config: DQNTrialConfig, env: gym.Env) -> DQN:
    """Construct the SB3 learner from ``config``; every hyperparameter comes from ``config.params``."""
    params = config.params
    algorithm = DoubleDQN if params.double_dqn else DQN
    return algorithm(
        MaskedDQNPolicy if config.env.is_keydoor else "MlpPolicy",
        env,
        policy_kwargs={"net_arch": list(params.network)},
        learning_rate=make_lr_schedule(params, config.schedule_fraction),
        buffer_size=params.buffer_size,
        learning_starts=params.learning_starts,
        batch_size=params.batch_size,
        gamma=params.gamma,
        train_freq=params.train_freq,
        target_update_interval=params.target_update_interval,
        exploration_fraction=params.exploration_fraction * config.schedule_fraction,
        exploration_initial_eps=params.exploration_initial_eps,
        exploration_final_eps=params.exploration_final_eps,
        max_grad_norm=params.max_grad_norm,
        seed=config.trial_seed,
        device="auto",
        verbose=0,
    )


def train_trial(
    config: DQNTrialConfig, trial_dir: Path, *, wandb: WandbSettings | None = None
) -> None:
    """Train one DQN trial and write its artifacts into ``trial_dir``.

    With ``config.eval.stop_on_sustained`` the trial ends at its sustained
    optimal solve (see ``GreedyEvalCallback``).
    """
    env = gym.wrappers.RecordEpisodeStatistics(build_dqn_env(config.env, seed=config.trial_seed))
    settings = config.to_dict()
    wandb_run = None
    if wandb is not None:
        wandb_run = start_wandb_run(
            enabled=True, project=wandb.project, entity=wandb.entity, group=wandb.group,
            name=f"delay-{config.env.delay}-trial-{config.trial}", config=settings,
        )
    callback = EpisodeMetricsCallback(
        reward_attribution=config.reward_attribution,
        wandb_run=wandb_run,
        log_every=wandb.log_every if wandb is not None else 1000,
    )
    callbacks: list[BaseCallback] = [callback]
    eval_callback: GreedyEvalCallback | None = None
    if config.eval.every_episodes is not None:
        eval_callback = GreedyEvalCallback(
            eval_env=build_dqn_env(config.env, seed=config.trial_seed),
            spec=config.eval,
            max_steps=config.env.max_steps or 2000,
        )
        callbacks.append(eval_callback)
    model = build_model(config, env)
    try:
        model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
    finally:
        env.close()
        if eval_callback is not None:
            eval_callback.eval_env.close()

    best = eval_callback.best_in_window() if eval_callback is not None else None
    if best is not None:
        model.policy.load_state_dict(best["state_dict"])
        checkpoint_selection: dict[str, object] = {
            "strategy": "best_of_last_n_eval_checkpoints",
            "window": config.eval.checkpoint_window,
            "selected_train_episode": best["train_episode"],
            "selected_timestep": best["timestep"],
            "selected_eval_return": best["return"],
            "selected_eval_steps": best["steps"],
        }
    else:
        checkpoint_selection = {"strategy": "final_weights"}

    time_to_optimal = (
        eval_callback.time_to_optimal(config.total_timesteps) if eval_callback is not None else None
    )
    save_metrics(
        trial_dir,
        rewards=callback.rewards,
        steps=callback.steps,
        config={**settings, "device": str(model.device), "checkpoint_selection": checkpoint_selection},
        window=100,
        extra={"time_to_optimal": time_to_optimal} if time_to_optimal is not None else None,
    )
    if eval_callback is not None:
        save_eval_metrics(
            trial_dir,
            train_episodes=eval_callback.eval_train_episodes,
            train_timesteps=eval_callback.eval_timesteps,
            returns=eval_callback.eval_returns,
            steps=eval_callback.eval_steps,
            optimal=eval_callback.eval_optimal if config.eval.optimal_steps is not None else None,
        )
    model.save(trial_dir / "model")
    _save_policy_gif(model, config.env, seed=config.trial_seed, path=trial_dir / "policy.gif")
    recent = callback.rewards[-100:]
    print(
        f"trial={config.trial} episodes={len(callback.rewards)} "
        f"last_100_mean_return={np.mean(recent) if recent else float('nan'):.3f} "
        f"device={model.device} "
        + (
            f"sustained_optimal_timestep={time_to_optimal['sustained_optimal_timestep']} "
            if time_to_optimal is not None
            else ""
        )
        + f"path={trial_dir}",
        flush=True,
    )
    if wandb_run is not None:
        if time_to_optimal is not None:
            wandb_run.summary.update(
                {f"time_to_optimal/{k}": v for k, v in time_to_optimal.items()}
            )
        wandb_run.finish()


def _save_policy_gif(model: DQN, spec: EnvSpec, *, seed: int, path: Path) -> None:
    max_steps = spec.max_steps or 200
    env = build_dqn_env(replace(spec, max_steps=max_steps), seed=seed, render_mode="rgb_array")
    frames: list[Image.Image] = []
    try:
        observation, _ = env.reset()
        frames.append(Image.fromarray(np.asarray(env.render())))
        for _ in range(max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, _info = env.step(int(action))
            frames.append(Image.fromarray(np.asarray(env.render())))
            if terminated or truncated:
                break
    finally:
        env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)


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
    parser.add_argument(
        "--lr-final", type=float, default=0.0, help="Final learning rate for --lr-schedule linear"
    )
    parser.add_argument(
        "--eval-every-episodes",
        type=int,
        default=20,
        help=(
            "Run a deterministic (exploration-off) evaluation every N completed "
            "training episodes. Pass 0 to disable."
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
