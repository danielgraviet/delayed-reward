"""Train Stable-Baselines3 DQN on the project's fully observed MiniGrid."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import LinearSchedule

from delayed_reward.double_dqn import DoubleDQN
from delayed_reward.environments.factory import (
    DEFAULT_ENV_ID,
    KEYDOOR_ENV_ID,
    create_empty_dqn_env,
    create_keydoor_dqn_env,
    fixed_layout_source,
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


def _make_env(
    *,
    env_kind: str,
    env_id: str,
    seed: int,
    delay: int,
    subgoal_reward: float,
    render_mode: str | None = None,
    max_steps: int | None = None,
    step_penalty: float = 0.0,
    randomize: LayoutRandomization = LayoutRandomization(),
) -> gym.Env:
    if env_kind == "keydoor":
        return create_keydoor_dqn_env(
            env_id=env_id,
            seed=seed,
            delay=delay,
            subgoal_reward=subgoal_reward,
            render_mode=render_mode,
            max_steps=max_steps,
            step_penalty=step_penalty,
            randomize=randomize,
        )
    if randomize.any:
        raise ValueError("layout randomization is only supported for env_kind='keydoor'")
    return create_empty_dqn_env(
        env_id=env_id, seed=seed, delay=delay, render_mode=render_mode, max_steps=max_steps
    )


class EpisodeMetricsCallback(BaseCallback):
    """Collect episode metrics and optionally relabel delayed subgoal rewards."""

    def __init__(self, *, reward_attribution: str = "arrival", wandb_run=None, log_every: int = 1000) -> None:
        super().__init__()
        if reward_attribution not in {"arrival", "origin"}:
            raise ValueError("reward_attribution must be 'arrival' or 'origin'")
        self.reward_attribution = reward_attribution
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
                if self.reward_attribution == "origin" and "subgoal_reward_origin_step" in info:
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
    rolling out ``n_eval_episodes`` episodes with ``deterministic=True`` and
    no exploration, every ``eval_every_episodes`` completed training
    episodes.

    By default the environment is reseeded to the same fixed layout every
    episode (see ``ReseedWrapper`` in ``environments/factory.py``), so a greedy
    policy's rollout is itself deterministic -- there's no layout or
    start-state randomness left once exploration is off. That's why
    ``n_eval_episodes`` defaults to 1: running it more times would just repeat
    the identical trajectory. With layout randomization on (see
    ``LayoutRandomization``) every eval episode draws a new start/key/door, so
    raise ``n_eval_episodes`` to average over them.

    Online TD updates keep nudging the Q-network after it has already found
    a working greedy policy, and the near-tied Q-values this task's sparse
    reward produces (see ``StepPenaltyWrapper``) mean a late update can flip
    the final argmax into a degenerate loop even after many evals showed a
    clean solve. ``best_window`` keeps a deep-copied policy snapshot for each
    of the last ``best_window`` eval checkpoints so ``best_in_window`` can
    hand back whichever of those recent snapshots actually solved the task
    fastest, instead of trusting whatever the raw final-timestep weights do.

    With ``optimal_steps`` set, a checkpoint counts as an *optimal solve* when
    every eval episode reaches the goal (terminates, not truncated) in at most
    ``optimal_steps`` steps. The callback records the first optimal checkpoint
    and the first checkpoint closing a run of ``sustain_evals`` consecutive
    optimal ones (the sustained solve), and with ``stop_on_sustained`` it ends
    training there. Consecutive checkpoints are separated by training, so on a
    fixed layout they are not repeats of one deterministic rollout.
    """

    def __init__(
        self,
        *,
        eval_env: gym.Env,
        eval_every_episodes: int,
        max_steps: int,
        n_eval_episodes: int = 1,
        best_window: int = 100,
        optimal_steps: int | None = None,
        sustain_evals: int = 3,
        stop_on_sustained: bool = False,
    ) -> None:
        super().__init__()
        if eval_every_episodes < 1:
            raise ValueError("eval_every_episodes must be >= 1")
        if n_eval_episodes < 1:
            raise ValueError("n_eval_episodes must be >= 1")
        if best_window < 1:
            raise ValueError("best_window must be >= 1")
        if optimal_steps is not None and optimal_steps < 1:
            raise ValueError("optimal_steps must be >= 1 or None")
        if sustain_evals < 1:
            raise ValueError("sustain_evals must be >= 1")
        if stop_on_sustained and optimal_steps is None:
            raise ValueError("stop_on_sustained requires optimal_steps")
        self.eval_env = eval_env
        self.eval_every_episodes = eval_every_episodes
        self.max_steps = max_steps
        self.n_eval_episodes = n_eval_episodes
        self.best_window = best_window
        self.optimal_steps = optimal_steps
        self.sustain_evals = sustain_evals
        self.stop_on_sustained = stop_on_sustained
        self._train_episode_count = 0
        self.eval_train_episodes: list[int] = []
        self.eval_timesteps: list[int] = []
        self.eval_returns: list[float] = []
        self.eval_steps: list[float] = []
        self.eval_optimal: list[bool] = []
        self._optimal_streak = 0
        self.first_optimal: dict[str, int] | None = None
        self.sustained_optimal: dict[str, int] | None = None
        self._window: deque[dict[str, object]] = deque(maxlen=best_window)

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


def train_trial(
    *,
    trial: int,
    env_id: str,
    seed: int,
    total_timesteps: int,
    delay: int,
    max_steps: int | None,
    trial_dir: Path,
    env_kind: str = "empty",
    subgoal_reward: float = 1.0,
    reward_attribution: str = "arrival",
    wandb_enabled: bool = False,
    wandb_project: str = "delayed-reward",
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_log_every: int = 1000,
    eval_every_episodes: int | None = 20,
    step_penalty: float = 0.0,
    checkpoint_window: int = 100,
    randomize: LayoutRandomization = LayoutRandomization(),
    eval_episodes: int = 1,
    learning_rate: float = 1e-3,
    lr_schedule: str = "constant",
    lr_final: float = 0.0,
    double_dqn: bool = False,
    schedule_timesteps: int | None = None,
    optimal_steps: int | None = None,
    sustain_evals: int = 3,
    stop_on_sustained: bool = False,
) -> None:
    """Train one DQN trial and write its artifacts into ``trial_dir``.

    ``total_timesteps`` is the cap; ``schedule_timesteps`` (default: the cap)
    is the horizon the epsilon and linear-LR schedules anneal over, so raising
    the cap leaves those schedules unchanged. With ``stop_on_sustained`` the
    trial ends at its sustained optimal solve (see ``GreedyEvalCallback``).
    """
    schedule_timesteps = total_timesteps if schedule_timesteps is None else schedule_timesteps
    if not 1 <= schedule_timesteps <= total_timesteps:
        raise ValueError("schedule_timesteps must be in [1, total_timesteps]")
    if optimal_steps is not None and eval_every_episodes is None:
        raise ValueError("optimal_steps needs periodic evals (eval_every_episodes)")
    schedule_fraction = schedule_timesteps / total_timesteps
    if wandb_log_every < 1:
        raise ValueError("wandb_log_every must be >= 1")
    if eval_every_episodes is not None and eval_every_episodes < 1:
        raise ValueError("eval_every_episodes must be >= 1 or None")
    env = _make_env(
        env_kind=env_kind,
        env_id=env_id,
        seed=seed + trial,
        delay=delay,
        subgoal_reward=subgoal_reward,
        max_steps=max_steps,
        step_penalty=step_penalty,
        randomize=randomize,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    config: dict[str, object] = {
        "algorithm": "stable-baselines3-dqn", "env": env_id, "env_kind": env_kind,
        "delay": delay, "trial": trial, "seed": seed + trial,
        "total_timesteps": total_timesteps, "max_steps": max_steps,
        "subgoal_reward": subgoal_reward if env_kind == "keydoor" else None,
        "reward_attribution": reward_attribution,
        "step_penalty": step_penalty if env_kind == "keydoor" else None,
        "layout_randomization": randomize.names() if env_kind == "keydoor" else None,
        "fixed_layout": fixed_layout_source(env_id) if env_kind == "keydoor" else None,
        "eval_episodes": eval_episodes,
        "eval_every_episodes": eval_every_episodes,
        **lr_config(learning_rate, lr_schedule, lr_final),
        "double_dqn": double_dqn,
        **_time_to_optimal_config(schedule_timesteps, optimal_steps, sustain_evals, stop_on_sustained),
    }
    wandb_run = start_wandb_run(
        enabled=wandb_enabled, project=wandb_project, entity=wandb_entity,
        group=wandb_group, name=f"delay-{delay}-trial-{trial}", config=config,
    )
    callback = EpisodeMetricsCallback(
        reward_attribution=reward_attribution, wandb_run=wandb_run, log_every=wandb_log_every
    )
    callbacks: list[BaseCallback] = [callback]
    eval_callback: GreedyEvalCallback | None = None
    if eval_every_episodes is not None:
        eval_env = _make_env(
            env_kind=env_kind,
            env_id=env_id,
            seed=seed + trial,
            delay=delay,
            subgoal_reward=subgoal_reward,
            max_steps=max_steps,
            step_penalty=step_penalty,
            randomize=randomize,
        )
        eval_callback = GreedyEvalCallback(
            eval_env=eval_env,
            eval_every_episodes=eval_every_episodes,
            n_eval_episodes=eval_episodes,
            max_steps=max_steps or 2000,
            best_window=checkpoint_window,
            optimal_steps=optimal_steps,
            sustain_evals=sustain_evals,
            stop_on_sustained=stop_on_sustained,
        )
        callbacks.append(eval_callback)
    policy = MaskedDQNPolicy if env_kind == "keydoor" else "MlpPolicy"
    algorithm = DoubleDQN if double_dqn else DQN
    model = algorithm(
        policy,
        env,
        policy_kwargs={"net_arch": [128, 128]},
        learning_rate=make_lr_schedule(learning_rate, lr_schedule, lr_final, schedule_fraction),
        buffer_size=150_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.995,
        train_freq=4,
        target_update_interval=1_000,
        # 0.4 of the schedule horizon, expressed as SB3's fraction of the cap.
        exploration_fraction=EXPLORATION_FRACTION * schedule_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10.0,
        seed=seed + trial,
        device="auto",
        verbose=0,
    )
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    finally:
        env.close()
        if eval_callback is not None:
            eval_callback.eval_env.close()

    best = eval_callback.best_in_window() if eval_callback is not None else None
    if best is not None:
        model.policy.load_state_dict(best["state_dict"])
        checkpoint_selection: dict[str, object] = {
            "strategy": "best_of_last_n_eval_checkpoints",
            "window": checkpoint_window,
            "selected_train_episode": best["train_episode"],
            "selected_timestep": best["timestep"],
            "selected_eval_return": best["return"],
            "selected_eval_steps": best["steps"],
        }
    else:
        checkpoint_selection = {"strategy": "final_weights"}

    config: dict[str, object] = {
        "algorithm": "stable-baselines3-dqn",
        "env": env_id,
        "env_kind": env_kind,
        "action_masking": env_kind == "keydoor",
        "delay": delay,
        "subgoal_reward": subgoal_reward if env_kind == "keydoor" else None,
        "reward_attribution": reward_attribution,
        "step_penalty": step_penalty if env_kind == "keydoor" else None,
        "layout_randomization": randomize.names() if env_kind == "keydoor" else None,
        "fixed_layout": fixed_layout_source(env_id) if env_kind == "keydoor" else None,
        "eval_episodes": eval_episodes,
        "trial": trial,
        "seed": seed + trial,
        "total_timesteps": total_timesteps,
        "max_steps": max_steps,
        "observation": "fully-observed encoded grid plus normalized agent pose",
        "network": [128, 128],
        **lr_config(learning_rate, lr_schedule, lr_final),
        "double_dqn": double_dqn,
        "eval_every_episodes": eval_every_episodes,
        "buffer_size": 150_000,
        "learning_starts": 1_000,
        "batch_size": 64,
        "gamma": 0.995,
        "train_freq": 4,
        "target_update_interval": 1_000,
        "exploration_fraction": EXPLORATION_FRACTION,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "device": str(model.device),
        "checkpoint_selection": checkpoint_selection,
        **_time_to_optimal_config(schedule_timesteps, optimal_steps, sustain_evals, stop_on_sustained),
    }
    time_to_optimal = (
        eval_callback.time_to_optimal(total_timesteps) if eval_callback is not None else None
    )
    save_metrics(
        trial_dir,
        rewards=callback.rewards,
        steps=callback.steps,
        config=config,
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
            optimal=eval_callback.eval_optimal if optimal_steps is not None else None,
        )
    model.save(trial_dir / "model")
    _save_policy_gif(
        model,
        env_kind=env_kind,
        env_id=env_id,
        seed=seed + trial,
        delay=delay,
        subgoal_reward=subgoal_reward,
        max_steps=max_steps or 200,
        path=trial_dir / "policy.gif",
        step_penalty=step_penalty,
        randomize=randomize,
    )
    recent = callback.rewards[-100:]
    print(
        f"trial={trial} episodes={len(callback.rewards)} "
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


def _save_policy_gif(
    model: DQN,
    *,
    env_kind: str,
    env_id: str,
    seed: int,
    delay: int,
    subgoal_reward: float,
    max_steps: int,
    path: Path,
    step_penalty: float = 0.0,
    randomize: LayoutRandomization = LayoutRandomization(),
) -> None:
    env = _make_env(
        env_kind=env_kind,
        env_id=env_id,
        seed=seed,
        delay=delay,
        subgoal_reward=subgoal_reward,
        render_mode="rgb_array",
        max_steps=max_steps,
        step_penalty=step_penalty,
        randomize=randomize,
    )
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


def add_layout_args(parser: argparse.ArgumentParser) -> None:
    """Add the layout-randomization and eval-episode options shared by the DQN CLIs."""
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
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1,
        help="Greedy episodes averaged per eval; raise it when using --randomize",
    )


LR_SCHEDULES = ("constant", "linear")
EXPLORATION_FRACTION = 0.4


def _time_to_optimal_config(
    schedule_timesteps: int,
    optimal_steps: int | None,
    sustain_evals: int,
    stop_on_sustained: bool,
) -> dict[str, object]:
    """Summary/resume fields for the schedule horizon and optimal-solve criterion."""
    return {
        "schedule_timesteps": schedule_timesteps,
        "optimal_steps": optimal_steps,
        "sustain_evals": sustain_evals if optimal_steps is not None else None,
        "stop_on_sustained": stop_on_sustained,
    }


def add_time_to_optimal_args(parser: argparse.ArgumentParser) -> None:
    """Add the schedule-horizon and optimal-solve options shared by the DQN CLIs."""
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


def time_to_optimal_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Validate the ``add_time_to_optimal_args`` options and return ``train_trial`` kwargs."""
    if args.schedule_timesteps is not None and not 1 <= args.schedule_timesteps <= args.timesteps:
        raise SystemExit("schedule-timesteps must be in [1, timesteps]")
    if args.optimal_steps is not None and args.optimal_steps < 1:
        raise SystemExit("optimal-steps must be >= 1")
    if args.sustain_evals < 1:
        raise SystemExit("sustain-evals must be >= 1")
    if args.optimal_steps is not None and not args.eval_every_episodes:
        raise SystemExit("--optimal-steps needs --eval-every-episodes > 0")
    if args.stop_on_sustained and args.optimal_steps is None:
        raise SystemExit("--stop-on-sustained requires --optimal-steps")
    return {
        "schedule_timesteps": args.schedule_timesteps,
        "optimal_steps": args.optimal_steps,
        "sustain_evals": args.sustain_evals,
        "stop_on_sustained": args.stop_on_sustained,
    }


def make_lr_schedule(
    learning_rate: float, lr_schedule: str, lr_final: float, end_fraction: float = 1.0
) -> float | LinearSchedule:
    """Return SB3's ``learning_rate`` argument: a constant, or a linear anneal.

    ``linear`` decays from ``learning_rate`` at the first update to ``lr_final``
    after ``end_fraction`` of training (SB3 passes the remaining progress,
    1 -> 0) and holds ``lr_final`` after that.
    """
    if not 0 < end_fraction <= 1:
        raise ValueError(f"end_fraction must be in (0, 1], got {end_fraction}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if lr_schedule == "constant":
        return learning_rate
    if lr_schedule == "linear":
        if not 0 <= lr_final <= learning_rate:
            raise ValueError(f"lr_final must be in [0, learning_rate], got {lr_final}")
        return LinearSchedule(learning_rate, lr_final, end_fraction)
    raise ValueError(f"unknown lr_schedule {lr_schedule!r}; choose from {LR_SCHEDULES}")


def lr_config(learning_rate: float, lr_schedule: str, lr_final: float) -> dict[str, object]:
    """Summary/resume fields describing the learning-rate setup."""
    return {
        "learning_rate": learning_rate,
        "lr_schedule": lr_schedule,
        "lr_final": lr_final if lr_schedule == "linear" else None,
    }


def add_lr_args(parser: argparse.ArgumentParser) -> None:
    """Add the learning-rate and learner options shared by the DQN CLIs."""
    parser.add_argument(
        "--double-dqn",
        action="store_true",
        help="Double DQN targets: the online net picks the next action, the target net scores it",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial Adam learning rate")
    parser.add_argument(
        "--lr-schedule",
        choices=LR_SCHEDULES,
        default="constant",
        help=(
            "constant, or linear decay from --learning-rate to --lr-final over "
            "--schedule-timesteps (default --timesteps), then held"
        ),
    )
    parser.add_argument(
        "--lr-final", type=float, default=0.0, help="Final learning rate for --lr-schedule linear"
    )


def lr_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Validate the ``add_lr_args`` options and return them as ``train_trial`` kwargs."""
    try:
        make_lr_schedule(args.learning_rate, args.lr_schedule, args.lr_final)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return {
        "learning_rate": args.learning_rate,
        "lr_schedule": args.lr_schedule,
        "lr_final": args.lr_final,
        "double_dqn": args.double_dqn,
    }


def layout_from_args(args: argparse.Namespace) -> LayoutRandomization:
    """Validate and return the layout randomization chosen by ``add_layout_args`` options."""
    if args.eval_episodes < 1:
        raise SystemExit("eval-episodes must be >= 1")
    randomize = LayoutRandomization.from_names(args.randomize)
    if randomize.any and args.env_kind != "keydoor":
        raise SystemExit("--randomize requires --env-kind keydoor")
    return randomize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-kind", choices=("empty", "keydoor"), default="empty")
    parser.add_argument(
        "--env-id",
        default=None,
        help=f"Defaults to {DEFAULT_ENV_ID!r} (empty) or {KEYDOOR_ENV_ID!r} (keydoor)",
    )
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument(
        "--subgoal-reward",
        type=float,
        default=0.3,
        help="Reward for picking up the key (--env-kind keydoor only)",
    )
    parser.add_argument(
        "--reward-attribution",
        choices=("arrival", "origin"),
        default="arrival",
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
        "--checkpoint-window",
        type=int,
        default=100,
        help=(
            "Save and GIF the best deterministic-eval checkpoint from the last "
            "N eval checkpoints (see --eval-every-episodes), instead of the raw "
            "final-timestep weights."
        ),
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=750_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="delayed-reward")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-log-every", type=int, default=1000)
    parser.add_argument(
        "--eval-every-episodes",
        type=int,
        default=20,
        help=(
            "Run a deterministic (exploration-off) evaluation episode every "
            "N completed training episodes. Pass 0 to disable."
        ),
    )
    add_layout_args(parser)
    add_lr_args(parser)
    add_time_to_optimal_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.trials < 1 or args.timesteps < 1:
        raise SystemExit("trials and timesteps must be >= 1")
    if args.delay < 0 or (args.max_steps is not None and args.max_steps < 1):
        raise SystemExit("delay must be >= 0 and max-steps must be >= 1")
    if args.step_penalty < 0 or args.checkpoint_window < 1:
        raise SystemExit("step-penalty must be >= 0 and checkpoint-window must be >= 1")
    randomize = layout_from_args(args)
    env_id = args.env_id or (KEYDOOR_ENV_ID if args.env_kind == "keydoor" else DEFAULT_ENV_ID)
    run_dir = args.run_dir or create_run_dir(args.results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    delay_dir = create_delay_dir(run_dir, args.delay)
    print(
        f"RUN_DIR {run_dir.resolve()} trials={args.trials} "
        f"timesteps={args.timesteps} delay={args.delay} env_kind={args.env_kind}",
        flush=True,
    )
    for trial in range(args.trials):
        trial_dir = create_trial_dir(delay_dir, trial)
        train_trial(
            trial=trial,
            env_id=env_id,
            seed=args.seed,
            total_timesteps=args.timesteps,
            delay=args.delay,
            max_steps=args.max_steps,
            trial_dir=trial_dir,
            env_kind=args.env_kind,
            subgoal_reward=args.subgoal_reward,
            reward_attribution=args.reward_attribution,
            wandb_enabled=args.wandb, wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity, wandb_group=run_dir.name,
            wandb_log_every=args.wandb_log_every,
            eval_every_episodes=args.eval_every_episodes or None,
            step_penalty=args.step_penalty,
            checkpoint_window=args.checkpoint_window,
            randomize=randomize,
            eval_episodes=args.eval_episodes,
            **lr_from_args(args),
            **time_to_optimal_from_args(args),
        )


if __name__ == "__main__":
    main()
