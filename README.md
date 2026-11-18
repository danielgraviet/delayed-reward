# delayed-reward

This project measures how much delay between an achievement and its reward
tabular Q-learning and DQN can tolerate on MiniGrid tasks. Each delay is run
across independent trials to see whether the learner still finds a policy that
solves the task.

Learning is judged from periodic greedy (exploration-off) evaluations. A trial
has *solved* the task once three evaluations in a row reach the goal. Every run
writes:

- `time_to_solve.png`: one row per delay, one dot per trial at the training
  timestep it solved the task; hollow marks are trials that never did.
- `success_rate.png`: for each delay, the share of trials whose greedy policy
  reaches the goal at each point in training.
- `trial_timelines.png`: every evaluation of every trial, to spot policies that
  solve the task and then lose it.
- `analysis.json`: the same numbers per delay.

`uv run delayed-reward-inspect RUN_DIR` prints that summary as a table, and
`uv run delayed-reward-replot RUN_DIR` regenerates the charts for an existing run.

## Weights & Biases tracking

Install the project dependencies (`uv sync` when using uv), then authenticate
with `wandb login` or set `WANDB_API_KEY`. Tracking is opt-in; without
`--wandb`, training continues to write local result files as before.

Each trial is a separate W&B run grouped under the local run directory name.
Tabular runs log episode return, episode length, and rolling means. DQN runs
also log Stable-Baselines3 training metrics such as loss, learning rate, and
exploration rate when available. Use `--wandb-project` and `--wandb-entity` to
choose the W&B project and account.

Example parallel tabular sweep:

```sh
uv run delayed-reward-sweep \
  --delay 0 2 5 10 --trials 10 --workers 6 --wandb \
  --wandb-project delayed-reward
```

Example parallel DQN sweep:

```sh
uv run delayed-reward-dqn-sweep \
  --env-kind keydoor --delay 0 2 5 10 --trials 10 --workers 4 --wandb \
  --wandb-project delayed-reward
```

The sequential entrypoints also accept `--wandb`. DQN's
`--wandb-log-every` controls how often training metrics are sampled, in
timesteps (default 1000).

## Scaling DQN sweeps on Daytona (CPU sandboxes)

DQN trials are CPU-bound, so they can be fanned out to Daytona sandboxes with
one trial per sandbox. Results land in a normal local `results/run_*`
directory:

```sh
uv sync --group cloud && export DAYTONA_API_KEY=...
uv run --group cloud delayed-reward-daytona-snapshot   # once per uv.lock change
uv run --group cloud delayed-reward-daytona \
  --env-kind keydoor --delay 0 2 5 10 --trials 10 --wandb --max-sandboxes 40
```

See [docs/DAYTONA_RUNBOOK.md](docs/DAYTONA_RUNBOOK.md) for setup, resume,
cleanup, and the current status/next steps.
