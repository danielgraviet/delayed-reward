# delayed-reward

This project measures how much delay between reaching the goal and receiving
reward tabular Q-learning can tolerate. Each delay is run across independent
trials to see whether the learner still discovers a successful policy.

Every run's `delay_survival.png` shows, by training episode, the percentage of
trials at each delay that have passed. A trial passes when its rolling mean
return reaches 0.5 over 100 episodes. The chart makes it easy to compare both
how quickly each delay is learned and how many trials learn it by the end.

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
uv run python scripts/parallel_sweep.py \
  --delay 0 2 5 10 --trials 10 --workers 6 --wandb \
  --wandb-project delayed-reward
```

Example parallel DQN sweep:

```sh
uv run python scripts/dqn_parallel_sweep.py \
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
uv run --group cloud scripts/daytona_snapshot.py          # once per uv.lock change
uv run --group cloud scripts/daytona_sweep.py \
  --env-kind keydoor --delay 0 2 5 10 --trials 10 --wandb --max-sandboxes 40
```

See [docs/DAYTONA_RUNBOOK.md](docs/DAYTONA_RUNBOOK.md) for setup, resume,
cleanup, and the current status/next steps.
