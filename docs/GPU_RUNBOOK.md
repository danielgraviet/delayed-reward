# GPU experiment runbook

Use this runbook for experiments on the Tailscale GPU box. The standard CLI is
sequential: setting `--trials 10` does **not** start parallel workers.

## Connect and locate the checkout

1. Bring up Tailscale on the local machine.
2. Connect with `ssh gpu` (the SSH host alias is `gpu`).
3. The checkout is `/home/dtgraviet/Projects/delayed-reward`.
4. Before running anything, check `git status --short --branch` in that checkout.
   The GPU checkout may contain uncommitted code changes and experiment results.
   Preserve them; do not reset, clean, or overwrite them as part of a run.

## Run sweeps

Run multi-trial sweeps with the repository's parallel launcher, not the ordinary
`python -m delayed_reward` CLI. It keeps all delays and trials in one run
directory, runs independent trials in separate processes, and can resume a run
by skipping trials that already have `summary.json`.

When the local checkout has changes that are not present on the GPU box, copy
the launcher explicitly after ensuring the destination directory exists:

```sh
ssh gpu 'mkdir -p /home/dtgraviet/Projects/delayed-reward/scripts'
scp scripts/parallel_sweep.py gpu:/home/dtgraviet/Projects/delayed-reward/scripts/parallel_sweep.py
```

Example for ten trials at each of the established delay levels:

```sh
cd /home/dtgraviet/Projects/delayed-reward
.venv/bin/python scripts/parallel_sweep.py \
  --env-id MiniGrid-Empty-10x10-v0 \
  --episodes 10000 \
  --trials 10 \
  --workers 6 \
  --delay 0 2 5 10 25 50 100 \
  --seed 0 \
  --log-every 100
```

Choose workers based on available CPU cores and competing GPU-box jobs. Start at
six workers for this tabular workload, inspect CPU and memory use, and lower the
count if the box is shared or memory pressure rises. Do not start multiple full
sweeps to obtain parallelism; parallelize trials within one run directory.

To resume an interrupted run, pass its existing path with `--run-dir`, keeping
the original environment, delays, seed, episode count, and learning settings.
The launcher skips completed trials and fills in missing ones. Do not resume a
run with changed parameters; start a new run directory instead.

## Monitor

The launcher prints the run directory and a completion line as each trial
finishes. Check progress and active workers with:

```sh
pgrep -af 'parallel_sweep.py|resume_delayed_reward_sweep.py'
find results/RUN_DIRECTORY -name summary.json | wc -l
```

There should be one coordinator plus up to the requested number of trial worker
processes. The total expected trial count is `number of delays × trials`.
Completed summaries and `delay_survival.json` / `.png` are stored in the run
directory under `results/`.

## Current 10×10 rerun

The active rerun is `results/run_20261008_134350`, using delays
`0, 2, 5, 10, 25, 50, 100`, ten trials each, and 10,000 episodes per trial.
It was switched from a sequential CLI launch to six parallel workers. The
coordinator is `/tmp/resume_delayed_reward_sweep.py` on the GPU box and its log
is `results/ten_trial_sweep.log`. Keep this run intact and let the coordinator
finish; do not start a duplicate sweep. Future sweeps should use the checked-in
`scripts/parallel_sweep.py` instead of a temporary launcher.
