# Daytona CPU sweep runbook

The DQN workload (128×128 MLP, batch 64) is bound by env stepping on the CPU,
so trials can run on Daytona CPU sandboxes instead of the GPU box. Each pending
`(delay, trial)` gets its own sandbox. Finished `trial_N/` folders are
downloaded into a local `results/run_*` directory, which uses the same layout
and resume rules as `scripts/parallel_dqn_sweep.py`.

## Status (2026-11-09): pick up here

**Done (uncommitted on `main`):**
- `daytona` SDK 0.231.1 is in the optional `cloud` uv group (`pyproject.toml`, `uv.lock`).
- `scripts/daytona_snapshot.py` builds the snapshot, and `scripts/daytona_sweep.py` is the launcher.
- `scripts/parallel_dqn_sweep.py` gained `--trial-ids` and `--no-plot`, and its
  planning moved into `resolve_sweep()` / `plan_tasks()`, which the Daytona
  launcher reuses. Local behaviour is unchanged.
- Checked locally only: `--help`, `--dry-run`, argument forwarding, and the code tarball (46 files, ~39 KiB).

**Verified on real sandboxes (2026-11-10):** snapshot `delayed-reward-cpu-2c4g-357557123c`
builds (torch 2.14.1+cpu). The 2-trial smoke test (steps 1–2 below) passes end to
end: `summary.json`, `device == "cpu"`, `daytona.log`, and the plot. Also verified:
resume skips finished trials, a remote failure writes its log to `_daytona_logs/`
without retrying, `--keep-failed` keeps the sandbox, `--cleanup`'s label filter
finds live sandboxes, and SIGINT/SIGTERM/SIGHUP delete every live sandbox (deletes
are async and finish about 20 s later). Underscores in sandbox names and `ttl_minutes`
are accepted. The launcher doesn't read `.env`, so run `set -a; source .env; set +a`
first. Note: `uv run` doesn't forward a SIGINT that's sent with `kill`, so signal
the Python child (a terminal Ctrl-C is fine). Steps 3–5 are still open.

**Timing (2026-11-10):** a 500k-step keydoor-hallway trial takes 58–76 min on a
2-vCPU sandbox. `run_20261110_171314` ran 10 trials in parallel, about 76 min
wall clock. W&B works from `~/.netrc`, with no extra setup.

**Original checklist:**
1. `export DAYTONA_API_KEY=...`, then build the snapshot:
   `uv run --group cloud scripts/daytona_snapshot.py`
   (first real check of the SDK's image build: the CPU torch index and requirement pins).
2. Smoke test, 2 short trials end to end:
   ```sh
   uv run --group cloud scripts/daytona_sweep.py --env-kind keydoor --delay 0 \
     --trials 2 --timesteps 20000 --max-sandboxes 2
   ```
   Check that `trial_*/summary.json` downloaded, `config.device == "cpu"`, and
   that `daytona.log` is present. Run `--cleanup` afterwards to confirm no
   sandboxes are left.
3. Benchmark: time the 20k-step smoke trial against the same settings on the GPU
   box (`ssh gpu`, `parallel_dqn_sweep.py ... --timesteps 20000 --workers 1`).
   If a 2-vCPU sandbox is much slower, try `--cpu 1` (more sandboxes per quota)
   or `--cpu 4`. Each size gets its own snapshot, built automatically.
4. Check the Tier 3 vCPU quota in the Daytona dashboard and size
   `--max-sandboxes` so that `max-sandboxes × cpu` stays under it.
5. Full sweep, e.g. the delays from `run_20261102_164946` (keydoor hallway,
   `--step-penalty 0.001`, 500k timesteps, 10 trials) with `--wandb --max-sandboxes 80`.
   Start it in a new run directory. Don't resume GPU runs on Daytona.

**Open questions / things to verify on first real run:**
- Unverified SDK behaviour: whether `ttl_minutes` and the `labels` filter in
  `--cleanup` behave as documented, and whether underscores in sandbox names are
  allowed (they are replaced with `-`).
- W&B inside sandboxes: the key is passed as a plain env var. A Daytona org
  Secret would be safer if the key format check allows it.
- Only the DQN launcher has a Daytona path. Tabular `parallel_sweep.py` trials are
  short and probably don't need one.
- The README's DQN example uses the older `scripts/dqn_parallel_sweep.py`.
  `parallel_dqn_sweep.py` is the newer one (step penalty, checkpoint window,
  evals) and the one used for the latest runs. Consider removing the older script.

## One-time setup

```sh
uv sync --group cloud                    # installs the daytona SDK (optional group)
export DAYTONA_API_KEY=...               # https://app.daytona.io/dashboard/keys
uv run --group cloud scripts/daytona_snapshot.py   # build CPU snapshot (~minutes, once)
```

The snapshot holds Python 3.13, CPU-only torch, and the exact versions from
`uv.lock` (no CUDA wheels). Its name contains a hash of the locked deps and the
resources (`delayed-reward-cpu-2c4g-<hash>`), so after changing `uv.lock` the
next sweep builds a new snapshot automatically. Project source is **not** baked
in. It is uploaded to each sandbox at launch, so code edits take effect without
a rebuild. Use `--rebuild` to force a rebuild.

## Run

```sh
uv run --group cloud scripts/daytona_sweep.py \
  --env-kind keydoor --delay 0 1 2 5 10 25 50 100 --trials 10 \
  --step-penalty 0.001 --wandb --max-sandboxes 80
```

- All `parallel_dqn_sweep.py` flags are forwarded. `--run-dir`, `--delay`,
  `--trial-ids`, and `--workers` are set per sandbox by the launcher.
- Resume with `--run-dir results/run_X`. Completed trials are skipped, and
  settings mismatches are refused.
- `--cpu/--memory/--disk` pick the snapshot (default 2 vCPU / 4 GiB / 10 GiB).
  Total vCPU in use = `max-sandboxes × cpu`, which must fit within the org quota.
- W&B: uses `WANDB_API_KEY`, or the `~/.netrc` entry for `WANDB_BASE_URL`
  (default `https://api.forge.coreweave.com`).
- `--dry-run` prints the plan. `--spot` uses cheaper preemptible sandboxes, and
  evicted trials are restarted from scratch.

## Safety

- Sandboxes have auto-stop disabled, because training generates no sandbox
  activity for hours, and a hard `--ttl-hours` cap (default 24) so nothing
  bills forever.
- Ctrl-C cancels trials and deletes every live sandbox. Stray sandboxes left by a
  crashed launcher can be removed with `scripts/daytona_sweep.py --cleanup`
  (deletes everything labelled `app=delayed-reward`).
- Failed trial logs go to `RUN_DIR/_daytona_logs/`. `--keep-failed` leaves
  those sandboxes up for debugging (delete them afterwards).
- Sandboxes record `device: cpu`. Don't mix them with GPU-box trials in one run
  directory.
