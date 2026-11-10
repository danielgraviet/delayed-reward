# Repository agent notes

When working on experiments on the GPU box, read [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md)
first. In particular, the normal delayed-reward CLI runs trials sequentially;
use `scripts/parallel_sweep.py` for multi-trial sweeps, check the GPU checkout's
working tree before changes, and preserve existing run directories and local
modifications.

For CPU-bound DQN sweeps on Daytona sandboxes instead of the GPU box, see
[docs/DAYTONA_RUNBOOK.md](docs/DAYTONA_RUNBOOK.md) (`scripts/daytona_sweep.py`).
