# Repository agent notes

When working on experiments on the GPU box, read [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md)
first. In particular, the normal delayed-reward CLI runs trials sequentially;
use `scripts/parallel_sweep.py` for multi-trial sweeps, check the GPU checkout's
working tree before changes, and preserve existing run directories and local
modifications.

For CPU-bound DQN sweeps on Daytona sandboxes instead of the GPU box, see
[docs/DAYTONA_RUNBOOK.md](docs/DAYTONA_RUNBOOK.md) (`scripts/daytona_sweep.py`).

Work is tracked in Linear: project **Delayed Reward** (team DTGOS,
https://linear.app/dtgos/project/delayed-reward-c5d43b75b48e). Before starting
experiment work, read the project description (standing decisions) and its open
issues. Keep it current: record each run's results on its issue, move issues
through Todo/In Progress/Done, and file new issues for follow-ups.
