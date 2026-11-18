# Repository agent notes

When working on experiments on the GPU box, read [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md)
first. In particular, the normal delayed-reward CLI runs trials sequentially;
use `delayed-reward-sweep` (tabular) or `delayed-reward-dqn-sweep` for multi-trial sweeps, check the GPU checkout's
working tree before changes, and preserve existing run directories and local
modifications.

For CPU-bound DQN sweeps on Daytona sandboxes instead of the GPU box, see
[docs/DAYTONA_RUNBOOK.md](docs/DAYTONA_RUNBOOK.md) (`delayed-reward-daytona`).

## Code layout

Each package under `src/delayed_reward/` is a layer, and a layer imports only
from the layers before it:

`protocols`, `tracking` < `results` < `environments` < `config` < `agents` <
`experiments` < `sweep` < `cli`

- `config.py`: `DQNTrialConfig` is the single definition of a DQN trial. Add a
  new setting there (field, validation, `to_dict`), never as a loose kwarg.
- `environments/`: Gymnasium wrappers, env ids (`registry.py`), `factory.py`.
- `agents/tabular/`, `agents/dqn/`: learners, callbacks, model construction.
- `experiments/`: run one trial end to end and write its artifacts.
- `sweep/`: plan/resume many trials, local process pool, Daytona launcher.
- `results/`: file IO and plotting only; never builds environments.
- `cli/`: argparse to config, nothing else. Every CLI is a `[project.scripts]`
  entry point (`uv run delayed-reward-dqn-sweep ...`); there is no `scripts/`.

Work is tracked in Linear: project **Delayed Reward** (team DTGOS,
https://linear.app/dtgos/project/delayed-reward-c5d43b75b48e). Before starting
experiment work, read the project description (standing decisions) and its open
issues. Keep it current: record each run's results on its issue, move issues
through Todo/In Progress/Done, and file new issues for follow-ups.
