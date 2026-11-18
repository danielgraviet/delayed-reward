# Repository agent notes

## Running experiments

DQN sweeps run on Daytona CPU sandboxes, one trial per sandbox: see
[docs/DAYTONA_RUNBOOK.md](docs/DAYTONA_RUNBOOK.md) (`delayed-reward-daytona`).
For small local runs, use the parallel launchers, `delayed-reward-dqn-sweep` and
`delayed-reward-sweep` (tabular); the `delayed-reward-dqn` and `delayed-reward`
CLIs run trials one after another.

Every sweep keeps all delays and trials in one run directory. To resume an
interrupted run, pass `--run-dir` with the original settings: finished trials are
skipped, and changed settings are refused. To change settings, start a new run
directory. Never delete or overwrite existing run directories under `results/`.

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

## Checks

Run `make check` before calling work done; `make fix` applies formatting and
safe lint fixes. It runs ruff (format + lint), pyright, import-linter (the layer
order above) and pytest in parallel. `tests/test_codebase_rules.py` enforces the
house rules no linter covers: no `#` comments except tool pragmas (put intent in
names and docstrings), and no new markdown outside `README.md`, `AGENTS.md`,
`TODO.md` and `docs/`. Fix the code, not the rule.

Verification is live, not mocked: tests run real envs, real SB3 models and the
real CLIs as subprocesses (`tests/test_cli_live.py`); patching and fake objects
are banned by the rules test. Configs and env wrappers are fuzzed with
Hypothesis (deterministic in `make check`; `make fuzz` runs a long random
search). When a fuzz run finds a bug, add the failing input as an `@example`.
`make smoke-daytona` runs one real sandbox trial end to end (needs `.env`).

Work is tracked in Linear: project **Delayed Reward** (team DTGOS,
https://linear.app/dtgos/project/delayed-reward-c5d43b75b48e). Before starting
experiment work, read the project description (standing decisions) and its open
issues. Keep it current: record each run's results on its issue, move issues
through Todo/In Progress/Done, and file new issues for follow-ups.
