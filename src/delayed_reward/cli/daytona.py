"""Fan out DQN sweep trials to Daytona CPU sandboxes, one trial per sandbox.

Accepts every ``delayed-reward-dqn-sweep`` argument (env, delays, trials,
timesteps, wandb, --run-dir, ...) plus the Daytona options below. Planning,
resume, and settings checks are the same as the local launcher: trials with a
matching ``summary.json`` in the local run directory are skipped. Each pending
trial runs ``delayed_reward.cli.dqn_sweep --trial-ids N`` in its own sandbox, and the
finished ``trial_N/`` folder is downloaded into the local run directory.

Sandboxes run on CPU (``device`` is recorded as ``cpu`` in each summary); keep a
run directory to one kind of machine so timings stay comparable.

Examples (needs DAYTONA_API_KEY):
    uv run --group cloud delayed-reward-daytona --env-kind keydoor \\
        --delay 0 1 2 5 10 25 50 100 --trials 10 --wandb --max-sandboxes 80
    uv run --group cloud delayed-reward-daytona ... --run-dir results/run_X   # resume
    uv run --group cloud delayed-reward-daytona --cleanup   # delete stray sandboxes
"""

from __future__ import annotations

import argparse
import signal
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from delayed_reward.cli import dqn_sweep
from delayed_reward.sweep.daytona import (
    LABEL_APP,
    Launcher,
    TrialFailedError,
    cleanup,
    strip_overridden,
    wandb_env,
)
from delayed_reward.sweep.daytona_snapshot import ensure_snapshot
from delayed_reward.sweep.dqn import summarize_time_to_optimal
from delayed_reward.sweep.trials import plan_tasks, write_run_overview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    group = parser.add_argument_group("daytona")
    group.add_argument("-h", "--help", action="store_true")
    group.add_argument("--max-sandboxes", type=int, default=40, help="Concurrent sandboxes")
    group.add_argument("--cpu", type=int, default=2, help="vCPUs per sandbox (selects snapshot)")
    group.add_argument("--memory", type=int, default=4, help="GiB RAM per sandbox")
    group.add_argument("--disk", type=int, default=10, help="GiB disk per sandbox")
    group.add_argument("--ttl-hours", type=float, default=24.0, help="Hard sandbox lifetime cap")
    group.add_argument("--poll-seconds", type=float, default=60.0)
    group.add_argument("--retries", type=int, default=2, help="Re-runs per trial after infra errors")
    group.add_argument("--spot", action="store_true", help="Use spot sandboxes (evictions are retried)")
    group.add_argument("--keep-failed", action="store_true", help="Don't delete sandboxes of failed trials")
    group.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    group.add_argument("--cleanup", action="store_true", help=f"Delete all sandboxes labelled app={LABEL_APP}")
    return parser


def _raise_interrupt(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(signal.Signals(signum).name)


def _route_termination_to_interrupt() -> None:
    """Send SIGTERM/SIGHUP (kill, closed SSH session) through the Ctrl-C path.

    That path deletes every live sandbox, instead of leaving them billing until ``--ttl-hours``.
    """
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _raise_interrupt)


def main() -> None:
    dparser = build_parser()
    dargs, sweep_argv = dparser.parse_known_args()
    if dargs.help:
        dparser.print_help()
        print("\nsweep options (forwarded to delayed-reward-dqn-sweep):")
        dqn_sweep.build_parser().print_help()
        return
    if dargs.cleanup:
        cleanup()
        return
    if dargs.max_sandboxes < 1:
        raise SystemExit("max-sandboxes must be >= 1")

    sweep_args = dqn_sweep.build_parser().parse_args(sweep_argv)
    if dargs.dry_run and sweep_args.run_dir is None:
        n_trials = len(set(sweep_args.trial_ids or range(sweep_args.trials)))
        print(
            f"dry run: new run directory, {len(set(sweep_args.delay)) * n_trials} trials, "
            f"remote args: {' '.join(strip_overridden(sweep_argv))}"
        )
        return
    template, delays, trial_ids, run_dir = dqn_sweep.resolve_sweep(sweep_args)
    pending_configs, skipped = plan_tasks(template, run_dir, delays, trial_ids)
    pending = [(config.env.delay, config.trial) for config in pending_configs]

    print(f"RUN_DIR {run_dir.resolve()}", flush=True)
    print(
        f"ENV {template.env.env_id} DELAYS {delays} TRIALS {sweep_args.trials} "
        f"SANDBOXES {min(dargs.max_sandboxes, len(pending))}x{dargs.cpu}vCPU "
        f"PENDING {len(pending)} SKIPPED {skipped}",
        flush=True,
    )
    if dargs.dry_run or not pending:
        return

    env_vars = wandb_env() if sweep_args.wandb else {}
    snapshot = ensure_snapshot(dargs.cpu, dargs.memory, dargs.disk)
    launcher = Launcher(dargs, strip_overridden(sweep_argv), run_dir, snapshot, env_vars)

    _route_termination_to_interrupt()

    failures: list[tuple[int, int, BaseException]] = []
    pool = ThreadPoolExecutor(max_workers=dargs.max_sandboxes)
    futures = {pool.submit(launcher.run_trial, d, t): (d, t) for d, t in pending}
    try:
        remaining = set(futures)
        while remaining:
            done, remaining = wait(remaining, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                delay, trial = futures[future]
                exc = future.exception()
                if exc is not None:
                    failures.append((delay, trial, exc))
                    print(f"FAILED delay={delay} trial={trial}: {exc}", flush=True)
                    if not isinstance(exc, TrialFailedError):
                        traceback.print_exception(exc)
    except KeyboardInterrupt:
        print("\nInterrupted: cancelling trials and deleting sandboxes...", flush=True)
        launcher.stop.set()
        for future in futures:
            future.cancel()
        raise SystemExit(130) from None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        launcher.delete_all_live()

    if not sweep_args.no_plot:
        write_run_overview(run_dir)
    summarize_time_to_optimal(run_dir)
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed; resume with --run-dir {run_dir}")
    print(f"All DQN trials finished under {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
