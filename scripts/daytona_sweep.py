#!/usr/bin/env python3
"""Fan out DQN sweep trials to Daytona CPU sandboxes, one trial per sandbox.

Accepts every ``scripts/parallel_dqn_sweep.py`` argument (env, delays, trials,
timesteps, wandb, --run-dir, ...) plus the Daytona options below. Planning,
resume, and settings checks are the same as the local launcher: trials with a
matching ``summary.json`` in the local run directory are skipped. Each pending
trial runs ``parallel_dqn_sweep.py --trial-ids N`` in its own sandbox, and the
finished ``trial_N/`` folder is downloaded into the local run directory.

Sandboxes run on CPU (``device`` is recorded as ``cpu`` in each summary), so
don't mix these trials with GPU-box trials in one run directory.

Examples (needs DAYTONA_API_KEY):
    uv run --group cloud scripts/daytona_sweep.py --env-kind keydoor \\
        --delay 0 1 2 5 10 25 50 100 --trials 10 --wandb --max-sandboxes 80
    uv run --group cloud scripts/daytona_sweep.py ... --run-dir results/run_X   # resume
    uv run --group cloud scripts/daytona_sweep.py --cleanup   # delete stray sandboxes
"""

from __future__ import annotations

import argparse
import io
import netrc
import os
import random
import signal
import sys
import tarfile
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import urlparse

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

import parallel_dqn_sweep  # noqa: E402
from daytona_snapshot import ensure_snapshot  # noqa: E402

LABEL_APP = "delayed-reward"
# Options whose value the launcher sets per sandbox; user values are replaced.
_OVERRIDDEN = {"--run-dir", "--delay", "--trial-ids", "--workers", "--results-dir"}


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


def _strip_overridden(argv: list[str]) -> list[str]:
    out: list[str] = []
    skipping = False
    for token in argv:
        name = token.split("=", 1)[0]
        if name in _OVERRIDDEN:
            skipping = "=" not in token
            continue
        if skipping and not token.startswith("-"):
            continue
        skipping = False
        out.append(token)
    return out


def _wandb_env() -> dict[str, str]:
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.forge.coreweave.com")
    env = {"WANDB_BASE_URL": base_url}
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        host = urlparse(base_url).hostname or base_url
        try:
            auth = netrc.netrc().authenticators(host)
        except (FileNotFoundError, netrc.NetrcParseError):
            auth = None
        key = auth[2] if auth else None
    if not key:
        raise SystemExit(f"--wandb needs WANDB_API_KEY or a ~/.netrc entry for {base_url}")
    env["WANDB_API_KEY"] = key
    for name in ("WANDB_ENTITY", "WANDB_PROJECT"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    return env


def _code_tarball() -> bytes:
    """Package the project source and scripts (no results, caches, or venv)."""
    buf = io.BytesIO()

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if "__pycache__" in parts or info.name.endswith(".pyc"):
            return None
        return info

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in ("src/delayed_reward", "scripts", "pyproject.toml"):
            tar.add(REPO_ROOT / rel, arcname=rel, filter=keep)
    return buf.getvalue()


class TrialFailed(Exception):
    """The training command itself failed; retrying will not help."""


class Launcher:
    def __init__(self, dargs, sweep_argv, run_dir: Path, snapshot: str, env_vars: dict[str, str]):
        from daytona import Daytona, DaytonaConfig

        self.dargs = dargs
        self.sweep_argv = sweep_argv
        self.run_dir = run_dir
        self.snapshot = snapshot
        self.env_vars = env_vars
        self.code = _code_tarball()
        self.daytona = Daytona(DaytonaConfig(connection_pool_maxsize=max(10, dargs.max_sandboxes * 2)))
        self.stop = threading.Event()
        self.live: dict[str, object] = {}
        self.lock = threading.Lock()

    # -- sandbox lifecycle -------------------------------------------------
    def _create(self, delay: int, trial: int):
        from daytona import CreateSandboxFromSnapshotParams, DaytonaRateLimitError

        params = CreateSandboxFromSnapshotParams(
            snapshot=self.snapshot,
            name=f"dr-{self.run_dir.name}-d{delay}-t{trial}-{random.randrange(16**4):04x}"
            .lower()
            .replace("_", "-"),
            labels={"app": LABEL_APP, "run": self.run_dir.name, "delay": str(delay), "trial": str(trial)},
            env_vars={**self.env_vars, "OMP_NUM_THREADS": str(self.dargs.cpu)},
            auto_stop_interval=0,  # training produces no sandbox events for hours
            ttl_minutes=int(self.dargs.ttl_hours * 60),
            spot=self.dargs.spot or None,
        )
        for attempt in range(8):
            try:
                sandbox = self.daytona.create(params, timeout=600)
            except DaytonaRateLimitError:
                time.sleep(min(120, 5 * 2**attempt) + random.random() * 5)
                continue
            with self.lock:
                self.live[sandbox.id] = sandbox
            return sandbox
        raise RuntimeError("sandbox creation kept hitting rate limits")

    def _delete(self, sandbox) -> None:
        from daytona import DaytonaConflictError, DaytonaNotFoundError

        # Only the caller that removes the sandbox from `live` deletes it, so the
        # worker's cleanup and delete_all_live() never race (409 conflicts).
        with self.lock:
            if self.live.pop(sandbox.id, None) is None:
                return
        for attempt in range(6):
            try:
                self.daytona.delete(sandbox, timeout=120)
                return
            except DaytonaNotFoundError:
                return
            except DaytonaConflictError:  # state change in progress (e.g. still starting)
                time.sleep(5 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001
                print(f"WARN failed to delete sandbox {sandbox.id}: {type(exc).__name__}: {exc}", flush=True)
                return
        print(f"WARN sandbox {sandbox.id} still busy; run --cleanup later", flush=True)

    def delete_all_live(self) -> None:
        with self.lock:
            sandboxes = list(self.live.values())
        if sandboxes:
            print(f"Deleting {len(sandboxes)} live sandbox(es)...", flush=True)
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(self._delete, sandboxes))

    # -- one trial ---------------------------------------------------------
    def run_trial(self, delay: int, trial: int) -> Path:
        last_exc: Exception | None = None
        for attempt in range(self.dargs.retries + 1):
            if self.stop.is_set():
                raise RuntimeError("cancelled")
            try:
                return self._run_once(delay, trial)
            except TrialFailed:
                raise
            except Exception as exc:  # noqa: BLE001 - infra errors (eviction, network) are retried
                last_exc = exc
                if self.stop.is_set():
                    raise
                print(
                    f"RETRY delay={delay} trial={trial} attempt={attempt + 1}: {exc!r}",
                    flush=True,
                )
                time.sleep(10 + random.random() * 20)
        raise RuntimeError(f"giving up after {self.dargs.retries + 1} attempts") from last_exc

    def _run_once(self, delay: int, trial: int) -> Path:
        from daytona import SessionExecuteRequest

        sandbox = self._create(delay, trial)
        failed = False
        try:
            home = sandbox.get_work_dir().rstrip("/")
            repo, remote_run = f"{home}/repo", f"{home}/results/{self.run_dir.name}"
            sandbox.fs.upload_file(self.code, f"{home}/code.tar.gz")
            setup = sandbox.process.exec(
                f"mkdir -p {repo} {remote_run} && tar xzf {home}/code.tar.gz -C {repo}", timeout=120
            )
            if setup.exit_code != 0:
                raise RuntimeError(f"setup failed: {setup.result}")

            argv = " ".join(_shell_quote(a) for a in self.sweep_argv)
            command = (
                f"cd {repo} && python scripts/parallel_dqn_sweep.py {argv} "
                f"--run-dir {remote_run} --delay {delay} --trial-ids {trial} --workers 1 --no-plot "
                f"> {home}/trial.log 2>&1"
            )
            sandbox.process.create_session("trial")
            started = sandbox.process.execute_session_command(
                "trial", SessionExecuteRequest(command=command, run_async=True)
            )
            print(f"STARTED delay={delay} trial={trial} sandbox={sandbox.id}", flush=True)
            t0 = time.monotonic()
            poll_errors = 0
            while True:
                if self.stop.wait(self.dargs.poll_seconds):
                    raise RuntimeError("cancelled")
                try:
                    status = sandbox.process.get_session_command("trial", started.cmd_id)
                except Exception:  # noqa: BLE001 - tolerate transient API/network blips
                    poll_errors += 1
                    if poll_errors >= 15:
                        raise
                    continue
                poll_errors = 0
                if status.exit_code is not None:
                    break

            log = sandbox.fs.download_file(f"{home}/trial.log") or b""
            if status.exit_code != 0:
                failed = True
                log_dir = self.run_dir / "_daytona_logs"
                log_dir.mkdir(exist_ok=True)
                log_path = log_dir / f"delay_{delay}_trial_{trial}.log"
                log_path.write_bytes(log)
                tail = log.decode(errors="replace").strip().splitlines()[-5:]
                raise TrialFailed(f"exit={status.exit_code}, log={log_path}\n  " + "\n  ".join(tail))

            pack = sandbox.process.exec(
                f"tar czf {home}/trial.tgz -C {remote_run}/delay_{delay} trial_{trial}", timeout=600
            )
            if pack.exit_code != 0:
                raise RuntimeError(f"packing results failed: {pack.result}")
            data = sandbox.fs.download_file(f"{home}/trial.tgz")
            delay_dir = self.run_dir / f"delay_{delay}"
            delay_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(delay_dir, filter="data")
            trial_dir = delay_dir / f"trial_{trial}"
            (trial_dir / "daytona.log").write_bytes(log)
            if not (trial_dir / "summary.json").exists():
                raise RuntimeError("downloaded trial has no summary.json")
            minutes = (time.monotonic() - t0) / 60
            print(f"DONE delay={delay} trial={trial} {trial_dir} ({minutes:.1f} min)", flush=True)
            return trial_dir
        finally:
            if failed and self.dargs.keep_failed:
                with self.lock:
                    self.live.pop(sandbox.id, None)
                print(f"KEPT sandbox {sandbox.id} for delay={delay} trial={trial}", flush=True)
            else:
                self._delete(sandbox)


def _shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def cleanup() -> None:
    from daytona import Daytona, ListSandboxesQuery

    daytona = Daytona()
    sandboxes = list(daytona.list(ListSandboxesQuery(labels={"app": LABEL_APP})))
    print(f"Deleting {len(sandboxes)} sandbox(es) labelled app={LABEL_APP}", flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda s: daytona.delete(s, timeout=120), sandboxes))


def main() -> None:
    dparser = build_parser()
    dargs, sweep_argv = dparser.parse_known_args()
    if dargs.help:
        dparser.print_help()
        print("\nsweep options (forwarded to parallel_dqn_sweep.py):")
        parallel_dqn_sweep.build_parser().print_help()
        return
    if dargs.cleanup:
        cleanup()
        return
    if dargs.max_sandboxes < 1:
        raise SystemExit("max-sandboxes must be >= 1")

    sweep_args = parallel_dqn_sweep.build_parser().parse_args(sweep_argv)
    if dargs.dry_run and sweep_args.run_dir is None:
        n_trials = len(set(sweep_args.trial_ids or range(sweep_args.trials)))
        print(
            f"dry run: new run directory, {len(set(sweep_args.delay)) * n_trials} trials, "
            f"remote args: {' '.join(_strip_overridden(sweep_argv))}"
        )
        return
    delays, trial_ids, env_id, run_dir = parallel_dqn_sweep.resolve_sweep(sweep_args)
    pending, skipped = parallel_dqn_sweep.plan_tasks(sweep_args, env_id, run_dir, delays, trial_ids)

    print(f"RUN_DIR {run_dir.resolve()}", flush=True)
    print(
        f"ENV {env_id} DELAYS {delays} TRIALS {sweep_args.trials} "
        f"SANDBOXES {min(dargs.max_sandboxes, len(pending))}x{dargs.cpu}vCPU "
        f"PENDING {len(pending)} SKIPPED {skipped}",
        flush=True,
    )
    if dargs.dry_run or not pending:
        return

    env_vars = _wandb_env() if sweep_args.wandb else {}
    snapshot = ensure_snapshot(dargs.cpu, dargs.memory, dargs.disk)
    launcher = Launcher(dargs, _strip_overridden(sweep_argv), run_dir, snapshot, env_vars)

    # Route SIGTERM/SIGHUP (kill, closed SSH session) through the Ctrl-C path so
    # live sandboxes are deleted instead of billing until --ttl-hours.
    def _interrupt(signum, _frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)

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
                    if not isinstance(exc, TrialFailed):
                        traceback.print_exception(exc)
    except KeyboardInterrupt:
        print("\nInterrupted: cancelling trials and deleting sandboxes...", flush=True)
        launcher.stop.set()
        for future in futures:
            future.cancel()
        raise SystemExit(130)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        launcher.delete_all_live()

    has_metrics = any(run_dir.glob("delay_*/trial_*/metrics.csv"))
    if has_metrics and not sweep_args.no_plot:
        from delayed_reward.results.plotting import plot_delay_survival

        plot_path = plot_delay_survival(
            run_dir, smooth_window=100, onset_threshold=sweep_args.onset_threshold
        )
        print(f"PLOT {plot_path.resolve()}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed; resume with --run-dir {run_dir}")
    print(f"All DQN trials finished under {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
