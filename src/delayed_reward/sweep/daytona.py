"""Run planned DQN sweep trials on Daytona CPU sandboxes, one trial per sandbox.

Each sandbox gets the project source as a tarball, runs
``python -m delayed_reward.cli.dqn_sweep`` for a single ``(delay, trial)``, and
the finished ``trial_N/`` folder is downloaded into the local run directory.
"""

from __future__ import annotations

import io
import netrc
import os
import random
import shlex
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[3]
LABEL_APP = "delayed-reward"
NEVER_AUTO_STOP = 0
OVERRIDDEN_OPTIONS = {"--run-dir", "--delay", "--trial-ids", "--workers", "--results-dir"}


def strip_overridden(argv: list[str]) -> list[str]:
    """Drop the sweep options the launcher sets per sandbox (see ``OVERRIDDEN_OPTIONS``)."""
    out: list[str] = []
    skipping = False
    for token in argv:
        name = token.split("=", 1)[0]
        if name in OVERRIDDEN_OPTIONS:
            skipping = "=" not in token
            continue
        if skipping and not token.startswith("-"):
            continue
        skipping = False
        out.append(token)
    return out


def wandb_env() -> dict[str, str]:
    """W&B credentials to forward into sandboxes, from the environment or ``~/.netrc``."""
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
    """Package the project source (no results, caches, or venv)."""
    buf = io.BytesIO()

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if "__pycache__" in parts or info.name.endswith(".pyc"):
            return None
        return info

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in ("src/delayed_reward", "pyproject.toml"):
            tar.add(REPO_ROOT / rel, arcname=rel, filter=keep)
    return buf.getvalue()


class TrialFailedError(Exception):
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

    def _create(self, delay: int, trial: int):
        from daytona import CreateSandboxFromSnapshotParams, DaytonaRateLimitError

        params = CreateSandboxFromSnapshotParams(
            snapshot=self.snapshot,
            name=f"dr-{self.run_dir.name}-d{delay}-t{trial}-{random.randrange(16**4):04x}".lower().replace("_", "-"),
            labels={"app": LABEL_APP, "run": self.run_dir.name, "delay": str(delay), "trial": str(trial)},
            env_vars={**self.env_vars, "OMP_NUM_THREADS": str(self.dargs.cpu)},
            auto_stop_interval=NEVER_AUTO_STOP,
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
        """Delete once: only the caller that pops it from ``live`` deletes, so cleanups never race (409s)."""
        from daytona import DaytonaConflictError, DaytonaNotFoundError

        with self.lock:
            if self.live.pop(sandbox.id, None) is None:
                return
        for attempt in range(6):
            try:
                self.daytona.delete(sandbox, timeout=120)
                return
            except DaytonaNotFoundError:
                return
            except DaytonaConflictError:
                time.sleep(5 * (attempt + 1))
            except Exception as exc:
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

    def run_trial(self, delay: int, trial: int) -> Path:
        last_exc: Exception | None = None
        for attempt in range(self.dargs.retries + 1):
            if self.stop.is_set():
                raise RuntimeError("cancelled")
            try:
                return self._run_once(delay, trial)
            except TrialFailedError:
                raise
            except Exception as exc:
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

            argv = " ".join(shlex.quote(a) for a in self.sweep_argv)
            command = (
                f"cd {repo} && PYTHONPATH=src python -m delayed_reward.cli.dqn_sweep {argv} "
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
                except Exception:
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
                raise TrialFailedError(f"exit={status.exit_code}, log={log_path}\n  " + "\n  ".join(tail))

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


def cleanup() -> None:
    from daytona import Daytona, ListSandboxesQuery

    daytona = Daytona()
    sandboxes = list(daytona.list(ListSandboxesQuery(labels={"app": LABEL_APP})))
    print(f"Deleting {len(sandboxes)} sandbox(es) labelled app={LABEL_APP}", flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda s: daytona.delete(s, timeout=120), sandboxes))
