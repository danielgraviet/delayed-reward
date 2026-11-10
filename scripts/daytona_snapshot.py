#!/usr/bin/env python3
"""Build (or reuse) a CPU-only Daytona snapshot with this project's pinned dependencies.

The snapshot contains Python and the locked third-party packages only. Project
source is uploaded per sandbox by ``scripts/daytona_sweep.py``, so code changes
do not require rebuilding. The snapshot name embeds a hash of the pinned
requirements and resources, so changing ``uv.lock`` produces a new snapshot.

Usage (needs DAYTONA_API_KEY):
    uv run --group cloud scripts/daytona_snapshot.py            # build if missing
    uv run --group cloud scripts/daytona_snapshot.py --print-name
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.13"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# GPU-only wheels; the CPU torch build does not need them.
_EXCLUDED_PREFIXES = ("nvidia-", "cuda-", "triton", "torch==")


def locked_requirements() -> tuple[str, list[str]]:
    """Return (torch version, other pinned requirement lines) from uv.lock."""
    exported = subprocess.run(
        [
            "uv", "export", "--frozen", "--no-dev", "--no-hashes",
            "--no-emit-project", "--format", "requirements-txt",
        ],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    torch_version = None
    lines = []
    for raw in exported.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("torch=="):
            torch_version = line.split("==", 1)[1].split(";", 1)[0].strip()
        if line.startswith(_EXCLUDED_PREFIXES):
            continue
        lines.append(line)
    if torch_version is None:
        raise SystemExit("torch not found in uv export output")
    return torch_version, lines


def snapshot_name(cpu: int, memory: int, disk: int) -> str:
    torch_version, lines = locked_requirements()
    digest = hashlib.sha256(
        "\n".join([PYTHON_VERSION, torch_version, f"{cpu}/{memory}/{disk}", *lines]).encode()
    ).hexdigest()[:10]
    return f"delayed-reward-cpu-{cpu}c{memory}g-{digest}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", type=int, default=2, help="vCPUs per sandbox")
    parser.add_argument("--memory", type=int, default=4, help="GiB RAM per sandbox")
    parser.add_argument("--disk", type=int, default=10, help="GiB disk per sandbox")
    parser.add_argument("--print-name", action="store_true", help="Print the snapshot name and exit")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild an existing snapshot")
    return parser


def ensure_snapshot(cpu: int, memory: int, disk: int, *, rebuild: bool = False) -> str:
    from daytona import CreateSnapshotParams, Daytona, DaytonaNotFoundError, Image, Resources

    name = snapshot_name(cpu, memory, disk)
    daytona = Daytona()
    try:
        existing = daytona.snapshot.get(name)
    except DaytonaNotFoundError:
        existing = None
    if existing is not None:
        state = str(getattr(existing, "state", "")).lower()
        if rebuild or "error" in state or "fail" in state:
            print(f"Deleting snapshot {name} (state={state})", flush=True)
            daytona.snapshot.delete(existing)
        else:
            print(f"Snapshot {name} already exists (state={state})", flush=True)
            return name

    torch_version, lines = locked_requirements()
    with tempfile.TemporaryDirectory() as tmp:
        requirements = Path(tmp) / "requirements.txt"
        requirements.write_text("\n".join(lines) + "\n")
        image = (
            Image.debian_slim(PYTHON_VERSION)
            .pip_install(f"torch=={torch_version}", index_url=TORCH_CPU_INDEX)
            .pip_install_from_requirements(str(requirements))
            .env({"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"})
            .run_commands(
                "python -c 'import torch, stable_baselines3, minigrid, wandb; "
                "assert not torch.cuda.is_available(); print(torch.__version__)'"
            )
        )
        print(f"Building snapshot {name} (torch {torch_version}+cpu, {len(lines)} pinned deps)", flush=True)
        daytona.snapshot.create(
            CreateSnapshotParams(
                name=name,
                image=image,
                resources=Resources(cpu=cpu, memory=memory, disk=disk),
            ),
            on_logs=lambda chunk: print(chunk, end="" if chunk.endswith("\n") else "\n", flush=True),
        )
    print(f"Snapshot {name} ready", flush=True)
    return name


def main() -> None:
    args = build_parser().parse_args()
    if args.print_name:
        print(snapshot_name(args.cpu, args.memory, args.disk))
        return
    ensure_snapshot(args.cpu, args.memory, args.disk, rebuild=args.rebuild)


if __name__ == "__main__":
    sys.exit(main())
