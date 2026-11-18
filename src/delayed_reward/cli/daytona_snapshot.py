"""Build (or reuse) the CPU-only Daytona snapshot with this project's pinned dependencies."""

from __future__ import annotations

import argparse

from delayed_reward.sweep.daytona_snapshot import ensure_snapshot, snapshot_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", type=int, default=2, help="vCPUs per sandbox")
    parser.add_argument("--memory", type=int, default=4, help="GiB RAM per sandbox")
    parser.add_argument("--disk", type=int, default=10, help="GiB disk per sandbox")
    parser.add_argument("--print-name", action="store_true", help="Print the snapshot name and exit")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild an existing snapshot")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.print_name:
        print(snapshot_name(args.cpu, args.memory, args.disk))
        return
    ensure_snapshot(args.cpu, args.memory, args.disk, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
