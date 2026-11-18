"""Print how a run's trials are learning, per delay, without opening any chart."""

from __future__ import annotations

import argparse
from pathlib import Path

from delayed_reward.results.analysis import SUSTAIN_EVALS, DelaySummary, load_trial_evals, summarize


def _count(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1_000_000:.2f}M" if value >= 1_000_000 else f"{value / 1_000:.0f}k"


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def format_summary(run_dir: Path, summaries: list[DelaySummary]) -> str:
    """The per-delay table ``delayed-reward-inspect`` prints."""
    finished = sum(1 for _ in run_dir.glob("delay_*/trial_*/summary.json"))
    started = sum(1 for path in run_dir.glob("delay_*/trial_*") if path.is_dir())
    lines = [
        f"{run_dir.name}: {finished} trials finished, {started - finished} running or incomplete",
        f"solved = {SUSTAIN_EVALS} greedy evals in a row reach the goal; unsolved trials count as slowest",
        "",
        f"{'delay':>6}  {'solved':>7}  {'median time':>11}  {'final success':>13}  {'stability':>9}",
    ]
    for s in summaries:
        lines.append(
            f"{s.delay:>6}  {f'{s.n_solved}/{s.n_trials}':>7}  {_count(s.median_time_to_solve):>11}  "
            f"{_percent(s.final_success_rate):>13}  {_percent(s.mean_stability):>9}"
        )
    lines += [
        "",
        "final success: trials whose last greedy eval reached the goal",
        "stability: share of evals still solved after a trial first solved",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory (contains delay_*/trial_*)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.run_dir.is_dir():
        raise SystemExit(f"not a directory: {args.run_dir}")
    trials = load_trial_evals(args.run_dir)
    if not trials:
        raise SystemExit(f"no trial under {args.run_dir} recorded greedy evals (eval_metrics.csv)")
    print(format_summary(args.run_dir, summarize(trials)))


if __name__ == "__main__":
    main()
