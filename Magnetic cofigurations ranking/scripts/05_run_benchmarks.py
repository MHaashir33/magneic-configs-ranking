from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from altermagnet_screening.benchmarking import run_benchmarks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run structure-only and spin-aware Top-K DFT-saving benchmarks, "
            "energy-regret/ranking diagnostics, plus Task-ID counts and the final "
            "candidate table for the preliminary-to-strict symmetry screen. "
            "Automatically detects results/new_preliminary_candidates.csv and "
            "results/new_candidates_spin_space_test.csv. No historical AM "
            "candidate reference list is used."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--symmetry-screen-table",
        type=Path,
        default=None,
        help=(
            "One combined CSV containing compensated/PT-broken flags and a "
            "strict spin-space pass flag."
        ),
    )
    parser.add_argument(
        "--preliminary-screen-table",
        type=Path,
        default=None,
        help=(
            "CSV from the preliminary magnetic-symmetry stage. If it contains "
            "compensation/PT flags, they are applied; otherwise rows are treated "
            "as already passing compensated + PT-broken screening."
        ),
    )
    parser.add_argument(
        "--strict-screen-table",
        type=Path,
        default=None,
        help=(
            "CSV from the strict spin-space-symmetry stage. If it contains a "
            "strict-pass flag, that flag is applied; otherwise rows are treated "
            "as already passing strict screening."
        ),
    )
    args = parser.parse_args()

    if bool(args.preliminary_screen_table) != bool(args.strict_screen_table):
        parser.error(
            "--preliminary-screen-table and --strict-screen-table must be supplied together."
        )

    summary = run_benchmarks(
        ROOT,
        preliminary_screen_table_path=args.preliminary_screen_table,
        strict_screen_table_path=args.strict_screen_table,
        symmetry_screen_table_path=args.symmetry_screen_table,
    )
    print("\nFinished:", summary)


if __name__ == "__main__":
    main()
