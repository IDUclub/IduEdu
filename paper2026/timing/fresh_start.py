#!/usr/bin/env python3
"""Move the previous timing run and its caches aside before a fresh run.

A rerun has to measure current OpenStreetMap data with the current library, and
every stage of the benchmark resumes from what it finds on disk: CSV rows already
written are skipped, the Overpass responses of the warm-ups are reused, a PBF extract
already in ``pbf_cache/`` is not downloaded again, and ``bench_od.py`` reuses the
Saint Petersburg graph it cached under ``results/`` without checking which library
version built it. Left in place, any of them would mix the old run into the new one
without a warning.

Nothing is deleted. Everything listed below moves into
``archive/timing_run_<date>/``, keeping its relative path. Paper A's results
(``results/wide_tier/``) and files from older runs that carry a date suffix are not
touched.

    python fresh_start.py          # list what would move
    python fresh_start.py --yes    # move it
"""

import argparse
import shutil
from datetime import date
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PAPER_DIR / "results"

#: Outputs of the previous run, relative to ``results/``.
RESULT_ITEMS = [
    "build_benchmark.csv",
    "simplify_benchmark.csv",
    "intermodal_benchmark.csv",
    "gtfs_benchmark.csv",
    "od_benchmark.csv",
    "validity_report.json",
    "library_provenance.json",
    "pbf_snapshots.json",
    "s7_tables.txt",
    "spb_intermodal.urbangraph",
    "smoke.urbangraph",
    "accessibility",
    "accessibility_cache",
    "logs",
    "logs_smoke",
]

#: Caches the arms read instead of fetching, relative to ``paper2026/``.
CACHE_ITEMS = [
    "timing_cache",  # the IduEdu Overpass cache of the timing benchmarks
    "pbf_cache",  # BBBike extracts read by pyrosm and iduedu_pbf
    "cityseer_cache",  # cityseer's raw Overpass responses
    "timing/cache",  # OSMnx's response cache in the benchmarks' working directory
]


def planned_moves() -> list[tuple[Path, Path]]:
    target = PAPER_DIR / "archive" / f"timing_run_{date.today().isoformat()}"
    moves = []
    for name in RESULT_ITEMS:
        source = RESULTS_DIR / name
        if source.exists():
            moves.append((source, target / "results" / name))
    for env in sorted(RESULTS_DIR.glob("env_*.json")):
        if env.stem.count(".") == 0:  # env_build.json, not env_build.2026-08.json
            moves.append((env, target / "results" / env.name))
    for name in CACHE_ITEMS:
        source = PAPER_DIR / name
        if source.exists():
            moves.append((source, target / name))
    return moves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--yes", action="store_true", help="move the files; without it only list them")
    args = parser.parse_args()

    moves = planned_moves()
    if not moves:
        print("nothing to move: no previous run found")
        return
    for source, destination in moves:
        print(
            f"{'move' if args.yes else 'would move'}  {source.relative_to(PAPER_DIR)}  ->  "
            f"{destination.relative_to(PAPER_DIR)}"
        )
        if args.yes:
            if destination.exists():
                raise SystemExit(f"{destination} already exists; move it away first")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
    if not args.yes:
        print("\nnothing moved; rerun with --yes")


if __name__ == "__main__":
    main()
