#!/usr/bin/env python3
"""
Run every paper benchmark back-to-back (overnight runner).

Each step is a separate subprocess; a failure (non-zero exit, crash, or timeout)
is logged and the next step still runs. All steps are resume-safe, so re-running
this script after a partial night continues where it stopped.

Per-step output is streamed to the console and saved to results/logs/<step>.log;
a summary (status + duration per step) is printed at the end and written to
results/logs/run_all_<timestamp>.summary.

Usage (from this directory, with the benchmark venv's python):
    ../.venv-bench/Scripts/python.exe run_all.py
    ../.venv-bench/Scripts/python.exe run_all.py --smoke            # quick dry run
    ../.venv-bench/Scripts/python.exe run_all.py --only build,od_rect  # subset
    ../.venv-bench/Scripts/python.exe run_all.py --step-timeout-hours 6
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE.parent / "results" / "logs"

# (step name, script, extra args). Order matters: bench_od (rect) builds and
# caches the SPb intermodal graph that validity reuses.
STEPS: list[tuple[str, str, list[str]]] = [
    ("build", "bench_build.py", []),
    ("intermodal", "bench_intermodal.py", []),
    ("od_rect", "bench_od.py", ["--mode", "rect"]),
    ("od_square", "bench_od.py", ["--mode", "square"]),
    ("validity", "bench_validity.py", []),
]


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def run_step(name: str, script: str, extra: list[str], *, smoke: bool, timeout: float | None) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    cmd = [sys.executable, str(HERE / script), *extra]
    if smoke:
        cmd.append("--smoke")

    header = f"\n{'=' * 70}\n[{datetime.now():%Y-%m-%d %H:%M:%S}] START {name}: {' '.join(cmd)}\n{'=' * 70}"
    print(header, flush=True)

    start = time.perf_counter()
    status = "ok"
    returncode: int | None = None
    # Stream child output live to console and tee it into the per-step log.
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(header + "\n")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(HERE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_file.write(line)
                    if timeout is not None and time.perf_counter() - start > timeout:
                        proc.kill()
                        status = "timeout"
                        break
                returncode = proc.wait()
            finally:
                if proc.stdout:
                    proc.stdout.close()
        except Exception as exc:  # pragma: no cover - runner robustness
            status = "crashed"
            log_file.write(f"\n[runner] exception launching step: {exc!r}\n")
            print(f"[runner] exception launching {name}: {exc!r}", flush=True)

    duration = time.perf_counter() - start
    if status == "ok" and returncode != 0:
        status = f"failed(rc={returncode})"
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] END   {name}: {status} in {fmt_duration(duration)} "
        f"(log: {log_path})",
        flush=True,
    )
    return {"name": name, "status": status, "duration": duration, "log": log_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="pass --smoke to every step (quick dry run)")
    parser.add_argument("--only", default=None, help="comma-separated subset of step names to run")
    parser.add_argument(
        "--step-timeout-hours",
        type=float,
        default=None,
        help="kill a step after this many hours and move on (default: no timeout)",
    )
    args = parser.parse_args()

    steps = STEPS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        unknown = wanted - {name for name, _, _ in STEPS}
        if unknown:
            parser.error(f"unknown step(s): {sorted(unknown)}; available: {[n for n, _, _ in STEPS]}")
        steps = [s for s in STEPS if s[0] in wanted]

    timeout = args.step_timeout_hours * 3600 if args.step_timeout_hours else None
    started_at = datetime.now()
    print(
        f"[runner] starting {len(steps)} step(s) at {started_at:%Y-%m-%d %H:%M:%S} " f"with {sys.executable}",
        flush=True,
    )

    results = []
    try:
        for name, script, extra in steps:
            results.append(run_step(name, script, extra, smoke=args.smoke, timeout=timeout))
    except KeyboardInterrupt:
        print("\n[runner] interrupted by user; writing summary for completed steps", flush=True)

    summary_lines = [
        f"run_all summary — started {started_at:%Y-%m-%d %H:%M:%S}, finished {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"{'step':<12} {'status':<16} {'duration':>10}",
        f"{'-' * 12} {'-' * 16} {'-' * 10}",
    ]
    for r in results:
        summary_lines.append(f"{r['name']:<12} {r['status']:<16} {fmt_duration(r['duration']):>10}")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    summary_lines += ["", f"{n_ok}/{len(results)} steps ok"]
    summary = "\n".join(summary_lines)

    print("\n" + summary, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_DIR / f"run_all_{started_at:%Y%m%d_%H%M%S}.summary"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    print(f"\n[runner] summary written to {summary_path}", flush=True)

    # Non-zero exit if anything did not finish cleanly (handy for CI/cron chaining).
    if n_ok != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
