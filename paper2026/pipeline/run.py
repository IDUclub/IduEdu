"""Run the pipeline, or part of it, in order. Every stage resumes where it stopped.

    python pipeline/run.py list               # the stages
    python pipeline/run.py all                # everything after the frozen inputs
    python pipeline/run.py from waits         # a stage and all after it
    python pipeline/run.py gtfs coverage      # these stages only

Each step runs in its own process, writes ``results/wide_tier/logs/<stage>.log`` and
stops the run on failure. ``PAPER_RUN=<name>`` sends all outputs to a test copy.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import LOG_DIR  # noqa: E402

#: (stage, what it makes, steps). A step is the argument list of one script.
STAGES = [
    ("city_feeds", "one merged, clipped, repaired GTFS archive per city", [["city_feeds.py"]]),
    ("osm_split", "one .osm.pbf per city from its Geofabrik region", [["osm.py", "split"]]),
    ("osm_pt", "OSM transit graphs (registry constants)", [["osm.py", "graphs", "--kind", "pt"]]),
    ("osm_walk", "OSM walk graphs", [["osm.py", "graphs", "--kind", "walk"]]),
    ("gtfs", "timetable graphs, average-day waits", [["gtfs.py"]]),
    ("coverage", "stop coverage both ways, provenance, displacement", [["coverage.py", "all"]]),
    ("relations", "route relations and stops in the OSM snapshot", [["osm_relations.py", "census"]]),
    ("waits", "waits per city and mode, distribution by mode", [["waits.py", "measure"], ["waits.py", "table"]]),
    ("speed_fit", "run time by segment length on every shaped segment", [["speeds.py", "fit"]]),
    ("variants", "four accessibility variants and their decomposition",
     [["access.py", "graphs"], ["access.py", "measure"], ["access.py", "decompose"]]),
    ("report", "the share of route relations listing a stop; METRICS.md", [["osm_relations.py", "proxy"], ["report.py"]]),
    ("figures", "every figure of the paper, into figures/", [["figures.py"]]),
]  # fmt: skip
NAMES = [name for name, _, _ in STAGES]


def run_stage(name: str, steps: list[list[str]]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "ENABLE_TQDM": "0"}
    with open(LOG_DIR / f"{name}.log", "a", encoding="utf-8") as log:
        for step in steps:
            started = time.strftime("%Y-%m-%dT%H:%M:%S")
            print(f"=== {name}: {' '.join(step)} ({started})", flush=True)
            log.write(f"=== {' '.join(step)} start {started}\n")
            log.flush()
            code = subprocess.run(
                [sys.executable, str(HERE / step[0]), *step[1:]],
                cwd=HERE,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
            log.write(f"=== {' '.join(step)} exit {code} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            if code:
                raise SystemExit(f"{name}: {' '.join(step)} failed with {code}; see {LOG_DIR / (name + '.log')}")


def main() -> None:
    arguments = sys.argv[1:] or ["list"]
    if arguments == ["list"]:
        for index, (name, what, steps) in enumerate(STAGES, start=1):
            print(f"{index:2}. {name:12} {what}\n    " + "\n    ".join("python pipeline/" + " ".join(s) for s in steps))
        return
    if arguments == ["all"]:
        chosen = NAMES
    elif arguments[0] == "from":
        chosen = NAMES[NAMES.index(arguments[1]) :]
    else:
        unknown = set(arguments) - set(NAMES)
        if unknown:
            raise SystemExit(f"unknown stages: {', '.join(sorted(unknown))}; see: python pipeline/run.py list")
        chosen = [name for name in NAMES if name in arguments]
    for name, _, steps in STAGES:
        if name in chosen:
            run_stage(name, steps)


if __name__ == "__main__":
    main()
