# Timing rerun on IduEdu 2.3.0: instructions for the agent

This branch carries what the timing benchmarks of the IduEdu paper need on the
benchmark machine. You run them; the author takes the results back to their own
machine, where the figures and the manuscript are generated from them (the
manuscript is not in git). Do not edit the scripts during the run: a harness
changed halfway makes the run incomparable with itself. If something fails,
report it with the log and ask the author.

## What and why

The paper's timings were measured in September 2026 on IduEdu before 2.3.0.
Version 2.3.0 rewrote routing, averages GTFS waits over service days, changed
the model constants and added a local PBF source, so the whole bench is rerun:
ratios between libraries are only read within one run. New since September:

- the `iduedu_pbf` arm in `bench_build.py`: IduEdu reads the same BBBike extract
  as pyrosm (`config.use_pbf`), with and without simplification;
- the `simplify` stage (`bench_simplify.py`): 32 full traversals from the same
  sources on the simplified and the raw graph of each city;
- one measurement per cell (`BENCH_ATTEMPTS`, default 1); further attempts are
  added later, only when the author asks;
- `results/library_provenance.json` (version, commit, changed files of the
  package, model constants) and `results/pbf_snapshots.json` (replication time
  of each extract), both written by the run.

`README.md` in this folder predates these changes and names an old cache path;
where it disagrees with this file, this file is current.

## Inputs that are not in git

The author brings these; they go into `paper2026/` and are git-ignored. Check
sizes and hashes before the run.

| file | bytes | sha256, first 16 |
|---|---:|---|
| `bench_feeds/hsl.zip` | 71,120,847 | `8b368864c7c930e7` |
| `bench_feeds/sankt_peterburg.zip` | 34,221,487 | `3de09c3d14941082` |
| `bench_feeds/moscow.zip` | 61,699,328 | `7b344145d45d6bf2` |
| `bench_feeds/london.zip` | 2,912,729 | `6cf0b27d8b15dc86` |
| `bench_feeds/nyc_feeds/gtfs_subway.zip` | 5,615,228 | `2c9e69b82ac596a2` |
| `bench_feeds/nyc_feeds/gtfs_bx.zip` | 6,808,243 | `e47c7bd67749048c` |
| `bench_feeds/nyc_feeds/gtfs_b.zip` | 13,522,834 | `2dec79e490ede975` |
| `bench_feeds/nyc_feeds/gtfs_m.zip` | 6,708,453 | `e020b62046914003` |
| `bench_feeds/nyc_feeds/gtfs_q.zip` | 4,699,835 | `d721863ae22c512a` |
| `bench_feeds/nyc_feeds/gtfs_si.zip` | 5,013,238 | `0d54de782e7094cd` |
| `bench_feeds/nyc_feeds/gtfs_busco.zip` | 5,621,861 | `94933c1af1f182b3` |
| `bench_feeds/nyc_feeds/gtfslirr.zip` | 1,979,146 | `156969c4a3f9ce38` |
| `bench_feeds/nyc_feeds/gtfsmnr.zip` | 3,371,581 | `b1b673dffe067da8` |
| `buildings_spb.parquet` | 19,890,385 | `46545fe9c9b55919` |
| `schools_spb.parquet` | 42,098 | `32df600d128391f3` |

Everything else is downloaded during the run and has to be fresh: BBBike
extracts into `pbf_cache/`, Overpass responses into `timing_cache/`, cityseer's
into `cityseer_cache/`. Never copy these caches in from elsewhere.

## Environments

Two interpreters, both Python 3.11 (September: 3.11.9). Below, `<bench python>`
is the interpreter of `paper2026/.venv-bench` (`Scripts/python.exe` on Windows,
`bin/python` on Linux) and `<peartree python>` that of the peartree environment;
commands in this section run from `paper2026/`.

**Bench environment.** If `.venv-bench` survives from the September run, keep it
and its competitor versions, and only reinstall IduEdu from this checkout:

    <bench python> -m pip install -e "..[pbf]"

Do this even if IduEdu is already installed editable: it refreshes the package
metadata that `dump_environment` records (the manuscript checks that it reads
2.3.0) and installs `osmium>=4` for the PBF arm. Then check:

    <bench python> -c "import importlib.metadata as m, iduedu, osmium; print(m.version('iduedu'), iduedu.__version__, osmium.__version__)"
    <bench python> -c "import osmnx, pyrosm, cityseer, networkx, networkit, igraph, rustworkx, objectnat"

Both IduEdu versions must read 2.3.0. A missing `osmium` does not stop the run:
`bench_build.py` drops the `iduedu_pbf` arm without an error. Install a missing
`cityseer` or `rustworkx` with pip; do not upgrade what is installed.

Without that environment, create `.venv-bench` with `python -m venv` (Python
3.11; a `uv venv` has no pip) and install the September versions (cityseer and
rustworkx were not recorded then; these are the releases of that time):

    <bench python> -m pip install -e "..[pbf]" osmnx==2.1.1 pyrosm==0.13.1 networkx==3.6.1 networkit==11.2.1 igraph==1.0.0 rustworkx==0.18.1 cityseer==5.8.0 objectnat==2.0.0 numba==0.65.1 numpy==2.4.6 pandas==3.0.5 geopandas==1.1.4 shapely==2.1.2 scipy==1.17.1 pyarrow psutil

pyrosm 0.13.1 compiles `cykhash` and needs a C compiler (MSVC Build Tools on
Windows). pyrosm 0.26.0 (2026-10-27) ships wheels, but a different pyrosm is the
author's decision. `../environment-bench.yml` does not resolve: conda-forge has
no `osmium` or `cityseer` package.

**peartree environment.** peartree 0.6.4 needs pandas 1.x and cannot share the
bench environment. Reuse the September one or create `.venv-peartree`:

    <peartree python> -m pip install peartree==0.6.4 pandas==1.5.3 numpy==1.26.4 networkx==3.6.1 geopandas==1.1.4 shapely==2.0.7

## Order

From `paper2026/timing/`:

1. `<bench python> fresh_start.py`, then `<bench python> fresh_start.py --yes`.
   It moves an earlier run's results and caches into
   `../archive/timing_run_<date>/`. Rename that folder (for example to
   `timing_run_<date>_previous`): step 4 calls the script again, and it refuses
   to overwrite.
2. `<bench python> run_all.py --smoke`. Every step must end `ok`, and
   `../results/build_benchmark.csv` must hold `smoke` rows for `iduedu_pbf`.
   Show the author the summary and the versions in `../results/env_build.json`,
   and wait for a go-ahead.
3. `<bench python> fresh_start.py --yes` again. The smoke run writes into the
   same files as the real one; in `od_benchmark.csv` its rows for 256 sources
   carry the keys of real cells, and the real run would skip those cells.
4. `<peartree python> -X utf8 bench_gtfs.py --only-peartree`, before
   `run_all.py`. peartree does not import in the bench environment, so the GTFS
   stage there writes an `unavailable` row for every city that has no peartree
   row yet, and that row would block the real measurement under the same key.
5. `<bench python> run_all.py`, in the background, overnight. Stages run one
   after another: build, simplify, intermodal, gtfs, od_rect, od_square,
   validity, accessibility, scenarios. Logs go to `../results/logs/`.

## Rules during the run

- One process talks to Overpass at a time. Do not start a second runner, a
  single stage, or anything else that fetches from Overpass next to
  `run_all.py`; do not change the stage order or the Overpass endpoint without
  the author.
- The run is the only load on the machine: no other benchmarks, builds, graph
  indexing (including `graphify`) or heavy agent work until it ends. Sleep and
  power settings are for the author to set.
- When a stage fails or Overpass keeps refusing, report the tail of
  `../results/logs/<stage>.log` and ask. Every stage resumes from its CSV, so
  `<bench python> run_all.py --only <stage>` continues where it stopped.
- Nothing from the run goes into git: `results/` is ignored on purpose, since
  data are published through Zenodo. Commit or push only when the author asks.

## Checks after the run

- `../results/logs/run_all_<timestamp>.summary` lists every stage as `ok`;
- `library_provenance.json` says `iduedu_version` 2.3.0 with empty
  `library_dirty_paths`, and every `env_*.json` but `env_gtfs_peartree.json`
  records `iduedu` 2.3.0;
- `build_benchmark.csv` has `iduedu_pbf` rows for all six cities, and
  `pbf_snapshots.json` a date for each extract;
- in `gtfs_benchmark.csv` peartree is `ok` for four cities and
  `failed: NotImplementedError` for New York (it cannot merge nine feeds), with
  no `unavailable` rows.

## What to hand back

One zip, `timing_run_<date>.zip`, of these items from `paper2026/results/`:
`build_benchmark.csv`, `simplify_benchmark.csv`, `intermodal_benchmark.csv`,
`gtfs_benchmark.csv`, `od_benchmark.csv`, `validity_report.json`,
`library_provenance.json`, `pbf_snapshots.json`, `s7_tables.txt` if present,
`spb_intermodal.urbangraph`, the undated `env_*.json`, `accessibility/` and
`logs/`. Tell the author where it is, which checks failed if any, and the
versions recorded.

## More attempts, later

Only when the author asks, and with nothing moved in between:

    <bench python> run_all.py --attempts 3
    BENCH_ATTEMPTS=3 <peartree python> -X utf8 bench_gtfs.py --only-peartree

In PowerShell set `$env:BENCH_ATTEMPTS = 3` first. Attempts 2 and 3 are
appended to the same CSVs; attempt 1 stays as it is.
