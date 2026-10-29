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
ratios between libraries are only read within one run. The rerun also moves
every library to its newest release on Python 3.14 (September: Python 3.11 with
the releases of that time), as the author decided on 29 October 2026. New since
September:

- the `iduedu_pbf` arm in `bench_build.py`: IduEdu reads the same BBBike extract
  as pyrosm (`config.use_pbf`), with and without simplification;
- the `simplify` stage (`bench_simplify.py`): 32 full traversals from the same
  sources on the simplified and the raw graph of each city;
- one measurement per cell (`BENCH_ATTEMPTS`, default 1); further attempts are
  added later, only when the author asks;
- `results/library_provenance.json` (version, commit, changed files of the
  package, model constants) and `results/pbf_snapshots.json` (replication time
  of each extract), both written by the run;
- the pyrosm arm passes `simplify=False` to `to_graph` and clears pyrosm's
  result cache before each measured read. From 0.19 pyrosm simplifies by
  default and keeps the result of each read in the temporary folder, so without
  these two changes the arm would time a simplified graph, and a repeated read
  would time the cache. Otherwise pyrosm runs with its defaults: since 0.19 it
  reads the file out of core, and since 0.23 on several threads.

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

Two interpreters. Below, `<bench python>` is the interpreter of
`paper2026/.venv-bench-latest` and `<peartree python>` that of the peartree
environment (`Scripts/python.exe` on Windows, `bin/python` on Linux); commands
in this section run from `paper2026/` unless they say otherwise.

**Bench environment: a new one, with the newest releases.** Leave the September
`.venv-bench` as it is and create `.venv-bench-latest` on Python 3.14, the newest
Python that IduEdu 2.3.0 supports. Python 3.11 cannot take the newest networkx,
numpy, scipy or pyproj, and networkx 3.7 refuses Python 3.14.1. With uv, which
fetches Python 3.14 if the machine lacks it (`--seed` puts pip into the
environment):

    uv venv --python 3.14 --seed .venv-bench-latest

Without uv, run `python -m venv .venv-bench-latest` with a Python 3.14
interpreter. Then install everything in one command and without versions, so
that pip chooses the newest releases that fit together and within IduEdu's own
bounds:

    <bench python> -m pip install -e "..[pbf,io]" osmnx pyrosm cityseer networkx networkit igraph rustworkx objectnat numba numpy pandas geopandas shapely scipy pyarrow psutil
    <bench python> -m pip check

IduEdu 2.3.0 keeps numba below 0.68 and pyarrow below 26. Do not install or
upgrade any package with a second command afterwards: pip would lift those
bounds with no more than a warning. On 29 October 2026 the command resolved to
iduedu 2.3.0, osmium 4.3.2, osmnx 2.1.1, pyrosm 0.26.0, cityseer 5.8.0,
networkx 3.7, networkit 11.2.2, igraph 1.0.0, rustworkx 0.18.1, objectnat 2.1.0,
numba 0.67.0, numpy 2.5.4, pandas 3.1.0, geopandas 1.2.0, shapely 2.2.0,
scipy 1.18.1 and pyarrow 25.0.1 on Python 3.14.7; a release newer than these on
the day of the run is fine. Then check, from `paper2026/timing/`:

    <bench python> -c "import importlib.metadata as m, iduedu, osmium; print(m.version('iduedu'), iduedu.__version__, m.version('osmium'))"
    <bench python> -c "import osmnx, pyrosm, cityseer, networkx, networkit, igraph, rustworkx, objectnat"
    <bench python> -c "import bench_build as b, pyrosm.data as d; f = d.get_data('test_pbf'); b.clear_pyrosm_cache(f); print([b.build_pyrosm(f, n)[:2] for n in ('walk', 'drive')])"

Both IduEdu versions must read 2.3.0. A missing `osmium` does not stop the run:
`bench_build.py` drops the `iduedu_pbf` arm without an error. The last command
runs the pyrosm arm on the small test file that ships with pyrosm, because the
smoke run does not reach pyrosm; with pyrosm 0.26.0 it prints
`[(781, 1650), (647, 1237)]`.

**peartree environment.** peartree 0.6.4 (2021) is its newest release. It runs
only on pandas 1.x, which has no wheels for Python above 3.11, so its stack
cannot follow the bench environment. Reuse the September peartree environment as
it is. Without it, create one on Python 3.11 and let pip choose the rest:

    <peartree python> -m pip install peartree "pandas<2" "numpy<2"

On 29 October 2026 this gave pandas 1.5.3, numpy 1.26.4, networkx 3.6.1 and
geopandas 1.0.1.

## Order

From `paper2026/timing/`:

1. `<bench python> fresh_start.py`, then `<bench python> fresh_start.py --yes`.
   It moves an earlier run's results and caches into
   `../archive/timing_run_<date>/`. Rename that folder (for example to
   `timing_run_<date>_previous`): step 3 calls the script again, and it refuses
   to overwrite.
2. `<bench python> run_all.py --smoke`. Every step must end `ok`, and
   `../results/build_benchmark.csv` must hold `smoke` rows for `iduedu_pbf`.
   If so, send the author the summary and the versions in
   `../results/env_build.json` and go on without waiting; if not, stop and ask.
3. `<bench python> fresh_start.py --yes` again. The smoke run writes into the
   same files as the real one; in `od_benchmark.csv` its rows for 256 sources
   carry the keys of real cells, and the real run would skip those cells.
4. Record both environments as UTF-8 text: the output of
   `<bench python> -m pip freeze` in `../results/pip_freeze_bench.txt` and that
   of `<peartree python> -m pip freeze` in `../results/pip_freeze_peartree.txt`.
5. `<peartree python> -X utf8 bench_gtfs.py --only-peartree`, before
   `run_all.py`. peartree does not import in the bench environment, so the GTFS
   stage there writes an `unavailable` row for every city that has no peartree
   row yet, and that row would block the real measurement under the same key.
6. `<bench python> run_all.py`, in the background, overnight. Stages run one
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
- When a newer release breaks a call in the scripts (the traceback ends inside
  a library), report the traceback and the change you would make, and wait for
  the author: a fix may follow a renamed function or argument, but it must not
  change what is measured.
- Nothing from the run goes into git: `results/` is ignored on purpose, since
  data are published through Zenodo. Commit or push only when the author asks.

## Checks after the run

- `../results/logs/run_all_<timestamp>.summary` lists every stage as `ok`;
- `library_provenance.json` says `iduedu_version` 2.3.0 with empty
  `library_dirty_paths`, and every `env_*.json` but `env_gtfs_peartree.json`
  records Python 3.14, `iduedu` 2.3.0 and the versions in
  `pip_freeze_bench.txt`;
- `build_benchmark.csv` has `iduedu_pbf` and `pyrosm` rows for all six cities,
  and `pbf_snapshots.json` a date for each extract;
- the `n_nodes` of each `pyrosm` row is within a few per cent of the
  `iduedu_pbf` row with `simplify` False for the same city and network (in
  September, within 0.4 per cent of IduEdu's); a simplified graph would have a
  tenth to a half as many nodes;
- in `gtfs_benchmark.csv` peartree is `ok` for four cities and
  `failed: NotImplementedError` for New York (it cannot merge nine feeds), with
  no `unavailable` rows.

## What to hand back

One zip, `timing_run_<date>.zip`, of these items from `paper2026/results/`:
`build_benchmark.csv`, `simplify_benchmark.csv`, `intermodal_benchmark.csv`,
`gtfs_benchmark.csv`, `od_benchmark.csv`, `validity_report.json`,
`library_provenance.json`, `pbf_snapshots.json`, `s7_tables.txt` if present,
`spb_intermodal.urbangraph`, the undated `env_*.json`, `pip_freeze_bench.txt`,
`pip_freeze_peartree.txt`, `accessibility/` and `logs/`. Tell the author where
it is, which checks failed if any, and the versions recorded.

## More attempts, later

Only when the author asks, and with nothing moved in between:

    <bench python> run_all.py --attempts 3
    BENCH_ATTEMPTS=3 <peartree python> -X utf8 bench_gtfs.py --only-peartree

In PowerShell set `$env:BENCH_ATTEMPTS = 3` first. Attempts 2 and 3 are
appended to the same CSVs; attempt 1 stays as it is.
