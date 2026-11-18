# timing — how fast IduEdu builds and routes, against other libraries

Measurements for the companion paper; they measure the library, not the data, and
are not part of the study pipeline (`../pipeline/`). How to run them on the
benchmark machine, with which environments and in which order, is in
`AGENT_RUN.md`.

| script | measures |
|---|---|
| `bench_build.py` | building drive and walk graphs (IduEdu from Overpass and from the PBF, OSMnx, pyrosm, cityseer) |
| `bench_simplify.py` | full traversals on simplified and unsimplified graphs |
| `bench_gtfs.py` | building a graph from a GTFS feed (IduEdu, peartree) |
| `bench_intermodal.py` | building an intermodal graph |
| `bench_od.py` | origin-destination matrices |
| `bench_accessibility.py`, `bench_scenarios.py` | accessibility workloads on Saint Petersburg (`accessibility_common.py`) |
| `bench_validity.py` | that the libraries' results agree |
| `run_all.py` | all of the above back to back (overnight) |
| `fresh_start.py` | moves an earlier run and its caches aside before a new one |

Every measurement is appended to a CSV in `../results/` as it is taken, so a run
resumes. They need a machine doing nothing else; absolute seconds do not transfer
between machines, ratios between libraries do. Inputs: `../bench_feeds/` (copies of
the feeds, kept apart from the study corpus), `../buildings_spb.parquet` and
`../schools_spb.parquet`. Downloaded during the run: the BBBike extracts in
`../pbf_cache/`, the Overpass cache of the benchmarks in `../timing_cache/`
(`timing_paths.py`, kept apart from the study's `../.iduedu_cache/`) and
cityseer's responses in `../cityseer_cache/`.
