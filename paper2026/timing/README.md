# timing — how fast IduEdu builds and routes, against other libraries

Measurements for the companion paper; they measure the library, not the data, and
are not part of the study pipeline (`../pipeline/`).

| script | measures |
|---|---|
| `bench_build.py` | building drive/walk/transit graphs (IduEdu, OSMnx, pyrosm, cityseer) |
| `bench_gtfs.py` | building a graph from a GTFS feed (IduEdu, peartree) |
| `bench_intermodal.py` | building an intermodal graph |
| `bench_od.py` | origin-destination matrices |
| `bench_accessibility.py`, `bench_scenarios.py` | accessibility workloads on Saint Petersburg (`accessibility_common.py`) |
| `bench_validity.py` | that the libraries' results agree |
| `run_all.py` | all of the above back to back (overnight) |

Every measurement is appended to a CSV in `../results/` as it is taken, so a run
resumes. They need `../environment-bench.yml` (conda-forge: `pyrosm` cannot be
installed with pip) and a machine doing nothing else; absolute seconds do not
transfer between machines, ratios between libraries do. Inputs: `../bench_feeds/`
(copies of the feeds, kept apart from the study corpus), `../pbf_cache/`,
`../buildings_spb.parquet`, `../schools_spb.parquet`, and the Overpass cache
`../.iduedu_cache/` (`timing_paths.py`).
