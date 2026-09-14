# benchmarks — the pipeline that produces every number in the paper

Run in this order. Every stage is resume-safe: it appends one row per city as it
goes and skips what is already recorded, so an interrupted sweep continues where
it stopped. Run with `PYTHONIOENCODING=utf-8` — the console here is not UTF-8 by
default and a stray non-ASCII character will otherwise kill a sweep mid-run.

| # | Stage | Produces |
|---|---|---|
| 1 | `wide_catalogs.py` | the union of the public GTFS catalogues |
| 2 | `wide_selection.py` | `selection.csv` — the sampling frame, round-robin over regions |
| 3 | `wide_dt4a.py` | adds the African and Latin American repositories to the frame |
| 4 | `wide_feeds.py` | downloads archives → `downloads.csv` |
| 5 | `wide_profile.py` | describes each archive from its own contents → `feed_profile.csv` |
| 6 | `wide_places.py` | asks OSM which settlement a feed belongs to → `places_review.csv` |
| 7 | `wide_resolve.py` | picks each city's boundary by data → `cities.csv`, `city_members.csv` |
| 8 | `wide_complete.py` | adds the other feeds serving a chosen city |
| 9 | `wide_city_feed.py` | merges and clips one archive per city → `city_feeds/`, `city_feeds.csv` |
| 10 | `wide_osm.py --run` | OSM transit and pedestrian graphs → `osm_graphs/`, `osm_graphs.csv` |
| 11 | `wide_gtfs.py` | reference graphs from the schedules → `gtfs_graphs/`, `gtfs_graphs.csv` |
| 12 | `wide_indicators.py` | two-directional coverage → `indicators_coverage.csv` |
| 13 | `wide_waits.py` | waiting times → `indicators_waits.csv` |
| 14 | `wide_provenance.py` | which feeds were generated from OSM → `feed_provenance.csv` |
| 15 | `wide_match.py` | paired segments → `matched_segments*.csv`, `speed_by_length.csv` |
| 16 | `wide_calibrate.py` | kinematic constants and their transfer → `kinematics_*.csv` |
| 17 | `wide_access.py` | accessibility, OSM against the schedule → `accessibility.csv` |

Then `wide_tier_metrics.py` here and `../figures/figures_cus.py`
regenerate `results/wide_tier/METRICS.md` and every figure. Neither reads
anything a sweep did not write, so the paper cannot drift from the data.

`wide_sanity.py` is a check rather than a stage: it re-counts route relations
straight from Overpass for cities whose graph came out suspiciously small, to
separate "OSM has nothing here" from "we failed to fetch it".

## Two modules everything else depends on

**`wide_paths.py`** declares every directory and table. Change the archive layout
here and nowhere else.

**`wide_cohort.py`** decides which cities a stage runs over. Before it existed
each stage chose for itself and they disagreed, so the denominators of the result
tables differed by code rather than by data. A city enters the study when it has
a boundary, because the boundary is what makes the two sides comparable: the same
OSM relation clips the feed and the OSM extract. Run it directly to print the
funnel and the cities left out, with reasons.

## Timing benchmarks

`bench_*.py` and `run_all.py` measure the library rather than the data, and they
belong to the companion paper. They need `environment-bench.yml` (conda-forge —
`pyrosm` cannot be installed with pip on any platform) and a machine that is not
doing anything else. Absolute seconds do not transfer between machines; ratios
between libraries do.
