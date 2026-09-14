# paper2026 — open transit data against OpenStreetMap

Source, pipeline and figures for two papers built from one body of work.

**A — open urban data.** How complete OpenStreetMap is for public transport, and
what the difference against published timetables costs when accessibility is
computed. 126 cities, 169 GTFS feeds, 54 countries. Completeness is measured in
**both directions without designating either source as ground truth**, and the
asymmetry between the two directions stands in for the missing reference.
Targeted at *Computational Urban Science* (Springer Nature).

**B — the library.** The `UrbanGraph` representation and its computational
properties, measured against OSMnx, pyrosm, igraph, NetworKit, rustworkx and
cityseer. Targeted at *Geographical Analysis*; the timing runs are waiting for
a machine that is not doing anything else.

## Layout

| Directory | What is in it |
|---|---|
| `benchmarks/` | the pipeline that produces every number — see its own README for the order |
| `figures/` | one generator per paper — `figures_cus.py` here — and the images they write |
| `results/wide_tier/` | one CSV per stage, plus `METRICS.md` with every headline number |
| `manuscript_cus/` | the manuscript: text modules, LaTeX generator, Springer template, built PDF — `python manuscript_cus/build_submission.py` (not in Git) |
| `deposit_cus/` | the Zenodo upload for paper A, assembled by `benchmarks/make_deposit.py --run` (not in Git) |
| `drafts/`, `archive/`, `tools/`, `PLAN.md` | working notes, superseded drafts, and scripts that answered a question once (not in Git) |
| `references/` | PDFs of the work the manuscript cites closely (not in Git) |
| `city_feeds/` | one merged, boundary-clipped GTFS archive per city (not in Git) |
| `osm_graphs/`, `gtfs_graphs/` | built graphs (not in Git) |

## Reproducing the study

Every stage is resume-safe and writes one row per city as it goes. Run with
`PYTHONIOENCODING=utf-8` — the console here is not UTF-8 by default, and a stray
non-ASCII character in a log line will otherwise kill a sweep mid-run.

```bash
cd paper2026/benchmarks && python wide_cohort.py
```

That prints the funnel — how many cities survive each stage, and which are left
out with the reason. `benchmarks/README.md` lists the seventeen stages in order.
The full sweep is hours of Overpass traffic; the graphs and merged feeds are
published through Zenodo so the measuring stages can be rerun without refetching.

To regenerate the tables and figures from result CSVs already on disk:

```bash
cd paper2026 && python benchmarks/wide_tier_metrics.py && python figures/figures_cus.py
```

Neither reads anything a sweep did not write, so the manuscript cannot drift away
from the data.

## Two modules worth knowing about

`benchmarks/wide_paths.py` declares every directory and table in one place.

`benchmarks/wide_cohort.py` decides which cities a stage runs over. A city enters
the study when it has a boundary, because the boundary is what makes the two
sides comparable: the same OSM relation clips the feed and the OSM extract.
Cities without one are reported with a reason rather than dropped in silence.

## What is in Git and what is not

Tracked: the pipeline, the figure generators, the figures a manuscript uses, and
the environment files that rebuild the whole thing.

Not tracked, for three different reasons. Downloaded archives, catalogue
snapshots, built graphs and result tables are gigabytes and belong in a citable
archive rather than in history. Manuscripts stay out until their papers are
published, because a public repository is a preprint and that is a decision to
take deliberately rather than by commit. Working notes and superseded drafts are
ours to read and nobody else's.

> **Previous Zenodo record** (superseded, kept for reference):
> <https://doi.org/10.5281/zenodo.21610964>

A new deposit covering the 126-city study is a condition of submission — the
journal returns a manuscript without a Data Availability Statement as incomplete.

## Environment

Python 3.11. The study pipeline runs in the repository's own environment with
IduEdu installed from the root. The timing benchmarks need `environment-bench.yml`
(conda-forge: `pyrosm` cannot be installed with pip on any platform, because
`cykhash` publishes no wheels).

## Data provenance and licences

Street, public-transport, school and building geometries come from
OpenStreetMap. Schedules come from the operators' own published GTFS feeds, via
the Mobility Database, the Transitland Atlas and DigitalTransport4Africa.
Population values were allocated to residential buildings by an author-developed
algorithm; no file contains individual-level data.

- IduEdu and the code here: BSD 3-Clause.
- OSM-derived databases and graph artefacts: ODbL-1.0, attribution to
  OpenStreetMap contributors.
- Author-written text and figures: CC BY 4.0, subject to the attribution
  requirements of the OSM-derived content they incorporate.

Individual feeds carry their publishers' own licences, and two of them
(Santiago, Saint Petersburg) do not permit republication. The archive therefore
holds derived tables for those cities rather than the source archives.
