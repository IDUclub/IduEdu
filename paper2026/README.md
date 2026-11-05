# paper2026 — open transit data against OpenStreetMap

Source, pipeline and figures for two papers built from one body of work.

**A — open urban data.** How complete OpenStreetMap is for public transport, and
what the difference against published timetables costs when accessibility is
computed. 265 cities, 626 GTFS feeds, 91 countries (October 2026 snapshot). Completeness is measured in
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
| `pipeline/` | the study pipeline: every number of paper A from the frozen inputs — [`pipeline/README.md`](pipeline/README.md) |
| `inputs/` | the frozen inputs: study cities, their feeds (sha256), boundaries, Geofabrik regions; `pipeline/cohort/` says how they were drawn (not in Git; in the Zenodo deposit) |
| `timing/` | the library timing benchmarks of paper B — [`timing/README.md`](timing/README.md) |
| `figures/` | the images of both papers; paper A's are drawn by `pipeline/figures.py` |
| `results/wide_tier/` | one table per stage, `METRICS.md` with every headline number, `logs/` (not in Git; the tables are in the Zenodo deposit) |
| `manuscript_cus/` | the manuscript: text modules, LaTeX generator, Springer template, built PDF (not in Git) |
| `deposit_cus/` | the Zenodo deposit, assembled by `python pipeline/deposit.py` (not in Git) |
| `drafts/`, `archive/`, `tools/`, `PLAN.md` | working notes, superseded versions, one-off scripts (not in Git) |
| `references/` | PDFs of the work the manuscript cites closely (not in Git) |
| `feed_cache/`, `osm/`, `city_feeds/`, `osm_graphs/`, `gtfs_graphs/`, `osm_graphs_variants/` | downloaded feeds, OSM extracts and built artefacts (not in Git; `osm/` is a junction to `D:/iduedu_osm_pbf` here) |

## Reproducing the study

```bash
python paper2026/pipeline/run.py list   # the stages and the command behind each
python paper2026/pipeline/run.py all    # everything after the frozen inputs; every stage resumes
python paper2026/pipeline/deposit.py    # the Zenodo deposit in deposit_cus/ (nothing is uploaded)
```

No stage queries a live service: feeds are the archived copies, OSM is one Geofabrik
snapshot, boundaries are frozen polygons.

## What is in Git and what is not

Tracked: the pipeline, the figure generators, the figures a manuscript uses, and
the environment files that rebuild the whole thing.

Not tracked, for three different reasons. Downloaded archives, catalogue
snapshots, built graphs and result tables are gigabytes and belong in a citable
archive rather than in history. Manuscripts stay out until their papers are
published, because a public repository is a preprint and that is a decision to
take deliberately rather than by commit. Working notes and superseded drafts are
ours to read and nobody else's.

> **Zenodo deposit of paper A** (265 cities, version 3.0.0):
> <https://doi.org/10.5281/zenodo.23732451>
>
> **Previous Zenodo record** (superseded, kept for reference):
> <https://doi.org/10.5281/zenodo.21610964>

## Environment

Python 3.11–3.12. The study pipeline runs in `.venv-bench` with IduEdu installed
from the root with the `pbf` extra (pyosmium). The timing benchmarks need `environment-bench.yml`
(conda-forge: `pyrosm` cannot be installed with pip on any platform, because
`cykhash` publishes no wheels).

## Data provenance and licences

Street, public-transport, school and building geometries come from
OpenStreetMap. Schedules come from the operators' own published GTFS feeds, via
the Mobility Database, the Transitland Atlas, DigitalTransport4Africa and Transitous.
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
