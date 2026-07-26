# IduEdu Smart Cities Paper

This directory contains the manuscript source, reproducibility scripts and
figure-generation code for:

> *IduEdu: An Open-Source GeoDataFrame-Native Infrastructure for Multimodal
> Accessibility and Rapid Scenario Screening in Smart Cities*

The manuscript targets the MDPI journal *Smart Cities*. The research evaluates
IduEdu as a GeoDataFrame-native infrastructure for multimodal graph
construction, batch origin-destination routing, school-accessibility analysis
and rapid public-transport scenario screening.

## Repository policy

The Git repository contains:

- benchmark and scenario source code in `benchmarks/`;
- figure-generation notebooks and publication figures in `figures/`;
- the manuscript source in `manuscript_smartcities.tex`;
- the bibliography in `sample-base.bib`;
- the benchmark dependency specification in `requirements-bench.txt`.

Research inputs, processed results, graph caches, raw Overpass responses and
Zenodo upload packages are intentionally not tracked by Git. They are preserved
in a separate Zenodo Dataset record:

> **Zenodo Dataset:** <https://doi.org/10.5281/zenodo.21610964>

The exact IduEdu source release used by the paper will be preserved through the
Zenodo–GitHub integration as a separate Software record.


## Experiment groups

- **B1 — graph construction:** IduEdu, OSMnx and Pyrosm across six cities,
  including construction time and representation size.
- **B2 — intermodal construction:** walking, public-transport and graph-joining
  stages.
- **B3 — batch OD routing:** IduEdu, NetworKit and igraph in rectangular and
  square workloads, including cutoff experiments.
- **B4 — validity:** reachability and travel-time agreement with NetworkX and
  structural comparison with OSMnx.
- **B5/B6 — city-scale case study:** school accessibility, waiting-time
  degradation, mode and route removal, and leave-one-route-out criticality for
  Saint Petersburg.

## Environment

The experiments used Python 3.11. Principal package versions and hardware
metadata are recorded separately for each run in the archived `env_*.json`
files. `requirements-bench.txt` describes the benchmark dependencies, but a
complete freeze of all transitive packages from the original workstation is
not available.

To create a local benchmark environment:

```bash
cd paper2026
uv venv .venv-bench --python 3.11
uv pip install --python .venv-bench/Scripts/python.exe -r requirements-bench.txt
```

The first line of `requirements-bench.txt` installs IduEdu from the repository
root in editable mode. For archival reproduction, use the exact tagged
software release referenced by the Zenodo Software record.

## Running the scripts

From `paper2026/benchmarks`:

```bash
../.venv-bench/Scripts/python.exe run_all.py
```

Individual scripts support `--smoke` for a reduced validation run. Full
construction benchmarks download OSM data and can take hours.

The city-scale case-study scripts are:

```bash
../.venv-bench/Scripts/python.exe bench_accessibility.py
../.venv-bench/Scripts/python.exe bench_scenarios.py
```

To regenerate the publication figures:

```bash
cd ../figures
../.venv-bench/Scripts/python.exe -m jupyter nbconvert \
  --to notebook --execute paper_figures.ipynb \
  --output paper_figures.ipynb
../.venv-bench/Scripts/python.exe -m jupyter nbconvert \
  --to notebook --execute accessibility_figures.ipynb \
  --output accessibility_figures.ipynb
```

## Data provenance and licenses

Street, public-transport, school and building geometries were obtained from
OpenStreetMap. Population values were allocated to residential buildings by an
author-developed algorithm; the files contain no individual-level data.

- IduEdu and benchmark code: BSD 3-Clause License.
- OSM-derived databases and graph artifacts: ODbL-1.0 with attribution to
  OpenStreetMap contributors.
- Author-created documentation and figures: CC BY 4.0, subject to the
  attribution requirements of incorporated OSM-derived content.

Detailed attribution and licensing statements are included in the Zenodo
package.
