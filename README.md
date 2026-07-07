# IduEdu

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/iduedu.svg)](https://pypi.org/project/iduedu/)
[![CI](https://github.com/DDonnyy/IduEdu/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/DDonnyy/IduEdu/actions/workflows/ci_pipeline.yml)
[![Coverage](https://codecov.io/gh/DDonnyy/IduEdu/graph/badge.svg?token=VN8CBP8ZW3)](https://codecov.io/gh/DDonnyy/IduEdu)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Docs](https://img.shields.io/badge/docs-latest-4aa0d5?logo=readthedocs)](https://iduclub.github.io/IduEdu/)
[![GitHub](https://img.shields.io/badge/GitHub-IDUclub%2FIduEdu-181717?logo=github)](https://github.com/IDUclub/IduEdu)

<p align="center">
<img src="./docs/_static/iduedu_header.svg" alt="IduEdu logo banner" width="100%">
</p>

**IduEdu** is an open-source Python toolkit for building and analyzing multimodal city networks from
OpenStreetMap data. It downloads OSM data via Overpass, builds drive, walk, public-transport and
intermodal networks, and stores them as `UrbanGraph` objects backed by GeoDataFrame node and edge tables.

## Benchmark snapshot

The benchmark below isolates city-scale pedestrian graph construction, comparing IduEdu's tabular
`UrbanGraph` representation with OSMnx's widely used NetworkX-backed workflow on the same area of
interest for each city. Both simplification modes are shown because `simplify` changes the trade-off
between graph-construction time and the size of the resulting graph.

![Pedestrian graph construction benchmark](./docs/_static/benchmark_walk_summary.png)

Across these repeated B1 runs, IduEdu built pedestrian graphs in about **5.8-9.3x less time** without
simplification and **3.3-6.0x less time** with simplification enabled. The resulting geospatial graph
object had a **10-12x lower deterministic representation-size estimate**. Edge-row counts are shown as
a representation detail:
`UrbanGraph` can keep bidirectional walking edges as one row with a direction flag, while NetworkX-style
directed multigraphs store separate directed edge rows.

Full protocol notes, limitations and raw-result links are in the
[benchmark documentation](https://iduclub.github.io/IduEdu/benchmarks.html).

## Documentation

Full documentation is published at <https://iduclub.github.io/IduEdu/>.
Migrating from the old NetworkX-first API? See the
[UrbanGraph migration guide](https://iduclub.github.io/IduEdu/migration_to_urban_graph.html).
Runnable examples are available for
[graph construction](https://iduclub.github.io/IduEdu/examples/get_any_graph.html),
[UrbanGraph basics](https://iduclub.github.io/IduEdu/examples/urban_graph_basics.html),
[object projection](https://iduclub.github.io/IduEdu/examples/objects_and_nearest_nodes.html), and
[shortest paths](https://iduclub.github.io/IduEdu/examples/shortest_paths.html).

## Features

- Store graph topology, geometry, CRS and edge weights in `UrbanGraph`, a GeoDataFrame-native graph model.
- Build drive and walk graphs from OpenStreetMap with local metric projection and optional simplification.
- Build static public-transport graphs directly from OSM relations for bus, trolleybus, tram and subway.
- Combine pedestrian and public-transport layers into one intermodal graph by projecting stops, platforms
  and station access points onto the walking network.
- Compute shortest paths and OD matrices with Numba-backed CSR routines, cutoff thresholds and adaptive
  graph reversal for unbalanced origin-destination sets.
- Work with connected, weakly connected and strongly connected UrbanGraph components.
- Convert to and from NetworkX through optional compatibility utilities.
- Snap geometries to graph nodes with `nearest_nodes` and validate custom graph edits.
- Store graphs as `.urbangraph` archives with parquet tables and metadata.
- Cache Overpass responses and query historical OSM snapshots.

## Installation

```bash
pip install iduedu
```

IduEdu requires Python 3.11 or 3.12. The core package uses the standard geospatial stack
including GeoPandas, Shapely, PyProj, NumPy, Pandas and SciPy. NetworkX utilities are optional
compatibility helpers.

Use `pip install "iduedu[io]"` to enable `.urbangraph` parquet archive read/write helpers.

## Quickstart

### Build an intermodal graph

```python
from iduedu import get_intermodal_graph

graph = get_intermodal_graph(
    osm_id=1114252,  # for example, Saint Petersburg's Vasileostrovsky District
    pt_kwargs={"transport_types": ["bus", "tram", "subway"]},
)

print(graph.nodes_gdf.head())
print(graph.edges_gdf.head())
```

Graph builders return `UrbanGraph`. Nodes are stored in `graph.nodes_gdf`; edges are stored in
`graph.edges_gdf` and include `u`, `v`, `geometry`, `length_meter` and `time_min`.

### Compute an OD matrix

```python
from iduedu import od_matrix

nodes = graph.nodes_gdf.index.to_list()

matrix = od_matrix(
    graph,
    origins_nodes=nodes[:5],
    destination_nodes=nodes[5:15],
    weight="time_min",
    threshold=30,
)

print(matrix)
```

Use `weight="time_min"` for travel time in minutes or `weight="length_meter"` for distance in meters.
Pairs without a path, or outside `threshold`, are returned as `inf`.

### Work with graph components

```python
from iduedu import largest_component, subgraph_by_nodes

component_nodes = largest_component(graph)
main_graph = subgraph_by_nodes(graph, component_nodes)
```

## Public API

Common entry points are available directly from `iduedu`:

- Builders: `get_drive_graph`, `get_walk_graph`, `get_public_transport_graph`, `get_intermodal_graph`.
- Graph model: `UrbanGraph`, `UrbanGraphChanges`.
- Editing and transforms: `clip_urban_graph`, `join_urban_graphs`, `project_objects2urban_graph`,
  `relabel_urban_graph`, `simplify_multiedges`, `to_directed`, `to_undirected`.
- Graph utilities: `nearest_nodes`, `validate_graph`, `UrbanGraph.nearest_nodes`,
  `UrbanGraph.validate`.
- Graph IO: `read_urban_graph`, `write_urban_graph`, `UrbanGraph.read`, `UrbanGraph.write`.
- Components: `connected_components`, `weakly_connected_components`, `strongly_connected_components`,
  `largest_component`.
- Shortest paths and matrices: `single_source_dijkstra_path_length`, `multi_source_dijkstra_path_length`,
  `multi_source_dijkstra_nearest_source`, `dijkstra_path_length_parallel`, `od_matrix`.
- Optional NetworkX helpers: `graph2gdf`, `gdf2graph`, `read_gml`, `write_gml`, `clip_nx_graph`,
  `reproject_graph`.

## Configuration

```python
from iduedu import config

config.set_overpass_url("https://overpass-api.de/api/interpreter")
config.set_timeout(120)
config.set_rate_limit(min_interval=1.0, max_retries=3, backoff_base=0.5)
config.set_enable_tqdm(True)
config.configure_logging(level="INFO")
```

### Overpass cache

Overpass caching is enabled by default and uses `.iduedu_cache` in the current working directory.

```python
from iduedu import config

config.set_overpass_cache(enabled=False)
config.set_overpass_cache(cache_dir="/tmp/overpass_cache", enabled=True)
```

Environment variables:

```bash
export OVERPASS_CACHE_DIR="/tmp/overpass_cache"
export OVERPASS_CACHE_ENABLED="1"
```

### Historical snapshots

```python
from iduedu import config

config.set_overpass_date(date="2020-01-01")
config.set_overpass_date(year=2020, month=5)
config.set_overpass_date()  # reset to current OSM data
```

When a historical date is set, detailed subway stop-area queries may be skipped because Overpass does
not always support those relation patterns at arbitrary timestamps.

## Development

This project uses `uv`.

```bash
uv sync --all-groups
```

Useful commands:

```bash
make test           # fast tests, no network
make test-network   # Overpass/network tests
make test-all       # all tests
make coverage       # terminal coverage for fast tests
make coverage-xml   # CI coverage, includes network tests
make format-check
make lint
make docs
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development, testing, Conventional Commits and
release workflow.

## Releases

Releases are automated from Conventional Commit messages on `main` using python-semantic-release.
Do not bump versions or create tags manually. The single source of truth for the package version is
`iduedu/_version.py`; `pyproject.toml` reads it dynamically at build time.

## Contacts

- [NCCR](https://actcognitive.org/) - National Center for Cognitive Research
- [IDU](https://idu.itmo.ru/) - Institute of Design and Urban Studies
- [Natalya Chichkova](https://t.me/nancy_nat) - project manager
- [Danila Oleynikov (Donny)](https://t.me/ddonny_dd) - lead software engineer

## Acknowledgments

Реализовано при финансовой поддержке Фонда поддержки проектов Национальной технологической инициативы
в рамках реализации "дорожной карты" развития высокотехнологичного направления "Искусственный интеллект"
на период до 2030 года (Договор № 70-2021-00187)

This research is financially supported by the Foundation for National Technology Initiative's Projects
Support as a part of the roadmap implementation for the development of the high-tech field of Artificial
Intelligence for the period up to 2030 (agreement 70-2021-00187)

---

## License

IduEdu is distributed under the BSD 3-Clause License. See [LICENSE.txt](LICENSE.txt) for details.

---

## Publications

_Coming soon..._
