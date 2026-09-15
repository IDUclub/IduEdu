
```{toctree}
:hidden:
:maxdepth: 2

High-level functions <api/high_level>
Configuration <configuration>
Graph data model <api/graph_data_model>
Migrating to UrbanGraph <migration_to_urban_graph>
Benchmarks and design notes <benchmarks>
Transport registry <api/transport_registry>
GTFS public transport <api/gtfs>
Graph utilities <api/utilities>
Matrices <api/matrices>
Overpass helpers <api/overpass>
Examples <examples/index>
```
# **IduEdu** is an open-source Python library for building and analyzing multimodal city networks from [OpenStreetMap](https://www.openstreetmap.org) and GTFS Schedule feeds.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/iduedu.svg)](https://pypi.org/project/iduedu/)
[![Tests and Coverage](https://github.com/IDUclub/IduEdu/actions/workflows/quality.yml/badge.svg)](https://github.com/IDUclub/IduEdu/actions/workflows/quality.yml)
[![Coverage](https://raw.githubusercontent.com/IDUclub/IduEdu/python-coverage-comment-action-data/badge.svg)](https://github.com/IDUclub/IduEdu/tree/python-coverage-comment-action-data)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Docs](https://img.shields.io/badge/docs-latest-4aa0d5?logo=readthedocs)](https://iduclub.github.io/IduEdu/)
[![GitHub](https://img.shields.io/badge/GitHub-IDUclub%2FIduEdu-181717?logo=github)](https://github.com/IDUclub/IduEdu)

---

## Features

- **UrbanGraph model**: graph topology, geometry, CRS and weights are stored in explicit
  `GeoDataFrame` node and edge tables, with lazy CSR adjacency for numerical routines.
- **Street graph builders**: `get_drive_graph` and `get_walk_graph` build OSM-based networks with
  local metric projection, travel-time weights and optional simplification.
- **Public transport from OSM**: `get_public_transport_graph` builds static bus, trolleybus, tram, subway,
  monorail and share-taxi graphs directly from OSM route relations; trains are available through
  `DEFAULT_REGISTRY_W_TRAIN`.
- **Public transport from GTFS**: `get_gtfs_public_transport_graph` reads a local GTFS Schedule directory
  or ZIP archive, or several feeds merged with `merge_gtfs_feeds`, and builds a static graph with
  timetable-derived boarding waits.
- **Intermodal graphs**: `get_intermodal_graph` combines public transport and walk networks by projecting
  stops, platforms and subway access points onto pedestrian edges.
- **Matrices and shortest paths**: `od_matrix` and Dijkstra helpers use Numba-backed CSR kernels, cutoff
  thresholds and adaptive graph reversal for large accessibility workloads. Ordered node routes can
  be converted back to their real edge geometries for map visualization.
- **Interoperability**: optional NetworkX adapters are available for projects that need graph exchange or
  compatibility with older workflows.

See [Benchmarks and design notes](benchmarks.md) for the construction benchmark summary, raw-result
locations and limitations of the static public-transport model.

---

## Installation

```bash
pip install iduedu
```

> Requires Python 3.11 or 3.12 and the geospatial stack (GeoPandas, Shapely, PyProj, NumPy, Pandas, SciPy,
> Numba). Install `iduedu[io]` to read and write `.urbangraph` archives. NetworkX is not installed with
> IduEdu; install it separately to use the optional NetworkX helpers.

---

## Quickstart

### 1) Build an intermodal graph

```python
from iduedu import get_intermodal_graph

# Define a territory (use OSM relation id or a shapely polygon/geodataframe)
G = get_intermodal_graph(osm_id=1114252)  # e.g., Saint Petersburg, Vasileostrovsky District
```

### 2) Compute an OD matrix (time or length)

```python
import geopandas as gpd
from iduedu import od_matrix

# A "graph_node_id" column is used as is; without it, each geometry is matched to its nearest graph node
origins = gpd.GeoDataFrame({"graph_node_id": [...]}, geometry=[...], crs=G.crs)
destinations = gpd.GeoDataFrame({"graph_node_id": [...]}, geometry=[...], crs=G.crs)

M = od_matrix(
    G,
    gdf_origins=origins,
    gdf_destinations=destinations,
    weight="time_min",
    threshold=30,  # pairs without a path or beyond 30 minutes are inf
)
print(M.head())
```

### 3) Reconstruct a route

```python
from iduedu import path_to_edges, single_source_dijkstra_path

nodes = G.nodes_gdf.index
path = single_source_dijkstra_path(G, nodes[0], nodes[-1], weight="time_min")  # [] if unreachable
route = path_to_edges(G, path)  # traversed edges in order, with geometry, type and time_min

print(route[["type", "length_meter", "time_min"]])
```

---

## Configuration

IduEdu reads its settings from the global `config` object, and most of them also from environment variables. Use it
to point requests at another Overpass instance, pace them, cache responses, query a historical OSM snapshot, choose
the OSM tags kept on edges, or set up logging:

```python
from iduedu import config

config.set_overpass_url("https://overpass-api.de/api/interpreter")
config.set_rate_limit(min_interval=2.0, max_retries=5)
config.set_overpass_cache(cache_dir=".iduedu_cache", enabled=True)
config.set_overpass_date(date="2020-01-01")  # query OSM as it stood on this date
config.configure_logging(level="INFO")
```

See the [configuration guide](configuration.md) for every setting and its environment variable.

---

## Roadmap / Ideas

- GTFS transfers and time-dependent timetable routing
- Richer edge attributes (e.g., elevation, turn costs)

> Contributions and ideas are welcome! Please open an issue or PR.

## Contacts

- [NCCR](https://actcognitive.org/) - National Center for Cognitive Research
- [IDU](https://idu.itmo.ru/) - Institute of Design and Urban Studies
- [Natalya Chichkova](https://t.me/nancy_nat) - project manager
- [Danila Oleynikov (Donny)](https://t.me/ddonny_dd) - lead software engineer
---

## Acknowledgments

Реализовано при финансовой поддержке Фонда поддержки проектов Национальной технологической инициативы в рамках реализации "дорожной карты" развития высокотехнологичного направления "Искусственный интеллект" на период до 2030 года (Договор № 70-2021-00187)

This research is financially supported by the Foundation for National Technology Initiative's Projects Support as a part of the roadmap implementation for the development of the high-tech field of Artificial Intelligence for the period up to 2030 (agreement 70-2021-00187)


## Citing IduEdu

If you use IduEdu in research, please cite it. Citation metadata is kept in
[`CITATION.cff`](https://github.com/IDUclub/IduEdu/blob/main/CITATION.cff); on GitHub,
**Cite this repository** in the repository sidebar copies it as APA or BibTeX. Publications describing
IduEdu will be listed below.

## Publications

_Coming soon..._
