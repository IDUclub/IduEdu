
```{toctree}
:hidden:
:maxdepth: 2

High-level functions <api/high_level>
Graph data model <api/graph_data_model>
Migrating to UrbanGraph <migration_to_urban_graph>
Benchmarks and design notes <benchmarks>
Transport registry <api/transport_registry>
Graph utilities <api/utilities>
Matrices <api/matrices>
Overpass helpers <api/overpass>
Examples <examples/index>
```
# **IduEdu** is an open-source Python library for building and analyzing multimodal city networks from [OpenStreetMap](https://www.openstreetmap.org).

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/iduedu.svg)](https://pypi.org/project/iduedu/)
[![Tests and Coverage](https://github.com/IDUclub/IduEdu/actions/workflows/quality.yml/badge.svg)](https://github.com/IDUclub/IduEdu/actions/workflows/quality.yml)
[![Coverage](https://codecov.io/gh/IDUclub/IduEdu/graph/badge.svg)](https://codecov.io/gh/IDUclub/IduEdu)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Docs](https://img.shields.io/badge/docs-latest-4aa0d5?logo=readthedocs)](https://iduclub.github.io/IduEdu/)
[![GitHub](https://img.shields.io/badge/GitHub-IDUclub%2FIduEdu-181717?logo=github)](https://github.com/IDUclub/IduEdu)

---

## Features

- **UrbanGraph model**: graph topology, geometry, CRS and weights are stored in explicit
  `GeoDataFrame` node and edge tables, with lazy CSR adjacency for numerical routines.
- **Street graph builders**: `get_drive_graph` and `get_walk_graph` build OSM-based networks with
  local metric projection, travel-time weights and optional simplification.
- **Public transport from OSM**: `get_public_transport_graph` builds static bus, tram, trolleybus and
  subway graphs directly from OSM route relations.
- **Intermodal graphs**: `get_intermodal_graph` combines public transport and walk networks by projecting
  stops, platforms and subway access points onto pedestrian edges.
- **Matrices and shortest paths**: `od_matrix` and Dijkstra helpers use Numba-backed CSR kernels, cutoff
  thresholds and adaptive graph reversal for large accessibility workloads.
- **Interoperability**: optional NetworkX adapters are available for projects that need graph exchange or
  compatibility with older workflows.

See [Benchmarks and design notes](benchmarks.md) for the construction benchmark summary, raw-result
locations and limitations of the static public-transport model.

---

## Installation

```bash
pip install iduedu
```

> Requires Python 3.11+ and common geospatial stack (GeoPandas, Shapely, PyProj, NetworkX, NumPy, Pandas).

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

# origins/destinations contain projected points already attached to graph nodes
origins = gpd.GeoDataFrame({"graph_node_id": [...]}, geometry=[...], crs=G.crs)
destinations = gpd.GeoDataFrame({"graph_node_id": [...]}, geometry=[...], crs=G.crs)

M = od_matrix(
    G,
    gdf_sources=origins,
    gdf_targets=destinations,
    weight="time_min",
    dtype="float32",
)
print(M.head())
```

---

## Configuration

Tweak Overpass endpoint, timeouts, and rate limits globally:

```python
from iduedu import config

config.set_overpass_url("https://overpass-api.de/api/interpreter")
config.set_timeout(120)
config.set_rate_limit(min_interval=1.0, max_retries=3, backoff_base=0.5)

# Optional progress bars and logging
config.set_enable_tqdm(True)
config.configure_logging(level="INFO")
```

### Overpass caching
IduEdu provides optional file-based caching of Overpass JSON responses to speed repeated queries. This cache is used for boundaries, network queries, route relation queries and member fetches.

- Runtime API:

```python
from iduedu import config

# Disable cache for this session
config.set_overpass_cache(enabled=False)

# Enable cache and change cache directory
config.set_overpass_cache(cache_dir="/tmp/overpass_cache", enabled=True)
```

- Environment variables:

```bash
export OVERPASS_CACHE_DIR="/tmp/overpass_cache"
export OVERPASS_CACHE_ENABLED="1"  # "0" or "false" disables cache
```

- Behavior notes:
  - Cache is enabled by default and uses ".iduedu_cache" as the default directory.
  - The cache stores raw Overpass JSON responses; it does not cache processed graphs or derived data.
  - To force fresh downloads, clear the cache directory or disable caching for that run.

### Historical snapshots

You can fix queries to a specific OSM snapshot using the Overpass `date` parameter.
This allows retrieving map data as it existed at a given moment in time.

```python
from iduedu import config

# Specific day
config.set_overpass_date(date="2020-01-01")

# Or build from components
config.set_overpass_date(year=2020)            # → 2020-01-01T00:00:00Z
config.set_overpass_date(year=2020, month=5)   # → 2020-05-01T00:00:00Z


# To reset and use the latest data again:

config.set_overpass_date()  # or config.set_overpass_date(None)
```

>When a historical date is set, complex subway stop-area relations are skipped automatically
> (as Overpass may not support those at arbitrary timestamps). A warning is logged in such cases.

> IduEdu respects Overpass API etiquette. Please keep sensible rate limits.

---

## Roadmap / Ideas

- More PT modes and GTFS import
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


## Publications

_Coming soon..._
