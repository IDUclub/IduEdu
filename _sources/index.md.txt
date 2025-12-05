
```{toctree}
:hidden:
:maxdepth: 2

High-level functions <api/high_level>
Graph utilities <api/utilities>
Matrices <api/matrices>
Overpass helpers <api/overpass>
Examples <examples/index>
```
# **IduEdu** is an open-source Python library for the creation and manipulation of complex city networks from [OpenStreetMap](https://www.openstreetmap.org).

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/iduedu.svg)](https://pypi.org/project/iduedu/)
[![CI](https://github.com/DDonnyy/IduEdu/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/DDonnyy/IduEdu/actions/workflows/ci_pipeline.yml)
[![Coverage](https://codecov.io/gh/DDonnyy/IduEdu/graph/badge.svg?token=VN8CBP8ZW3)](https://codecov.io/gh/DDonnyy/IduEdu)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-IDUclub%2FIduEdu-181717?logo=github)](https://github.com/IDUclub/IduEdu)

---

## Features

- **Graph Builders**
  - `get_drive_graph` — driving network with speeds & categories
  - `get_walk_graph` — pedestrian network (bi‑directional)
  - `get_all_public_transport_graph` / `get_single_public_transport_graph` — bus, tram, trolleybus, subway
  - `get_intermodal_graph` — compose PT + walk with platform snapping
- **Geometry & CRS Correctness**
  - Local UTM estimation for accurate metric lengths
  - Safe graph ↔ GeoDataFrame conversion; optional geometry restoration
- **Matrices**
  - `get_adj_matrix_gdf_to_gdf` — OD matrices by length/time using Numba accelerated Dijkstra
  - `get_closest_nodes` — nearest node snapping
- **Utilities**
  - `clip_nx_graph`, `reproject_graph`, `read_gml`/`write_gml`, etc.

---

## Installation

```bash
pip install iduedu
```

> Requires Python 3.10+ and common geospatial stack (GeoPandas, Shapely, PyProj, NetworkX, NumPy, Pandas).

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
from iduedu import get_adj_matrix_gdf_to_gdf

# origins/destinations can be any geometries; representative points are used
origins = gpd.GeoDataFrame(geometry=[...], crs=...)
destinations = gpd.GeoDataFrame(geometry=[...], crs=...)

M = get_adj_matrix_gdf_to_gdf(
    origins, destinations, G, weight="time_min", dtype="float32", threshold=None
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
# Specific day
config.set_overpass_date(date="2020-01-01")

# Or build from components
config.set_overpass_date(year=2020)            # → 2020-01-01T00:00:00Z
config.set_overpass_date(year=2020, month=5)   # → 2020-05-01T00:00:00Z
```

To reset and use the latest data again:

```python
config.set_overpass_date()  # or config.set_overpass_date(None)
```

>When a historical date is set, complex subway stop-area relations are skipped automatically
> (as Overpass may not support those at arbitrary timestamps). A warning is logged in such cases.

> IduEdu respects Overpass API etiquette. Please keep sensible rate limits.

---

## Roadmap / Ideas

- More PT modes and GTFS import
- Caching of Overpass responses
- Richer edge attributes (e.g., elevation, turn costs)

> Contributions and ideas are welcome! Please open an issue or PR.

---