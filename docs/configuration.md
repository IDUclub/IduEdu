# Configuration

IduEdu keeps its runtime settings in one global object, `iduedu.config`. Downloaders and builders read it on
every call, so a change applies to everything that runs after it in the same process. Most settings can also
be given as environment variables, which are read once when `iduedu` is imported.

```python
from iduedu import config

config.to_dict()  # a snapshot of every current setting
```

## Overpass endpoint and requests

```python
from iduedu import config

config.set_overpass_url("https://overpass-api.de/api/interpreter")
config.set_timeout(120)
config.set_rate_limit(min_interval=2.0, max_retries=5, backoff_base=2.0)
```

- `set_overpass_url` accepts any `http://` or `https://` Overpass interpreter, for example a self-hosted
  instance or a mirror.
- `set_timeout` sets the request timeout in seconds. The same value is sent to Overpass as `[timeout:...]`
  in every query.
- `set_rate_limit` spaces requests at least `min_interval` seconds apart. Before each attempt IduEdu asks the
  instance's `/status` endpoint for a free slot and waits for it. Network errors and HTTP 429, 502, 503 and
  504 responses are retried up to `max_retries` times with an exponentially growing delay based on
  `backoff_base`; a 429 response honours the server's `Retry-After` header.
- `config.user_agent` is sent with every request. Set it to something that identifies your project:
  `config.user_agent = "my-project/1.0 (me@example.org)"`.
- `config.proxies` (a `requests`-style mapping such as `{"https": "http://proxy:3128"}`) and
  `config.verify_ssl` are passed to every request.

The public Overpass instance is shared by many users. Keep the default pacing unless you run your own
instance.

## Overpass cache

IduEdu caches raw Overpass JSON responses on disk: boundaries, street-network queries, route relations and
their members. The cache is enabled by default and lives in `.iduedu_cache` relative to the current working
directory.

```python
from iduedu import config

config.set_overpass_cache(enabled=False)  # ignore the cache completely
config.set_overpass_cache(cache_dir="/tmp/overpass_cache", enabled=True)
```

The cache stores responses only, never processed graphs. Clear the directory or disable the cache to force
fresh downloads, for example after OSM data you depend on has been edited.

## Historical snapshots

Overpass can answer a query as OpenStreetMap stood at a given moment:

```python
from iduedu import config

config.set_overpass_date(date="2020-01-01")                # 2020-01-01T00:00:00Z
config.set_overpass_date(date="2020-01-01T12:34:56Z")
config.set_overpass_date(year=2020, month=5)               # 2020-05-01T00:00:00Z
config.set_overpass_date()                                 # back to current data
```

While a date is set, subway routes are queried as plain route relations: stop areas, stations and entrances
are skipped because Overpass does not reliably support those queries at arbitrary timestamps, and a warning is
logged.

## Local OSM files instead of Overpass

Every OSM-based builder can read its data from a local OpenStreetMap file instead of the Overpass API: the
same street networks, public-transport routes, subway stop areas and boundaries, answered with the same
filters and the same `poly:` selection rules, so the graphs come out the same. This needs the optional
dependency [pyosmium](https://osmcode.org/pyosmium/):

```bash
pip install "iduedu[pbf]"
```

```python
from iduedu import config, get_public_transport_graph, get_walk_graph

config.set_osm_source("pbf", path="city.osm.pbf")   # every following call reads the file
walk = get_walk_graph(territory=city_polygon)
config.set_osm_source("overpass")                   # back to the default

with config.use_pbf("city.osm.pbf"):                # or only inside a block
    pt = get_public_transport_graph(territory=city_polygon, transport_types=["bus", "tram", "subway"])
```

The environment variables `IDUEDU_OSM_SOURCE=pbf` and `IDUEDU_PBF_PATH=/data/city.osm.pbf` do the same at import
time. In pbf mode the Overpass cache is not used and `overpass_date` is ignored: the file itself is the
snapshot, and subway stop areas are always read. `snapshot_time(path)` returns the moment the file reflects,
from its header (`osmosis_replication_timestamp`, set in Geofabrik extracts).

### Per-city files from a regional extract

Querying a country-sized file scans all of it on every call. Cut it into small city files once, in one pass
for all cities of the region:

```python
from iduedu import snapshot_time, split_by_polygons

paths = split_by_polygons(
    "russia-northwestern-fed-district-latest.osm.pbf",
    {"spb": spb_polygon, "petrozavodsk": ptz_polygon},
    "cities/",
    location_storage="dense_file_array",     # disk-backed node index for large extracts
)
snapshot_time(paths["spb"])                  # the extract's replication timestamp is kept
```

Each city file keeps, complete and with tags: the street ways (`highway`, `railway`, `public_transport`)
touching the polygon; the public-transport route relations touching it, with all their members also outside
the polygon; for subways, every relation sharing a node with the line (its stop areas), their members and
the stop-area groups containing them; and relations named in `keep_relations` (for instance the boundary, so
`osm_id=` keeps working). Building graphs from a city file then gives the same result as from the regional
file, for the same polygon or any polygon inside it.

### Differences from Overpass

- Elements are returned in id order. Overpass returns route responses in quadtile order (`qt`), so node and
  edge ids of a public-transport graph can be numbered differently even though the graph is the same.
- Points on the polygon boundary count as inside. Overpass' handling of exact boundary hits and of very long
  polygon edges may differ in rare cases.
- Only tag filters are supported (`["k"]`, `[!"k"]`, `["k"="v"]`, `["k"!="v"]`, `["k"~"re"]`, `["k"!~"re"]`,
  with an optional `,i` flag on regular expressions); other Overpass syntax in `custom_filter` raises
  `UnsupportedFilterError`. Regular expressions use Python's `re` module.
- Data missing from the file cannot be found: members of a relation that are not in the file have no
  geometry. Use `split_by_polygons` (or a complete regional extract) rather than a tight clip.
- `split_by_polygons` selects ways spatially only when they carry one of `way_keys`; a `custom_filter` on
  other keys needs `way_keys=None` (or a list including them) when cutting the city files.

## OSM tags kept on edges

Street and public-transport edges keep only a small set of OSM tags:

| Graph | Default tags | Global setter |
| --- | --- | --- |
| Drive | `highway`, `name`, `lanes` | `config.set_drive_useful_edges_attr(...)` |
| Walk | `highway`, `name` | `config.set_walk_useful_edges_attr(...)` |
| Public transport | `name` | `config.set_transport_useful_edges_attr(...)` |

The builders' `osm_edge_tags` argument replaces the global set for a single call:

```python
from iduedu import get_drive_graph

graph = get_drive_graph(osm_id=1114252, osm_edge_tags=["highway", "name", "maxspeed", "surface"])
```

## Progress bars and logging

```python
from iduedu import config

config.set_enable_tqdm(False)  # hide progress bars while public-transport routes are parsed
config.configure_logging(level="INFO")
config.configure_logging(level="DEBUG", to_file="iduedu.log")
```

IduEdu logs through [loguru](https://loguru.readthedocs.io/), available as `config.logger`. The default level
is `WARNING`. `configure_logging` replaces every sink of the shared loguru logger, including sinks your own
application added, then writes to standard error and, with `to_file`, to a file rotated at 20 MB and kept for
14 days. `json=True` writes the bare message without time and level, for collectors that add their own.

## Environment variables

| Variable | Default | Equivalent setting |
| --- | --- | --- |
| `OVERPASS_URL` | `https://overpass-api.de/api/interpreter` | `set_overpass_url` |
| `OVERPASS_USER_AGENT` | `iduedu/<version> (+https://github.com/IDUclub/IduEdu)` | `config.user_agent` |
| `OVERPASS_TIMEOUT` | `120` | `set_timeout` |
| `OVERPASS_MIN_INTERVAL` | `2` | `set_rate_limit(min_interval=...)` |
| `OVERPASS_MAX_RETRIES` | `5` | `set_rate_limit(max_retries=...)` |
| `OVERPASS_BACKOFF_BASE` | `2` | `set_rate_limit(backoff_base=...)` |
| `OVERPASS_DATE` | unset | `set_overpass_date` |
| `OVERPASS_CACHE_DIR` | `.iduedu_cache` | `set_overpass_cache(cache_dir=...)` |
| `OVERPASS_CACHE_ENABLED` | `1`; `0` or `false` disables | `set_overpass_cache(enabled=...)` |
| `ENABLE_TQDM` | `1`; `0` or `false` disables | `set_enable_tqdm` |
| `LOG_LEVEL` | `WARNING` | `configure_logging(level=...)` |
| `IDUEDU_OSM_SOURCE` | `overpass`; `pbf` reads a local file | `set_osm_source` |
| `IDUEDU_PBF_PATH` | unset | `set_osm_source("pbf", path=...)` |

```bash
export OVERPASS_URL="https://overpass.example.org/api/interpreter"
export OVERPASS_CACHE_DIR="/tmp/overpass_cache"
export LOG_LEVEL="INFO"
```

An invalid `OVERPASS_DATE` is ignored with a warning rather than failing the import.

## API reference

```{eval-rst}
.. autoclass:: iduedu.config.Config
    :members:
```
