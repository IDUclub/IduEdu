# CHANGELOG

<!-- version list -->

## v2.1.0 (2026-09-14)

IduEdu 2.1.0 builds public transport graphs from GTFS Schedule feeds, reconstructs shortest paths as actual routes, and fixes how public transport routes are assembled from OpenStreetMap. The public transport travel-time model is recalibrated against published timetables, so travel times and waits differ from 2.0; see **Upgrading from 2.0** below.

### Highlights

- **Public transport from GTFS.** `get_gtfs_public_transport_graph` builds a static graph from a GTFS directory, a ZIP archive or several feeds at once, with waiting times taken from the timetable.
- **Routes, not only distances.** `single_source_dijkstra_path`, `multi_source_dijkstra_path` and `path_to_edges` return the nodes and edges a shortest path actually follows.
- **OSM routes without teleports.** Route relations are stitched in the right order and direction and cut at holes, instead of being bridged by straight lines across the city.
- **Recalibrated travel times.** Per-mode free-flow speed, dwell and waiting time in `TransportSpec`, fitted against published timetables.

### Upgrading from 2.0

- `TransportSpec` no longer has `traffic_coef`, and `base_speed_kmh` is required. A call such as `TransportSpec("bus", ..., traffic_coef=0.7)` now raises `TypeError`. A road speed limit still caps the speed where it is lower than the mode's own.
- `get_public_transport_graph` no longer takes `avg_boarding_time_min`. The wait is set per mode through `TransportSpec.avg_wait_time_min`. To keep the flat one-minute wait of 2.0:

```python
from dataclasses import replace

from iduedu import DEFAULT_REGISTRY, TransportRegistry, get_public_transport_graph

one_minute_wait = TransportRegistry(
    {name: replace(DEFAULT_REGISTRY.get(name), avg_wait_time_min=1.0) for name in DEFAULT_REGISTRY.list_types()}
)
graph = get_public_transport_graph(osm_id=1114252, transport_registry=one_minute_wait)
```

- Travel times change with default settings too: OSM boarding edges now cost 3-8.2 minutes depending on the mode instead of 1 minute, and in-vehicle times follow the new speed model. Rebuild matrices and accessibility results computed with 2.0.
- With `transport_types=None` the OSM builder also fetches `monorail` and share taxi (`taxi`) routes, and `tram` includes OSM `light_rail` routes. Pass `transport_types` explicitly to limit the modes.
- OSM route relations with a gap over 1 km are split into parts, and routes without stops or platforms are left out, so node and edge counts differ from 2.0.
- In intermodal graphs `subway_platform` nodes are no longer attached to the walk network directly; they are reached through stations and entrances.

### Features

#### GTFS Schedule graphs

- `get_gtfs_public_transport_graph(feed, *, service_date, start_time, end_time, crs, single_departure_wait_min, walk_speed_m_per_min)` accepts a GTFS directory, a ZIP archive, a `GTFSFeed` or a list of them.
- `service_date` applies `calendar.txt` and `calendar_dates.txt`; `start_time` and `end_time` select a time window, overnight windows included.
- Trips with the same route, direction, shape and stop order form a route pattern; a segment's time is the median scheduled time.
- Boarding waits come from the timetable: half the mean gap between departures, or `headway_secs / 120` for `frequencies.txt`. A stop with a single departure gets no boarding edge unless `single_departure_wait_min` is set.
- Segment geometry is cut from `shapes.txt`; without shapes it is a straight line whose length is multiplied by `sqrt(2)`.
- `pathways.txt` becomes `pathway` edges inside stations, and `parent_station` becomes zero-cost `station_link` edges, so a metro is not dropped as a separate component.
- GTFS route types 0-7, 11 and 12 and Google's extended route types are mapped by explicit ranges (for example 200-209 -> `coach`, 405 -> `monorail`); unknown codes stay `public_transport`.
- New `iduedu.gtfs` module with `read_gtfs_feed`, `GTFSFeed`, `validate_gtfs_feed`, `GTFSValidationError` and `merge_gtfs_feeds`.
- `merge_gtfs_feeds` combines several feeds of one city (New York publishes nine) and qualifies identifiers by source, so routes, trips and stops of different feeds do not collide. `merge_stops_within` optionally fuses stops across feeds within a radius.
- A GTFS graph joins an OSM walk graph through `join_pt_walk_graph` when both use the same CRS. See the [GTFS guide](https://iduclub.github.io/IduEdu/api/gtfs.html).
- Not supported yet: `transfers.txt` and time-dependent routing.

#### Shortest path reconstruction

- `single_source_dijkstra_path(graph, source_node, target_node, *, weight, cutoff, reverse)` returns one shortest path as a list of node ids.
- `multi_source_dijkstra_path` takes node ids or GeoDataFrames of origins and destinations, `mode="all_to_all"` or `mode="pairwise"`, a `threshold` and `max_workers`, and reconstructs paths in parallel Numba kernels.
- `path_to_edges` turns a node path into an ordered GeoDataFrame of graph edges, taking the lightest of parallel edges, ready for mapping.
- All three are also available as `UrbanGraph` methods.

#### Transport modes and travel time

- `TransportSpec` gains `base_speed_kmh` (free-flow speed of the mode), `dwell_min` (time standing at a stop, added per segment) and `avg_wait_time_min` (wait on OSM boarding edges).
- On segments too short to reach full speed, the peak speed scales as `sqrt(L / span)`, so short hops no longer all take the same time.
- Default specs are fitted against GTFS timetables matched to OSM. Default waits are medians across cities of timetable waits over regular services (bus 8.2 min over 115 cities, tram 4.95, subway 3.02, train 7.71). Monorail and share taxi borrow the subway and bus profiles, because no calibration city publishes a timetable for them.

| Mode | Base speed, km/h | Wait, min | Dwell, min |
|---|---|---|---|
| bus | 41.0 | 8.2 | 0.475 |
| trolleybus | 20.0 | 5.92 | 0.325 |
| tram | 41.5 | 4.95 | 1.2 |
| subway | 49.0 | 3.02 | 0.35 |
| monorail | 49.0 | 3.0 | 0.35 |
| taxi (share taxi) | 41.0 | 8.0 | 0.475 |
| train (`DEFAULT_REGISTRY_W_TRAIN`) | 50.5 | 7.71 | 0.375 |

- `OSM_ROUTE_ALIASES` in `iduedu.constants.transport_specs` maps OSM route tags to registry modes: `light_rail` -> `tram`, `share_taxi` -> `taxi`, while `monorail` keeps its own mode.
- Every mode except the subway is parsed from OSM the same way, so any mode in the registry reaches the graph.

### Bug fixes

#### OSM public transport routes

- Route pieces are ordered by cheapest insertion with local search. The old chain growing from both ends could glue a mid-route detour onto the far end and draw a straight line across the city (14.4 km in Delhi); this dated back to v1.1.0.
- A route is cut into parts at gaps over 1 km instead of bridging a hole in the relation with a straight line.
- Direction follows the stop order when at least three stops near the path agree by a clear majority, otherwise the ways. A first way drawn backwards no longer reverses the whole route. Share of one-way streets driven the legal way: Delhi 91.2% -> 97.5%, Saint Petersburg 84.1% -> 99.8%, Moscow 80.3% -> 99.3%.
- Speeds stay aligned with segments when pieces are reversed or joined.
- Across the 124 study cities with a public transport graph all graphs build, road edges with a straight segment over 1.5 km fall from 1,866 to 558 (the rest is OSM way geometry), and 1,037 routes lose their teleports.
- Subway routes parse in territories without stop area data.
- Route relations with no members, or with members lacking a ref, no longer abort the city's graph.

#### Graph editing and intermodal graphs

- An object that coincides with the graph is mapped onto the existing node instead of getting an isolated node with a zero-length connector, and connector edges are returned when every object snaps to an existing node.
- Equally distant nearest edges are resolved deterministically (an endpoint wins, then the lower node id) instead of raising `ValueError`.
- Intermodal graphs are composed with `join_urban_graphs`, and edge keys `(u, v, k)` stay unique across walk and public transport edges.
- Platforms out of reach of the walk network are logged, and a warning names the public transport node types dropped by the largest-component filter.
- `write_urban_graph` writes tables whose object columns mix numbers and strings, as OSM tags often do, instead of failing with `ArrowInvalid`.

### Documentation

- New [GTFS guide](https://iduclub.github.io/IduEdu/api/gtfs.html) and API reference for shortest path reconstruction; the shortest path notebook maps intermodal routes by transport type.
- README, high-level API and transport registry pages describe GTFS graphs and the new speed model.
- Documentation is published at https://iduclub.github.io/IduEdu/.

### Project

- The repository moved to [IDUclub/IduEdu](https://github.com/IDUclub/IduEdu); old GitHub links redirect.
- Releases run only in the official repository after Tests and Coverage pass on `main`, and the documentation is rebuilt after each release.
- Coverage is reported in pull request comments with an in-repo badge instead of Codecov.
- Added `CITATION.cff`.
- `paper2026/` holds the reproducibility pipeline, benchmarks and figures of the accompanying paper; it is not shipped in the wheel or sdist.
- No new runtime dependencies; Python 3.11-3.12 as before.

---

**Full diff**: [v2.0.0...v2.1.0](https://github.com/IDUclub/IduEdu/compare/v2.0.0...v2.1.0)


## v2.0.0 (2026-07-07)

### Features

- **graph**: Migrate IduEdu to UrbanGraph ([#21](https://github.com/IDUclub/IduEdu/pull/21),
  [`1f6781e`](https://github.com/IDUclub/IduEdu/commit/1f6781e405d2fd576410c5b3cd420ffd2eecc262))

* chore(todos): - added todos for feature release

* more TODOS

* refactor(od_matrix) wip: - some todos - perf upgrade in matrix_builder.py

* refactor: - Graph types refactor WIP

* fix(config): OVERPASS_USER_AGENT naming changed

* feat(urban_graph) WIP: - oneway attribute in directed graph

* feat(graph)!: introduce UrbanGraph core and flatten package layout Move iduedu out of the src
  layout and add the UrbanGraph graph core under iduedu.graph.

This starts the migration away from NetworkX as the primary internal graph contract while keeping
  compatibility work separate.

BREAKING CHANGE: internal package paths and repository layout changed from src/iduedu to iduedu.

* feat(graph)!: migrate OSM builders to UrbanGraph - return UrbanGraph from OSM graph builders
  without compatibility wrappers - add UrbanGraph crs/type metadata, empty graph support and
  clip/relabel helpers - move NetworkX adapters out of graph_transformers - update Overpass parsers
  to emit GeoDataFrame nodes/edges with geometry - propagate oneway flags from parsers and remove
  duplicate bidirectional PT edges - drop NetworkX from the package dependency surface

* feat(graph)!: migrate intermodal routing to UrbanGraph - migrate intermodal graph composition from
  NetworkX to UrbanGraph - add UrbanGraph join, projection and direction-transform helpers - expose
  routing, graph editor and transformer APIs through public package API - move OD/search exports to
  the routing module and drop old matrix API surface - keep legacy NetworkX utilities behind
  optional compatibility imports - switch package metadata from Poetry config to PEP 621/uv with
  hatchling backend

* feat(graph): amend commit

* feat(graph)!: add UrbanGraph components and reorganize graph modules - add connected, weakly
  connected and strongly connected component algorithms for UrbanGraph - add numba-backed CSR
  utilities for component traversal and shortest-path search - move UrbanGraph validation and
  adjacency construction out of the core class - expose component, editor, transformer and
  shortest-path APIs through graph and package API - move routing search internals into
  graph.shortest_paths and _numba - rename graph and overpass modules to shorter public paths -
  restore largest-component filtering in drive, walk and intermodal builders - move UrbanGraph
  validation and weighted/boolean adjacency construction out of the core class

* test(graph): expand UrbanGraph coverage and CI test workflow

- split tests into unit and network layers with explicit pytest markers - add synthetic graph
  factories and unit coverage for adjacency, components, adapters, editors, shortest paths, nx utils
  and UrbanGraph methods - add direct Numba kernel tests and enable coverage reporting without
  pragma exclusions - update intermodal builder regressions for PT attribute collapsing and walk
  graph component defaults - split default transport registries into base and train-enabled variants
  - update Makefile and CI to use uv, format checks, coverage XML and Codecov OIDC upload

* feat(graph)!: migrate public API to UrbanGraph - refresh README and API docs for the
  UrbanGraph-based API - document uv-based development, network coverage and automated releases -
  expand graph, Overpass and builder test coverage - update CI, docs and release workflows for
  semantic-release

* feat(graph): add UrbanGraph archive IO and helper methods

- add .urbangraph read/write support with parquet tables and metadata - expose
  read_urban_graph/write_urban_graph and UrbanGraph.read/write APIs - move nearest node lookup into
  graph input utilities and expose it publicly - add public graph validation helper and
  UrbanGraph.validate - document UrbanGraph migration, archive IO and uv-based contribution notes -
  cover nearest-node, validation and .urbangraph roundtrip workflows with tests

* fix(graph): preserve float weights and refresh API docs

- keep numba shortest-path weights as float32 in original graph units - JSON-encode non-scalar
  object columns in .urbangraph parquet archives - translate graph, builder, Overpass and config
  docstrings to English - add missing docstrings for public helpers and numba CSR wrappers - enforce
  LF line endings with gitattributes and editorconfig - update tests for float32 shortest-path
  outputs and UrbanGraph IO roundtrips

* fix(graph): preserve route-specific PT nodes and nearest-node order

- remap public-transport parser node ids before grouping to avoid merging shared OSM stop ids across
  routes - split PT access links into directed alighting and boarding edges with zero-cost alighting
  - filter leaked unrequested ground PT route types before parsing - preserve input order when
  snapping objects with nearest_nodes via spatial index - cover PT node remapping, boarding
  direction and nearest-node ordering with unit tests

* docs: add benchmark paper materials and examples - add the paper2026 benchmark suite, result
  tables and generated figures for build, intermodal and OD workloads - add README and documentation
  benchmark summaries with pedestrian graph comparisons and representation-size wording - expand
  runnable documentation notebooks for graph construction, operations, connectivity, object
  projection, shortest paths and transport registry - refresh API docs and docstrings with links
  from methods to notebook examples - add repository agent guidance and quiet default logging for
  documentation workflows

### Breaking Changes

- **graph**: Internal package paths and repository layout changed from src/iduedu to iduedu.


## v1.2.2 (2026-05-14)

### Bug Fixes

- **config**: (#20)
  ([`7809972`](https://github.com/IDUclub/IduEdu/commit/780997245211ee84710a6622a4d258a41b71585d))

OVERPASS_USER_AGENT naming changed

### Chores

- Version bumped
  ([`3e51b7d`](https://github.com/IDUclub/IduEdu/commit/3e51b7d8e5c7ffc732857a406e80dbc5cdb0efe8))


## v1.2.1 (2026-02-12)

### Bug Fixes

- **dependencies**: - added requests to required dep (lost earlier)
  ([`e51214b`](https://github.com/IDUclub/IduEdu/commit/e51214b1bce2b534201c23d017ce54d766aeef02))


## v1.2.0 (2026-01-13)

### Features

- **pt speed**: Wip
  ([`db42c30`](https://github.com/IDUclub/IduEdu/commit/db42c302c2a1dcb89051fdfbefd662be6b02fc6b))

- **pt speed**: Wip
  ([`dc0b059`](https://github.com/IDUclub/IduEdu/commit/dc0b059550fc7eb9f736a0071e93fa4f751eb501))

- **pt_routes**: - ways speed in overpass response data
  ([`fa4a826`](https://github.com/IDUclub/IduEdu/commit/fa4a82683c201d417043de711b2e2ba2bc59f9e7))

### Refactoring

- **pt_parser**: - new interface for public transport graph, get_all_public_transport_graph and
  get_single_public_transport_graph will be deprecated
  ([`266b065`](https://github.com/IDUclub/IduEdu/commit/266b065109ab7e0d585f1915928e4734bfd07e13))

- **pt_parser**: Wip
  ([`848dbc8`](https://github.com/IDUclub/IduEdu/commit/848dbc80a0966f668ba854681bcadce7efd16f95))

- **pt_routes**: - ground public transport parser to edgenode refactor
  ([`07acc47`](https://github.com/IDUclub/IduEdu/commit/07acc47d0066b276582d470ec808cf0f717c38a7))


## v1.1.0 (2025-12-05)

### Bug Fixes

- Overpass_backoff_base default set to 2 sec
  ([`bfbdfb1`](https://github.com/IDUclub/IduEdu/commit/bfbdfb1a2541d649c76301b3987707520e4134b2))

- **0.5.1**: Removed duplicated nodes
  ([`d7d582e`](https://github.com/IDUclub/IduEdu/commit/d7d582ea697cfc2f2b1cfd4a46bbd6786226e210))

- **0.5.2**: Fix KeyError
  ([`95e9cec`](https://github.com/IDUclub/IduEdu/commit/95e9cec3ded787feed9d37e801cdbf788c2e8031))

- **0.5.4**: Fixed platform projections
  ([`21a749b`](https://github.com/IDUclub/IduEdu/commit/21a749b6abed76355c45232fc7b8390422452f5e))

- **intermodal_walk_builders**: - order in concating edges with their reverse copy for correct uvk
  in graph
  ([`e982494`](https://github.com/IDUclub/IduEdu/commit/e9824944c1213f7b75e4be012d0e776b8cf760f9))

- **matrix_builder**: - version 0.5.8
  ([`e36b9d0`](https://github.com/IDUclub/IduEdu/commit/e36b9d073822fe624aba2aeabf3339dbf888018f))

- **matrix_builder**: Added force node relabeling on matrix validation
  ([`ca6e38a`](https://github.com/IDUclub/IduEdu/commit/ca6e38a2961c2b467f54d7fe863de2dbf8f2a83c))

- **reg_status**: Replaced deprecated code
  ([`9691f42`](https://github.com/IDUclub/IduEdu/commit/9691f4231a3b0e436f206290d65c870dc7ac1999))

### Code Style

- Added line
  ([`094841d`](https://github.com/IDUclub/IduEdu/commit/094841d24ead13d2a7acb1078b6b7ef2df3bbe9f))

- Black fix
  ([`b50c2e7`](https://github.com/IDUclub/IduEdu/commit/b50c2e77faf031160dd0fc3346b7a7f06160e896))

- Blacked
  ([`7db7b6c`](https://github.com/IDUclub/IduEdu/commit/7db7b6c4f75fae02249b80d18bd73153c513af88))

### Continuous Integration

- Fixing deploy-docs step
  ([`8de4677`](https://github.com/IDUclub/IduEdu/commit/8de4677e4d81a1d92f861c739e425dc41c9a5d99))

- Fixing deploy-docs step
  ([`7c00608`](https://github.com/IDUclub/IduEdu/commit/7c006087384f1091e48d304985a1b054b7c41bed))

- Fixing deploy-docs step
  ([`e3aa296`](https://github.com/IDUclub/IduEdu/commit/e3aa2965e2517adac2359a53ae9a9ca36faeea22))

- Removed pyarrow install
  ([`7ecf66b`](https://github.com/IDUclub/IduEdu/commit/7ecf66b04285d72854d6ac512b31d3ebe559bb92))

- Removed pyarrow install
  ([`e45f33d`](https://github.com/IDUclub/IduEdu/commit/e45f33d1e05e937e5302c05056e7a31aecfb083e))

- Removed wrong dependency group
  ([`3eb44d5`](https://github.com/IDUclub/IduEdu/commit/3eb44d50e7cc64ef21d447c2b59e4f84320e7a0a))

### Documentation

- - README.md update
  ([`0972371`](https://github.com/IDUclub/IduEdu/commit/09723715359c69f01d4c47ee604456a93749a035))

- Added info about caching
  ([`bccc69e`](https://github.com/IDUclub/IduEdu/commit/bccc69e645ae7c342efb144b150cdbf5902539df))

### Features

- **0.4.0**: Tests & train enum
  ([`bd7b500`](https://github.com/IDUclub/IduEdu/commit/bd7b50021becdbebad5a4b94a0e76c69bc2e4a22))

- **graph_transformer**: Added module for converting graph to GeoDataFrame
  ([`253fbdb`](https://github.com/IDUclub/IduEdu/commit/253fbdb8ea19a2b0b6cf3c54a6f5f4f354bc8cc4))

- **matrix_builder**: New Numba accelerated matrix computation instead networkit spsp
  ([`24cb5ec`](https://github.com/IDUclub/IduEdu/commit/24cb5ec5ce32204c7f227cadc778453a621fa9df))

- **overpass_cache**: - caching overpass requests
  ([`e64d746`](https://github.com/IDUclub/IduEdu/commit/e64d7468feee6ace0e40f2a8a5e3c323650c0f3a))

### Refactoring

- **0.5.3**: Added coordinate rounds in any graph geometry
  ([`7c8971a`](https://github.com/IDUclub/IduEdu/commit/7c8971a4396f00008ebca158527a0acc57f25d71))

- **get_any_graph**: Added crs as optional and exception
  ([`8b1c371`](https://github.com/IDUclub/IduEdu/commit/8b1c3719a9b14a7d64fbfc27fee986e8c60e07a7))

- **get_any_graph**: Updated example
  ([`7808f0a`](https://github.com/IDUclub/IduEdu/commit/7808f0a36e51a444ab971ae1c1e65090e31807ef))

- **graph_transformer**: Edges and nodes to gdf made shorter
  ([`7f1b969`](https://github.com/IDUclub/IduEdu/commit/7f1b969cd113607e720b96a49db78639525cab0a))

### Testing

- - added conftest.py
  ([`20f3720`](https://github.com/IDUclub/IduEdu/commit/20f372001bfb5067cf48774c5f6767b78ef9c8c9))

- - added some tests
  ([`560aa82`](https://github.com/IDUclub/IduEdu/commit/560aa8246da4b674e79275e6922de126ec9dce96))
