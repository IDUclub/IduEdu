# IduEdu 3.0.0 plan

Working document for the `dev` branch. Everything is committed straight to `dev` (no pull requests into it); at the
end `dev` is squash-merged into `main` with a `feat!:` title and a `BREAKING CHANGE:` footer, which makes
python-semantic-release cut **3.0.0**. Delete this file before that merge.

Status legend: `[ ]` to do, `[~]` in progress, `[x]` done.

## Compatibility policy

- 3.0.0 is a major release: the Numba core is replaced by Rust, and the API may change.
- Everything deprecated in 2.4.1 with "will be removed in IduEdu 3.0" is removed in 3.0.0:
  `single_source_dijkstra_path_length`, `multi_source_dijkstra_path_length`, `dijkstra_path_length_parallel`,
  `multi_source_dijkstra_path` (functions and `UrbanGraph` methods).
- Anything else that 3.0.0 renames or replaces keeps a shim that raises `DeprecationWarning`
  ("... is deprecated and will be removed in IduEdu 3.1.0") and is removed in 3.1.0.
- An "Upgrading to 3.0" page in the docs lists every removal and every deprecation (as ObjectNat does).

## Stage 1. Rust core instead of Numba

Benchmark that motivates it: Rust 7–12x faster than the current Numba Dijkstra on one thread and 5–13x on 16-thread
OD workloads, no JIT warm-up (the Numba cache never hits today: ~5.7 s per fresh process), 130 MB less to install
(numba + llvmlite). Prototype with identical results: `.claude/worktrees/agent-acf223dbc4b5c170f/bench/rust_prototype/`
(`REPORT.md` there).

1. [x] **Packaging and CI**, as in ObjectNat:
   - maturin build backend, crate `rust/` (`iduedu-rs`), extension module `iduedu._rust`, PyO3 abi3-py311, rayon;
     release profile `lto = "fat"`, `codegen-units = 1`;
   - `[tool.uv] cache-keys` so editable installs rebuild after Rust edits;
   - `quality.yml`: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test`; Python jobs build the extension;
   - `release.yml`: version job (semantic-release bumps `pyproject.toml` and `rust/Cargo.toml`, builds the sdist with
     maturin) + five abi3 wheels (linux x86_64/aarch64, windows x64, macOS arm64/x86_64) + PyPI Trusted Publishing;
   - `quality.yml` / `docs.yml` also run on pushes to `dev`;
   - Makefile, AGENTS.md, CONTRIBUTING.md: Rust toolchain, `maturin develop`, rules for where kernels live.
2. [x] **Kernels** (from the prototype, cleaned up):
   - `CsrGraph` (`#[pyclass]`): owns the CSR arrays (validated once) and a pool of per-search scratch buffers;
     cached on `UrbanGraph` next to the SciPy matrix, a second one for the transposed matrix (reverse searches);
   - Dijkstra: one or many sources, cutoff, early exit on targets, predecessors, owner (nearest source), parallel
     searches over sources with shared or per-source targets, dense OD rows; GIL released, rayon pool sized by
     `max_workers`;
   - path reconstruction in batches;
   - connected / weakly / strongly connected components.
3. [x] **CSR builder in Rust** with min/max reduction of parallel edges. This fixes the bug where undirected
   multigraphs summed parallel edges of different orientation (walk graph: 15→16 0.114 min + 16→15 2.213 min gave
   2.327 min). Vectorized node-id lookup in Python (`Index.get_indexer`) instead of a generator over a dict.
   Regression test with parallel edges in both orientations.
4. [x] **Remove Numba**: `iduedu/_numba/`, the dependency, `NUMBA_DISABLE_JIT` in `conftest.py`, `nb.set_num_threads`;
   before removal, freeze reference results of the Numba kernels as test fixtures and compare the Rust kernels to them.
5. [x] **Python wrappers**: drop the per-call `_prepare_numba_graph`; faster `_distance_frame` and result assembly
   (74–96 % of `dijkstra_parallel` / `single_source_dijkstra` time is Python today); remove the 2.4.1 deprecations.
6. [x] **Other utilities in Rust** where measurements justify it (see "Survey of existing utilities" below):
   - [x] PT route segmentation: the cost was algorithmic (quadratic `substring`, `iterrows`, `Series.map(dict)`,
     a lambda per group), not computation; fixed in Python/numpy (`iduedu.graph.lines`), 89 → 28 s on
     St Petersburg with byte-identical output. Rust would add nothing here.
   - [x] `project_objects2urban_graph`: 2.1 s per 20k objects without the profiler, spread over pandas/shapely
     overhead; a vectorized rewrite gave identical output and no speed-up, so it stays as is.
   - [x] PBF reading of street ways in Rust (`rust/src/pbf.rs`, `osmpbf`): two parallel passes (filtered
     ways, then their node locations), output identical to pyosmium; `PbfSource.ways_table` hands the builders
     coordinate arrays instead of Overpass point dicts. St Petersburg: walk ways 20.2 → 0.64 s, `get_walk_graph`
     49.6 → 28.5 s, `get_drive_graph` 16.2 → 5.5 s, identical graphs.
   - [x] pyosmium dropped entirely: `PbfFile` (`rust/src/pbf.rs`) reads nodes, ways and relations with a block
     index of the first pass (later queries read only the blocks they need), `rust/src/pbf_write.rs` writes
     `.pbf` (zlib, dense nodes, no metadata) for `split_by_polygons`. NW Russia extract: split for St Petersburg
     and Leningrad oblast 86.5 → 17.2 s with identical elements (1.6M nodes, 425k ways, 1854 relations);
     `routes_by_poly` on St Petersburg 6.7 → 1.8 s, identical output. Only `.osm.pbf` is read now (`.osm`/`.opl`:
     convert with `osmium cat`); the `pbf` extra stays empty until 3.1, `location_storage` is deprecated.
   - [x] Edge merging rewritten (`rust/src/chains.rs`): segments are joined into junction-to-junction edges only
     where they are alone at a point, follow one another and carry equal tags, instead of `line_merge` over
     everything and a nearest-midpoint join that lost the tags of all but one way. Bridges and tunnels always
     stay separate edges and are kept as columns. St Petersburg: `get_walk_graph` 28.5 → 8.2 s, `get_drive_graph`
     5.5 → 3.5 s; length, intersections and dead ends unchanged, +37k (walk) and +7.6k (drive) two-street nodes
     where tags change.

## Stage 2. New functionality (new kernels in Rust from the start)

7. [x] **Utilities**: `nearest_edges` (public); `unique_segments` (deduplicated by elementary segments — consecutive
   coordinate pairs with sorted endpoints, so partly shared there-and-back geometries count once);
   `consolidate_intersections` (KD-tree single-linkage clusters, split by connectivity inside each cluster, only edge
   end points moved, mapping old → new nodes).
8. [x] **`stats`** (`iduedu/graph/stats.py`): `graph_stats` for the whole graph, an `area` or `zones` (node degrees
   from the whole graph, boundary dead ends excluded), `edge_stats` by any edge column, `transport_stats` per mode,
   `local_stats` within a network radius, `streets_per_node`, `route_directness`. St Petersburg walk graph
   (480k nodes): whole graph 2.0 s, 400 zones 3.8 s, local stats of every node within 500 m 1.7 s.
9. [x] **Aggregations** (`iduedu/graph/aggregations.py`, kernels in `rust/src/analysis.rs`): `nearest_objects`,
   `aggregate_objects` (count / sum / mean, several cutoffs, linear / exp / inverse decay), `diversity` (richness,
   Shannon, Hill). Sums run from the smaller side: per origin, or per object on the reversed graph.
   50k origins × 2k objects on the 533k-node intermodal graph: 0.2 s.
10. [x] **Centralities** (`iduedu/graph/centrality.py`): `centrality` (reach, harmonic, gravity in one search per node;
    Brandes betweenness, one backward sweep per threshold; node weights; `sample`; `area` against edge effects) and
    `edge_betweenness`. Equal to networkx. Drive graph (24.6k nodes) full betweenness 4.3 s; walk graph (480k)
    reach + harmonic + gravity at 400 / 800 m 4.2 s, betweenness at 400 / 800 m 5.4 s.
11. [x] **Elevation** (`iduedu/graph/elevation.py`, extra `elevation` = rasterio): `add_elevation` from a raster,
    points, contours or a function; `grade`, `grade_max`, `ascent`, `descent`; bridges and tunnels interpolated
    between their ends; `apply_grade_speed` (Tobler, cycling, custom); relief measures in `graph_stats`.
    Walk graph of St Petersburg from a 30 m raster: 3.1 s. Checked on Copernicus GLO-30: the method works
    (structures get straight profiles), but GLO-30 is a surface model and measures roofs in built-up areas
    (historic centre: ground 2–5 m, raster median 13.7 m), so street grades need a terrain model (DTM).
12. [x] **`get_bike_graph`** (`iduedu/graph_builders/bike_builders.py`): `cycle_infra`, `lts` 1–4 (simplified
    Mekuria, Furth and Nixon 2012), bicycle one-way rules, `max_lts`, `elevation`. St Petersburg: 10.8 s,
    360k nodes, 20.3k km.
13. [x] **Point-to-point routes** (`rust/src/bidirectional.rs`, `astar.rs`, `yen.rs`): `shortest_path` with
    `method="bidirectional"` (default), `"astar"` or `"dijkstra"`; `k_shortest_paths` (Yen, Lawler's rule, spur
    searches in parallel on the bidirectional search); `od_matrix(mode="pairwise", method=...)`, where `"auto"`
    takes the bidirectional search when most origins occur once. A* estimates the rest as straight distance times
    the smallest cost per metre over the graph's own edges of non-zero cost; edges of zero cost (four 1–4 cm edges
    of the walk graph whose rounded time is 0) add a slack that keeps it exact. St Petersburg walk graph, pairs
    4 km apart: Dijkstra 2.3 ms, bidirectional 1.0 ms, A* 0.5 ms by length / 0.8 ms by time; random pairs across
    the city gain less (the one-sided search is cut by the city border). Intermodal graph by time: bidirectional
    2.2x, A* none (metro sets the scale). Yen on the walk graph: 0.57 s for k=3 on a 536-node route; the
    alternatives differ by tenths of a metre (other sidewalk).
    - [x] **ALT** (`rust/src/alt.rs`, `method="alt"`): 16 landmarks, farthest first, tables cached with the
      matrix (walk graph 0.7 s, 33 MB). Walk by time: 10 km pairs 0.83 ms (Dijkstra 11.7, bidirectional 4.8,
      A* 3.7), random pairs 1.9 ms (18.0); drive 2–5x faster than bidirectional; intermodal by time on par
      with bidirectional, 1.7x on random pairs.
    - [x] **Penalty alternatives** (`rust/src/penalty.rs`, `alternative_paths`): edges of every route x1.5 in
      both directions, kept within 1.4x the shortest cost and 0.7 shared cost. 4.9 km walk: 4.9 / 5.3 / 5.5 km
      with 0–12 % shared streets (Yen: three routes within 1.7 m), 2–8 ms. Sidewalks on both sides of an
      avenue are separate edges, so a route along the other sidewalk counts as another street.

## Stage 3. Release and after

14. [ ] "Upgrading to 3.0" docs page, README/AGENTS/CONTRIBUTING updated, Russian translation (`docs-i18n`,
    `docs-i18n-check`), delete this file, squash `dev` → `main` as `feat!: ...` → 3.0.0.
    Breaking changes to list there (so far):
    - the four 2.4.1 routing helpers are removed;
    - `.osm.pbf` only (no `.osm` / `.opl`), split files carry no element metadata; `location_storage` and the
      `pbf` extra are deprecated;
    - simplified drive / walk graphs: an edge per run of equal tags (more nodes and edges), `bridge` and `tunnel`
      always kept;
    - `clip_urban_graph`, `UrbanGraph.clip`, `clip_nx_graph`: a bare shapely polygon is read in EPSG:4326
      (pass `gpd.GeoSeries([polygon], crs=graph.crs)` for metres);
    - walk graph weights: parallel edges are no longer summed (shorter routes, the bug fix of step 3).
15. [ ] 3.1.0: remove the shims deprecated in 3.0.0 (`location_storage` of `PbfSource`, `split_by_polygons`,
    `download_osm_pbf`; the empty `pbf` extra).
16. [x] Docs rework along ObjectNat's structure: building / graph / routing / analysis pages (Overview, Inputs,
    Method, Outputs, Functions, Figures, Limitations, References, Example notebook), figures from
    `docs/figures/make_figures.py` on the `assets` branch, notebooks `routing_methods`, `network_analysis`,
    `elevation`, function index, upgrade page; README / AGENTS / CONTRIBUTING updated; Russian complete.
    The performance page is a stub until new benchmarks.
17. [ ] Design document for schedule-based routing (CSA / RAPTOR in Rust).

Not planned now: GTFS feed metrics (headways, frequency).

Every step: unit tests on factories, a network test on real data where it matters, docs page with the Russian
translation, `graphify update`, black / isort / pylint, commits without co-author lines.

## Survey of existing utilities

Measured on St Petersburg (relation 337422) built from the Geofabrik extract: walk 480k nodes / 661k edges,
intermodal 533k / 785k, Python 3.13, Numba 0.67.

| piece | time | where it goes | Rust? |
|---|---:|---|---|
| `get_public_transport_graph` | 84 s | `overpass_ground_transport2edgenode` 63 s: per-route shapely `substring` (35k calls, 17 s), coordinate iteration (20M points, 9 s), pandas Series per stop; `_graph_data_to_urban_graph` 14 s (pandas) | **yes**: linear referencing of stops on a route and cutting it into stop-to-stop segments on coordinate arrays. The biggest win found |
| `project_objects2urban_graph`, 20k objects | 4.8 s | per-edge `substring` and `line_interpolate_point` when edges are split at projections | **yes**, same "cut a line at distances" kernel as the PT routes; matters for ObjectNat (buildings) |
| `get_walk_graph` | 66 s | PBF reading through pyosmium callbacks and per-way shapely geometry (~2.1M small shapely calls, 14 s), `STRtree.query_nearest` 6.6 s, `line_merge` 4.5 s, `MultiLineString` assembly 6.1 s, `_build_nodes_and_uv` 2.1 s, largest component 2.4 s | **maybe**: topological chain merging on node ids instead of `line_merge`, and a Rust PBF reader (`osmpbf` crate) that could replace pyosmium and the `pbf` extra. Large scope: measure the reading share separately first |
| `get_drive_graph` | 22 s | the same PBF reading, on fewer ways | follows the walk decision |
| `update_adjacency_matrix` | 1.3 s | `np.fromiter` over a dict 0.7 s, pandas `groupby` | covered by the Rust CSR builder (step 3) |
| `connected_components` | 1.4–1.8 s | boolean matrix build 0.65 s, Python grouping of labels 0.5 s | Rust labels + vectorized grouping (step 2) |
| `subgraph_by_nodes`, `validate`, `relabel`, `clip`, `simplify_multiedges` | 0.5–1.0 s each | pandas | no: vectorize in Python where needed |
| `nearest_nodes`, 20k points | 0.2 s | GEOS STRtree | no |
| Overpass JSON parsing | — | network-bound | no |
| GTFS builders | not measured | — | measure on a large feed before deciding |

So step 6 is: (a) PT route segmentation in Rust, (b) the shared line-cutting kernel for `project_objects2urban_graph`,
(c) a separate measurement of PBF reading and simplification before deciding on a Rust OSM reader.
