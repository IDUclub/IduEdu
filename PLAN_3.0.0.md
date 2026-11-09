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
6. [~] **Other utilities in Rust** where measurements justify it (see "Survey of existing utilities" below):
   PT route segmentation, edge splitting in `project_objects2urban_graph`; PBF reading after a separate measurement.

## Stage 2. New functionality (new kernels in Rust from the start)

7. [ ] **Utilities**: `nearest_edges` (public); `unique_segments` (deduplicated by elementary segments — consecutive
   coordinate pairs with sorted endpoints, so partly shared there-and-back geometries count once);
   `consolidate_intersections` (KD-tree single-linkage clusters, split by connectivity inside each cluster, only edge
   end points moved, mapping old → new nodes).
8. [ ] **`stats`**: global, per zone (GeoDataFrame of polygons) and local (within a network radius) metrics; grouping by
   `type` / `category`; boundary dead ends excluded. Size and length (by unique segments), streets per node and its
   distribution, intersection count and density, dead-end share, street density, mean segment length, edge circuity,
   β / γ / cyclomatic number / meshedness, orientation entropy, self-loops and multi-edges, components; drive: one-way
   share, length by category, length-weighted speed; PT: stops and routes by mode, stop spacing; intermodal:
   connectors and unattached platforms. Route directness on an OD matrix.
9. [ ] **Aggregations**: k nearest objects; sum / mean / count within a cutoff with optional decay; diversity of
   categories (count, Shannon, Hill); `direction="from" | "to"` via the transposed matrix.
10. [ ] **Centralities**: reach, harmonic closeness, gravity, betweenness (Brandes) for nodes and edges; several
    thresholds in one call; buffer against edge effects.
11. [ ] **Elevation** from a raster or a GeoDataFrame (points / contours): node heights, signed edge grade, maximum
    grade along the geometry, bridges and tunnels interpolated between the ends, ascent / descent, Tobler walking
    speed, relief metrics in `stats`.
12. [ ] **`get_bike_graph`**: cycle infrastructure from OSM tags, level of traffic stress (LTS), optional grade penalties.
13. [ ] **Bidirectional Dijkstra** for point-to-point queries.

## Stage 3. Release and after

14. [ ] "Upgrading to 3.0" docs page, README/AGENTS/CONTRIBUTING updated, Russian translation (`docs-i18n`,
    `docs-i18n-check`), delete this file, squash `dev` → `main` as `feat!: ...` → 3.0.0.
15. [ ] 3.1.0: remove the shims deprecated in 3.0.0.
16. [ ] Docs rework along ObjectNat's structure.
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
