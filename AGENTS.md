# AGENTS.md

Guidance for any coding agent (Claude Code, Cursor, Copilot, Codex, Aider, Windsurf, etc.) working in this
repository. All commands below are plain shell/CLI and assume no specific agent or IDE.

## graphify

graphify builds a knowledge graph of the codebase in `graphify-out/` with god nodes, community structure, and
cross-file relationships. Prefer it over blind grep/file-walking when answering questions about the codebase.

`graphify-out/` is git-ignored and is not shipped with the repository. Generate it manually when you start working
with an agent: use the `/graphify .` command or skill if your runtime has one, or run
`uv tool run --from graphifyy graphify update .` (AST-only, no API cost). The rules below apply once it exists.

If your runtime exposes a `/graphify` command or a graphify skill, use it first. Otherwise run the CLI through
uv tool as `uv tool run --from graphifyy graphify <command>` (the PyPI package is `graphifyy`, while the executable
is `graphify`). If uv cannot run the tool, fall back to reading `graphify-out/graph.json` or
`graphify-out/GRAPH_REPORT.md` directly.

Rules:

- For codebase questions, first run `uv tool run --from graphifyy graphify query "<question>"` when
  graphify-out/graph.json exists. Use `uv tool run --from graphifyy graphify path "<A>" "<B>"` for relationships and
  `uv tool run --from graphifyy graphify explain "<concept>"` for focused concepts. These return a scoped subgraph,
  usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip
  graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to
  use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough
  context.
- After modifying code, run `uv tool run --from graphifyy graphify update .` to keep the graph current
  (AST-only, no API cost).

# Project guide

## Project Overview

**IduEdu** is a Python library for building and analyzing multi-modal city transport networks from OpenStreetMap and
GTFS Schedule data. It downloads OSM data via Overpass API (or reads local `.osm.pbf` files with its Rust
reader), reads local GTFS feeds, constructs `drive`, `walk`, and `public transport` graphs, joins them into intermodal
networks, and computes OD-matrices and routes with Dijkstra kernels written in Rust.

Python 3.11–3.14. Package manager: **uv** (`uv.lock` is generated locally and not committed). Build
backend: maturin; the graph kernels are a Rust crate in `rust/` compiled into the extension `iduedu._rust` (PyO3, abi3),
so a stable Rust toolchain (1.82+) is needed to build from source.

## Commands

```bash
# Install dev dependencies (builds the Rust extension; rerun after editing rust/)
uv sync --all-groups

# Rust core: format, lint, unit tests (or: make rust-check)
cd rust && cargo fmt --check && cargo clippy --release --all-targets -- -D warnings && cargo test --release

# Run all unit tests (fast, no network)
uv run pytest

# Run a single test file
uv run pytest tests/test_urban_graph_unit.py -v

# Run a single test by name
uv run pytest tests/test_urban_graph_unit.py -k "test_name" -v

# Run network tests (calls Overpass API — requires internet)
uv run pytest --run-network

# Network tests on the Geofabrik extract instead of Overpass (auto, the default, falls back to it when Overpass is down)
uv run pytest --run-network --osm-source=pbf

# Run with coverage
uv run pytest --cov=iduedu --cov-report=term-missing

# Format code
uv run isort iduedu
uv run black iduedu

# Lint
uv run pylint iduedu
uv run isort --check-only iduedu
uv run black --check iduedu

# Build docs (English, then Russian into html/ru)
uv run sphinx-build -b html docs docs/_build/html
uv run sphinx-build -b html -D language=ru -d docs/_build/doctrees-ru docs docs/_build/html/ru

# Russian translation of the docs: add new/changed text to the catalogues, then check nothing is untranslated
uv run python docs/i18n.py update                   # or: make docs-i18n
uv run python docs/i18n.py check                    # or: make docs-i18n-check (CI fails on untranslated text)
```

**Rust note:** `uv sync` rebuilds the extension when `rust/Cargo.toml` or `rust/src/**/*.rs` change (`[tool.uv]
cache-keys`); `uv run maturin develop --release` rebuilds it by hand. Coverage measures the Python layer only; the Rust
kernels have their own `cargo test` and the Python-level checks in `tests/test_rust_unit.py`.

## Architecture

### Core data structure: `UrbanGraph`

`iduedu/graph/urban_graph.py` — The central class. Unlike NetworkX, it stores the graph as two GeoDataFrames:

- `nodes_gdf` — point geometries, unique index = node ID
- `edges_gdf` — required columns: `u`, `v`, `geometry`, `length_meter`, `time_min`; multigraph adds `k`

The adjacency matrix (`scipy.sparse.csr_matrix`) is built lazily via `update_adjacency_matrix()` and stored on the
object. It is invalidated whenever nodes/edges change — callers must call `update_adjacency_matrix()` again after any
structural edit.

Default weight for routing: `time_min`. Alternative: `length_meter`.

### Module layout

```
iduedu/
  __init__.py        — re-exports everything from _api.py (public surface)
  _api.py            — single import hub; all public symbols come from here
  config.py          — global config (OSM source, Overpass URL, cache, rate limits, logging)
  graph/
    urban_graph.py   — UrbanGraph class (data + adjacency matrix + method dispatch)
    adjacency.py     — build_adjacency_matrix() from edges_gdf → CSR
    components.py    — connected/weakly/strongly connected component algorithms
    editors.py       — subgraph_by_nodes, clip, join, relabel, project_objects2urban_graph
    transformers.py  — keep_largest_component, simplify_multiedges, to_directed/undirected
    routing.py       — Dijkstra wrappers (single/multi-source, parallel, OD matrix; distances, paths or both)
    routing_result.py — ShortestPathResult: distances with eager or lazy (predecessor-tree) paths
    shortest_paths.py — public re-exports of the routing API, path_to_edges()
    adapters.py      — UrbanGraph ↔ NetworkX conversion
    nx_utils.py      — legacy NetworkX helpers (optional dep, soft-imported)
    validation.py    — node/edge schema validation, CRS sync
    graph_inputs.py  — nearest_nodes(), resolve_graph_nodes_input() helpers
    segments.py      — unique_segments(): every physical segment of the geometry once
    within.py        — sums of node values within a travel cost (shared by stats, aggregations, centralities)
    stats.py         — graph_stats(), edge_stats(), transport_stats(), local_stats(), route_directness()
    io.py            — .urbangraph archive read/write
  _rust.pyi        — type stubs of the compiled extension built from rust/
  graph_builders/
    drive_walk_builders.py      — get_drive_graph(), get_walk_graph()
    public_transport_builders.py — get_public_transport_graph()
    gtfs_builders.py            — get_gtfs_public_transport_graph()
    intermodal_builders.py      — get_intermodal_graph(), join_pt_walk_graph()
  gtfs/
    reader.py        — local GTFS directory/ZIP loading
    validation.py    — GTFS table and reference validation
    merge.py         — merge_gtfs_feeds(), deduplicate_across_feeds(): several feeds into one
    service_days.py  — calendar/calendar_dates → service days for average-day boarding waits
  osm_pbf/           — local .osm.pbf files instead of Overpass (read and written by the Rust core)
    reader.py        — PbfSource: answers the Overpass queries of the builders from a file; snapshot_time()
    filters.py       — Overpass tag-filter parser (parse_way_filter, WayFilter)
    extract.py       — split_by_polygons(): per-city files from a regional extract in one pass
    download.py      — download_osm_pbf(): smallest covering Geofabrik extract, cached, cut to a territory
  overpass/
    downloaders.py   — Overpass HTTP requests with retry/rate-limit; dispatch to osm_pbf in pbf mode
    parsers.py       — JSON → GeoDataFrame (nodes/edges)
    cache.py         — file-based JSON cache for Overpass responses
  constants/
    transport_specs.py — TransportSpec, TransportRegistry, DEFAULT_REGISTRY
    highway_enums.py   — HighwayType enum
    network_enums.py   — network type enums
rust/                — Rust core, compiled into iduedu._rust (see rust/README.md)
  src/csr.rs         — CSR view and validation, edge list → CSR (parallel arcs merged by min/max), transpose
  src/dijkstra.rs    — Dijkstra (multi-source, cutoff, early exit, predecessors, nearest source), route rebuild
  src/components.rs  — connected and strongly connected components
  src/analysis.rs    — sums of node values within a travel cost (gather/scatter, decay), nearest objects, betweenness
  src/pbf.rs         — .pbf reading (osmpbf, parallel, block index): ways by tag filter, nodes, relations, header
  src/pbf_write.rs   — .pbf writing: per-territory extracts (osm_pbf/extract.py) and files from element lists
  src/lib.rs         — PyO3 bindings: CsrGraph, build_csr, reconstruct_paths, *_components, PbfFile, write_pbf
```

### Data flow

```
Overpass API                       Local .osm.pbf (config.set_osm_source("pbf"))
  → downloaders.py (HTTP + cache)    → osm_pbf/ (same Overpass-shaped elements)
  → parsers.py (JSON → GeoDataFrame)
  → graph_builders/ (GeoDataFrame → UrbanGraph)

Local GTFS directory/ZIP
  → gtfs/ (tables + validation)
  → gtfs_builders.py (schedule aggregation → UrbanGraph)

UrbanGraph layers
  → intermodal_builders.py (UrbanGraph + UrbanGraph → joined UrbanGraph)
  → graph/editors.py (project objects onto graph)
  → graph/routing.py → iduedu._rust (CSR Dijkstra → distances, paths, OD matrix)
```

### Public transport speed model

`TransportSpec` (frozen dataclass) encodes per-mode physics and OSM boarding waits: `vmax_tech_kmh`, `accel_dist_m`,
`brake_dist_m`, `base_speed_kmh`, `dwell_min`, `avg_wait_time_min`, and optionally `long_speed_kmh` with
`speed_half_length_m` (free speed growing with segment length; set for tram and train). `DEFAULT_REGISTRY` covers
bus/trolleybus/tram/subway/monorail/taxi; `DEFAULT_REGISTRY_W_TRAIN` adds train. `OSM_ROUTE_ALIASES` maps OSM
`light_rail` to `tram` and `share_taxi` to `taxi`. Pass a custom `TransportRegistry` to
`get_public_transport_graph()` to override speeds or waiting times. GTFS boarding waits are schedule-derived and do not use the registry.

### Test organization

Tests are split into `_unit` (no network, fast) and `_network` (real OSM data). Network test modules are listed in
`conftest.py::NETWORK_ONLY_TEST_MODULES` and are excluded from collection unless `--run-network` is passed. They read
Overpass, or the test territory cut out of its Geofabrik extract (`northwestern-fed-district`, about 650 MB, cached in
`.iduedu_cache/geofabrik/`) when Overpass does not answer; `--osm-source=overpass|pbf|auto` forces one.
Network tests check real data only; options and edge cases belong in the unit tests with mocked responses.

Pytest markers: `unit`, `integration`, `network`, `slow`, `rust`.

Shared fixtures in `tests/conftest.py`: `territory_osm_id` (OSM relation 1114252), `osm_source` (Overpass or the
PBF fallback; a `pytest_runtest_protocol` hook switches to the file only around network tests, so unit tests in the
same run keep their mocked Overpass), `bounds`, `intermodal_graph` (session-scoped, downloaded once per network run).

Object factories for unit tests: `tests/factories.py`.

### Documentation in two languages

English is the source; Russian lives in gettext catalogues `docs/locale/ru/LC_MESSAGES/<page>.po` (pages, notebook
markdown and the docstrings autodoc shows). `docs/conf.py` sets `locale_dirs`, one catalogue per page and the
language switch (`docs/_templates/sidebar/lang.html`: the Russian site is the `ru/` subdirectory of the English one).
**Whenever English text changes — a page, a notebook text cell, a public docstring — run `docs/i18n.py update`,
translate every empty or `#, fuzzy` entry (keep markup, roles, literals and links; decimal commas in Russian prose)
and finish with `docs/i18n.py check`.** Containers of nested lists (`Text:  - a - b`) stay identical to the
English; the update step keeps them so. Translations on MyST pages (`.md`, including the `Config` docstrings on
`configuration.md`) are parsed as Markdown: roles there are `{func}`, not `:func:`.

### Coding conventions

- Line length: 120 characters (black + pylint)
- Imports sorted with isort (multi-line style 3, trailing comma)
- `isort` skips `__init__.py` files
- All public API must be reachable via `from iduedu import ...` through `_api.py`
- NetworkX is an **optional** dependency — always soft-import it inside `try/except ModuleNotFoundError` and check
  `exc.name == "networkx"` before re-raising
- Compiled kernels live in the Rust crate `rust/` (`csr.rs`, `dijkstra.rs`, `components.rs`, bindings in `lib.rs`)
  and are called only through the `graph/` layer (`adjacency.py`, `routing.py`, `components.py`, `within.py`, `centrality.py`) and `osm_pbf/` (`reader.py`, `extract.py`), never directly
  from builders or the public API. Keep `iduedu/_rust.pyi` in sync with the bindings. Kernels release the GIL and
  run parallel work on rayon; `max_workers` maps to the number of rayon threads
