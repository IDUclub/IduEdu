# AGENTS.md

Guidance for any coding agent (Claude Code, Cursor, Copilot, Codex, Aider, Windsurf, etc.) working in this
repository. All commands below are plain shell/CLI and assume no specific agent or IDE.

## graphify

This project ships a prebuilt knowledge graph at `graphify-out/` with god nodes, community structure, and
cross-file relationships. Prefer it over blind grep/file-walking when answering questions about the codebase.

If your runtime exposes a `/graphify` command or a graphify skill, use it first. Otherwise call the `graphify`
CLI directly (rules below); if the `graphify` binary is not installed, fall back to reading `graphify-out/graph.json`
or `graphify-out/GRAPH_REPORT.md` directly.

Rules:

- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use
  `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a
  scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip
  graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to
  use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough
  context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

# Project guide

## Project Overview

**IduEdu** is a Python library for building and analyzing multi-modal city transport networks from OpenStreetMap data.
It downloads OSM data via Overpass API, constructs `drive`, `walk`, and `public transport` graphs, joins them into
intermodal networks, and computes OD-matrices with Numba-accelerated Dijkstra.

Python 3.11–3.12 only. Package manager: **uv** (lockfile: `uv.lock`). Build backend: hatchling.

## Commands

```bash
# Install dev dependencies
uv sync --group dev

# Run all unit tests (fast, no network)
uv run pytest

# Run a single test file
uv run pytest tests/test_urban_graph_unit.py -v

# Run a single test by name
uv run pytest tests/test_urban_graph_unit.py -k "test_name" -v

# Run network tests (calls Overpass API — requires internet)
uv run pytest --run-network

# Run with coverage
uv run pytest --cov=iduedu --cov-report=term-missing

# Format code
uv run isort iduedu
uv run black iduedu

# Lint
uv run pylint iduedu
uv run isort --check-only iduedu
uv run black --check iduedu

# Build docs
uv run sphinx-build -b html docs docs/_build/html
```

**Numba JIT note:** When running tests under coverage (`--cov`), `conftest.py` automatically sets `NUMBA_DISABLE_JIT=1`
to allow line-level coverage. Don't set this manually when you want to test actual Numba performance.

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
  config.py          — global config (Overpass URL, cache, rate limits, logging)
  graph/
    urban_graph.py   — UrbanGraph class (data + adjacency matrix + method dispatch)
    adjacency.py     — build_adjacency_matrix() from edges_gdf → CSR
    components.py    — connected/weakly/strongly connected component algorithms
    editors.py       — subgraph_by_nodes, clip, join, relabel, project_objects2urban_graph
    transformers.py  — keep_largest_component, simplify_multiedges, to_directed/undirected
    shortest_paths.py — Dijkstra wrappers (single/multi-source, parallel, OD matrix)
    adapters.py      — UrbanGraph ↔ NetworkX conversion
    nx_utils.py      — legacy NetworkX helpers (optional dep, soft-imported)
    validation.py    — node/edge schema validation, CRS sync
    graph_inputs.py  — resolve_graph_nodes_input() helper
  _numba/
    csr.py           — CSR builder (Numba-compiled)
    components.py    — BFS/DFS traversal (Numba-compiled)
    shortest_paths.py — Dijkstra kernels (Numba-compiled)
  graph_builders/
    drive_walk_builders.py      — get_drive_graph(), get_walk_graph()
    public_transport_builders.py — get_public_transport_graph()
    intermodal_builders.py      — get_intermodal_graph(), join_pt_walk_graph()
  overpass/
    downloaders.py   — Overpass HTTP requests with retry/rate-limit
    parsers.py       — JSON → GeoDataFrame (nodes/edges)
    cache.py         — file-based JSON cache for Overpass responses
  constants/
    transport_specs.py — TransportSpec, TransportRegistry, DEFAULT_REGISTRY
    highway_enums.py   — HighwayType enum
    network_enums.py   — network type enums
```

### Data flow

```
Overpass API
  → downloaders.py (HTTP + cache)
  → parsers.py (JSON → GeoDataFrame)
  → graph_builders/ (GeoDataFrame → UrbanGraph)
  → intermodal_builders.py (UrbanGraph + UrbanGraph → joined UrbanGraph)
  → graph/editors.py (project objects onto graph)
  → graph/shortest_paths.py → _numba/ (CSR Dijkstra → OD matrix)
```

### Public transport speed model

`TransportSpec` (frozen dataclass) encodes per-mode physics: `vmax_tech_kmh`, `accel_dist_m`, `brake_dist_m`,
`traffic_coef`. `DEFAULT_REGISTRY` covers bus/tram/trolleybus/subway; `DEFAULT_REGISTRY_W_TRAIN` adds train. Pass a
custom `TransportRegistry` to `get_public_transport_graph()` to override speeds.

### Test organization

Tests are split into `_unit` (no network, fast) and `_network` (calls Overpass). Network test modules are listed in
`conftest.py::NETWORK_ONLY_TEST_MODULES` and are excluded from collection unless `--run-network` is passed.

Pytest markers: `unit`, `integration`, `network`, `slow`, `numba`.

Shared fixtures in `tests/conftest.py`: `territory_osm_id` (OSM relation 1114252), `bounds`, `intermodal_graph` (
session-scoped, downloaded once per network run).

Object factories for unit tests: `tests/factories.py`.

### Coding conventions

- Line length: 120 characters (black + pylint)
- Imports sorted with isort (multi-line style 3, trailing comma)
- `isort` skips `__init__.py` files
- All public API must be reachable via `from iduedu import ...` through `_api.py`
- NetworkX is an **optional** dependency — always soft-import it inside `try/except ModuleNotFoundError` and check
  `exc.name == "networkx"` before re-raising
- Numba JIT functions live in `iduedu/_numba/` and are called only through the `graph/` layer, never directly from
  builders or public API
