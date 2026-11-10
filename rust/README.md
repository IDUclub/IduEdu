# iduedu-rs

Rust core of [IduEdu](https://github.com/IDUclub/IduEdu), compiled with [maturin](https://www.maturin.rs/) into the
extension module `iduedu._rust`. It is not a separate package: `pip install iduedu` ships it inside the wheels, and
building IduEdu from source builds it.

## Contents

- `src/csr.rs`: borrowed CSR view with validation, the edge-list → CSR builder (parallel arcs merged with min or max)
  and the transpose.
- `src/dijkstra.rs`: Dijkstra searches (one or many sources, cutoff, early exit on targets, predecessors, nearest
  source) with packed heap keys and reusable per-search buffers; route reconstruction.
- `src/bidirectional.rs`: bidirectional Dijkstra for point-to-point queries (searches from both ends, stops where
  they meet), optionally avoiding nodes and arcs out of the source.
- `src/astar.rs`: A* with the straight distance between node coordinates as the estimate, scaled by the smallest
  cost per metre of the graph's own arcs; arcs of zero cost add a slack that keeps the answers exact.
- `src/yen.rs`: Yen's k shortest loopless routes (Lawler's rule, spur searches in parallel).
- `src/components.rs`: connected and strongly connected components.
- `src/analysis.rs`: cut-off searches that sum node values around every source (with distance decay), spread values
  from their own nodes, find the nearest objects, and Brandes betweenness for nodes and arcs in several cost bands.
- `src/chains.rs`: merging street segments into junction-to-junction edges; segments are joined only when they
  follow one another and carry equal tags, so bridges and tunnels stay separate.
- `src/pbf.rs`: reading OSM `.pbf` files (`osmpbf` crate): ways by Overpass tag filter with their node locations,
  nodes and relations by id, tag or member, the header. Blocks are decoded in parallel; the first pass records a
  block index (offsets and id ranges) so later queries read only the blocks they need. Backs `PbfSource`.
- `src/pbf_write.rs`: writing `.pbf` files (zlib blocks, dense nodes, no metadata): per-territory extracts of a
  larger file in one pass (`split_by_polygons`) and small files from element lists.
- `src/lib.rs`: PyO3 bindings (`CsrGraph`, `build_csr`, `reconstruct_paths`, `connected_components`,
  `strongly_connected_components`, `PbfFile`, `write_pbf`), abi3 for CPython 3.11+. Every kernel releases the GIL;
  parallel kernels use rayon.

The Python layer calls these only through `iduedu/graph/` (`adjacency.py`, `routing.py`, `components.py`) and
`iduedu/osm_pbf/` (`reader.py`, `extract.py`).

## Development

```bash
uv sync --all-groups                       # builds the extension into the editable install
(cd rust && cargo fmt --check && cargo clippy --release --all-targets -- -D warnings && cargo test --release)
uv run maturin build --release             # wheel into rust/target/wheels
```

The crate version is kept in sync with `pyproject.toml` by python-semantic-release; never bump it by hand.
