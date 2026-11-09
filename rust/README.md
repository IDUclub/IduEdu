# iduedu-rs

Rust core of [IduEdu](https://github.com/IDUclub/IduEdu), compiled with [maturin](https://www.maturin.rs/) into the
extension module `iduedu._rust`. It is not a separate package: `pip install iduedu` ships it inside the wheels, and
building IduEdu from source builds it.

## Contents

- `src/csr.rs`: borrowed CSR view with validation, the edge-list → CSR builder (parallel arcs merged with min or max)
  and the transpose.
- `src/dijkstra.rs`: Dijkstra searches (one or many sources, cutoff, early exit on targets, predecessors, nearest
  source) with packed heap keys and reusable per-search buffers; route reconstruction.
- `src/components.rs`: connected and strongly connected components.
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
