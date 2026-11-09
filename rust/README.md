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
- `src/pbf.rs`: reading ways from OSM `.pbf` files (`osmpbf` crate) with an Overpass tag filter, two passes with
  blocks decoded in parallel; backs `PbfSource` for `.pbf` files (other formats go through pyosmium).
- `src/lib.rs`: PyO3 bindings (`CsrGraph`, `build_csr`, `reconstruct_paths`, `connected_components`,
  `strongly_connected_components`, `read_pbf_ways`), abi3 for CPython 3.11+. Every kernel releases the GIL;
  parallel kernels use rayon.

The Python layer calls these only through `iduedu/graph/` (`adjacency.py`, `routing.py`, `components.py`) and
`iduedu/osm_pbf/reader.py`.

## Development

```bash
uv sync --all-groups                       # builds the extension into the editable install
(cd rust && cargo fmt --check && cargo clippy --release --all-targets -- -D warnings && cargo test --release)
uv run maturin build --release             # wheel into rust/target/wheels
```

The crate version is kept in sync with `pyproject.toml` by python-semantic-release; never bump it by hand.
