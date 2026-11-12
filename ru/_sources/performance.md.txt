# Performance

This page is being rewritten for IduEdu 3.0 and will hold new measurements.

IduEdu 3.0 replaces the Numba kernels of the 2.x series with a compiled Rust extension, ``iduedu._rust``. The
extension is built once into the wheel, so a new process starts without the just-in-time compilation that Numba
needed on its first call, and the ``numba`` and ``llvmlite`` dependencies (about 130 MB) are no longer installed.
The changes cover the whole pipeline:

- **Routing**: Dijkstra searches on a cached compressed sparse row matrix, parallel searches over sources, shared
  searches for objects on the same node, point-to-point searches (bidirectional, A*, ALT; see
  {doc}`routing/point_to_point`).
- **Graph construction**: OpenStreetMap ``.osm.pbf`` files are read and written by the extension, with parallel
  block decoding; street segments are merged into junction-to-junction edges in the extension.
- **Network analysis**: sums of node values within a travel cost, nearest objects, betweenness and the other
  centralities run in the extension in parallel threads.

The measurements of IduEdu 2.x, made with the Numba kernels, are kept in the
[repository history](https://github.com/IDUclub/IduEdu/blob/v2.4.2/docs/benchmarks.md).
