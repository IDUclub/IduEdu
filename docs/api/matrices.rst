Matrices
--------

.. currentmodule:: iduedu

Shortest-path workflows
=======================

IduEdu shortest-path helpers work on :class:`UrbanGraph` objects and reuse the
graph adjacency cache when possible. Distances use the units of the selected
edge weight:

``time_min``
    Travel time in minutes.

``length_meter``
    Network distance in meters.

See :doc:`../examples/shortest_paths` for runnable examples of every function
listed below.

Choosing a helper
~~~~~~~~~~~~~~~~~

Use :func:`single_source_dijkstra` for one source, :func:`multi_source_dijkstra`
for the nearest member of a source set, or :func:`dijkstra_parallel` for an
independent row per source. Each accepts ``output="distances"``, ``"paths"``
or ``"both"``.

Use :func:`od_matrix` for specified origin and destination sets, with
``mode="all_to_all"`` or ``"pairwise"``. Use :func:`iter_od_matrix` to consume
origin-row blocks. Existing specialised ``*_path`` and ``*_path_length``
helpers remain available for compatibility.

Use :func:`path_to_edges` to turn a node route into an ordered GeoDataFrame of
the actual graph edges. This preserves curved edge geometry and transport
attributes for mapping.

Distances and paths in one search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`single_source_dijkstra`, :func:`multi_source_dijkstra`,
:func:`dijkstra_parallel` and :func:`od_matrix` accept ``output``:

* ``"distances"`` returns sparse numeric results.
* ``"paths"`` returns sparse results containing ordered node-id lists.
* ``"both"`` returns :class:`ShortestPathResult` with matching ``distances``
  and ``paths`` tables, computed by the same searches.

``od_matrix`` defaults to ``output="distances"`` for compatibility. The three
general Dijkstra helpers default to ``output="both"``. Existing specialised
``*_path`` and ``*_path_length`` functions keep their return types and use the
same search implementation.

.. code-block:: python

    result = graph.od_matrix(
        origins_nodes=origins,
        destination_nodes=destinations,
        weight="time_min",
        cutoff=30,
        output="both",
    )
    distances = result.distances
    paths = result.paths
    route = result.get_path(origins[0], destinations[0])

Both matrices preserve input order and labels. Missing routes and routes
outside the cutoff have distance ``np.inf`` and path ``None`` (pandas can
display a sparse missing path as ``NaN``). Self routes have cost zero and path
``[node]``; zero-cost edges are retained. The cutoff is inclusive and uses the
selected weight units. ``threshold`` remains an alias on OD functions and
``multi_source_dijkstra_path``; specifying both names raises ``ValueError``.
``dtype`` controls returned distance storage; kernel arithmetic remains
``float32`` even when the requested output dtype is ``float64``.

Single, nearest-source and independent searches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # One source, with only the requested targets in the result.
    single = graph.single_source_dijkstra(
        source, target_nodes=destinations, cutoff=30,
    )
    route = single.get_path(destinations[0])

    # One combined source set: choose the closest source for each target.
    nearest = graph.multi_source_dijkstra(source_nodes=origins, cutoff=30)
    winning_sources = nearest.nearest_sources

    # Independent result row for every source object or node.
    independent = graph.dijkstra_parallel(
        source_nodes=origins, target_nodes=destinations,
        cutoff=30, output="both", max_workers=4,
    )

For single-source and nearest-source searches, ``target_nodes=None`` returns
only reachable nodes, indexed in graph node order. Explicit targets preserve
their input order and include unreachable entries. Independent searches
without explicit targets use the union of reachable nodes as columns.
Nearest-source ties choose the source appearing first in graph node order;
``nearest_sources`` contains graph node ids, not source object labels.

``reverse=True`` searches the transposed directed graph. Returned paths follow
the original graph direction, towards the source. For undirected graphs this
option has no effect. OD results always follow origin-to-destination direction,
including when the implementation searches from the smaller destination side.

Pairwise OD queries
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    pairs = graph.od_matrix(
        origins_nodes=origins,
        destination_nodes=destinations,
        mode="pairwise",
        output="both",
    )

Pairwise inputs must have equal lengths. Each origin is matched to the
destination at the same position. Results use an ``(origin, destination)``
MultiIndex and one column: ``dist`` for costs or ``path`` for routes. This
avoids constructing all origin-destination combinations.

Lazy route storage
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    result = graph.od_matrix(
        origins_nodes=origins,
        destination_nodes=destinations,
        cutoff=30,
        output="both",
        path_storage="lazy",
    )
    distances = result.distances        # Already calculated.
    route = result.get_path(origins[0], destinations[0])
    assert not result.paths_materialized
    paths = result.paths                # Materialize and cache the whole table.
    assert result.paths_materialized

Lazy storage requires ``output="both"``. It retains compact predecessor trees
for settled nodes, rather than creating every route list immediately.
``get_path`` reconstructs just one route without another Dijkstra search.
The node-id mapping and predecessors are snapshots: later graph edits do not
alter saved paths. Geometries are not saved; applying ``path_to_edges`` to a
modified graph uses that graph's current edges.

Use object labels in ``get_path`` when object tables were supplied. Repeated
node inputs are supported, but ambiguous duplicate labels require positional
access: ``result.get_path_at(row, column)``. Series results use
``get_path(target)`` or ``get_path_at(row)``. Unknown labels raise ``KeyError``;
unreachable known pairs return ``None``.

Lazy trees still consume memory proportional to the number of settled nodes
per unique search. They can be larger than a few eagerly materialized routes,
so lazy storage is most useful when many routes share tree segments and only
some of those routes will be inspected.

Large calculations and working memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``batch_size`` on ``od_matrix`` and ``dijkstra_parallel`` limits the number of
unique-source searches in flight. The full output and all retained lazy trees
still remain in memory. Use :func:`iter_od_matrix` to process origin-row blocks:

.. code-block:: python

    for block in graph.iter_od_matrix(
        origins_nodes=origins,
        destination_nodes=destinations,
        cutoff=30,
        output="both",
        path_storage="lazy",
        batch_size=256,
    ):
        process_distances(block.distances)
        # Inspect selected routes with block.get_path(...), then release block.

In pairwise mode, destination rows are sliced alongside their origins.
Object-to-node matching is performed once. Retaining all yielded blocks also
retains their memory; do not edit the graph while consuming an iterator.

Distance tables are built without a dense OD-sized intermediate. Searches are
shared by objects matched to the same graph node, stop when all requested
targets settle, and use the smaller unique OD side where possible.
``max_workers`` temporarily controls Numba threads; the previous setting is
restored, including when a search fails.

An offline synthetic benchmark is available with
``python -m benchmarks.bench_routing``. It compares separate calls with joint
eager/lazy calculations after JIT warmup. Its allocation measurements use
``tracemalloc`` and exclude Numba's native allocator; they are not process peak
RSS measurements. Timing ratios depend on graph structure and requested paths.

Tables retain node mappings as Series in ``attrs``. Before concatenating blocks
with pandas, clear their block-specific metadata, because pandas compares
``attrs`` using ordinary equality:

.. code-block:: python

    tables = []
    for table in graph.iter_od_matrix(
        origins_nodes=origins, destination_nodes=destinations,
    ):
        table.attrs = {}
        tables.append(table)
    combined = pd.concat(tables)

GeoDataFrame inputs
~~~~~~~~~~~~~~~~~~~

Several helpers accept GeoDataFrames through ``gdf_sources``,
``gdf_origins``, or ``gdf_destinations``. If the table already contains
``graph_node_id`` (or the column configured through ``graph_node_column``), that
mapping is used directly. Otherwise objects are matched to nearest graph nodes.
For object-to-edge projection before OD calculations, see
:doc:`../examples/objects_and_nearest_nodes`.

.. autosummary::
    :toctree: generated
    :nosignatures:

    single_source_dijkstra_path_length
    multi_source_dijkstra_path_length
    multi_source_dijkstra_nearest_source
    dijkstra_path_length_parallel
    single_source_dijkstra
    multi_source_dijkstra
    dijkstra_parallel
    od_matrix
    iter_od_matrix
    ShortestPathResult
    single_source_dijkstra_path
    multi_source_dijkstra_path
    path_to_edges
