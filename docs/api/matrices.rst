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
origin-row blocks. :func:`shortest_path` returns the cost and the route between
two nodes, and :func:`alternative_paths` and :func:`k_shortest_paths` several
alternative routes (see `Point-to-point routes`_). The legacy ``*_path_length`` helpers and ``multi_source_dijkstra_path``
were removed in 3.0 (see `Removed helpers`_).

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
general Dijkstra helpers default to ``output="both"``.

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
selected weight units. ``threshold`` remains an alias on OD functions;
specifying both names raises ``ValueError``.
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

``method`` chooses the search. ``"dijkstra"`` runs one search per unique origin
that stops once all destinations paired with it are reached. ``"bidirectional"``,
``"astar"`` and ``"alt"`` run one point-to-point search per pair (see
`Point-to-point routes`_). The default ``"auto"`` takes ``"bidirectional"``
when most origins occur once and ``"dijkstra"`` when they repeat. All-to-all
tables always use Dijkstra.

Point-to-point routes
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    cost, route = graph.shortest_path(source, target, weight="time_min")
    cost, route = graph.shortest_path(source, target, method="alt")
    alternatives = graph.alternative_paths(source, target, 3, weight="length_meter")

:func:`shortest_path` returns ``(cost, route)`` for one pair, or
``(inf, [])`` when the target is not reached within ``cutoff``. Its methods
return a shortest route and differ only in speed:

``"bidirectional"`` (default)
    Searches from the source and, on the reversed graph, from the target at the
    same time and stops where the two searches meet. Each search covers about
    half of the route cost, so on a street network it is two to three times
    faster than one search from the source. It works on any graph and weight.

``"astar"``
    Searches from the source and prefers nodes in the direction of the target.
    The estimate of the remaining cost is the straight distance to the target
    times the smallest cost per unit of straight distance over the edges of the
    graph, so it never exceeds the real cost and the result is exact whatever
    the weight. It pays off when costs follow distance: walking and driving by
    ``length_meter``, or walking by ``time_min``. On an intermodal graph the
    fastest edge (metro) sets the scale and A* gains little. It needs a point
    geometry for every node.

``"alt"``
    A* whose estimate comes from the network itself (ALT: A*, landmarks and the
    triangle inequality). Sixteen landmark nodes are spread along the edges of
    the graph, and the costs from each of them to every node and back are
    computed on the first call and kept with the graph until it changes (0.7 s
    and 33 MB for the walk graph below). The difference of these costs bounds
    the remaining cost from below on any graph, including intermodal ones, and
    no node coordinates are needed. It pays off for many queries on one graph.

``"dijkstra"``
    One search from the source that stops at the target.

.. list-table:: Milliseconds per pair on St Petersburg graphs, one thread
    :header-rows: 1

    * - Graph, weight, pairs
      - Dijkstra
      - bidirectional
      - A*
      - ALT
    * - walk (518 thousand nodes), length, 4 km
      - 2.6
      - 1.1
      - 0.56
      - 0.28
    * - walk, time, 10 km
      - 11.7
      - 4.8
      - 3.7
      - 0.83
    * - walk, time, random pairs
      - 18.0
      - 14.4
      - 10.9
      - 1.9
    * - drive (32 thousand nodes), time, random pairs
      - 0.77
      - 0.51
      - 0.54
      - 0.10
    * - intermodal (533 thousand nodes), time, 10 km
      - 21.3
      - 4.5
      - 23.7
      - 5.1

Random pairs lie across the whole city, where a one-sided search is cut by the
city border and the bidirectional search gains little. Paths of equal cost may
come out differently from different methods; their costs are equal up to
float32 rounding.

Alternative routes
~~~~~~~~~~~~~~~~~~

:func:`alternative_paths` finds routes that differ by whole streets (the
penalty method). After the shortest route, the costs of its edges are
multiplied by ``penalty`` (1.5 by default) in both directions and the search
runs again, so the next route leaves those streets wherever a detour exists
and keeps them where it does not, as on a bridge. A route is kept when its
cost is at most ``max_cost_ratio`` times the shortest one and at most
``max_overlap`` of its cost runs on the streets of a route kept before. The
table is indexed by ``rank`` with ``dist``, ``overlap`` (the largest share of
the cost on the streets of another route of the table) and ``path``.
Streets are graph edges: on a walk graph the sidewalk on the other side of an
avenue is another street.

:func:`k_shortest_paths` returns up to ``k`` loopless routes, cheapest first,
in a table indexed by ``rank`` with ``dist`` and ``path`` columns (Yen's
algorithm). Every next route leaves an earlier one at some node and takes the
shortest way on from there that repeats no node of the shared part and no way
taken before. These are exactly the next shortest routes, and on street graphs
they are small variations of one another: for a 4.9 km walk in St Petersburg
the first three differ by 1.7 m, while :func:`alternative_paths` gives 4.9,
5.3 and 5.5 km with 0 to 12 per cent of shared streets, in 2 to 8 ms.

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
``max_workers`` caps the number of worker threads of the searches (by default
one per CPU).

An offline synthetic benchmark is available with
``python -m benchmarks.bench_routing``. It compares separate calls with joint
eager/lazy calculations after a warm-up run. Its allocation measurements use
``tracemalloc`` and exclude native allocations of the compiled kernels; they are not process peak
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

    single_source_dijkstra
    multi_source_dijkstra
    multi_source_dijkstra_nearest_source
    dijkstra_parallel
    od_matrix
    iter_od_matrix
    ShortestPathResult
    shortest_path
    alternative_paths
    k_shortest_paths
    single_source_dijkstra_path
    path_to_edges

Removed helpers
~~~~~~~~~~~~~~~

These wrappers were deprecated in 2.4.1 and removed in IduEdu 3.0 (functions
and ``UrbanGraph`` methods). Each called one general helper with a fixed
``output``:

* ``single_source_dijkstra_path_length`` → ``single_source_dijkstra(..., output="distances")``
* ``multi_source_dijkstra_path_length`` → ``multi_source_dijkstra(..., output="distances")``
* ``dijkstra_path_length_parallel`` → ``dijkstra_parallel(..., output="distances")``
* ``multi_source_dijkstra_path`` → ``od_matrix(..., output="paths")``. Despite
  its name it computed independent origin-destination routes, not a
  nearest-source search.
