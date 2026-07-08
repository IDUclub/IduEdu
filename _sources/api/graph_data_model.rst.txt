Graph data model
================

IduEdu represents transport networks with :class:`iduedu.UrbanGraph`.
An ``UrbanGraph`` stores graph topology and geometry in two pandas-compatible
tables: ``nodes_gdf`` and ``edges_gdf``.

See :doc:`../examples/urban_graph_basics` for a runnable introduction to graph
tables, validation, adjacency matrices, empty graphs, and ``.urbangraph`` IO.

Nodes table
-----------

``nodes_gdf`` is a ``pandas.DataFrame`` or ``geopandas.GeoDataFrame`` whose
index is the node identifier used by all graph algorithms.

For spatial graphs, ``nodes_gdf`` should be a ``GeoDataFrame`` with point
geometries. The node index must be unique.

Edges table
-----------

``edges_gdf`` is a ``pandas.DataFrame`` or ``geopandas.GeoDataFrame`` with one
row per graph edge. Spatial graphs use ``LineString`` geometries.

Required columns:

``u``
    Source node id. Must reference ``nodes_gdf.index``.

``v``
    Target node id. Must reference ``nodes_gdf.index``.

``geometry``
    Edge geometry. For geospatial graphs this is a ``LineString``.

``length_meter``
    Edge length in meters.

``time_min``
    Edge traversal time in minutes.

Multigraphs
-----------

If ``UrbanGraph.is_multigraph`` is true, ``edges_gdf`` must also contain ``k``.
The tuple ``(u, v, k)`` uniquely identifies an edge. Non-multigraphs require
``(u, v)`` pairs to be unique.

Directed edges
--------------

Directed graphs are represented by ``UrbanGraph.is_directed``. Some builders
also provide an edge direction column, usually ``oneway``.

When ``edge_direction_column`` is set:

- ``True`` means movement is allowed only from ``u`` to ``v``;
- ``False`` means movement is allowed in both directions.

Coordinate reference systems
----------------------------

When nodes and edges are ``GeoDataFrame`` objects, their CRS must match the
graph CRS. Builders usually estimate a local projected CRS for metric lengths
and travel-time calculations.

API reference
-------------

.. autoclass:: iduedu.UrbanGraph
    :members:
    :member-order: bysource
