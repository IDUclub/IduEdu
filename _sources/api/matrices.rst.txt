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

Use :func:`single_source_dijkstra_path_length` when one source node should be
expanded to reachable graph nodes.

Use :func:`multi_source_dijkstra_path_length` when several sources should act
as one combined source set and each reachable node needs only the best distance
to the closest source.

Use :func:`multi_source_dijkstra_nearest_source` when the closest source id is
needed together with the distance.

Use :func:`dijkstra_path_length_parallel` when every source needs its own sparse
distance row, for example for per-origin accessibility profiles.

Use :func:`od_matrix` when you need an origin-destination table between two
sets of graph nodes or object tables.

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
    od_matrix
