Graph utilities
---------------

.. currentmodule:: iduedu

Connectivity
============

Connectivity helpers inspect graph components and are useful before routing or
OD-matrix calculations. For undirected graphs use
:func:`connected_components`; for directed graphs choose between
:func:`weakly_connected_components` and :func:`strongly_connected_components`.

The ``largest_*`` helpers return node-id sets. To actually keep only the largest
component, use :meth:`UrbanGraph.keep_largest_connected_component` or the
``keep_largest_subgraph`` parameter in graph builders.

See :doc:`../examples/connectivity` for a runnable connectivity workflow.

Editing and transformation
==========================

The editing helpers return new ``UrbanGraph`` objects and do not mutate the
source graph unless you use the corresponding ``UrbanGraph`` method with
``inplace=True``.

Use :func:`subgraph_by_nodes` for node-induced subgraphs,
:func:`clip_urban_graph` for geometry-based clipping (a shapely polygon is read
in EPSG:4326, as ``territory`` in the graph builders; pass a GeoSeries or
GeoDataFrame for a polygon in another CRS),
:func:`relabel_urban_graph` for dense node indexes, and
:func:`join_urban_graphs` for concatenating compatible graph tables.

Use :meth:`UrbanGraph.to_directed`, :meth:`UrbanGraph.to_undirected`, and
:meth:`UrbanGraph.simplify_multiedges` for topology transformations.

:func:`consolidate_intersections` merges groups of nearby nodes, such as the
several nodes OSM draws for one junction of dual carriageways, into one node.
Use it before network statistics or centralities, not for routing graphs.
:func:`unique_segments` returns the straight pieces of the edge geometries with
every physical segment once, also where two edges of a two-way street or a
there-and-back pair share their points; its total length is the network length
on the ground.

See :doc:`../examples/graph_operations` for editing examples.

Object matching and projection
==============================

Use :func:`nearest_nodes` when an object table only needs a nearest existing
graph node id, and :func:`nearest_edges` for the nearest edge with the distance
to it and the position along it.

Use :func:`project_objects2urban_graph` when objects should be inserted into
the graph topology. The function prepares an :class:`UrbanGraphChanges` object
with new nodes, new edges, and source edges to delete. Apply the changes with
:func:`apply_urban_graph_changes`, or use :meth:`UrbanGraph.project_objects`
for the in-memory shortcut.

See :doc:`../examples/objects_and_nearest_nodes` for both workflows, including
``add_link_edge=True`` and ``add_link_edge=False``.

Persistence
===========

Use :func:`write_urban_graph` and :func:`read_urban_graph` to store and restore
``.urbangraph`` archives. Archives contain node and edge tables and can
optionally include the cached adjacency matrix. See
:doc:`../examples/urban_graph_basics`.

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    connected_components
    weakly_connected_components
    strongly_connected_components
    number_connected_components
    number_weakly_connected_components
    number_strongly_connected_components
    largest_connected_component
    largest_weakly_connected_component
    largest_strongly_connected_component
    largest_component
    keep_largest_connected_component
    relabel_urban_graph
    subgraph_by_nodes
    clip_urban_graph
    join_urban_graphs
    simplify_multiedges
    to_directed
    to_undirected
    consolidate_intersections
    unique_segments
    nearest_nodes
    nearest_edges
    project_objects2urban_graph
    apply_urban_graph_changes
    UrbanGraphChanges
    read_urban_graph
    write_urban_graph
    validate_graph

Optional NetworkX utilities
---------------------------

NetworkX is not installed with IduEdu. Install it separately (``pip install networkx``) to use the
helpers below; without it they raise ``ImportError`` when called.

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    urban_graph2nx_graph
    nx_graph2urban_graph
    graph2gdf
    gdf2graph
    keep_largest_nx_component
    clip_nx_graph
    read_gml
    write_gml
    reproject_graph
