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
:func:`clip_urban_graph` for geometry-based clipping,
:func:`relabel_urban_graph` for dense node indexes, and
:func:`join_urban_graphs` for concatenating compatible graph tables.

Use :meth:`UrbanGraph.to_directed`, :meth:`UrbanGraph.to_undirected`, and
:meth:`UrbanGraph.simplify_multiedges` for topology transformations.

See :doc:`../examples/graph_operations` for editing examples.

Object matching and projection
==============================

Use :func:`nearest_nodes` when an object table only needs a nearest existing
graph node id.

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
    :toctree: generated
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
    relabel_urban_graph
    subgraph_by_nodes
    clip_urban_graph
    join_urban_graphs
    nearest_nodes
    project_objects2urban_graph
    apply_urban_graph_changes
    read_urban_graph
    write_urban_graph
    validate_graph

Optional NetworkX utilities
---------------------------

The following helpers are available when NetworkX support is installed.

.. autosummary::
    :toctree: generated
    :nosignatures:

    graph2gdf
    gdf2graph
    keep_largest_nx_component
    clip_nx_graph
    read_gml
    write_gml
    reproject_graph
