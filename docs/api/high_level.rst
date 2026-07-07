High-level functions
--------------------

.. currentmodule:: iduedu

Graph builders
==============

The high-level builders download OpenStreetMap data, normalize it into the
``UrbanGraph`` data model, estimate a local metric CRS, and compute standard edge
weights:

``length_meter``
    Edge geometry length in meters.

``time_min``
    Traversal time in minutes. For walk graphs this is based on ``walk_speed``;
    for drive graphs it is based on road class and speed tags; for public
    transport it is based on the active :class:`TransportRegistry`.

Most examples use ``osm_id=1114252`` and can be run directly from the notebooks.
See :doc:`../examples/get_any_graph` for end-to-end examples of all builders.

Drive and walk graphs
~~~~~~~~~~~~~~~~~~~~~

Use :func:`get_drive_graph` for car-accessible street networks and
:func:`get_walk_graph` for pedestrian networks. Both functions support:

``simplify``
    Merge raw OSM way segments into longer graph edges. Use ``False`` when the
    full per-segment topology is required.

``clip_by_territory``
    Clip edge geometries to the exact territory boundary before graph assembly.

``keep_largest_subgraph``
    Remove disconnected fragments. Directed graphs use the largest strongly
    connected component by default.

The :doc:`../examples/get_any_graph` notebook shows ``simplify=True`` and
``simplify=False`` side by side and highlights edges affected by clipping.

Public-transport and intermodal graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`get_public_transport_graph` to build bus, tram, trolleybus, subway,
or train graphs. Pass ``transport_types`` to restrict modes and
``transport_registry`` to customize time calculation.

Use :func:`join_pt_walk_graph` when you already have public-transport and walk
graphs. Use :func:`get_intermodal_graph` to build both networks and join them in
one call. Public-transport platform-like nodes are projected to nearby walk
edges, optionally creating link edges.

See :doc:`../examples/get_any_graph` for intermodal construction and
:doc:`../examples/objects_and_nearest_nodes` for the lower-level projection
workflow used during joining.

.. autosummary::
    :toctree: generated
    :nosignatures:

    get_drive_graph
    get_walk_graph
    get_public_transport_graph
    get_intermodal_graph
    join_pt_walk_graph
