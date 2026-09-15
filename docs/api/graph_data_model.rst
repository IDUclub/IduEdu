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

Node and edge types
-------------------

Builders record what a node or an edge stands for in a ``type`` column.
Routing ignores it, but it tells graph layers apart after they are joined, for
example to colour a route by mode or to select boarding edges.

Walk and drive graphs
~~~~~~~~~~~~~~~~~~~~~

:func:`iduedu.get_walk_graph` returns an undirected multigraph whose edges have
``type="walk"``. :func:`iduedu.get_drive_graph` returns a directed multigraph
with ``type="drive"`` edges, the ``oneway`` direction column and, with
``add_road_category=True``, a road ``category``. Street nodes carry only their
geometry. Edges also keep a configurable set of OSM tags: ``highway`` and
``name`` for walking, plus ``lanes`` for driving (see :doc:`../configuration`).

Public transport from OSM
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 32 68

    * - Type
      - Meaning
    * - ``bus``, ``trolleybus``, ``tram``, ``subway``, ``monorail``, ``taxi``, ``train``
      - On nodes, a stop of one route, with the route name in ``route``. On
        edges, a ride between consecutive stops of that route.
    * - ``platform``
      - Node where passengers wait; it is attached to the walking network in
        intermodal graphs. Subway platforms that are not linked to a station, or
        whose station has no mapped entrance and exit, are typed ``platform`` too.
    * - ``boarding``
      - Edge from a platform to a route stop, with zero length. Its ``time_min``
        is the mode's ``avg_wait_time_min`` from the transport registry.
    * - ``alighting``
      - Edge from a route stop back to its platform, with zero time and length.
    * - ``subway_station``
      - Node for a station, with OSM ``name`` and ``depth`` when mapped. As an
        edge type, the bidirectional link between a station and its platform.
    * - ``subway_platform``
      - Node for a platform reached only through its station.
    * - ``subway_entry_exit``, ``subway_entry``, ``subway_exit``
      - Nodes for station entrances, by the direction of travel they allow.
        They are attached to the walking network.
    * - ``subway_entrance``, ``subway_exit``
      - Edges from an entrance into the station and from the station to an
        exit, timed as an escalator ride when the station depth is known.
    * - ``subway_transfer``
      - Bidirectional edge between the stations of one interchange.

Public transport from GTFS
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 32 68

    * - Type
      - Meaning
    * - ``bus``, ``subway``, ``tram`` and the other modes
      - On nodes, a stop of one route pattern, with ``gtfs_stop_id``,
        ``route_id``, ``pattern_id`` and ``stop_position``. On edges, a ride
        between consecutive stops of that pattern.
    * - ``platform``
      - Node for a stop served by the selected trips; it is attached to the
        walking network in intermodal graphs.
    * - ``station_platform``
      - Node for a stop inside a station that the feed describes with pathways.
        It is reached through the station, not directly from the street.
    * - ``station``, ``station_entry_exit``, ``station_node``, ``station_boarding_area``
      - Nodes for stops with ``location_type`` 1, 2, 3 and 4 that pathways or
        served stops refer to. Entrances (``station_entry_exit``) are attached
        to the walking network.
    * - ``boarding``, ``alighting``
      - As in OSM graphs, except that the boarding time is the timetable wait.
    * - ``pathway``
      - Edge from ``pathways.txt`` with ``pathway_id`` and ``pathway_mode``,
        timed by ``traversal_time`` or by its length at ``walk_speed_m_per_min``.
    * - ``station_link``
      - Zero-cost, bidirectional edge from a stop to its ``parent_station``.

Intermodal graphs
~~~~~~~~~~~~~~~~~

:func:`iduedu.get_intermodal_graph` and :func:`iduedu.join_pt_walk_graph` keep
the types of both layers. Walking edges, including the connectors that attach
platforms and entrances to the street, have ``type="walk"``, and walking nodes
have no type. See :doc:`../examples/gtfs_public_transport` and
:doc:`../examples/shortest_paths` for routes coloured by edge type.

API reference
-------------

.. autoclass:: iduedu.UrbanGraph
    :members:
    :member-order: bysource
