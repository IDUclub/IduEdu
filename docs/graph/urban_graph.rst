The UrbanGraph model
====================

.. currentmodule:: iduedu

Overview
--------

Every network in IduEdu is an :class:`UrbanGraph`: a table of nodes and a table of edges, both GeoDataFrames in one
projected CRS, plus a few flags that say how to read them. Tables are what urban data already is: they filter, join,
plot and save with pandas and GeoPandas. The graph algorithms read a compressed sparse row (CSR) matrix built from the edge table and run in
the Rust core.

.. code-block:: python

   walk = iduedu.get_walk_graph(osm_id=1114252)
   walk
   # UrbanGraph(nodes=22289, edges=29149, is_multigraph=True, is_directed=False,
   #            edge_direction_column=None, crs='EPSG:32636', type='walk')

   walk.nodes_gdf.head(2)
   #                          geometry
   # 0  POINT (346518.381 6645885.289)
   # 1   POINT (346402.237 6645930.35)

   walk.edges_gdf[["u", "v", "k", "length_meter", "time_min", "highway"]].head(2)
   #    u  v  k  length_meter  time_min  highway
   # 0  0  2  0         1.696     0.020  footway
   # 1  1  2  0       134.782     1.617  footway

Tables
------

Nodes
~~~~~

``nodes_gdf`` is a DataFrame or GeoDataFrame whose index is the node id used by every function. The index must be
unique. Spatial graphs store a point geometry for every node; builders add columns such as ``type``, ``route`` or
``name`` where a node stands for something.

Edges
~~~~~

``edges_gdf`` has one row per edge and these required columns:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Meaning
   * - ``u``, ``v``
     - Source and target node ids, present in ``nodes_gdf.index``
   * - ``k``
     - Key of parallel edges, required when ``is_multigraph`` is true; ``(u, v, k)`` identifies an edge
   * - ``geometry``
     - Line geometry of the edge
   * - ``length_meter``
     - Length in metres
   * - ``time_min``
     - Travel time in minutes

Without ``is_multigraph`` every ``(u, v)`` pair must be unique. Any other column (OSM tags, ``type``, ``route``,
``grade``) can be added and used as a weight if it holds non-negative numbers.

Direction
~~~~~~~~~

``is_directed`` makes adjacency-based algorithms respect the order of ``u`` and ``v``. A directed graph can carry
an ``edge_direction_column``, usually ``oneway``: ``True`` allows movement only from ``u`` to ``v`` and ``False``
in both directions, so a two-way street is stored once. Walk graphs are undirected; drive, bike, public-transport
and intermodal graphs are directed.

CRS
~~~

When nodes and edges are GeoDataFrames, their CRS must equal the graph CRS. Builders choose a local projected CRS
(UTM) from the territory, so that lengths are metres. Functions that take areas or objects in another CRS reproject
them to the graph CRS.

Adjacency matrix
----------------

Routing and analysis functions turn the edge table into a CSR matrix of one weight, ``time_min`` by default:

1. The rows and columns follow ``nodes_gdf.index`` (``graph.adjacency_nodelist``).
2. Parallel edges between two nodes, in either orientation of an undirected graph, become one entry with the
   smallest cost (``multigraph_rule="max"`` takes the largest).
3. Every edge of an undirected graph and every two-way edge of a directed one gives entries in both directions.

The matrix is stored on the graph (``graph.adjacency_matrix``) with a copy for the Rust core and is reused by later
calls with the same weight. :meth:`UrbanGraph.update_adjacency_matrix` rebuilds it, and
:meth:`UrbanGraph.to_csr` builds a matrix without storing it. After changing the values of an edge column that a
graph has already been routed on, call ``graph.update_adjacency_matrix(weight=...)`` before the next query.
Functions that return a new graph (the editors, :func:`add_elevation`, :func:`apply_grade_speed`) need no such call.

Node and edge types
-------------------

Builders record what a node or an edge stands for in a ``type`` column. Routing ignores it, but it tells the layers
apart after they are joined, for example to colour a route by mode or to select boarding edges.

Streets
~~~~~~~

:func:`get_walk_graph` returns an undirected multigraph with ``type="walk"`` edges. :func:`get_drive_graph` returns
a directed multigraph with ``type="drive"`` edges, the ``oneway`` direction column and, with
``add_road_category=True``, a road ``category``. :func:`get_bike_graph` adds ``cycle_infra`` and ``lts``. Street
nodes carry only their geometry (see :doc:`../building/streets`).

Public transport from OSM
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 32 68

    * - Type
      - Meaning
    * - ``bus``, ``trolleybus``, ``tram``, ``subway``, ``monorail``, ``taxi``, ``train``
      - On nodes, a stop of one route, with the route name in ``route``. On edges, a ride between consecutive stops
        of that route.
    * - ``platform``
      - Node where passengers wait; it is attached to the walking network in intermodal graphs. Subway platforms
        that are not linked to a station, or whose station has no mapped entrance and exit, are typed ``platform``
        too.
    * - ``boarding``
      - Edge from a platform to a route stop, with zero length. Its ``time_min`` is the ``avg_wait_time_min`` of the
        mode in the transport registry.
    * - ``alighting``
      - Edge from a route stop back to its platform, with zero time and length.
    * - ``subway_station``
      - Node for a station, with OSM ``name`` and ``depth`` when mapped. As an edge type, the link in both
        directions between a station and its platform.
    * - ``subway_platform``
      - Node for a platform reached only through its station.
    * - ``subway_entry_exit``, ``subway_entry``, ``subway_exit``
      - Nodes for station entrances, by the direction of travel they allow. They are attached to the walking
        network.
    * - ``subway_entrance``, ``subway_exit``
      - Edges from an entrance into the station and from the station to an exit, timed as an escalator ride when
        the station depth is known.
    * - ``subway_transfer``
      - Edge in both directions between the stations of one interchange.

Public transport from GTFS
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 32 68

    * - Type
      - Meaning
    * - ``bus``, ``subway``, ``tram`` and the other modes
      - On nodes, a stop of one route pattern, with ``gtfs_stop_id``, ``route_id``, ``pattern_id`` and
        ``stop_position``. On edges, a ride between consecutive stops of that pattern.
    * - ``platform``
      - Node for a stop served by the selected trips; it is attached to the walking network in intermodal graphs.
    * - ``station_platform``
      - Node for a stop inside a station that the feed describes with pathways. It is reached through the
        station, not directly from the street.
    * - ``station``, ``station_entry_exit``, ``station_node``, ``station_boarding_area``
      - Nodes for stops with ``location_type`` 1, 2, 3 and 4 that pathways or served stops refer to. Entrances
        (``station_entry_exit``) are attached to the walking network.
    * - ``boarding``, ``alighting``
      - As in OSM graphs, except that the boarding time is the timetable wait.
    * - ``pathway``
      - Edge from ``pathways.txt`` with ``pathway_id`` and ``pathway_mode``, timed by ``traversal_time`` or by its
        length at ``walk_speed_m_per_min``.
    * - ``station_link``
      - Edge of zero cost in both directions from a stop to its ``parent_station``.

Intermodal graphs
~~~~~~~~~~~~~~~~~

:func:`get_intermodal_graph` and :func:`join_pt_walk_graph` keep the types of both layers. Walking edges, including
the connectors that attach platforms and entrances to the street, have ``type="walk"``, and walking nodes have no
type (see :doc:`../building/intermodal`).

Methods
-------

Most functions of the library are also methods of the graph, with the graph as the first argument left out:

.. code-block:: python

   walk.od_matrix(gdf_origins=homes, gdf_destinations=schools)        # iduedu.od_matrix(walk, ...)
   walk.shortest_path(source, target, method="alt")
   walk.graph_stats(zones=grid)
   walk = walk.add_elevation("dem.tif").apply_grade_speed("tobler")

Editing methods (``clip``, ``relabel``, ``join``, ``to_directed``, ``to_undirected``, ``simplify_multiedges``,
``project_objects``) return a new graph, or change the graph itself with ``inplace=True``.

Validation
----------

The constructor checks the tables: unique node ids, edge ends present among the nodes, the required columns, line
geometries and one CRS. :meth:`UrbanGraph.validate` and :func:`validate_graph` repeat the checks after your own
edits:

.. code-block:: python

   graph.edges_gdf["time_min"] = graph.edges_gdf["length_meter"] / 80
   graph.validate()
   graph.update_adjacency_matrix(weight="time_min")

Storage
-------

A ``.urbangraph`` archive stores a graph without converting it: ``metadata.json``, ``nodes.parquet`` and
``edges.parquet``, and optionally the adjacency matrix as a cache.

.. code-block:: python

   graph.write("walk.urbangraph")                    # or iduedu.write_urban_graph(graph, "walk.urbangraph")
   graph = iduedu.UrbanGraph.read("walk.urbangraph")  # or iduedu.read_urban_graph("walk.urbangraph")

Install ``iduedu[io]`` when parquet support is not already available.

NetworkX
--------

NetworkX is an optional dependency. :func:`urban_graph2nx_graph` (``graph.to_nx_graph()``) and
:func:`nx_graph2urban_graph` (``UrbanGraph.from_nx_graph``) convert at the boundary with code that needs NetworkX;
the other NetworkX helpers are listed in :doc:`operations`. Code written for IduEdu 1.x maps as follows:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Before (NetworkX graph)
     - Now (UrbanGraph)
   * - Read nodes and edges from a NetworkX graph
     - Use ``graph.nodes_gdf`` and ``graph.edges_gdf``
   * - Convert GeoDataFrames to NetworkX before an OD matrix
     - Pass the ``UrbanGraph`` to :func:`od_matrix`
   * - Snap objects to node ids before every matrix call
     - Pass the objects as ``gdf_origins`` / ``gdf_destinations``, or store ``graph_node_id`` from
       :func:`nearest_nodes` once
   * - Keep NetworkX as the graph in memory
     - Keep ``UrbanGraph``; convert to NetworkX only where another library needs it

Functions
---------

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    UrbanGraph
    read_urban_graph
    write_urban_graph
    validate_graph

Limitations
-----------

- The adjacency matrix is not rebuilt when edge values change inside a graph that has been routed on; call
  :meth:`UrbanGraph.update_adjacency_matrix` after such edits.
- Weights must be non-negative and without missing values.
- Node ids can be any hashable values, but integer ids keep the tables and the archives compact.

Example notebook
----------------

:doc:`../examples/urban_graph_basics`
