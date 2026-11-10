High-level functions
--------------------

.. currentmodule:: iduedu

Graph builders
==============

The OSM high-level builders download OpenStreetMap data, while the GTFS builder
reads a local Schedule feed. Both normalize their inputs into the ``UrbanGraph``
data model, use a local metric CRS, and compute standard edge weights:

``length_meter``
    Edge geometry length in meters.

``time_min``
    Traversal time in minutes. For walk graphs this is based on ``walk_speed``;
    for drive graphs it is based on road class and speed tags. OSM-based public
    transport uses the active :class:`TransportRegistry`; GTFS graphs derive
    in-vehicle and boarding times from the selected schedule.

Most examples use ``osm_id=1114252`` and can be run directly from the notebooks.
See :doc:`../examples/get_any_graph` for end-to-end examples of all builders.

Drive and walk graphs
~~~~~~~~~~~~~~~~~~~~~

Use :func:`get_drive_graph` for car-accessible street networks and
:func:`get_walk_graph` for pedestrian networks. Both functions support:

``simplify``
    Merge raw OSM way segments into edges that run from junction to junction.
    Segments are merged only when they carry the same tags, so an edge keeps
    the attributes of its ways, and bridges and tunnels stay separate edges.
    Use ``False`` when the full per-segment topology is required.

``clip_by_territory``
    Clip edge geometries to the exact territory boundary before graph assembly.

``keep_largest_subgraph``
    Remove disconnected fragments. Directed graphs use the largest strongly
    connected component by default.

The :doc:`../examples/get_any_graph` notebook shows ``simplify=True`` and
``simplify=False`` side by side and highlights edges affected by clipping.

Edge tags and merging
~~~~~~~~~~~~~~~~~~~~~

The drive, walk and bike builders cut every OSM way into its straight
segments. With ``simplify=True`` the segments are joined back into edges at
the points where exactly two segments meet, follow one another in the same
direction, and carry the same values of every tag the builder reads. So an
edge runs from junction to junction, and it ends earlier wherever a tag
changes: an edge always has the tags of all its ways, never a mix of
different ways.

The builder reads three groups of tags:

- the tags kept as edge columns, set with ``osm_edge_tags`` or, by default,
  with ``config.drive_useful_edges_attr``, ``config.walk_useful_edges_attr``
  and ``config.bike_useful_edges_attr``;
- ``bridge`` and ``tunnel``, which are always read and always kept, whatever
  ``osm_edge_tags`` holds, because a bridge or a tunnel does not follow the
  ground and must not be merged with the road it continues
  (:func:`add_elevation` relies on them);
- tags the builder needs for its own calculations; they split edges too but
  are not kept as columns.

.. list-table::
    :header-rows: 1

    * - Graph
      - Kept tags (default, changed with ``config`` or ``osm_edge_tags``)
      - Always kept
      - Read but not kept
    * - drive
      - ``highway``, ``name``, ``lanes``
      - ``bridge``, ``tunnel``, ``oneway``, ``category``
      - ``maxspeed`` (for the travel time)
    * - walk
      - ``highway``, ``name``
      - ``bridge``, ``tunnel``
      - none
    * - bike
      - ``highway``, ``name``, ``lanes``, ``maxspeed``, ``surface``
      - ``bridge``, ``tunnel``, ``oneway``, ``cycle_infra``, ``lts``
      - ``cycleway``, ``cycleway:left``, ``cycleway:right``,
        ``cycleway:both``, ``bicycle``, ``oneway:bicycle``

The drive builder also always reads ``highway`` and ``oneway``, and the bike
builder ``highway``, ``maxspeed``, ``lanes`` and ``oneway``, even when
``osm_edge_tags`` leaves them out. Fewer kept tags therefore give longer
edges: with ``osm_edge_tags=[]`` a walk graph is split only at junctions,
bridges and tunnels, while a drive graph is still split where the speed limit
or the road class changes.

Bike graphs
~~~~~~~~~~~

Use :func:`get_bike_graph` for the network a bicycle may use: streets,
cycleways, paths and tracks, and footways where cycling is permitted. Every
edge gets ``cycle_infra`` (``track``, ``lane`` or ``mixed``) and ``lts``, the
level of traffic stress from 1 to 4 derived from the infrastructure, the speed
limit and the number of lanes. ``max_lts`` keeps only the low-stress network,
and ``elevation`` adds grades and makes travel times depend on them (see
:doc:`../analysis/elevation`).

OSM public-transport and intermodal graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`get_public_transport_graph` to build bus, trolleybus, tram, subway,
monorail, or share-taxi (``taxi``) graphs. Pass ``transport_types`` to restrict
modes and ``transport_registry`` to customize time calculation. A requested type
must exist in the registry, otherwise the builder raises ``ValueError``: trains
need a registry that defines them, such as ``DEFAULT_REGISTRY_W_TRAIN``. OSM
``light_rail`` routes are fetched as ``tram`` and ``share_taxi`` routes as
``taxi``. See :doc:`public_transport`.

Use :func:`join_pt_walk_graph` when you already have public-transport and walk
graphs. Use :func:`get_intermodal_graph` to build both networks and join them in
one call. Public-transport platform-like nodes are projected to nearby walk
edges, optionally creating link edges.

See :doc:`../examples/get_any_graph` for intermodal construction and
:doc:`../examples/objects_and_nearest_nodes` for the lower-level projection
workflow used during joining.

GTFS public-transport graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`get_gtfs_public_transport_graph` to build a static graph from a
local GTFS Schedule directory or ZIP archive. A list of feeds is merged with
:func:`merge_gtfs_feeds` first. Service-date and time-window filters select the
departures used to estimate boarding waits. GTFS graphs do not use
:class:`TransportRegistry` waiting-time defaults.

To make a GTFS graph intermodal, first build a walk graph, pass its CRS to the
GTFS builder, and combine the two with :func:`join_pt_walk_graph`. See
:doc:`gtfs` for the complete workflow and modeling details.

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    get_drive_graph
    get_walk_graph
    get_bike_graph
    get_public_transport_graph
    get_gtfs_public_transport_graph
    get_intermodal_graph
    join_pt_walk_graph
