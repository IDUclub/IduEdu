Network analysis
----------------

.. currentmodule:: iduedu

The functions on this page measure a graph that is already built: its shape and
connectivity, what can be reached from a place, which streets carry the
shortest paths, and how steep the network is. They read an ``UrbanGraph`` in a
projected CRS and return pandas tables indexed by node, edge, zone or origin.

Areas (``area``, ``zones``) follow the rule of ``territory`` in the graph
builders: a bare shapely geometry is in EPSG:4326 (longitude, latitude), and a
GeoSeries or GeoDataFrame carries its own CRS; both are reprojected to the
graph CRS.

Thresholds are given in the units of ``weight``. Statistics and centralities
measure by ``length_meter`` (metres) by default, as is usual for street
networks; the functions that work with objects measure by ``time_min``
(minutes) like the routing functions.

Statistics
==========

:func:`graph_stats` describes a street network as a whole, inside a study
``area`` or for every one of ``zones``. Streets are counted once whatever their
direction and however many copies of them the graph stores, and lengths are
taken on the unique segments of the geometry (:func:`unique_segments`). The
number of streets at a node always comes from the whole graph: build the graph
with a margin around the study area, pass the area, and dead ends and
intersections at its border are counted correctly.

:func:`edge_stats` gives counts, lengths and speeds for every value of an edge
column such as ``type``, ``category`` or ``highway``; :func:`transport_stats`
gives stops, routes and stop spacing for every public-transport mode.
:func:`local_stats` measures the surroundings of every node within a network
radius, :func:`streets_per_node` returns the number of streets at each node,
and :func:`route_directness` compares network distances with straight ones.

Junctions drawn as several nodes inflate node and intersection counts; merge
them with :func:`consolidate_intersections` first when that matters.

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    graph_stats
    edge_stats
    transport_stats
    local_stats
    streets_per_node
    route_directness

Objects around origins
======================

These functions answer what is near a place along the network. Origins and
objects are node ids, tables with a column of node ids, or GeoDataFrames that
are matched to their nearest nodes; project objects into the graph with
:func:`project_objects2urban_graph` first when exact distances matter.

:func:`nearest_objects` finds the ``k`` nearest objects of every origin.
:func:`aggregate_objects` counts objects and sums or averages their values
within one or several travel costs, optionally with a distance decay.
:func:`diversity` measures how mixed the categories of the objects in reach
are.

``direction="from"`` measures the way from the origin to the object and
``direction="to"`` the way back; they differ on directed graphs. Sums are
computed from the smaller side: by one search per origin that collects what it
reaches, or by one search per object on the reversed graph that hands its
values to the origins around it.

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    nearest_objects
    aggregate_objects
    diversity

Centrality
==========

:func:`centrality` computes reach, harmonic closeness, gravity and betweenness
of the nodes within one or several thresholds; :func:`edge_betweenness` gives
the betweenness of the edges. Reach, harmonic and gravity take one cut-off
search per node. Betweenness takes a search from every node of the graph and is
much slower without a threshold; ``sample`` estimates it from a random part of
the nodes.

Values near the border of a graph are biased, because the network beyond the
border is missing. Build the graph with a margin at least as wide as the
largest threshold and pass the study ``area``: only the nodes inside are
reported, and the searches still use the whole graph.

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    centrality
    edge_betweenness

Elevation
=========

:func:`add_elevation` reads heights from a raster file, from spot heights or
contour lines in a GeoDataFrame, or from a function, and stores ``elevation``
on the nodes and ``grade``, ``grade_max``, ``ascent`` and ``descent`` on the
edges. Reading rasters needs the optional dependency rasterio
(``pip install 'iduedu[elevation]'``).

Bridges and tunnels do not follow the ground. Edges with a ``bridge`` or
``tunnel`` tag get a straight profile between their ends, and nodes inside
such structures are interpolated between the ends of the structure. The drive,
walk and bike builders keep these tags and never merge a bridge or a tunnel
with the road it continues.

:func:`apply_grade_speed` rescales travel times with a speed model that depends
on the grade: Tobler's hiking function for walking, a linear model for cycling,
or your own function. :func:`graph_stats` reports elevation and grade measures
for graphs that carry them.

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    add_elevation
    apply_grade_speed
