Function index
==============

.. currentmodule:: iduedu

Every public name is imported from the package itself, ``from iduedu import ...``. The tables link to the reference
page of each function; the topic pages explain how they work.

Building graphs
---------------

:doc:`building/sources`, :doc:`building/streets`, :doc:`building/public_transport`, :doc:`building/gtfs`,
:doc:`building/intermodal`

.. autosummary::
    :nosignatures:

    get_drive_graph
    get_walk_graph
    get_bike_graph
    get_public_transport_graph
    get_gtfs_public_transport_graph
    get_intermodal_graph
    join_pt_walk_graph
    merge_gtfs_feeds
    deduplicate_across_feeds
    TransportSpec
    TransportRegistry
    get_4326_boundary
    use_pbf
    download_osm_pbf
    download_geofabrik_region
    find_geofabrik_region
    geofabrik_regions
    PbfSource
    split_by_polygons
    snapshot_time

The graph model
---------------

:doc:`graph/urban_graph`, :doc:`graph/operations`

.. autosummary::
    :nosignatures:

    UrbanGraph
    read_urban_graph
    write_urban_graph
    validate_graph
    subgraph_by_nodes
    clip_urban_graph
    join_urban_graphs
    relabel_urban_graph
    to_directed
    to_undirected
    simplify_multiedges
    consolidate_intersections
    unique_segments
    connected_components
    weakly_connected_components
    strongly_connected_components
    keep_largest_connected_component
    nearest_nodes
    nearest_edges
    project_objects2urban_graph
    apply_urban_graph_changes
    UrbanGraphChanges

Routing
-------

:doc:`routing/shortest_paths`, :doc:`routing/point_to_point`

.. autosummary::
    :nosignatures:

    single_source_dijkstra
    multi_source_dijkstra
    multi_source_dijkstra_nearest_source
    dijkstra_parallel
    single_source_dijkstra_path
    od_matrix
    iter_od_matrix
    path_to_edges
    ShortestPathResult
    shortest_path
    alternative_paths
    k_shortest_paths

Network analysis
----------------

:doc:`analysis/statistics`, :doc:`analysis/objects`, :doc:`analysis/centrality`, :doc:`analysis/elevation`

.. autosummary::
    :nosignatures:

    graph_stats
    local_stats
    edge_stats
    transport_stats
    streets_per_node
    route_directness
    nearest_objects
    aggregate_objects
    diversity
    centrality
    edge_betweenness
    add_elevation
    apply_grade_speed

NetworkX
--------

Optional; install ``networkx`` separately (see :doc:`graph/operations`).

.. autosummary::
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

Settings
--------

``iduedu.config`` holds the Overpass endpoint, the cache, the OSM source and logging; see :doc:`configuration`.
