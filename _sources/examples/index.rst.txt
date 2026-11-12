Examples
========

Every notebook runs on Vasilyevsky Island in Saint Petersburg (OSM relation 1114252) and ends with the changes
needed for your own area. The notebooks are stored with their outputs; to run one, install IduEdu and open it in
Jupyter.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Notebook
     - Contents
   * - :doc:`get_any_graph`
     - Drive, walk, public-transport and intermodal graphs; ``simplify`` and clipping
   * - :doc:`urban_graph_basics`
     - The node and edge tables, validation, adjacency matrices, ``.urbangraph`` archives
   * - :doc:`graph_operations`
     - Subgraphs, clipping, joins, relabelling, direction and parallel edges
   * - :doc:`connectivity`
     - Connected, weakly and strongly connected components
   * - :doc:`objects_and_nearest_nodes`
     - Nearest nodes and edges, objects projected onto the graph
   * - :doc:`shortest_paths`
     - Costs and routes from one or many sources, OD matrices, lazy paths, matrices in blocks
   * - :doc:`routing_methods`
     - Bidirectional search, A* and ALT compared; alternative routes and k shortest routes
   * - :doc:`network_analysis`
     - Network statistics by zone and around nodes, residents within a walking time, nearest objects, diversity,
       centralities
   * - :doc:`elevation`
     - Heights and grades, walking times on slopes, a bike graph with its level of traffic stress
   * - :doc:`transport_registry`
     - The default speeds and waits of public transport, a custom registry
   * - :doc:`gtfs_public_transport`
     - A graph from GTFS feeds: service days, waits, merging feeds, joining with a walk graph

.. toctree::
    :hidden:
    :maxdepth: 1

    get_any_graph
    urban_graph_basics
    graph_operations
    connectivity
    objects_and_nearest_nodes
    shortest_paths
    routing_methods
    network_analysis
    elevation
    transport_registry
    gtfs_public_transport
