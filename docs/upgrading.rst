Upgrading to 3.0
================

.. currentmodule:: iduedu

IduEdu 3.0 replaces the Numba kernels with a compiled Rust core and adds point-to-point routing, network
statistics, object aggregations, centralities, elevation and bike graphs. Most code written for 2.x runs unchanged.
This page lists what was removed and what changed its results.

Installation
------------

- Numba and llvmlite are no longer dependencies, and there is no JIT warm-up on the first call. Wheels are
  published for Linux (x86_64, aarch64), Windows (x64) and macOS (arm64, x86_64); building from source needs a
  stable Rust toolchain (1.82 or newer).
- pyosmium is no longer used: ``.osm.pbf`` files are read and written by the Rust core. The ``pbf`` extra is
  removed; drop it from your requirements, since ``pip install "iduedu[pbf]"`` now warns about an unknown extra.
- The new ``elevation`` extra installs rasterio for :func:`add_elevation` with raster files.

Removed functions and arguments
-------------------------------

The routing wrappers deprecated in 2.4.1 are removed, as functions and as :class:`UrbanGraph` methods. Each called a
general function with a fixed ``output``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Removed
     - Replacement
   * - ``single_source_dijkstra_path_length``
     - ``single_source_dijkstra(..., output="distances")``
   * - ``multi_source_dijkstra_path_length``
     - ``multi_source_dijkstra(..., output="distances")``
   * - ``dijkstra_path_length_parallel``
     - ``dijkstra_parallel(..., output="distances")``
   * - ``multi_source_dijkstra_path``
     - ``od_matrix(..., output="paths")``; despite its name it computed independent origin–destination routes,
       not a nearest-source search

The ``location_storage`` argument of :class:`PbfSource`, :func:`split_by_polygons` and :func:`download_osm_pbf` is
removed too: it chose the node index of pyosmium, and the Rust reader needs none.

Changed behaviour
-------------------------------

Areas in EPSG:4326
   :func:`clip_urban_graph`, :meth:`UrbanGraph.clip` and :func:`clip_nx_graph` read a bare shapely polygon in
   EPSG:4326, like ``territory`` in the graph builders and ``area`` in the new analysis functions. A GeoSeries or
   GeoDataFrame carries its own CRS. To clip by a polygon in the graph CRS, pass
   ``gpd.GeoSeries([polygon], crs=graph.crs)``.

Edges of simplified street graphs
   With ``simplify=True``, segments are merged only where the tags of the ways are equal, so an edge ends wherever
   a tag changes and keeps the tags of all its ways (see :doc:`building/streets`). Drive and walk graphs have more
   nodes and edges than in 2.x; network length, intersections and dead ends are unchanged. The ``bridge`` and
   ``tunnel`` columns are always present, and bridges and tunnels are separate edges.

Parallel edges of undirected graphs
   The adjacency matrix takes the cheapest of the parallel edges between two nodes in either orientation. In 2.x
   two edges of opposite orientation in an undirected multigraph were summed, which made some walking routes too
   long. Travel times on walk graphs can therefore be shorter than in 2.x.

Local OSM files
   Only ``.osm.pbf`` files are read; convert ``.osm`` and ``.opl`` files with
   ``osmium cat city.osm -o city.osm.pbf``. Files written by :func:`split_by_polygons` carry no element metadata
   (versions, timestamps, users).

Edits of edge weights
   The cached adjacency matrix is rebuilt when the weights, directions or ends of the edges change after a query,
   also when the weights are edited in place. In 2.x such edits were ignored until
   :meth:`UrbanGraph.update_adjacency_matrix` was called, and the queries returned the old costs.

Single routes
   :func:`single_source_dijkstra_path` and :func:`od_matrix` with ``mode="pairwise"`` run the bidirectional search
   (see :doc:`routing/point_to_point`). The costs are the same; where several routes are equally short, another of
   them can be returned.

New in 3.0
----------

- Routes between two points with bidirectional search, A* and ALT, alternative routes and k shortest routes:
  :doc:`routing/point_to_point`.
- Network statistics, objects around origins and centralities: :doc:`analysis/statistics`,
  :doc:`analysis/objects`, :doc:`analysis/centrality`.
- Heights, grades and speeds on slopes: :doc:`analysis/elevation`.
- Bike graphs with the level of traffic stress: :doc:`building/streets`.
- :func:`nearest_edges`, :func:`unique_segments` and :func:`consolidate_intersections`: :doc:`graph/operations`.
