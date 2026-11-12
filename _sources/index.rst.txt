.. toctree::
    :hidden:
    :caption: Building graphs
    :maxdepth: 1

    building/sources
    building/streets
    building/public_transport
    building/gtfs
    building/intermodal

.. toctree::
    :hidden:
    :caption: Graph model
    :maxdepth: 1

    graph/urban_graph
    graph/operations

.. toctree::
    :hidden:
    :caption: Routing
    :maxdepth: 1

    routing/shortest_paths
    routing/point_to_point

.. toctree::
    :hidden:
    :caption: Network analysis
    :maxdepth: 1

    analysis/statistics
    analysis/objects
    analysis/centrality
    analysis/elevation

.. toctree::
    :hidden:
    :caption: Reference
    :maxdepth: 1

    examples/index
    configuration
    api
    performance
    upgrading

IduEdu
======

Multimodal city networks from OpenStreetMap and GTFS
----------------------------------------------------

**IduEdu** is an open-source Python library that builds street, cycling, public-transport and intermodal
networks of a city from `OpenStreetMap <https://www.openstreetmap.org>`__ and GTFS Schedule feeds, routes on
them and measures them. It is developed by the IDU team at ITMO University.

|badge-black| |badge-pypi| |badge-ci| |badge-cov| |badge-license| |badge-github|

.. |badge-black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

.. |badge-pypi| image:: https://img.shields.io/pypi/v/iduedu.svg
   :target: https://pypi.org/project/iduedu/
   :alt: PyPI version

.. |badge-ci| image:: https://github.com/IDUclub/IduEdu/actions/workflows/quality.yml/badge.svg
   :target: https://github.com/IDUclub/IduEdu/actions/workflows/quality.yml
   :alt: Tests and coverage

.. |badge-cov| image:: https://raw.githubusercontent.com/IDUclub/IduEdu/python-coverage-comment-action-data/badge.svg
   :target: https://github.com/IDUclub/IduEdu/tree/python-coverage-comment-action-data
   :alt: Coverage

.. |badge-license| image:: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License

.. |badge-github| image:: https://img.shields.io/badge/GitHub-IDUclub%2FIduEdu-181717?logo=github
   :target: https://github.com/IDUclub/IduEdu
   :alt: GitHub

----

Overview
--------

Every network is an :class:`~iduedu.UrbanGraph`: a table of nodes and a table of edges (GeoDataFrames in a local
metric CRS) with the edge costs ``length_meter`` (metres) and ``time_min`` (minutes). The graph algorithms run
in parallel threads in a compiled Rust extension, ``iduedu._rust``. Every page below describes what is computed,
the method, the parameters and outputs with their units, figures and the limitations.

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Topic
     - Result
     - Functions
   * - :doc:`building/sources`
     - OSM data from the Overpass API or from a local ``.osm.pbf`` file, downloaded from Geofabrik and cut to
       the territory
     - :func:`~iduedu.use_pbf`, :func:`~iduedu.download_osm_pbf`, :func:`~iduedu.split_by_polygons`
   * - :doc:`building/streets`
     - Walk, drive and bike graphs with travel times, kept OSM tags, bridges and tunnels, cycling stress levels
     - :func:`~iduedu.get_walk_graph`, :func:`~iduedu.get_drive_graph`, :func:`~iduedu.get_bike_graph`
   * - :doc:`building/public_transport`
     - Bus, trolleybus, tram, subway, monorail and train routes from OSM, timed by a speed model per mode
     - :func:`~iduedu.get_public_transport_graph`, :class:`~iduedu.TransportRegistry`
   * - :doc:`building/gtfs`
     - Public transport from GTFS Schedule feeds, with waiting times from the timetable
     - :func:`~iduedu.get_gtfs_public_transport_graph`, :func:`~iduedu.merge_gtfs_feeds`
   * - :doc:`building/intermodal`
     - Walking and public transport in one graph, joined at platforms and station entrances
     - :func:`~iduedu.get_intermodal_graph`, :func:`~iduedu.join_pt_walk_graph`
   * - :doc:`graph/urban_graph`
     - The graph model: node and edge tables, types, validation, storage, NetworkX
     - :class:`~iduedu.UrbanGraph`, :func:`~iduedu.read_urban_graph`
   * - :doc:`graph/operations`
     - Clipping, subgraphs, joins, components, merged junctions, objects projected onto edges
     - :func:`~iduedu.clip_urban_graph`, :func:`~iduedu.project_objects2urban_graph`
   * - :doc:`routing/shortest_paths`
     - Costs and routes from one or many sources, origin–destination matrices
     - :func:`~iduedu.od_matrix`, :func:`~iduedu.single_source_dijkstra`
   * - :doc:`routing/point_to_point`
     - One route by bidirectional search, A* or ALT; alternative routes; k shortest routes
     - :func:`~iduedu.shortest_path`, :func:`~iduedu.alternative_paths`, :func:`~iduedu.k_shortest_paths`
   * - :doc:`analysis/statistics`
     - Length, density, junctions and directness of a network, by area or zone
     - :func:`~iduedu.graph_stats`, :func:`~iduedu.local_stats`
   * - :doc:`analysis/objects`
     - Nearest objects, counts and sums of objects within a travel cost, diversity
     - :func:`~iduedu.aggregate_objects`, :func:`~iduedu.nearest_objects`
   * - :doc:`analysis/centrality`
     - Reach, closeness, gravity and betweenness of nodes and edges
     - :func:`~iduedu.centrality`, :func:`~iduedu.edge_betweenness`
   * - :doc:`analysis/elevation`
     - Heights of nodes, grades of edges, travel times that depend on the grade
     - :func:`~iduedu.add_elevation`, :func:`~iduedu.apply_grade_speed`

:doc:`examples/index` contains an executable notebook for every topic.

----

Installation
------------

::

   pip install iduedu

IduEdu supports Python 3.11–3.14 on Linux, macOS and Windows. Wheels contain the compiled extension, so ``pip``
needs no compiler; building from a source checkout requires a Rust toolchain (`rustup <https://rustup.rs>`__).
Optional extras: ``iduedu[io]`` writes and reads ``.urbangraph`` archives, ``iduedu[elevation]`` reads elevation
rasters (rasterio). NetworkX is not installed with IduEdu and is needed only by the NetworkX helpers. Changes for
code written for 2.x are listed in :doc:`upgrading`.

----

Quickstart
----------

The builders download the territory from the Overpass API, or read it from a local OSM file. The second way is
faster for large areas and does not depend on the public Overpass instance:

.. code-block:: python

   import iduedu
   from iduedu import get_4326_boundary, get_intermodal_graph

   # Vasilyevsky Island district, Saint Petersburg (OSM relation 1114252)
   with iduedu.use_pbf(osm_id=1114252, region="northwestern-fed-district"):
       boundary = get_4326_boundary(osm_id=1114252)
       graph = get_intermodal_graph(territory=boundary, clip_by_territory=True)

Without the ``with`` block the same calls query the Overpass API. Routes and matrices then take node ids or
GeoDataFrames of objects, which are matched to their nearest nodes:

.. code-block:: python

   nodes = graph.nodes_gdf.index
   cost, route = graph.shortest_path(nodes[0], nodes[-1], weight="time_min")  # minutes, node ids
   edges = graph.path_to_edges(route)  # the edges of the route with their geometry

   matrix = graph.od_matrix(gdf_origins=buildings, gdf_destinations=schools, cutoff=30)

Network measures work on the same graph:

.. code-block:: python

   from iduedu import aggregate_objects, centrality

   reach = centrality(graph, ["reach", "harmonic"], cutoffs=[10, 20], weight="time_min")
   schools_within = aggregate_objects(graph, [10, 20], gdf_origins=buildings, gdf_objects=schools)

Configuration
-------------

Settings such as the Overpass endpoint, the response cache, a historical OSM date or the OSM tags kept on edges
live in the global ``config`` object (see :doc:`configuration`):

.. code-block:: python

   from iduedu import config

   config.set_overpass_cache(cache_dir=".iduedu_cache", enabled=True)
   config.configure_logging(level="INFO")

----

Contributing, contacts and citation
-----------------------------------

Issues and pull requests are welcome on `GitHub <https://github.com/IDUclub/IduEdu>`__; the workflow is described
in ``CONTRIBUTING.md``.

- `NCCR <https://actcognitive.org/>`_, National Center for Cognitive Research
- `IDU <https://idu.itmo.ru/>`_, Institute of Design and Urban Studies
- `Natalya Chichkova <https://t.me/nancy_nat>`_, project manager
- `Danila Oleynikov (Donny) <https://t.me/ddonny_dd>`_, lead software engineer

This research is financially supported by the Foundation for National Technology Initiative's Projects Support
as a part of the roadmap implementation for the development of the high-tech field of Artificial Intelligence
for the period up to 2030 (agreement 70-2021-00187).

IduEdu is distributed under the BSD 3-Clause license (``LICENSE.txt``). To cite it, use the metadata in
`CITATION.cff <https://github.com/IDUclub/IduEdu/blob/main/CITATION.cff>`__ (on GitHub, **Cite this repository**
exports it as APA or BibTeX). Publications describing IduEdu will be listed here.
