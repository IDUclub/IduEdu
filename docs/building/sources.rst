Data sources
============

.. currentmodule:: iduedu

Overview
--------

The OSM graph builders read OpenStreetMap from one of two sources, and the GTFS builder reads a local timetable.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Source
     - Use it for
     - Selected with
   * - Overpass API (default)
     - One territory at a time, current data or a past date, no files to manage
     - ``config.set_overpass_url``, ``config.set_overpass_date`` (see :doc:`../configuration`)
   * - Local ``.osm.pbf`` file
     - Many territories, repeatable runs, work without Overpass
     - ``config.set_osm_source("pbf", path=...)`` or ``with use_pbf(...)``
   * - GTFS Schedule feed
     - Public transport with timetable travel and waiting times
     - :func:`get_gtfs_public_transport_graph` (see :doc:`gtfs`)

Every builder answers the same questions from either OSM source: the street ways of a network type, the
public-transport route relations, the stop areas of the subway and the territory boundary. The filters and the
selection by polygon are the same, so a graph built from a file equals the graph built from Overpass at the moment
the file reflects.

Territory
---------

Every builder takes the territory as ``osm_id`` (an OSM relation, for example a municipal boundary) or as
``territory``: a shapely polygon in EPSG:4326 (longitude, latitude) or a GeoDataFrame in any CRS.
:func:`get_4326_boundary` returns the polygon the builders use. The graph is built in a local projected CRS
estimated from the territory, so lengths are in metres. Ways and routes that touch the polygon are read whole;
``clip_by_territory=True`` cuts the edges at the boundary.

.. code-block:: python

   boundary = iduedu.get_4326_boundary(osm_id=1114252)        # Vasilyevsky Island, Saint Petersburg
   walk = iduedu.get_walk_graph(territory=boundary, clip_by_territory=True)

Overpass API
------------

By default the builders send queries to the public Overpass instance. Responses are cached on disk in
``.iduedu_cache``, requests are spaced and retried, and ``config.set_overpass_date`` answers a query as OpenStreetMap
stood at a past moment. :doc:`../configuration` describes the endpoint, the cache, the pacing and the snapshot
dates.

Local OSM files
---------------

The Rust core of IduEdu reads ``.osm.pbf`` files with no further dependency. Only ``.osm.pbf`` is read; convert
``.osm`` or ``.opl`` files first, for example with ``osmium cat city.osm -o city.osm.pbf``.

.. code-block:: python

   from iduedu import config, get_public_transport_graph, get_walk_graph, use_pbf

   config.set_osm_source("pbf", path="city.osm.pbf")   # every following call reads the file
   walk = get_walk_graph(territory=city_polygon)
   config.set_osm_source("overpass")                   # back to the default

   with use_pbf("city.osm.pbf"):                       # or only inside a block
       pt = get_public_transport_graph(territory=city_polygon, transport_types=["bus", "tram", "subway"])

The environment variables ``IDUEDU_OSM_SOURCE=pbf`` and ``IDUEDU_PBF_PATH=/data/city.osm.pbf`` do the same at
import time. In pbf mode the Overpass cache is not used and ``overpass_date`` is ignored: the file is the snapshot,
and :func:`snapshot_time` returns the moment it reflects, read from its header
(``osmosis_replication_timestamp``, set in Geofabrik extracts).

Downloading from Geofabrik
~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`download_osm_pbf` takes the territory in the same form as the builders and returns a small file of that
territory. It picks the smallest `Geofabrik <https://download.geofabrik.de/>`__ extract that covers the territory,
downloads it once (an interrupted download resumes, and the md5 sum Geofabrik publishes is checked), and cuts the
territory out with :func:`split_by_polygons`. :func:`use_pbf` accepts the same arguments, downloads the file
and reads from it inside the block.

.. code-block:: python

   with iduedu.use_pbf(osm_id=1114252, region="northwestern-fed-district"):
       boundary = iduedu.get_4326_boundary(osm_id=1114252)
       graph = iduedu.get_intermodal_graph(territory=boundary, clip_by_territory=True)

   with iduedu.use_pbf(territory=city_polygon):        # the extract is chosen from the territory
       walk = iduedu.get_walk_graph(territory=city_polygon)

With ``region`` the boundary of ``osm_id`` is read from the extract and no Overpass request is made; without it the
boundary is fetched once through Overpass to choose the extract. ``territory`` needs neither. Extracts and cut
files are cached under ``config.overpass_cache_dir`` (``geofabrik/`` and ``geofabrik/territories/``). An existing
cut file is returned without a download, and a cached extract is reused until ``update=True`` asks for the newer
file of Geofabrik. :func:`geofabrik_regions` lists the extracts with their outlines, and
:func:`find_geofabrik_region` names the smallest one that covers a territory.

Per-city files from a regional extract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A query to a country-sized file scans all of it for every territory. :func:`split_by_polygons` cuts it into small
city files in one pass, for all cities of the region at once:

.. code-block:: python

   from iduedu import snapshot_time, split_by_polygons

   paths = split_by_polygons(
       "russia-northwestern-fed-district-latest.osm.pbf",
       {"spb": spb_polygon, "petrozavodsk": ptz_polygon},
       "cities/",
   )
   snapshot_time(paths["spb"])                  # the replication timestamp of the extract is kept

Each city file keeps, complete and with their tags:

- the street ways (``highway``, ``railway``, ``public_transport``) that touch the polygon;
- the public-transport route relations that touch it, with all their members, also those outside the polygon;
- for the subway, every relation that shares a node with a line (its stop areas), their members and the stop-area
  groups that contain them;
- the relations named in ``keep_relations``, for instance the boundary, so that ``osm_id`` keeps working.

Element metadata (versions, timestamps, users) is not copied. Graphs built from a city file equal those built from
the regional file, for the same polygon or any polygon inside it.

Differences from Overpass
~~~~~~~~~~~~~~~~~~~~~~~~~

- Elements are returned in id order. Overpass returns route responses in quadtile order, so the node and edge ids
  of a public-transport graph can be numbered differently although the graph is the same.
- Points on the polygon boundary count as inside. Overpass can treat exact boundary hits and long polygon edges
  differently in rare cases.
- Only tag filters are supported (``["k"]``, ``[!"k"]``, ``["k"="v"]``, ``["k"!="v"]``, ``["k"~"re"]``,
  ``["k"!~"re"]``, with an optional ``,i`` flag on regular expressions); other Overpass syntax in ``custom_filter``
  raises :class:`~iduedu.osm_pbf.UnsupportedFilterError`. Regular expressions use the Python ``re`` module.
- Data missing from the file cannot be found: members of a relation that are not in the file have no geometry. Use
  :func:`split_by_polygons` or a complete regional extract, and avoid a tight clip.
- :func:`split_by_polygons` selects ways by location only when they carry one of ``way_keys``; a ``custom_filter``
  on other keys needs ``way_keys=None`` (or a list that includes them) when the city files are cut.

Functions
---------

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    get_4326_boundary
    use_pbf
    download_osm_pbf
    download_geofabrik_region
    find_geofabrik_region
    geofabrik_regions
    PbfSource
    split_by_polygons
    snapshot_time

Overpass tag filters of the file reader:

.. currentmodule:: iduedu.osm_pbf

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    parse_way_filter
    WayFilter
    UnsupportedFilterError

Limitations
-----------

- A file reflects one moment; for another date, download a newer extract or use ``config.set_overpass_date`` with
  Overpass.
- The Geofabrik extract of a large region is a large download (about 650 MB for the Northwestern Federal District
  of Russia); it is downloaded once and cached.
- Graphs from Overpass and from a file of a different date differ by the edits made between them.

Example notebook
----------------

:doc:`../examples/get_any_graph`
