Local OSM files
---------------

.. currentmodule:: iduedu

Every OSM-based builder can read local ``.osm.pbf`` files instead of
querying Overpass; IduEdu's Rust core reads and writes them, with no further
dependency. Select the source with ``config.set_osm_source("pbf", path=...)`` or
``with use_pbf(path): ...``; ``use_pbf(osm_id=...)`` downloads the territory
from Geofabrik first. See :doc:`../configuration` for the full workflow and the
known differences from Overpass.

Downloading from Geofabrik
==========================

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    get_4326_boundary
    use_pbf
    download_osm_pbf
    download_geofabrik_region
    find_geofabrik_region
    geofabrik_regions

Reading and splitting files
===========================

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    PbfSource
    split_by_polygons
    snapshot_time

Overpass tag filters
====================

.. currentmodule:: iduedu.osm_pbf

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    parse_way_filter
    WayFilter
    UnsupportedFilterError

Module reference
================

.. automodule:: iduedu.osm_pbf.reader
    :no-members:

.. automodule:: iduedu.osm_pbf.extract
    :no-members:

.. automodule:: iduedu.osm_pbf.download
    :no-members:

.. automodule:: iduedu.osm_pbf.filters
    :no-members:
