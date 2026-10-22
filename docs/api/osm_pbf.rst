Local OSM files
---------------

.. currentmodule:: iduedu

With the optional dependency pyosmium (``pip install 'iduedu[pbf]'``) every
OSM-based builder can read local ``.osm.pbf`` files instead of querying
Overpass. Select the source with ``config.set_osm_source("pbf", path=...)`` or
``with config.use_pbf(path): ...``; see :doc:`../configuration` for the full
workflow and the known differences from Overpass.

Reading and splitting files
===========================

.. autosummary::
    :toctree: generated
    :nosignatures:

    PbfSource
    split_by_polygons
    snapshot_time

Overpass tag filters
====================

.. currentmodule:: iduedu.osm_pbf

.. autosummary::
    :toctree: generated
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

.. automodule:: iduedu.osm_pbf.filters
    :no-members:
