"""
Local OpenStreetMap files as an alternative to the Overpass API.

``config.set_osm_source("pbf", path=...)`` (or ``with config.use_pbf(path): ...``) makes every graph builder
read OSM data from a local ``.osm.pbf`` file instead of Overpass. :func:`split_by_polygons` cuts compact
per-city files out of a regional extract, :func:`snapshot_time` reads the date a file reflects.

Requires the optional dependency pyosmium: ``pip install 'iduedu[pbf]'``. Importing this package works
without it; using it raises an ``ImportError`` with the installation hint.
"""

from .extract import split_by_polygons
from .filters import UnsupportedFilterError, WayFilter, parse_way_filter
from .reader import PbfSource, snapshot_time

__all__ = [
    "PbfSource",
    "UnsupportedFilterError",
    "WayFilter",
    "parse_way_filter",
    "snapshot_time",
    "split_by_polygons",
]
