Overpass helpers
----------------

.. currentmodule:: iduedu

Boundary normalization
======================

Use :func:`get_4326_boundary` to resolve a territory into a single EPSG:4326
polygon. The input can be an OSM relation id, a Shapely polygon or multipolygon,
or a GeoDataFrame/GeoSeries. Graph builders call this helper internally, but it
is also useful when one boundary should be reused across several builders.

See :doc:`../examples/get_any_graph` for the common ``osm_id`` workflow.

.. autosummary::
    :toctree: generated
    :nosignatures:

    get_4326_boundary
