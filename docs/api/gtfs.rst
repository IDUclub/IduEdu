GTFS public transport
=====================

IduEdu can build a static public-transport :class:`iduedu.UrbanGraph` from a
local `GTFS Schedule <https://gtfs.org/documentation/schedule/reference/>`_
feed. The source may be either a directory containing ``*.txt`` tables or a
ZIP archive.

Building a graph
----------------

.. code-block:: python

    from iduedu import get_gtfs_public_transport_graph

    public_transport = get_gtfs_public_transport_graph(
        "feed.zip",
        service_date="2026-08-04",
        start_time="07:00:00",
        end_time="10:00:00",
    )

``service_date`` applies ``calendar.txt`` and ``calendar_dates.txt``. It accepts
a Python ``date`` or ``datetime``, ``YYYY-MM-DD``, or ``YYYYMMDD``. ``start_time``
is inclusive and ``end_time`` is exclusive; both accept GTFS time strings,
Python ``time`` values, or seconds. An end time earlier than the start time
selects an overnight window.

Omitting all three filters intentionally aggregates every trip in the feed.
This can combine weekday, weekend, and seasonal service, so pass a date and
time window when the graph should represent a particular operating period.

Required and optional tables
----------------------------

The builder requires ``agency.txt``, ``stops.txt``, ``routes.txt``,
``trips.txt``, and ``stop_times.txt``. At least one of ``calendar.txt`` and
``calendar_dates.txt`` must also be present.

The first implementation additionally reads ``frequencies.txt``,
``shapes.txt``, ``pathways.txt``, ``levels.txt``, and ``feed_info.txt`` when
available. ``transfers.txt`` and exact time-dependent transfer routing are not
implemented yet.

Standard route types are represented as ``tram`` (0), ``subway`` (1),
``train`` (2), ``bus`` (3), ``trolleybus`` (11), and ``monorail`` (12).
Common extended route-type ranges for rail, subway, bus, and tram are also
normalized; unknown types remain ``public_transport``.

Static time model
-----------------

Trips with the same route, direction, shape, stop order, and pickup/drop-off
rules form a route pattern. For each pattern segment, ``time_min`` is the
median scheduled time from departure at one stop to arrival at the next.

Boarding edges represent expected waiting rather than vehicle motion:

- scheduled service uses half the mean gap between consecutive departures at
  the pattern stop;
- ``frequencies.txt`` service uses ``headway_secs / 120`` and duration-weighted
  averaging when frequency periods overlap the selected window;
- a pattern stop with only one scheduled departure has no estimable headway,
  so its boarding edge is omitted by default. Set
  ``single_departure_wait_min`` to retain it with an explicit fallback.

Boarding and alighting edges always have ``length_meter = 0``. Alighting also
has ``time_min = 0``. GTFS waiting times come from the feed and do not use
:class:`iduedu.TransportRegistry` defaults.

Geometry and distance
---------------------

When ``shapes.txt`` supplies usable route geometry, segment geometry is cut
from the matching shape and ``length_meter`` is its measured length. Otherwise
the builder creates a straight line between consecutive route-stop points and
assigns ``sqrt(2) * straight_line_length`` as the estimated travel distance.
The square-root-of-two correction applies only to vehicle movement edges such
as ``bus``, ``tram``, or ``subway``; it never applies to boarding or alighting.

The graph CRS must be projected in metres. If ``crs`` is omitted, IduEdu
estimates a local UTM CRS from the feed stops.

Stations, pathways, and intermodal joining
------------------------------------------

``pathways.txt`` creates station-internal ``pathway`` edges. Explicit
``traversal_time`` is converted from seconds to minutes; otherwise time is
estimated from pathway length and ``walk_speed_m_per_min``. Station entrances,
nodes, boarding areas, and platforms are retained when referenced by the
served stops or pathways.

To join GTFS public transport to an OSM walking graph, build the walk layer
first and use the same CRS:

.. code-block:: python

    from iduedu import get_gtfs_public_transport_graph, get_walk_graph, join_pt_walk_graph

    walk = get_walk_graph(osm_id=1114252)
    public_transport = get_gtfs_public_transport_graph(
        "feed.zip",
        service_date="2026-08-04",
        start_time="07:00:00",
        end_time="10:00:00",
        crs=walk.crs,
    )
    intermodal = join_pt_walk_graph(public_transport, walk)

API reference
-------------

.. currentmodule:: iduedu

.. autofunction:: get_gtfs_public_transport_graph
    :no-index:
