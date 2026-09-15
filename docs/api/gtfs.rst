GTFS public transport
=====================

IduEdu can build a static public-transport :class:`iduedu.UrbanGraph` from a
local `GTFS Schedule <https://gtfs.org/documentation/schedule/reference/>`_
feed. The source may be a directory containing ``*.txt`` tables, a ZIP archive,
an already-read :class:`iduedu.gtfs.GTFSFeed`, or a list of those merged into
one feed (see `Merging several feeds`_). The :doc:`../examples/gtfs_public_transport`
notebook walks through every step below on small synthetic feeds.

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

The builder also reads ``frequencies.txt``, ``shapes.txt``, ``pathways.txt``,
``levels.txt``, and ``feed_info.txt`` when available. ``transfers.txt`` and
exact time-dependent transfer routing are not implemented yet.

Base route types are represented as ``tram`` (0), ``subway`` (1), ``train`` (2),
``bus`` (3), ``ferry`` (4), ``cable_tram`` (5), ``aerial_lift`` (6),
``funicular`` (7), ``trolleybus`` (11), and ``monorail`` (12). Google's extended
route types are mapped by explicit ranges: 100-117 ``train``, 200-209 ``coach``
(kept apart from city buses), 400-404 and 406-499 ``subway``, 405 ``monorail``,
700-716 ``bus``, 800-899 ``trolleybus``, 900-906 ``tram``, 1000 and 1200
``ferry``, 1300 ``aerial_lift``, 1400 ``funicular``, and 1500 ``taxi``. Any
other code, including the miscellaneous 300, 500, 600, 1600 and 1700 ranges,
remains ``public_transport`` rather than a guess from neighbouring codes.

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

Merging several feeds
---------------------

A city is often published as several feeds, and every feed numbers its own
routes, trips, and stops. Pass a list of sources to merge them before the graph
is built:

.. code-block:: python

    public_transport = get_gtfs_public_transport_graph(
        ["subway.zip", "bus.zip"],
        service_date="2026-08-04",
        start_time="07:00:00",
        end_time="10:00:00",
    )

A list is merged with :func:`iduedu.merge_gtfs_feeds`, which qualifies every
identifier by its source, so route ``1`` of one feed and route ``1`` of another
stay distinct. Call it directly to choose the prefixes or to fuse stops that
different agencies publish for the same place:

.. code-block:: python

    from iduedu import get_gtfs_public_transport_graph, merge_gtfs_feeds

    feed = merge_gtfs_feeds(["subway.zip", "bus.zip"], merge_stops_within=20)
    public_transport = get_gtfs_public_transport_graph(feed, service_date="2026-08-04")

Stops are kept distinct by default: fusing them changes the network topology,
and an inferred transfer would look like a published one. Distances of about
15-25 m suit feeds from different agencies.

Stations, pathways, and intermodal joining
------------------------------------------

``pathways.txt`` creates station-internal ``pathway`` edges. Explicit
``traversal_time`` is converted from seconds to minutes; otherwise time is
estimated from pathway length and ``walk_speed_m_per_min``. Station entrances,
nodes, boarding areas, and platforms are retained when referenced by the
served stops or pathways.

A stop that names a ``parent_station`` is linked to it with a zero-cost,
bidirectional ``station_link`` edge. Only stops already present in the graph
are linked. Without these edges, a station described down to boarding areas
would leave its platforms in a separate component that ``keep_largest_subgraph``
removes.

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

.. autofunction:: merge_gtfs_feeds

.. currentmodule:: iduedu.gtfs

.. autofunction:: read_gtfs_feed

.. autofunction:: validate_gtfs_feed

.. autoclass:: GTFSFeed
    :members:

.. autoexception:: GTFSValidationError
