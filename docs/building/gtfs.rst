Public transport from GTFS
==========================

.. currentmodule:: iduedu

Overview
--------

:func:`get_gtfs_public_transport_graph` builds a static public-transport :class:`UrbanGraph` from a local
`GTFS Schedule <https://gtfs.org/documentation/schedule/reference/>`__ feed. Travel times between stops are the
scheduled times, and waiting times at boarding come from the gaps between departures, so the graph follows the
timetable of the city for the chosen days and hours.

.. code-block:: python

   from iduedu import get_gtfs_public_transport_graph

   public_transport = get_gtfs_public_transport_graph(
       "feed.zip",
       service_date="2026-08-04",
       start_time="07:00:00",
       end_time="10:00:00",
   )

The source may be a directory of ``*.txt`` tables, a ZIP archive, a feed already read with
:func:`~iduedu.gtfs.read_gtfs_feed`, or a list of those merged into one feed (see `Merging several feeds`_).

Inputs
------

Tables
~~~~~~

The builder requires ``agency.txt``, ``stops.txt``, ``routes.txt``, ``trips.txt`` and ``stop_times.txt``, and at
least one of ``calendar.txt`` and ``calendar_dates.txt``. It also reads ``frequencies.txt``, ``shapes.txt``,
``pathways.txt``, ``levels.txt`` and ``feed_info.txt`` when they are present. ``transfers.txt`` is not read.

Service days and hours
~~~~~~~~~~~~~~~~~~~~~~

``service_date`` selects one day through ``calendar.txt`` and ``calendar_dates.txt``. It accepts a Python ``date``
or ``datetime``, ``YYYY-MM-DD`` or ``YYYYMMDD``. ``start_time`` (inclusive) and ``end_time`` (exclusive) select the
departures of a time window; both accept GTFS time strings, Python ``time`` values or seconds. An end time earlier
than the start time selects an overnight window.

Without ``service_date`` the graph holds every route pattern that runs on at least one day, and its waiting times
are those of the *average day*. Every service day is computed on its own trips, and the waits are averaged over the
days a pattern runs, with every calendar day weighing the same. The days are those of the feed calendar (at most
366 days from its start) or of ``service_period=(first, last)``, for example one reference week. A gap between
departures is always a gap within one day: a weekday departure at 08:00 and a Saturday departure at 08:10 do not
make a wait of 10 minutes. ``start_time`` and ``end_time`` apply within each day, for example
``start_time="06:00", end_time="22:00"`` for daytime service.

.. code-block:: python

   # the average day of one reference week, daytime service only
   average_week = get_gtfs_public_transport_graph(
       "feed.zip",
       service_period=("2026-10-12", "2026-10-18"),
       start_time="06:00",
       end_time="22:00",
   )

``service_dates=[...]`` lists the days explicitly, for example a reference week plus the weeks of merged sources that
were not in service then. Every day still holds only its own trips, so days of different years do not mix.
``service_date``, ``service_period`` and ``service_dates`` cannot be combined. A feed without ``calendar.txt`` and
``calendar_dates.txt`` is taken as a single day.

Route types
~~~~~~~~~~~

The ``type`` of the travel edges comes from ``route_type``. The base types are ``tram`` (0), ``subway`` (1),
``train`` (2), ``bus`` (3), ``ferry`` (4), ``cable_tram`` (5), ``aerial_lift`` (6), ``funicular`` (7),
``trolleybus`` (11) and ``monorail`` (12). The extended route types of Google are mapped by explicit ranges: 100–117
``train``, 200–209 ``coach`` (kept apart from city buses), 400–404 and 406–499 ``subway``, 405 ``monorail``, 700–716
``bus``, 800–899 ``trolleybus``, 900–906 ``tram``, 1000 and 1200 ``ferry``, 1300 ``aerial_lift``, 1400
``funicular`` and 1500 ``taxi``. Any other code, including the miscellaneous ranges 300, 500, 600, 1600 and 1700,
stays ``public_transport``, with no guess from neighbouring codes.

Method
------

Travel times
~~~~~~~~~~~~

Trips with the same route, direction, shape, stop order and pickup and drop-off rules form a route pattern. For each
segment of a pattern, ``time_min`` is the median scheduled time from the departure at one stop to the arrival at
the next.

Waiting times
~~~~~~~~~~~~~

Boarding edges carry the expected wait:

- scheduled service waits half of the mean gap between consecutive departures of the *line* at the stop on the
  same day, averaged over days as described above. A line is the route leaving the stop for the same next stop: all
  its patterns count (branches on their shared section, short workings, expresses), so the wait is that of the line
  a rider waits for. The next stop decides the direction, not ``direction_id``;
- ``frequencies.txt`` service waits ``headway_secs / 120`` minutes, averaged by duration when frequency periods
  overlap the window;
- a pattern stop with a single departure has no gap to estimate, so its boarding edge is left out by default;
  ``single_departure_wait_min`` keeps it with the given wait.

Boarding and alighting edges have ``length_meter = 0``, and alighting also ``time_min = 0``. GTFS waits come from
the feed and do not use the averages of :class:`TransportRegistry`.

Geometry and distance
~~~~~~~~~~~~~~~~~~~~~

When ``shapes.txt`` gives a usable route geometry, the segment geometry is cut from the shape and ``length_meter``
is its measured length. Otherwise the segment is a straight line between consecutive stops and ``length_meter`` is
``sqrt(2)`` times its length, an estimate of the distance along the streets. The correction applies to travel edges
(``bus``, ``tram``, ``subway`` and so on), never to boarding or alighting.

The graph CRS must be projected in metres. When ``crs`` is omitted, a local UTM CRS is estimated from the stops.

Stations and pathways
~~~~~~~~~~~~~~~~~~~~~

``pathways.txt`` adds ``pathway`` edges inside stations. An explicit ``traversal_time`` is converted from seconds to
minutes; otherwise the time is the pathway length over ``walk_speed_m_per_min``. Station entrances, generic nodes,
boarding areas and platforms are kept when the served stops or the pathways refer to them.

A stop that names a ``parent_station`` is linked to it by a ``station_link`` edge of zero cost in both directions.
Only stops already in the graph are linked. Without these edges a station described down to its boarding areas
would leave its platforms in a separate component, which ``keep_largest_subgraph`` removes.

Merging several feeds
~~~~~~~~~~~~~~~~~~~~~

A city is often published as several feeds, and every feed numbers its own routes, trips and stops. A list of
sources is merged with :func:`merge_gtfs_feeds` before the graph is built:

.. code-block:: python

   public_transport = get_gtfs_public_transport_graph(
       ["subway.zip", "bus.zip"],
       service_date="2026-08-04",
       start_time="07:00:00",
       end_time="10:00:00",
   )

:func:`merge_gtfs_feeds` qualifies every identifier by its source, so route ``1`` of one feed and route ``1`` of
another stay distinct. Call it directly to choose the prefixes or to fuse the stops that different agencies publish
for the same place:

.. code-block:: python

   from iduedu import get_gtfs_public_transport_graph, merge_gtfs_feeds

   feed = merge_gtfs_feeds(["subway.zip", "bus.zip"], merge_stops_within=20)
   public_transport = get_gtfs_public_transport_graph(feed, service_date="2026-08-04")

Stops stay distinct by default: fusing them changes the network, and an inferred transfer would look like a
published one. Distances of 15 to 25 m suit feeds from different agencies.

Feeds gathered from several catalogues often overlap: two versions of one feed, a national and a city feed of the
same operator, a mirror and its original. ``deduplicate=True`` counts that service once. Platforms of different
sources closer than ``duplicate_platform_tolerance`` (0.5 m by default, the same coordinate published again, while
distinct platforms of one feed already occur 1 to 3 m apart) become one platform, and a trip that another source
already runs through the same platforms at the same times is dropped. Platforms of one feed are never fused with
each other, and neither are stations, entrances or boarding areas.

.. code-block:: python

   feed = merge_gtfs_feeds(["city.zip", "national.zip"], deduplicate=True)

:func:`deduplicate_across_feeds` applies the same rules to tables merged elsewhere and returns how many platforms
and trips it removed.

Outputs
-------

A directed :class:`UrbanGraph` with the node and edge types of the OSM public-transport graph (:doc:`public_transport`):
platforms, route stops, travel edges by mode, ``boarding`` and ``alighting`` edges, and for stations with pathways
``pathway`` and ``station_link`` edges. To walk to the stops, build the walk graph first and give its CRS to the GTFS
builder:

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

Functions
---------

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    get_gtfs_public_transport_graph
    merge_gtfs_feeds
    deduplicate_across_feeds

.. currentmodule:: iduedu.gtfs

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    read_gtfs_feed
    validate_gtfs_feed
    GTFSFeed
    GTFSValidationError

Limitations
-----------

- The graph is static: a wait is an average over the selected days and hours, and routes are found without the
  times of individual departures. ``transfers.txt`` and timed transfers are not used.
- Segments without ``shapes.txt`` get an estimated length and a straight geometry.
- Real-time feeds (GTFS Realtime) are not read.

Example notebook
----------------

:doc:`../examples/gtfs_public_transport`
