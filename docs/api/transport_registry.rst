Transport registry
==================

The transport registry defines how different public-transport modes are represented and how
travel time is computed on graph edges.

It is used by the OSM public-transport builder to validate transport types and to estimate per-edge
travel time based on segment length, speed limits, and mode-specific parameters. GTFS graph weights
are derived from the feed schedule and do not use the registry.

Overview
--------

A transport registry is an instance of :class:`iduedu.TransportRegistry` that stores one or more
transport specifications (:class:`iduedu.TransportSpec`).

Each transport specification describes:

- the transport mode identifier (e.g. ``"bus"``, ``"tram"``, ``"subway"``);
- technical maximum speed;
- typical acceleration and braking distances;
- free-flow speed between stops;
- time lost standing at a stop;
- average passenger waiting time before boarding.

The registry is consulted during graph construction to compute the ``time_min`` attribute
for each edge.

See :doc:`../examples/transport_registry` for a runnable example that inspects
the default registries, creates a custom registry, updates mode parameters, and
passes the registry into public-transport graph construction.

Default registry
----------------

The library provides a predefined registry:

.. code-block:: python

    from iduedu import DEFAULT_REGISTRY

The default registry includes buses, trolleybuses, trams, subways, monorails, and share taxis
(``taxi``). ``DEFAULT_REGISTRY_W_TRAIN`` additionally includes trains.

.. list-table::
    :header-rows: 1

    * - Mode
      - ``base_speed_kmh``
      - ``long_speed_kmh`` / ``speed_half_length_m``
      - ``avg_wait_time_min``
      - ``dwell_min``
    * - ``bus``
      - 29.5
      - --
      - 12.24
      - 0.387
    * - ``trolleybus``
      - 20.0
      - --
      - 8.02
      - 0.159
    * - ``tram``
      - 11.5
      - 74.1 / 3506
      - 5.22
      - 0.0
    * - ``subway``
      - 56.7
      - --
      - 3.26
      - 0.694
    * - ``monorail``
      - 56.7
      - --
      - 3.26
      - 0.694
    * - ``taxi``
      - 29.5
      - --
      - 12.24
      - 0.387
    * - ``train`` (``DEFAULT_REGISTRY_W_TRAIN`` only)
      - 33.5
      - 78.9 / 5270
      - 21.5
      - 0.365

Speeds and dwell times are fitted on every published GTFS segment that has its own
geometry (``shapes.txt``) -- about 859,000 segments in 239 cities -- minimising the error in
scheduled minutes, every city weighing the same, and cross-validated on held-out cities. A
free speed growing with the segment length is used for tram and train, where it carries to
unseen cities; bus, subway and trolleybus keep one free speed, as the growing form does not
improve them. Waiting times are medians across cities of the average-day timetable wait over
regular services (06:00-22:00, a line counted towards the same next stop). Too few cities
publish a timetable for monorail or share taxi, so these two borrow the subway and bus
profiles.

OpenStreetMap tags some services under names the registry does not use. A request for ``tram`` also
fetches ``route=light_rail`` relations, and ``taxi`` fetches ``route=share_taxi``. The mapping is
``OSM_ROUTE_ALIASES`` in ``iduedu.constants.transport_specs``.

If no registry is explicitly provided, the OSM public-transport builder automatically falls back to
``DEFAULT_REGISTRY``.

TransportSpec
-------------

A single transport mode is described by :class:`iduedu.TransportSpec`.

.. code-block:: python

    from iduedu import TransportSpec

    bus = TransportSpec(
        name="bus",
        vmax_tech_kmh=90,
        accel_dist_m=220,
        brake_dist_m=140,
        base_speed_kmh=37.0,
        dwell_min=0.45,
        avg_wait_time_min=8.0,
    )

The parameters have the following meaning:

- ``name`` – transport type identifier, usually matching the OSM ``route=*`` value;
- ``vmax_tech_kmh`` – technical maximum speed in kilometers per hour;
- ``accel_dist_m`` – typical distance required to accelerate to cruising speed (meters);
- ``brake_dist_m`` – typical distance required to decelerate from cruising speed (meters);
- ``base_speed_kmh`` – free-flow speed of the mode between stops. A posted road limit
  does **not** scale it; the limit is honoured only where it is lower, so a motorway
  does not make a bus faster. Fitting free speed as ``base + coef * limit`` against
  published timetables showed the two terms are not separately identifiable, and
  dropping the limit term improved accuracy for every mode measured;
- ``dwell_min`` – time lost standing at a stop, added once per segment;
- ``avg_wait_time_min`` – average waiting time assigned to boarding edges in OSM-based graphs;
- ``long_speed_kmh`` and ``speed_half_length_m`` – optional, set together. The free speed then
  grows with the segment length ``L`` from ``base_speed_kmh`` towards ``long_speed_kmh``:
  ``v(L) = base_speed_kmh + (long_speed_kmh - base_speed_kmh) * L / (L + speed_half_length_m)``,
  halfway at ``L = speed_half_length_m``. Timetables of every mode show it: a long stretch
  between stops runs on roads and tracks built for speed. Left at ``None`` (default), one free
  speed applies to every segment.

Travel time
~~~~~~~~~~~

For a segment of length ``L``, the cruising speed ``v`` is the free speed (``base_speed_kmh``, or
``v(L)`` above when ``long_speed_kmh`` is set), capped by
``vmax_tech_kmh`` and by the road speed limit where that is lower, and
``span = accel_dist_m + brake_dist_m``:

- if ``L >= span``, the time is ``dwell_min + (L + span) / v``: accelerating and braking cost as much
  as covering ``span`` twice at cruising speed;
- if ``L < span``, the vehicle never reaches ``v``. Its peak speed is ``v * sqrt(L / span)`` and the
  time is ``dwell_min + 2 * L / v_peak``. Both cases give the same time at ``L == span``.

Creating a custom registry
--------------------------

You can create your own registry and fully control how travel time is computed.

.. code-block:: python

    from iduedu import TransportRegistry, TransportSpec

    registry = TransportRegistry()

    registry.add(
        TransportSpec(
            name="bus",
            vmax_tech_kmh=80,
            accel_dist_m=200,
            brake_dist_m=120,
            base_speed_kmh=34.0,
            avg_wait_time_min=8.0,
        )
    )

    registry.add(
        TransportSpec(
            name="tram",
            vmax_tech_kmh=70,
            accel_dist_m=180,
            brake_dist_m=110,
            base_speed_kmh=40.0,
            avg_wait_time_min=6.0,
        )
    )

All transport type names are normalized internally (lowercase, stripped).

Updating and extending the registry
-----------------------------------

Existing transport specifications can be updated:

.. code-block:: python

    registry.update("bus", base_speed_kmh=32.0)
    registry.update("bus", long_speed_kmh=40.0, speed_half_length_m=5000.0)  # faster on long stretches
    registry.update("tram", vmax_tech_kmh=75)
    registry.update("bus", avg_wait_time_min=5.0)

Transport types can also be renamed:

.. code-block:: python

    registry.update("bus", name="express_bus")

If a transport type is encountered during parsing but is missing from the registry,
it can be created automatically using ``ensure``:

.. code-block:: python

    spec = registry.ensure("ferry")

This is useful when working with less common OSM transport modes.

Using the registry in graph builders
------------------------------------

The OSM :func:`iduedu.get_public_transport_graph` builder accepts a registry via the
``transport_registry`` parameter. :func:`iduedu.get_intermodal_graph` forwards it through ``pt_kwargs``.

For example:

.. code-block:: python

    from iduedu import get_public_transport_graph

    graph = get_public_transport_graph(
        osm_id=123456,
        transport_types=["bus", "tram"],
        transport_registry=registry,
    )

If ``transport_registry`` is not provided, ``DEFAULT_REGISTRY`` is used automatically.

The former ``avg_boarding_time_min`` builder argument has been removed. To apply one waiting-time
value to selected modes, create a registry and update those specifications instead:

.. code-block:: python

    from iduedu import DEFAULT_REGISTRY, TransportRegistry

    registry = TransportRegistry({mode: DEFAULT_REGISTRY.get(mode) for mode in DEFAULT_REGISTRY.list_types()})
    for mode in registry.list_types():
        registry.update(mode, avg_wait_time_min=1.0)

    graph = get_public_transport_graph(
        osm_id=123456,
        transport_registry=registry,
    )

For OSM-based public-transport graphs, the registry controls:

- which transport types are considered valid;
- how per-edge travel time (``time_min``) is computed;
- how short segments are handled (acceleration and braking effects);
- the mode-specific ``time_min`` of boarding edges.

GTFS graph boarding weights remain timetable-derived and do not use this fallback.

API reference
-------------

.. currentmodule:: iduedu

.. autosummary::
    :toctree: generated
    :nosignatures:

    TransportSpec
    TransportRegistry
