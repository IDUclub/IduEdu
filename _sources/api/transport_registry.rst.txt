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
      - ``avg_wait_time_min``
      - ``dwell_min``
    * - ``bus``
      - 41.0
      - 8.2
      - 0.475
    * - ``trolleybus``
      - 20.0
      - 5.92
      - 0.325
    * - ``tram``
      - 41.5
      - 4.95
      - 1.2
    * - ``subway``
      - 49.0
      - 3.02
      - 0.35
    * - ``monorail``
      - 49.0
      - 3.0
      - 0.35
    * - ``taxi``
      - 41.0
      - 8.0
      - 0.475
    * - ``train`` (``DEFAULT_REGISTRY_W_TRAIN`` only)
      - 50.5
      - 7.71
      - 0.375

Speeds and dwell times are fitted against published GTFS timetables matched to OSM routes. Waiting
times are medians across cities of the timetable wait over regular services. No city in the
calibration sample publishes a timetable for monorail or share taxi, so these two borrow the subway
and bus profiles.

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
- ``avg_wait_time_min`` – average waiting time assigned to boarding edges in OSM-based graphs.

Travel time
~~~~~~~~~~~

For a segment of length ``L``, the cruising speed ``v`` is ``base_speed_kmh``, capped by
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
