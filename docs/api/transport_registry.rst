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
- a traffic slowdown coefficient;
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

The default registry includes buses, trams, trolleybuses, and subways. Their OSM boarding waits are
8, 6, 8, and 2 minutes respectively. ``DEFAULT_REGISTRY_W_TRAIN`` additionally includes trains with a
1-minute default wait.

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

The OSM :func:`get_public_transport_graph` builder accepts a registry via the
``transport_registry`` parameter. :func:`get_intermodal_graph` forwards it through ``pt_kwargs``.

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
