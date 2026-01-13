Transport registry
==================

The transport registry defines how different public-transport modes are represented and how
travel time is computed on graph edges.

It is used by all public-transport graph builders to validate transport types and to estimate
per-edge travel time based on segment length, speed limits, and mode-specific parameters.

Overview
--------

A transport registry is an instance of :class:`iduedu.TransportRegistry` that stores one or more
transport specifications (:class:`iduedu.TransportSpec`).

Each transport specification describes:

- the transport mode identifier (e.g. ``"bus"``, ``"tram"``, ``"subway"``);
- technical maximum speed;
- typical acceleration and braking distances;
- a traffic slowdown coefficient.

The registry is consulted during graph construction to compute the ``time_min`` attribute
for each edge.

Default registry
----------------

The library provides a predefined registry:

.. code-block:: python

    from iduedu import DEFAULT_REGISTRY

The default registry includes common public-transport modes such as buses, trams, trolleybuses,
subways, and trains. These defaults are suitable for most use cases and require no configuration.

If no registry is explicitly provided, all public-transport graph builders automatically fall back
to ``DEFAULT_REGISTRY``.

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
        traffic_coef=0.75,
    )

The parameters have the following meaning:

- ``name`` – transport type identifier, usually matching the OSM ``route=*`` value;
- ``vmax_tech_kmh`` – technical maximum speed in kilometers per hour;
- ``accel_dist_m`` – typical distance required to accelerate to cruising speed (meters);
- ``brake_dist_m`` – typical distance required to decelerate from cruising speed (meters);
- ``traffic_coef`` – traffic slowdown coefficient (values below 1.0 reduce effective speed).

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
            traffic_coef=0.7,
        )
    )

    registry.add(
        TransportSpec(
            name="tram",
            vmax_tech_kmh=70,
            accel_dist_m=180,
            brake_dist_m=110,
            traffic_coef=0.85,
        )
    )

All transport type names are normalized internally (lowercase, stripped).

Updating and extending the registry
-----------------------------------

Existing transport specifications can be updated:

.. code-block:: python

    registry.update("bus", traffic_coef=0.6)
    registry.update("tram", vmax_tech_kmh=75)

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

All public-transport graph builders accept a registry via the ``transport_registry`` parameter.

For example:

.. code-block:: python

    from iduedu import get_public_transport_graph

    graph = get_public_transport_graph(
        osm_id=123456,
        transport_types=["bus", "tram"],
        transport_registry=registry,
    )

If ``transport_registry`` is not provided, ``DEFAULT_REGISTRY`` is used automatically.

The registry controls:

- which transport types are considered valid;
- how per-edge travel time (``time_min``) is computed;
- how short segments are handled (acceleration and braking effects).

API reference
-------------

.. currentmodule:: iduedu

.. autosummary::
    :toctree: generated
    :nosignatures:

    TransportSpec
    TransportRegistry

