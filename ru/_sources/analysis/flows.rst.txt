Flow assignment
===============

.. currentmodule:: iduedu

Overview
--------

:func:`assign_flows` loads a network with trips: every group of sources sends its flow to the nodes it reaches,
in proportion to their attraction and to a decay of the travel cost, and the trips run along the shortest paths.
The result is the flow on every edge.

.. code-block:: python

   loads = iduedu.assign_flows(
       drive,
       source_nodes=entrances,          # graph node ids, grouped by building
       source_flows=trips,              # trips from every entrance
       cluster_offsets=offsets,         # building b owns entrances offsets[b]:offsets[b + 1]
       node_weights="jobs",             # attraction of the destinations
       mean=20.0, std=10.0, cutoff=60.0,
   )
   drive.edges_gdf["flow"] = loads["flow"]

Inputs
------

- ``source_nodes`` and ``source_flows``: the sources as graph node ids and the flow each of them sends.
- ``cluster_offsets``: groups of sources that share one search, such as the entrances of one building. By default
  every source is a group of its own.
- ``node_weights``: attraction of every node (a ``nodes_gdf`` column, a Series by node id or an array).
- ``decay`` with its parameters, ``cutoff`` and ``weight`` (``time_min`` by default): how the attraction fades with
  the travel cost and the largest cost of a trip.
- ``min_flow``: the flow of one group below which an arc is ignored.

Method
------

Every group runs one search from all of its sources at once, cut off at ``cutoff``. A node joins its nearest
source; when two sources are equally near, the one that comes first in ``nodes_gdf`` wins. Every reached node other than a source is a
destination with the weight ``node_weights[node] · decay(cost)``:

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - ``decay``
     - Weight
     - Meaning
   * - ``"gaussian"``
     - ``exp(-0.5 · ((cost - mean) / std)²)``
     - Trips of a typical cost ``mean`` with the spread ``std``
   * - ``"exp"``
     - ``exp(-beta · cost)``
     - Nearer destinations are more attractive
   * - ``"flat"``
     - 1
     - Every destination within ``cutoff`` counts by its attraction only

The summed flow of the group's sources is split between its destinations in proportion to these weights, which
is the distribution step of a gravity model (Ortúzar and Willumsen, 2011). The flow of every destination then runs
back along the search tree to its source, from the farthest nodes to the nearest, so every arc carries the trips
that pass it: an all-or-nothing assignment. An arc keeps the flow of a group only when it exceeds ``min_flow``, or
when the arc leaves a source and the flow is positive, so the far tails of small groups do not add noise. The kept
flows of all groups are summed. Groups run in parallel. Costs are summed in float32 and flows in float64.

Congestion is not part of one call. Equilibrium methods repeat the assignment with edge times updated from the
flows, for example by the BPR function and the method of successive averages (Sheffi, 1985).

Outputs
-------

A table indexed like ``edges_gdf`` with ``flow_uv`` (along the edge from ``u`` to ``v``), ``flow_vu`` (against it,
for edges travelled both ways) and their sum ``flow``. Among parallel edges between two nodes only the cheapest one
carries flow, and equally cheap ones share it. ``per="arcs"`` returns the flows of the arcs of
``adjacency_matrix`` instead, one value per entry of its ``data``.

On the intermodal graph of St Petersburg (422 000 nodes, 1.1 million arcs) on 16 threads, 4 000 sources in 800
groups with ``cutoff=45`` minutes take 3.8 s.

Functions
---------

.. autosummary::
    :toctree: ../api/generated
    :nosignatures:

    assign_flows

Limitations
-----------

- One call is an all-or-nothing assignment: every trip takes one shortest path, and the edge costs do not depend
  on the flow.
- Within a group the destinations compete only through their weights: the flows do not depend on how many other
  groups reach the same destination (no balancing of destinations).
- A node reached from no source of a group receives nothing from it, also when it lies just beyond ``cutoff``.

References
----------

- Ortúzar, J. de D. and Willumsen, L. G. (2011). *Modelling Transport*, 4th edition. Wiley.
- Sheffi, Y. (1985). *Urban Transportation Networks: Equilibrium Analysis with Mathematical Programming Methods*.
  Prentice-Hall.
