"""Public-transport routes of a graph: the ``routes`` table of :class:`UrbanGraph` and the queries over it.

A route is one variant of a line, as OSM maps it (a ``type=route`` relation) or as a GTFS feed runs it (the
trips of a ``route_id`` in one direction calling at the same stops). The table holds its attributes only: the
route's own stop nodes and the edges between them, which carry its ``route_key``, give its geometry, the order
of its stops, its length and its travel time.
"""

from collections.abc import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

#: Name of the route key: the index of the ``routes`` table and the column of the edges and nodes that use it.
ROUTE_KEY = "route_key"

#: Columns every ``routes`` table has, in this order; a builder may add more after them.
ROUTE_COLUMNS = (
    "source",
    "mode",
    "ref",
    "name",
    "from",
    "to",
    "network",
    "operator",
    "colour",
    "line_key",
    "line_name",
)

#: Edge types that link a stop to its platform; the other edges with a route key are its travel edges.
_LINK_TYPES = frozenset({"boarding", "alighting"})


def empty_routes() -> pd.DataFrame:
    """A ``routes`` table without rows."""
    return pd.DataFrame(columns=list(ROUTE_COLUMNS), index=pd.Index([], name=ROUTE_KEY, dtype=object))


def normalize_routes(routes: pd.DataFrame | None) -> pd.DataFrame | None:
    """Check a ``routes`` table and give it its index name and leading columns.

    Raises:
        TypeError: If ``routes`` is not a DataFrame.
        ValueError: If its route keys are missing or repeated.
    """
    if routes is None:
        return None
    if not isinstance(routes, pd.DataFrame) or isinstance(routes, gpd.GeoDataFrame):
        raise TypeError(f"routes must be a pandas DataFrame, got {type(routes).__name__}")
    if ROUTE_KEY in routes.columns:
        routes = routes.set_index(ROUTE_KEY)
    if routes.index.hasnans:
        raise ValueError("routes has rows without a route key")
    if not routes.index.is_unique:
        repeated = routes.index[routes.index.duplicated()].unique().tolist()[:5]
        raise ValueError(f"routes repeats route keys, e.g. {repeated}")
    routes = routes.copy()
    routes.index.name = ROUTE_KEY
    for column in ROUTE_COLUMNS:
        if column not in routes.columns:
            routes[column] = None
    rest = [column for column in routes.columns if column not in ROUTE_COLUMNS]
    return routes[list(ROUTE_COLUMNS) + rest]


def used_routes(routes: pd.DataFrame | None, edges: pd.DataFrame) -> pd.DataFrame | None:
    """The rows of ``routes`` that some edge still refers to; all of them when the edges have no route keys."""
    if routes is None or ROUTE_KEY not in edges.columns:
        return routes
    keys = pd.unique(edges[ROUTE_KEY].dropna().to_numpy())
    if len(keys) == len(routes) and routes.index.isin(keys).all():
        return routes
    return routes.loc[routes.index.isin(keys)]


def join_routes(*tables: pd.DataFrame | None) -> pd.DataFrame | None:
    """One ``routes`` table from several; a key in more than one keeps its first row."""
    present = [table for table in tables if table is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    joined = pd.concat(present, axis=0, sort=False)
    return joined[~joined.index.duplicated(keep="first")]


def _require_routes(graph) -> pd.DataFrame:
    if graph.routes is None:
        raise ValueError("the graph has no routes table; build it with a public-transport builder")
    return graph.routes


def _travel_edges(graph, route_key: str) -> pd.DataFrame:
    edges = graph.edges_gdf
    if ROUTE_KEY not in edges.columns:
        raise ValueError(f"the graph edges have no {ROUTE_KEY!r} column")
    if route_key not in _require_routes(graph).index:
        raise KeyError(f"route {route_key!r} is not in the routes table")
    mine = edges[edges[ROUTE_KEY].eq(route_key)]
    if "type" in mine.columns:
        mine = mine[~mine["type"].astype(str).isin(_LINK_TYPES)]
    return mine


def _ordered(travel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of the travel edges in the order of travel, and the part of the route of each.

    A part starts at a stop no edge of the route arrives at; a route cut by a hole in its mapping, or clipped
    by a boundary, has several. From a stop the route leaves more than once (a loop through it), the first
    unused edge in table order is taken. What is left after the parts is closed rings, started at their first
    edge in table order.
    """
    u = travel["u"].to_numpy()
    v = travel["v"].to_numpy()
    outgoing: dict = {}
    for position, tail in enumerate(u):
        outgoing.setdefault(tail, []).append(position)
    arriving = set(v.tolist())
    used = np.zeros(len(travel), dtype=bool)
    order: list[int] = []
    parts: list[int] = []

    def follow(position: int, part: int) -> None:
        while position is not None:
            used[position] = True
            order.append(position)
            parts.append(part)
            nxt = None
            for candidate in outgoing.get(v[position], ()):
                if not used[candidate]:
                    nxt = candidate
                    break
            position = nxt

    part = 0
    for position in range(len(travel)):
        if not used[position] and u[position] not in arriving:
            follow(position, part)
            part += 1
    for position in range(len(travel)):
        if not used[position]:
            follow(position, part)
            part += 1
    return np.asarray(order, dtype=np.int64), np.asarray(parts, dtype=np.int64)


def route_edges(graph, route_key: str) -> gpd.GeoDataFrame:
    """Travel edges of one route in the order of travel.

    Args:
        graph: Graph with a ``routes`` table.
        route_key: Key of the route.

    Returns:
        Its edges between stops (no boarding or alighting edges), with a ``position`` column (0, 1, ... along
        the route) and a ``part`` column (0 unless the route is broken into several pieces, by a gap in its data
        or by clipping).

    Raises:
        ValueError: If the graph has no routes table or its edges no ``route_key`` column.
        KeyError: If the route is not in the table.
    """
    travel = _travel_edges(graph, route_key)
    order, parts = _ordered(travel)
    result = travel.iloc[order].copy()
    result["position"] = np.arange(len(result), dtype=np.int64)
    result["part"] = parts
    return result


def route_stops(graph, route_key: str) -> gpd.GeoDataFrame:
    """Stop nodes of one route in the order the route calls at them.

    Args:
        graph: Graph with a ``routes`` table.
        route_key: Key of the route.

    Returns:
        Rows of ``nodes_gdf`` (a stop visited twice appears twice) with ``position`` and ``part`` columns, as
        :func:`route_edges` numbers them.

    Raises:
        ValueError: If the graph has no routes table or its edges no ``route_key`` column.
        KeyError: If the route is not in the table.
    """
    edges = route_edges(graph, route_key)
    nodes: list = []
    parts: list[int] = []
    previous_part = None
    previous_head = None
    for tail, head, part in zip(edges["u"], edges["v"], edges["part"]):
        if part != previous_part or tail != previous_head:
            nodes.append(tail)
            parts.append(int(part))
        nodes.append(head)
        parts.append(int(part))
        previous_part, previous_head = part, head
    if not nodes and route_key in graph.nodes_gdf.get(ROUTE_KEY, pd.Series(dtype=object)).to_numpy():
        lone = graph.nodes_gdf.index[graph.nodes_gdf[ROUTE_KEY].eq(route_key)]
        nodes, parts = list(lone), [0] * len(lone)
    result = graph.nodes_gdf.loc[nodes].copy()
    result["position"] = np.arange(len(result), dtype=np.int64)
    result["part"] = np.asarray(parts, dtype=np.int64)
    return result


def routes_in_area(graph, area: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame) -> pd.DataFrame:
    """Routes of the graph with a travel edge or a stop in an area.

    Args:
        graph: Graph with a ``routes`` table.
        area: A shapely geometry in EPSG:4326 (longitude, latitude), or a GeoSeries or GeoDataFrame with
            its own CRS, as for :func:`clip_urban_graph`; several geometries are united.

    Returns:
        The rows of ``graph.routes`` that have an edge or a stop node intersecting the area.

    Raises:
        ValueError: If the graph has no routes table.
    """
    # Imported here: graph_inputs imports the graph package at import time.
    from iduedu.graph.graph_inputs import area_in_crs  # pylint: disable=import-outside-toplevel

    routes = _require_routes(graph)
    area = area_in_crs(area, graph.crs, "area")
    keys: set = set()
    edges = graph.edges_gdf
    if ROUTE_KEY in edges.columns and len(edges):
        with_key = edges[edges[ROUTE_KEY].notna()]
        hit = with_key.geometry.intersects(area).to_numpy()
        keys.update(with_key.loc[hit, ROUTE_KEY].tolist())
    nodes = graph.nodes_gdf
    if ROUTE_KEY in nodes.columns and len(nodes):
        stops = nodes[nodes[ROUTE_KEY].map(lambda key: isinstance(key, str))]
        hit = stops.geometry.intersects(area).to_numpy()
        keys.update(stops.loc[hit, ROUTE_KEY].tolist())
    return routes.loc[routes.index.isin(keys)]


def route_keys_of(values: Iterable) -> list[str]:
    """Route keys in first-seen order, without missing values and repeats."""
    return list(dict.fromkeys(value for value in values if isinstance(value, str)))
