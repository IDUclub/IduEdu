"""Network statistics of an UrbanGraph: for the whole graph, per zone and around every node.

All measures are taken on the *street* view of the graph: an undirected street is one edge between two
nodes however many directed or parallel copies of it the graph stores, and lengths are measured on the
unique segments of the geometry (:func:`~iduedu.unique_segments`), so the two directions of a street count
once.
"""

from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy import sparse
from scipy.sparse import csgraph
from shapely.geometry.base import BaseGeometry

from iduedu.constants.transport_specs import DEFAULT_REGISTRY_W_TRAIN
from iduedu.graph.graph_inputs import area_in_crs, geometries_in_crs, resolve_graph_nodes_input
from iduedu.graph.routing import _positions, _prepare_graph
from iduedu.graph.segments import segment_arrays
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.graph.within import band_label, cost_bands, sum_within, value_rows

ORIENTATION_BINS = 36


def _projected(urban_graph: UrbanGraph, what: str):
    if not isinstance(urban_graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(urban_graph).__name__}")
    crs = getattr(urban_graph.edges_gdf, "crs", None) or getattr(urban_graph.nodes_gdf, "crs", None) or urban_graph.crs
    if crs is not None and getattr(crs, "is_geographic", False):
        raise ValueError(f"{what} needs a projected CRS; reproject the graph first")
    return crs


def _zone_of(xy: np.ndarray, zones: np.ndarray) -> np.ndarray:
    """Position of the first zone that holds each point, ``-1`` for points in no zone."""
    result = np.full(len(xy), -1, dtype=np.int64)
    if len(xy) and len(zones):
        points, found = shapely.STRtree(zones).query(shapely.points(xy), predicate="intersects")
        result[points[::-1]] = found[::-1]
    return result


class _Streets:  # pylint: disable=too-many-instance-attributes
    """Street view of a graph: unique undirected streets and the number of streets at every node."""

    def __init__(self, urban_graph: UrbanGraph, edge_types: Iterable[str] | None):
        nodes, edges = urban_graph.nodes_gdf, urban_graph.edges_gdf
        if edge_types is not None:
            if "type" not in edges.columns:
                raise KeyError("edge_types needs a 'type' column in edges_gdf")
            edges = edges[edges["type"].isin(list(edge_types))]
        self.edges = edges
        self.n = len(nodes)
        self.xy = np.column_stack([nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()]).reshape(-1, 2)
        index = pd.Index(nodes.index)
        u, v = index.get_indexer(edges["u"]), index.get_indexer(edges["v"])
        if (u < 0).any() or (v < 0).any():
            raise ValueError("edges_gdf refers to nodes that are absent in nodes_gdf")
        self.length = shapely.length(edges.geometry.values) if len(edges) else np.zeros(0)
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        # One street per unordered node pair and length: the two directions of a street and exact copies merge.
        pair = pd.DataFrame({"lo": lo, "hi": hi, "length": np.round(self.length, 1)})
        self.first = ~pair.duplicated().to_numpy()
        self.lo, self.hi = lo[self.first], hi[self.first]
        self.street_length = self.length[self.first]
        self.straight = np.hypot(*(self.xy[self.lo] - self.xy[self.hi]).T) if len(self.lo) else np.zeros(0)
        self.degree = np.bincount(self.lo, minlength=self.n) + np.bincount(self.hi, minlength=self.n)
        self.active = self.degree > 0 if edge_types is not None else np.ones(self.n, dtype=bool)
        pairs = pd.DataFrame({"lo": self.lo, "hi": self.hi})
        self.parallel = pairs.duplicated(keep=False).to_numpy() & (self.lo != self.hi)

    def midpoints(self) -> np.ndarray:
        """Midpoint of every edge geometry (all edges, not only the unique streets)."""
        if not len(self.edges):
            return np.zeros((0, 2))
        return shapely.get_coordinates(shapely.line_interpolate_point(self.edges.geometry.values, 0.5, normalized=True))


def streets_per_node(urban_graph: UrbanGraph, *, edge_types: Iterable[str] | None = None) -> pd.Series:
    """Number of streets that meet at every node.

    A street is counted once whatever its direction and however many copies of it the graph stores; a loop
    counts twice. One street marks a dead end, three or more an intersection.

    Args:
        urban_graph: The graph.
        edge_types: Count only edges whose ``type`` is one of these (for example ``["walk"]`` in an
            intermodal graph).

    Returns:
        Series indexed like ``nodes_gdf``, named ``streets``.
    """
    if not isinstance(urban_graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(urban_graph).__name__}")
    streets = _Streets(urban_graph, edge_types)
    return pd.Series(streets.degree, index=urban_graph.nodes_gdf.index, name="streets")


def _entropy(histogram: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of each row of a histogram."""
    total = histogram.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0, histogram / total, 0.0)
        terms = np.where(share > 0, share * np.log(share), 0.0)
    return np.where(total[:, 0] > 0, -terms.sum(axis=1), np.nan)


def _ratio(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


def graph_stats(  # pylint: disable=too-many-locals, too-many-statements
    urban_graph: UrbanGraph,
    *,
    area: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    zones: gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    edge_types: Iterable[str] | None = None,
    boundary_buffer: float | None = 1.0,
    precision: float = 0.01,
) -> pd.Series | pd.DataFrame:
    """Measures of size, density, shape and connectivity of a street network.

    Without ``area`` and ``zones`` the whole graph is measured. With ``area`` only the part of the graph
    inside it is measured, and with ``zones`` every zone separately. In both cases the number of streets at
    a node is taken from the whole graph, so a graph built with a margin around the study area gives dead
    ends and intersections free of edge effects. Nodes belong to the zone they lie in, streets and segments
    to the zone of their midpoint.

    Measures (``NaN`` where a measure is undefined):

    - ``node_count``, ``edge_count`` (edges as stored), ``street_count`` (undirected streets);
    - ``street_length_total`` (metres, every physical segment once), ``street_length_mean``;
    - ``streets_per_node_mean``, ``dead_end_count``, ``dead_end_share``, ``intersection_count`` (nodes with
      three or more streets), ``intersection_4way_share`` (four or more streets, among intersections);
    - ``circuity_mean`` (street length over the straight distance between its ends, loops left out),
      ``self_loop_share``, ``parallel_street_share``;
    - ``component_count``, ``largest_component_share`` (of nodes);
    - ``beta_index`` (streets per node), ``gamma_index`` (streets over the most a planar graph can have,
      ``3 (v - 2)``), ``cyclomatic_number`` (independent cycles, ``e - v + p``), ``meshedness`` (cycles over
      the most a planar graph can have, ``2 v - 5``); they count streets with both ends inside;
    - ``orientation_entropy`` (Shannon entropy in nats of segment bearings in 36 bins, weighted by length)
      and ``orientation_order`` (1 for a perfect grid, 0 for uniform bearings);
    - ``oneway_share`` (of street length) when the graph has a direction column;
    - ``speed_mean_kmh`` (total length over total travel time) when edges have ``length_meter`` and
      ``time_min``;
    - ``elevation_min``, ``elevation_max``, ``elevation_mean`` (nodes), ``grade_mean`` (mean absolute grade
      weighted by length) and ``grade_max`` when the graph carries elevation
      (:func:`~iduedu.add_elevation`);
    - ``platform_count`` (nodes of type ``platform``), ``unattached_platform_count`` (platforms no walking
      edge leads to, in graphs that have walking edges) and ``boarding_edge_count`` for graphs with
      public-transport platforms;
    - with ``area`` or ``zones``: ``area_km2``, ``node_density``, ``intersection_density`` (per km²) and
      ``street_density`` (km per km²).

    Args:
        urban_graph: Graph in a projected (metric) CRS.
        area: Study area: a shapely geometry in EPSG:4326, as ``territory`` of the graph builders, or a
            GeoSeries / GeoDataFrame in any CRS. Several geometries are merged into one area.
        zones: Zones to measure separately: a GeoSeries or GeoDataFrame in any CRS. The result has one row
            per zone, indexed like ``zones``.
        edge_types: Measure only edges whose ``type`` is one of these, and the nodes they touch.
        boundary_buffer: With ``area``: nodes with one street closer than this to the area boundary are not
            counted as dead ends (streets cut by clipping the graph to the area). ``None`` counts them.
        precision: Grid for comparing segment end points, see :func:`~iduedu.unique_segments`.

    Returns:
        Series of measures, or with ``zones`` a DataFrame with one row per zone.

    Raises:
        TypeError: If ``urban_graph`` is not an ``UrbanGraph`` or ``area`` / ``zones`` have a wrong type.
        ValueError: If the CRS is geographic, or both ``area`` and ``zones`` are passed.
    """
    crs = _projected(urban_graph, "graph_stats")
    if area is not None and zones is not None:
        raise ValueError("Pass either area or zones, not both")
    streets = _Streets(urban_graph, edge_types)
    nodes, edges = urban_graph.nodes_gdf, streets.edges

    boundary = None
    if zones is not None:
        zone_series = geometries_in_crs(zones, crs, "zones")
        zone_geoms, zone_index = zone_series.values, zone_series.index
    elif area is not None:
        area_geom = area_in_crs(area, crs)
        zone_geoms, zone_index, boundary = np.array([area_geom], dtype=object), None, area_geom.boundary
    else:
        zone_geoms, zone_index = None, None
    count = 1 if zone_geoms is None else len(zone_geoms)

    def assign(xy):
        return np.zeros(len(xy), dtype=np.int64) if zone_geoms is None else _zone_of(xy, zone_geoms)

    def total(zone, weights=None, mask=None):
        keep = zone >= 0 if mask is None else (zone >= 0) & mask
        return np.bincount(
            zone[keep], weights=None if weights is None else np.asarray(weights, dtype=float)[keep], minlength=count
        ).astype(float)

    node_zone = np.where(streets.active, assign(streets.xy), -1)
    edge_zone = assign(streets.midpoints())
    street_zone = edge_zone[streets.first]
    if len(edges):
        seg_start, seg_end, _, _ = segment_arrays(edges, precision)
    else:
        seg_start = seg_end = np.zeros((0, 2))
    seg_length = np.hypot(*(seg_end - seg_start).T) if len(seg_start) else np.zeros(0)
    seg_zone = assign((seg_start + seg_end) / 2.0)

    degree = streets.degree
    dead_end = degree == 1
    if boundary is not None and boundary_buffer is not None and dead_end.any():
        at_edge = np.flatnonzero(dead_end)
        near = shapely.dwithin(shapely.points(streets.xy[at_edge]), boundary, boundary_buffer)
        dead_end[at_edge[near]] = False
    intersection = degree >= 3

    out: dict[str, Any] = {}
    out["node_count"] = total(node_zone)
    out["edge_count"] = total(edge_zone)
    out["street_count"] = total(street_zone)
    out["street_length_total"] = total(seg_zone, seg_length)
    out["street_length_mean"] = _ratio(total(street_zone, streets.street_length), out["street_count"])
    out["streets_per_node_mean"] = _ratio(total(node_zone, degree), out["node_count"])
    out["dead_end_count"] = total(node_zone, mask=dead_end)
    out["dead_end_share"] = _ratio(out["dead_end_count"], out["node_count"])
    out["intersection_count"] = total(node_zone, mask=intersection)
    out["intersection_4way_share"] = _ratio(total(node_zone, mask=degree >= 4), out["intersection_count"])

    straight = (streets.lo != streets.hi) & (streets.straight > 0)
    out["circuity_mean"] = _ratio(
        total(street_zone, streets.street_length, straight), total(street_zone, streets.straight, straight)
    )
    out["self_loop_share"] = _ratio(total(street_zone, mask=streets.lo == streets.hi), out["street_count"])
    out["parallel_street_share"] = _ratio(total(street_zone, mask=streets.parallel), out["street_count"])

    # Components and connectivity indices on the streets with both ends in the same zone.
    inside = (node_zone[streets.lo] == node_zone[streets.hi]) & (node_zone[streets.lo] >= 0)
    links = sparse.coo_matrix(
        (np.ones(int(inside.sum())), (streets.lo[inside], streets.hi[inside])), shape=(streets.n, streets.n)
    )
    _, label = csgraph.connected_components(links, directed=False)
    member = np.flatnonzero(node_zone >= 0)
    parts = pd.DataFrame({"zone": node_zone[member], "label": label[member]}).value_counts()
    part_zone = parts.index.get_level_values("zone").to_numpy()
    components = np.bincount(part_zone, minlength=count).astype(float)
    largest = np.zeros(count)
    np.maximum.at(largest, part_zone, parts.to_numpy())
    out["component_count"] = components
    out["largest_component_share"] = _ratio(largest, out["node_count"])
    v = out["node_count"]
    e = total(node_zone[streets.lo], mask=inside)
    out["beta_index"] = _ratio(e, v)
    out["gamma_index"] = _ratio(e, 3 * (v - 2))
    out["cyclomatic_number"] = np.where(v > 0, e - v + components, np.nan)
    out["meshedness"] = _ratio(e - v + components, 2 * v - 5)

    # Bearings of both directions of every segment, in bins centred on 0°, 10°, ...
    if len(seg_start):
        delta = seg_end - seg_start
        bearing = np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) % 360.0
        width = 360.0 / ORIENTATION_BINS
        histogram = np.zeros((count, ORIENTATION_BINS))
        keep = seg_zone >= 0
        for angles in (bearing, (bearing + 180.0) % 360.0):
            bins = np.floor(((angles + width / 2.0) % 360.0) / width).astype(np.int64) % ORIENTATION_BINS
            np.add.at(histogram, (seg_zone[keep], bins[keep]), seg_length[keep])
        entropy = _entropy(histogram)
    else:
        entropy = np.full(count, np.nan)
    out["orientation_entropy"] = entropy
    # Boeing (2019): 1 for a perfect grid (entropy ln 4), 0 for bearings spread evenly (entropy ln 36).
    grid, uniform = np.log(4.0), np.log(ORIENTATION_BINS)
    out["orientation_order"] = 1.0 - np.clip((entropy - grid) / (uniform - grid), 0.0, 1.0) ** 2

    direction = urban_graph.edge_direction_column
    if urban_graph.is_directed and direction is not None and direction in edges.columns:
        oneway = edges[direction].to_numpy(dtype=bool)[streets.first]
        out["oneway_share"] = _ratio(
            total(street_zone, streets.street_length, oneway), total(street_zone, streets.street_length)
        )
    if {"length_meter", "time_min"} <= set(edges.columns):
        out["speed_mean_kmh"] = 0.06 * _ratio(
            total(edge_zone, edges["length_meter"].to_numpy()), total(edge_zone, edges["time_min"].to_numpy())
        )

    if "elevation" in nodes.columns:
        z = pd.to_numeric(nodes["elevation"], errors="coerce").to_numpy(dtype=float)
        known = np.isfinite(z)
        low, high = np.full(count, np.inf), np.full(count, -np.inf)
        valid = known & (node_zone >= 0)
        np.minimum.at(low, node_zone[valid], z[valid])
        np.maximum.at(high, node_zone[valid], z[valid])
        out["elevation_min"] = np.where(np.isfinite(low), low, np.nan)
        out["elevation_max"] = np.where(np.isfinite(high), high, np.nan)
        out["elevation_mean"] = _ratio(total(node_zone, np.where(known, z, 0.0), known), total(node_zone, mask=known))
    if "grade" in edges.columns:
        grade = np.abs(pd.to_numeric(edges["grade"], errors="coerce").to_numpy(dtype=float))[streets.first]
        known = np.isfinite(grade)
        weighted = np.where(known, grade, 0.0) * streets.street_length
        out["grade_mean"] = _ratio(
            total(street_zone, weighted, known), total(street_zone, streets.street_length, known)
        )
        steepest = np.full(count, -np.inf)
        column = "grade_max" if "grade_max" in edges.columns else "grade"
        steep = np.abs(pd.to_numeric(edges[column], errors="coerce").to_numpy(dtype=float))[streets.first]
        valid = np.isfinite(steep) & (street_zone >= 0)
        np.maximum.at(steepest, street_zone[valid], steep[valid])
        out["grade_max"] = np.where(np.isfinite(steepest), steepest, np.nan)

    if edge_types is None and "type" in nodes.columns and "type" in edges.columns:
        platform = (nodes["type"].astype("string").fillna("") == "platform").to_numpy(dtype=bool)
        if platform.any():
            edge_type = edges["type"].astype("string").fillna("").to_numpy()
            walk = edge_type == "walk"
            index = pd.Index(nodes.index)
            reached = np.zeros(streets.n, dtype=bool)
            reached[index.get_indexer(edges["u"])[walk]] = True
            reached[index.get_indexer(edges["v"])[walk]] = True
            out["platform_count"] = total(node_zone, mask=platform)
            if walk.any():
                out["unattached_platform_count"] = total(node_zone, mask=platform & ~reached)
            out["boarding_edge_count"] = total(edge_zone, mask=edge_type == "boarding")

    if zone_geoms is not None:
        area_km2 = shapely.area(zone_geoms) / 1e6
        out["area_km2"] = area_km2
        out["node_density"] = _ratio(out["node_count"], area_km2)
        out["intersection_density"] = _ratio(out["intersection_count"], area_km2)
        out["street_density"] = _ratio(out["street_length_total"] / 1000.0, area_km2)

    if zones is not None:
        return pd.DataFrame(out, index=zone_index)
    return pd.Series({key: float(value[0]) for key, value in out.items()}, name="value")


def edge_stats(
    urban_graph: UrbanGraph,
    by: str = "type",
    *,
    area: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    precision: float = 0.01,
) -> pd.DataFrame:
    """Edge counts, lengths and speeds for every value of an edge column.

    Args:
        urban_graph: Graph in a projected (metric) CRS.
        by: Edge column to group by, for example ``"type"``, ``"category"`` or ``"highway"``. Edges without a
            value form the group ``"<missing>"``.
        area: Measure only edges and segments whose midpoint lies inside (see :func:`graph_stats`).
        precision: Grid for comparing segment end points, see :func:`~iduedu.unique_segments`.

    Returns:
        DataFrame indexed by the values of ``by``, sorted by length, with columns ``edge_count``,
        ``length_km`` (every physical segment once, counted for the group of the first edge that has it),
        ``length_share``, ``edge_length_mean`` (metres) and ``speed_kmh`` (total length over total travel time,
        when edges have ``length_meter`` and ``time_min``).

    Raises:
        KeyError: If ``edges_gdf`` has no column ``by``.
        ValueError: If the CRS is geographic.
    """
    crs = _projected(urban_graph, "edge_stats")
    edges = urban_graph.edges_gdf
    if by not in edges.columns:
        raise KeyError(f"graph.edges_gdf has no column {by!r}")
    columns = ["edge_count", "length_km", "length_share", "edge_length_mean", "speed_kmh"]
    if edges.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name=by))
    group = edges[by].map(lambda value: value if isinstance(value, str) else str(value), na_action="ignore")
    group = group.fillna("<missing>").to_numpy()
    length = shapely.length(edges.geometry.values)
    seg_start, seg_end, seg_owner, _ = segment_arrays(edges, precision)
    seg_length = np.hypot(*(seg_end - seg_start).T)
    inside_edge, inside_seg = np.ones(len(edges), dtype=bool), np.ones(len(seg_owner), dtype=bool)
    if area is not None:
        zone = np.array([area_in_crs(area, crs)], dtype=object)
        midpoints = shapely.get_coordinates(shapely.line_interpolate_point(edges.geometry.values, 0.5, normalized=True))
        inside_edge = _zone_of(midpoints, zone) >= 0
        inside_seg = _zone_of((seg_start + seg_end) / 2.0, zone) >= 0

    table = pd.DataFrame({"group": group[inside_edge], "length": length[inside_edge]})
    result = table.groupby("group").agg(edge_count=("length", "size"), edge_length_mean=("length", "mean"))
    segments = pd.DataFrame({"group": group[seg_owner][inside_seg], "length": seg_length[inside_seg]})
    result["length_km"] = segments.groupby("group")["length"].sum().reindex(result.index).fillna(0.0) / 1000.0
    whole = result["length_km"].sum()
    result["length_share"] = result["length_km"] / whole if whole > 0 else np.nan
    if {"length_meter", "time_min"} <= set(edges.columns):
        travel = (
            pd.DataFrame(
                {
                    "group": group[inside_edge],
                    "length": edges["length_meter"].to_numpy(dtype=float)[inside_edge],
                    "time": edges["time_min"].to_numpy(dtype=float)[inside_edge],
                }
            )
            .groupby("group")[["length", "time"]]
            .sum()
        )
        result["speed_kmh"] = 0.06 * _ratio(
            travel["length"].reindex(result.index), travel["time"].reindex(result.index)
        )
    else:
        result["speed_kmh"] = np.nan
    result.index.name = by
    return result.sort_values("length_km", ascending=False)[columns]


def transport_stats(urban_graph: UrbanGraph, *, modes: Iterable[str] | None = None) -> pd.DataFrame:
    """Stops, routes and stop spacing of every public-transport mode of a graph.

    Works on public-transport and intermodal graphs, where the edges between stops and the stop nodes of a
    mode carry the mode as their ``type`` (``bus``, ``tram``, ``subway``, ...).

    Args:
        urban_graph: Graph with a ``type`` column on nodes and edges.
        modes: Modes to report. By default the edge types that are modes of the transport registry
            (``DEFAULT_REGISTRY_W_TRAIN``) or GTFS route types; pass the names of custom modes here.

    Returns:
        DataFrame indexed by mode with columns ``stop_count`` (stop nodes of the mode), ``route_count``
        (distinct values of the edge column ``route``, ``NaN`` without it), ``edge_count``, ``length_km`` (sum
        of ``length_meter``, both directions of a route counted), ``stop_spacing_mean`` (mean edge length,
        metres) and ``speed_kmh`` (total length over total travel time).

    Raises:
        KeyError: If nodes or edges have no ``type`` column.
    """
    if not isinstance(urban_graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(urban_graph).__name__}")
    nodes, edges = urban_graph.nodes_gdf, urban_graph.edges_gdf
    if "type" not in edges.columns or "type" not in nodes.columns:
        raise KeyError("transport_stats needs a 'type' column on nodes_gdf and edges_gdf")
    stops = nodes["type"].value_counts()
    edge_type = edges["type"]
    if modes is None:
        # Imported here: the builders import the graph package.
        from iduedu.graph_builders.gtfs_builders import GTFS_ROUTE_TYPES  # pylint: disable=import-outside-toplevel

        known = set(DEFAULT_REGISTRY_W_TRAIN.list_types()) | set(GTFS_ROUTE_TYPES.values())
        modes = [mode for mode in edge_type.dropna().unique() if mode in known]
    rows = {}
    for mode in modes:
        part = edges[edge_type == mode]
        length = part["length_meter"].to_numpy(dtype=float) if "length_meter" in part.columns else np.zeros(0)
        time = part["time_min"].to_numpy(dtype=float) if "time_min" in part.columns else np.zeros(0)
        rows[mode] = {
            "stop_count": int(stops.get(mode, 0)),
            "route_count": float(part["route"].nunique()) if "route" in part.columns else np.nan,
            "edge_count": len(part),
            "length_km": float(length.sum()) / 1000.0,
            "stop_spacing_mean": float(length.mean()) if len(length) else np.nan,
            "speed_kmh": 0.06 * float(length.sum() / time.sum()) if len(time) and time.sum() > 0 else np.nan,
        }
    columns = ["stop_count", "route_count", "edge_count", "length_km", "stop_spacing_mean", "speed_kmh"]
    result = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    result.index.name = "mode"
    return result.sort_values("length_km", ascending=False)


def local_stats(
    urban_graph: UrbanGraph,
    radius: float | Iterable[float],
    *,
    weight: str = "length_meter",
    nodes: Iterable[Any] | None = None,
    edge_types: Iterable[str] | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Network measures around every node, within a travel cost along the network.

    For each node the nodes reachable within ``radius`` are collected with a cut-off shortest-path search, and
    the streets are counted through them: every street gives half of its length to each of its two ends.

    Args:
        urban_graph: Graph in a projected (metric) CRS.
        radius: Network radius, or several radii, in the units of ``weight`` (metres by default).
        weight: Edge column that measures the radius: ``"length_meter"`` or ``"time_min"``.
        nodes: Nodes to measure; by default every node. The search always runs on the whole graph.
        edge_types: Count only the streets whose ``type`` is one of these. The search still uses every edge.
        max_workers: Number of threads; ``None`` uses all cores.

    Returns:
        DataFrame indexed by node id. For every radius ``r`` the columns ``node_count_r``,
        ``intersection_count_r``, ``dead_end_count_r``, ``street_length_r`` (metres),
        ``streets_per_node_mean_r``, ``dead_end_share_r`` and ``intersections_per_km_r`` (intersections per
        kilometre of street). The node itself is included.

    Raises:
        ValueError: If the CRS is geographic, ``radius`` is invalid or a node is absent from the graph.
    """
    _projected(urban_graph, "local_stats")
    bands = cost_bands(radius, "radius")
    streets = _Streets(urban_graph, edge_types)
    n, degree = streets.n, streets.degree
    index = pd.Index(urban_graph.nodes_gdf.index)
    if nodes is None:
        positions, labels = np.arange(n, dtype=np.int32), index
    else:
        labels = pd.Index(list(nodes))
        positions = index.get_indexer(labels)
        if (positions < 0).any():
            raise ValueError(f"nodes absent in graph: {labels[positions < 0][:10].tolist()}")
    half = np.zeros(n)
    np.add.at(half, streets.lo, streets.street_length / 2.0)
    np.add.at(half, streets.hi, streets.street_length / 2.0)
    active = np.flatnonzero(streets.active)
    columns = [np.ones(len(active)), degree[active] >= 3, degree[active] == 1, degree[active], half[active]]
    values = value_rows(
        n,
        np.tile(active, len(columns)),
        np.repeat(np.arange(len(columns)), len(active)),
        np.concatenate([np.asarray(c, dtype=float) for c in columns]),
        len(columns),
    )
    sums = sum_within(urban_graph, positions, values, len(columns), bands, weight=weight, max_workers=max_workers)
    out = {}
    for b, band in enumerate(bands):
        suffix = band_label(band)
        node_count, intersections, dead_ends, degrees, length = (sums[:, b, c] for c in range(len(columns)))
        out[f"node_count_{suffix}"] = node_count
        out[f"intersection_count_{suffix}"] = intersections
        out[f"dead_end_count_{suffix}"] = dead_ends
        out[f"street_length_{suffix}"] = length
        out[f"streets_per_node_mean_{suffix}"] = _ratio(degrees, node_count)
        out[f"dead_end_share_{suffix}"] = _ratio(dead_ends, node_count)
        out[f"intersections_per_km_{suffix}"] = _ratio(intersections, length / 1000.0)
    return pd.DataFrame(out, index=labels.rename("node"))


def route_directness(
    urban_graph: UrbanGraph,
    *,
    origins_nodes: Iterable[Any] | None = None,
    destination_nodes: Iterable[Any] | None = None,
    gdf_origins: pd.DataFrame | None = None,
    gdf_destinations: pd.DataFrame | None = None,
    graph_node_column: str = "graph_node_id",
    sample: int = 200,
    seed: int | None = 0,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Ratio of the shortest network distance to the straight distance between origins and destinations.

    1 means a straight route; a street grid gives about 1.27 on average. Distances are measured by
    ``length_meter`` between graph nodes, and the straight distance between the same nodes, so objects are
    represented by their nearest nodes.

    Args:
        urban_graph: Graph in a projected (metric) CRS.
        origins_nodes: Origin node ids, or
        gdf_origins: origin objects (with ``graph_node_column`` or matched to their nearest nodes).
        destination_nodes: Destination node ids, or
        gdf_destinations: destination objects. Without destinations the origins are used.
        graph_node_column: Column of object tables that holds graph node ids.
        sample: Without origins, this many random nodes are taken as origins and destinations.
        seed: Seed of the random sample.
        max_workers: Number of threads; ``None`` uses all cores.

    Returns:
        DataFrame of ratios, origins in rows and destinations in columns; ``NaN`` for unreachable pairs and
        for pairs at the same place. ``result.stack().mean()`` is the mean directness.
    """
    _projected(urban_graph, "route_directness")

    def resolve(node_ids, gdf, name):
        return resolve_graph_nodes_input(
            urban_graph=urban_graph,
            nodes=node_ids,
            gdf=gdf,
            graph_node_column=graph_node_column,
            nodes_name=f"{name}_nodes",
            gdf_name=f"gdf_{name}",
        )

    if origins_nodes is None and gdf_origins is None:
        if destination_nodes is not None or gdf_destinations is not None:
            raise ValueError("Pass origins together with destinations")
        if isinstance(sample, bool) or not isinstance(sample, (int, np.integer)) or sample < 2:
            raise ValueError("sample must be an integer of at least 2")
        index = urban_graph.nodes_gdf.index
        picked = np.sort(np.random.default_rng(seed).choice(len(index), size=min(sample, len(index)), replace=False))
        origins = destinations = pd.Series(index[picked], index=index[picked])
    else:
        origins = resolve(origins_nodes, gdf_origins, "origins")
        if destination_nodes is None and gdf_destinations is None:
            destinations = origins
        else:
            destinations = resolve(destination_nodes, gdf_destinations, "destination")

    csr = _prepare_graph(urban_graph, weight="length_meter", cutoff=None, reverse=False)
    sources = _positions(urban_graph, origins.to_numpy())
    targets = _positions(urban_graph, destinations.to_numpy())
    network = csr.distance_matrix(sources, targets, np.float32(np.inf), max_workers).astype(float)
    nodes = urban_graph.nodes_gdf.geometry
    order = pd.Index(urban_graph.adjacency_nodelist)
    xy = np.column_stack([nodes.x.to_numpy(), nodes.y.to_numpy()])[pd.Index(nodes.index).get_indexer(order)]
    straight = np.hypot(
        xy[sources, 0][:, None] - xy[targets, 0][None, :], xy[sources, 1][:, None] - xy[targets, 1][None, :]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.isfinite(network) & (straight > 0), network / straight, np.nan)
    return pd.DataFrame(ratio, index=origins.index, columns=destinations.index)
