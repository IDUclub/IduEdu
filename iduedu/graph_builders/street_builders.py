"""One street network for walking, driving and cycling, and the graph of each mode taken from it."""

from collections.abc import Iterable
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Polygon
from shapely.geometry.multipolygon import MultiPolygon

from iduedu import config
from iduedu.constants.network_enums import Network
from iduedu.graph.simplify import simplify_graph
from iduedu.graph.transformers import estimate_crs_for_bounds, keep_largest_connected_component
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.graph_builders.bike_builders import (
    _CYCLE_TAGS,
    _PERMITTED,
    FOOTWAYS,
    _text,
    cycle_infrastructure,
    level_of_traffic_stress,
)
from iduedu.graph_builders.drive_walk_builders import (
    STRUCTURE_TAGS,
    _assign_edge_keys,
    _build_edges_from_overpass,
    _build_nodes_and_uv,
    _get_highway_properties,
)
from iduedu.osm_pbf.filters import parse_way_filter
from iduedu.overpass.downloaders import get_4326_boundary, osm_source_label

logger = config.logger

Mode = Literal["walk", "drive", "bike"]
STREET_MODES: tuple[str, ...] = ("walk", "drive", "bike")
#: Columns that say where each mode may travel along the drawing of the edge (``forward``) or against it.
MODE_FLAGS = {
    "walk": ("walk",),
    "drive": ("drive_forward", "drive_backward"),
    "bike": ("bike_forward", "bike_backward"),
}
#: Columns that belong to one mode only; the graph of another mode leaves them out.
MODE_COLUMNS = {
    "walk": ("walk", "time_walk_min"),
    "drive": ("drive_forward", "drive_backward", "time_drive_min", "category"),
    "bike": ("bike_forward", "bike_backward", "time_bike_min", "cycle_infra", "lts"),
}
#: Tag columns the graph of each mode keeps, like its own builder (``osm_edge_tags`` add to all of them).
MODE_TAGS = {
    "walk": ("highway", "name"),
    "drive": ("highway", "name", "lanes", "maxspeed"),
    "bike": ("highway", "name", "lanes", "maxspeed", "surface"),
}
_STREET_TAGS = ("highway", "name", "lanes", "maxspeed", "surface")
_FORWARD_ONLY = {"yes", "true", "1"}


def _mode_filters(modes: tuple[str, ...], drive_network: str) -> dict[str, str]:
    presets = {
        "walk": Network.WALK.filter,
        "drive": Network.DRIVE_SERVICE.filter if drive_network == "drive_service" else Network.DRIVE.filter,
        "bike": Network.BIKE.filter,
    }
    return {mode: presets[mode] for mode in modes}


def _way_modes(
    ways: pd.DataFrame, filters: dict[str, str], *, allow_footways: bool
) -> pd.DataFrame:  # pylint: disable=too-many-locals
    """Per way: where each mode may travel, and the attributes the modes derive from the tags."""
    tags = ways["tags"].map(lambda value: value if isinstance(value, dict) else {})
    keys = set(_CYCLE_TAGS) | {"highway", "maxspeed", "lanes"}
    table = pd.DataFrame({key: tags.map(lambda d, k=key: d.get(k)) for key in keys}, index=ways.index)
    inside = {mode: tags.map(parse_way_filter(text)).to_numpy(dtype=bool) for mode, text in filters.items()}
    none = np.zeros(len(ways), dtype=bool)

    oneway = _text(table, "oneway")
    forward_only = oneway.isin(_FORWARD_ONLY).to_numpy()
    backward_only = oneway.eq("-1").to_numpy()
    columns: dict[str, object] = {}
    if "walk" in filters:
        columns["walk"] = inside["walk"]
    if "drive" in filters:
        drive = inside["drive"]
        columns["drive_forward"] = drive & ~backward_only
        columns["drive_backward"] = drive & ~forward_only
        category, speed = zip(*table["highway"].map(_get_highway_properties)) if len(table) else ((), ())
        speed = pd.Series(speed, index=ways.index, dtype=float)
        limit = (pd.to_numeric(table["maxspeed"], errors="coerce") * 1000.0 / 60.0).round(3)
        columns["category"] = pd.Series(category, index=ways.index, dtype=object).where(drive)
        columns["_speed_drive"] = limit.fillna(speed).where(drive).to_numpy(dtype=float)
    if "bike" in filters:
        highway, bicycle = _text(table, "highway"), _text(table, "bicycle")
        usable = inside["bike"] & (bicycle != "no").to_numpy()
        if not allow_footways:
            usable &= (~highway.isin(FOOTWAYS) | bicycle.isin(_PERMITTED)).to_numpy()
        contraflow = (_text(table, "oneway:bicycle") == "no").to_numpy()
        for column in ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both"):
            contraflow |= _text(table, column).str.startswith("opposite").to_numpy()
        columns["bike_forward"] = usable & ~(backward_only & ~contraflow)
        columns["bike_backward"] = usable & ~(forward_only & ~contraflow)
        columns["cycle_infra"] = cycle_infrastructure(table).where(usable)
        columns["lts"] = level_of_traffic_stress(table).astype("Int64").where(usable)

    result = pd.DataFrame(columns, index=ways.index)
    flags = [c for flags in MODE_FLAGS.values() for c in flags if c in result.columns]
    any_mode = result[flags].any(axis=1).to_numpy() if flags else none
    return result[any_mode]


def get_street_graph(  # pylint: disable=too-many-locals
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    modes: Iterable[Mode] = STREET_MODES,
    simplify: bool = True,
    keep_largest_subgraph: bool = False,
    clip_by_territory: bool = False,
    walk_speed: float = 5 * 1000 / 60,
    bike_speed: float = 15 * 1000 / 60,
    allow_footways: bool = False,
    drive_network: Literal["drive", "drive_service"] = "drive",
    osm_edge_tags: list[str] | None = None,
) -> UrbanGraph:
    """
    Build one street network for walking, driving and cycling, with every street stored once.

    The ways of all ``modes`` are read in one request (or one pass over the local file) with the filters of
    :func:`get_walk_graph`, :func:`get_drive_graph` and :func:`get_bike_graph`. For each way and mode the
    builder decides, as those builders do, whether the mode may use it and in which directions: walking both
    ways; driving after ``oneway`` (``-1`` against the drawing, ``reversible`` both ways); cycling after
    ``oneway`` unless ``oneway:bicycle=no`` or a ``cycleway*=opposite*`` tag lets bicycles ride against it.
    Segments are merged into edges only where all of this, and the kept tags, stay the same, so an edge never
    spans a change for any mode; bridges and tunnels are separate edges. The geometry of every edge runs from
    ``u`` to ``v`` in the direction the way is drawn. :func:`mode_graph` takes the graph of one mode from it.

    Parameters:
        osm_id (int | None): OSM relation id of the territory boundary. Provide this or ``territory``.
        territory: Boundary in EPSG:4326 or a GeoDataFrame, used when ``osm_id`` is not given.
        modes: Modes to include, any of ``"walk"``, ``"drive"`` and ``"bike"``.
        simplify (bool): Merge segments into edges that run from junction to junction.
        keep_largest_subgraph (bool): Keep only the largest connected part of the street network.
        clip_by_territory (bool): Clip edges by the exact boundary.
        walk_speed (float): Walking speed in metres per minute.
        bike_speed (float): Cycling speed on flat ground in metres per minute.
        allow_footways (bool): Let bicycles use footways and pedestrian streets without ``bicycle=yes``.
        drive_network: ``"drive"``, or ``"drive_service"`` to drive on service roads too.
        osm_edge_tags (list[str] | None): OSM tags kept on the edges. By default ``highway``, ``name``,
            ``lanes``, ``maxspeed`` and ``surface``; ``bridge`` and ``tunnel`` are always kept.

    Returns:
        UrbanGraph: Undirected multigraph of type ``"street"``. Edges have ``length_meter``; the flags
        ``walk``, ``drive_forward``, ``drive_backward``, ``bike_forward`` and ``bike_backward`` (``forward`` is
        from ``u`` to ``v``); the travel times ``time_walk_min``, ``time_drive_min`` and ``time_bike_min``, NaN
        where the mode does not go; ``category`` for driving; ``cycle_infra`` and ``lts`` for cycling; the kept
        tags and ``bridge`` and ``tunnel``. ``time_min`` is NaN: route on the graph of a mode.

    Raises:
        ValueError: If ``modes`` is empty or names an unknown mode.
    """
    modes = tuple(dict.fromkeys(modes))
    if not modes or set(modes) - set(STREET_MODES):
        raise ValueError(f"modes must be a non-empty subset of {STREET_MODES}, got {modes}")
    if drive_network not in ("drive", "drive_service"):
        raise ValueError(f"drive_network must be 'drive' or 'drive_service', got {drive_network!r}")
    polygon4326 = get_4326_boundary(osm_id=osm_id, territory=territory)
    local_crs = estimate_crs_for_bounds(*polygon4326.bounds)
    filters = _mode_filters(modes, drive_network)
    tags = set(_STREET_TAGS if osm_edge_tags is None else osm_edge_tags)

    logger.info(f"Retrieving street network for {', '.join(modes)} via {osm_source_label()}...")
    edges = _build_edges_from_overpass(
        polygon4326,
        list(filters.values()),
        needed_tags=tags,
        local_crs=local_crs,
        simplify=simplify,
        way_columns=lambda ways: _way_modes(ways, filters, allow_footways=allow_footways),
    )
    if len(edges) == 0:
        logger.warning("No edges found, returning empty graph")
        return UrbanGraph.empty(crs=local_crs, is_multigraph=True, is_directed=False, graph_type="street")
    if clip_by_territory:
        clip_poly_gdf = gpd.GeoDataFrame(geometry=[polygon4326], crs=4326).to_crs(local_crs)
        edges = edges.clip(clip_poly_gdf, keep_geom_type=True).explode(ignore_index=True)
        if len(edges) == 0:
            logger.warning("No edges left after clipping, returning empty graph")
            return UrbanGraph.empty(crs=local_crs, is_multigraph=True, is_directed=False, graph_type="street")

    nodes, edges = _build_nodes_and_uv(edges, local_crs)
    length = edges.geometry.length.round(3)
    edges["length_meter"] = length
    edges["time_min"] = np.nan
    if "walk" in modes:
        edges["time_walk_min"] = (length / float(walk_speed)).round(3).where(edges["walk"])
    if "drive" in modes:
        drives = edges["drive_forward"] | edges["drive_backward"]
        edges["time_drive_min"] = (length / edges["_speed_drive"]).round(3).where(drives)
    if "bike" in modes:
        bikes = edges["bike_forward"] | edges["bike_backward"]
        edges["time_bike_min"] = (length / float(bike_speed)).round(3).where(bikes)
    edges["type"] = "street"
    edges = _assign_edge_keys(edges)

    flags = [c for mode in modes for c in MODE_FLAGS[mode]]
    times = [f"time_{mode}_min" for mode in modes]
    derived = [c for c in ("category", "cycle_infra", "lts") if c in edges.columns]
    order = ["u", "v", "k", "length_meter", "time_min", *flags, *times, *sorted(tags), *STRUCTURE_TAGS, *derived]
    edges = edges[[c for c in dict.fromkeys(order) if c in edges.columns] + ["type", "geometry"]]
    edges = gpd.GeoDataFrame(edges, geometry="geometry", crs=local_crs)

    graph = UrbanGraph(
        nodes_gdf=nodes, edges_gdf=edges, is_multigraph=True, is_directed=False, crs=local_crs, graph_type="street"
    )
    if keep_largest_subgraph:
        graph = keep_largest_connected_component(graph)
    logger.debug("Street graph built.")
    return graph


def mode_graph(
    street_graph: UrbanGraph, mode: Mode, *, simplify: bool = True, keep_largest_subgraph: bool = False
) -> UrbanGraph:
    """
    Take the graph of one mode from a street graph of :func:`get_street_graph`.

    The graph holds the edges the mode may use in at least one direction, with ``time_min`` from the time of
    the mode, and the nodes of those edges. Walking gives an undirected graph. Driving and cycling give
    directed graphs with an ``oneway`` column, as their builders do; an edge the mode may use only against its
    drawing is turned (``u`` and ``v`` swap and the geometry is reversed). With ``simplify`` the edges are
    merged through the nodes that have only two edges of this mode (see :func:`simplify_graph`), so a road is
    not cut where only a footway joins it.

    Parameters:
        street_graph (UrbanGraph): Graph of :func:`get_street_graph`.
        mode: ``"walk"``, ``"drive"`` or ``"bike"``.
        simplify (bool): Merge chains of edges through nodes of degree two of this mode.
        keep_largest_subgraph (bool): Keep only the largest connected part (strongly connected for driving and
            cycling).

    Returns:
        UrbanGraph: Multigraph of type ``mode`` with the columns of the matching builder.

    Raises:
        ValueError: If the street graph has no columns for ``mode``.
    """
    if mode not in MODE_FLAGS:
        raise ValueError(f"mode must be one of {tuple(MODE_FLAGS)}, got {mode!r}")
    edges = street_graph.edges_gdf
    flags = MODE_FLAGS[mode]
    if not set(flags) <= set(edges.columns):
        raise ValueError(f"The street graph has no {mode!r} columns; build it with {mode!r} in modes")
    known = {c for columns in MODE_COLUMNS.values() for c in columns} | set(_STREET_TAGS)
    foreign = (known - set(MODE_COLUMNS[mode]) - set(MODE_TAGS[mode])) | {"time_min", "type"}
    directed = mode != "walk"

    if directed:
        forward = edges[flags[0]].to_numpy(dtype=bool)
        backward = edges[flags[1]].to_numpy(dtype=bool)
        chosen = edges[forward | backward].copy()
        forward, backward = forward[forward | backward], backward[forward | backward]
        turn = backward & ~forward
        chosen.loc[turn, ["u", "v"]] = chosen.loc[turn, ["v", "u"]].to_numpy()
        chosen.loc[turn, "geometry"] = chosen.loc[turn, "geometry"].reverse()
        chosen["oneway"] = ~(forward & backward)
    else:
        chosen = edges[edges[flags[0]].to_numpy(dtype=bool)].copy()
    chosen["time_min"] = chosen[f"time_{mode}_min"]
    chosen["type"] = mode
    chosen = chosen.drop(columns=[c for c in chosen.columns if c in foreign - {"time_min", "type"}])
    chosen = chosen.drop(columns=[c for c in (*flags, f"time_{mode}_min") if c in chosen.columns])
    chosen = _assign_edge_keys(chosen.reset_index(drop=True))
    used = pd.Index(chosen["u"]).union(pd.Index(chosen["v"]))
    nodes = street_graph.nodes_gdf.loc[street_graph.nodes_gdf.index.isin(used)]

    graph = UrbanGraph(
        nodes_gdf=nodes,
        edges_gdf=chosen,
        is_multigraph=True,
        is_directed=directed,
        edge_direction_column="oneway" if directed else None,
        crs=street_graph.crs,
        graph_type=mode,
    )
    if simplify:
        graph = simplify_graph(graph)
    if keep_largest_subgraph:
        graph = keep_largest_connected_component(graph)
    return graph
