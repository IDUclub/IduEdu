"""Cycling graph from OpenStreetMap: where a bicycle may ride, what protects it and how stressful it is."""

import os
from typing import Callable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import LineString, Polygon
from shapely.geometry.multipolygon import MultiPolygon

from iduedu import config
from iduedu.constants.highway_enums import HighwayType
from iduedu.constants.network_enums import Network
from iduedu.graph.elevation import add_elevation, apply_grade_speed
from iduedu.graph.transformers import estimate_crs_for_bounds, keep_largest_connected_component
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.graph_builders.drive_walk_builders import (
    STRUCTURE_TAGS,
    _assign_edge_keys,
    _build_edges_from_overpass,
    _build_nodes_and_uv,
)
from iduedu.overpass.downloaders import get_4326_boundary, osm_source_label

logger = config.logger

#: Tags read to classify the infrastructure and the direction of travel (not kept as columns).
_CYCLE_TAGS = ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both", "bicycle", "oneway", "oneway:bicycle")
#: Ways without motor traffic.
CAR_FREE = frozenset({"cycleway", "path", "footway", "pedestrian", "track", "bridleway"})
#: Ways for pedestrians, used only with ``allow_footways`` or an explicit ``bicycle`` permission.
FOOTWAYS = frozenset({"footway", "pedestrian"})
_QUIET = frozenset({"residential", "living_street", "unclassified", "service"})
_TRACK = frozenset({"track", "opposite_track"})
_LANE = frozenset({"lane", "opposite_lane", "share_busway", "opposite_share_busway"})
_PERMITTED = frozenset({"yes", "designated", "permissive", "official"})
_DEFAULT_SPEED_KMH = 40.0


def _text(edges: pd.DataFrame, column: str) -> pd.Series:
    """A tag column as lower-case text, empty where the tag is absent."""
    if column not in edges.columns:
        return pd.Series("", index=edges.index, dtype="string")
    return edges[column].astype("string").str.strip().str.lower().fillna("")


def _highway_speed_kmh(highway: pd.Series) -> np.ndarray:
    """Default speed limit of each highway class, km/h (40 for classes outside ``HighwayType``)."""
    known = {item.value: item.max_speed * 0.06 for item in HighwayType}
    return highway.map(known).fillna(_DEFAULT_SPEED_KMH).to_numpy(dtype=float)


def cycle_infrastructure(edges: pd.DataFrame) -> pd.Series:
    """Classify the cycling infrastructure of edges with OSM tags.

    Returns a Series with ``"track"`` (a way of its own or without motor traffic: ``highway=cycleway``,
    paths and footways, or ``cycleway*=track``), ``"lane"`` (a marked lane on the carriageway:
    ``cycleway*=lane`` or a shared bus lane) or ``"mixed"`` (riding with the traffic).
    """
    highway = _text(edges, "highway")
    tags = [_text(edges, column) for column in ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both")]
    track = highway.isin(CAR_FREE)
    lane = pd.Series(False, index=edges.index)
    for tag in tags:
        track |= tag.isin(_TRACK)
        lane |= tag.isin(_LANE)
    return pd.Series(np.where(track, "track", np.where(lane, "lane", "mixed")), index=edges.index, name="cycle_infra")


def level_of_traffic_stress(edges: pd.DataFrame) -> pd.Series:
    """Level of traffic stress (LTS) of edges with OSM tags, from 1 (anyone rides here) to 4 (only the fearless).

    A simplified form of the criteria of Mekuria, Furth and Nixon (2012) for the tags OSM has: the kind of
    infrastructure (:func:`cycle_infrastructure`), the speed limit (``maxspeed``, or the default of the
    highway class) and the number of lanes (``lanes``, 2 when absent).

    - track: 1;
    - lane: 1 up to 30 km/h and two lanes, 2 up to 50 km/h and two lanes, 3 up to 60 km/h or with more
      lanes, otherwise 4;
    - mixed traffic on residential, living, unclassified and service streets: 1 up to 30 km/h and two
      lanes (always on living streets), 2 up to 50 km/h and two lanes, 3 with more lanes, 4 above 50 km/h;
    - mixed traffic on other roads: 2 up to 30 km/h and two lanes, 3 up to 50 km/h and three lanes,
      otherwise 4.
    """
    highway = _text(edges, "highway")
    infra = cycle_infrastructure(edges).to_numpy()
    limit = pd.to_numeric(edges["maxspeed"], errors="coerce") if "maxspeed" in edges.columns else pd.Series(np.nan)
    speed = np.where(np.isfinite(limit.to_numpy(dtype=float)), limit.to_numpy(dtype=float), _highway_speed_kmh(highway))
    lanes = pd.to_numeric(edges["lanes"], errors="coerce") if "lanes" in edges.columns else pd.Series(np.nan)
    lanes = np.where(np.isfinite(lanes.to_numpy(dtype=float)), lanes.to_numpy(dtype=float), 2.0)
    quiet = highway.isin(_QUIET).to_numpy()
    living = (highway == "living_street").to_numpy()

    lane_lts = np.select(
        [(speed <= 30) & (lanes <= 2), (speed <= 50) & (lanes <= 2), speed <= 60],
        [1, 2, 3],
        default=4,
    )
    quiet_lts = np.select(
        [living | ((speed <= 30) & (lanes <= 2)), (speed <= 50) & (lanes <= 2), speed <= 50],
        [1, 2, 3],
        default=4,
    )
    busy_lts = np.select([(speed <= 30) & (lanes <= 2), (speed <= 50) & (lanes <= 3)], [2, 3], default=4)
    lts = np.where(infra == "track", 1, np.where(infra == "lane", lane_lts, np.where(quiet, quiet_lts, busy_lts)))
    return pd.Series(lts.astype(np.int64), index=edges.index, name="lts")


def _empty(local_crs, network_type):
    return UrbanGraph.empty(
        crs=local_crs, is_multigraph=True, is_directed=True, edge_direction_column="oneway", graph_type=network_type
    )


def get_bike_graph(  # pylint: disable=too-many-locals
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    simplify: bool = True,
    clip_by_territory: bool = False,
    keep_largest_subgraph: bool = True,
    bike_speed: float = 15 * 1000 / 60,  # m/min
    max_lts: int | None = None,
    allow_footways: bool = False,
    elevation: str | os.PathLike | gpd.GeoDataFrame | Callable | None = None,
    network_type: Literal["bike", "custom"] = "bike",
    custom_filter: str | None = None,
    osm_edge_tags: list[str] | None = None,
) -> UrbanGraph:
    """
    Build a cycling UrbanGraph from OpenStreetMap within a given territory.

    The graph holds every way a bicycle may use: streets (except motorways and ways closed to bicycles),
    cycleways, paths and tracks, and footways and pedestrian streets where cycling is permitted
    (``bicycle=yes``, ``designated`` or ``permissive``). Every edge gets the kind of cycling infrastructure
    (``cycle_infra``: ``track``, ``lane`` or ``mixed``) and the level of traffic stress (``lts``, 1 to 4, see
    :func:`~iduedu.graph_builders.bike_builders.level_of_traffic_stress`).

    The graph is directed. One-way streets are one-way for bicycles too, unless the way allows riding against
    the traffic (``oneway:bicycle=no`` or a ``cycleway*=opposite*`` tag). Travel time is the length over
    ``bike_speed``; with ``elevation`` it also depends on the grade.

    Parameters:
        osm_id (int | None): OSM relation/area ID of the territory boundary. Provide this or ``territory``.
        territory (Polygon | MultiPolygon | gpd.GeoDataFrame | None): Boundary geometry in EPSG:4326 or GeoDataFrame.
        simplify (bool): If True, merges the segments of the ways into edges that run from junction to
            junction; segments are merged only when they carry the same tags.
        clip_by_territory (bool): If True, clips edges by the exact boundary geometry before graph construction.
        keep_largest_subgraph (bool): If True, returns only the largest strongly connected component.
        bike_speed (float): Cycling speed on flat ground in meters per minute.
        max_lts (int | None): Keep only edges with a level of traffic stress up to this one, for example 2 for
            the network that most adults are willing to ride.
        allow_footways (bool): If True, footways and pedestrian streets are used also without an explicit
            ``bicycle`` permission.
        elevation: Source of heights for :func:`~iduedu.add_elevation` (a raster path, a GeoDataFrame of points
            or contours, or a function). When given, edges get grades, and travel times are rescaled with the
            cycling speed model of :func:`~iduedu.apply_grade_speed`.
        network_type: ``"bike"``, or ``"custom"`` together with ``custom_filter``.
        custom_filter (str | None): Custom Overpass way filter used in custom mode.
        osm_edge_tags (list[str] | None): Which OSM tags to retain on edges. Overrides
            ``config.bike_useful_edges_attr`` (``highway``, ``name``, ``lanes``, ``maxspeed``, ``surface``).
            ``bridge`` and ``tunnel`` are always kept. The ``cycleway*``, ``bicycle``, ``oneway`` and
            ``oneway:bicycle`` tags are always read, and with ``simplify`` a change of any of them, or of a kept
            tag, starts a new edge.

    Returns:
        UrbanGraph: Directed multigraph of the cycling network. Edges include geometry in the local CRS,
        ``length_meter``, ``time_min``, ``oneway``, ``cycle_infra``, ``lts``, the selected OSM tags and, with
        ``elevation``, ``grade``, ``grade_max``, ``ascent`` and ``descent``.

    Raises:
        ValueError: If ``max_lts`` is not between 1 and 4, or the network type is unknown.
    """
    if max_lts is not None and max_lts not in (1, 2, 3, 4):
        raise ValueError("max_lts must be 1, 2, 3, 4 or None")
    polygon4326 = get_4326_boundary(osm_id=osm_id, territory=territory)
    filters = {"bike": Network.BIKE.filter, "custom": custom_filter}
    try:
        road_filter = filters[network_type]
    except KeyError:
        raise ValueError(f"Unknown road_type: {network_type!r}")
    if network_type == "custom" and road_filter is None:
        raise ValueError("For road_type='custom' you must provide custom_filter")

    needed_tags = set(config.bike_useful_edges_attr if osm_edge_tags is None else osm_edge_tags)
    tags_to_retrieve = needed_tags | set(_CYCLE_TAGS) | {"highway", "maxspeed", "lanes"}
    local_crs = estimate_crs_for_bounds(*polygon4326.bounds)

    logger.info(f"Retrieving bike network via {osm_source_label()}...")
    edges = _build_edges_from_overpass(
        polygon4326, road_filter, needed_tags=tags_to_retrieve, local_crs=local_crs, simplify=simplify
    )
    if len(edges) == 0:
        logger.warning("No edges found, returning empty graph")
        return _empty(local_crs, network_type)

    highway, bicycle = _text(edges, "highway"), _text(edges, "bicycle")
    usable = bicycle != "no"
    if not allow_footways:
        usable &= ~highway.isin(FOOTWAYS) | bicycle.isin(_PERMITTED)
    edges = edges[usable.to_numpy()].reset_index(drop=True)
    if clip_by_territory and len(edges):
        clip_poly_gdf = gpd.GeoDataFrame(geometry=[polygon4326], crs=4326).to_crs(local_crs)
        edges = edges.clip(clip_poly_gdf, keep_geom_type=True).explode(ignore_index=True)
    edges["cycle_infra"] = cycle_infrastructure(edges)
    edges["lts"] = level_of_traffic_stress(edges)
    if max_lts is not None:
        edges = edges[edges["lts"] <= max_lts].reset_index(drop=True)
    if len(edges) == 0:
        logger.warning("No edges left for bicycles, returning empty graph")
        return _empty(local_crs, network_type)

    # One-way for bicycles: the direction of the traffic, unless riding against it is allowed.
    oneway = _text(edges, "oneway")
    contraflow = (_text(edges, "oneway:bicycle") == "no").to_numpy()
    for column in ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both"):
        contraflow |= _text(edges, column).str.startswith("opposite").to_numpy()
    reverse_mask = oneway.eq("-1").to_numpy() & ~contraflow
    if reverse_mask.any():
        edges.loc[reverse_mask, "geometry"] = edges.loc[reverse_mask, "geometry"].apply(
            lambda geom: LineString(list(geom.coords)[::-1])
        )
    edges["oneway"] = oneway.isin({"yes", "true", "1", "-1"}).to_numpy() & ~contraflow

    nodes, edges = _build_nodes_and_uv(edges, local_crs)
    edges["length_meter"] = edges.geometry.length.round(3)
    edges["time_min"] = (edges["length_meter"] / float(bike_speed)).round(3)
    edges["type"] = "bike"
    edges = _assign_edge_keys(edges)
    keep = needed_tags | set(STRUCTURE_TAGS) | {"u", "v", "k", "geometry", "length_meter", "time_min", "type"}
    keep |= {"oneway", "cycle_infra", "lts"}
    edges = edges[[column for column in edges.columns if column in keep]].copy()

    graph = UrbanGraph(
        nodes_gdf=nodes,
        edges_gdf=edges,
        is_multigraph=True,
        is_directed=True,
        edge_direction_column="oneway",
        crs=local_crs,
        graph_type=network_type,
    )
    if keep_largest_subgraph:
        graph = keep_largest_connected_component(graph)
    if elevation is not None:
        graph = apply_grade_speed(add_elevation(graph, elevation), "cycling")
    logger.debug("Bike graph built.")
    return graph
