import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import LineString, MultiPolygon, Polygon
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from iduedu import config
from iduedu.constants.transport_specs import DEFAULT_REGISTRY, TransportRegistry
from iduedu.modules.graph_transformers import clip_nx_graph, estimate_crs_for_bounds
from iduedu.modules.overpass.overpass_downloaders import (
    get_4326_boundary,
    get_routes_by_poly,
)
from iduedu.modules.overpass.overpass_parsers import (
    overpass_ground_transport2edgenode,
    overpass_routes_to_df,
    overpass_subway2edgenode,
)

logger = config.logger


def _graph_data_to_nx(
    graph_nodes_df, graph_edges_df, transport_registry: TransportRegistry, keep_geometry: bool = True
) -> nx.DiGraph:
    """
    Build a directed public-transport graph from a mixed edge/node DataFrame, with optional subway add-ons.

    Input `graph_df` contains both node rows (non-null `node_id`) and edge rows (null `node_id`).
    Nodes are grouped/merged (platforms by rounded coords; other types by (coord, route, type, then by ref_id)),
    then edges are mapped to these merged nodes. If `additional_data` (subway entrances/transfers) is provided,
    edges with `u_ref`/`v_ref` are joined by `ref_id`.

    Parameters:
        graph_nodes_df (pd.DataFrame): columns: `node_id`, `point` (projected meters), `route`, `type`, `ref_id`, `extra_data`;
        graph_edges_df (pd.DataFrame): columns: `u`, `v`, `type`, `extra_data`, `route`, `geometry`.
        keep_geometry (bool): If True, store shapely `geometry` on edges; otherwise drop it.

    Returns:
        (nx.DiGraph): Directed graph
    """

    def avg_point(points):
        arr = np.asarray([p for p in points if p is not None], dtype=float)
        return tuple(arr.mean(axis=0)) if len(arr) else None

    def merge_dicts_last(dicts):
        out = {}
        for d in dicts.dropna():
            for k, v in d.items():
                out[k] = v
        return out

    platforms = graph_nodes_df[graph_nodes_df["type"] == "platform"].copy()
    platforms["point_group"] = platforms["point"].apply(lambda p: (round(p[0]), round(p[1])))
    platforms = platforms.groupby("point_group", as_index=False).agg(
        point=("point", "first"),
        node_id=("node_id", lambda s: tuple(s.dropna())),
        route=("route", lambda s: tuple(s)),
        extra_data=("extra_data", merge_dicts_last),
    )
    platforms["type"] = "platform"

    not_platforms = graph_nodes_df[graph_nodes_df["type"] != "platform"].copy()
    not_platforms["point_group"] = not_platforms["point"].apply(lambda p: (round(p[0]), round(p[1])))
    not_platforms = not_platforms.groupby(["point_group", "route", "type"], as_index=False, dropna=False).agg(
        point=("point", "first"),
        node_id=("node_id", lambda s: tuple(s.dropna())),
        extra_data=("extra_data", merge_dicts_last),
    )

    all_nodes = pd.concat([platforms, not_platforms], ignore_index=True)

    map_nodeid_to_idx = {}

    for idx, row in all_nodes.iterrows():
        for nid in row["node_id"] or []:
            map_nodeid_to_idx[nid] = idx

    graph_edges_df["u"] = graph_edges_df["u"].map(map_nodeid_to_idx)
    graph_edges_df["v"] = graph_edges_df["v"].map(map_nodeid_to_idx)

    if "length_meter" not in graph_edges_df.columns:
        graph_edges_df["length_meter"] = np.nan
    if "time_min" not in graph_edges_df.columns:
        graph_edges_df["time_min"] = np.nan
    if "speed_m_min" not in graph_edges_df.columns:
        graph_edges_df["speed_m_min"] = np.nan

    def calc_len_time(row):
        if row.type == "boarding":
            return 0.0, 0.0
        geom = row.geometry
        if geom is None:
            return 0.0, 0.0
        length_m = float(round(geom.length, 3))
        spec = transport_registry.get(str(row.type))
        speed_limit_mpm = row.speed_m_min

        time_min = spec.travel_time_min(
            length_m,
            speed_limit_mpm=speed_limit_mpm,
        )
        time_min = float(round(time_min, 3))

        return length_m, time_min

    mask_missing = graph_edges_df["length_meter"].isna() | graph_edges_df["time_min"].isna()

    vals = graph_edges_df.loc[mask_missing].apply(calc_len_time, axis=1, result_type="expand")
    vals.columns = ["length_meter", "time_min"]
    graph_edges_df.loc[mask_missing, ["length_meter", "time_min"]] = vals

    graph = nx.DiGraph()

    for idx, node in all_nodes.iterrows():
        route = list(set(node["route"])) if isinstance(node["route"], tuple) else [node["route"]]
        if len(route) == 1:
            route = route[0]
        node_extra = node.get("extra_data")
        graph.add_node(
            idx,
            x=float(node["point"][0]),
            y=float(node["point"][1]),
            type=node["type"],
            route=route,
            **(node_extra if isinstance(node_extra, dict) else {}),
        )

    for _, e in graph_edges_df.iterrows():
        edge_extra = e.get("extra_data")
        payload = {
            "route": e["route"],
            "type": e["type"],
            "length_meter": e["length_meter"],
            "time_min": e["time_min"],
            **(edge_extra if isinstance(edge_extra, dict) else {}),
        }
        if keep_geometry and not pd.isna(e["geometry"]):
            payload["geometry"] = e["geometry"]
        graph.add_edge(
            int(e["u"]),
            int(e["v"]),
            **payload,
        )
    return graph


def _multi_ground_to_edgenode(args):
    # args: (row, local_crs, ref2speed, needed_tags)
    return overpass_ground_transport2edgenode(*args)


def _get_public_transport_graph(
    osm_id: int,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None,
    transport_types: list[str],
    osm_edge_tags: list[str],
    transport_registry: TransportRegistry,
    clip_by_territory: bool = False,
    keep_edge_geometry: bool = True,
):
    """
    Build a directed public-transport graph for one or several transport types inside a territory.

    The function:
    1) resolves a boundary polygon (EPSG:4326);
    2) downloads routes via Overpass in parallel;
    3) (for subway) additionally parses stop areas/groups and stations, producing entrances/transfers;
    4) parses each route into edge/node rows (parallel for large inputs);
    5) assembles a single `nx.DiGraph` via `_graph_data_to_nx`, computing missing edge length/time;
    6) optionally clips the graph by the territory.

    Parameters:
        osm_id (int): OSM relation/area id of the territory; used if `territory` is not provided.
        territory (Polygon | MultiPolygon | gpd.GeoDataFrame | None): Boundary geometry in EPSG:4326.
        transport_types (list[str]): Transport types to include (e.g., `["bus", "tram", "subway"]`).
        osm_edge_tags (list[str]): Which route/member tags to retain on edges/nodes (overrides defaults).
        clip_by_territory (bool): If True, clip the final graph to the (projected) boundary.
        keep_edge_geometry (bool): If True, store shapely `geometry` on edges.

    Returns:
        (nx.DiGraph): Directed PT graph. Graph attributes set by this function:
            - `graph["crs"]` (int/EPSG of the local projected CRS),
            - `graph["type"]` = "public_trasport" (sic).
    """

    polygon = get_4326_boundary(osm_id=osm_id, territory=territory)

    # Если парсим метро - ожидаем в ответе информацию о станциях
    expect_subway = False
    if "subway" in transport_types and config.overpass_date is None:
        expect_subway = True

    ptts = ", ".join(transport_types)
    logger.info(f"Downloading routes via Overpass with types {ptts} ...")
    overpass_response: list[dict] = get_routes_by_poly(polygon, transport_types)
    overpass_data = overpass_routes_to_df(overpass_response, enable_subway_details=expect_subway)
    logger.info(f"Downloading routes via Overpass with types {ptts} done!")

    if overpass_data.shape[0] == 0:
        logger.warning("No routes found for public transport.")
        return nx.Graph()

    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    # необходимые osm теги из relation маршрута
    needed_tags = set(config.transport_useful_edges_attr) if osm_edge_tags is None else set(osm_edge_tags)

    way_data = overpass_data[overpass_data["is_way_data"]].copy()
    if len(way_data) > 0:
        ref2speed = (
            way_data[["id", "way_speed_m_per_min"]]
            .dropna(subset=["id"])
            .set_index("id")["way_speed_m_per_min"]
            .to_dict()
        )
    else:
        ref2speed = {}

    graph_edges_df = []
    graph_nodes_df = []

    ground_types = {"bus", "tram", "trolley"}
    ground_pt_data = overpass_data[overpass_data["transport_type"].isin(ground_types)].copy()

    if len(ground_pt_data) > 0:
        if not config.enable_tqdm_bar:
            logger.debug("Parsing ground public transport routes")

        if len(ground_pt_data) > 100:
            results = process_map(
                _multi_ground_to_edgenode,
                [(row, local_crs, ref2speed, needed_tags) for _, row in ground_pt_data.iterrows()],
                desc="Parsing ground PT routes",
                chunksize=1,
                disable=not config.enable_tqdm_bar,
            )
        else:
            tqdm.pandas(desc="Parsing ground PT routes", disable=not config.enable_tqdm_bar)
            results = ground_pt_data.progress_apply(
                lambda row: overpass_ground_transport2edgenode(row, local_crs, ref2speed, needed_tags),
                axis=1,
            ).tolist()

        for edges_df, nodes_df in results:
            if len(edges_df) > 0:
                graph_edges_df.append(edges_df)
            if len(nodes_df) > 0:
                graph_nodes_df.append(nodes_df)

    if expect_subway:
        subway_data = overpass_data[overpass_data["transport_type"] == "subway"].copy()
        if len(subway_data) > 0:
            subway_edges, subway_nodes = overpass_subway2edgenode(subway_data, local_crs)
            if len(subway_edges) > 0:
                graph_edges_df.append(subway_edges)
            if len(subway_nodes) > 0:
                graph_nodes_df.append(subway_nodes)

    if not graph_edges_df and not graph_nodes_df:
        logger.warning("No routes were parsed for public transport.")
        return nx.DiGraph()

    graph_edges_df = pd.concat(graph_edges_df, ignore_index=True) if graph_edges_df else pd.DataFrame()
    graph_nodes_df = pd.concat(graph_nodes_df, ignore_index=True) if graph_nodes_df else pd.DataFrame()

    nx_graph = _graph_data_to_nx(
        graph_nodes_df, graph_edges_df, transport_registry=transport_registry, keep_geometry=keep_edge_geometry
    )
    nx_graph.graph["crs"] = local_crs
    nx_graph.graph["type"] = "public_trasport"

    if clip_by_territory:
        poly_proj = gpd.GeoSeries([polygon], crs=4326).to_crs(local_crs).union_all()
        nx_graph = clip_nx_graph(nx_graph, poly_proj)

    nx_graph = nx.convert_node_labels_to_integers(nx_graph)
    logger.debug("Done!")
    return nx_graph


def get_single_public_transport_graph(
    public_transport_type: str,
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    clip_by_territory: bool = False,
    keep_edge_geometry: bool = True,
    osm_edge_tags: list[str] | None = None,  # overrides default tags
    transport_registry: TransportRegistry | None = None,
):
    """
    Build a directed graph for a single public-transport mode within a given territory.

    The function resolves a boundary (by `osm_id` or `territory`), downloads OpenStreetMap routes  inside that boundary,
    converts them into a projected `nx.DiGraph`, and computes per-edge length (meters) and travel time (minutes).
    When the mode is **subway**, additional station context is incorporated: entrances/exits and inter-station
    transfers are added; station metadata (e.g., name, depth) is attached to nodes where available.

    Parameters:
        public_transport_type (str | PublicTrasport): One mode, e.g. `"bus"`, `"tram"`, `"trolleybus"`, `"subway"`.
            You may pass a `PublicTrasport` enum or its string value.
        osm_id (int | None): OSM relation/area id of the territory. Provide this or `territory`.
        territory (Polygon | MultiPolygon | gpd.GeoDataFrame | None): Boundary geometry in EPSG:4326 (or a GeoDataFrame).
            Used when `osm_id` is not given.
        clip_by_territory (bool): If True, the resulting graph is clipped to the boundary (in the local CRS).
        keep_edge_geometry (bool): If True, edge shapes (`shapely` geometries in local CRS) are stored on edges.
        osm_edge_tags (list[str] | None): Subset of OSM tags to retain on edges/nodes. If None, a sensible default
            is used from configuration; only requested keys are joined from OSM element tags.
        transport_registry (TransportRegistry | None): Transport registry used to validate available transport types and
            to compute per-edge travel times (via each mode's parameters such as max speed, acceleration/braking
            distances, and traffic coefficient). If None, ``DEFAULT_REGISTRY`` is used.

    Returns:
        (nx.DiGraph): Directed PT graph with:
            - node attrs: `x`, `y` (floats, local CRS), `type`, `route`, `ref_id`, plus merged station `extra_data` (if any);
            - edge attrs: `type`, `route`, `length_meter`, `time_min`, optional `geometry`, and selected OSM tags.

          Graph attrs: `graph["crs"]` (EPSG int of the local projected CRS), `graph["type"]` = `"public_trasport"`.

    Notes:
        - Lengths and times are computed in a **local projected CRS** estimated from the boundary; per-edge speeds are
          taken from mode-specific defaults (and, for subway connectors, from connector-type defaults).
    """
    registry = transport_registry or DEFAULT_REGISTRY

    pt = str(public_transport_type).strip().lower()
    registry_types = set(registry.list_types() if hasattr(registry, "list_types") else registry.type_list())

    if pt not in registry_types:
        raise ValueError(f"Unknown transport type: {pt!r}. Available: {sorted(registry_types)}")

    return _get_public_transport_graph(
        osm_id=osm_id,
        territory=territory,
        transport_types=[pt],
        osm_edge_tags=osm_edge_tags,
        transport_registry=registry,
        clip_by_territory=clip_by_territory,
        keep_edge_geometry=keep_edge_geometry,
    )


def get_all_public_transport_graph(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    clip_by_territory: bool = False,
    keep_edge_geometry: bool = True,
    transport_types: list[str] = None,
    osm_edge_tags: list[str] | None = None,  # overrides default tags
    transport_registry: TransportRegistry | None = None,
) -> nx.Graph:
    """
    Build a combined directed graph for multiple public-transport modes within a territory.

    The function collects routes for the requested modes (e.g. ``bus``, ``tram``, ``trolleybus``, ``subway``),
    converts them into a single projected graph, and computes per-edge length and travel time. Edges from different
    modes coexist in the same ``nx.DiGraph``; node ids are unified across modes. For the subway mode, station context
    may be added (entrances/exits and inter-station transfers), and available station metadata is merged into node
    attributes.

    Parameters:
        osm_id (int | None): OSM relation/area id of the territory. Provide this or `territory`.
        territory (Polygon | MultiPolygon | gpd.GeoDataFrame | None): Boundary geometry in EPSG:4326 (or a GeoDataFrame).
        clip_by_territory (bool): If True, clip the final graph to the boundary (in the local CRS).
        keep_edge_geometry (bool): If True, retain `shapely` geometries on edges.
        transport_types (list[str] | None): List of OSM public-transport route types to include (strings), e.g.
            ``["tram", "bus", "trolleybus", "subway"]``. If None, defaults to all types available in the registry.
        osm_edge_tags (list[str] | None): Which OSM tags to keep on edges/nodes. If None, a default subset from
            configuration is used; only these keys are joined from OSM.
        transport_registry (TransportRegistry | None): Transport registry used to validate available transport types and
            to compute per-edge travel times (via each mode's parameters such as max speed, acceleration/braking
            distances, and traffic coefficient). If None, ``DEFAULT_REGISTRY`` is used.
    Returns:
        (nx.DiGraph): Combined directed PT graph. Typical attributes:

            - node attrs: `x`, `y` (local CRS), `type`, `route`, `ref_id`, station `extra_data` where applicable;
            - edge attrs: `type`, `route`, `length_meter`, `time_min`, optional `geometry`, plus selected OSM tags.
            Graph attrs: `graph["crs"]` (EPSG int), `graph["type"]` = `"public_trasport"`.

    Notes:
        Each mode’s ways are downloaded inside the boundary and transformed into directed edges. Per-edge travel time
        is derived from the transport registry settings and (optionally) road speed limits when available.
    """

    registry = transport_registry or DEFAULT_REGISTRY
    registry_types = set(registry.list_types())

    if transport_types is None:
        transport_types = list(registry_types)
    else:
        transport_types = [t.strip().lower() for t in transport_types]

    unknown = [t for t in transport_types if t not in registry_types]
    if unknown:
        raise ValueError(f"Unknown transport type(s): {unknown}. Available: {sorted(registry_types)}")

    return _get_public_transport_graph(
        osm_id=osm_id,
        territory=territory,
        transport_types=transport_types,
        osm_edge_tags=osm_edge_tags,
        transport_registry=registry,
        clip_by_territory=clip_by_territory,
        keep_edge_geometry=keep_edge_geometry,
    )
