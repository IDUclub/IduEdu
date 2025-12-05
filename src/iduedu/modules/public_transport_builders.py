import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import LineString, MultiPolygon, Polygon
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from iduedu import config
from iduedu.enums.pt_enums import PublicTrasport
from iduedu.modules.overpass_parsers import parse_overpass_subway_data, parse_overpass_to_edgenode

from .graph_transformers import clip_nx_graph, estimate_crs_for_bounds
from .overpass_downloaders import (
    get_4326_boundary,
    get_routes_by_poly,
)

logger = config.logger


def _graph_data_to_nx(graph_df, keep_geometry: bool = True, additional_data=None) -> nx.DiGraph:
    """
    Build a directed public-transport graph from a mixed edge/node DataFrame, with optional subway add-ons.

    Input `graph_df` contains both node rows (non-null `node_id`) and edge rows (null `node_id`).
    Nodes are grouped/merged (platforms by rounded coords; other types by (coord, route, type, then by ref_id)),
    then edges are mapped to these merged nodes. If `additional_data` (subway entrances/transfers) is provided,
    edges with `u_ref`/`v_ref` are joined by `ref_id`.

    Parameters:
        graph_df (pd.DataFrame): Mixed table with:
            node rows → columns: `node_id`, `point` (projected meters), `route`, `type`, `ref_id`, `extra_data`;
            edge rows → columns: `u`, `v`, `type`, `extra_data`, `route`, `geometry`.
        keep_geometry (bool): If True, store shapely `geometry` on edges; otherwise drop it.
        additional_data (tuple | None): Optional `(additional_edges, additional_nodes)` produced by
            `parse_overpass_subway_data`. `additional_edges` must have `u_ref`, `v_ref`, `type`, optional
            `geometry`, `route`, `extra_data`; `additional_nodes` must have `ref_id`, `point`, `type`, `extra_data`.

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

    nodes_col = ["node_id", "point", "route", "type", "ref_id", "extra_data"]
    edges_col = ["u", "v", "type", "extra_data", "route", "geometry"]

    graph_nodes = graph_df[~graph_df["node_id"].isna()][nodes_col].copy()
    mask = graph_nodes["ref_id"].isna()
    graph_nodes["ref_id"] = graph_nodes["ref_id"].astype(int, errors="ignore").astype("string")
    new_ids = [f"new_node_{i}" for i in range(1, mask.sum() + 1)]
    graph_nodes.loc[mask, "ref_id"] = new_ids

    graph_edges = graph_df[graph_df["node_id"].isna()][edges_col].copy()

    if additional_data is not None:
        additional_edges, additional_nodes = additional_data
        additional_nodes["ref_id"] = additional_nodes["ref_id"].astype(int, errors="ignore").astype("string")
        additional_edges["v_ref"] = additional_edges["v_ref"].astype(int, errors="ignore").astype("string")
        additional_edges["u_ref"] = additional_edges["u_ref"].astype(int, errors="ignore").astype("string")
        graph_nodes_combined = graph_nodes.merge(additional_nodes, left_on="ref_id", right_on="ref_id", how="outer")
        for column in nodes_col:
            if column in graph_nodes_combined.columns:
                continue
            else:
                graph_nodes_combined[column] = graph_nodes_combined[f"{column}_y"].combine_first(
                    graph_nodes_combined[f"{column}_x"]
                )
        graph_nodes_combined = graph_nodes_combined[nodes_col]

        no_point_refs = graph_nodes_combined[
            (graph_nodes_combined["point"].isna()) & (~graph_nodes_combined["ref_id"].isna())
        ][["ref_id"]].copy()

        if len(no_point_refs) > 0:
            no_point_refs = no_point_refs.merge(
                additional_edges[["v_ref", "u_ref"]], left_on="ref_id", right_on="v_ref"
            )[["v_ref", "u_ref"]].drop_duplicates(subset="u_ref")
            no_point_refs = no_point_refs.merge(
                graph_nodes[["node_id", "type", "ref_id"]], left_on="u_ref", right_on="ref_id"
            ).drop(columns=["ref_id"])
            no_point_refs = no_point_refs.merge(
                graph_edges[["u", "v"]], left_on="node_id", right_on="u", how="left"
            ).drop(columns=["u"])
            no_point_refs = no_point_refs.merge(
                graph_nodes[["node_id", "type", "ref_id"]].add_prefix("potential_"),
                left_on="v",
                right_on="potential_node_id",
            )
            no_point_refs = no_point_refs[no_point_refs["potential_type"] == "platform"]
            remap_refs = no_point_refs.set_index("potential_ref_id")["v_ref"].to_dict()
            s = graph_nodes_combined["ref_id"].copy()
            mapped = s.map(remap_refs)
            mask = mapped.notna()
            graph_nodes_combined.loc[mask, "ref_id"] = mapped[mask]
            graph_nodes_combined.loc[mask, "type"] = "subway_platform"

        graph_nodes_combined["route"] = graph_nodes_combined["route"].fillna("subway_transit")
        graph_nodes = graph_nodes_combined.dropna(subset=["node_id", "point"], how="all").copy()

    platforms = graph_nodes[graph_nodes["type"] == "platform"].copy()
    platforms["point_group"] = platforms["point"].apply(lambda p: (round(p[0]), round(p[1])))
    platforms = platforms.groupby("point_group", as_index=False).agg(
        point=("point", "first"),
        node_id=("node_id", lambda s: tuple(s.dropna())),
        route=("route", lambda s: tuple(s)),
        ref_id=("ref_id", lambda s: tuple(s.dropna())),
        extra_data=("extra_data", merge_dicts_last),
    )
    platforms["type"] = "platform"

    not_platforms = graph_nodes[graph_nodes["type"] != "platform"].copy()
    not_platforms["point_group"] = not_platforms["point"].apply(lambda p: (round(p[0]), round(p[1])))
    not_platforms = not_platforms.groupby(["point_group", "route", "type"], as_index=False).agg(
        point=("point", "first"),
        node_id=("node_id", lambda s: tuple(s.dropna())),
        ref_id=("ref_id", lambda s: tuple(set(s.dropna()))),
        extra_data=("extra_data", merge_dicts_last),
    )
    not_platforms = not_platforms.groupby(["ref_id", "route", "type"], as_index=False).agg(
        point=("point", avg_point),
        node_id=("node_id", "sum"),
        extra_data=("extra_data", merge_dicts_last),
    )

    all_nodes = pd.concat([platforms, not_platforms], ignore_index=True)

    map_nodeid_to_idx = {}
    map_refid_to_idx = {}

    for idx, row in all_nodes.iterrows():
        for nid in row["node_id"] or []:
            map_nodeid_to_idx[nid] = idx
        rids = row["ref_id"]
        if not isinstance(rids, (list, tuple)):
            rids = [rids]
        for rid in rids or []:
            map_refid_to_idx[rid] = idx

    edges_existing = graph_edges[["route", "type", "u", "v", "geometry", "extra_data"]].copy()
    edges_existing["u"] = edges_existing["u"].map(map_nodeid_to_idx)
    edges_existing["v"] = edges_existing["v"].map(map_nodeid_to_idx)
    edges_existing = edges_existing.dropna(subset=["u", "v"]).copy()

    if additional_data is not None:
        additional_edges["u"] = additional_edges["u_ref"].map(map_refid_to_idx)
        additional_edges["v"] = additional_edges["v_ref"].map(map_refid_to_idx)
        additional_edges = additional_edges.dropna(subset=["u", "v"]).copy()

        def _ensure_geom(row):
            if row.get("geometry") is not None:
                return row["geometry"]
            pu = all_nodes.loc[int(row["u"]), "point"]
            pv = all_nodes.loc[int(row["v"]), "point"]
            return LineString([pu, pv])

        additional_edges["geometry"] = additional_edges.apply(_ensure_geom, axis=1)
    else:
        additional_edges = pd.DataFrame()

    edges = pd.concat([edges_existing, additional_edges], ignore_index=True)

    if "length_meter" not in edges.columns:
        edges["length_meter"] = np.nan
    if "time_min" not in edges.columns:
        edges["time_min"] = np.nan

    def calc_len_time(row):
        if row.type == "boarding":
            return 0.0, 0.0
        geom = row.geometry
        if geom is None:
            return 0.0, 0.0
        length = float(round(geom.length, 3))
        speed = PublicTrasport[row.type.upper()].avg_speed
        return length, float(round(length / speed, 3))

    mask_missing = edges["length_meter"].isna() | edges["time_min"].isna()

    vals = edges.loc[mask_missing].apply(calc_len_time, axis=1, result_type="expand")
    vals.columns = ["length_meter", "time_min"]
    edges.loc[mask_missing, ["length_meter", "time_min"]] = vals

    graph = nx.DiGraph()

    for idx, node in all_nodes.iterrows():
        route = list(set(node["route"])) if isinstance(node["route"], tuple) else [node["route"]]
        if len(route) == 1:
            route = route[0]
        graph.add_node(
            idx,
            x=float(node["point"][0]),
            y=float(node["point"][1]),
            type=node["type"],
            route=route,
            ref_id=(node["ref_id"][0] if isinstance(node["ref_id"], tuple) and node["ref_id"] else node["ref_id"]),
            **(node["extra_data"] if isinstance(node["extra_data"], dict) else {}),
        )

    for _, e in edges.iterrows():
        payload = {
            "route": e["route"],
            "type": e["type"],
            "length_meter": e["length_meter"],
            "time_min": e["time_min"],
            **(e["extra_data"] if isinstance(e["extra_data"], dict) else {}),
        }
        if keep_geometry and not pd.isna(e["geometry"]):
            payload["geometry"] = e["geometry"]
        graph.add_edge(
            int(e["u"]),
            int(e["v"]),
            **payload,
        )
    return graph


def _multi_parse_overpass_to_edgenode(args):
    return parse_overpass_to_edgenode(*args)


def _get_public_transport_graph(
    osm_id: int,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None,
    transport_types: list[str],
    osm_edge_tags: list[str],
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
    platform_stop_data_use = False
    if "subway" in transport_types and config.overpass_date is None:
        platform_stop_data_use = True

    ptts = ", ".join(transport_types)
    logger.info(f"Downloading routes via Overpass with types {ptts} ...")
    overpass_data = get_routes_by_poly(polygon, transport_types)
    logger.info(f"Downloading routes via Overpass with types {ptts} done!")

    if overpass_data.shape[0] == 0:
        logger.warning("No routes found for public transport.")
        return nx.Graph()

    overpass_data = overpass_data.dropna(subset=["transport_type"])

    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    # необходимые osm теги из relation маршрута
    if osm_edge_tags is None:
        needed_tags = set(config.transport_useful_edges_attr)
    else:
        needed_tags = set(osm_edge_tags)

    # Отделяем станции от маршрутов при необходимости
    if platform_stop_data_use:
        for column in ["is_stop_area", "is_stop_area_group", "is_station"]:
            overpass_data[column] = overpass_data[column].astype("boolean").fillna(False)
        routes_data = overpass_data[
            ~((overpass_data["is_stop_area"]) | (overpass_data["is_stop_area_group"]) | (overpass_data["is_station"]))
        ].copy()
        stop_areas = overpass_data[overpass_data["is_stop_area"]].copy()
        stop_areas_group = overpass_data[overpass_data["is_stop_area_group"]].copy()
        stations_data = overpass_data[overpass_data["is_station"]].copy()
        add_data = parse_overpass_subway_data(stop_areas, stop_areas_group, stations_data, local_crs)
    else:
        routes_data = overpass_data.copy()
        add_data = None

    if not config.enable_tqdm_bar:
        logger.debug("Parsing public transport routes")
    if overpass_data.shape[0] > 100:
        # Если много маршрутов - обрабатываем в параллели
        edgenode_for_routes = process_map(
            _multi_parse_overpass_to_edgenode,
            [(row, local_crs, needed_tags) for _, row in routes_data.iterrows()],
            desc="Parsing public transport routes",
            chunksize=1,
            disable=not config.enable_tqdm_bar,
        )
    else:
        tqdm.pandas(desc="Parsing public transport routes", disable=not config.enable_tqdm_bar)
        edgenode_for_routes = [
            data
            for data in routes_data.progress_apply(
                lambda x: parse_overpass_to_edgenode(x, local_crs, needed_tags), axis=1
            ).tolist()
            if data is not None
        ]

    if len(edgenode_for_routes) == 0:
        logger.warning("No routes were parsed for public transport.")
        return nx.DiGraph()
    graph_df = pd.concat(edgenode_for_routes, ignore_index=True)
    to_return = _graph_data_to_nx(graph_df, keep_geometry=keep_edge_geometry, additional_data=add_data)
    to_return.graph["crs"] = local_crs
    to_return.graph["type"] = "public_trasport"

    if clip_by_territory:
        polygon = gpd.GeoSeries([polygon], crs=4326).to_crs(local_crs).union_all()
        to_return = clip_nx_graph(to_return, polygon)

    to_return = nx.convert_node_labels_to_integers(to_return)
    logger.debug("Done!")
    return to_return


def get_single_public_transport_graph(
    public_transport_type: str | PublicTrasport,
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    clip_by_territory: bool = False,
    keep_edge_geometry: bool = True,
    osm_edge_tags: list[str] | None = None,  # overrides default tags
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

    Returns:
        (nx.DiGraph): Directed PT graph with:
            - node attrs: `x`, `y` (floats, local CRS), `type`, `route`, `ref_id`, plus merged station `extra_data` (if any);
            - edge attrs: `type`, `route`, `length_meter`, `time_min`, optional `geometry`, and selected OSM tags.

          Graph attrs: `graph["crs"]` (EPSG int of the local projected CRS), `graph["type"]` = `"public_trasport"`.

    Notes:
        - Lengths and times are computed in a **local projected CRS** estimated from the boundary; per-edge speeds are
          taken from mode-specific defaults (and, for subway connectors, from connector-type defaults).
    """
    public_transport_type = (
        public_transport_type.value() if isinstance(public_transport_type, PublicTrasport) else public_transport_type
    )
    return _get_public_transport_graph(
        osm_id=osm_id,
        territory=territory,
        transport_types=[public_transport_type],
        osm_edge_tags=osm_edge_tags,
        clip_by_territory=clip_by_territory,
        keep_edge_geometry=keep_edge_geometry,
    )


def get_all_public_transport_graph(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    clip_by_territory: bool = False,
    keep_edge_geometry: bool = True,
    transport_types: list[PublicTrasport] = None,
    osm_edge_tags: list[str] | None = None,  # overrides default tags
) -> nx.Graph:
    """
    Build a combined directed graph for multiple public-transport modes within a territory.

    The function collects routes for the requested modes (by default: **tram**, **bus**, **trolleybus**, **subway**),
    converts them into a single projected graph, and computes per-edge length and time. Edges from different modes
    coexist in the same `nx.DiGraph`; node ids are unified across modes. For the **subway** mode, station context
    is added (entrances/exits and inter-station transfers), and available station metadata is merged into node attrs.

    Parameters:
        osm_id (int | None): OSM relation/area id of the territory. Provide this or `territory`.
        territory (Polygon | MultiPolygon | gpd.GeoDataFrame | None): Boundary geometry in EPSG:4326 (or a GeoDataFrame).
        clip_by_territory (bool): If True, clip the final graph to the boundary (in the local CRS).
        keep_edge_geometry (bool): If True, retain `shapely` geometries on edges.
        transport_types (list[PublicTrasport] | None): List of modes to include. Defaults to
            `[PublicTrasport.TRAM, PublicTrasport.BUS, PublicTrasport.TROLLEYBUS, PublicTrasport.SUBWAY]`.
            All items must be `PublicTrasport` enums.
        osm_edge_tags (list[str] | None): Which OSM tags to keep on edges/nodes. If None, a default subset from
            configuration is used; only these keys are joined from OSM.

    Returns:
        (nx.DiGraph): Combined directed PT graph. Typical attributes:
            - node attrs: `x`, `y` (local CRS), `type`, `route`, `ref_id`, station `extra_data` where applicable;
            - edge attrs: `type`, `route`, `length_meter`, `time_min`, optional `geometry`, plus selected OSM tags.

          Graph attrs: `graph["crs"]` (EPSG int), `graph["type"]` = `"public_trasport"`.

    Notes:
        Each mode’s ways are downloaded inside the boundary and transformed into directed edges; per-edge speeds are
        taken from mode-specific defaults (and, for subway connectors, from connector-type defaults).
    """

    if transport_types is None:
        transport_types = [PublicTrasport.TRAM, PublicTrasport.BUS, PublicTrasport.TROLLEYBUS, PublicTrasport.SUBWAY]
    else:
        for transport_type in transport_types:
            if not isinstance(transport_type, PublicTrasport):
                raise ValueError(f"transport_type {transport_type} is not a valid transport type.")

    transports = [transport.value for transport in transport_types if isinstance(transport, PublicTrasport)]

    return _get_public_transport_graph(
        osm_id=osm_id,
        territory=territory,
        transport_types=transports,
        osm_edge_tags=osm_edge_tags,
        clip_by_territory=clip_by_territory,
        keep_edge_geometry=keep_edge_geometry,
    )
