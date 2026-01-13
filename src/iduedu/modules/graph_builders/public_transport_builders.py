import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import MultiPolygon, Polygon
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

    def merge_dicts_last(dicts):
        out = {}
        for d in dicts.dropna():
            for k, v in d.items():
                out[k] = v
        return out

    if "extra_data" not in graph_nodes_df.columns:
        graph_nodes_df["extra_data"] = [{} for _ in range(len(graph_nodes_df))]

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


def _build_public_transport_graph(
    osm_id: int | None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None,
    transport_types: list[str],
    osm_edge_tags: list[str] | None,
    transport_registry: TransportRegistry,
    clip_by_territory: bool = False,
    keep_edge_geometry: bool = True,
):
    """
    Build a directed public-transport graph for one or multiple OSM public-transport modes inside a territory.

    Pipeline:
      1) Resolve a boundary polygon (EPSG:4326) from ``osm_id`` and/or ``territory``.
      2) Download PT routes via Overpass for the requested ``transport_types``.
      3) Optionally (subway) parse station/stop-area context (entrances, exits, transfers).
      4) Parse each route into node/edge tables (parallelized for large inputs).
      5) Assemble a single ``nx.DiGraph`` via ``_graph_data_to_nx`` and compute missing edge ``length_meter`` and
         ``time_min`` using ``transport_registry.rst``.
      6) Optionally clip the graph by the territory boundary (in the projected CRS).

    Parameters:
        osm_id:
            OSM relation/area id of the territory. Used if ``territory`` is not provided.
        territory:
            Boundary geometry in EPSG:4326 (or a GeoDataFrame). Used when ``osm_id`` is not given.
        transport_types:
            List of OSM public-transport route types to include (already normalized), e.g. ``["bus", "tram", "subway"]``.
        osm_edge_tags:
            Optional list of OSM tag keys to retain on edges/nodes. If None, defaults from configuration are used.
        transport_registry:
            Registry used to compute per-edge travel time (minutes) based on mode parameters (max speed, accel/brake
            distances, traffic coefficient). Also used for transport-type validation in public APIs.
        clip_by_territory:
            If True, clip the final graph to the (projected) boundary.
        keep_edge_geometry:
            If True, store ``shapely`` edge geometries (in the local projected CRS) on graph edges.

    Returns:
        ``nx.DiGraph``: Directed PT graph. Graph attributes set by this function:
          - ``graph["crs"]``: EPSG integer of the local projected CRS
          - ``graph["type"]``: ``"public_transport"``
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

    ground_types = {"bus", "tram", "trolleybus"}
    ground_pt_data = overpass_data[
        (overpass_data["transport_type"].isin(ground_types)) & (~overpass_data["is_way_data"])
    ].copy()

    if len(ground_pt_data) > 0:
        if not config.enable_tqdm_bar:
            logger.debug("Parsing ground public transport routes")

        if len(ground_pt_data) > 500:
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
                graph_edges_df.append(edges_df.dropna(axis=1, how="all"))
            if len(nodes_df) > 0:
                graph_nodes_df.append(nodes_df.dropna(axis=1, how="all"))

    if expect_subway:
        subway_data = overpass_data[overpass_data["transport_type"] == "subway"].copy()
        if len(subway_data) > 0:
            subway_edges, subway_nodes = overpass_subway2edgenode(subway_data, local_crs)
            if len(subway_edges) > 0:
                graph_edges_df.append(subway_edges.dropna(axis=1, how="all"))
            if len(subway_nodes) > 0:
                graph_nodes_df.append(subway_nodes.dropna(axis=1, how="all"))

    if not graph_edges_df and not graph_nodes_df:
        logger.warning("No routes were parsed for public transport.")
        return nx.DiGraph()

    graph_edges_df = pd.concat(graph_edges_df, ignore_index=True) if graph_edges_df else pd.DataFrame()
    graph_nodes_df = pd.concat(graph_nodes_df, ignore_index=True) if graph_nodes_df else pd.DataFrame()

    nx_graph = _graph_data_to_nx(
        graph_nodes_df, graph_edges_df, transport_registry=transport_registry, keep_geometry=keep_edge_geometry
    )
    nx_graph.graph["crs"] = local_crs
    nx_graph.graph["type"] = "public_transport"

    if clip_by_territory:
        poly_proj = gpd.GeoSeries([polygon], crs=4326).to_crs(local_crs).union_all()
        nx_graph = clip_nx_graph(nx_graph, poly_proj)

    nx_graph = nx.convert_node_labels_to_integers(nx_graph)
    logger.debug("Done!")
    return nx_graph


def get_public_transport_graph(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    transport_types: str | list[str] | None = None,
    clip_by_territory: bool = False,
    keep_edge_geometry: bool = True,
    osm_edge_tags: list[str] | None = None,
    transport_registry: TransportRegistry | None = None,
) -> nx.Graph:
    """
    Build a directed public-transport graph for one or multiple transport modes within a territory.

    The function resolves a boundary (by ``osm_id`` or ``territory``), downloads OpenStreetMap public-transport routes
    inside that boundary, converts them into a projected ``nx.DiGraph``, and computes per-edge length (meters) and
    travel time (minutes). Multiple modes can coexist in the same graph; node ids are unified across modes.

    For the ``subway`` mode, additional station context may be added (entrances/exits and inter-station transfers),
    and available station metadata may be merged into node attributes.

    Parameters:
        osm_id:
            OSM relation/area id of the territory. Provide this or ``territory``.
        territory:
            Boundary geometry in EPSG:4326 (or a GeoDataFrame). Used when ``osm_id`` is not given.
        transport_types:
            Transport mode(s) to include. Accepts:
              - ``None``: include all types available in ``transport_registry.rst``;
              - ``str``: a single OSM route type, e.g. ``"bus"``;
              - ``Sequence[str]``: multiple types, e.g. ``["tram", "bus", "trolleybus", "subway"]``.

            Values are normalized with ``strip().lower()`` and validated against the registry.
        clip_by_territory:
            If True, clip the resulting graph to the boundary (in the local CRS).
        keep_edge_geometry:
            If True, store ``shapely`` geometries (in local CRS) on edges.
        osm_edge_tags:
            Subset of OSM tags to retain on edges/nodes. If None, a default subset is used.
        transport_registry:
            Transport registry used to validate transport types and to compute per-edge travel times (via each mode's
            parameters such as max speed, acceleration/braking distances, and traffic coefficient).
            If None, ``DEFAULT_REGISTRY`` is used.

    Returns:
        Directed PT graph (``nx.DiGraph``) with:
            - node attrs: ``x``, ``y`` (local CRS), ``type``, ``route``, ``ref_id``, plus optional station metadata;
            - edge attrs: ``type``, ``route``, ``length_meter``, ``time_min``, optional ``geometry``, and selected OSM tags.

        Graph attrs typically include: ``graph["crs"]`` (EPSG int), ``graph["type"]`` = ``"public_transport"``.
    """
    registry = transport_registry or DEFAULT_REGISTRY
    registry_types = set(registry.list_types())

    # normalize transport_types -> list[str]
    if transport_types is None:
        types = list(registry_types)
    elif isinstance(transport_types, str):
        types = [transport_types.strip().lower()]
    else:
        types = [str(t).strip().lower() for t in transport_types]

    unknown = [t for t in types if t not in registry_types]
    if unknown:
        raise ValueError(f"Unknown transport type(s): {unknown}. Available: {sorted(registry_types)}")

    return _build_public_transport_graph(
        osm_id=osm_id,
        territory=territory,
        transport_types=types,
        osm_edge_tags=osm_edge_tags,
        transport_registry=registry,
        clip_by_territory=clip_by_territory,
        keep_edge_geometry=keep_edge_geometry,
    )


def get_single_public_transport_graph(
    public_transport_type: str,
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    clip_by_territory: bool = False,
    keep_edge_geometry: bool = True,
    osm_edge_tags: list[str] | None = None,
    transport_registry: TransportRegistry | None = None,
) -> nx.Graph:  # pragma: no cover
    """
    Deprecated wrapper for ``get_public_transport_graph``.

    This function will be removed in a future release.
    Use ``get_public_transport_graph(transport_types="...")`` instead.
    """
    warnings.warn(
        "get_single_public_transport_graph() is deprecated and will be removed in the next release. "
        "Use get_public_transport_graph(transport_types='...') instead.",
        FutureWarning,
        stacklevel=2,
    )
    return get_public_transport_graph(
        osm_id=osm_id,
        territory=territory,
        transport_types=str(public_transport_type),
        clip_by_territory=clip_by_territory,
        keep_edge_geometry=keep_edge_geometry,
        osm_edge_tags=osm_edge_tags,
        transport_registry=transport_registry,
    )


def get_all_public_transport_graph(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    clip_by_territory: bool = False,
    keep_edge_geometry: bool = True,
    transport_types: list[str] | None = None,
    osm_edge_tags: list[str] | None = None,
    transport_registry: TransportRegistry | None = None,
) -> nx.Graph:  # pragma: no cover
    """
    Deprecated wrapper for ``get_public_transport_graph``.

    This function will be removed in a future release.
    Use ``get_public_transport_graph(transport_types=[...])`` instead.
    """
    warnings.warn(
        "get_all_public_transport_graph() is deprecated and will be removed in the next release. "
        "Use get_public_transport_graph(transport_types=[...]) instead.",
        FutureWarning,
        stacklevel=2,
    )
    return get_public_transport_graph(
        osm_id=osm_id,
        territory=territory,
        transport_types=transport_types,  # None or list[str]
        clip_by_territory=clip_by_territory,
        keep_edge_geometry=keep_edge_geometry,
        osm_edge_tags=osm_edge_tags,
        transport_registry=transport_registry,
    )
