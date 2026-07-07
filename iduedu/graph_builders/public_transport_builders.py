import multiprocessing

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import MultiPolygon, Polygon
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from iduedu import config
from iduedu.constants.transport_specs import DEFAULT_REGISTRY, TransportRegistry
from iduedu.graph.transformers import estimate_crs_for_bounds
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.overpass.downloaders import (
    get_4326_boundary,
    get_routes_by_poly,
)
from iduedu.overpass.parsers import (
    overpass_ground_transport2edgenode,
    overpass_routes_to_df,
    overpass_subway2edgenode,
)

logger = config.logger


def _merge_dicts_last(dicts):
    out = {}
    for d in dicts.dropna():
        for k, v in d.items():
            out[k] = v
    return out


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _empty_public_transport_graph(crs) -> UrbanGraph:
    return UrbanGraph.empty(
        crs=crs,
        is_multigraph=True,
        is_directed=True,
        edge_direction_column="oneway",
        graph_type="public_transport",
    )


def _graph_data_to_urban_graph(
    graph_nodes_gdf: gpd.GeoDataFrame,
    graph_edges_gdf: gpd.GeoDataFrame,
    transport_registry: TransportRegistry,
    local_crs,
    avg_boarding_time_min: float,
) -> UrbanGraph:
    """
    Build a directed public-transport UrbanGraph from parser node/edge tables.
    """
    if graph_nodes_gdf.empty:
        return _empty_public_transport_graph(local_crs)

    graph_nodes_gdf = gpd.GeoDataFrame(graph_nodes_gdf, geometry="geometry", crs=local_crs)
    missing_node_geometry = graph_nodes_gdf.geometry.isna()
    if missing_node_geometry.any():
        logger.warning(f"Dropping {int(missing_node_geometry.sum())} PT nodes with missing geometry")
        graph_nodes_gdf = graph_nodes_gdf.loc[~missing_node_geometry].copy()
    if graph_nodes_gdf.empty:
        return _empty_public_transport_graph(local_crs)
    graph_nodes_gdf["_x_group"] = graph_nodes_gdf.geometry.x.round()
    graph_nodes_gdf["_y_group"] = graph_nodes_gdf.geometry.y.round()

    for column in ["node_id", "type", "route"]:
        if column not in graph_nodes_gdf.columns:
            graph_nodes_gdf[column] = np.nan

    graph_nodes_gdf["_source_node_id"] = graph_nodes_gdf["node_id"]
    graph_nodes_gdf["_geometry_key"] = graph_nodes_gdf.geometry.to_wkb()
    duplicate_nodes = graph_nodes_gdf.duplicated(
        subset=["_source_node_id", "_geometry_key", "route", "type"], keep="first"
    )
    if duplicate_nodes.any():
        graph_nodes_gdf = graph_nodes_gdf.loc[~duplicate_nodes].copy()

    node_lookup = graph_nodes_gdf[["_source_node_id", "route"]].copy()
    node_lookup["node_id"] = np.arange(len(graph_nodes_gdf), dtype=np.int64)
    graph_nodes_gdf["node_id"] = node_lookup["node_id"].to_numpy()

    if not graph_edges_gdf.empty:
        for column in ["u", "v", "route"]:
            if column not in graph_edges_gdf.columns:
                graph_edges_gdf[column] = np.nan

        route_lookup = node_lookup.drop_duplicates(subset=["_source_node_id", "route"], keep="first")
        source_counts = node_lookup.groupby("_source_node_id", dropna=False)["node_id"].nunique()
        unique_source_ids = source_counts[source_counts == 1].index
        source_lookup = (
            node_lookup[node_lookup["_source_node_id"].isin(unique_source_ids)]
            .drop_duplicates(subset=["_source_node_id"])
            .set_index("_source_node_id")["node_id"]
        )

        for endpoint in ["u", "v"]:
            mapped_column = f"_{endpoint}_mapped"
            graph_edges_gdf = graph_edges_gdf.merge(
                route_lookup.rename(columns={"_source_node_id": endpoint, "node_id": mapped_column}),
                on=[endpoint, "route"],
                how="left",
            )
            graph_edges_gdf[mapped_column] = graph_edges_gdf[mapped_column].fillna(
                graph_edges_gdf[endpoint].map(source_lookup)
            )
            graph_edges_gdf[endpoint] = graph_edges_gdf[mapped_column]
            graph_edges_gdf = graph_edges_gdf.drop(columns=[mapped_column])
        graph_edges_gdf = gpd.GeoDataFrame(graph_edges_gdf, geometry="geometry", crs=local_crs)

    if "extra_data" not in graph_nodes_gdf.columns:
        graph_nodes_gdf["extra_data"] = [{} for _ in range(len(graph_nodes_gdf))]

    platforms = graph_nodes_gdf[graph_nodes_gdf["type"] == "platform"].copy()
    platforms = platforms.groupby(["_x_group", "_y_group"], as_index=False).agg(
        geometry=("geometry", "first"),
        node_id=("node_id", lambda s: tuple(s.dropna())),
        route=("route", lambda s: tuple(s)),
        extra_data=("extra_data", _merge_dicts_last),
    )
    platforms["type"] = "platform"

    not_platforms = graph_nodes_gdf[graph_nodes_gdf["type"] != "platform"].copy()
    not_platforms = not_platforms.groupby(["_x_group", "_y_group", "route", "type"], as_index=False, dropna=False).agg(
        geometry=("geometry", "first"),
        node_id=("node_id", lambda s: tuple(s.dropna())),
        extra_data=("extra_data", _merge_dicts_last),
    )

    all_nodes = gpd.GeoDataFrame(
        pd.concat([platforms, not_platforms], ignore_index=True), geometry="geometry", crs=local_crs
    )
    if all_nodes.empty:
        return _empty_public_transport_graph(local_crs)

    node_ids = all_nodes["node_id"].explode().dropna()
    map_nodeid_to_idx = dict(zip(node_ids.to_numpy(), node_ids.index.to_numpy()))

    node_routes = all_nodes["route"].copy()
    route_is_sequence = node_routes.map(lambda route: isinstance(route, (tuple, list, set)))
    node_routes.loc[route_is_sequence] = node_routes.loc[route_is_sequence].map(
        lambda route: list(dict.fromkeys(item for item in route if not _is_missing(item)))
    )
    single_route = route_is_sequence & node_routes.map(lambda route: isinstance(route, list) and len(route) == 1)
    node_routes.loc[single_route] = node_routes.loc[single_route].map(lambda route: route[0])
    node_routes.loc[~route_is_sequence & node_routes.map(_is_missing)] = None

    nodes_df = pd.DataFrame(
        {
            "type": all_nodes["type"],
            "route": node_routes,
        },
        index=all_nodes.index,
    )
    node_extra = all_nodes["extra_data"].map(lambda value: value if isinstance(value, dict) else {})
    node_extra_df = pd.DataFrame.from_records(node_extra.tolist(), index=all_nodes.index).drop(
        columns=["geometry"], errors="ignore"
    )
    node_extra_df = node_extra_df.drop(
        columns=[column for column in node_extra_df.columns if column in nodes_df.columns], errors="ignore"
    )
    nodes_df = pd.concat([nodes_df, node_extra_df], axis=1).reset_index(drop=True)

    node_geometry = gpd.GeoSeries(all_nodes.geometry.to_numpy(), crs=local_crs)
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=node_geometry, crs=local_crs)

    if graph_edges_gdf.empty:
        return UrbanGraph(
            nodes_gdf=nodes_gdf,
            edges_gdf=gpd.GeoDataFrame(),
            is_multigraph=True,
            is_directed=True,
            edge_direction_column="oneway",
            crs=local_crs,
            graph_type="public_transport",
        )

    for column in ["u", "v", "type", "route", "geometry", "extra_data", "oneway"]:
        if column not in graph_edges_gdf.columns:
            graph_edges_gdf[column] = np.nan

    graph_edges_gdf["u"] = graph_edges_gdf["u"].map(map_nodeid_to_idx)
    graph_edges_gdf["v"] = graph_edges_gdf["v"].map(map_nodeid_to_idx)
    missing_endpoints = graph_edges_gdf[["u", "v"]].isna().any(axis=1)
    if missing_endpoints.any():
        logger.warning(f"Dropping {int(missing_endpoints.sum())} PT edges with missing endpoint nodes")
        graph_edges_gdf = graph_edges_gdf.loc[~missing_endpoints].copy()

    if graph_edges_gdf.empty:
        return UrbanGraph(
            nodes_gdf=nodes_gdf,
            edges_gdf=gpd.GeoDataFrame(),
            is_multigraph=True,
            is_directed=True,
            edge_direction_column="oneway",
            crs=local_crs,
            graph_type="public_transport",
        )

    graph_edges_gdf[["u", "v"]] = graph_edges_gdf[["u", "v"]].astype(int)

    if "length_meter" not in graph_edges_gdf.columns:
        graph_edges_gdf["length_meter"] = np.nan
    if "time_min" not in graph_edges_gdf.columns:
        graph_edges_gdf["time_min"] = np.nan
    if "speed_m_min" not in graph_edges_gdf.columns:
        graph_edges_gdf["speed_m_min"] = np.nan

    boarding_mask = graph_edges_gdf["type"].astype(str).eq("boarding")
    if boarding_mask.any():
        alighting_edges = graph_edges_gdf.loc[boarding_mask].copy()
        alighting_edges["type"] = "alighting"
        alighting_edges["length_meter"] = 0.0
        alighting_edges["time_min"] = 0.0
        alighting_edges["oneway"] = True

        boarding_edges = graph_edges_gdf.loc[boarding_mask].copy()
        boarding_edges[["u", "v"]] = boarding_edges[["v", "u"]].to_numpy()
        boarding_edges["geometry"] = boarding_edges.geometry.reverse()
        boarding_edges["length_meter"] = 0.0
        boarding_edges["time_min"] = float(avg_boarding_time_min)
        boarding_edges["oneway"] = True

        graph_edges_gdf = gpd.GeoDataFrame(
            pd.concat([graph_edges_gdf.loc[~boarding_mask], alighting_edges, boarding_edges], ignore_index=True),
            geometry="geometry",
            crs=local_crs,
        )

    missing_oneway = graph_edges_gdf["oneway"].isna()
    if missing_oneway.any():
        logger.warning(f"Filling {int(missing_oneway.sum())} PT edges with missing oneway values")
        graph_edges_gdf["oneway"] = graph_edges_gdf["oneway"].astype(object)
        graph_edges_gdf.loc[missing_oneway, "oneway"] = True
    graph_edges_gdf["oneway"] = graph_edges_gdf["oneway"].astype(bool)

    def calc_len_time(row):
        """Calculate edge length and travel time for a public-transport edge."""
        geom = row.geometry
        length_m = float(round(geom.length, 3))
        spec = transport_registry.get(str(row.type))
        speed_limit_mpm = row.speed_m_min

        time_min = spec.travel_time_min(
            length_m,
            speed_limit_mpm=speed_limit_mpm,
        )
        time_min = float(round(time_min, 3))

        return length_m, time_min

    free_pt_link_mask = graph_edges_gdf["type"].astype(str).isin({"boarding", "alighting"})
    graph_edges_gdf.loc[free_pt_link_mask, "length_meter"] = 0.0
    graph_edges_gdf.loc[graph_edges_gdf["type"].astype(str).eq("boarding"), "time_min"] = float(avg_boarding_time_min)
    graph_edges_gdf.loc[graph_edges_gdf["type"].astype(str).eq("alighting"), "time_min"] = 0.0

    mask_missing = graph_edges_gdf["length_meter"].isna() | graph_edges_gdf["time_min"].isna()

    vals = graph_edges_gdf.loc[mask_missing].apply(calc_len_time, axis=1, result_type="expand")
    if not vals.empty:
        vals.columns = ["length_meter", "time_min"]
        graph_edges_gdf.loc[mask_missing, ["length_meter", "time_min"]] = vals

    edge_columns = ["u", "v", "geometry", "route", "type", "length_meter", "time_min", "oneway"]
    edges_df = graph_edges_gdf[edge_columns].copy()
    edges_df[["u", "v"]] = edges_df[["u", "v"]].astype(int)
    edges_df["oneway"] = edges_df["oneway"].astype(bool)

    edge_extra = graph_edges_gdf["extra_data"].map(lambda value: value if isinstance(value, dict) else {})
    edge_extra_df = pd.DataFrame.from_records(edge_extra.tolist(), index=graph_edges_gdf.index)
    protected_edge_columns = set(edge_columns) | {"k"}
    edge_extra_df = edge_extra_df.drop(
        columns=[column for column in edge_extra_df.columns if column in protected_edge_columns],
        errors="ignore",
    )

    edges_df = pd.concat([edges_df, edge_extra_df], axis=1).reset_index(drop=True)
    edges_gdf = gpd.GeoDataFrame(edges_df, geometry="geometry", crs=local_crs)
    edges_gdf["k"] = edges_gdf.groupby(["u", "v"], sort=False).cumcount()
    edge_columns_ordered = ["u", "v", "k"] + [col for col in edges_gdf.columns if col not in {"u", "v", "k"}]
    edges_gdf = edges_gdf[edge_columns_ordered].copy()

    return UrbanGraph(
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        is_multigraph=True,
        is_directed=True,
        edge_direction_column="oneway",
        crs=local_crs,
        graph_type="public_transport",
    )


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
    avg_boarding_time_min: float = 1.0,
) -> UrbanGraph:
    """
    Build a directed public-transport graph for one or multiple OSM public-transport modes inside a territory.

    Pipeline:
      1) Resolve a boundary polygon (EPSG:4326) from ``osm_id`` and/or ``territory``.
      2) Download PT routes via Overpass for the requested ``transport_types``.
      3) Optionally (subway) parse station/stop-area context (entrances, exits, transfers).
      4) Parse each route into node/edge tables (parallelized for large inputs).
      5) Assemble a single ``UrbanGraph`` and compute missing edge ``length_meter`` and ``time_min`` using
         ``transport_registry.rst``.
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
        avg_boarding_time_min:
            Time penalty for directed platform-to-stop ``boarding`` edges. Reverse stop-to-platform ``alighting``
            edges are added with zero travel time.

    Returns:
        ``UrbanGraph``: Directed public-transport graph with ``oneway`` edge direction column.
    """

    polygon = get_4326_boundary(osm_id=osm_id, territory=territory)
    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    # Subway parsing expects station information in the response.
    expect_subway = False
    if "subway" in transport_types and config.overpass_date is None:
        expect_subway = True

    ptts = ", ".join(transport_types)
    logger.info(f"Downloading routes via Overpass with types {ptts} ...")
    overpass_response: list[dict] = get_routes_by_poly(polygon, transport_types)
    overpass_data = overpass_routes_to_df(overpass_response, enable_subway_details=expect_subway)

    if overpass_data.shape[0] == 0:
        logger.warning("No routes found for public transport.")
        return _empty_public_transport_graph(local_crs)

    # Required OSM tags from the route relation.
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

    graph_edges_gdf = []
    graph_nodes_gdf = []

    ground_types = {"bus", "tram", "trolleybus", "train"} & set(transport_types)
    ground_pt_data = overpass_data[
        (overpass_data["transport_type"].isin(ground_types)) & (~overpass_data["is_way_data"])
    ].copy()

    if len(ground_pt_data) > 0:
        if not config.enable_tqdm_bar:
            logger.debug("Parsing ground public transport routes")

        use_processes = len(ground_pt_data) > 500 and multiprocessing.get_start_method(allow_none=False) == "fork"
        if use_processes:
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

        for edges_gdf, nodes_gdf in results:
            if len(edges_gdf) > 0:
                graph_edges_gdf.append(edges_gdf.dropna(axis=1, how="all"))
            if len(nodes_gdf) > 0:
                graph_nodes_gdf.append(nodes_gdf.dropna(axis=1, how="all"))

    if expect_subway:
        subway_data = overpass_data[overpass_data["transport_type"] == "subway"].copy()
        if len(subway_data) > 0:
            subway_edges, subway_nodes = overpass_subway2edgenode(subway_data, local_crs)
            if len(subway_edges) > 0:
                graph_edges_gdf.append(subway_edges.dropna(axis=1, how="all"))
            if len(subway_nodes) > 0:
                graph_nodes_gdf.append(subway_nodes.dropna(axis=1, how="all"))

    if not graph_edges_gdf and not graph_nodes_gdf:
        logger.warning("No routes were parsed for public transport.")
        return _empty_public_transport_graph(local_crs)

    graph_edges_gdf = pd.concat(graph_edges_gdf, ignore_index=True) if graph_edges_gdf else gpd.GeoDataFrame()
    graph_nodes_gdf = pd.concat(graph_nodes_gdf, ignore_index=True) if graph_nodes_gdf else gpd.GeoDataFrame()

    urban_graph: UrbanGraph = _graph_data_to_urban_graph(
        graph_nodes_gdf, graph_edges_gdf, transport_registry, local_crs, avg_boarding_time_min
    )

    if clip_by_territory:
        poly_proj = gpd.GeoSeries([polygon], crs=4326).to_crs(local_crs).union_all()
        urban_graph = urban_graph.clip(poly_proj).relabel()

    return urban_graph


def get_public_transport_graph(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    transport_types: str | list[str] | None = None,
    clip_by_territory: bool = False,
    osm_edge_tags: list[str] | None = None,
    transport_registry: TransportRegistry | None = None,
    avg_boarding_time_min: float = 1.0,
) -> UrbanGraph:
    """
    Build a directed public-transport graph for one or multiple transport modes within a territory.

    The function resolves a boundary (by ``osm_id`` or ``territory``), downloads OpenStreetMap public-transport routes
    inside that boundary, converts them into a projected ``UrbanGraph``, and computes per-edge length (meters) and
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
        osm_edge_tags:
            Subset of OSM tags to retain on edges/nodes. If None, a default subset is used.
        transport_registry:
            Transport registry used to validate transport types and to compute per-edge travel times (via each mode's
            parameters such as max speed, acceleration/braking distances, and traffic coefficient).
            If None, ``DEFAULT_REGISTRY`` is used.
        avg_boarding_time_min:
            Time penalty added to directed platform-to-stop boarding edges. Stop-to-platform alighting edges are added
            with zero travel time. Default is 1 minute.

    Returns:
        Directed PT ``UrbanGraph`` with ``crs`` and ``type="public_transport"``.

    See also:
        https://iduclub.github.io/IduEdu/examples/get_any_graph.html
        https://iduclub.github.io/IduEdu/examples/transport_registry.html
    """
    if avg_boarding_time_min < 0:
        raise ValueError(f"avg_boarding_time_min must be >= 0, got {avg_boarding_time_min}")

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
        avg_boarding_time_min=float(avg_boarding_time_min),
    )
