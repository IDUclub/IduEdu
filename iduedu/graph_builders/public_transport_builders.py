from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS
from shapely import MultiPolygon, Polygon

from iduedu import config
from iduedu.constants.transport_specs import DEFAULT_REGISTRY, TransportRegistry
from iduedu.graph.routes import empty_routes, normalize_routes, route_keys_of
from iduedu.graph.transformers import estimate_crs_for_bounds
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.overpass.downloaders import (
    active_overpass_date,
    get_4326_boundary,
    get_routes_by_poly,
    osm_source_label,
    split_stop_area_members,
)
from iduedu.overpass.parsers import (
    ground_routes_tables,
    overpass_routes_to_df,
    overpass_subway2edgenode,
    route_member_tags,
)

logger = config.logger

DEFAULT_WALK_SPEED_M_PER_MIN = 5 * 1000 / 60


def _aggregate_nodes(nodes: pd.DataFrame, keys: list[str], *, dropna: bool, with_routes: bool) -> pd.DataFrame:
    """``nodes.groupby(keys, as_index=False).agg(...)`` of the node merge, without a Python call per group.

    One row per group in sorted key order: the keys, the first geometry, the tuple of non-missing node ids
    (and of all route labels and route keys, with ``with_routes``) in row order, and the extra-data dicts
    merged with later keys winning.
    """
    columns = keys + ["geometry", "node_id"] + (["route", "route_key"] if with_routes else []) + ["extra_data"]
    codes = nodes.groupby(keys, sort=True, dropna=dropna).ngroup().to_numpy()
    kept = np.flatnonzero(codes >= 0)  # rows with a missing key are dropped when dropna
    if len(kept) == 0:
        return pd.DataFrame(columns=columns)
    order = kept[np.argsort(codes[kept], kind="stable")]
    starts = np.flatnonzero(np.r_[True, np.diff(codes[order]) != 0])
    ends = np.r_[starts[1:], len(order)]
    firsts = order[starts]
    result = nodes.iloc[firsts][keys + ["geometry"]].reset_index(drop=True)

    def chunks(column):
        values = nodes[column].to_numpy(dtype=object)[order].tolist()
        return [values[a:b] for a, b in zip(starts, ends)]

    result["node_id"] = [tuple(v for v in chunk if not _is_missing(v)) for chunk in chunks("node_id")]
    if with_routes:
        result["route"] = [tuple(chunk) for chunk in chunks("route")]
        result["route_key"] = [tuple(chunk) for chunk in chunks("route_key")]
    merged = []
    for chunk in chunks("extra_data"):
        out = {}
        for item in chunk:
            if isinstance(item, dict):
                out.update(item)
        merged.append(out)
    result["extra_data"] = merged
    return result[columns]


def _snap_edge_ends(geometry: np.ndarray, node_points: np.ndarray, u: pd.Series, v: pd.Series) -> np.ndarray:
    """Edge lines that start at the node ``u`` and end at the node ``v``.

    Nodes in one metre cell are merged into the first of them, while the edges of the others keep their own ends,
    up to a metre away; their first and last points are moved onto the node.
    """
    geometry = np.asarray(geometry, dtype=object).copy()
    lines = ~shapely.is_missing(geometry) & (shapely.get_num_coordinates(geometry) >= 2)
    if not lines.any():
        return geometry
    coords = shapely.get_coordinates(geometry[lines])
    lasts = np.cumsum(shapely.get_num_coordinates(geometry[lines])) - 1
    firsts = np.r_[0, lasts[:-1] + 1]
    coords[firsts] = shapely.get_coordinates(node_points[u.to_numpy()[lines]])
    coords[lasts] = shapely.get_coordinates(node_points[v.to_numpy()[lines]])
    geometry[lines] = shapely.set_coordinates(geometry[lines].copy(), coords)
    return geometry


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
        routes=empty_routes(),
    )


def _text_tag(tags: dict, key: str):
    value = tags.get(key)
    return None if value is None else str(value)


def osm_routes_table(elements: pd.DataFrame, transport_types, route_tags=None) -> pd.DataFrame:
    """Routes table of OSM route relations: one row per relation of ``transport_types``.

    The line of a route is the ``route_master`` relation that lists it: ``line_key`` is
    ``osm:r<master id>`` and ``line_name`` its ``ref`` (or ``name``). ``route_tags`` (by default
    ``config.transport_route_attr``) are copied from the relation; ``osm_id`` and ``osm_route`` (the raw
    ``route`` value, e.g. ``light_rail`` for a tram) are added.
    """
    route_tags = sorted(config.transport_route_attr if route_tags is None else set(route_tags))
    if elements is None or elements.empty or "type" not in elements.columns:
        return empty_routes()
    relations = elements[elements["type"].eq("relation")]
    tags = relations["tags"].map(lambda value: value if isinstance(value, dict) else {})
    masters: dict[int, tuple[str, str | None]] = {}
    for rid, members, master_tags in zip(relations["id"], relations.get("members", pd.Series(dtype=object)), tags):
        if master_tags.get("type") != "route_master" or not isinstance(members, list):
            continue
        line = (f"osm:r{int(rid)}", _text_tag(master_tags, "ref") or _text_tag(master_tags, "name"))
        for member in members:
            if isinstance(member, dict) and member.get("type") == "relation":
                masters.setdefault(int(member["ref"]), line)
    wanted = set(transport_types)
    rows = []
    for rid, route_tags_of, mode in zip(relations["id"], tags, relations.get("transport_type")):
        if not route_tags_of.get("route") or mode not in wanted:
            continue
        line_key, line_name = masters.get(int(rid), (None, None))
        row = {
            "route_key": f"osm:r{int(rid)}",
            "source": "osm",
            "mode": mode,
            "line_key": line_key,
            "line_name": line_name,
            "osm_id": int(rid),
            "osm_route": str(route_tags_of["route"]),
        }
        for key in route_tags:
            row[key] = _text_tag(route_tags_of, key)
        rows.append(row)
    if not rows:
        return empty_routes()
    table = pd.DataFrame(rows).drop_duplicates("route_key").set_index("route_key")
    return normalize_routes(table)


def _graph_data_to_urban_graph(
    graph_nodes_gdf: gpd.GeoDataFrame,
    graph_edges_gdf: gpd.GeoDataFrame,
    transport_registry: TransportRegistry,
    local_crs,
    routes: pd.DataFrame | None = None,
) -> UrbanGraph:
    """
    Build a directed public-transport UrbanGraph from parser node/edge tables.

    ``routes`` is the routes table; the graph keeps the rows its edges refer to.
    """
    routes = empty_routes() if routes is None else routes
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

    for column in ["node_id", "type", "route", "route_key"]:
        if column not in graph_nodes_gdf.columns:
            graph_nodes_gdf[column] = np.nan

    # The route a node belongs to: its route key, or its label for parser tables made without keys.
    graph_nodes_gdf["_route_id"] = graph_nodes_gdf["route_key"].where(
        graph_nodes_gdf["route_key"].notna(), graph_nodes_gdf["route"]
    )
    graph_nodes_gdf["_source_node_id"] = graph_nodes_gdf["node_id"]
    graph_nodes_gdf["_geometry_key"] = graph_nodes_gdf.geometry.to_wkb()
    duplicate_nodes = graph_nodes_gdf.duplicated(
        subset=["_source_node_id", "_geometry_key", "_route_id", "type"], keep="first"
    )
    if duplicate_nodes.any():
        graph_nodes_gdf = graph_nodes_gdf.loc[~duplicate_nodes].copy()

    # A stop node belongs to one route variant: nodes and edges are matched by the source node and the route key.
    node_lookup = graph_nodes_gdf[["_source_node_id", "_route_id"]].copy()
    node_lookup["node_id"] = np.arange(len(graph_nodes_gdf), dtype=np.int64)
    graph_nodes_gdf["node_id"] = node_lookup["node_id"].to_numpy()

    if not graph_edges_gdf.empty:
        for column in ["u", "v", "route", "route_key"]:
            if column not in graph_edges_gdf.columns:
                graph_edges_gdf[column] = np.nan

        graph_edges_gdf["_route_id"] = graph_edges_gdf["route_key"].where(
            graph_edges_gdf["route_key"].notna(), graph_edges_gdf["route"]
        )
        route_lookup = node_lookup.drop_duplicates(subset=["_source_node_id", "_route_id"], keep="first")
        source_counts = node_lookup.groupby("_source_node_id", dropna=False)["node_id"].nunique()
        unique_source_ids = source_counts[source_counts == 1].index
        source_lookup = (
            node_lookup[node_lookup["_source_node_id"].isin(unique_source_ids)]
            .drop_duplicates(subset=["_source_node_id"])
            .set_index("_source_node_id")["node_id"]
        )
        shared_lookup = (
            node_lookup[~node_lookup["_source_node_id"].isin(unique_source_ids)]
            .dropna(subset=["_source_node_id"])
            .groupby("_source_node_id")["node_id"]
            .agg(list)
        )

        for endpoint in ["u", "v"]:
            mapped_column = f"_{endpoint}_mapped"
            graph_edges_gdf = graph_edges_gdf.merge(
                route_lookup.rename(columns={"_source_node_id": endpoint, "node_id": mapped_column}),
                on=[endpoint, "_route_id"],
                how="left",
            )
            graph_edges_gdf[mapped_column] = graph_edges_gdf[mapped_column].fillna(
                graph_edges_gdf[endpoint].map(source_lookup)
            )
            # A link without a route (a subway station to its platforms) ending at an OSM node that several routes
            # share, such as a cross-platform interchange, joins every node made from it.
            shared = (
                graph_edges_gdf[mapped_column].isna()
                & graph_edges_gdf["_route_id"].isna()
                & graph_edges_gdf[endpoint].isin(shared_lookup.index)
            )
            if shared.any():
                mapped = graph_edges_gdf[mapped_column].astype(object)
                mapped[shared] = graph_edges_gdf.loc[shared, endpoint].map(shared_lookup)
                graph_edges_gdf = pd.DataFrame(graph_edges_gdf).assign(**{mapped_column: mapped})
                graph_edges_gdf = graph_edges_gdf.explode(mapped_column, ignore_index=True)
            graph_edges_gdf[endpoint] = graph_edges_gdf[mapped_column]
            graph_edges_gdf = graph_edges_gdf.drop(columns=[mapped_column])
        graph_edges_gdf = gpd.GeoDataFrame(graph_edges_gdf, geometry="geometry", crs=local_crs)

    if "extra_data" not in graph_nodes_gdf.columns:
        graph_nodes_gdf["extra_data"] = [{} for _ in range(len(graph_nodes_gdf))]

    platforms = graph_nodes_gdf[graph_nodes_gdf["type"] == "platform"]
    platforms = _aggregate_nodes(platforms, ["_x_group", "_y_group"], dropna=True, with_routes=True)
    platforms["type"] = "platform"

    not_platforms = graph_nodes_gdf[graph_nodes_gdf["type"] != "platform"]
    not_platforms = _aggregate_nodes(
        not_platforms,
        ["_x_group", "_y_group", "_route_id", "route_key", "route", "type"],
        dropna=False,
        with_routes=False,
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

    # A platform lists the keys of every route calling there; a stop node has the key of its own route.
    node_route_keys = all_nodes["route_key"].map(
        lambda keys: (
            route_keys_of(keys) if isinstance(keys, (tuple, list)) else (keys if isinstance(keys, str) else None)
        )
    )
    nodes_df = pd.DataFrame(
        {
            "type": all_nodes["type"],
            "route": node_routes,
            "route_key": node_route_keys,
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
            routes=routes,
        )

    for column in ["u", "v", "type", "route", "route_key", "geometry", "extra_data", "oneway"]:
        if column not in graph_edges_gdf.columns:
            graph_edges_gdf[column] = np.nan

    graph_edges_gdf["u"] = graph_edges_gdf["u"].map(map_nodeid_to_idx)
    graph_edges_gdf["v"] = graph_edges_gdf["v"].map(map_nodeid_to_idx)
    missing_endpoints = graph_edges_gdf[["u", "v"]].isna().any(axis=1)
    if missing_endpoints.any():
        logger.warning(f"Dropping {int(missing_endpoints.sum())} PT edges with missing endpoint nodes")
        graph_edges_gdf = graph_edges_gdf.loc[~missing_endpoints].copy()
    # Stops of one route in one metre cell are one node, so an edge between them (a stop listed twice in a row, a
    # terminus passed on the way in and out) leads nowhere.
    graph_edges_gdf = graph_edges_gdf.loc[graph_edges_gdf["u"].to_numpy() != graph_edges_gdf["v"].to_numpy()]

    if graph_edges_gdf.empty:
        return UrbanGraph(
            nodes_gdf=nodes_gdf,
            edges_gdf=gpd.GeoDataFrame(),
            is_multigraph=True,
            is_directed=True,
            edge_direction_column="oneway",
            crs=local_crs,
            graph_type="public_transport",
            routes=routes,
        )

    graph_edges_gdf[["u", "v"]] = graph_edges_gdf[["u", "v"]].astype(int)
    graph_edges_gdf["geometry"] = _snap_edge_ends(
        graph_edges_gdf.geometry.to_numpy(), nodes_gdf.geometry.to_numpy(), graph_edges_gdf["u"], graph_edges_gdf["v"]
    )

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
        # The waiting time belongs to the mode being boarded, and the node type is
        # not a reliable name for it: in the subway branch a boarding edge ends at
        # a platform, so the type reads "platform" or "subway_platform" and the
        # registry lookup raised, taking every city with a metro down with it.
        # The mode is therefore read from the travel edges the node participates
        # in, which carry the mode by construction, and the node type is used only
        # when it is itself a known mode.
        known_modes = set(transport_registry.list_types())
        travel = graph_edges_gdf[graph_edges_gdf["type"].astype(str).isin(known_modes)]
        node_mode = pd.concat(
            [
                pd.Series(travel["type"].to_numpy(), index=travel["u"].to_numpy()),
                pd.Series(travel["type"].to_numpy(), index=travel["v"].to_numpy()),
            ]
        )
        node_mode = node_mode[~node_mode.index.duplicated(keep="first")]

        boarding_modes = boarding_edges["v"].map(node_mode)
        by_type = boarding_edges["v"].map(nodes_gdf["type"])
        boarding_modes = boarding_modes.fillna(by_type.where(by_type.isin(known_modes)))

        unresolved = boarding_modes.isna()
        if unresolved.any():
            # Dropping the edge is wrong -- the platform would become unreachable --
            # so the boarding is kept with the registry's own fallback and reported.
            logger.warning(
                f"{int(unresolved.sum())} boarding edges have no resolvable transport mode; "
                f"using the default waiting time"
            )
        default_wait = min(
            (transport_registry.get(mode).avg_wait_time_min for mode in known_modes),
            default=0.0,
        )
        boarding_edges["time_min"] = boarding_modes.map(
            lambda mode: (
                float(transport_registry.get(str(mode)).avg_wait_time_min) if isinstance(mode, str) else default_wait
            )
        )
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

    def calc_len_time(geom, edge_type, speed_m_min):
        """Calculate edge length and travel time for a public-transport edge.

        Only travel edges carry a transport mode as their type. Station interiors
        (``subway_station``, ``subway_entrance``, ``subway_exit``) are walked, not
        ridden, and asking the registry about them raised a ``KeyError`` that
        aborted the whole graph.
        """
        length_m = float(round(geom.length, 3))
        spec = transport_registry.try_get(str(edge_type))
        if spec is None:
            return length_m, float(round(length_m / DEFAULT_WALK_SPEED_M_PER_MIN, 3))

        time_min = spec.travel_time_min(
            length_m,
            speed_limit_mpm=speed_m_min,
        )
        return length_m, float(round(time_min, 3))

    free_pt_link_mask = graph_edges_gdf["type"].astype(str).isin({"boarding", "alighting"})
    graph_edges_gdf.loc[free_pt_link_mask, "length_meter"] = 0.0
    graph_edges_gdf.loc[graph_edges_gdf["type"].astype(str).eq("alighting"), "time_min"] = 0.0

    mask_missing = graph_edges_gdf["length_meter"].isna() | graph_edges_gdf["time_min"].isna()

    missing = graph_edges_gdf.loc[mask_missing]
    if len(missing):
        # A loop over plain values: a row-wise DataFrame.apply builds a Series per edge.
        vals = [
            calc_len_time(geometry, edge_type, speed)
            for geometry, edge_type, speed in zip(missing.geometry, missing["type"], missing["speed_m_min"])
        ]
        graph_edges_gdf.loc[mask_missing, ["length_meter", "time_min"]] = np.asarray(vals, dtype=float)

    edge_columns = ["u", "v", "geometry", "route", "route_key", "type", "length_meter", "time_min", "oneway"]
    edges_df = graph_edges_gdf[edge_columns].copy()
    edges_df[["u", "v"]] = edges_df[["u", "v"]].astype(int)
    edges_df["oneway"] = edges_df["oneway"].astype(bool)
    edges_df["type"] = pd.Series(edges_df["type"].tolist(), index=edges_df.index)  # the text dtype of other builders

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
        routes=routes,
    )


def _build_public_transport_graph(
    osm_id: int | None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None,
    transport_types: list[str],
    osm_edge_tags: list[str] | None,
    transport_registry: TransportRegistry,
    clip_by_territory: bool = False,
    crs: Any | None = None,
) -> UrbanGraph:
    """
    Build a directed public-transport graph for one or multiple OSM public-transport modes inside a territory.

    Pipeline:
      1) Resolve a boundary polygon (EPSG:4326) from ``osm_id`` and/or ``territory``.
      2) Download PT routes via Overpass (or read them from the local OSM file in pbf mode) for the requested
         ``transport_types``.
      3) Optionally (subway) parse station/stop-area context (entrances, exits, transfers).
      4) Parse each route into node/edge tables (the ground routes by one parallel Rust kernel call).
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
        crs:
            Projected CRS of the graph; by default the local UTM zone of the territory.
    Returns:
        ``UrbanGraph``: Directed public-transport graph with ``oneway`` edge direction column.
    """

    if crs is not None:
        local_crs = CRS.from_user_input(crs)
        if not local_crs.is_projected:
            raise ValueError(f"crs must be projected in metres, got {local_crs.to_string()}")
        local_crs = local_crs.to_epsg() or local_crs
    polygon = get_4326_boundary(osm_id=osm_id, territory=territory)
    if crs is None:
        local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    # Subway parsing expects station information in the response.
    expect_subway = False
    if "subway" in transport_types and active_overpass_date() is None:
        expect_subway = True

    ptts = ", ".join(transport_types)
    logger.info(f"Downloading routes via {osm_source_label()} with types {ptts} ...")
    overpass_response: list[dict] = get_routes_by_poly(polygon, transport_types)
    # Stop-area members arrive in the same response, after a marker; separate them
    # so the route parser never sees them and the subway parser need not ask again.
    overpass_response, stop_area_members = split_stop_area_members(overpass_response)
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

    # Everything except the subway is parsed the same way, from a route relation and
    # its stops; the subway is special only because OSM describes it with stop areas,
    # entrances and interchanges. Listing the ground modes by hand meant a mode the
    # registry knew about was silently discarded here after being downloaded: taxi
    # and monorail routes arrived from Overpass and never reached a graph.
    ground_types = set(transport_types) - {"subway"}
    # Stop and platform nodes come with the routes for their tags; they are not routes themselves.
    is_node = overpass_data["type"].eq("node") if "type" in overpass_data else pd.Series(False, overpass_data.index)
    ground_pt_data = overpass_data[
        (overpass_data["transport_type"].isin(ground_types)) & (~overpass_data["is_way_data"]) & ~is_node
    ].copy()

    if len(ground_pt_data) > 0:
        logger.debug(f"Parsing {len(ground_pt_data)} ground public transport routes")
        # One Rust kernel call for all routes, in parallel; one table each for their edges and nodes.
        ground_edges, ground_nodes = ground_routes_tables(
            ground_pt_data, local_crs, ref2speed, needed_tags, member_tags=route_member_tags(overpass_data)
        )
        if len(ground_edges):
            graph_edges_gdf.append(ground_edges)
        if len(ground_nodes):
            graph_nodes_gdf.append(ground_nodes)

    if expect_subway:
        subway_data = overpass_data[overpass_data["transport_type"] == "subway"].copy()
        if len(subway_data) > 0:
            subway_edges, subway_nodes = overpass_subway2edgenode(
                subway_data, local_crs, known_members=stop_area_members
            )
            if len(subway_edges) > 0:
                graph_edges_gdf.append(subway_edges.dropna(axis=1, how="all"))
            if len(subway_nodes) > 0:
                graph_nodes_gdf.append(subway_nodes.dropna(axis=1, how="all"))

    if not graph_edges_gdf and not graph_nodes_gdf:
        logger.warning("No routes were parsed for public transport.")
        return _empty_public_transport_graph(local_crs)

    graph_edges_gdf = pd.concat(graph_edges_gdf, ignore_index=True) if graph_edges_gdf else gpd.GeoDataFrame()
    graph_nodes_gdf = pd.concat(graph_nodes_gdf, ignore_index=True) if graph_nodes_gdf else gpd.GeoDataFrame()

    routes = osm_routes_table(overpass_data, transport_types)
    urban_graph: UrbanGraph = _graph_data_to_urban_graph(
        graph_nodes_gdf, graph_edges_gdf, transport_registry, local_crs, routes=routes
    )

    if clip_by_territory:
        urban_graph = urban_graph.clip(polygon).relabel()

    return urban_graph


def get_public_transport_graph(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    transport_types: str | list[str] | None = None,
    clip_by_territory: bool = False,
    osm_edge_tags: list[str] | None = None,
    transport_registry: TransportRegistry | None = None,
    crs: Any | None = None,
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
              - ``None``: include every type of ``transport_registry``;
              - ``str``: a single OSM route type, e.g. ``"bus"``;
              - ``Sequence[str]``: multiple types, e.g. ``["tram", "bus", "trolleybus", "subway"]``.

            Values are normalized with ``strip().lower()`` and validated against the registry.
        clip_by_territory:
            If True, clip the resulting graph to the boundary (in the local CRS).
        osm_edge_tags:
            Subset of OSM tags to retain on edges/nodes. If None, a default subset is used.
        transport_registry:
            Transport registry used to validate transport types and to compute per-edge travel times (via each mode's
            parameters such as free speed, acceleration and braking distances, dwell time and average waiting
            time for boarding edges).
            If None, ``DEFAULT_REGISTRY`` is used.
        crs:
            Projected CRS in metres for the graph (anything ``pyproj.CRS.from_user_input`` accepts). If None, the
            UTM zone of the territory is estimated. Give one CRS to graphs built in parts to share coordinates,
            or a conformal CRS of a large area (such as a Lambert conic) when routes reach far beyond one UTM zone.

    Returns:
        Directed PT ``UrbanGraph`` with ``crs`` and ``type="public_transport"``. Route stops and the edges of a
        route carry its ``route_key`` (``osm:r<relation id>``), platforms the list of keys of the routes calling
        there, and ``graph.routes`` is the table of the routes: ``mode``, the relation tags of
        ``config.transport_route_attr``, ``osm_id``, ``osm_route`` and the ``route_master`` line as ``line_key``
        and ``line_name``. With ``clip_by_territory`` it keeps the routes with an edge inside.

    See also:
        https://iduclub.github.io/IduEdu/examples/get_any_graph.html
        https://iduclub.github.io/IduEdu/examples/transport_registry.html
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
        crs=crs,
    )
