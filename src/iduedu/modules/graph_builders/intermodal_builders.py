import concurrent.futures
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely import LineString, MultiPolygon, Polygon
from shapely.ops import substring

from iduedu import config
from iduedu.modules.graph_builders.drive_walk_builders import get_walk_graph
from iduedu.modules.graph_builders.public_transport_builders import get_public_transport_graph
from iduedu.modules.graph_transformers import keep_largest_strongly_connected_component
from iduedu.modules.overpass.overpass_downloaders import get_4326_boundary

logger = config.logger


def join_pt_walk_graph(
    public_transport_g: nx.Graph,
    walk_g: nx.Graph,
    max_dist=20,
    keep_largest_subgraph: bool = True,
) -> nx.Graph:
    """
    Join a public-transport graph with a pedestrian graph into a single intermodal network.

    The function relabels node ids to avoid collisions, finds PT platforms (including subway
    entries/exits), projects them onto the nearest walking edges within `max_dist`, and connects
    the two networks. When multiple platforms project to the same point on a walk edge, PT nodes
    are merged. For each platform–edge match, the walking edge is split at the projection point
    (or its endpoint is replaced by the platform node), and new directed walk edges are created
    with length/time attributes. Finally, both graphs are composed and (optionally) pruned to
    the largest strongly connected component.

    Parameters:
        public_transport_g (nx.Graph): PT graph with node attributes `x`, `y`, `type`, `route`
            and a graph CRS in `graph["crs"]`. Platform-like nodes are those with
            `type in {"platform","subway_entry_exit","subway_entry","subway_exit"}`.
        walk_g (nx.Graph): Pedestrian graph in the **same** CRS with edge `geometry` and graph
            attribute `graph["walk_speed"]` (meters/min). If `walk_speed` is missing, a default
            of 83.33 m/min is used.
        max_dist (float): Max search radius in meters to snap platforms to walk edges.
        keep_largest_subgraph (bool): If True, keep only the largest strongly connected component.

    Returns:
        (nx.Graph): Intermodal `MultiDiGraph`:
            - nodes: union of PT and Walk nodes (platforms become vertices of the walk network);
            - edges: original PT + split/updated Walk edges with `geometry`, `length_meter`,
              `time_min`, `type="walk"` (for inserted walking segments).

            Graph attrs: `graph["type"]="intermodal"`; CRS is inherited from inputs.

    Notes:
        - Node ids of PT and Walk parts are relabeled to disjoint ranges before composition,
          then relabeled densely (0..N-1) in the final graph.
        - Platforms projecting exactly to a walk edge endpoint trigger in-place node relabeling
          of that endpoint to the platform id (no edge splitting needed).
        - Duplicate projection points along the same walk edge are detected; PT platforms are merged
          and their incident PT edges are rewired onto the chosen representative.
    """

    if public_transport_g.graph["crs"] != walk_g.graph["crs"]:
        raise AttributeError(
            f"CRS mismatching, public_transport_graph crs {public_transport_g.graph['crs']}, walk_graph crs {walk_g.graph['crs']}"
        )
    logger.info("Composing intermodal graph...")

    num_nodes_g1 = len(public_transport_g.nodes)
    mapping_g1 = {node: idx for idx, node in enumerate(public_transport_g.nodes)}
    mapping_g2 = {node: idx + num_nodes_g1 for idx, node in enumerate(walk_g.nodes)}
    transport: nx.DiGraph = nx.relabel_nodes(public_transport_g, mapping_g1)
    walk: nx.MultiDiGraph = nx.relabel_nodes(walk_g, mapping_g2)

    platforms = pd.DataFrame.from_dict(dict(transport.nodes(data=True)), orient="index")
    platforms = platforms[platforms["type"].isin(["platform", "subway_entry_exit", "subway_entry", "subway_exit"])]
    platforms["geometry"] = gpd.points_from_xy(platforms["x"], platforms["y"])
    platforms = gpd.GeoDataFrame(platforms, crs=transport.graph["crs"])

    logger.debug("creating walk edges gdf")
    walk_edges = gpd.GeoDataFrame(
        nx.to_pandas_edgelist(walk.to_undirected(), source="u", target="v", edge_key="k"), crs=walk.graph["crs"]
    )
    walk_edges["edge_geometry"] = walk_edges["geometry"]

    logger.debug("joining nearest")
    projection_join = platforms.reset_index().sjoin_nearest(walk_edges, max_distance=max_dist, distance_col="dist")
    projection_join["project_dist"] = projection_join["edge_geometry"].project(projection_join["geometry"])
    projection_join["project_point"] = gpd.GeoSeries(
        projection_join["edge_geometry"].interpolate(projection_join["project_dist"]), crs=walk.graph["crs"]
    ).set_precision(1)
    projection_join["route"] = projection_join["route"].apply(lambda x: x if isinstance(x, list) else [x])

    logger.debug("searching for duplicated project points")
    projection_join = projection_join.groupby(by="project_point", as_index=False).agg(
        {
            "index": lambda x: list(dict.fromkeys(item for item in x)),
            "index_right": "first",
            "geometry": "first",
            "u": "first",
            "v": "first",
            "k": "first",
            "project_dist": "first",
            "route": lambda x: list(dict.fromkeys(item for sublist in x for item in sublist)),
        }
    )

    for ind, row in projection_join.iterrows():
        if len(row["index"]) == 1:
            projection_join.loc[ind, "index"] = row["index"][0]
        platform_in = row["index"][0]
        for platform_out in row["index"][1:]:
            if platform_out == platform_in:
                continue
            for neighbor in transport.neighbors(platform_out):
                edge_data = transport.get_edge_data(neighbor, platform_out)
                if edge_data:
                    transport.add_edge(neighbor, platform_in, **edge_data)
                edge_data = transport.get_edge_data(platform_out, neighbor)
                if edge_data:
                    transport.add_edge(platform_in, neighbor, **edge_data)
            transport.remove_node(platform_out)
        transport.nodes[platform_in]["route"] = row["route"]
        projection_join.loc[ind, "index"] = platform_in

    logger.debug("Projecting platforms to walk graph")
    points_grouped_by_edge = projection_join.groupby(by="index_right", as_index=False).agg(
        {
            "index": tuple,
            "geometry": tuple,
            "u": "first",
            "v": "first",
            "k": "first",
        }
    )
    try:
        speed = walk.graph["walk_speed"]
    except KeyError:  # pragma: no branch
        logger.warning(
            "There is no walk_speed in graph, set to the default speed - 83.33 m/min"
        )  # посчитать примерную скорость по length timemin для любой эджи
        speed = 83.33

    edges_to_del = []

    # for name, row in points_grouped_by_edge.iterrows():
    for i in range(len(points_grouped_by_edge)):
        row = points_grouped_by_edge.iloc[i]
        u, v, k = row[["u", "v", "k"]]
        edge: LineString = walk.get_edge_data(u, v, k)["geometry"]
        # Если платформа одна единственная на эдж
        if len(row["index"]) == 1:
            platform_id = row["index"][0]
            dist = edge.project(row.geometry[0])
            projected_point = edge.interpolate(dist)
            if dist == 0:  # pragma: no branch
                # Если платформа проецируется на начало эджа
                mapping = {u: platform_id}
                nx.relabel_nodes(walk, mapping, copy=False)
                points_grouped_by_edge.loc[points_grouped_by_edge["u"] == u, "u"] = platform_id
                points_grouped_by_edge.loc[points_grouped_by_edge["v"] == u, "v"] = platform_id
            elif dist == edge.length:
                # Если на конец
                mapping = {v: platform_id}
                nx.relabel_nodes(walk, mapping, copy=False)
                points_grouped_by_edge.loc[points_grouped_by_edge["u"] == v, "u"] = platform_id
                points_grouped_by_edge.loc[points_grouped_by_edge["v"] == v, "v"] = platform_id
            else:
                line1 = substring(edge, 0, dist)
                line2 = substring(edge, dist, edge.length)
                # Убираем старые эджи и добавляем новые, в серединке нода - наша платформа для соединения
                edges_to_del.append((u, v, k))
                if u != v:
                    edges_to_del.append((v, u, k))
                walk.add_node(platform_id, x=round(projected_point.x, 5), y=round(projected_point.y, 5))
                walk.add_edge(
                    u,
                    platform_id,
                    geometry=line1,
                    length_meter=round(line1.length, 3),
                    time_min=round(line1.length / speed, 3),
                    type="walk",
                )
                walk.add_edge(
                    platform_id,
                    u,
                    geometry=line1.reverse(),
                    length_meter=round(line1.length, 3),
                    time_min=round(line1.length / speed, 3),
                    type="walk",
                )
                walk.add_edge(
                    platform_id,
                    v,
                    geometry=line2,
                    length_meter=round(line2.length, 3),
                    time_min=round(line2.length / speed, 3),
                    type="walk",
                )
                walk.add_edge(
                    v,
                    platform_id,
                    geometry=line2.reverse(),
                    length_meter=round(line2.length, 3),
                    time_min=round(line2.length / speed, 3),
                    type="walk",
                )
        # Если платформ несколько на эдж
        else:
            dist_project = [
                (edge.project(platform), edge.interpolate(edge.project(platform)), ind)
                for ind, platform in zip(row["index"], row.geometry)
            ]
            dist_project.sort(key=lambda x: x[0])

            eps = 1e-6 * edge.length  # допуск по расстоянию вдоль эджа

            # сгруппируем платформы по расстоянию(fallback на случай проецирования в одну точку пешеходки)
            grouped = []
            for dist, projected_point, cur_index in dist_project:
                if not grouped:
                    grouped.append([dist, projected_point, [cur_index]])
                else:
                    last_dist, last_point, idxs = grouped[-1]
                    if abs(dist - last_dist) <= eps:
                        idxs.append(cur_index)
                    else:
                        grouped.append([dist, projected_point, [cur_index]])

            dist_project_grouped = grouped

            merge_mapping = {}

            u_to_del = u
            v_to_del = v
            last_dist = 0
            last_u = u

            for dist, projected_point, idxs in dist_project_grouped:
                main_index = idxs[0]
                for extra_idx in idxs[1:]:
                    merge_mapping[extra_idx] = main_index

                if dist == 0:  # pragma: no branch
                    # Если платформа проецируются на начало эджа
                    mapping = {u: main_index}
                    u_to_del = main_index
                    nx.relabel_nodes(walk, mapping, copy=False)

                    points_grouped_by_edge.loc[points_grouped_by_edge["u"] == u, "u"] = main_index
                    points_grouped_by_edge.loc[points_grouped_by_edge["v"] == u, "v"] = main_index

                    last_u = main_index

                elif dist == edge.length:  # pragma: no branch
                    # Если на конец
                    mapping = {v: main_index}
                    v_to_del = main_index
                    nx.relabel_nodes(walk, mapping, copy=False)

                    points_grouped_by_edge.loc[points_grouped_by_edge["u"] == v, "u"] = main_index
                    points_grouped_by_edge.loc[points_grouped_by_edge["v"] == v, "v"] = main_index

                    last_u = main_index

                else:
                    # По очереди добавляем ноды/линии
                    line = substring(edge, last_dist, dist)

                    if main_index not in walk:
                        walk.add_node(
                            main_index,
                            x=round(projected_point.x, 5),
                            y=round(projected_point.y, 5),
                        )

                    walk.add_edge(
                        last_u,
                        main_index,
                        geometry=line,
                        length_meter=round(line.length, 3),
                        time_min=round(line.length / speed, 3),
                        type="walk",
                    )
                    walk.add_edge(
                        main_index,
                        last_u,
                        geometry=line.reverse(),
                        length_meter=round(line.length, 3),
                        time_min=round(line.length / speed, 3),
                        type="walk",
                    )
                    last_u = main_index

                last_dist = dist

            # Если последняя остановка спроецировалась не на конец эджа, надо добавить остаток
            if last_dist != edge.length:
                line = substring(edge, last_dist, edge.length)
                walk.add_edge(
                    last_u,
                    v,
                    geometry=line,
                    length_meter=round(line.length, 3),
                    time_min=round(line.length / speed, 3),
                    type="walk",
                )
                walk.add_edge(
                    v,
                    last_u,
                    geometry=line.reverse(),
                    length_meter=round(line.length, 3),
                    time_min=round(line.length / speed, 3),
                    type="walk",
                )

            if merge_mapping:  # pragma: no branch
                nx.relabel_nodes(walk, merge_mapping, copy=False)
                points_grouped_by_edge["u"] = points_grouped_by_edge["u"].replace(merge_mapping)
                points_grouped_by_edge["v"] = points_grouped_by_edge["v"].replace(merge_mapping)

                u_to_del = merge_mapping.get(u_to_del, u_to_del)
                v_to_del = merge_mapping.get(v_to_del, v_to_del)

            edges_to_del.append((u_to_del, v_to_del, k))
            if u != v:
                edges_to_del.append((v_to_del, u_to_del, k))
    walk.remove_edges_from(edges_to_del)
    logger.debug("Composing graphs")
    intermodal = nx.compose(nx.MultiDiGraph(transport), nx.MultiDiGraph(walk))
    if keep_largest_subgraph:
        intermodal = keep_largest_strongly_connected_component(intermodal)
    intermodal.remove_nodes_from([node for node, data in intermodal.nodes(data=True) if "x" not in data.keys()])
    mapping = {old_label: new_label for new_label, old_label in enumerate(intermodal.nodes())}
    logger.debug("Relabeling")
    nx.relabel_nodes(intermodal, mapping, copy=False)
    intermodal.graph["type"] = "intermodal"
    logger.debug("Done composing!")
    return intermodal


def get_intermodal_graph(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    clip_by_territory: bool = False,
    keep_edge_geometry: bool = True,
    osm_edge_tags: list[str] | None = None,
    max_dist: float = 30,
    keep_largest_subgraph: bool = True,
    walk_kwargs: dict[str, Any] | None = None,
    pt_kwargs: dict[str, Any] | None = None,
) -> nx.Graph:
    """
    Build an intermodal (PT+walking) graph for a territory by downloading, parsing, and joining both networks.

    The function resolves a boundary polygon (by `osm_id` or `territory`), runs **in parallel**:
    1) pedestrian network construction (`get_walk_graph`),
    2) public-transport network construction for selected modes (`get_all_public_transport_graph`),
    then connects PT platforms to nearby walk edges via `join_pt_walk_graph` using a snapping radius `max_dist`.
    Edge lengths (m) and times (min) come from the underlying builders and from the walk-edge splits.

    Parameters:
        osm_id (int | None): OSM relation/area id for the territory; provide this or `territory`.
        territory (Polygon | MultiPolygon | gpd.GeoDataFrame | None): Boundary geometry in EPSG:4326.
        clip_by_territory (bool): If True, both PT and Walk graphs are clipped to the boundary.
        keep_edge_geometry (bool): If True, keep `shapely` geometries on edges for both sub-graphs.
        osm_edge_tags (list[str] | None): Subset of OSM tags to retain (forwarded to both builders).
        max_dist (float): Max distance in meters to connect PT platforms to walk edges.
        keep_largest_subgraph (bool): If True, keep only the largest strongly connected component after joining.
        walk_kwargs (dict[str, Any] | None): Extra keyword args for `get_walk_graph` (e.g., `walk_speed`,
            `simplify`, `osm_edge_tags`, `keep_largest_subgraph`, …). Defaults are sensible.
        pt_kwargs (dict[str, Any] | None): Extra keyword args for `get_all_public_transport_graph`
            (e.g., `transport_types`, `osm_edge_tags`, `keep_edge_geometry`, …).

    Returns:
        (nx.Graph): Intermodal `MultiDiGraph` combining public transport and pedestrian networks, with:
            - node attrs: `x`, `y` (local CRS), plus PT metadata for platform/station nodes where present;
            - edge attrs: `type` (e.g., "walk", PT edge types), `length_meter`, `time_min`, optional `geometry`,
              and selected OSM tags.

          Graph CRS equals the builders’ local projected CRS.

    Notes:
        - `keep_edge_geometry`, `clip_by_territory`, and `osm_edge_tags` are propagated to both builders unless
          overridden in `walk_kwargs`/`pt_kwargs`.
        - If the PT graph is empty for the area, the function returns the walking graph alone (with a warning).
        - Joining requires both sub-graphs to share the same CRS; builders derive a **local projected CRS**
          from the boundary’s extent, so lengths/times are in meters/minutes.
    """
    boundary = get_4326_boundary(osm_id=osm_id, territory=territory)

    walk_kwargs = dict(walk_kwargs or {})
    pt_kwargs = dict(pt_kwargs or {})

    for d in (walk_kwargs, pt_kwargs):
        d.setdefault("keep_edge_geometry", keep_edge_geometry)
        if osm_edge_tags is not None:
            d.setdefault("osm_edge_tags", osm_edge_tags)
        d.setdefault("clip_by_territory", clip_by_territory)

    walk_kwargs.setdefault("keep_largest_subgraph", keep_largest_subgraph)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        walk_future = executor.submit(get_walk_graph, territory=boundary, **walk_kwargs)
        logger.debug("Started downloading and parsing walk graph...")

        pt_future = executor.submit(get_public_transport_graph, territory=boundary, **pt_kwargs)
        logger.debug("Started downloading and parsing public transport graph...")

        pt_g = pt_future.result()
        logger.debug("Public transport graph done!")

        walk_g = walk_future.result()
        logger.debug("Walk graph done!")

    if len(pt_g.nodes()) == 0:  # pragma: no branch
        logger.warning("Public transport graph is empty! Returning only walk graph.")
        return walk_g

    intermodal = join_pt_walk_graph(pt_g, walk_g, max_dist=max_dist, keep_largest_subgraph=keep_largest_subgraph)
    return intermodal
