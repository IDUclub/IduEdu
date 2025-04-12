import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely import LineString, Point
from shapely.ops import substring

from iduedu import config
from iduedu.utils.utils import remove_weakly_connected_nodes

logger = config.logger


def join_pt_walk_graph(public_transport_g: nx.Graph, walk_g: nx.Graph, max_dist=20) -> nx.Graph:
    """
    Combine a public transport network graph with a pedestrian network graph, creating an intermodal transport graph.
    Platforms in the public transport network are connected to nearby pedestrian network edges based on the specified
    maximum distance.

    Parameters
    ----------
    public_transport_g : nx.Graph
        The public transport network graph, which should contain platform nodes with geometry and CRS information.
    walk_g : nx.Graph
        The pedestrian network graph. It must have the same CRS as the public transport graph.
    max_dist : float, optional
        Maximum distance (in meters) to search for connections between platforms and pedestrian edges. Defaults to 20.

    Returns
    -------
    nx.Graph
        A combined intermodal graph where public transport platforms are connected to nearby pedestrian routes.

    Raises
    ------
    AssertionError
        If the CRS of the public transport graph and pedestrian graph do not match.

    Examples
    --------
    >>> intermodal_graph = join_pt_walk_graph(public_transport_g, walk_g, max_dist=50)

    Notes
    -----
    The function relabels nodes in both graphs to ensure unique node IDs before composing them.
    It connects public transport platforms to the closest edges in the pedestrian network by projecting platforms
    onto edges.
    Walking speed is taken from the pedestrian graph, and default speed is set to 83.33 m/min if not available.
    """

    assert public_transport_g.graph["crs"] == walk_g.graph["crs"], "CRS mismatching."
    logger.info("Composing intermodal graph...")

    num_nodes_g1 = len(public_transport_g.nodes)
    mapping_g1 = {node: idx for idx, node in enumerate(public_transport_g.nodes)}
    mapping_g2 = {node: idx + num_nodes_g1 for idx, node in enumerate(walk_g.nodes)}
    transport: nx.DiGraph = nx.relabel_nodes(public_transport_g, mapping_g1)
    walk: nx.MultiDiGraph = nx.relabel_nodes(walk_g, mapping_g2)

    platforms = pd.DataFrame.from_dict(dict(transport.nodes(data=True)), orient="index")
    platforms = platforms[platforms["type"] == "platform"]
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
    projection_join["project_point"] = projection_join["edge_geometry"].interpolate(projection_join["project_dist"])
    projection_join["route"] = projection_join["route"].apply(lambda x: x if isinstance(x, list) else [x])

    logger.debug("searching for duplicated project points")
    projection_join = projection_join.groupby(by="project_point", as_index=False).agg(
        {
            "index": tuple,
            "index_right": "first",
            "geometry": "first",
            "u": "first",
            "v": "first",
            "k": "first",
            "project_dist": "first",
            "route": lambda x: [item for sublist in x for item in sublist],
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
    except KeyError:
        logger.warning(
            "There is no walk_speed in graph, set to the default speed - 83.33 m/min"
        )  #  посчитать примерную скорость по length timemin для любой эджи
        speed = 83.33

    edges_to_del = []

    for _, row in points_grouped_by_edge.iterrows():
        u, v, k = row[["u", "v", "k"]]
        edge: LineString = walk.get_edge_data(u, v, k)["geometry"]
        # Если платформа одна единственная на эдж
        if len(row["index"]) == 1:
            platform_id = row["index"][0]
            dist = edge.project(row.geometry[0])
            projected_point = edge.interpolate(dist)
            if dist == 0:
                # Если платформа проецируется на начало эджа
                mapping = {u: platform_id}
                nx.relabel_nodes(walk, mapping, copy=False)
            elif dist == edge.length:
                # Если на конец
                mapping = {v: platform_id}
                nx.relabel_nodes(walk, mapping, copy=False)
            else:
                line1 = substring(edge, 0, dist)
                line2 = substring(edge, dist, edge.length)
                # Убираем старые эджи и добавляем новые, в серединке нода - наша платформа для соединения
                edges_to_del.append((u, v, k))
                if u != v:
                    edges_to_del.append((v, u, k))
                walk.add_node(platform_id, x=projected_point.x, y=projected_point.y)
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
        # Если платформм несколько на эдж
        else:
            dist_project = [
                (edge.project(platform), edge.interpolate(edge.project(platform)), ind)
                for ind, platform in zip(row["index"], row.geometry)
            ]
            dist_project.sort(key=lambda x: x[0])
            u_to_del = u
            v_to_del = v
            last_dist = 0
            last_u = u
            for dist, projected_point, cur_index in dist_project:
                if dist == 0:
                    # Если платформа проецируется на начало эджа
                    mapping = {u: cur_index}
                    u_to_del = cur_index
                    nx.relabel_nodes(walk, mapping, copy=False)
                elif dist == edge.length:
                    # Если на конец
                    mapping = {v: cur_index}
                    v_to_del = cur_index
                    nx.relabel_nodes(walk, mapping, copy=False)
                else:
                    # По очереди добавляем ноды/линии
                    line = substring(edge, last_dist, dist)
                    if isinstance(line, Point):
                        raise ValueError(f"wtf {line} cannot be linestring")
                    walk.add_node(cur_index, x=projected_point.x, y=projected_point.y)
                    walk.add_edge(
                        last_u,
                        cur_index,
                        geometry=line,
                        length_meter=round(line.length, 3),
                        time_min=round(line.length / speed, 3),
                        type="walk",
                    )
                    walk.add_edge(
                        cur_index,
                        last_u,
                        geometry=line.reverse(),
                        length_meter=round(line.length, 3),
                        time_min=round(line.length / speed, 3),
                        type="walk",
                    )
                    last_u = cur_index
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
            edges_to_del.append((u_to_del, v_to_del, k))
            if u != v:
                edges_to_del.append((v_to_del, u_to_del, k))
    walk.remove_edges_from(edges_to_del)
    logger.debug("Composing graphs")
    intermodal = nx.compose(nx.MultiDiGraph(transport), nx.MultiDiGraph(walk))
    intermodal = remove_weakly_connected_nodes(intermodal)
    intermodal.remove_nodes_from([node for node, data in intermodal.nodes(data=True) if "x" not in data.keys()])
    mapping = {old_label: new_label for new_label, old_label in enumerate(intermodal.nodes())}
    logger.debug("Relabeling")
    nx.relabel_nodes(intermodal, mapping, copy=False)
    intermodal.graph["type"] = "intermodal"
    logger.debug("Done composing!")
    return intermodal
