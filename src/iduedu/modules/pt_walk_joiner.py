import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely import LineString, Point
from shapely.ops import substring

from iduedu import config

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

    transport = nx.relabel_nodes(public_transport_g, mapping_g1)
    walk = nx.relabel_nodes(walk_g, mapping_g2)

    platforms = pd.DataFrame.from_dict(dict(transport.nodes(data=True)), orient="index")
    platforms = platforms[platforms["desc"] == "platform"]
    platforms["geometry"] = platforms.apply(lambda x: Point(x.x, x.y), axis=1)
    platforms = gpd.GeoDataFrame(platforms, crs=transport.graph["crs"])
    walk_edges = gpd.GeoDataFrame(
        nx.to_pandas_edgelist(walk, source="u", target="v", edge_key="k"), crs=walk.graph["crs"]
    )

    projection_join = platforms.sjoin_nearest(walk_edges, max_distance=max_dist, distance_col="dist")

    points_grouped_by_edge = (
        projection_join.reset_index()
        .groupby("index_right", group_keys=True)
        .agg({"index": tuple, "geometry": tuple, "u": "first", "v": "first", "k": "first"})
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
        edge: LineString = walk_edges.loc[row.name].geometry

        # Если платформа одна единственная на эдж
        if len(row["index"]) == 1:
            dist = edge.project(row.geometry[0])
            projected_point = edge.interpolate(dist)
            if dist == 0:
                # Если платформа проецируется на начало эджа
                mapping = {u: row["index"][0]}
                nx.relabel_nodes(walk, mapping, copy=False)

            elif dist == edge.length:
                # Если на конец
                mapping = {v: row["index"][0]}
                nx.relabel_nodes(walk, mapping, copy=False)

            else:

                line1 = substring(edge, 0, dist)
                line2 = substring(edge, dist, edge.length)
                # Убираем старую эджу и добавляем новые, в серединке нода - наша платформа для соединения
                edges_to_del.append((u, v, k))
                walk.add_node(row["index"][0], x=projected_point.x, y=projected_point.y)
                walk.add_edge(
                    u,
                    row["index"][0],
                    geometry=line1,
                    length_meter=round(line1.length, 3),
                    time_min=round(line1.length / speed, 3),
                    type="walk",
                )
                walk.add_edge(
                    row["index"][0],
                    v,
                    geometry=line2,
                    length_meter=round(line2.length, 3),
                    time_min=round(line2.length / speed, 3),
                    type="walk",
                )
        # Если платформм несколько на эдж
        else:
            edges_to_del.append((u, v, k))
            dist_project = [
                (edge.project(platform), edge.interpolate(edge.project(platform)), ind)
                for ind, platform in zip(row["index"], row.geometry)
            ]
            dist_project.sort(key=lambda x: x[0])
            last_dist = 0
            last_u = u
            for dist, projected_point, cur_index in dist_project:

                if dist == 0:
                    # Если платформа проецируется на начало эджа
                    mapping = {u: cur_index}
                    nx.relabel_nodes(walk, mapping, copy=False)

                elif dist == edge.length:
                    # Если на конец
                    mapping = {v: cur_index}
                    nx.relabel_nodes(walk, mapping, copy=False)

                else:
                    # По очереди добавляем ноды/линии

                    line = substring(edge, last_dist, dist)
                    #
                    walk.add_node(cur_index, x=projected_point.x, y=projected_point.y)
                    walk.add_edge(
                        last_u,
                        cur_index,
                        geometry=line,
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
    walk.remove_edges_from(edges_to_del)
    intermodal = nx.compose(nx.MultiDiGraph(transport), nx.MultiDiGraph(walk))
    island_count = nx.number_weakly_connected_components(intermodal)
    if island_count > 1:
        logger.warning(
            f"Weakly connected components detected. {island_count} graph islands were removed. "
            f"These are probably pt routes that are not connected to walking routes. "
            f"You may try adjusting the max_dist parameter, but this can lead to incorrect data."
        )
    intermodal.remove_nodes_from([node for node, data in intermodal.nodes(data=True) if "x" not in data.keys()])
    components = sorted(nx.strongly_connected_components(intermodal), key=len)
    components = list(components)[:-1]
    intermodal.remove_nodes_from([node for comp in components for node in comp])
    mapping = {old_label: new_label for new_label, old_label in enumerate(intermodal.nodes())}
    intermodal = nx.relabel_nodes(intermodal, mapping)
    intermodal.graph["type"] = "intermodal"
    logger.debug("Done composing!")
    return intermodal
