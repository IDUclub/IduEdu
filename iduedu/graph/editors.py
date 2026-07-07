from dataclasses import dataclass
from typing import Any, Literal

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

from iduedu.graph.urban_graph import UrbanGraph


@dataclass(slots=True)
class UrbanGraphChanges:
    """Set of table changes that can be applied to an ``UrbanGraph``.

    The class is used as an intermediate result when projecting objects onto a
    graph. It stores new nodes, new edges and edge keys that should be removed from
    the source graph before split edge segments are appended.

    Args:
        edges_gdf: New graph edges.
        nodes_gdf: New graph nodes.
        edges_to_delete: Edge-key table for edges that should be removed.
        is_multigraph: Must match ``UrbanGraph.is_multigraph``.
        is_directed: Whether edge keys should be interpreted as directed ``u -> v``
            keys.

    Raises:
        TypeError: If change tables use unsupported types.
        ValueError: If change keys or geometries are invalid.
    """

    edges_gdf: gpd.GeoDataFrame | None = None
    nodes_gdf: gpd.GeoDataFrame | None = None
    edges_to_delete: pd.DataFrame | None = None
    is_multigraph: bool = False
    is_directed: bool = False

    def __post_init__(self) -> None:
        if self.nodes_gdf is not None:
            self._validate_nodes()
        if self.edges_gdf is not None:
            self._validate_edges()
        if self.edges_to_delete is not None:
            self._validate_edges_to_delete()

    def _edge_key_cols(self) -> list[str]:
        return ["u", "v", "k"] if self.is_multigraph else ["u", "v"]

    def _validate_nodes(self) -> None:
        nodes = self.nodes_gdf
        if not isinstance(nodes, gpd.GeoDataFrame):
            raise TypeError(f"nodes_gdf must be GeoDataFrame, got {type(nodes).__name__}")
        if nodes.empty:
            raise ValueError("nodes_gdf is empty")
        if nodes.index.has_duplicates:
            raise ValueError("nodes_gdf.index must be unique")
        if nodes.geometry.isna().any():
            raise ValueError("nodes_gdf.geometry contains NaN")
        if (~nodes.geometry.geom_type.isin(["Point"])).any():
            raise ValueError("All nodes_gdf geometries must be Point")

    def _validate_edges(self) -> None:
        edges = self.edges_gdf
        if not isinstance(edges, gpd.GeoDataFrame):
            raise TypeError(f"edges_gdf must be GeoDataFrame, got {type(edges).__name__}")
        if edges.empty:
            raise ValueError("edges_gdf is empty")

        key_cols = self._edge_key_cols()
        required = set(key_cols + ["geometry"])
        missing = required - set(edges.columns)
        if missing:
            raise ValueError(f"edges_gdf missing required columns: {sorted(missing)}")

        if edges[key_cols].isna().any().any():
            raise ValueError(f"edges_gdf columns {key_cols} must not contain NaN")
        if edges[key_cols].duplicated().any():
            raise ValueError(f"edges_gdf must have unique {key_cols} keys")
        if (~edges.geometry.geom_type.isin(["LineString"])).any():
            raise ValueError("All edges_gdf geometries must be LineString")

    def _validate_edges_to_delete(self) -> None:
        etd = self.edges_to_delete
        if not isinstance(etd, pd.DataFrame):
            raise TypeError(f"edges_to_delete must be DataFrame, got {type(etd).__name__}")
        if etd.empty:
            return
        key_cols = self._edge_key_cols()
        missing = set(key_cols) - set(etd.columns)
        if missing:
            raise ValueError(f"edges_to_delete missing required columns: {sorted(missing)}")
        if etd[key_cols].isna().any().any():
            raise ValueError(f"edges_to_delete columns {key_cols} must not contain NaN")


def relabel_urban_graph(graph_gdf: UrbanGraph) -> UrbanGraph:
    """
    Relabel node ids to a dense ``RangeIndex`` and update edge endpoints.

    The function does not mutate the source graph. For multigraphs, ``k`` is
    recalculated after relabeling to keep ``['u', 'v', 'k']`` unique.

    See also:
        https://iduclub.github.io/IduEdu/examples/graph_operations.html
    """

    nodes = graph_gdf.nodes_gdf.copy()
    edges = graph_gdf.edges_gdf.copy()

    mapping = {old: new for new, old in enumerate(nodes.index)}
    nodes.index = pd.RangeIndex(len(nodes))

    if not edges.empty:
        edges["u"] = edges["u"].map(mapping)
        edges["v"] = edges["v"].map(mapping)
        edges = edges.dropna(subset=["u", "v"]).copy()

        if not edges.empty:
            edges[["u", "v"]] = edges[["u", "v"]].astype(int)
            if graph_gdf.is_multigraph:
                edges["k"] = edges.groupby(["u", "v"], sort=False).cumcount()
            edges = edges.reset_index(drop=True)

    relabeled_graph = UrbanGraph(
        nodes_gdf=nodes,
        edges_gdf=edges,
        is_multigraph=graph_gdf.is_multigraph,
        is_directed=graph_gdf.is_directed,
        edge_direction_column=graph_gdf.edge_direction_column,
        adjacency_weight=graph_gdf.adjacency_weight,
        crs=graph_gdf.crs,
        graph_type=graph_gdf.type,
    )
    return relabeled_graph


def subgraph_by_nodes(graph_gdf: UrbanGraph, nodes) -> UrbanGraph:
    """
    Return an induced ``UrbanGraph`` subgraph for the provided node ids.

    Node ids are preserved. Edges are retained only when both endpoints are in
    ``nodes``.

    See also:
        https://iduclub.github.io/IduEdu/examples/graph_operations.html
    """

    if not isinstance(graph_gdf, UrbanGraph):
        raise TypeError(f"graph_gdf must be UrbanGraph, got {type(graph_gdf).__name__}")

    nodes_index = pd.Index(nodes)
    graph_nodes = graph_gdf.nodes_gdf
    graph_edges = graph_gdf.edges_gdf
    missing_nodes = nodes_index.difference(graph_nodes.index)

    if len(missing_nodes) > 0:
        raise ValueError(f"nodes contain ids absent in graph.nodes_gdf.index: {missing_nodes[:10].tolist()}")

    sub_nodes = graph_nodes.loc[graph_nodes.index.isin(nodes_index)].copy()

    if graph_edges.empty:
        sub_edges = graph_edges.copy()
    else:
        sub_node_ids = sub_nodes.index
        sub_edges = graph_edges.loc[graph_edges["u"].isin(sub_node_ids) & graph_edges["v"].isin(sub_node_ids)].copy()
        sub_edges = sub_edges.reset_index(drop=True)

    return UrbanGraph(
        nodes_gdf=sub_nodes,
        edges_gdf=sub_edges,
        is_multigraph=graph_gdf.is_multigraph,
        is_directed=graph_gdf.is_directed,
        edge_direction_column=graph_gdf.edge_direction_column,
        adjacency_weight=graph_gdf.adjacency_weight,
        crs=graph_gdf.crs,
        graph_type=graph_gdf.type,
    )


def clip_urban_graph(graph_gdf: UrbanGraph, polygon: BaseGeometry) -> UrbanGraph:
    """
    Clip graph nodes by geometry and keep only edges with retained endpoints.

    ``polygon`` is expected to be in the same CRS as ``graph_gdf``. Node ids are
    preserved; call :func:`relabel_urban_graph` if dense labels are needed.

    See also:
        https://iduclub.github.io/IduEdu/examples/graph_operations.html
    """

    nodes = graph_gdf.nodes_gdf.copy()
    edges = graph_gdf.edges_gdf.copy()

    if not isinstance(nodes, gpd.GeoDataFrame):
        raise TypeError(f"graph.nodes_gdf must be GeoDataFrame, got {type(nodes).__name__}")
    if not isinstance(polygon, BaseGeometry):
        raise TypeError(f"polygon must be a shapely geometry, got {type(polygon).__name__}")

    nodes = nodes.clip(polygon, keep_geom_type=True)

    if not edges.empty:
        node_ids = nodes.index
        edges = edges.loc[edges["u"].isin(node_ids) & edges["v"].isin(node_ids)].copy()
        edges = edges.reset_index(drop=True)

    clipped_graph = UrbanGraph(
        nodes_gdf=nodes,
        edges_gdf=edges,
        is_multigraph=graph_gdf.is_multigraph,
        is_directed=graph_gdf.is_directed,
        edge_direction_column=graph_gdf.edge_direction_column,
        adjacency_weight=graph_gdf.adjacency_weight,
        crs=graph_gdf.crs,
        graph_type=graph_gdf.type,
    )
    return clipped_graph


def join_urban_graphs(
    left: UrbanGraph,
    right: UrbanGraph,
    *,
    graph_type: str | None = None,
    node_conflict: Literal["left", "right"] = "left",
) -> UrbanGraph:
    """
    Join two compatible ``UrbanGraph`` objects by concatenating node and edge tables.

    Shared node indexes are allowed and are merged according to
    ``node_conflict``. Edge keys must be unique across both graphs because
    duplicate edges are ambiguous at this layer.

    See also:
        https://iduclub.github.io/IduEdu/examples/graph_operations.html
    """

    if not isinstance(left, UrbanGraph):
        raise TypeError(f"left must be UrbanGraph, got {type(left).__name__}")
    if not isinstance(right, UrbanGraph):
        raise TypeError(f"right must be UrbanGraph, got {type(right).__name__}")
    if left.crs != right.crs:
        raise ValueError(f"CRS mismatch: left.crs={left.crs}, right.crs={right.crs}")
    if left.is_multigraph != right.is_multigraph:
        raise ValueError("left.is_multigraph and right.is_multigraph mismatch")
    if left.is_directed != right.is_directed:
        raise ValueError("left.is_directed and right.is_directed mismatch")
    if left.edge_direction_column != right.edge_direction_column:
        raise ValueError("left.edge_direction_column and right.edge_direction_column mismatch")
    if node_conflict not in {"left", "right"}:
        raise ValueError(f"node_conflict must be 'left' or 'right', got {node_conflict!r}")

    key_cols = ["u", "v", "k"] if left.is_multigraph else ["u", "v"]

    def _edge_keys(graph: UrbanGraph) -> pd.DataFrame:
        if graph.edges_gdf.empty:
            return pd.DataFrame(columns=key_cols)
        return graph.edges_gdf[key_cols].copy()

    edge_keys = pd.concat([_edge_keys(left), _edge_keys(right)], ignore_index=True)
    duplicated_edge_keys = edge_keys[edge_keys.duplicated(keep=False)]
    if not duplicated_edge_keys.empty:
        duplicated_records = duplicated_edge_keys.drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"Duplicate edge keys across graphs: {duplicated_records}")

    node_parts = [left.nodes_gdf, right.nodes_gdf] if node_conflict == "left" else [right.nodes_gdf, left.nodes_gdf]
    if left.nodes_gdf.empty:
        nodes = right.nodes_gdf.copy()
    elif right.nodes_gdf.empty:
        nodes = left.nodes_gdf.copy()
    else:
        nodes = gpd.GeoDataFrame(
            pd.concat(node_parts, axis=0, sort=False),
            geometry=left.nodes_gdf.geometry.name,
            crs=left.crs,
        )
    nodes = nodes.loc[~nodes.index.duplicated(keep="first")].copy()

    if left.edges_gdf.empty:
        edges = right.edges_gdf.copy()
    elif right.edges_gdf.empty:
        edges = left.edges_gdf.copy()
    else:
        edges = gpd.GeoDataFrame(
            pd.concat([left.edges_gdf, right.edges_gdf], axis=0, ignore_index=True, sort=False),
            geometry=left.edges_gdf.geometry.name,
            crs=left.crs,
        )

    return UrbanGraph(
        nodes_gdf=nodes,
        edges_gdf=edges,
        is_multigraph=left.is_multigraph,
        is_directed=left.is_directed,
        edge_direction_column=left.edge_direction_column,
        adjacency_weight=left.adjacency_weight,
        crs=left.crs,
        graph_type=graph_type if graph_type is not None else left.type,
    )


def project_objects2urban_graph(
    graph_gdf: UrbanGraph,
    objects_gdf: gpd.GeoDataFrame,
    speed_m_per_min: float,
    *,
    max_dist: float | None = None,
    add_link_edge: bool = True,
) -> tuple[UrbanGraphChanges, pd.Series]:
    """Prepare graph changes that connect objects to their nearest edges.

    The function does not mutate the source graph. It computes nodes and edges to
    add, source edges to remove, and an ``object2node`` mapping. In backend services
    this is the preferred low-level workflow: ``changes`` can be persisted to graph
    tables and ``object2node_map`` can be saved on the object table as
    ``graph_node_id``. For in-memory workflows, apply the result with
    :func:`apply_urban_graph_changes`.

    Objects are projected by their ``representative_point()``, so the function works
    for buildings, services and other polygon or point layers with a stable unique
    index.

    Args:
        graph_gdf: Source urban graph.
        objects_gdf: Objects to connect to the graph. The index should be a stable
            object identifier, such as ``object_id`` or ``building_id``.
        speed_m_per_min: Movement speed on created connector edges, in meters per
            minute. For 5 km/h use ``5 * 1000 / 60``.
        max_dist: Optional maximum distance from an object to the nearest edge. If
            ``None``, nearest edges are searched without a distance limit.
        add_link_edge: If ``True``, create a dedicated object node and connector
            edge. If ``False``, map the object to the projection node on the graph.

    Returns:
        Pair ``(changes, object2node_map)``. ``changes.nodes_gdf`` contains new
        nodes, ``changes.edges_gdf`` contains new edges, and
        ``changes.edges_to_delete`` contains replaced edge keys. ``object2node_map``
        is a ``Series`` indexed by the original object index with graph node ids as
        values.

    Raises:
        TypeError: If ``objects_gdf`` is not a GeoDataFrame.
        ValueError: If objects are empty or speed/distance parameters are invalid.

    See also:
        https://iduclub.github.io/IduEdu/examples/objects_and_nearest_nodes.html
    """

    gdf_edges = graph_gdf.edges_gdf.copy()
    gdf_nodes = graph_gdf.nodes_gdf.copy()

    is_multigraph = graph_gdf.is_multigraph
    is_directed = graph_gdf.is_directed
    edge_direction_column = graph_gdf.edge_direction_column

    if not isinstance(objects_gdf, gpd.GeoDataFrame):
        raise TypeError(f"objects_gdf must be GeoDataFrame, got {type(objects_gdf).__name__}")
    if objects_gdf.empty:
        raise ValueError("objects_gdf is empty")
    if objects_gdf.index.has_duplicates:
        raise ValueError("objects_gdf.index must be unique")
    if speed_m_per_min <= 0:
        raise ValueError(f"default_speed_m_per_min must be > 0, got {speed_m_per_min}")
    if max_dist is not None and max_dist <= 0:
        raise ValueError(f"max_dist must be > 0 or None, got {max_dist}")

    original_crs = graph_gdf.crs or gdf_edges.crs or gdf_nodes.crs
    local_crs = gdf_nodes.estimate_utm_crs()

    if gdf_edges.crs != local_crs:
        gdf_edges = gdf_edges.to_crs(local_crs)
    if gdf_nodes.crs != local_crs:
        gdf_nodes = gdf_nodes.to_crs(local_crs)
    if objects_gdf.crs != local_crs:
        objects_gdf = objects_gdf.to_crs(local_crs)

    rep_gdf = objects_gdf[["geometry"]].copy()
    rep_gdf.geometry = rep_gdf.geometry.representative_point()

    new_nodes_records: list[dict[str, Any]] = []
    new_edges_records: list[dict[str, Any]] = []
    edges_to_delete_records: list[dict[str, Any]] = []
    object2node_records: list[dict[str, Any]] = []

    next_node_id = max(gdf_nodes.index) + 1 if len(gdf_nodes) else 0

    def _edge_record(u, v, geometry: LineString, k=0, attrs: dict[str, Any] | None = None) -> dict[str, Any]:
        record = dict(attrs) if attrs is not None else {}
        length = round(geometry.length, 3)
        time = round(length / speed_m_per_min, 2)
        record.update({"u": u, "v": v, "length_meter": length, "time_min": time, "geometry": geometry})
        if is_multigraph:
            record["k"] = k
        return record

    def _add_connector_edges(object_node, connect_node, object_point: Point, connect_point: Point) -> None:
        object_to_graph = LineString([object_point, connect_point])
        if object_to_graph.length <= 0:
            return
        connector_attrs = {edge_direction_column: False} if edge_direction_column is not None else None
        new_edges_records.append(_edge_record(object_node, connect_node, object_to_graph, attrs=connector_attrs))
        if is_directed and edge_direction_column is None:
            graph_to_object = LineString([connect_point, object_point])
            new_edges_records.append(_edge_record(connect_node, object_node, graph_to_object))

    def _object2node_map_from_records() -> pd.Series:
        if not object2node_records:
            return pd.Series(dtype=int)
        object2node = (
            pd.DataFrame(object2node_records)
            .astype({"object_index": int, "node_id": int})
            .drop_duplicates(subset=["object_index"], keep="first")
            .set_index("object_index")["node_id"]
        )
        return object2node

    gdf_edges["edge_geometry"] = gdf_edges["geometry"]
    projection_join = rep_gdf.reset_index(names="object_index").sjoin_nearest(
        gdf_edges, how="left", max_distance=max_dist
    )
    projection_join = projection_join.rename_geometry("object_geometry")
    projection_join = projection_join.rename(columns={"index_right": "edge_index"})
    projection_join = projection_join.dropna(subset=["edge_index"]).copy()

    if projection_join.empty:
        key_cols = ["u", "v", "k"] if is_multigraph else ["u", "v"]
        return (
            UrbanGraphChanges(
                edges_to_delete=pd.DataFrame(columns=key_cols),
                is_multigraph=is_multigraph,
                is_directed=is_directed,
            ),
            pd.Series(dtype=int),
        )

    projection_join["project_dist"] = projection_join["edge_geometry"].project(projection_join["object_geometry"])
    edge_lengths = projection_join["edge_geometry"].length
    endpoint_snap_tolerance = (1e-5 * edge_lengths).clip(lower=1e-6)

    start_mask = projection_join["project_dist"].abs() <= endpoint_snap_tolerance
    end_mask = (projection_join["project_dist"] - edge_lengths).abs() <= endpoint_snap_tolerance

    projection_join["node2connect"] = pd.NA
    projection_join.loc[start_mask, "node2connect"] = projection_join.loc[start_mask, "u"]
    projection_join.loc[end_mask, "node2connect"] = projection_join.loc[end_mask, "v"]

    endpoint_candidate = projection_join["node2connect"].notna()
    endpoint_object_index = projection_join.loc[endpoint_candidate, "object_index"].drop_duplicates()

    if not endpoint_object_index.empty:
        endpoint_rows = projection_join[projection_join["object_index"].isin(endpoint_object_index)]

        mixed_projection_object_index = endpoint_rows.loc[endpoint_rows["node2connect"].isna(), "object_index"].unique()
        if len(mixed_projection_object_index) > 0:
            raise ValueError(
                "Some objects have both endpoint and non-endpoint nearest edges: "
                f"{mixed_projection_object_index[:10].tolist()}"
            )

        nodes_per_object = endpoint_rows.groupby("object_index", sort=False)["node2connect"].nunique(dropna=True)
        bad_object_index = nodes_per_object[nodes_per_object != 1].index
        if len(bad_object_index) > 0:
            raise ValueError(
                "Some endpoint-projected objects connect to multiple graph nodes: " f"{bad_object_index[:10].tolist()}"
            )

        object_to_existing_node = endpoint_rows.groupby("object_index", sort=False)["node2connect"].first()

        # Objects projected to edge endpoints are attached to existing graph nodes.

        for object_index, connect_node in object_to_existing_node.items():
            object_point = rep_gdf.loc[object_index, "geometry"]
            connect_point = gdf_nodes.loc[connect_node, "geometry"]

            if add_link_edge:
                object_node = next_node_id
                next_node_id += 1
                new_nodes_records.append(
                    {"node_id": object_node, "object_index": object_index, "geometry": object_point}
                )
                _add_connector_edges(object_node, connect_node, object_point, connect_point)
                object2node_records.append({"object_index": object_index, "node_id": object_node})
            else:
                object2node_records.append({"object_index": object_index, "node_id": connect_node})

        projection_join = projection_join[~projection_join["object_index"].isin(endpoint_object_index)].copy()

    if projection_join.empty:
        new_nodes = gpd.GeoDataFrame(new_nodes_records, geometry="geometry", crs=local_crs)
        if not new_nodes.empty:
            new_nodes = new_nodes.set_index("node_id", drop=True).drop(columns="object_index", errors="ignore")
            new_nodes = new_nodes.to_crs(original_crs)
        object2node_map = _object2node_map_from_records()
        return (
            UrbanGraphChanges(
                nodes_gdf=new_nodes if not new_nodes.empty else None,
                edges_to_delete=pd.DataFrame(columns=["u", "v", "k"] if is_multigraph else ["u", "v"]),
                is_multigraph=is_multigraph,
                is_directed=is_directed,
            ),
            object2node_map,
        )

    projection_join["project_point"] = gpd.GeoSeries(
        projection_join["edge_geometry"].interpolate(projection_join["project_dist"]), crs=local_crs
    )
    projection_join["groupby_point"] = projection_join["project_point"].set_precision(10)
    projection_join["object_pair"] = list(zip(projection_join["object_index"], projection_join["object_geometry"]))

    projection_join["projection_group"] = projection_join.groupby(["groupby_point", "edge_index"], sort=False).ngroup()

    grouped_proj_points = projection_join.groupby(by="projection_group", as_index=False, sort=False).agg(
        {
            "project_point": "first",
            "project_dist": "first",
            "object_pair": lambda x: list(dict.fromkeys(x)),
        }
    )

    new_projection_count = len(grouped_proj_points)
    grouped_proj_points["projection_node_id"] = range(next_node_id, next_node_id + new_projection_count)
    next_node_id += new_projection_count

    grouped_proj_points["projection_node_id"] = grouped_proj_points["projection_node_id"].astype(int)
    projection_join = projection_join.merge(
        grouped_proj_points[["projection_group", "projection_node_id", "project_dist"]],
        on="projection_group",
        how="left",
        suffixes=("", "_group"),
    )
    projection_join["project_dist"] = projection_join["project_dist_group"]
    projection_join = projection_join.drop(columns="project_dist_group")

    # Attach grouped objects to the first projection node in each group.

    for _, row in grouped_proj_points.iterrows():
        projection_node = row.projection_node_id
        projection_point: Point = row.project_point
        new_nodes_records.append({"node_id": projection_node, "geometry": projection_point})
        for object_index, object_point in row["object_pair"]:
            if add_link_edge:
                object_node = next_node_id
                next_node_id += 1
                new_nodes_records.append(
                    {"node_id": object_node, "object_index": object_index, "geometry": object_point}
                )
                _add_connector_edges(object_node, projection_node, object_point, projection_point)
                object2node_records.append({"object_index": object_index, "node_id": object_node})
            else:
                object2node_records.append({"object_index": object_index, "node_id": projection_node})

    edge_group_agg = {
        "projection_node_id": tuple,
        "project_dist": tuple,
    }
    edge_lookup_cols = ["u", "v", "edge_geometry"]
    if is_multigraph:
        edge_lookup_cols.append("k")

    points_grouped_by_edge = projection_join.groupby(by="edge_index", as_index=False).agg(edge_group_agg)
    points_grouped_by_edge = points_grouped_by_edge.merge(
        gdf_edges[edge_lookup_cols],
        left_on="edge_index",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    for _, row in points_grouped_by_edge.iterrows():
        u = row["u"]
        v = row["v"]
        edge: LineString = row["edge_geometry"]
        edge_key = row["k"] if is_multigraph else 0
        split_attrs = {"edge_index": row["edge_index"]}

        if is_multigraph:
            edges_to_delete_records.append({"u": u, "v": v, "k": edge_key})
        else:
            edges_to_delete_records.append({"u": u, "v": v})

        eps = 1e-6 * edge.length

        dist_project = sorted(zip(row["project_dist"], row["projection_node_id"]), key=lambda x: x[0])

        grouped: list[list[Any]] = []
        for dist, node_id in dist_project:
            if not grouped:
                grouped.append([dist, [node_id]])
                continue

            last_dist, node_ids = grouped[-1]
            if abs(dist - last_dist) <= eps:
                node_ids.append(node_id)
            else:
                grouped.append([dist, [node_id]])

        last_dist = 0.0
        last_u = u

        for dist, node_ids in grouped:
            main_node = node_ids[0]
            if abs(dist) <= eps:
                last_u = u
                last_dist = 0.0
                continue

            if abs(dist - edge.length) <= eps:
                line = substring(edge, last_dist, edge.length)
                if line.length > 0 and last_u != v:
                    new_edges_records.append(_edge_record(last_u, v, line, k=edge_key, attrs=split_attrs))

                last_u = v
                last_dist = edge.length
                continue

            line = substring(edge, last_dist, dist)
            if line.length > 0 and last_u != main_node:
                new_edges_records.append(_edge_record(last_u, main_node, line, k=edge_key, attrs=split_attrs))

            last_u = main_node
            last_dist = dist

        if last_dist < edge.length:
            line = substring(edge, last_dist, edge.length)
            if line.length > 0 and last_u != v:
                new_edges_records.append(_edge_record(last_u, v, line, k=edge_key, attrs=split_attrs))

    new_nodes = gpd.GeoDataFrame(new_nodes_records, geometry="geometry", crs=local_crs)

    if object2node_records:
        object2node_map = _object2node_map_from_records()
    elif not new_nodes.empty and "object_index" in new_nodes.columns:
        object2node_map = (
            new_nodes[["node_id", "object_index"]]
            .dropna(subset=["object_index"])
            .astype({"object_index": int, "node_id": int})
            .set_index("object_index")["node_id"]
        )
    else:
        object2node_map = pd.Series(dtype=int)

    if not new_nodes.empty:
        new_nodes = new_nodes.set_index("node_id", drop=True)
        if "object_index" in new_nodes.columns:
            new_nodes = new_nodes.drop(columns="object_index")

    new_edges = gpd.GeoDataFrame(new_edges_records, geometry="geometry", crs=local_crs)
    if not new_edges.empty and "edge_index" in new_edges.columns:
        changed_edge_columns = {"u", "v", "k", "geometry", "edge_geometry", "length_meter", "time_min", "edge_index"}
        edge_attr_cols = [col for col in gdf_edges.columns if col not in changed_edge_columns]
        edge_attrs = gdf_edges[edge_attr_cols].reset_index(names="edge_index")
        new_edges = new_edges.merge(edge_attrs, on="edge_index", how="left", suffixes=("", "_source"))

        for col in edge_attr_cols:
            source_col = f"{col}_source"
            if source_col in new_edges.columns:
                new_edges[col] = new_edges[col].combine_first(new_edges[source_col])
                new_edges = new_edges.drop(columns=source_col)
        new_edges = new_edges.drop(columns="edge_index")
        new_edges = gpd.GeoDataFrame(new_edges, geometry="geometry", crs=local_crs)
    edges_to_delete = pd.DataFrame(edges_to_delete_records).drop_duplicates()

    new_edges = new_edges.to_crs(original_crs)
    new_nodes = new_nodes.to_crs(original_crs)

    return (
        UrbanGraphChanges(
            edges_gdf=new_edges if not new_edges.empty else None,
            nodes_gdf=new_nodes if not new_nodes.empty else None,
            edges_to_delete=edges_to_delete,
            is_multigraph=is_multigraph,
            is_directed=is_directed,
        ),
        object2node_map,
    )


def apply_urban_graph_changes(graph_gdf: UrbanGraph, changes: UrbanGraphChanges) -> UrbanGraph:
    """Apply prepared changes to an ``UrbanGraph``.

    The function removes edges listed in ``changes.edges_to_delete``, appends new
    nodes and edges, and returns a new graph. Any previously cached adjacency matrix
    is considered stale after applying structural changes; rebuild it before running
    routing or accessibility calculations.

    Args:
        graph_gdf: Source graph.
        changes: Changes prepared by :func:`project_objects2urban_graph` or created
            manually.

    Returns:
        New ``UrbanGraph`` with changes applied.

    Raises:
        ValueError: If graph and change topology flags differ or if
            ``edges_to_delete`` references edges that are absent from the graph.

    See also:
        https://iduclub.github.io/IduEdu/examples/objects_and_nearest_nodes.html
    """

    if graph_gdf.is_multigraph != changes.is_multigraph:
        raise ValueError("graph.is_multigraph and changes.is_multigraph mismatch")
    if graph_gdf.is_directed != changes.is_directed:
        raise ValueError("graph.is_directed and changes.is_directed mismatch")

    nodes = graph_gdf.nodes_gdf.copy()
    edges = graph_gdf.edges_gdf.copy()

    new_nodes = changes.nodes_gdf
    new_edges = changes.edges_gdf
    edges_to_delete = changes.edges_to_delete

    key_cols = ["u", "v", "k"] if graph_gdf.is_multigraph else ["u", "v"]

    if edges_to_delete is not None and not edges_to_delete.empty:

        delete_keys = edges_to_delete[key_cols].drop_duplicates()
        edge_keys = edges[key_cols].drop_duplicates()

        missing_delete_keys = delete_keys.merge(edge_keys, on=key_cols, how="left", indicator=True)

        missing_delete_keys = missing_delete_keys[missing_delete_keys["_merge"] == "left_only"]

        if not missing_delete_keys.empty:
            raise ValueError(
                "Some edges_to_delete are not present in graph.edges_gdf: "
                f"{missing_delete_keys[key_cols].head(10).to_dict('records')}"
            )

        edges = edges.merge(delete_keys.assign(_delete_marker=True), on=key_cols, how="left")
        edges = edges[edges["_delete_marker"].isna()].drop(columns="_delete_marker")

    if new_nodes is not None and not new_nodes.empty:
        intersect = nodes.index.intersection(new_nodes.index)
        if len(intersect) > 0:
            raise ValueError(f"Node id collision: {intersect[:10].tolist()}")

        nodes = pd.concat([nodes, new_nodes], axis=0)

    if new_edges is not None and not new_edges.empty:
        edges = pd.concat([edges, new_edges], axis=0, ignore_index=True)

    return UrbanGraph(
        nodes_gdf=nodes,
        edges_gdf=edges,
        is_multigraph=graph_gdf.is_multigraph,
        is_directed=graph_gdf.is_directed,
        edge_direction_column=graph_gdf.edge_direction_column,
        adjacency_weight=graph_gdf.adjacency_weight,
        crs=graph_gdf.crs,
        graph_type=graph_gdf.type,
    )
