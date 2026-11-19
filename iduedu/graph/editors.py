import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

from iduedu import _rust
from iduedu.graph.routes import join_routes
from iduedu.graph.urban_graph import UrbanGraph

# Objects closer than this to their attachment point are mapped onto the existing node instead of
# getting a dedicated node: a shorter connector rounds to a zero length edge and cannot connect
# anything, which would leave the object node isolated.
MIN_CONNECTOR_LENGTH_M = 1e-3


@dataclass(slots=True)
class UrbanGraphChanges:
    """Set of table changes that can be applied to an ``UrbanGraph``.

    The class is used as an intermediate result when projecting objects onto a
    graph or resolving geometric edits. It stores new nodes, new edges, edge keys that
    should be removed from the source graph and new attribute values of kept edges.

    Args:
        edges_gdf: New graph edges.
        nodes_gdf: New graph nodes.
        edges_to_delete: Edge-key table for edges that should be removed.
        edges_to_update: Edge keys with new attribute values: one row per edge, the key
            columns and any other columns except ``geometry``. A missing value (NaN) keeps
            the old value of the edge; a column absent from the graph is added.
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
    edges_to_update: pd.DataFrame | None = None
    is_multigraph: bool = False
    is_directed: bool = False

    def __post_init__(self) -> None:
        if self.nodes_gdf is not None:
            self._validate_nodes()
        if self.edges_gdf is not None:
            self._validate_edges()
        if self.edges_to_delete is not None:
            self._validate_edges_to_delete()
        if self.edges_to_update is not None:
            self._validate_edges_to_update()

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

    def _validate_edges_to_update(self) -> None:
        etu = self.edges_to_update
        if not isinstance(etu, pd.DataFrame):
            raise TypeError(f"edges_to_update must be DataFrame, got {type(etu).__name__}")
        if etu.empty:
            return
        key_cols = self._edge_key_cols()
        missing = set(key_cols) - set(etu.columns)
        if missing:
            raise ValueError(f"edges_to_update missing required columns: {sorted(missing)}")
        if etu[key_cols].isna().any().any():
            raise ValueError(f"edges_to_update columns {key_cols} must not contain NaN")
        if etu[key_cols].duplicated().any():
            raise ValueError(f"edges_to_update must have unique {key_cols} keys")
        forbidden = {"geometry", "u", "v", "k"} & (set(etu.columns) - set(key_cols))
        if isinstance(etu, gpd.GeoDataFrame) or forbidden:
            raise ValueError(f"edges_to_update cannot change {sorted(forbidden) or ['geometry']}")
        if len(etu.columns) == len(key_cols):
            raise ValueError("edges_to_update has no value columns")


# Packed coordinate ids: latitude and longitude in units of 1e-7 degree (the precision of OSM), shifted to be
# non-negative and combined as lat * LON_SPAN + lon, which stays below 2**63.
_COORD_SCALE = 10_000_000
_LAT_SHIFT = 90 * _COORD_SCALE
_LON_SHIFT = 180 * _COORD_SCALE
_LON_SPAN = 2 * _LON_SHIFT + 1


def coordinate_node_ids(points: gpd.GeoSeries | gpd.GeoDataFrame) -> np.ndarray:
    """Stable int64 node ids packed from the coordinates of points.

    The longitude and latitude in EPSG:4326 are rounded to 1e-7 degree (the precision of OpenStreetMap), shifted
    to be non-negative and combined as ``(lat + 90°) * 3_600_000_001 + (lon + 180°)`` in units of 1e-7 degree, so
    ``divmod(node_id, 3_600_000_001)`` gives them back. The ids are below ``2**63`` and never negative.

    Args:
        points: Points with a CRS, for example ``graph.nodes_gdf``.

    Returns:
        Array of int64 ids in the order of ``points``.

    Raises:
        ValueError: If the points have no CRS.
    """
    if isinstance(points, gpd.GeoDataFrame):
        points = points.geometry
    if points.crs is None:
        raise ValueError("coordinate node ids need points with a CRS")
    lonlat = points.to_crs(4326) if not points.crs.equals("EPSG:4326") else points
    lon = np.rint(lonlat.x.to_numpy() * _COORD_SCALE).astype(np.int64) + _LON_SHIFT
    lat = np.rint(lonlat.y.to_numpy() * _COORD_SCALE).astype(np.int64) + _LAT_SHIFT
    return lat * _LON_SPAN + lon


def _hashed_node_ids(nodes: gpd.GeoDataFrame, coordinate_ids: np.ndarray, shared: np.ndarray) -> np.ndarray:
    """Negative ids of nodes that share a coordinate id, from that id, their attributes and their rank among equals."""
    columns = sorted((c for c in nodes.columns if c != nodes.geometry.name), key=str)
    rows = nodes.loc[shared, columns].itertuples(index=False, name=None) if columns else iter(())
    keys = [
        f"{cid}|{json.dumps(list(row), default=str, ensure_ascii=False)}"
        for cid, row in zip(coordinate_ids[shared], rows if columns else ([] for _ in range(int(shared.sum()))))
    ]
    rank = pd.Series(keys).groupby(keys, sort=False).cumcount().to_numpy()
    ids = np.empty(len(keys), dtype=np.int64)
    for i, (key, r) in enumerate(zip(keys, rank)):
        digest = hashlib.blake2b(f"{key}|{r}".encode(), digest_size=8).digest()
        ids[i] = -(int.from_bytes(digest, "big") & (2**63 - 1)) - 1
    return ids


def relabel_urban_graph(
    graph_gdf: UrbanGraph,
    *,
    by: Literal["range", "coordinates"] = "range",
    on_duplicate: Literal["raise", "hash"] = "raise",
) -> UrbanGraph:
    """
    Relabel node ids and update edge endpoints.

    The function does not mutate the source graph. For multigraphs, ``k`` is
    recalculated after relabeling to keep ``['u', 'v', 'k']`` unique.

    Args:
        graph_gdf: Source graph.
        by: ``"range"`` numbers the nodes ``0 .. n - 1`` in their order. ``"coordinates"``
            packs the latitude and longitude of every node, rounded to 1e-7 degree (the
            precision of OpenStreetMap), into one int64 id. Such ids do not change when the
            graph is built again from newer data, and graphs of neighbouring territories get
            equal ids for their shared nodes (see :func:`coordinate_node_ids`).
        on_duplicate: What ``by="coordinates"`` does with nodes closer than 1e-7 degree (about
            1 cm), which would get one id, such as the stops of several routes at one platform
            in a public transport graph. ``"raise"`` raises ``ValueError``. ``"hash"`` gives
            each of them a negative id hashed from the coordinate id, the node attributes and
            the rank among nodes with equal attributes, so the ids stay stable when the graph
            is built again in the same way; the other nodes keep their coordinate ids.

    Raises:
        ValueError: If ``by="coordinates"`` and the graph has no CRS, or nodes share a
            coordinate id and ``on_duplicate="raise"``.

    See also:
        https://iduclub.github.io/IduEdu/examples/graph_operations.html
    """

    nodes = graph_gdf.nodes_gdf.copy()
    edges = graph_gdf.edges_gdf.copy()

    if by == "range":
        new_index = pd.RangeIndex(len(nodes))
    elif by == "coordinates":
        if on_duplicate not in ("raise", "hash"):
            raise ValueError(f"on_duplicate must be 'raise' or 'hash', got {on_duplicate!r}")
        geometry = nodes.geometry if nodes.crs is not None else nodes.geometry.set_crs(graph_gdf.crs)
        ids = coordinate_node_ids(geometry)
        shared = pd.Index(ids).duplicated(keep=False)
        if shared.any():
            if on_duplicate == "raise":
                examples = nodes.index[shared][:10].tolist()
                raise ValueError(
                    f"Nodes closer than 1e-7 degree would share a coordinate id: {examples}; " "use on_duplicate='hash'"
                )
            ids[shared] = _hashed_node_ids(nodes, ids, shared)
            if pd.Index(ids).has_duplicates:  # pragma: no cover - a 63-bit hash collision
                raise ValueError("Hashed node ids collide; relabel with by='range'")
        new_index = pd.Index(ids, name=nodes.index.name)
    else:
        raise ValueError(f"by must be 'range' or 'coordinates', got {by!r}")
    old_index = nodes.index
    nodes.index = new_index

    if not edges.empty:
        # Positions instead of a dict map: a missing endpoint would turn the column into floats, which cannot hold
        # the coordinate ids above 2**53.
        u_position = old_index.get_indexer(edges["u"])
        v_position = old_index.get_indexer(edges["v"])
        kept = (u_position >= 0) & (v_position >= 0)
        edges = edges[kept].copy()
        edges["u"] = new_index.to_numpy()[u_position[kept]]
        edges["v"] = new_index.to_numpy()[v_position[kept]]

        if not edges.empty:
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
        routes=graph_gdf.routes,
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
    missing_nodes = nodes_index.difference(graph_nodes.index)

    if len(missing_nodes) > 0:
        raise ValueError(f"nodes contain ids absent in graph.nodes_gdf.index: {missing_nodes[:10].tolist()}")

    sub_nodes, sub_edges = _induced_tables(graph_gdf, graph_nodes.index.isin(nodes_index))

    return UrbanGraph(
        nodes_gdf=sub_nodes,
        edges_gdf=sub_edges,
        is_multigraph=graph_gdf.is_multigraph,
        is_directed=graph_gdf.is_directed,
        edge_direction_column=graph_gdf.edge_direction_column,
        adjacency_weight=graph_gdf.adjacency_weight,
        crs=graph_gdf.crs,
        graph_type=graph_gdf.type,
        routes=graph_gdf.routes,
    )


def _induced_tables(graph, keep_nodes):
    nodes = graph.nodes_gdf.loc[keep_nodes]
    edges = graph.edges_gdf
    if edges.empty:
        return nodes, edges.copy()
    keep_edges = edges["u"].isin(nodes.index) & edges["v"].isin(nodes.index)
    edges = edges.loc[keep_edges]
    edges.index = pd.RangeIndex(len(edges))
    return nodes, edges


def clip_urban_graph(graph_gdf: UrbanGraph, polygon: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame) -> UrbanGraph:
    """
    Clip graph nodes by geometry and keep only edges with retained endpoints.

    ``polygon`` follows the rule of ``territory`` in the graph builders: a shapely geometry is in EPSG:4326
    (longitude, latitude); a GeoSeries or GeoDataFrame carries its own CRS. Either is reprojected to the
    graph CRS. Node ids are preserved; call :func:`relabel_urban_graph` if dense labels are needed.

    See also:
        https://iduclub.github.io/IduEdu/examples/graph_operations.html
    """
    # Imported here: graph_inputs imports this module's package at import time.
    from iduedu.graph.graph_inputs import area_in_crs  # pylint: disable=import-outside-toplevel

    nodes = graph_gdf.nodes_gdf.copy()
    edges = graph_gdf.edges_gdf.copy()

    if not isinstance(nodes, gpd.GeoDataFrame):
        raise TypeError(f"graph.nodes_gdf must be GeoDataFrame, got {type(nodes).__name__}")
    polygon = area_in_crs(polygon, nodes.crs or graph_gdf.crs, "polygon")

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
        routes=graph_gdf.routes,
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

    nodes, edges = _join_graph_tables(
        left.nodes_gdf, left.edges_gdf, right.nodes_gdf, right.edges_gdf, left.crs, node_conflict
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
        routes=join_routes(left.routes, right.routes),
    )


def _join_graph_tables(left_nodes, left_edges, right_nodes, right_edges, crs, node_conflict="left"):
    if left_nodes.empty:
        nodes = right_nodes.copy()
    elif right_nodes.empty:
        nodes = left_nodes.copy()
    else:
        node_parts = [left_nodes, right_nodes] if node_conflict == "left" else [right_nodes, left_nodes]
        nodes = gpd.GeoDataFrame(pd.concat(node_parts, axis=0, sort=False), geometry=left_nodes.geometry.name, crs=crs)
    if nodes.index.has_duplicates:
        nodes = nodes.loc[~nodes.index.duplicated(keep="first")]

    if left_edges.empty:
        edges = right_edges.copy()
    elif right_edges.empty:
        edges = left_edges.copy()
    else:
        edges = gpd.GeoDataFrame(
            pd.concat([left_edges, right_edges], axis=0, ignore_index=True, sort=False),
            geometry=left_edges.geometry.name,
            crs=crs,
        )
    return nodes, edges


def _substring_pieces(edges, distances, cut_offsets):
    """Retain Shapely's handling of XYZ lines and unusual coordinates."""
    source, starts, ends, pieces = [], [], [], []
    for row, edge in enumerate(edges):
        length = edge.length
        eps = 1e-6 * length
        boundaries = [(0.0, -1)]
        group_start = -np.inf
        for cut in range(cut_offsets[row], cut_offsets[row + 1]):
            distance = distances[cut]
            if abs(distance - group_start) <= eps:
                continue
            group_start = distance
            if abs(distance) <= eps:
                continue
            if abs(distance - length) <= eps:
                break
            boundaries.append((distance, cut))
        if boundaries[-1][0] < length:
            boundaries.append((length, -2))
        for (start, start_cut), (end, end_cut) in zip(boundaries, boundaries[1:]):
            piece = substring(edge, start, end)
            if piece.length > 0:
                source.append(row)
                starts.append(start_cut)
                ends.append(end_cut)
                pieces.append(piece)
    return (
        np.asarray(source, dtype=np.intp),
        np.asarray(starts, dtype=np.intp),
        np.asarray(ends, dtype=np.intp),
        np.asarray(pieces, dtype=object),
    )


def _projection_pieces(edges, distances, cut_offsets):
    """Split XY edges in one call, keeping the source edge and the two cut positions per piece."""
    xy = shapely.get_coordinates(edges).ravel()
    if (
        (shapely.get_coordinate_dimension(edges) != 2).any()
        or not np.isfinite(xy).all()
        or not np.isfinite(distances).all()
    ):
        return _substring_pieces(edges, distances, cut_offsets)

    coord_offsets = np.r_[0, shapely.get_num_coordinates(edges).cumsum(dtype=np.int64)]
    xy, offsets, sources, starts, ends = _rust.split_edges(xy, coord_offsets, distances, cut_offsets)
    if sources.size:
        pieces = shapely.linestrings(xy.reshape(-1, 2), indices=np.repeat(np.arange(len(sources)), np.diff(offsets)))
    else:
        pieces = np.empty(0, dtype=object)
    return sources, starts, ends, pieces


def _projection_edge_frame(u, v, geometries, speed, keys=None):
    # Python's round rounds the binary float directly, including values around a decimal tie.
    lengths = [round(length, 3) for length in shapely.length(geometries).tolist()]
    frame = pd.DataFrame(
        {
            "u": u.tolist(),
            "v": v.tolist(),
            "length_meter": lengths,
            "time_min": [round(length / speed, 2) for length in lengths],
            "geometry": geometries,
        }
    )
    if keys is not None:
        frame["k"] = keys
    return frame


def _split_projected_edges(projections, edges, key_cols, speed):
    if projections.empty:
        return pd.DataFrame(), pd.DataFrame(columns=key_cols)

    cuts = projections.sort_values(["edge_index", "project_dist"], kind="stable")
    cut_counts = cuts.groupby("edge_index", sort=False).size()
    edge_rows = (
        cut_counts.rename("cut_count")
        .to_frame()
        .merge(edges[key_cols + ["geometry"]], left_index=True, right_index=True, how="left", validate="many_to_one")
    )
    cut_offsets = np.r_[0, edge_rows["cut_count"].to_numpy(dtype=np.int64).cumsum()]
    cut_nodes = cuts["projection_node_id"].to_numpy(dtype=object)
    sources, starts, ends, pieces = _projection_pieces(
        edge_rows["geometry"].to_numpy(), cuts["project_dist"].to_numpy(), cut_offsets
    )
    u = np.where(starts < 0, edge_rows["u"].to_numpy(dtype=object)[sources], cut_nodes[np.maximum(starts, 0)])
    v = np.where(ends < 0, edge_rows["v"].to_numpy(dtype=object)[sources], cut_nodes[np.maximum(ends, 0)])
    keep = u != v
    sources = sources[keep]
    keys = edge_rows["k"].to_numpy()[sources].tolist() if "k" in key_cols else None
    split = _projection_edge_frame(u[keep], v[keep], pieces[keep], speed, keys)
    split["edge_index"] = edge_rows.index.to_numpy()[sources]
    deleted = pd.DataFrame({key: edge_rows[key].tolist() for key in key_cols}).drop_duplicates()
    return split, deleted


def _projection_connectors(objects, targets, object_nodes, target_nodes, speed, direction_column, topology):
    """Build connectors in object order, with a reverse edge after each forward edge when needed."""
    if objects.size == 0:
        return pd.DataFrame()
    paired = np.column_stack((objects, targets))
    if topology["is_directed"] and direction_column is None:
        both = np.empty((2 * len(paired), 2), dtype=object)
        both[::2], both[1::2] = paired, paired[:, ::-1]
        paired = both
        u = np.column_stack((object_nodes, target_nodes)).ravel()
        v = np.column_stack((target_nodes, object_nodes)).ravel()
    else:
        u, v = object_nodes, target_nodes

    flat = paired.ravel()
    if (shapely.get_coordinate_dimension(flat) != 2).any() or shapely.is_empty(flat).any():
        lines = np.asarray([LineString(pair) for pair in paired], dtype=object)
    else:
        lines = shapely.linestrings(shapely.get_coordinates(flat).reshape(-1, 2, 2))
    keep = ~(shapely.length(lines) <= 0)
    if not keep.any():
        return pd.DataFrame()
    connectors = _projection_edge_frame(u[keep], v[keep], lines[keep], speed, 0 if topology["is_multigraph"] else None)
    if direction_column is not None:
        connectors.insert(0, direction_column, False)
    return connectors


def _projection_attachments(
    endpoints,
    projections,
    attachments,
    speed,
    add_link_edge,
    next_id,
    direction_column,
    topology,
):
    """Allocate object nodes and assemble them in the original endpoint and projection-group order."""
    groups = attachments["projection_group"].to_numpy(dtype=np.intp)
    projection_points = projections["project_point"].to_numpy()
    objects = np.concatenate((endpoints["object_geometry"], attachments["object_geometry"]))
    targets = np.concatenate((endpoints["project_point"], projection_points[groups]))
    linked = (
        ~(shapely.distance(objects, targets) < MIN_CONNECTOR_LENGTH_M)
        if add_link_edge
        else np.zeros(len(objects), dtype=bool)
    )
    n_endpoint_links = int(linked[: len(endpoints)].sum())
    n_object_links = int(linked.sum())
    n_projections = len(projections)
    first_projection_id = next_id + n_endpoint_links
    projection_ids = range(first_projection_id, first_projection_id + n_projections) if n_projections else range(0)
    projections["projection_node_id"] = pd.Index(projection_ids).astype(int)
    projection_nodes = projections["projection_node_id"].to_numpy(dtype=object)
    target_nodes = np.concatenate((endpoints["projection_node_id"].to_numpy(dtype=object), projection_nodes[groups]))
    object_nodes = np.asarray(
        [next_id + offset for offset in range(n_endpoint_links)]
        + [first_projection_id + n_projections + offset for offset in range(n_object_links - n_endpoint_links)],
        dtype=object,
    )
    node_ids = np.empty(n_projections + n_object_links, dtype=object)
    points = np.empty(len(node_ids), dtype=object)
    # Each group contributes its projection node, followed by the group's dedicated object nodes.
    linked_groups = groups[linked[len(endpoints) :]]
    projection_positions = np.arange(n_projections)
    projection_positions += n_endpoint_links + np.searchsorted(linked_groups, projection_positions, side="left")
    object_positions = np.ones(len(node_ids), dtype=bool)
    object_positions[projection_positions] = False
    node_ids[projection_positions] = projection_nodes
    node_ids[object_positions] = object_nodes
    points[projection_positions] = projection_points
    points[object_positions] = objects[linked]

    nodes = gpd.GeoDataFrame(geometry=points, index=pd.Index(node_ids.tolist(), name="node_id"), crs=endpoints.crs)
    connectors = _projection_connectors(
        objects[linked], targets[linked], object_nodes, target_nodes[linked], speed, direction_column, topology
    )
    target_nodes[linked] = object_nodes
    mapping = pd.Series(dtype=int)
    if len(objects):
        mapping = pd.Series(
            target_nodes.tolist(),
            index=pd.Index(
                endpoints["object_index"].tolist() + attachments["object_index"].tolist(),
                name="object_index",
                tupleize_cols=False,
            ),
            name="node_id",
        )
    return nodes, connectors, mapping.loc[~mapping.index.duplicated()]


def _restore_split_attributes(edges, connectors, split, direction_column):
    if split.empty:
        return connectors
    if connectors.empty:
        new_edges = split
    else:
        new_edges = pd.concat([connectors, split], ignore_index=True)
        for key in ("u", "v", "k"):
            if key in split and connectors[key].dtype != split[key].dtype:
                new_edges[key] = connectors[key].tolist() + split[key].tolist()
        if direction_column is not None:
            # The original record table has False for connectors and a missing value for splits.
            direction = np.empty(len(new_edges), dtype=object)
            direction[: len(connectors)] = False
            direction[len(connectors) :] = np.nan
            new_edges[direction_column] = direction

    replaced = {"u", "v", "k", "geometry", "edge_geometry", "length_meter", "time_min", "edge_index"}
    columns = [column for column in edges.columns if column not in replaced]
    attrs = edges.loc[split["edge_index"].unique(), columns].reset_index(names="edge_index")
    new_edges = new_edges.merge(attrs, on="edge_index", how="left", suffixes=("", "_source"))
    for col in columns:
        source_col = f"{col}_source"
        if source_col in new_edges.columns:
            new_edges[col] = new_edges[col].combine_first(new_edges[source_col])
            new_edges = new_edges.drop(columns=source_col)
    return new_edges.drop(columns="edge_index")


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
            An object that already lies on the graph is mapped to the node it
            coincides with in both modes, because a connector edge shorter than
            ``MIN_CONNECTOR_LENGTH_M`` cannot connect anything.

    Returns:
        Pair ``(changes, object2node_map)``. ``changes.nodes_gdf`` contains new
        nodes, ``changes.edges_gdf`` contains new edges, and
        ``changes.edges_to_delete`` contains replaced edge keys. ``object2node_map``
        is a ``Series`` indexed by the original object index with graph node ids as
        values. Node ids in the mapping always reference connected nodes; several
        objects may share one node when they project to the same point.

    Raises:
        TypeError: If ``objects_gdf`` is not a GeoDataFrame.
        ValueError: If objects are empty or speed/distance parameters are invalid.

    See also:
        https://iduclub.github.io/IduEdu/examples/objects_and_nearest_nodes.html
    """

    return _project_objects(
        graph_gdf.nodes_gdf,
        graph_gdf.edges_gdf,
        objects_gdf,
        speed_m_per_min,
        crs=graph_gdf.crs,
        is_multigraph=graph_gdf.is_multigraph,
        is_directed=graph_gdf.is_directed,
        direction_column=graph_gdf.edge_direction_column,
        max_dist=max_dist,
        add_link_edge=add_link_edge,
    )


def _project_objects(
    gdf_nodes,
    gdf_edges,
    objects_gdf,
    speed_m_per_min,
    *,
    crs,
    is_multigraph,
    is_directed,
    direction_column,
    max_dist,
    add_link_edge,
):
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

    original_crs = crs or gdf_edges.crs or gdf_nodes.crs
    local_crs = gdf_nodes.estimate_utm_crs()
    key_cols = ["u", "v", "k"] if is_multigraph else ["u", "v"]
    match_edges = gdf_edges if gdf_edges.empty else gdf_edges[key_cols + ["geometry"]]
    if match_edges.crs != local_crs:
        match_edges = match_edges.to_crs(local_crs)
    if gdf_nodes.crs != local_crs:
        gdf_nodes = gdf_nodes.to_crs(local_crs)

    rep_points = objects_gdf["geometry"]
    if rep_points.crs != local_crs:
        rep_points = rep_points.to_crs(local_crs)
    rep_gdf = gpd.GeoDataFrame(geometry=rep_points.representative_point())

    # The inner join preserves endpoint ids above 2**53 by avoiding NaN in the endpoint columns.
    projection_join = rep_gdf.reset_index(names="object_index").sjoin_nearest(
        match_edges.assign(edge_geometry=match_edges.geometry), how="inner", max_distance=max_dist
    )
    projection_join = projection_join.rename_geometry("object_geometry").rename(columns={"index_right": "edge_index"})
    projection_join = projection_join.dropna(subset=["edge_index"])
    topology = {"is_multigraph": is_multigraph, "is_directed": is_directed}
    if projection_join.empty:
        return UrbanGraphChanges(edges_to_delete=pd.DataFrame(columns=key_cols), **topology), pd.Series(dtype=int)

    edge_geometry = projection_join["edge_geometry"]
    distances = edge_geometry.project(projection_join.geometry)
    edge_lengths = edge_geometry.length
    endpoint_snap_tolerance = (1e-5 * edge_lengths).clip(lower=1e-6)
    start_mask = distances.abs() <= endpoint_snap_tolerance
    end_mask = (distances - edge_lengths).abs() <= endpoint_snap_tolerance
    projection_join["project_dist"] = distances
    projection_join["node2connect"] = np.where(
        end_mask,
        projection_join["v"].to_numpy(dtype=object),
        np.where(start_mask, projection_join["u"].to_numpy(dtype=object), None),
    )

    # Prefer endpoints to splits and resolve endpoint ties by smallest node id.
    endpoint_rows = projection_join.loc[projection_join["node2connect"].notna()]
    existing = endpoint_rows.sort_values("node2connect").groupby("object_index", sort=False)["node2connect"].first()
    endpoints = rep_gdf.loc[existing.index].rename_geometry("object_geometry").reset_index(names="object_index")
    endpoints["projection_node_id"] = existing.to_numpy()
    endpoints["project_point"] = gdf_nodes.loc[existing.to_numpy(), "geometry"].to_numpy()

    interior = projection_join.loc[~projection_join["object_index"].isin(endpoint_rows["object_index"])].copy()
    interior["project_point"] = interior["edge_geometry"].interpolate(interior["project_dist"])
    interior["groupby_point"] = interior["project_point"].set_precision(10)
    interior["projection_group"] = interior.groupby(["groupby_point", "edge_index"], sort=False).ngroup()
    projections = interior.drop_duplicates("projection_group").copy()
    next_id = (
        int(gdf_nodes.index.max())
        if pd.api.types.is_integer_dtype(gdf_nodes.index)
        else max(gdf_nodes.index, default=-1)
    ) + 1
    attachments = (
        interior[["projection_group", "object_index", "object_geometry"]]
        .drop_duplicates()
        .sort_values("projection_group", kind="stable")
    )
    nodes, connectors, mapping = _projection_attachments(
        endpoints, projections, attachments, speed_m_per_min, add_link_edge, next_id, direction_column, topology
    )
    split, deleted = _split_projected_edges(projections, match_edges, key_cols, speed_m_per_min)
    edges = _restore_split_attributes(gdf_edges, connectors, split, direction_column)
    return (
        UrbanGraphChanges(
            nodes_gdf=nodes.to_crs(original_crs) if not nodes.empty else None,
            edges_gdf=(
                gpd.GeoDataFrame(edges, geometry="geometry", crs=local_crs).to_crs(original_crs)
                if not edges.empty
                else None
            ),
            edges_to_delete=deleted,
            **topology,
        ),
        mapping,
    )


def _update_edges(edges: gpd.GeoDataFrame, updates: pd.DataFrame, key_cols: list[str]) -> gpd.GeoDataFrame:
    """Edges with the values of ``updates`` set by key; NaN in ``updates`` keeps the old value."""
    position = pd.MultiIndex.from_frame(edges[key_cols]).get_indexer(pd.MultiIndex.from_frame(updates[key_cols]))
    if (position < 0).any():
        missing = updates.loc[position < 0, key_cols]
        raise ValueError(
            f"Some edges_to_update are not present in graph.edges_gdf: {missing.head(10).to_dict('records')}"
        )
    edges = edges.copy()
    for column in updates.columns.difference(key_cols, sort=False):
        new = pd.Series(updates[column].to_numpy(), index=position)
        if column in edges.columns:
            old = edges[column].reset_index(drop=True)
        else:
            old = pd.Series(np.nan, index=pd.RangeIndex(len(edges)))
        edges[column] = new.combine_first(old).sort_index().to_numpy()
    return edges


def apply_urban_graph_changes(graph_gdf: UrbanGraph, changes: UrbanGraphChanges) -> UrbanGraph:
    """Apply prepared changes to an ``UrbanGraph``.

    The function removes edges listed in ``changes.edges_to_delete``, sets the values of
    ``changes.edges_to_update``, appends new nodes and edges, and returns a new graph.
    The new graph has no cached adjacency matrix; routing builds it on the first query.

    Args:
        graph_gdf: Source graph.
        changes: Changes prepared by :func:`project_objects2urban_graph` or created
            manually.

    Returns:
        New ``UrbanGraph`` with changes applied.

    Raises:
        ValueError: If graph and change topology flags differ, if
            ``edges_to_delete`` or ``edges_to_update`` reference edges that are absent
            from the graph, or if an edge is both deleted and updated.

    See also:
        https://iduclub.github.io/IduEdu/examples/objects_and_nearest_nodes.html
    """

    if graph_gdf.is_multigraph != changes.is_multigraph:
        raise ValueError("graph.is_multigraph and changes.is_multigraph mismatch")
    if graph_gdf.is_directed != changes.is_directed:
        raise ValueError("graph.is_directed and changes.is_directed mismatch")

    nodes, edges = _apply_graph_changes_to_tables(graph_gdf.nodes_gdf, graph_gdf.edges_gdf, changes)
    return UrbanGraph(
        nodes_gdf=nodes,
        edges_gdf=edges,
        is_multigraph=graph_gdf.is_multigraph,
        is_directed=graph_gdf.is_directed,
        edge_direction_column=graph_gdf.edge_direction_column,
        adjacency_weight=graph_gdf.adjacency_weight,
        crs=graph_gdf.crs,
        graph_type=graph_gdf.type,
        routes=graph_gdf.routes,
    )


def _apply_graph_changes_to_tables(source_nodes, source_edges, changes):
    nodes, edges = source_nodes, source_edges
    new_nodes = changes.nodes_gdf
    new_edges = changes.edges_gdf
    edges_to_delete = changes.edges_to_delete
    edges_to_update = changes.edges_to_update

    key_cols = ["u", "v", "k"] if changes.is_multigraph else ["u", "v"]

    if (
        edges_to_delete is not None
        and not edges_to_delete.empty
        and edges_to_update is not None
        and not edges_to_update.empty
    ):
        both = edges_to_update[key_cols].merge(edges_to_delete[key_cols].drop_duplicates(), on=key_cols)
        if not both.empty:
            raise ValueError(f"Edges both deleted and updated: {both.head(10).to_dict('records')}")

    if edges_to_delete is not None and not edges_to_delete.empty:

        delete_keys = edges_to_delete[key_cols].drop_duplicates()
        delete_index = pd.MultiIndex.from_frame(delete_keys)
        matched = delete_index.get_indexer(pd.MultiIndex.from_frame(edges[key_cols]))
        keep = matched < 0
        absent = np.bincount(matched[~keep], minlength=len(delete_keys)) == 0
        if absent.any():
            raise ValueError(
                "Some edges_to_delete are not present in graph.edges_gdf: "
                f"{delete_keys.loc[absent].head(10).to_dict('records')}"
            )
        edges = edges.loc[keep]
        edges.index = pd.Index(np.flatnonzero(keep))

    if edges_to_update is not None and not edges_to_update.empty:
        edges = _update_edges(edges, edges_to_update, key_cols)

    if new_nodes is not None and not new_nodes.empty:
        intersect = nodes.index.intersection(new_nodes.index)
        if len(intersect) > 0:
            raise ValueError(f"Node id collision: {intersect[:10].tolist()}")

        nodes = pd.concat([nodes, new_nodes], axis=0)

    if new_edges is not None and not new_edges.empty:
        edges = pd.concat([edges, new_edges], axis=0, ignore_index=True)

    return (
        nodes.copy() if nodes is source_nodes else nodes,
        edges.copy() if edges is source_edges else edges,
    )
