from pathlib import Path
from typing import Any, Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse

from iduedu.graph.adjacency import build_adjacency_matrix
from iduedu.graph.validation import gdf_crs, sync_graph_crs, validate_graph


class UrbanGraph:
    """Tabular representation of an urban transport graph.

    ``UrbanGraph`` stores nodes and edges as pandas-compatible tables and
    builds SciPy CSR adjacency matrices for shortest-path and OD-matrix
    calculations. Spatial graphs use ``GeoDataFrame`` tables: nodes are points,
    edges are lines, and both tables share the graph CRS.

    Args:
        nodes_gdf: Node table. Its index is the node id and must be unique.
            Spatial graphs should use point geometries.
        edges_gdf: Edge table. Required columns are ``u``, ``v``,
            ``geometry``, ``length_meter`` and ``time_min``. Multigraphs also
            require ``k``. Edge endpoint columns reference ``nodes_gdf.index``.
        is_multigraph: Whether multiple edges may exist between the same node
            pair. If true, ``(u, v, k)`` uniquely identifies an edge.
        is_directed: Whether edge direction is respected by adjacency-based
            algorithms.
        edge_direction_column: Optional boolean edge column. ``True`` means
            movement is allowed only from ``u`` to ``v``; ``False`` means both
            directions are allowed.
        adjacency_weight: Default edge column used when building weighted
            adjacency matrices.
        crs: Optional graph CRS. If omitted, it is inferred from GeoDataFrames
            when possible.
        graph_type: Optional semantic graph type such as ``"drive"``,
            ``"walk"`` or ``"intermodal"``.

    Raises:
        TypeError: If node or edge tables use unsupported types.
        ValueError: If graph table contracts are violated.

    See also:
        https://iduclub.github.io/IduEdu/examples/urban_graph_basics.html
    """

    __slots__ = (
        "nodes_gdf",
        "edges_gdf",
        "crs",
        "type",
        "is_multigraph",
        "is_directed",
        "edge_direction_column",
        "adjacency_matrix",
        "adjacency_nodelist",
        "node_to_adjacency_pos",
        "adjacency_weight",
    )

    def __init__(
        self,
        nodes_gdf: gpd.GeoDataFrame | pd.DataFrame,
        edges_gdf: gpd.GeoDataFrame | pd.DataFrame,
        is_multigraph: bool,
        is_directed: bool,
        *,
        edge_direction_column: str | None = None,
        adjacency_weight: str = "time_min",
        crs: Any | None = None,
        graph_type: str | None = None,
    ):
        """Initialize graph tables, topology flags and adjacency cache state."""

        self.nodes_gdf = nodes_gdf
        self.edges_gdf = edges_gdf
        self.crs = crs
        self.type = graph_type
        self.is_multigraph = is_multigraph
        self.edge_direction_column = edge_direction_column
        self.is_directed = is_directed or edge_direction_column is not None
        self.adjacency_matrix = None
        self.adjacency_nodelist = []
        self.node_to_adjacency_pos = {}
        self.adjacency_weight = adjacency_weight

        self.validate()

    def __repr__(self) -> str:
        crs = self.crs
        if crs is not None:
            epsg = crs.to_epsg()
            crs_str = f"EPSG:{epsg}" if epsg is not None else crs.to_string()
        else:
            crs_str = None
        return (
            f"UrbanGraph(nodes={len(self.nodes_gdf)}, edges={len(self.edges_gdf)}, "
            f"is_multigraph={self.is_multigraph}, is_directed={self.is_directed}, "
            f"edge_direction_column={self.edge_direction_column!r}, crs={crs_str!r}, type={self.type!r})"
        )

    @classmethod
    def empty(
        cls,
        *,
        crs: Any | None = None,
        is_multigraph: bool = True,
        is_directed: bool = False,
        edge_direction_column: str | None = None,
        adjacency_weight: str = "time_min",
        graph_type: str | None = None,
    ) -> "UrbanGraph":
        """Create an empty graph with the requested topology metadata."""
        nodes_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs), crs=crs)
        edge_columns = ["u", "v", "geometry", "length_meter", "time_min"]
        if is_multigraph:
            edge_columns.insert(2, "k")
        if edge_direction_column is not None:
            edge_columns.append(edge_direction_column)
        edges_gdf = gpd.GeoDataFrame(columns=edge_columns, geometry="geometry", crs=crs)
        if edge_direction_column is not None:
            edges_gdf[edge_direction_column] = pd.Series(dtype=bool)
        return cls(
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            is_multigraph=is_multigraph,
            is_directed=is_directed,
            edge_direction_column=edge_direction_column,
            adjacency_weight=adjacency_weight,
            crs=crs,
            graph_type=graph_type,
        )

    @staticmethod
    def _gdf_crs(frame) -> Any | None:
        return gdf_crs(frame)

    def _sync_crs(self) -> None:
        sync_graph_crs(self)

    def validate(self) -> None:
        """Validate node, edge, topology and CRS contracts of the graph.

        Raises:
            TypeError: If graph tables use unsupported types.
            ValueError: If graph table contracts are violated.
        """

        validate_graph(self)

    def copy(self) -> "UrbanGraph":
        """Return an independent copy of the graph and cached adjacency state."""

        graph = UrbanGraph(
            nodes_gdf=self.nodes_gdf.copy(),
            edges_gdf=self.edges_gdf.copy(),
            is_multigraph=self.is_multigraph,
            is_directed=self.is_directed,
            edge_direction_column=self.edge_direction_column,
            adjacency_weight=self.adjacency_weight,
            crs=self.crs,
            graph_type=self.type,
        )

        if self.adjacency_matrix is not None:
            graph.adjacency_matrix = self.adjacency_matrix.copy()
            graph.adjacency_nodelist = list(self.adjacency_nodelist)
            graph.node_to_adjacency_pos = dict(self.node_to_adjacency_pos)

        return graph

    def write(self, path: str | Path, *, include_adjacency: bool = False) -> Path:
        """Write the graph to an ``.urbangraph`` archive.

        Args:
            path: Destination path with the ``.urbangraph`` suffix.
            include_adjacency: Whether to persist the cached adjacency matrix.

        Returns:
            Path to the written archive.

        See also:
            https://iduclub.github.io/IduEdu/examples/urban_graph_basics.html
        """

        from iduedu.graph.io import write_urban_graph

        return write_urban_graph(self, path, include_adjacency=include_adjacency)

    @classmethod
    def read(cls, path: str | Path, *, validate: bool = True) -> "UrbanGraph":
        """Read an ``UrbanGraph`` from an ``.urbangraph`` archive.

        Args:
            path: Source path with the ``.urbangraph`` suffix.
            validate: Whether to validate the graph after reading.

        Returns:
            Restored graph instance.

        See also:
            https://iduclub.github.io/IduEdu/examples/urban_graph_basics.html
        """

        from iduedu.graph.io import read_urban_graph

        return read_urban_graph(path, validate=validate)

    def _replace_state_from(self, other: "UrbanGraph") -> None:
        if not isinstance(other, UrbanGraph):
            raise TypeError(f"other must be UrbanGraph, got {type(other).__name__}")
        self.nodes_gdf = other.nodes_gdf
        self.edges_gdf = other.edges_gdf
        self.is_multigraph = other.is_multigraph
        self.is_directed = other.is_directed
        self.edge_direction_column = other.edge_direction_column
        self.crs = other.crs
        self.type = other.type
        self.validate()
        self._sync_crs()
        self.adjacency_matrix = None
        self.adjacency_nodelist = []
        self.node_to_adjacency_pos = {}

    def _build_adjacency_matrix(
        self, *, nodelist: Iterable[Any], weight: str, multigraph_rule: Literal["min", "max"] = "min"
    ) -> sparse.csr_matrix:
        return build_adjacency_matrix(self, nodelist=nodelist, weight=weight, multigraph_rule=multigraph_rule)

    def update_adjacency_matrix(
        self,
        *,
        nodelist: Iterable[Any] | None = None,
        weight: str | None = None,
        multigraph_rule: Literal["min", "max"] = "min",
    ) -> sparse.csr_matrix:
        """Rebuild and store the graph adjacency matrix.

        Args:
            nodelist: Node ids to include in matrix order. If omitted, all
                graph nodes are used.
            weight: Edge weight column. If omitted, ``adjacency_weight`` is
                used.
            multigraph_rule: Aggregation rule for parallel edges.

        Returns:
            Built SciPy CSR adjacency matrix.

        Raises:
            KeyError: If ``weight`` is not present in ``edges_gdf``.
            ValueError: If edge weights are invalid.
        """

        if nodelist is None:
            nodelist = self.nodes_gdf.index.to_list()
        else:
            nodelist = list(nodelist)

        if weight is None:
            weight = self.adjacency_weight

        self.adjacency_matrix = self._build_adjacency_matrix(
            nodelist=nodelist, weight=weight, multigraph_rule=multigraph_rule
        )
        self.adjacency_nodelist = nodelist
        self.node_to_adjacency_pos = {node: i for i, node in enumerate(nodelist)}
        self.adjacency_weight = weight

        return self.adjacency_matrix

    def to_csr(
        self,
        *,
        nodelist: Iterable[Any] | None = None,
        weight: str | None = None,
        multigraph_rule: Literal["min", "max"] = "min",
    ) -> sparse.csr_matrix:
        """Build a CSR adjacency matrix without changing graph state.

        Args:
            nodelist: Node ids to include in matrix order. If omitted, all
                graph nodes are used.
            weight: Edge weight column. If omitted, ``adjacency_weight`` is
                used.
            multigraph_rule: Aggregation rule for parallel edges.

        Returns:
            Built SciPy CSR adjacency matrix.
        """

        if nodelist is None:
            nodelist = self.nodes_gdf.index.to_list()
        else:
            nodelist = list(nodelist)

        if weight is None:
            weight = self.adjacency_weight

        return self._build_adjacency_matrix(nodelist=nodelist, weight=weight, multigraph_rule=multigraph_rule)

    def connected_components(self) -> list[set[Any]]:
        """Return connected components for an undirected graph.

        See also:
            https://iduclub.github.io/IduEdu/examples/connectivity.html
        """

        from iduedu.graph.components import connected_components

        return connected_components(self)

    def weakly_connected_components(self) -> list[set[Any]]:
        """Return weakly connected components, ignoring edge direction.

        See also:
            https://iduclub.github.io/IduEdu/examples/connectivity.html
        """

        from iduedu.graph.components import weakly_connected_components

        return weakly_connected_components(self)

    def strongly_connected_components(self) -> list[set[Any]]:
        """Return strongly connected components.

        See also:
            https://iduclub.github.io/IduEdu/examples/connectivity.html
        """

        from iduedu.graph.components import strongly_connected_components

        return strongly_connected_components(self)

    def largest_component(self, *, mode: Literal["auto", "connected", "weak", "strong"] = "auto") -> set[Any]:
        """Return the largest component according to the selected mode.

        See also:
            https://iduclub.github.io/IduEdu/examples/connectivity.html
        """

        from iduedu.graph.components import largest_component

        return largest_component(self, mode=mode)

    def subgraph_by_nodes(self, nodes: Iterable[Any]) -> "UrbanGraph":
        """Return the node-induced subgraph for ``nodes``.

        See also:
            https://iduclub.github.io/IduEdu/examples/graph_operations.html
        """

        from iduedu.graph.editors import subgraph_by_nodes

        return subgraph_by_nodes(self, nodes)

    def keep_largest_connected_component(
        self,
        *,
        mode: Literal["auto", "connected", "weak", "strong"] = "auto",
        inplace: bool = False,
    ) -> "UrbanGraph":
        """Keep only the largest graph component.

        See also:
            https://iduclub.github.io/IduEdu/examples/connectivity.html
        """

        from iduedu.graph.transformers import keep_largest_connected_component

        filtered = keep_largest_connected_component(self, mode=mode)

        if not inplace:
            return filtered

        self._replace_state_from(filtered)
        return self

    def single_source_dijkstra_path_length(
        self,
        source_node: Any,
        *,
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        reverse: bool = False,
        dtype: np.dtype = np.float32,
    ) -> pd.Series:
        """Run single-source Dijkstra shortest path search on this graph.

        See also:
            https://iduclub.github.io/IduEdu/examples/shortest_paths.html
        """

        from iduedu.graph.shortest_paths import single_source_dijkstra_path_length

        return single_source_dijkstra_path_length(
            self,
            source_node,
            weight=weight,
            cutoff=cutoff,
            reverse=reverse,
            dtype=dtype,
        )

    def multi_source_dijkstra_path_length(
        self,
        *,
        source_nodes: Iterable[Any] | None = None,
        gdf_sources: pd.DataFrame | None = None,
        graph_node_column: str = "graph_node_id",
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        reverse: bool = False,
        dtype: np.dtype = np.float32,
    ) -> pd.Series:
        """Run multi-source Dijkstra shortest path search on this graph.

        See also:
            https://iduclub.github.io/IduEdu/examples/shortest_paths.html
        """

        from iduedu.graph.shortest_paths import multi_source_dijkstra_path_length

        return multi_source_dijkstra_path_length(
            self,
            source_nodes=source_nodes,
            gdf_sources=gdf_sources,
            graph_node_column=graph_node_column,
            weight=weight,
            cutoff=cutoff,
            reverse=reverse,
            dtype=dtype,
        )

    def multi_source_dijkstra_nearest_source(
        self,
        *,
        source_nodes: Iterable[Any] | None = None,
        gdf_sources: pd.DataFrame | None = None,
        graph_node_column: str = "graph_node_id",
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        reverse: bool = False,
        dtype: np.dtype = np.float32,
    ) -> pd.DataFrame:
        """Find the nearest source node and distance for each reachable graph node.

        See also:
            https://iduclub.github.io/IduEdu/examples/shortest_paths.html
        """

        from iduedu.graph.shortest_paths import multi_source_dijkstra_nearest_source

        return multi_source_dijkstra_nearest_source(
            self,
            source_nodes=source_nodes,
            gdf_sources=gdf_sources,
            graph_node_column=graph_node_column,
            weight=weight,
            cutoff=cutoff,
            reverse=reverse,
            dtype=dtype,
        )

    def dijkstra_path_length_parallel(
        self,
        *,
        source_nodes: Iterable[Any] | None = None,
        gdf_sources: pd.DataFrame | None = None,
        graph_node_column: str = "graph_node_id",
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        reverse: bool = False,
        dtype: np.dtype = np.float32,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Run independent Dijkstra searches for multiple source nodes.

        See also:
            https://iduclub.github.io/IduEdu/examples/shortest_paths.html
        """

        from iduedu.graph.shortest_paths import dijkstra_path_length_parallel

        return dijkstra_path_length_parallel(
            self,
            source_nodes=source_nodes,
            gdf_sources=gdf_sources,
            graph_node_column=graph_node_column,
            weight=weight,
            cutoff=cutoff,
            reverse=reverse,
            dtype=dtype,
            max_workers=max_workers,
        )

    def od_matrix(
        self,
        *,
        gdf_origins: pd.DataFrame | None = None,
        gdf_destinations: pd.DataFrame | None = None,
        origins_nodes: Iterable[Any] | None = None,
        destination_nodes: Iterable[Any] | None = None,
        graph_node_column: str = "graph_node_id",
        weight: Literal["length_meter", "time_min"] = "time_min",
        dtype: np.dtype = np.float32,
        threshold: float | None = None,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Calculate an OD matrix of shortest paths on this graph.

        See also:
            https://iduclub.github.io/IduEdu/examples/shortest_paths.html
        """

        from iduedu.graph.shortest_paths import od_matrix

        return od_matrix(
            self,
            gdf_origins=gdf_origins,
            gdf_destinations=gdf_destinations,
            origins_nodes=origins_nodes,
            destination_nodes=destination_nodes,
            graph_node_column=graph_node_column,
            weight=weight,
            dtype=dtype,
            threshold=threshold,
            max_workers=max_workers,
        )

    @classmethod
    def from_nx_graph(
        cls,
        nx_graph,
        restore_edge_geom: bool = False,
        *,
        check_oneway: bool = True,
        oneway_column: str = "oneway",
    ) -> "UrbanGraph":
        """Create an ``UrbanGraph`` from a NetworkX graph.

        This constructor is useful for graphs received from external libraries when
        they already contain node coordinates, CRS metadata and edge attributes such as
        ``length_meter`` and ``time_min``. The conversion itself is performed by
        :func:`iduedu.graph.adapters.nx_graph2urban_graph`.

        Args:
            nx_graph: NetworkX graph, directed graph, multigraph or multidigraph.
            restore_edge_geom: If ``True``, empty edge geometries are restored as
                straight segments between endpoint nodes.
            check_oneway: If ``True`` and ``oneway_column`` exists on edges, that column
                is used as the edge direction column.
            oneway_column: Boolean edge attribute that marks one-way movement.

        Returns:
            Converted ``UrbanGraph`` instance.
        """

        from .adapters import nx_graph2urban_graph

        return nx_graph2urban_graph(
            nx_graph,
            restore_edge_geom,
            check_oneway=check_oneway,
            oneway_column=oneway_column,
        )

    def to_nx_graph(self):
        """Convert this graph to a NetworkX graph.

        The method delegates to :func:`iduedu.graph.adapters.urban_graph2nx_graph` and
        preserves node and edge attributes where possible.

        Returns:
            NetworkX graph type matching this graph topology.
        """

        from .adapters import urban_graph2nx_graph

        return urban_graph2nx_graph(self)

    def simplify_multiedges(
        self, *, weight: str = "time_min", rule: Literal["min", "max"] = "min", inplace: bool = False
    ) -> "UrbanGraph":
        """Collapse a multigraph to a simple graph.

        For each node pair, one edge is selected by the ``weight`` column. ``rule="min"``
        keeps the smallest weight and ``rule="max"`` keeps the largest weight.
        Functional equivalent: :func:`iduedu.graph.transformers.simplify_multiedges`.

        Args:
            weight: Edge column used to choose the representative edge.
            rule: Selection rule, either ``"min"`` or ``"max"``.
            inplace: If ``True``, replace this object with the simplified graph.

        Returns:
            Simplified ``UrbanGraph``.

        See also:
            https://iduclub.github.io/IduEdu/examples/graph_operations.html
        """

        from iduedu.graph.transformers import simplify_multiedges

        simplified = simplify_multiedges(self, weight=weight, rule=rule)
        simplified.adjacency_weight = self.adjacency_weight

        if not inplace:
            return simplified

        self._replace_state_from(simplified)
        return self

    def relabel(self, *, inplace: bool = False) -> "UrbanGraph":
        """Relabel graph nodes to a dense ``RangeIndex``.

        Functional equivalent: :func:`iduedu.graph.editors.relabel_urban_graph`.

        Args:
            inplace: If ``True``, replace this object with the relabeled graph.

        Returns:
            ``UrbanGraph`` with updated node indexes and edge endpoints.

        See also:
            https://iduclub.github.io/IduEdu/examples/graph_operations.html
        """

        from .editors import relabel_urban_graph

        relabeled = relabel_urban_graph(self)

        if not inplace:
            return relabeled

        self._replace_state_from(relabeled)
        return self

    def clip(self, polygon, *, inplace: bool = False) -> "UrbanGraph":
        """Clip the graph by geometry and keep only nodes inside it.

        Edges are retained only when both endpoints remain in the graph. Node ids are
        preserved; call :meth:`relabel` if dense labels are needed. Functional
        equivalent: :func:`iduedu.graph.editors.clip_urban_graph`.

        Args:
            polygon: Shapely geometry in the graph CRS.
            inplace: If ``True``, replace this object with the clipped graph.

        Returns:
            Clipped ``UrbanGraph``.

        See also:
            https://iduclub.github.io/IduEdu/examples/graph_operations.html
        """

        from .editors import clip_urban_graph

        clipped = clip_urban_graph(self, polygon)

        if not inplace:
            return clipped

        self._replace_state_from(clipped)
        return self

    def join(
        self,
        other: "UrbanGraph",
        *,
        graph_type: str | None = None,
        node_conflict: str = "left",
        inplace: bool = False,
    ) -> "UrbanGraph":
        """Join this graph with another compatible ``UrbanGraph``.

        Shared node indexes are allowed and resolved with ``node_conflict``. Duplicate
        edge keys are treated as conflicts.

        Args:
            other: Graph to append.
            graph_type: Optional graph type for the result. If ``None``, keep this graph
                type.
            node_conflict: Which side wins when node indexes overlap: ``"left"`` or
                ``"right"``.
            inplace: If ``True``, replace this object with the joined graph.

        Returns:
            Joined ``UrbanGraph``.

        See also:
            https://iduclub.github.io/IduEdu/examples/graph_operations.html
        """

        from iduedu.graph.editors import join_urban_graphs

        joined = join_urban_graphs(self, other, graph_type=graph_type, node_conflict=node_conflict)

        if not inplace:
            return joined

        self._replace_state_from(joined)
        return self

    def to_directed(
        self,
        *,
        edge_direction_column: str = "oneway",
        default_direction_value: bool = False,
        inplace: bool = False,
    ) -> "UrbanGraph":
        """Return a directed version of the graph with an edge direction column.

        Functional equivalent: :func:`iduedu.graph.transformers.to_directed`.

        Args:
            edge_direction_column: Name of the boolean one-way edge column.
            default_direction_value: Value used for edges where the column is missing or
                null.
            inplace: If ``True``, replace this object with the directed graph.

        Returns:
            Directed ``UrbanGraph``.

        See also:
            https://iduclub.github.io/IduEdu/examples/graph_operations.html
        """

        from iduedu.graph.transformers import to_directed

        directed = to_directed(
            self,
            edge_direction_column=edge_direction_column,
            default_direction_value=default_direction_value,
        )

        if not inplace:
            return directed

        self._replace_state_from(directed)
        return self

    def to_undirected(self, *, inplace: bool = False) -> "UrbanGraph":
        """Return an undirected version of the graph.

        Functional equivalent: :func:`iduedu.graph.transformers.to_undirected`.

        Args:
            inplace: If ``True``, replace this object with the undirected graph.

        Returns:
            Undirected ``UrbanGraph``.

        See also:
            https://iduclub.github.io/IduEdu/examples/graph_operations.html
        """

        from .transformers import to_undirected

        undirected = to_undirected(self)

        if not inplace:
            return undirected

        self._replace_state_from(undirected)
        return self

    def nearest_nodes(
        self,
        objects_gdf: gpd.GeoDataFrame,
        *,
        graph_node_column: str = "graph_node_id",
    ) -> pd.Series:
        """Return nearest graph node ids for object geometries.

        Functional equivalent: :func:`iduedu.graph.graph_inputs.nearest_nodes`.

        Args:
            objects_gdf: GeoDataFrame with geometries to match to graph nodes.
            graph_node_column: Name assigned to the returned ``Series``.

        Returns:
            Series indexed like ``objects_gdf`` with nearest node ids as values.

        See also:
            https://iduclub.github.io/IduEdu/examples/objects_and_nearest_nodes.html
        """

        from iduedu.graph.graph_inputs import nearest_nodes

        return nearest_nodes(self, objects_gdf, graph_node_column=graph_node_column)

    def project_objects(
        self,
        objects_gdf: gpd.GeoDataFrame,
        speed_m_per_min: float,
        *,
        max_dist: float | None = None,
        add_link_edge: bool = True,
        inplace: bool = False,
    ) -> tuple["UrbanGraph", pd.Series]:
        """Project objects onto nearest graph edges and add them to the graph.

        The method creates graph nodes for objects, projects their representative points
        onto nearest edges, splits those edges when needed and adds connector edges. It
        is convenient for in-memory preparation of buildings, services or other objects
        before OD-matrix calculations. For backend workflows where graph changes should
        be persisted separately, use :func:`iduedu.graph.editors.project_objects2urban_graph`.

        Args:
            objects_gdf: Objects with a unique index and geometry. The index becomes the
                ``object2node_map`` index.
            speed_m_per_min: Movement speed on connector edges, in meters per minute.
                For 5 km/h use ``5 * 1000 / 60``.
            max_dist: Optional maximum distance to the nearest edge. If ``None``, no
                distance limit is applied.
            add_link_edge: If ``True``, create a dedicated object node and connector
                edge. If ``False``, map objects to projection nodes on the graph.
            inplace: If ``True``, apply changes to this graph.

        Returns:
            Pair ``(graph, object2node_map)``. ``object2node_map`` is indexed by the
            original object index and contains graph node ids.

        See also:
            https://iduclub.github.io/IduEdu/examples/objects_and_nearest_nodes.html
        """

        from iduedu.graph.editors import apply_urban_graph_changes, project_objects2urban_graph

        graph = self if inplace else self.copy()
        changes, object2node_map = project_objects2urban_graph(
            graph,
            objects_gdf,
            speed_m_per_min,
            max_dist=max_dist,
            add_link_edge=add_link_edge,
        )

        changed_graph = apply_urban_graph_changes(graph, changes)
        changed_graph.adjacency_weight = graph.adjacency_weight

        if not inplace:
            return changed_graph, object2node_map

        self._replace_state_from(changed_graph)
        return self, object2node_map
