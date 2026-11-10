import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
from shapely.geometry.base import BaseGeometry

from iduedu.graph.adjacency import build_adjacency_matrix
from iduedu.graph.routing_result import ShortestPathResult
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
        "_csr_cache",
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
        self._csr_cache = None

        self.validate()

    def __getstate__(self):
        # The Rust view of the adjacency matrix is rebuilt on demand and is not pickled.
        return {name: getattr(self, name) for name in self.__slots__ if name != "_csr_cache"}

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)
        self._csr_cache = None

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

    def single_source_dijkstra(
        self,
        source_node: Any,
        *,
        target_nodes: Iterable[Any] | None = None,
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        reverse: bool = False,
        output: Literal["distances", "paths", "both"] = "both",
        path_storage: Literal["eager", "lazy"] = "eager",
        dtype: np.dtype = np.float32,
    ) -> pd.Series | ShortestPathResult:
        """Search once for distances and/or routes. See :func:`iduedu.single_source_dijkstra`."""
        from iduedu.graph.routing import single_source_dijkstra

        return single_source_dijkstra(
            self,
            source_node,
            target_nodes=target_nodes,
            weight=weight,
            cutoff=cutoff,
            reverse=reverse,
            output=output,
            path_storage=path_storage,
            dtype=dtype,
        )

    def multi_source_dijkstra(
        self,
        *,
        source_nodes: Iterable[Any] | None = None,
        gdf_sources: pd.DataFrame | None = None,
        graph_node_column: str = "graph_node_id",
        target_nodes: Iterable[Any] | None = None,
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        reverse: bool = False,
        output: Literal["distances", "paths", "both"] = "both",
        path_storage: Literal["eager", "lazy"] = "eager",
        dtype: np.dtype = np.float32,
    ) -> pd.Series | ShortestPathResult:
        """Search from the nearest source. See :func:`iduedu.multi_source_dijkstra`."""
        from iduedu.graph.routing import multi_source_dijkstra

        return multi_source_dijkstra(
            self,
            source_nodes=source_nodes,
            gdf_sources=gdf_sources,
            graph_node_column=graph_node_column,
            target_nodes=target_nodes,
            weight=weight,
            cutoff=cutoff,
            reverse=reverse,
            output=output,
            path_storage=path_storage,
            dtype=dtype,
        )

    def dijkstra_parallel(
        self,
        *,
        source_nodes: Iterable[Any] | None = None,
        gdf_sources: pd.DataFrame | None = None,
        graph_node_column: str = "graph_node_id",
        target_nodes: Iterable[Any] | None = None,
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        reverse: bool = False,
        output: Literal["distances", "paths", "both"] = "both",
        path_storage: Literal["eager", "lazy"] = "eager",
        dtype: np.dtype = np.float32,
        max_workers: int | None = None,
        batch_size: int | None = None,
    ) -> pd.DataFrame | ShortestPathResult:
        """Run independent source searches. See :func:`iduedu.dijkstra_parallel`."""
        from iduedu.graph.routing import dijkstra_parallel

        return dijkstra_parallel(
            self,
            source_nodes=source_nodes,
            gdf_sources=gdf_sources,
            graph_node_column=graph_node_column,
            target_nodes=target_nodes,
            weight=weight,
            cutoff=cutoff,
            reverse=reverse,
            output=output,
            path_storage=path_storage,
            dtype=dtype,
            max_workers=max_workers,
            batch_size=batch_size,
        )

    def single_source_dijkstra_path(
        self,
        source_node: Any,
        target_node: Any,
        *,
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        reverse: bool = False,
    ) -> list[Any]:
        """Return one shortest path as an ordered list of graph node ids."""

        from iduedu.graph.shortest_paths import single_source_dijkstra_path

        return single_source_dijkstra_path(
            self,
            source_node,
            target_node,
            weight=weight,
            cutoff=cutoff,
            reverse=reverse,
        )

    def shortest_path(
        self,
        source_node: Any,
        target_node: Any,
        *,
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        method: Literal["bidirectional", "astar", "alt", "dijkstra"] = "bidirectional",
    ) -> tuple[float, list[Any]]:
        """Return the cost and node route of one shortest path. See :func:`iduedu.shortest_path`."""

        from iduedu.graph.shortest_paths import shortest_path

        return shortest_path(self, source_node, target_node, weight=weight, cutoff=cutoff, method=method)

    def k_shortest_paths(
        self,
        source_node: Any,
        target_node: Any,
        k: int,
        *,
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Return up to k shortest loopless paths. See :func:`iduedu.k_shortest_paths`."""

        from iduedu.graph.shortest_paths import k_shortest_paths

        return k_shortest_paths(
            self, source_node, target_node, k, weight=weight, cutoff=cutoff, max_workers=max_workers
        )

    def alternative_paths(
        self,
        source_node: Any,
        target_node: Any,
        k: int = 3,
        *,
        weight: Literal["length_meter", "time_min"] = "time_min",
        penalty: float = 1.5,
        max_overlap: float = 0.7,
        max_cost_ratio: float = 1.4,
        cutoff: float | None = None,
    ) -> pd.DataFrame:
        """Return up to k distinct routes by the penalty method. See :func:`iduedu.alternative_paths`."""

        from iduedu.graph.shortest_paths import alternative_paths

        return alternative_paths(
            self,
            source_node,
            target_node,
            k,
            weight=weight,
            penalty=penalty,
            max_overlap=max_overlap,
            max_cost_ratio=max_cost_ratio,
            cutoff=cutoff,
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
        cutoff: float | None = None,
        output: Literal["distances", "paths", "both"] = "distances",
        path_storage: Literal["eager", "lazy"] = "eager",
        mode: Literal["all_to_all", "pairwise"] = "all_to_all",
        batch_size: int | None = None,
        method: Literal["auto", "bidirectional", "astar", "dijkstra"] = "auto",
    ) -> pd.DataFrame | ShortestPathResult:
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
            cutoff=cutoff,
            output=output,
            path_storage=path_storage,
            mode=mode,
            batch_size=batch_size,
            method=method,
        )

    def iter_od_matrix(
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
        cutoff: float | None = None,
        output: Literal["distances", "paths", "both"] = "distances",
        path_storage: Literal["eager", "lazy"] = "eager",
        mode: Literal["all_to_all", "pairwise"] = "all_to_all",
        batch_size: int = 256,
        method: Literal["auto", "bidirectional", "astar", "dijkstra"] = "auto",
    ) -> Iterator[pd.DataFrame | ShortestPathResult]:
        """Yield OD origin-row blocks. See :func:`iduedu.iter_od_matrix`."""
        from iduedu.graph.routing import iter_od_matrix

        return iter_od_matrix(
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
            cutoff=cutoff,
            output=output,
            path_storage=path_storage,
            mode=mode,
            batch_size=batch_size,
            method=method,
        )

    def path_to_edges(
        self,
        path: Iterable[Any],
        *,
        weight: Literal["length_meter", "time_min"] = "time_min",
    ) -> gpd.GeoDataFrame:
        """Convert an ordered node path to ordered graph edges."""

        from iduedu.graph.shortest_paths import path_to_edges

        return path_to_edges(self, path, weight=weight)

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
        :func:`iduedu.nx_graph2urban_graph`.

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

        The method delegates to :func:`iduedu.urban_graph2nx_graph` and
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
        Functional equivalent: :func:`iduedu.simplify_multiedges`.

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

        Functional equivalent: :func:`iduedu.relabel_urban_graph`.

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
        equivalent: :func:`iduedu.clip_urban_graph`.

        Args:
            polygon: Shapely geometry in EPSG:4326 (as ``territory`` of the graph builders), or a
                GeoSeries / GeoDataFrame in any CRS.
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

        Functional equivalent: :func:`iduedu.to_directed`.

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

        Functional equivalent: :func:`iduedu.to_undirected`.

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

        Functional equivalent: :func:`iduedu.nearest_nodes`.

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

    def nearest_edges(self, objects_gdf: gpd.GeoDataFrame, *, max_dist: float | None = None) -> pd.DataFrame:
        """Return the nearest graph edge for each object, with the distance and the position along it.

        Functional equivalent: :func:`iduedu.nearest_edges`.
        """

        from iduedu.graph.graph_inputs import nearest_edges

        return nearest_edges(self, objects_gdf, max_dist=max_dist)

    def unique_segments(self, *, precision: float = 0.01) -> gpd.GeoDataFrame:
        """Return the straight segments of the edge geometries, each physical segment once.

        Functional equivalent: :func:`iduedu.unique_segments`.
        """

        from iduedu.graph.segments import unique_segments

        return unique_segments(self, precision=precision)

    def consolidate_intersections(
        self, tolerance: float = 10.0, *, max_span: float | None = None, return_mapping: bool = False
    ):
        """Merge groups of nearby nodes, such as the nodes of one junction, into one node.

        Functional equivalent: :func:`iduedu.consolidate_intersections`.
        """

        from iduedu.graph.transformers import consolidate_intersections

        return consolidate_intersections(self, tolerance, max_span=max_span, return_mapping=return_mapping)

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
        be persisted separately, use :func:`iduedu.project_objects2urban_graph`.

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

    def graph_stats(
        self,
        *,
        area: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
        zones: gpd.GeoSeries | gpd.GeoDataFrame | None = None,
        edge_types: Iterable[str] | None = None,
        boundary_buffer: float | None = 1.0,
        precision: float = 0.01,
    ) -> pd.Series | pd.DataFrame:
        """Measures of size, density, shape and connectivity of the street network.

        Functional equivalent: :func:`iduedu.graph_stats`.
        """

        from iduedu.graph.stats import graph_stats

        return graph_stats(
            self, area=area, zones=zones, edge_types=edge_types, boundary_buffer=boundary_buffer, precision=precision
        )

    def edge_stats(
        self,
        by: str = "type",
        *,
        area: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
        precision: float = 0.01,
    ) -> pd.DataFrame:
        """Edge counts, lengths and speeds for every value of an edge column.

        Functional equivalent: :func:`iduedu.edge_stats`.
        """

        from iduedu.graph.stats import edge_stats

        return edge_stats(self, by, area=area, precision=precision)

    def transport_stats(
        self,
        *,
        modes: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Stops, routes and stop spacing of every public-transport mode.

        Functional equivalent: :func:`iduedu.transport_stats`.
        """

        from iduedu.graph.stats import transport_stats

        return transport_stats(self, modes=modes)

    def local_stats(
        self,
        radius: float | Iterable[float],
        *,
        weight: str = "length_meter",
        nodes: Iterable[Any] | None = None,
        edge_types: Iterable[str] | None = None,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Network measures around every node, within a travel cost along the network.

        Functional equivalent: :func:`iduedu.local_stats`.
        """

        from iduedu.graph.stats import local_stats

        return local_stats(self, radius, weight=weight, nodes=nodes, edge_types=edge_types, max_workers=max_workers)

    def streets_per_node(
        self,
        *,
        edge_types: Iterable[str] | None = None,
    ) -> pd.Series:
        """Number of streets that meet at every node.

        Functional equivalent: :func:`iduedu.streets_per_node`.
        """

        from iduedu.graph.stats import streets_per_node

        return streets_per_node(self, edge_types=edge_types)

    def route_directness(
        self,
        *,
        origins_nodes: Iterable[Any] | None = None,
        destination_nodes: Iterable[Any] | None = None,
        gdf_origins: pd.DataFrame | None = None,
        gdf_destinations: pd.DataFrame | None = None,
        graph_node_column: str = "graph_node_id",
        sample: int = 200,
        seed: int | None = 0,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Ratio of network to straight distance between origins and destinations.

        Functional equivalent: :func:`iduedu.route_directness`.
        """

        from iduedu.graph.stats import route_directness

        return route_directness(
            self,
            origins_nodes=origins_nodes,
            destination_nodes=destination_nodes,
            gdf_origins=gdf_origins,
            gdf_destinations=gdf_destinations,
            graph_node_column=graph_node_column,
            sample=sample,
            seed=seed,
            max_workers=max_workers,
        )

    def nearest_objects(
        self,
        *,
        gdf_origins: pd.DataFrame | None = None,
        gdf_objects: pd.DataFrame | None = None,
        origins_nodes: Iterable[Any] | None = None,
        object_nodes: Iterable[Any] | None = None,
        k: int = 1,
        cutoff: float | None = None,
        weight: Literal["length_meter", "time_min"] = "time_min",
        direction: Literal["from", "to"] = "from",
        graph_node_column: str = "graph_node_id",
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Find the ``k`` nearest objects of every origin along the network.

        Functional equivalent: :func:`iduedu.nearest_objects`.
        """

        from iduedu.graph.aggregations import nearest_objects

        return nearest_objects(
            self,
            gdf_origins=gdf_origins,
            gdf_objects=gdf_objects,
            origins_nodes=origins_nodes,
            object_nodes=object_nodes,
            k=k,
            cutoff=cutoff,
            weight=weight,
            direction=direction,
            graph_node_column=graph_node_column,
            max_workers=max_workers,
        )

    def aggregate_objects(
        self,
        cutoff: float | Iterable[float],
        *,
        gdf_origins: pd.DataFrame | None = None,
        gdf_objects: pd.DataFrame | None = None,
        origins_nodes: Iterable[Any] | None = None,
        object_nodes: Iterable[Any] | None = None,
        columns: Iterable[str] | None = None,
        weight: Literal["length_meter", "time_min"] = "time_min",
        decay: Literal["linear", "exp", "inverse"] | None = None,
        beta: float | Iterable[float] | None = None,
        direction: Literal["from", "to"] = "from",
        graph_node_column: str = "graph_node_id",
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Count objects and sum their values within a travel cost of every origin.

        Functional equivalent: :func:`iduedu.aggregate_objects`.
        """

        from iduedu.graph.aggregations import aggregate_objects

        return aggregate_objects(
            self,
            cutoff,
            gdf_origins=gdf_origins,
            gdf_objects=gdf_objects,
            origins_nodes=origins_nodes,
            object_nodes=object_nodes,
            columns=columns,
            weight=weight,
            decay=decay,
            beta=beta,
            direction=direction,
            graph_node_column=graph_node_column,
            max_workers=max_workers,
        )

    def diversity(
        self,
        cutoff: float | Iterable[float],
        category: str,
        *,
        gdf_objects: pd.DataFrame,
        gdf_origins: pd.DataFrame | None = None,
        origins_nodes: Iterable[Any] | None = None,
        values: str | None = None,
        hill_order: float = 2.0,
        weight: Literal["length_meter", "time_min"] = "time_min",
        decay: Literal["linear", "exp", "inverse"] | None = None,
        beta: float | Iterable[float] | None = None,
        direction: Literal["from", "to"] = "from",
        graph_node_column: str = "graph_node_id",
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Diversity of object categories within a travel cost of every origin.

        Functional equivalent: :func:`iduedu.diversity`.
        """

        from iduedu.graph.aggregations import diversity

        return diversity(
            self,
            cutoff,
            category,
            gdf_objects=gdf_objects,
            gdf_origins=gdf_origins,
            origins_nodes=origins_nodes,
            values=values,
            hill_order=hill_order,
            weight=weight,
            decay=decay,
            beta=beta,
            direction=direction,
            graph_node_column=graph_node_column,
            max_workers=max_workers,
        )

    def centrality(
        self,
        measures: Iterable[str] = ("reach", "harmonic", "gravity"),
        *,
        cutoffs: float | Iterable[float] | None = None,
        weight: Literal["length_meter", "time_min"] = "length_meter",
        beta: float | Iterable[float] | None = None,
        node_weights: str | pd.Series | np.ndarray | None = None,
        direction: Literal["from", "to"] = "from",
        area: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
        sample: int | None = None,
        seed: int | None = 0,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Centralities of the nodes within travel-cost thresholds.

        Functional equivalent: :func:`iduedu.centrality`.
        """

        from iduedu.graph.centrality import centrality

        return centrality(
            self,
            measures,
            cutoffs=cutoffs,
            weight=weight,
            beta=beta,
            node_weights=node_weights,
            direction=direction,
            area=area,
            sample=sample,
            seed=seed,
            max_workers=max_workers,
        )

    def edge_betweenness(
        self,
        *,
        cutoffs: float | Iterable[float] | None = None,
        weight: Literal["length_meter", "time_min"] = "length_meter",
        node_weights: str | pd.Series | np.ndarray | None = None,
        sample: int | None = None,
        seed: int | None = 0,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Betweenness of the edges within travel-cost thresholds.

        Functional equivalent: :func:`iduedu.edge_betweenness`.
        """

        from iduedu.graph.centrality import edge_betweenness

        return edge_betweenness(
            self,
            cutoffs=cutoffs,
            weight=weight,
            node_weights=node_weights,
            sample=sample,
            seed=seed,
            max_workers=max_workers,
        )

    def add_elevation(
        self,
        elevation: str | os.PathLike | gpd.GeoDataFrame | Callable,
        *,
        column: str = "elevation",
        method: Literal["linear", "nearest", "idw"] = "linear",
        sample_distance: float = 20.0,
        structure_columns: Iterable[str] = ("bridge", "tunnel"),
        band: int = 1,
    ) -> "UrbanGraph":
        """Return a copy of the graph with node heights and edge grades.

        Functional equivalent: :func:`iduedu.add_elevation`.
        """

        from iduedu.graph.elevation import add_elevation

        return add_elevation(
            self,
            elevation,
            column=column,
            method=method,
            sample_distance=sample_distance,
            structure_columns=structure_columns,
            band=band,
        )

    def apply_grade_speed(
        self,
        model: Literal["tobler", "cycling"] | Callable[[np.ndarray], np.ndarray] = "tobler",
        *,
        edge_types: Iterable[str] | None = None,
    ) -> "UrbanGraph":
        """Return a copy of the graph with travel times that depend on the grade.

        Functional equivalent: :func:`iduedu.apply_grade_speed`.
        """

        from iduedu.graph.elevation import apply_grade_speed

        return apply_grade_speed(self, model, edge_types=edge_types)
