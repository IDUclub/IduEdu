"""Edits of a graph given by geometry: lines to add, and anchor lines that select the edges to remove or update.

Edits are stored as lines rather than node and edge ids, so they survive a rebuild of the graph from newer data, and
:func:`resolve_graph_edits` binds them to the current graph as an :class:`UrbanGraphChanges`.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import substring

from iduedu.graph.editors import UrbanGraphChanges, coordinate_node_ids
from iduedu.graph.graph_inputs import projected_crs
from iduedu.graph.urban_graph import UrbanGraph

REPORT_COLUMNS = ["kind", "edit", "reason", "coverage"]
_INPUT_ONLY = {"speed_kmh", "direction"}  # read from the edits, never stored on the edges
_GRID = 1e-3  # split points closer than this (in CRS units) become one node


@dataclass(slots=True)
class GraphEdits:
    """Edits of a graph given by lines in any CRS, resolved by :func:`resolve_graph_edits`.

    Args:
        add: Lines of new edges. Columns other than ``geometry`` become edge attributes, except:
            ``time_min`` and ``length_meter`` (computed when absent), ``speed_kmh`` (gives ``time_min``
            when it is absent) and ``oneway`` (bool, default ``False``: the edge can be travelled both ways).
        remove: Anchor lines; the network along each of them is removed. An optional ``direction`` column
            (``"both"``, the default, or ``"along"``) limits a directed graph to the direction the anchor is
            drawn in.
        update: Anchor lines with the new values of the edges along them, in any columns except ``u``, ``v``
            and ``k``. A missing value keeps the old one. ``speed_kmh`` sets ``time_min`` (and ``speed_mpm``
            when the graph has it) from the edge length unless ``time_min`` is given. ``direction`` works as in
            ``remove``.

    Raises:
        TypeError: If a table is not a GeoDataFrame.
        ValueError: If a table has no CRS, a duplicated index, geometries other than non-empty LineStrings,
            reserved columns, or an unknown ``direction``.
    """

    add: gpd.GeoDataFrame | None = None
    remove: gpd.GeoDataFrame | None = None
    update: gpd.GeoDataFrame | None = None

    def __post_init__(self) -> None:
        for name, table in (("add", self.add), ("remove", self.remove), ("update", self.update)):
            if table is None:
                continue
            if not isinstance(table, gpd.GeoDataFrame):
                raise TypeError(f"{name} must be GeoDataFrame, got {type(table).__name__}")
            if table.crs is None:
                raise ValueError(f"{name} needs a CRS")
            if table.index.has_duplicates:
                raise ValueError(f"{name}.index must be unique")
            geometry = table.geometry
            if (geometry.isna() | geometry.is_empty | (geometry.geom_type != "LineString")).any():
                raise ValueError(f"All {name} geometries must be non-empty LineString")
            if (shapely.length(np.asarray(geometry.values)) <= 0).any():  # any CRS, so no GeoSeries.length
                raise ValueError(f"{name} contains lines of zero length")
            reserved = {"u", "v", "k"} & set(table.columns)
            if reserved:
                raise ValueError(f"{name} cannot set {sorted(reserved)}")
            if name != "add" and "direction" in table.columns:
                unknown = set(table["direction"].dropna()) - {"both", "along"}
                if unknown:
                    raise ValueError(f"{name}.direction must be 'both' or 'along', got {sorted(unknown)}")
        if self.update is not None and not set(self.update.columns) - {"geometry", "direction"}:
            raise ValueError("update has no value columns")


@dataclass(slots=True)
class _Target:
    """Part ``start .. end`` of the edge at ``position`` selected by an anchor."""

    order: int
    start: float
    end: float
    along: bool  # the edge geometry runs in the direction of the anchor
    direction: str
    values: dict[str, Any] | None  # None removes


@dataclass(frozen=True, slots=True)
class _NewNode:
    """Reference to a new node by its position in :class:`_NewNodes`, until the ids are assigned."""

    index: int


@dataclass
class _NewNodes:
    """New nodes keyed by position, so that one point gives one node."""

    points: list[Point] = field(default_factory=list)
    cells: dict[tuple[int, int], int] = field(default_factory=dict)

    def at(self, point: Point) -> _NewNode:
        cx, cy = int(round(point.x / _GRID)), int(round(point.y / _GRID))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                found = self.cells.get((cx + dx, cy + dy))
                if found is not None and self.points[found].distance(point) <= _GRID:
                    return _NewNode(found)
        self.points.append(point)
        self.cells[(cx, cy)] = len(self.points) - 1
        return _NewNode(len(self.points) - 1)

    def near(self, point: Point, tolerance: float) -> _NewNode | None:
        if not self.points:
            return None
        distances = shapely.distance(np.asarray(self.points, dtype=object), point)
        best = int(np.argmin(distances))
        return _NewNode(best) if distances[best] <= tolerance else None


def resolve_graph_edits(
    graph: UrbanGraph,
    edits: GraphEdits,
    *,
    tolerance: float = 5.0,
    speed_kmh: float | None = None,
    min_coverage: float = 0.5,
    node_ids: Literal["next", "coordinates"] = "next",
    edge_types: Iterable[str] | None = None,
) -> tuple[UrbanGraphChanges, pd.DataFrame]:
    """Bind geometric edits to the current graph.

    Anchors of ``remove`` and ``update`` select the edges that lie along them within ``tolerance``; an edge
    covered only in part is split at the ends of the anchor, and only the covered part is changed. Crossing
    streets are not selected: the edge has to run along the anchor. Each end of an ``add`` line is attached to
    a node within ``tolerance``, or splits the nearest edge within ``tolerance`` at the projection of the end,
    or becomes a new node. Split edges keep their attributes, with length and travel time shared out by length.

    Edits that cannot be bound are skipped and reported instead of raising, so a stored scenario still applies
    after the graph is rebuilt.

    Args:
        graph: Graph in a projected CRS; ``tolerance`` is in its units.
        edits: Edits to bind.
        tolerance: Search distance for nodes and edges, in metres for the graphs of the builders.
        speed_kmh: Speed of added lines that have neither ``time_min`` nor ``speed_kmh``.
        min_coverage: Least share of an anchor's length that the selected edges must cover; an anchor with less
            is reported and skipped.
        node_ids: Ids of new nodes: ``"next"`` continues after the largest id of the graph, ``"coordinates"``
            packs the coordinates as :func:`relabel_urban_graph` with ``by="coordinates"`` does.
        edge_types: Edit only edges whose ``type`` is one of these, and attach added lines only to their nodes,
            for example ``["walk"]`` in an intermodal graph, so that a closed street does not remove the transit
            edges drawn along it.

    Returns:
        Pair ``(changes, report)``. ``changes`` is applied with :func:`apply_urban_graph_changes`. ``report`` has a
        row per skipped edit: ``kind`` (``"add"``, ``"remove"`` or ``"update"``), ``edit`` (its index), ``reason``
        and ``coverage`` (the share of an anchor covered by the graph).

    Raises:
        ValueError: If the graph CRS is missing or geographic, the parameters are out of range, or a new node id
            is already in the graph.
        KeyError: If ``edge_types`` is given and the edges have no ``type`` column.
    """
    crs = projected_crs(graph, "resolve_graph_edits", required=True)
    if tolerance <= 0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}")
    if speed_kmh is not None and speed_kmh <= 0:
        raise ValueError(f"speed_kmh must be > 0 or None, got {speed_kmh}")
    if not 0 < min_coverage <= 1:
        raise ValueError(f"min_coverage must be in (0, 1], got {min_coverage}")
    if node_ids not in ("next", "coordinates"):
        raise ValueError(f"node_ids must be 'next' or 'coordinates', got {node_ids!r}")

    if edge_types is not None and "type" not in graph.edges_gdf.columns:
        raise KeyError("edge_types needs a 'type' column in edges_gdf")
    resolver = _Resolver(graph, crs, tolerance, speed_kmh, min_coverage, node_ids)
    if edge_types is not None:
        resolver.restrict(list(edge_types))
    return resolver.run(edits)


class _Resolver:  # pylint: disable=too-many-instance-attributes
    def __init__(self, graph: UrbanGraph, crs, tolerance: float, speed_kmh, min_coverage: float, node_ids: str):
        self.graph = graph
        self.crs = crs
        self.tolerance = tolerance
        self.speed_kmh = speed_kmh
        self.min_coverage = min_coverage
        self.node_ids = node_ids
        self.edges = _in_crs(graph.edges_gdf, crs)
        self.nodes = _in_crs(graph.nodes_gdf, crs)
        self.geometry = self.edges.geometry.to_numpy()
        self.key_cols = ["u", "v", "k"] if graph.is_multigraph else ["u", "v"]
        self.direction_column = graph.edge_direction_column
        self.new_nodes = _NewNodes()
        self.cuts: dict[int, list[tuple[float, Any]]] = {}
        self.targets: dict[int, list[_Target]] = {}
        self.added: list[dict[str, Any]] = []
        self.report: list[dict[str, Any]] = []
        self.selectable: np.ndarray | None = None  # edge positions that edits may touch
        self.attachable = self.nodes.geometry  # nodes that added lines may attach to

    def restrict(self, edge_types: list[str]) -> None:
        mask = self.edges["type"].isin(edge_types).to_numpy()
        self.selectable = mask
        ends = pd.Index(self.edges["u"][mask]).union(pd.Index(self.edges["v"][mask]))
        self.attachable = self.nodes.geometry[self.nodes.index.isin(ends)]

    def _query(self, geometry, **kwargs) -> np.ndarray:
        positions = self.edges.sindex.query(geometry, **kwargs)
        return positions if self.selectable is None else positions[self.selectable[positions]]

    # --- anchors -----------------------------------------------------------------------------------------------

    def run(self, edits: GraphEdits) -> tuple[UrbanGraphChanges, pd.DataFrame]:
        order = 0
        for kind in ("update", "remove"):
            table = getattr(edits, kind)
            if table is None:
                continue
            for label, row in table.to_crs(self.crs).iterrows():
                values = None
                if kind == "update":
                    values = {
                        key: value
                        for key, value in row.items()
                        if key not in ("geometry", "direction") and not _missing(value)
                    }
                    if not values:
                        continue
                direction = row.get("direction", "both")
                self._anchor(kind, label, row.geometry, order, "both" if _missing(direction) else direction, values)
                order += 1
        if edits.add is not None:
            for label, row in edits.add.to_crs(self.crs).iterrows():
                self._add(label, row)
        return self._changes(), pd.DataFrame(self.report, columns=REPORT_COLUMNS)

    def _anchor(self, kind: str, label, anchor: LineString, order: int, direction: str, values) -> None:
        tolerance = self.tolerance
        zone = anchor.buffer(tolerance)
        found: list[tuple[int, _Target]] = []
        covered: list[tuple[float, float]] = []
        for position in self._query(zone, predicate="intersects"):
            edge = self.geometry[position]
            for part in _parts(edge.intersection(zone)):
                start, end = sorted((edge.project(Point(part.coords[0])), edge.project(Point(part.coords[-1]))))
                a_start = anchor.project(edge.interpolate(start))
                a_end = anchor.project(edge.interpolate(end))
                if abs(a_end - a_start) < 0.5 * (end - start) or end - start <= _GRID:
                    continue  # a crossing street, not one along the anchor
                start = edge.project(anchor.interpolate(a_start))
                end = edge.project(anchor.interpolate(a_end))
                start, end = min(start, end), max(start, end)
                if start < tolerance:
                    start = 0.0
                if end > edge.length - tolerance:
                    end = edge.length
                if end - start <= _GRID:
                    continue
                if direction == "along" and a_end < a_start and self._one_way(position):
                    continue  # the edge can only be travelled against the anchor
                found.append((int(position), _Target(order, start, end, a_end > a_start, direction, values)))
                covered.append((min(a_start, a_end), max(a_start, a_end)))
        coverage = _union_length(covered) / anchor.length
        if coverage < self.min_coverage:
            reason = "no edges along the anchor" if not found else "anchor not covered by the graph"
            self.report.append({"kind": kind, "edit": label, "reason": reason, "coverage": round(coverage, 3)})
            return
        for position, target in found:
            edge = self.geometry[position]
            for distance in (target.start, target.end):
                if 0.0 < distance < edge.length:
                    self._cut(position, distance)
            self.targets.setdefault(position, []).append(target)

    def _one_way(self, position: int) -> bool:
        if not self.graph.is_directed:
            return False
        if self.direction_column is None:
            return True
        return _flag(self.edges.iloc[position][self.direction_column])

    def _cut(self, position: int, distance: float, node=None) -> Any:
        edge = self.geometry[position]
        node = node if node is not None else self.new_nodes.at(edge.interpolate(distance))
        self.cuts.setdefault(position, []).append((distance, node))
        return node

    # --- added lines -------------------------------------------------------------------------------------------

    def _endpoint(self, point: Point) -> tuple[Any, Point]:
        tolerance = self.tolerance
        candidates = []
        new = self.new_nodes.near(point, tolerance)
        if new is not None:
            new_point = self.new_nodes.points[new.index]
            candidates.append((new_point.distance(point), new, new_point))
        nodes = self.attachable
        near_nodes = nodes.sindex.query(point, predicate="dwithin", distance=tolerance)
        if len(near_nodes):
            distances = shapely.distance(nodes.to_numpy()[near_nodes], point)
            best = near_nodes[int(np.argmin(distances))]
            candidates.append((float(distances.min()), nodes.index[best], nodes.iloc[best]))
        if candidates:
            _, node, node_point = min(candidates, key=lambda item: item[0])
            return node, node_point
        near_edges = self._query(point, predicate="dwithin", distance=tolerance)
        if len(near_edges):
            distances = shapely.distance(self.geometry[near_edges], point)
            nearest = near_edges[int(np.argmin(distances))]
            foot = self.geometry[nearest].interpolate(self.geometry[nearest].project(point))
            node = self.new_nodes.at(foot)
            for position in near_edges[distances <= distances.min() + _GRID]:  # also the reverse twin
                distance = self.geometry[position].project(foot)
                if 0.0 < distance < self.geometry[position].length:
                    self._cut(int(position), distance, node)
            return node, self.new_nodes.points[node.index]
        node = self.new_nodes.at(point)
        return node, point

    def _add(self, label, row: pd.Series) -> None:
        line: LineString = row.geometry
        coords = list(line.coords)
        attrs = {key: value for key, value in row.items() if key != "geometry" and not _missing(value)}
        speed = attrs.pop("speed_kmh", None) or self.speed_kmh
        if "time_min" not in attrs and speed is None:
            self.report.append({"kind": "add", "edit": label, "reason": "no time_min or speed", "coverage": np.nan})
            return
        u, u_point = self._endpoint(Point(coords[0]))
        v, v_point = self._endpoint(Point(coords[-1]))
        if _same_node(u, v):
            self.report.append({"kind": "add", "edit": label, "reason": "both ends at one node", "coverage": np.nan})
            return
        if not self.graph.is_multigraph and self._connected(u, v):
            self.report.append({"kind": "add", "edit": label, "reason": "edge already exists", "coverage": np.nan})
            return
        geometry = LineString([u_point.coords[0], *coords[1:-1], v_point.coords[0]])
        attrs.setdefault("length_meter", round(geometry.length, 3))
        if "time_min" not in attrs:
            attrs["time_min"] = round(attrs["length_meter"] / (speed * 1000.0 / 60.0), 3)
        oneway = bool(attrs.pop("oneway", False))
        if self.direction_column is not None:
            attrs[self.direction_column] = oneway
        self.added.append({**attrs, "u": u, "v": v, "geometry": geometry})
        if self.graph.is_directed and self.direction_column is None and not oneway:
            self.added.append({**attrs, "u": v, "v": u, "geometry": geometry.reverse()})

    def _connected(self, u, v) -> bool:
        if isinstance(u, _NewNode) or isinstance(v, _NewNode):
            return False
        edges = self.edges
        forward = (edges["u"] == u) & (edges["v"] == v)
        if not self.graph.is_directed:
            forward |= (edges["u"] == v) & (edges["v"] == u)
        return bool(forward.any())

    # --- output ------------------------------------------------------------------------------------------------

    def _changes(self) -> UrbanGraphChanges:
        deleted, updated, rows = [], [], []
        for position in sorted(set(self.cuts) | set(self.targets)):
            edge_row = self.edges.iloc[position]
            pieces = self._pieces(position, edge_row)
            targets = self.targets.get(position, [])
            if position not in self.cuts:
                halves = self._halves(edge_row, 0.0, self.geometry[position].length, targets)
                if halves is None:
                    continue
                if len(halves) == 1 and halves[0][0] == "same" and halves[0][1] is not None:
                    values = self._values(edge_row, float(edge_row["length_meter"]), halves[0][1])
                    updated.append({**{c: edge_row[c] for c in self.key_cols}, **values})
                    continue
                deleted.append({c: edge_row[c] for c in self.key_cols})
                rows.extend(self._rows(edge_row, edge_row.geometry, edge_row["u"], edge_row["v"], 1.0, halves))
                continue
            deleted.append({c: edge_row[c] for c in self.key_cols})
            for start, end, u, v, geometry in pieces:
                share = (end - start) / self.geometry[position].length
                halves = self._halves(edge_row, start, end, targets)
                if halves is None:
                    halves = [("same", {})]
                rows.extend(self._rows(edge_row, geometry, u, v, share, halves))
        rows.extend(self.added)
        return self._assemble(deleted, updated, rows)

    def _pieces(self, position: int, edge_row: pd.Series) -> list[tuple[float, float, Any, Any, LineString]]:
        edge = self.geometry[position]
        cuts = sorted(self.cuts.get(position, []), key=lambda cut: cut[0])
        points = [(0.0, edge_row["u"])]
        for distance, node in cuts:
            if distance - points[-1][0] > _GRID and not _same_node(node, points[-1][1]):
                points.append((distance, node))
        if edge.length - points[-1][0] <= _GRID:
            points.pop()
        points.append((edge.length, edge_row["v"]))
        return [
            (start, end, u, v, substring(edge, start, end)) for (start, u), (end, v) in zip(points[:-1], points[1:])
        ]

    def _halves(self, edge_row: pd.Series, start: float, end: float, targets: list[_Target]):
        """State of the part ``start .. end`` of an edge: None when untouched, else a list of
        ``(orientation, values)`` with orientation ``"same"`` (one row as the edge, two-way rows stay two-way),
        ``"forward"`` or ``"backward"`` (one-way rows), and values None for a removed part."""
        covering = sorted(
            (t for t in targets if t.start <= start + _GRID and t.end >= end - _GRID), key=lambda t: t.order
        )
        if not covering:
            return None
        directed = self.graph.is_directed
        two_way = directed and self.direction_column is not None and not _flag(edge_row[self.direction_column])

        def state(forward: bool) -> dict[str, Any] | None:
            values: dict[str, Any] = {}
            for target in covering:
                if directed and target.direction == "along" and target.along != forward:
                    continue
                if target.values is None:
                    return None
                values = {**values, **target.values}
            return values

        if not two_way:
            values = state(True)
            if values == {}:
                return None
            return [("same", values)] if values is not None else []
        forward, backward = state(True), state(False)
        if forward == backward:
            if forward == {}:
                return None
            return [("same", forward)] if forward is not None else []
        return [(side, values) for side, values in (("forward", forward), ("backward", backward)) if values is not None]

    def _values(self, edge_row: pd.Series, length: float, values: dict[str, Any]) -> dict[str, Any]:
        values = dict(values)
        speed = values.pop("speed_kmh", None)
        if speed is not None and "time_min" not in values:
            values["time_min"] = round(length / (float(speed) * 1000.0 / 60.0), 3)
            if "speed_mpm" in edge_row.index:
                values["speed_mpm"] = round(float(speed) * 1000.0 / 60.0, 3)
        return values

    def _rows(self, edge_row: pd.Series, geometry, u, v, share: float, halves) -> list[dict[str, Any]]:
        base = edge_row.to_dict()
        base["length_meter"] = round(float(edge_row["length_meter"]) * share, 3)
        base["time_min"] = round(float(edge_row["time_min"]) * share, 3)
        rows = []
        for orientation, values in halves:
            row = {**base, **self._values(edge_row, base["length_meter"], values), "geometry": geometry}
            if orientation == "same":
                rows.append({**row, "u": u, "v": v})
            elif orientation == "forward":
                rows.append({**row, "u": u, "v": v, self.direction_column: True})
            else:
                rows.append({**row, "u": v, "v": u, "geometry": geometry.reverse(), self.direction_column: True})
        return rows

    def _assemble(self, deleted, updated, rows) -> UrbanGraphChanges:
        graph = self.graph
        ids = self._node_ids()

        def final(node):
            return ids[node.index] if isinstance(node, _NewNode) else node

        new_nodes = None
        used = sorted({node.index for row in rows for node in (row["u"], row["v"]) if isinstance(node, _NewNode)})
        if used:
            new_nodes = gpd.GeoDataFrame(
                geometry=[self.new_nodes.points[i] for i in used],
                index=pd.Index([ids[i] for i in used], name=graph.nodes_gdf.index.name),
                crs=self.crs,
            ).to_crs(graph.nodes_gdf.crs or self.crs)
        new_edges = None
        if rows:
            new_edges = gpd.GeoDataFrame(rows, geometry="geometry", crs=self.crs)
            new_edges["u"] = new_edges["u"].map(final)
            new_edges["v"] = new_edges["v"].map(final)
            if graph.is_multigraph:
                new_edges["k"] = self._keys(new_edges, deleted)
            new_edges = new_edges.drop(columns=[c for c in _INPUT_ONLY if c in new_edges.columns])
            new_edges = new_edges.to_crs(graph.edges_gdf.crs or self.crs)
        updates = None
        if updated:
            updates = pd.DataFrame(updated).drop(columns=list(_INPUT_ONLY), errors="ignore")
        return UrbanGraphChanges(
            nodes_gdf=new_nodes,
            edges_gdf=new_edges,
            edges_to_delete=pd.DataFrame(deleted, columns=self.key_cols).drop_duplicates(),
            edges_to_update=updates,
            is_multigraph=graph.is_multigraph,
            is_directed=graph.is_directed,
        )

    def _node_ids(self) -> list[Any]:
        count = len(self.new_nodes.points)
        if not count:
            return []
        existing = self.graph.nodes_gdf.index
        if self.node_ids == "coordinates":
            ids = coordinate_node_ids(gpd.GeoSeries(self.new_nodes.points, crs=self.crs)).tolist()
        else:
            if len(existing) and not pd.api.types.is_integer_dtype(existing):
                raise ValueError("node_ids='next' needs integer node ids; use node_ids='coordinates'")
            start = int(existing.max()) + 1 if len(existing) else 0
            ids = list(range(start, start + count))
        clash = existing.intersection(pd.Index(ids))
        if len(clash) or len(set(ids)) < count:
            raise ValueError(f"New node ids collide with the graph or each other: {clash[:10].tolist()}")
        return ids

    def _keys(self, new_edges: gpd.GeoDataFrame, deleted: list[dict[str, Any]]) -> list[int]:
        kept = self.edges[self.key_cols]
        if deleted:
            gone = pd.DataFrame(deleted, columns=self.key_cols)
            kept = kept.merge(gone.assign(_gone=True), on=self.key_cols, how="left")
            kept = kept[kept["_gone"].isna()]
        used = set(map(tuple, kept[self.key_cols].to_numpy().tolist()))
        keys = []
        initial = new_edges["k"] if "k" in new_edges.columns else pd.Series(0, index=new_edges.index)
        for u, v, k in zip(new_edges["u"], new_edges["v"], initial.fillna(0).astype(int)):
            while (u, v, k) in used:
                k += 1
            used.add((u, v, k))
            keys.append(k)
        return keys


def _same_node(a, b) -> bool:
    if isinstance(a, _NewNode) or isinstance(b, _NewNode):
        return a == b
    return bool(a == b)


def _in_crs(gdf: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(crs)
    return gdf if gdf.crs == crs else gdf.to_crs(crs)


def _flag(value) -> bool:
    """One-way flag of an edge; a missing value means two-way."""
    return False if _missing(value) else bool(value)


def _missing(value) -> bool:
    return value is None or (np.isscalar(value) and pd.isna(value))


def _parts(geometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return [part for item in geometry.geoms for part in _parts(item)]
    return []


def _union_length(intervals: list[tuple[float, float]]) -> float:
    total, current_start, current_end = 0.0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total
