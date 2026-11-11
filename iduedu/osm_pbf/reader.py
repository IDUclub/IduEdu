"""Answer IduEdu's Overpass queries from a local OpenStreetMap ``.osm.pbf`` file.

:class:`PbfSource` returns the very JSON element shapes the Overpass downloaders return, so builders and
parsers work unchanged whichever source the data comes from. The file is read by IduEdu's Rust core, with no
further dependency; the first query records which blocks of the file hold which elements, so later queries
on the same source read only the blocks they need.

Spatial selection follows Overpass ``poly:`` semantics, applied to the polygon's exterior ring (the
Overpass downloaders send only the exterior ring too):

- a node is selected when it lies inside the polygon or on its boundary;
- a way is selected when one of its nodes is selected or one of its segments crosses the polygon;
- a relation is selected when at least one of its node or way members is selected.

The geometry of selected elements is always complete, also outside the polygon, as with ``out geom``.
Node order in ways and member order in relations are kept as in the file. Elements of one output
statement come in ``(type, id)`` order, the order of a sorted ``.pbf`` file; Overpass' ``qt`` (quadtile)
order is not reproduced.
"""

from __future__ import annotations

import datetime as dt
import os
from itertools import chain
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import shapely
from shapely import MultiPolygon, Polygon

from iduedu import _rust, config
from iduedu.constants.transport_specs import osm_route_values
from iduedu.osm_pbf.filters import parse_way_filter

logger = config.logger

#: Name of the marker element separating stop-area members from the rest of a route response.
#: Kept equal to ``iduedu.overpass.downloaders.STOP_AREA_MEMBERS_SECTION``.
STOP_AREA_MEMBERS_SECTION = "stop_area_members"

_TYPE_NAMES = {"n": "node", "w": "way", "r": "relation"}
_TYPE_CHARS = {"node": "n", "way": "w", "relation": "r"}
_TYPE_ORDER = {"n": 0, "w": 1, "r": 2}
_MEMBER_CHARS = np.array(["n", "w", "r"])


def open_pbf(path: str | os.PathLike) -> _rust.PbfFile:
    """Open an OSM ``.pbf`` file with the Rust reader.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        ValueError: When the file is not an OSM ``.pbf`` file (other formats, such as ``.osm`` XML or
            ``.opl``, can be converted with ``osmium cat in.osm -o out.osm.pbf``).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"OSM file not found: {path}")
    try:
        return _rust.PbfFile(path)
    except ValueError as exc:
        raise ValueError(
            f"{exc}. IduEdu reads OSM data from .osm.pbf files only; convert other formats first, "
            f"for example with: osmium cat {path.name} -o {path.stem}.osm.pbf"
        ) from None


def _utc(stamp: int | None) -> dt.datetime | None:
    return None if stamp is None else dt.datetime.fromtimestamp(stamp, dt.timezone.utc)


def snapshot_time(path: str | os.PathLike) -> dt.datetime | None:
    """Return the moment an OSM file reflects, from its header, as an aware UTC ``datetime``.

    Geofabrik and other replication-based extracts store it as ``osmosis_replication_timestamp``.
    Returns ``None`` when the header carries none.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        ValueError: When the file is not an OSM ``.pbf`` file.
    """
    return _utc(open_pbf(path).header()["osmosis_replication_timestamp"])


def _selection_polygon(polygon) -> Polygon | MultiPolygon:
    """The area Overpass searches: the exterior ring(s) of the polygon, holes ignored."""
    if isinstance(polygon, Polygon):
        poly = Polygon(polygon.exterior)
    elif isinstance(polygon, MultiPolygon):
        poly = MultiPolygon([Polygon(p.exterior) for p in polygon.geoms])
    else:
        raise TypeError(f"polygon must be a shapely Polygon or MultiPolygon, got {type(polygon).__name__}")
    shapely.prepare(poly)
    return poly


def coords_intersect(poly, coord_lists: list) -> np.ndarray:
    """Vectorised ``poly:`` test: for every coordinate list (a way or a node), does it touch ``poly``."""
    n = len(coord_lists)
    result = np.zeros(n, dtype=bool)
    if n == 0:
        return result
    lengths = np.fromiter((len(c) for c in coord_lists), dtype=np.int64, count=n)
    multi = np.flatnonzero(lengths >= 2)
    if multi.size:
        flat = np.asarray(list(chain.from_iterable(coord_lists[i] for i in multi)), dtype="f8").reshape(-1, 2)
        indices = np.repeat(np.arange(multi.size), lengths[multi])
        lines = shapely.linestrings(flat, indices=indices)
        result[multi] = shapely.intersects(poly, lines)
    single = np.flatnonzero(lengths == 1)
    if single.size:
        xy = np.asarray([coord_lists[i][0] for i in single], dtype="f8")
        result[single] = shapely.intersects_xy(poly, xy[:, 0], xy[:, 1])
    return result


def _bounds(coords) -> dict | None:
    if not coords:
        return None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return {"minlat": min(lats), "minlon": min(lons), "maxlat": max(lats), "maxlon": max(lons)}


def _center(coords) -> dict | None:
    """Centre of the bounding box, printed by Overpass with seven decimals."""
    box = _bounds(coords)
    if box is None:
        return None
    lat = (box["minlat"] + box["maxlat"]) / 2.0
    lon = (box["minlon"] + box["maxlon"]) / 2.0
    return {"lat": float(f"{lat:.7f}"), "lon": float(f"{lon:.7f}")}


def _geometry(coords) -> list[dict]:
    return [{"lat": lat, "lon": lon} for lon, lat in coords]


def _with_tags(element: dict, tags: dict) -> dict:
    if tags:
        element["tags"] = dict(tags)
    return element


def _sort_key(item):
    kind, eid = item
    return _TYPE_ORDER[kind], eid


class _Store:
    """Objects read from the file: ``nodes[id] = (lonlat | None, tags)``,
    ``ways[id] = (node_refs, coords, tags)``, ``rels[id] = (members, tags)``."""

    def __init__(self):
        self.nodes: dict[int, tuple] = {}
        self.ways: dict[int, tuple] = {}
        self.rels: dict[int, tuple] = {}

    def has(self, kind: str, eid: int) -> bool:
        return eid in {"n": self.nodes, "w": self.ways, "r": self.rels}[kind]

    def tags(self, kind: str, eid: int) -> dict:
        return {"n": self.nodes, "w": self.ways, "r": self.rels}[kind][eid][-1]

    def member_coords(self, members) -> list:
        coords = []
        for kind, ref, _ in members:
            if kind == "n" and ref in self.nodes and self.nodes[ref][0] is not None:
                coords.append(self.nodes[ref][0])
            elif kind == "w" and ref in self.ways:
                coords.extend(self.ways[ref][1])
        return coords

    # ----- Overpass element shapes -----
    def tags_element(self, kind: str, eid: int) -> dict:
        """``out tags``."""
        return _with_tags({"type": _TYPE_NAMES[kind], "id": eid}, self.tags(kind, eid))

    def center_element(self, kind: str, eid: int) -> dict:
        """``out center`` (body verbosity plus a bounding-box centre for ways and relations)."""
        element = {"type": _TYPE_NAMES[kind], "id": eid}
        if kind == "n":
            lonlat, tags = self.nodes[eid]
            if lonlat is not None:
                element["lat"], element["lon"] = lonlat[1], lonlat[0]
        elif kind == "w":
            refs, coords, tags = self.ways[eid]
            center = _center(coords)
            if center is not None:
                element["center"] = center
            element["nodes"] = list(refs)
        else:
            members, tags = self.rels[eid]
            center = _center(self.member_coords(members))
            if center is not None:
                element["center"] = center
            element["members"] = [{"type": _TYPE_NAMES[k], "ref": ref, "role": role} for k, ref, role in members]
        return _with_tags(element, tags)

    def body_relation(self, rid: int) -> dict:
        """``out body`` of a relation."""
        members, tags = self.rels[rid]
        element = {
            "type": "relation",
            "id": rid,
            "members": [{"type": _TYPE_NAMES[k], "ref": ref, "role": role} for k, ref, role in members],
        }
        return _with_tags(element, tags)

    def geom_way(self, wid: int) -> dict:
        """``out geom`` of a way."""
        refs, coords, tags = self.ways[wid]
        element = {"type": "way", "id": wid}
        box = _bounds(coords)
        if box is not None:
            element["bounds"] = box
        element["nodes"] = list(refs)
        element["geometry"] = _geometry(coords)
        return _with_tags(element, tags)

    def geom_relation(self, rid: int) -> dict:
        """``out geom`` of a relation: node members carry lat/lon, way members their full geometry."""
        members, tags = self.rels[rid]
        out_members = []
        for kind, ref, role in members:
            member = {"type": _TYPE_NAMES[kind], "ref": ref, "role": role}
            if kind == "n" and ref in self.nodes and self.nodes[ref][0] is not None:
                lon, lat = self.nodes[ref][0]
                member["lat"], member["lon"] = lat, lon
            elif kind == "w" and ref in self.ways:
                member["geometry"] = _geometry(self.ways[ref][1])
            out_members.append(member)
        element = {"type": "relation", "id": rid}
        box = _bounds(self.member_coords(members))
        if box is not None:
            element["bounds"] = box
        element["members"] = out_members
        return _with_tags(element, tags)


class PbfSource:
    """A local OpenStreetMap ``.osm.pbf`` file answering the queries IduEdu otherwise sends to Overpass.

    For a city, pass a compact per-city file cut with :func:`iduedu.osm_pbf.split_by_polygons`; reading a
    whole country extract works too, but the first query then scans the whole file. Later queries on the
    same source read only the blocks of the file they need.

    Parameters:
        path: The ``.osm.pbf`` file.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        ValueError: When the file is not an OSM ``.pbf`` file.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._file = open_pbf(self.path)

    def __repr__(self) -> str:
        return f"PbfSource({str(self.path)!r})"

    def snapshot_time(self) -> dt.datetime | None:
        """The replication timestamp from the file header, see :func:`snapshot_time`."""
        return _utc(self._file.header()["osmosis_replication_timestamp"])

    # ------------------------------------------------------------------ reading
    @staticmethod
    def _id_array(ids) -> np.ndarray:
        return np.fromiter((int(i) for i in ids), dtype=np.int64)

    def _read_relations(self, store: _Store, *, ids=None, tag_pairs=None, contains_nodes=None) -> list[int]:
        """Read relations into ``store``; returns the ids read (in file order)."""
        if ids is not None:
            ids = self._id_array(ids)
            if ids.size == 0:
                return []
        if contains_nodes is not None:
            contains_nodes = self._id_array(contains_nodes)
            if contains_nodes.size == 0:
                return []
        rel_ids, offsets, types, refs, roles, tags = self._file.relations(
            ids=ids, tags=list(tag_pairs) if tag_pairs else None, contains_nodes=contains_nodes
        )
        kinds, refs, offsets = _MEMBER_CHARS[types].tolist(), refs.tolist(), offsets.tolist()
        read = rel_ids.tolist()
        for i, rid in enumerate(read):
            a, b = offsets[i], offsets[i + 1]
            store.rels[rid] = (tuple(zip(kinds[a:b], refs[a:b], roles[a:b])), tags[i])
        return read

    def _read_nodes_and_ways(self, store: _Store, node_ids: Iterable[int], way_ids: Iterable[int]) -> None:
        """Read the given nodes (location and tags) and ways (node refs, coordinates, tags) into ``store``."""
        node_ids = {int(i) for i in node_ids} - store.nodes.keys()
        way_ids = {int(i) for i in way_ids} - store.ways.keys()
        if way_ids:
            wids, ref_offsets, refs, coord_offsets, lon, lat, tags = self._file.ways(ids=self._id_array(way_ids))
            refs, xy = refs.tolist(), list(zip(lon.tolist(), lat.tolist()))
            ref_offsets, coord_offsets = ref_offsets.tolist(), coord_offsets.tolist()
            for i, wid in enumerate(wids.tolist()):
                store.ways[wid] = (
                    tuple(refs[ref_offsets[i] : ref_offsets[i + 1]]),
                    xy[coord_offsets[i] : coord_offsets[i + 1]],
                    tags[i],
                )
        if node_ids:
            nids, lon, lat, tags = self._file.nodes(self._id_array(node_ids))
            for nid, x, y, node_tags in zip(nids.tolist(), lon.tolist(), lat.tolist(), tags):
                store.nodes[nid] = ((x, y), node_tags)

    def _read_members(self, store: _Store, relation_ids: Iterable[int]) -> None:
        node_ids, way_ids = set(), set()
        for rid in relation_ids:
            for kind, ref, _ in store.rels[rid][0]:
                if kind == "n":
                    node_ids.add(ref)
                elif kind == "w":
                    way_ids.add(ref)
        self._read_nodes_and_ways(store, node_ids, way_ids)

    @staticmethod
    def _selected_relations(store: _Store, poly, relation_ids: list[int]) -> list[int]:
        """Relations with at least one node or way member touching ``poly``."""
        node_refs = sorted({ref for rid in relation_ids for k, ref, _ in store.rels[rid][0] if k == "n"})
        way_refs = sorted({ref for rid in relation_ids for k, ref, _ in store.rels[rid][0] if k == "w"})
        node_refs = [n for n in node_refs if n in store.nodes and store.nodes[n][0] is not None]
        way_refs = [w for w in way_refs if w in store.ways and store.ways[w][1]]
        inside_nodes = {
            n for n, hit in zip(node_refs, coords_intersect(poly, [[store.nodes[n][0]] for n in node_refs])) if hit
        }
        inside_ways = {
            w for w, hit in zip(way_refs, coords_intersect(poly, [store.ways[w][1] for w in way_refs])) if hit
        }
        selected = []
        for rid in relation_ids:
            for kind, ref, _ in store.rels[rid][0]:
                if (kind == "n" and ref in inside_nodes) or (kind == "w" and ref in inside_ways):
                    selected.append(rid)
                    break
        return sorted(selected)

    # ------------------------------------------------------------------ queries
    def _read_ways(self, polygon: Polygon, way_filter: str):
        """Ways matching ``way_filter`` that touch ``polygon``.

        Returns ``(ids, refs, coords, tags)``: way ids in file order, per way the tuple of node refs and the
        ``(k, 2)`` lon/lat array of the nodes found in the file, and the tag dicts.
        """
        predicate = parse_way_filter(way_filter)
        poly = _selection_polygon(polygon)
        clauses = [(c.key, c.op, c.value, c.ignore_case) for c in predicate.clauses]
        bboxes = [poly.bounds]  # the reader keeps only the ways whose box meets the polygon's
        try:
            ids, ref_offsets, refs, coord_offsets, lon, lat, tags = self._file.ways(clauses, bboxes=bboxes)
            keep = np.ones(len(ids), dtype=bool)
        except ValueError:
            # A regular expression Rust's engine does not support: read by the other clauses, then filter here.
            plain = [c for c in clauses if c[1] not in {"~", "!~"}]
            ids, ref_offsets, refs, coord_offsets, lon, lat, tags = self._file.ways(plain, bboxes=bboxes)
            keep = np.fromiter((predicate(t) for t in tags), dtype=bool, count=len(tags))

        counts = np.diff(coord_offsets)
        candidates = np.flatnonzero(keep)
        hits = np.zeros(len(candidates), dtype=bool)
        lengths = counts[candidates]
        multi, single = np.flatnonzero(lengths >= 2), np.flatnonzero(lengths == 1)
        if multi.size:
            spans = [np.arange(coord_offsets[w], coord_offsets[w + 1]) for w in candidates[multi]]
            flat = np.concatenate(spans)
            lines = shapely.linestrings(lon[flat], lat[flat], indices=np.repeat(np.arange(multi.size), lengths[multi]))
            hits[multi] = shapely.intersects(poly, lines)
        if single.size:
            first = coord_offsets[candidates[single]]
            hits[single] = shapely.intersects_xy(poly, lon[first], lat[first])
        selected = candidates[hits]

        xy = np.column_stack([lon, lat])
        way_refs = refs.tolist()
        out_refs = [tuple(way_refs[ref_offsets[w] : ref_offsets[w + 1]]) for w in selected]
        out_coords = [xy[coord_offsets[w] : coord_offsets[w + 1]] for w in selected]
        return ids[selected].tolist(), out_refs, out_coords, [tags[w] for w in selected]

    def ways_by_filter(self, polygon: Polygon, way_filter: str) -> list[dict]:
        """Ways matching an Overpass tag filter inside a polygon.

        Equivalent to ``way<way_filter>(poly:"..."); out geom;``. Returns Overpass ``out geom`` way
        elements ``{"type": "way", "id", "bounds", "nodes", "geometry": [{"lat", "lon"}], "tags"}``.
        """
        store = _Store()
        out = []
        for wid, refs, coords, tags in zip(*self._read_ways(polygon, way_filter)):
            store.ways[wid] = (refs, [tuple(c) for c in coords.tolist()], tags)
            out.append(store.geom_way(wid))
        logger.debug(f"Read {len(out)} ways matching <{way_filter}> from {self.path.name}")
        return out

    def ways_table(self, polygon: Polygon, way_filter: str) -> pd.DataFrame:
        """The ways of :meth:`ways_by_filter` as a table, without the per-point dictionaries.

        Columns ``type`` (``"way"``), ``id``, ``nodes`` (list of node ids), ``geometry`` (``(k, 2)`` array of
        lon/lat) and ``tags``. The street builders read this; it holds the same data as the Overpass elements.
        """
        ids, refs, coords, tags = self._read_ways(polygon, way_filter)
        logger.debug(f"Read {len(ids)} ways matching <{way_filter}> from {self.path.name}")
        return pd.DataFrame(
            {
                "type": ["way"] * len(ids),
                "id": ids,
                "nodes": [list(r) for r in refs],
                "geometry": coords,
                "tags": tags,
            }
        )

    def routes_by_poly(self, polygon: Polygon, public_transport_types: list[str]) -> list[dict]:
        """Public-transport route elements in a polygon, shaped as ``get_routes_by_poly`` returns them.

        Reproduces, statement by statement and in the same order, the Overpass query of
        :func:`iduedu.overpass.downloaders.get_routes_by_poly`: tags of the member ways of ground routes
        (``out tags``), the ground route relations (``out geom``) and, for ``subway``, the relations
        sharing a node with a subway route (``out geom``), their stop-area groups (``out body``), the
        stations among their members (``out tags``), the ``stop_area_members`` section marker and all
        their members (``out center``).
        """
        types = sorted(set(public_transport_types))
        if not types:
            return []
        has_subway = "subway" in types
        simple_route_types = sorted({v for t in types if t != "subway" for v in osm_route_values(t)})
        wanted = simple_route_types + (["subway"] if has_subway else [])
        poly = _selection_polygon(polygon)

        store = _Store()
        candidates = self._read_relations(store, tag_pairs=[("route", v) for v in wanted])
        self._read_members(store, candidates)
        selected = self._selected_relations(store, poly, candidates)

        simple_set = set(simple_route_types)
        routes_basic = [r for r in selected if store.rels[r][1].get("route") in simple_set]
        subway_routes = [r for r in selected if store.rels[r][1].get("route") == "subway"]

        out: list[dict] = []
        if simple_route_types:
            member_ways = sorted({ref for r in routes_basic for k, ref, _ in store.rels[r][0] if k == "w"})
            out.extend(store.tags_element("w", w) for w in member_ways if w in store.ways)
            out.extend(store.geom_relation(r) for r in routes_basic)

        if has_subway:
            if config.overpass_date is not None:
                logger.debug("config.overpass_date is ignored in pbf mode; the file is the snapshot.")
            out.extend(self._subway_details(store, subway_routes))

        logger.info(f"Read routes with types {types} from {self.path.name}.")
        return out

    def _subway_details(self, store: _Store, subway_routes: list[int]) -> list[dict]:
        route_nodes = sorted(
            {ref for r in subway_routes for k, ref, _ in store.rels[r][0] if k == "n" and ref in store.nodes}
        )
        stop_areas = sorted(self._read_relations(store, contains_nodes=route_nodes))
        stop_area_set = set(stop_areas)

        group_store = _Store()
        group_candidates = self._read_relations(group_store, tag_pairs=[("public_transport", "stop_area_group")])
        groups = []
        for gid in group_candidates:
            members, tags = group_store.rels[gid]
            if tags.get("type") == "public_transport" and any(
                k == "r" and ref in stop_area_set for k, ref, _ in members
            ):
                groups.append(gid)
        groups.sort()

        member_keys = sorted({(k, ref) for r in stop_areas for k, ref, _ in store.rels[r][0]}, key=_sort_key)
        member_rels = [ref for k, ref in member_keys if k == "r" and ref not in store.rels]
        self._read_relations(store, ids=member_rels)
        nested = [ref for k, ref in member_keys if k == "r" and ref in store.rels]
        self._read_members(store, stop_areas + nested)
        existing_members = [(k, ref) for k, ref in member_keys if store.has(k, ref)]

        out = [store.geom_relation(r) for r in stop_areas]
        for gid in groups:
            store.rels.setdefault(gid, group_store.rels[gid])
            out.append(store.body_relation(gid))
        out.extend(
            store.tags_element(k, ref)
            for k, ref in existing_members
            if store.tags(k, ref).get("public_transport") == "station"
        )
        out.append({"type": "section", "id": 1, "tags": {"name": STOP_AREA_MEMBERS_SECTION}})
        out.extend(store.center_element(k, ref) for k, ref in existing_members)
        return out

    def member_elements(self, members: Iterable[Mapping]) -> list[dict]:
        """``out center`` elements of the given members (dicts with ``type`` and ``ref``), sorted by type and id."""
        keys = sorted({(_TYPE_CHARS[m["type"]], int(m["ref"])) for m in members}, key=_sort_key)
        if not keys:
            return []
        store = _Store()
        self._read_relations(store, ids=[ref for k, ref in keys if k == "r"])
        relation_ids = list(store.rels)
        node_ids = {ref for k, ref in keys if k == "n"}
        way_ids = {ref for k, ref in keys if k == "w"}
        for rid in relation_ids:
            for kind, ref, _ in store.rels[rid][0]:
                if kind == "n":
                    node_ids.add(ref)
                elif kind == "w":
                    way_ids.add(ref)
        self._read_nodes_and_ways(store, node_ids, way_ids)
        return [store.center_element(k, ref) for k, ref in keys if store.has(k, ref)]

    def member_tags(self, members: Iterable[Mapping]) -> dict:
        """Tags and coordinates of members, keyed by ``(type, id)``, as ``fetch_member_tags`` returns them."""
        # Imported here: the downloaders import this module lazily, never the other way round at import time.
        from iduedu.overpass.downloaders import member_payloads  # pylint: disable=import-outside-toplevel

        return member_payloads(self.member_elements(members))

    def boundary(self, osm_id: int) -> list[dict]:
        """The relation ``osm_id`` as ``relation(<osm_id>); out geom;`` returns it (empty list when absent)."""
        store = _Store()
        found = self._read_relations(store, ids=[int(osm_id)])
        if not found:
            return []
        self._read_members(store, found)
        return [store.geom_relation(int(osm_id))]
