"""Cut compact per-city OSM files out of a large regional extract.

:func:`split_by_polygons` reads a regional extract (for instance a Geofabrik country file) for the spatial
selection of every city at the same time, completes the references in a few passes that read only the
relation or way blocks of the file, and writes one ``.osm.pbf`` per city in a final pass. Each city file
holds what IduEdu's queries can reach from that city's polygon -- as ``osmium extract --strategy smart``
would, restricted to the objects IduEdu reads -- so :class:`~iduedu.osm_pbf.PbfSource` answers them on the
small file exactly as on the regional one:

- ways carrying one of ``way_keys`` (by default ``highway``, ``railway``, ``public_transport``) that touch
  the polygon under Overpass ``poly:`` semantics, with all their nodes;
- public-transport route relations (``route`` in ``route_values``) with a node or way member touching
  the polygon, with all their members complete, also outside the polygon;
- for subway routes, every relation sharing a node with them (their stop areas) with all members and
  their relation members complete, and the ``stop_area_group`` relations that contain those stop areas;
- relations listed in ``keep_relations`` (for example the city boundary, so that ``osm_id=`` keeps
  working) with their node and way members complete.

Tags are kept on every object, including referenced ones outside the polygon; element metadata (versions,
timestamps, users) is not copied. The header keeps the source's replication timestamp, so
:func:`~iduedu.osm_pbf.snapshot_time` works on the city files.
"""

import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import shapely
from shapely import MultiPolygon, Polygon, STRtree

from iduedu import config
from iduedu.constants.transport_specs import DEFAULT_REGISTRY_W_TRAIN, osm_route_values
from iduedu.osm_pbf.reader import _MEMBER_CHARS, _selection_polygon, open_pbf

logger = config.logger

DEFAULT_WAY_KEYS = ("highway", "railway", "public_transport")


def default_route_values() -> tuple[str, ...]:
    """OSM ``route`` values of every mode in ``DEFAULT_REGISTRY_W_TRAIN``, aliases included."""
    return tuple(sorted({v for t in DEFAULT_REGISTRY_W_TRAIN.list_types() for v in osm_route_values(t)}))


class _City:  # pylint: disable=too-many-instance-attributes
    """Selection state of one output file."""

    def __init__(self, name: str, polygon, path: Path):
        self.name = name
        self.polygon = polygon
        self.path = path
        self.node_parts: list[np.ndarray] = []
        self.way_parts: list[np.ndarray] = []
        self.relations: set[int] = set()
        self.selected_ways: set[int] = set()  # ways touching the polygon, not those added as members
        self.member_nodes: set[int] = set()
        self.member_ways: set[int] = set()
        self.subway_nodes: set[int] = set()
        self.stop_areas: set[int] = set()
        self.nested: set[int] = set()

    def complete(self, rid: int, members) -> None:
        """Take a relation and all its node and way members."""
        self.relations.add(rid)
        for kind, ref, _ in members:
            if kind == "n":
                self.member_nodes.add(ref)
            elif kind == "w":
                self.member_ways.add(ref)

    def ids(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sorted node, way and relation ids of the file."""

        def unique(parts, extra):
            arrays = [np.asarray(p, dtype=np.int64) for p in parts]
            arrays.append(np.fromiter(extra, dtype=np.int64, count=len(extra)))
            return np.unique(np.concatenate(arrays))

        return (
            unique(self.node_parts, self.member_nodes),
            unique(self.way_parts, self.member_ways),
            unique([], self.relations),
        )


def _relations(file, **query) -> list[tuple[int, tuple, dict]]:
    """``(id, members, tags)`` of the relations matching ``query``, members as ``(type, ref, role)``."""
    rel_ids, offsets, types, refs, roles, tags = file.relations(**query)
    members = list(zip(_MEMBER_CHARS[types].tolist(), refs.tolist(), roles))
    offsets = offsets.tolist()
    return [(rid, tuple(members[offsets[i] : offsets[i + 1]]), tags[i]) for i, rid in enumerate(rel_ids.tolist())]


def split_by_polygons(
    region_pbf: str | os.PathLike,
    polygons: Mapping[str, Polygon | MultiPolygon],
    out_dir: str | os.PathLike,
    *,
    way_keys: Iterable[str] | None = DEFAULT_WAY_KEYS,
    route_values: Iterable[str] | None = None,
    keep_relations: Mapping[str, Iterable[int]] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write one compact, referentially complete ``<name>.osm.pbf`` per polygon from a regional extract.

    Parameters:
        region_pbf: Regional ``.osm.pbf`` file, as distributed by Geofabrik.
        polygons: ``{name: polygon}`` in EPSG:4326. Use the polygons (or larger ones) you will later
            build graphs for; as with Overpass, only the exterior ring counts. ``name`` becomes the
            file name and must not contain path separators.
        out_dir: Output directory, created if missing.
        way_keys: Ways carrying at least one of these keys are selected spatially. ``None`` selects
            every way. Ways without them still enter a city file as members of selected relations.
        route_values: OSM ``route`` values of the relations to keep; defaults to every mode of
            ``DEFAULT_REGISTRY_W_TRAIN`` (bus, trolleybus, tram/light_rail, subway, monorail, train,
            taxi/share_taxi).
        keep_relations: ``{name: [relation ids]}`` relations to copy complete into that city's file,
            e.g. its boundary relation.
        overwrite: Replace existing output files.

    Returns:
        ``{name: path}`` of the written files.

    Raises:
        FileNotFoundError: When ``region_pbf`` does not exist.
        ValueError: When ``region_pbf`` is not an OSM ``.pbf`` file, or for an invalid name.
        FileExistsError: When an output exists and ``overwrite`` is False.
    """
    region_pbf = Path(region_pbf)
    file = open_pbf(region_pbf)
    if not polygons:
        return {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cities: list[_City] = []
    for name, polygon in polygons.items():
        name = str(name)
        if not name or any(sep in name for sep in ("/", "\\", os.sep)) or name in {".", ".."}:
            raise ValueError(f"Invalid city name for a file name: {name!r}")
        path = out_dir / f"{name}.osm.pbf"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        cities.append(_City(name, _selection_polygon(polygon), path))

    route_values = tuple(sorted(set(route_values))) if route_values is not None else default_route_values()
    keep = {str(k): {int(i) for i in v} for k, v in (keep_relations or {}).items()}
    unknown = set(keep) - {c.name for c in cities}
    if unknown:
        raise ValueError(f"keep_relations names without a polygon: {sorted(unknown)}")

    logger.info(f"Selecting ways and routes for {len(cities)} polygon(s) in {region_pbf.name} ...")
    tree = STRtree([c.polygon for c in cities])
    _select_ways(file, cities, tree, way_keys)
    route_rels = _select_routes(file, cities, tree, route_values)
    _complete_relations(file, cities, route_rels, keep)
    _complete_ways(file, cities)

    logger.info(f"Writing {len(cities)} city file(s) from {region_pbf.name} ...")
    file.extract(
        [(c.path, *c.ids(), c.polygon.bounds) for c in cities],
        "iduedu split_by_polygons",
    )
    return {c.name: c.path for c in cities}


def _select_ways(file, cities: list[_City], tree: STRtree, way_keys) -> None:
    """Ways with one of ``way_keys`` touching each city's polygon, with their nodes."""
    keys = list(way_keys) if way_keys is not None else None
    ids, ref_offsets, refs, coord_offsets, lon, lat, _ = file.ways(
        any_keys=keys, bboxes=[c.polygon.bounds for c in cities], tags=False
    )
    if ids.size == 0:
        return
    counts = np.diff(coord_offsets)
    geoms = np.empty(len(ids), dtype=object)
    multi, single = np.flatnonzero(counts >= 2), np.flatnonzero(counts == 1)
    if multi.size:
        spans = np.concatenate([np.arange(coord_offsets[w], coord_offsets[w + 1]) for w in multi])
        geoms[multi] = shapely.linestrings(
            lon[spans], lat[spans], indices=np.repeat(np.arange(multi.size), counts[multi])
        )
    if single.size:
        geoms[single] = shapely.points(lon[coord_offsets[single]], lat[coord_offsets[single]])
    way_idx, city_idx = tree.query(geoms, predicate="intersects")
    way_of_ref = np.repeat(np.arange(len(ids)), np.diff(ref_offsets))
    for c, city in enumerate(cities):
        selected = np.unique(way_idx[city_idx == c])
        city.way_parts.append(ids[selected])
        city.selected_ways = set(ids[selected].tolist())
        city.node_parts.append(refs[np.isin(way_of_ref, selected)])


def _select_routes(file, cities: list[_City], tree: STRtree, route_values) -> dict:
    """Route relations with a node member inside or a selected way member, per city.

    Returns ``{relation id: (members, route)}`` of the routes selected in any city.
    """
    routes = _relations(file, tags=[("route", v) for v in route_values])
    node_refs = np.unique(np.fromiter((ref for _, ms, _ in routes for k, ref, _ in ms if k == "n"), dtype=np.int64))
    nids, lon, lat, _ = file.nodes(node_refs)
    inside: list[set[int]] = [set() for _ in cities]
    if nids.size:
        node_idx, city_idx = tree.query(shapely.points(lon, lat), predicate="intersects")
        for n, c in zip(nids[node_idx].tolist(), city_idx.tolist()):
            inside[c].add(n)

    route_rels: dict[int, tuple] = {}
    for rid, members, tags in routes:
        route = tags.get("route")
        for c, city in enumerate(cities):
            hit = any(
                (kind == "w" and ref in city.selected_ways) or (kind == "n" and ref in inside[c])
                for kind, ref, _ in members
            )
            if not hit:
                continue
            route_rels[rid] = (members, route)
            city.complete(rid, members)
            if route == "subway":
                city.subway_nodes.update(ref for kind, ref, _ in members if kind == "n")
    return route_rels


def _complete_relations(file, cities: list[_City], route_rels: dict, keep: dict) -> None:
    """Add stop areas, stop-area groups, nested stop-area members and kept relations to the cities."""
    by_name = {c.name: c for c in cities}

    keep_ids = sorted({i for ids in keep.values() for i in ids})
    if keep_ids:
        for rid, members, _ in _relations(file, ids=np.asarray(keep_ids, dtype=np.int64)):
            for name, ids in keep.items():
                if rid in ids:
                    by_name[name].complete(rid, members)

    # Relations sharing a node with a subway route: Overpass' rel(bn.route_nodes), the stop areas.
    subway_nodes = sorted({n for c in cities for n in c.subway_nodes})
    if subway_nodes:
        for rid, members, _ in _relations(file, contains_nodes=np.asarray(subway_nodes, dtype=np.int64)):
            node_refs = {ref for kind, ref, _ in members if kind == "n"}
            for city in cities:
                if node_refs & city.subway_nodes:
                    city.complete(rid, members)
                    city.stop_areas.add(rid)
                    city.nested.update(ref for kind, ref, _ in members if kind == "r")

    if not any(c.stop_areas for c in cities):
        return

    # Groups containing those stop areas: rel(br.stop_areas)["public_transport"="stop_area_group"].
    for rid, members, tags in _relations(file, tags=[("public_transport", "stop_area_group")]):
        if tags.get("type") != "public_transport":
            continue
        rel_refs = {ref for kind, ref, _ in members if kind == "r"}
        for city in cities:
            if rel_refs & city.stop_areas:
                city.relations.add(rid)

    # Relation members of stop areas (station relations and the like), complete for their centres.
    nested_ids = sorted({i for c in cities for i in c.nested} - set(route_rels))
    if nested_ids:
        for rid, members, _ in _relations(file, ids=np.asarray(nested_ids, dtype=np.int64)):
            for city in cities:
                if rid in city.nested:
                    city.complete(rid, members)


def _complete_ways(file, cities: list[_City]) -> None:
    """Add the nodes of ways that entered a city as relation members."""
    member_ways = sorted({w for c in cities for w in c.member_ways})
    if not member_ways:
        return
    ids, ref_offsets, refs, *_ = file.ways(ids=np.asarray(member_ways, dtype=np.int64), locations=False, tags=False)
    way_of_ref = np.repeat(np.arange(len(ids)), np.diff(ref_offsets))
    for city in cities:
        wanted = np.flatnonzero(np.isin(ids, np.fromiter(city.member_ways, dtype=np.int64)))
        city.node_parts.append(refs[np.isin(way_of_ref, wanted)])
