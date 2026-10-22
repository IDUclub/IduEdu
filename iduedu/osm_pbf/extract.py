"""Cut compact per-city OSM files out of a large regional extract.

:func:`split_by_polygons` reads a regional extract (for instance a Geofabrik country file) once for the
spatial selection of every city at the same time, completes the references in a few cheap
relation-only and way-only passes, and writes one sorted ``.osm.pbf`` per city in a final pass. Each
city file holds what IduEdu's queries can reach from that city's polygon -- as ``osmium extract
--strategy smart`` would, restricted to the objects IduEdu reads -- so :class:`~iduedu.osm_pbf.PbfSource`
answers them on the small file exactly as on the regional one:

- ways carrying one of ``way_keys`` (by default ``highway``, ``railway``, ``public_transport``) that touch
  the polygon under Overpass ``poly:`` semantics, with all their nodes;
- public-transport route relations (``route`` in ``route_values``) with a node or way member touching
  the polygon, with all their members complete, also outside the polygon;
- for subway routes, every relation sharing a node with them (their stop areas) with all members and
  their relation members complete, and the ``stop_area_group`` relations that contain those stop areas;
- relations listed in ``keep_relations`` (for example the city boundary, so that ``osm_id=`` keeps
  working) with their node and way members complete.

Tags are kept on every object, including referenced ones outside the polygon. The header keeps the
source's replication timestamp, so :func:`~iduedu.osm_pbf.snapshot_time` works on the city files.
"""

import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import shapely
from shapely import MultiPolygon, Polygon, STRtree

from iduedu import config
from iduedu.constants.transport_specs import DEFAULT_REGISTRY_W_TRAIN, osm_route_values
from iduedu.osm_pbf.reader import _selection_polygon, require_osmium

logger = config.logger

DEFAULT_WAY_KEYS = ("highway", "railway", "public_transport")
_HEADER_KEYS = (
    "osmosis_replication_timestamp",
    "osmosis_replication_sequence_number",
    "osmosis_replication_base_url",
    "timestamp",
)
_CHUNK = 50_000


def default_route_values() -> tuple[str, ...]:
    """OSM ``route`` values of every mode in ``DEFAULT_REGISTRY_W_TRAIN``, aliases included."""
    return tuple(sorted({v for t in DEFAULT_REGISTRY_W_TRAIN.list_types() for v in osm_route_values(t)}))


class _City:  # pylint: disable=too-many-instance-attributes
    """Selection state of one output file. ``union`` tracks the objects of all cities together."""

    def __init__(self, name: str, polygon, path: Path, union):
        module = require_osmium()
        self.name = name
        self.polygon = polygon
        self.path = path
        self.tracker = module.IdTracker()
        self.union = union
        self.selected_ways = module.index.IdSet()  # ways touching the polygon, not those added as members
        self.subway_nodes: set[int] = set()
        self.stop_areas: set[int] = set()
        self.nested: set[int] = set()
        self.pending_ways: set[int] = set()

    def add_node(self, nid: int) -> None:
        self.tracker.add_node(nid)
        self.union.add_node(nid)

    def add_way(self, wid: int) -> None:
        self.tracker.add_way(wid)
        self.union.add_way(wid)

    def add_relation(self, rid: int) -> None:
        self.tracker.add_relation(rid)
        self.union.add_relation(rid)

    def complete(self, rid: int, members) -> None:
        """Track a relation and all its node and way members."""
        self.add_relation(rid)
        for kind, ref, _ in members:
            if kind == "n":
                self.add_node(ref)
            elif kind == "w" and ref not in self.tracker.way_ids():
                self.add_way(ref)
                self.pending_ways.add(ref)


def _members(rel) -> tuple:
    return tuple((m.type, m.ref, m.role) for m in rel.members)


def split_by_polygons(
    region_pbf: str | os.PathLike,
    polygons: Mapping[str, Polygon | MultiPolygon],
    out_dir: str | os.PathLike,
    *,
    location_storage: str = "flex_mem",
    way_keys: Iterable[str] | None = DEFAULT_WAY_KEYS,
    route_values: Iterable[str] | None = None,
    keep_relations: Mapping[str, Iterable[int]] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write one compact, referentially complete ``<name>.osm.pbf`` per polygon from a regional extract.

    Parameters:
        region_pbf: Sorted regional OSM file (``.osm.pbf`` as distributed by Geofabrik, or any format
            libosmium reads).
        polygons: ``{name: polygon}`` in EPSG:4326. Use the polygons (or larger ones) you will later
            build graphs for; as with Overpass, only the exterior ring counts. ``name`` becomes the
            file name and must not contain path separators.
        out_dir: Output directory, created if missing.
        location_storage: Node-location index for the spatial pass, as accepted by
            ``osmium.index.create_map``. ``"flex_mem"`` keeps it in memory. For country-sized files use a
            disk-backed index: ``"sparse_file_array"`` or ``"dense_file_array"`` alone put it in an
            anonymous temporary file that libosmium removes itself; ``"dense_file_array,/scratch/nodes.idx"``
            names the file (it is left in place, and on Windows stays locked until the process exits).
            ``dense_file_array`` suits extracts with a large share of all node ids (big countries),
            ``sparse_file_array`` smaller ones.
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
        ImportError: When pyosmium is not installed.
        FileExistsError: When an output exists and ``overwrite`` is False.
    """
    osmium = require_osmium()
    region_pbf = Path(region_pbf)
    if not region_pbf.is_file():
        raise FileNotFoundError(f"OSM file not found: {region_pbf}")
    if not polygons:
        return {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    union = osmium.IdTracker()
    cities: list[_City] = []
    for name, polygon in polygons.items():
        name = str(name)
        if not name or any(sep in name for sep in ("/", "\\", os.sep)) or name in {".", ".."}:
            raise ValueError(f"Invalid city name for a file name: {name!r}")
        path = out_dir / f"{name}.osm.pbf"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        cities.append(_City(name, _selection_polygon(polygon), path, union))

    route_values = tuple(sorted(set(route_values))) if route_values is not None else default_route_values()
    keep = {str(k): {int(i) for i in v} for k, v in (keep_relations or {}).items()}
    unknown = set(keep) - {c.name for c in cities}
    if unknown:
        raise ValueError(f"keep_relations names without a polygon: {sorted(unknown)}")

    locations = osmium.index.create_map(location_storage)
    route_rels = _spatial_pass(osmium, region_pbf, cities, locations, way_keys, route_values)
    locations.clear()
    del locations

    _complete_relations(osmium, region_pbf, cities, route_rels, keep)
    _complete_ways(osmium, region_pbf, cities)
    _write(osmium, region_pbf, cities, union, overwrite)
    return {c.name: c.path for c in cities}


def _spatial_pass(osmium, region_pbf: Path, cities: list[_City], locations, way_keys, route_values) -> dict:
    """Select ways and route relations per city; returns ``{relation id: (members, route)}`` of selected routes."""
    tree = STRtree([c.polygon for c in cities])
    fp = osmium.FileProcessor(str(region_pbf)).with_locations(locations)
    fp.with_filter(osmium.filter.EntityFilter(osmium.osm.WAY | osmium.osm.RELATION))
    if way_keys is not None:
        key_filter = osmium.filter.KeyFilter(*way_keys)
        key_filter.enable_for(osmium.osm.WAY)
        fp.with_filter(key_filter)
    route_filter = osmium.filter.TagFilter(*[("route", v) for v in route_values])
    route_filter.enable_for(osmium.osm.RELATION)
    fp.with_filter(route_filter)

    chunk: list[tuple] = []
    selected_ways = osmium.index.IdSet()
    route_rels: dict[int, tuple] = {}
    flushed = False

    def flush():
        if not chunk:
            return
        geoms = []
        for _, _, coords in chunk:
            geoms.append(shapely.LineString(coords) if len(coords) > 1 else shapely.Point(coords[0]))
        geom_idx, city_idx = tree.query(np.asarray(geoms, dtype=object), predicate="intersects")
        for g, c in zip(geom_idx.tolist(), city_idx.tolist()):
            wid, refs, _ = chunk[g]
            city = cities[c]
            city.add_way(wid)
            city.selected_ways.set(wid)
            for ref in refs:
                city.add_node(ref)
            selected_ways.set(wid)
        chunk.clear()

    logger.info(f"Selecting ways and routes for {len(cities)} polygon(s) in {region_pbf.name} ...")
    for obj in fp:
        if obj.is_way():
            refs = []
            coords = []
            for node in obj.nodes:
                refs.append(node.ref)
                if node.location.valid():
                    coords.append((node.location.lon, node.location.lat))
            if coords:
                chunk.append((obj.id, refs, coords))
                if len(chunk) >= _CHUNK:
                    flush()
            continue
        if not flushed:
            flush()
            flushed = True
        members = _members(obj)
        route = obj.tags.get("route")
        hit_cities = _relation_cities(members, cities, tree, selected_ways, locations)
        if hit_cities:
            route_rels[obj.id] = (members, route)
            for c in hit_cities:
                cities[c].complete(obj.id, members)
                if route == "subway":
                    cities[c].subway_nodes.update(ref for kind, ref, _ in members if kind == "n")
    flush()
    return route_rels


def _relation_cities(members, cities: list[_City], tree: STRtree, selected_ways, locations) -> list[int]:
    """Indices of the cities in which a relation has a node member inside or a selected way member."""
    hits = set()
    if any(kind == "w" and ref in selected_ways for kind, ref, _ in members):
        for c, city in enumerate(cities):
            if any(kind == "w" and ref in city.selected_ways for kind, ref, _ in members):
                hits.add(c)
    xs, ys = [], []
    for kind, ref, _ in members:
        if kind != "n":
            continue
        try:
            location = locations.get(ref)
        except KeyError:
            continue
        if location.valid():
            xs.append(location.lon)
            ys.append(location.lat)
    if xs and len(hits) < len(cities):
        points = shapely.points(np.asarray(xs), np.asarray(ys))
        _, city_idx = tree.query(points, predicate="intersects")
        hits.update(city_idx.tolist())
    return sorted(hits)


def _complete_relations(osmium, region_pbf: Path, cities: list[_City], route_rels: dict, keep: dict) -> None:
    """Add stop areas, stop-area groups, nested stop-area members and kept relations to the cities."""
    by_name = {c.name: c for c in cities}

    keep_ids = sorted({i for ids in keep.values() for i in ids})
    if keep_ids:
        fp = osmium.FileProcessor(str(region_pbf), osmium.osm.RELATION).with_filter(osmium.filter.IdFilter(keep_ids))
        for rel in fp:
            members = _members(rel)
            for name, ids in keep.items():
                if rel.id in ids:
                    by_name[name].complete(rel.id, members)

    # Relations sharing a node with a subway route: Overpass' rel(bn.route_nodes), the stop areas.
    tracker = osmium.IdTracker()
    for city in cities:
        for nid in city.subway_nodes:
            tracker.add_node(nid)
    if any(c.subway_nodes for c in cities):
        fp = osmium.FileProcessor(str(region_pbf), osmium.osm.RELATION).with_filter(tracker.contains_filter())
        for rel in fp:
            members = _members(rel)
            node_refs = {ref for kind, ref, _ in members if kind == "n"}
            for city in cities:
                if node_refs & city.subway_nodes:
                    city.complete(rel.id, members)
                    city.stop_areas.add(rel.id)
                    city.nested.update(ref for kind, ref, _ in members if kind == "r")

    if not any(c.stop_areas for c in cities):
        return

    # Groups containing those stop areas: rel(br.stop_areas)["public_transport"="stop_area_group"].
    fp = osmium.FileProcessor(str(region_pbf), osmium.osm.RELATION).with_filter(
        osmium.filter.TagFilter(("public_transport", "stop_area_group"))
    )
    for rel in fp:
        if rel.tags.get("type") != "public_transport":
            continue
        rel_refs = {m.ref for m in rel.members if m.type == "r"}
        for city in cities:
            if rel_refs & city.stop_areas:
                city.add_relation(rel.id)

    # Relation members of stop areas (station relations and the like), complete for their centres.
    nested_ids = sorted({i for c in cities for i in c.nested} - set(route_rels))
    if nested_ids:
        fp = osmium.FileProcessor(str(region_pbf), osmium.osm.RELATION).with_filter(osmium.filter.IdFilter(nested_ids))
        for rel in fp:
            members = _members(rel)
            for city in cities:
                if rel.id in city.nested:
                    city.complete(rel.id, members)


def _complete_ways(osmium, region_pbf: Path, cities: list[_City]) -> None:
    """Add the nodes of ways that entered a city as relation members."""
    tracker = osmium.IdTracker()
    for city in cities:
        for wid in city.pending_ways:
            tracker.add_way(wid)
    if not any(c.pending_ways for c in cities):
        return
    fp = osmium.FileProcessor(str(region_pbf), osmium.osm.WAY).with_filter(tracker.id_filter())
    for way in fp:
        refs = [n.ref for n in way.nodes]
        for city in cities:
            if way.id in city.pending_ways:
                for ref in refs:
                    city.add_node(ref)


def _write(osmium, region_pbf: Path, cities: list[_City], union, overwrite: bool) -> None:
    source_header = osmium.FileProcessor(str(region_pbf)).header
    node_sets, way_sets, rel_sets, writers = [], [], [], []
    try:
        for city in cities:
            node_sets.append(city.tracker.node_ids())
            way_sets.append(city.tracker.way_ids())
            rel_sets.append(city.tracker.relation_ids())
            header = osmium.io.Header()
            for key in _HEADER_KEYS:
                value = source_header.get(key, "")
                if value:
                    header.set(key, value)
            header.set("generator", "iduedu split_by_polygons")
            minx, miny, maxx, maxy = city.polygon.bounds
            header.add_box(osmium.osm.Box(osmium.osm.Location(minx, miny), osmium.osm.Location(maxx, maxy)))
            writers.append(osmium.SimpleWriter(str(city.path), header=header, overwrite=overwrite))

        logger.info(f"Writing {len(cities)} city file(s) from {region_pbf.name} ...")
        fp = osmium.FileProcessor(str(region_pbf)).with_filter(union.id_filter())
        for obj in fp:
            if obj.is_node():
                sets = node_sets
            elif obj.is_way():
                sets = way_sets
            else:
                sets = rel_sets
            oid = obj.id
            for writer, ids in zip(writers, sets):
                if oid in ids:
                    writer.add(obj)
    finally:
        for writer in writers:
            writer.close()
