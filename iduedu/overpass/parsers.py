import math
import re
from collections import defaultdict
from functools import lru_cache
from itertools import chain, combinations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely import LineString, Point

from iduedu import _rust, config
from iduedu.constants.highway_enums import HighwayType
from iduedu.constants.transport_specs import canonical_transport_type
from iduedu.overpass.downloaders import fetch_member_tags

logger = config.logger

PLATFORM_ROLES = ["platform_entry_only", "platform", "platform_exit_only"]
STOPS_ROLES = ["stop", "stop_exit_only", "stop_entry_only"]
SUBWAY_EDGE_COLUMNS = ["u_ref", "v_ref", "type", "length_meter", "time_min", "geometry", "oneway"]


def _empty_subway_edges(to_crs) -> gpd.GeoDataFrame:
    """Return an empty edge table shaped like the output of :func:`parse_overpass_subway_data`.

    Callers merge this table with route-derived edges on ``["u", "v", "type"]``, so the columns
    have to be present even when a territory has no stop-area data at all.
    """

    return gpd.GeoDataFrame(
        pd.DataFrame(columns=SUBWAY_EDGE_COLUMNS),
        geometry=gpd.GeoSeries([], crs=to_crs),
        crs=to_crs,
    )


def parse_maxspeed_to_m_per_min(raw: str | int | float | None) -> float | None:
    """Parse an OSM ``maxspeed`` value into meters per minute."""

    if raw is None:
        return None

    s = str(raw).strip().lower()
    if not s:
        return None

    try:
        v = float(s)
        return v * 1000.0 / 60.0
    except ValueError:
        pass

    # "50 km/h" / "50kph" / "50kmh"
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(km/h|kmh|kph)$", s)
    if m:
        v = float(m.group(1))
        return v * 1000.0 / 60.0

    # mph
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(mph)$", s)
    if m:
        v = float(m.group(1))
        v_kmh = v * 1.60934
        return v_kmh * 1000.0 / 60.0

    return None


def overpass_routes_to_df(json_routes: list[dict], enable_subway_details: bool) -> pd.DataFrame:
    """Normalize raw Overpass route elements into a parser DataFrame."""

    def _compute_way_speed(tags: dict) -> float:
        highway = tags.get("highway")
        if not highway:
            return HighwayType.UNCLASSIFIED.max_speed
        maxspeed_tag = tags.get("maxspeed")
        speed = parse_maxspeed_to_m_per_min(maxspeed_tag)
        if speed is None:
            try:
                ht = HighwayType(highway)
                speed = ht.max_speed
            except ValueError:
                speed = HighwayType.UNCLASSIFIED.max_speed
        return speed

    if not json_routes:
        empty = pd.DataFrame()
        for col in ("is_stop_area", "is_stop_area_group", "is_station"):
            empty[col] = pd.Series(dtype=bool)
        return empty

    for e in json_routes:
        tags = e.get("tags") or {}
        etype = e.get("type")

        # The tag is normalised to the registry's vocabulary here, once, so nothing
        # downstream has to know that OSM calls a tram "light_rail" in some cities.
        # The original value stays available for anyone who needs to tell them apart.
        route_type = tags.get("route")
        e["osm_route_value"] = route_type
        e["transport_type"] = canonical_transport_type(route_type) if route_type else route_type

        if etype == "way" and "highway" in tags:
            e["is_way_data"] = True
            speed = _compute_way_speed(tags)
            e["way_speed_m_per_min"] = speed

        if enable_subway_details:
            is_stop_area = etype == "relation" and tags.get("public_transport") == "stop_area"
            is_stop_area_group = (
                etype == "relation"
                and tags.get("public_transport") == "stop_area_group"
                and tags.get("type") == "public_transport"
            )
            is_station = tags.get("public_transport") == "station"

            e["is_stop_area"] = is_stop_area
            e["is_stop_area_group"] = is_stop_area_group
            e["is_station"] = is_station

            if is_stop_area or is_stop_area_group or is_station:
                e["transport_type"] = "subway"

    data = pd.DataFrame(json_routes)

    for col in ("is_stop_area", "is_stop_area_group", "is_station", "is_way_data"):
        if col not in data.columns:
            data[col] = False
        else:
            data[col] = data[col].astype("boolean").fillna(False).astype(bool)

    return data


def extract_needed(loc_obj: pd.Series, keys: list[str]) -> dict | None:  # pragma: no cover
    """Extract selected nested fields and tags from an Overpass element."""
    out = {}
    tags = loc_obj.get("tags", {}) if isinstance(loc_obj.get("tags"), dict) else {}
    for k in keys:
        value_found = None
        if "." in k:
            # dotted path
            parts = k.split(".")
            cur = loc_obj
            ok = True
            for part in parts:
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if ok:
                value_found = cur
        else:
            if k in tags:
                value_found = tags[k]
            elif k in loc_obj:
                value_found = loc_obj[k]
        if value_found is not None:
            out[k] = value_found
    return out or None


GROUND_EDGE_COLUMNS = ["u", "v", "type", "geometry", "oneway", "extra_data", "speed_m_min", "route", "route_key"]
GROUND_NODE_COLUMNS = ["node_id", "geometry", "type", "route", "route_key", "extra_data"]
# Tags a stop, platform or entrance node keeps of its OSM object, as columns of the same name.
NODE_TAG_COLUMNS = ("name", "ref", "local_ref", "wheelchair")
_WAY_ROLES = frozenset({"", "forward", "backward"})
_PLATFORM_ROLE_SET = frozenset(PLATFORM_ROLES)
_STOP_ROLE_SET = frozenset(STOPS_ROLES)
_NO_POINT = (np.nan, np.nan)


@lru_cache(maxsize=8)
def _to_local(local_crs) -> Transformer:
    """EPSG:4326 to the local CRS, made once per CRS rather than once per route."""
    return Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)


def _missing(value) -> bool:
    """Missing as pandas sees a scalar: None or NaN."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def osm_node_attributes(kind, ref, tags) -> dict:
    """Attributes a graph node keeps of the OSM object it is made from.

    The tags of ``NODE_TAG_COLUMNS`` that the object has, and ``osm_id`` as the object's path on
    openstreetmap.org (``node/<id>``, ``way/<id>`` or ``relation/<id>``) when ``kind`` is given; a node made up
    from another object (a platform from its stop) takes that object's tags but no ``osm_id``.
    """
    tags = tags if isinstance(tags, dict) else {}
    out = {key: str(tags[key]).strip() for key in NODE_TAG_COLUMNS if not _missing(tags.get(key))}
    out = {key: value for key, value in out.items() if value}
    if kind in ("node", "way", "relation") and ref is not None:
        out["osm_id"] = f"{kind}/{int(ref)}"
    return out


def route_member_tags(elements: pd.DataFrame) -> dict:
    """Tags of the node and way members of route relations in a route response, keyed by ``(type, id)``."""
    if elements is None or len(elements) == 0 or "type" not in elements or "tags" not in elements:
        return {}
    members = elements[elements["type"].isin(("node", "way"))]
    return {
        (kind, int(eid)): tags
        for kind, eid, tags in zip(members["type"], members["id"], members["tags"])
        if isinstance(tags, dict) and not _missing(eid)
    }


def _node_attributes(rows: list, node_route: np.ndarray, node_ids: list, member_tags: dict) -> list[dict]:
    """``osm_node_attributes`` of every kernel node: of its stop or platform member, or, for a node made up
    from a member (``from_<ref>``), that member's tags without ``osm_id``."""
    kinds: dict[int, dict] = {}
    out = []
    for route, node_id in zip(node_route.tolist(), node_ids):
        if route not in kinds:
            members = rows[route].get("members")
            kinds[route] = {
                str(member.get("ref")): member.get("type")
                for member in (members if isinstance(members, list) else [])
                if member.get("role") in _PLATFORM_ROLE_SET or member.get("role") in _STOP_ROLE_SET
            }
        made_up = isinstance(node_id, str) and node_id.startswith("from_")
        ref = node_id[len("from_") :] if made_up else str(node_id)
        kind = kinds[route].get(ref)
        try:
            number = int(float(ref))
        except (TypeError, ValueError):
            number = None
        tags = None if number is None else member_tags.get((kind, number))
        out.append(osm_node_attributes(None if made_up else kind, None if made_up else number, tags))
    return out


def _is_nan(value) -> bool:
    """NaN, which never finds itself again as a dict key: a stop without a reference has no rank."""
    return isinstance(value, float) and math.isnan(value)


def _int64(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _member_refs(members: list[dict]) -> list:
    """Member references as a pandas column of them holds them: floats when numbers miss some."""
    refs = [member.get("ref") for member in members]
    if not any(_missing(ref) for ref in refs):
        return refs
    if all(_missing(ref) or (isinstance(ref, (int, float, np.number)) and not isinstance(ref, bool)) for ref in refs):
        return [np.nan if _missing(ref) else float(ref) for ref in refs]
    return [np.nan if "ref" not in member else ref for ref, member in zip(refs, members)]


def _lonlat(geometry) -> np.ndarray:
    """A way's geometry as a ``(k, 2)`` lon/lat array: the PBF reader's array, or Overpass ``{"lat", "lon"}`` dicts."""
    if isinstance(geometry, np.ndarray):
        return geometry
    if not isinstance(geometry, list):
        return np.empty((0, 2))
    return np.asarray([(point["lon"], point["lat"]) for point in geometry], dtype="f8").reshape(-1, 2)


class _RouteInputs:  # pylint: disable=too-many-instance-attributes
    """Flat arrays of route relations for the Rust kernel (``_rust.parse_pt_routes``), one pass over the members.

    ``stand_in`` (for the subway) maps a stop reference to a platform known from its stop area, ``(node id,
    (x, y) or None)``, which a stop left without a platform of the route takes instead of a made-up one.
    """

    def __init__(self, ref2speed: dict | None = None, stand_in=None):
        self.ref2speed = ref2speed or {}
        self.stand_in = stand_in
        # Way points, then member points: each run of offsets counts within its own chunks.
        self.way_chunks: list[np.ndarray] = []
        self.member_chunks: list[np.ndarray] = []
        self.way_offsets, self.way_points, self.way_speeds = [0], [0], []
        self.member_offsets, self.member_points = [0], [0]
        self.member_refs: list[int] = []
        self.member_ranks: list[int] = []
        self.member_platform: list[bool] = []
        self.member_stand_ins: list[int] = []
        self.member_stand_in_xy: list[tuple[float, float]] = []
        self.ref_index: dict = {}

    def _reference(self, value) -> int:
        return self.ref_index.setdefault(value, len(self.ref_index))

    def add_route(self, members) -> None:
        """Ways, stops and platforms of one relation; nothing for a route with no ways or nowhere to board."""
        if not isinstance(members, list):
            members = []
        kept = [
            (member, ref)
            for member, ref in zip(members, _member_refs(members))
            if not (_missing(member.get("lat")) and _missing(member.get("lon")) and _missing(member.get("geometry")))
        ]
        rank: dict = {}
        boarding = []
        ways = []
        for position, (member, ref) in enumerate(kept):
            role = member.get("role")
            if role in _PLATFORM_ROLE_SET or role in _STOP_ROLE_SET:
                if not _is_nan(ref):
                    rank.setdefault(ref, position)
                boarding.append((member, ref, role in _PLATFORM_ROLE_SET))
            elif member.get("type") == "way" and ("" if _missing(role) else role) in _WAY_ROLES:
                ways.append((member, ref))
        if ways and boarding:
            n_points = self.way_points[-1]
            for member, ref in ways:
                points = _lonlat(member.get("geometry"))
                self.way_chunks.append(points)
                n_points += len(points)
                self.way_points.append(n_points)
                self.way_speeds.append(float(self.ref2speed.get(ref, np.nan)))
            n_points = self.member_points[-1]
            for member, ref, is_platform in boarding:
                geometry = member.get("geometry")
                if isinstance(geometry, (list, np.ndarray)):
                    points = _lonlat(geometry)
                else:
                    points = np.asarray([(member.get("lon", np.nan), member.get("lat", np.nan))], dtype="f8")
                self.member_chunks.append(points)
                n_points += len(points)
                self.member_points.append(n_points)
                self.member_refs.append(self._reference(ref))
                self.member_ranks.append(-1 if _is_nan(ref) else rank[ref])
                self.member_platform.append(is_platform)
                known = None if is_platform or self.stand_in is None else self.stand_in(ref)
                if known is None:
                    self.member_stand_ins.append(-1)
                    self.member_stand_in_xy.append(_NO_POINT)
                else:
                    self.member_stand_ins.append(self._reference(known[0]))
                    self.member_stand_in_xy.append(_NO_POINT if known[1] is None else known[1])
        self.way_offsets.append(len(self.way_speeds))
        self.member_offsets.append(len(self.member_refs))

    def parse(self, local_crs, kinds_apart: bool = False) -> tuple:
        """Project the points and run the kernel; its arrays and the reference values in index order."""
        lonlat = np.concatenate(self.way_chunks + self.member_chunks or [np.empty((0, 2))]).astype("f8", copy=False)
        xy = np.empty(lonlat.shape, dtype=np.float64)
        if len(lonlat):
            xy[:, 0], xy[:, 1] = _to_local(local_crs).transform(lonlat[:, 0], lonlat[:, 1])
        result = _rust.parse_pt_routes(
            xy,
            _int64(self.way_offsets),
            _int64(self.way_points),
            np.asarray(self.way_speeds, dtype=np.float64),
            _int64(self.member_offsets),
            _int64(self.member_points) + self.way_points[-1],
            _int64(self.member_refs),
            _int64(self.member_ranks),
            np.asarray(self.member_platform, dtype=bool),
            _int64(self.member_stand_ins),
            np.asarray(self.member_stand_in_xy, dtype=np.float64).reshape(-1, 2),
            kinds_apart=kinds_apart,
        )
        return result, list(self.ref_index)


def _node_values(codes: np.ndarray, ref_values: list) -> list:
    """Node ids of the kernel's codes: the member reference, or ``from_<reference>`` for a made-up node."""
    if len(codes) == 0:
        return []
    unique, inverse = np.unique(codes, return_inverse=True)
    values = np.empty(len(unique), dtype=object)
    for i, code in enumerate(unique.tolist()):
        ref = ref_values[code >> 1]
        values[i] = f"from_{ref}" if code & 1 else ref
    return values[inverse].tolist()


def _route_label(tags: dict, loc_id) -> str:
    if "ref" in tags:
        return tags["ref"]
    if "name" in tags:
        return tags["name"]
    return f"Unnamed_{loc_id}"


def _parse_route_rows(routes: pd.DataFrame, local_crs, inputs: _RouteInputs, kinds_apart: bool = False) -> dict:
    """Run the kernel on the route relations of ``routes``: one call for all of them, in parallel.

    For every relation the kernel stitches its ways in member order into its path (pieces that share no vertex
    ordered by the shortest total join, cut where they lie more than a kilometre apart), pairs its stops with
    its platforms, makes up the missing half of a pair, turns a part round when its stops clearly run against
    it, and returns stop nodes projected onto the path, platform nodes, travel edges between consecutive stops
    with the length-weighted speed of the ways under them, and boarding edges. A route that cannot be parsed
    (no connected ways, a degenerate path) is left out with a warning.

    Returns the decoded arrays (``node_*`` and ``edge_*``) and, per route, ``labels``, ``keys`` and ``rows``.
    """
    labels, keys, rows = [], [], []
    for loc_id, loc in routes.iterrows():
        tags = loc.get("tags", {}) if isinstance(loc.get("tags"), dict) else {}
        keys.append(osm_route_key(loc.get("id"), loc_id))
        labels.append(_route_label(tags, loc_id))
        rows.append(loc)
        inputs.add_route(loc.get("members", []))
    result, ref_values = inputs.parse(local_crs, kinds_apart=kinds_apart)
    (node_route, node_id, node_xy, node_platform, edge_route, edge_u, edge_v, boarding, offsets, edge_xy, speed) = (
        result[:11]
    )
    for position, message in result[11]:
        logger.warning(f"Route {keys[position]} ({labels[position]}) is left out: {message}")
    return {
        "labels": np.asarray(labels, dtype=object),
        "keys": np.asarray(keys, dtype=object),
        "rows": rows,
        "node_route": node_route,
        "node_id": _node_values(node_id, ref_values),
        "node_points": shapely.points(node_xy),
        "node_platform": node_platform,
        "edge_route": edge_route,
        "edge_u": _node_values(edge_u, ref_values),
        "edge_v": _node_values(edge_v, ref_values),
        "edge_boarding": boarding,
        "edge_lines": shapely.linestrings(edge_xy, indices=np.repeat(np.arange(len(edge_u)), np.diff(offsets))),
        "edge_speed": speed,
    }


def ground_routes_tables(
    routes: pd.DataFrame, local_crs, ref2speed: dict, needed_tags, member_tags: dict | None = None
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Edge and node tables of ground public-transport route relations, parsed by one Rust kernel call.

    See :func:`_parse_route_rows` for what the kernel does. Travel edges carry the OSM tags ``needed_tags``
    of their relation in ``extra_data``, nodes the attributes of their stop or platform (``member_tags``, see
    :func:`osm_node_attributes`); records carry ``route`` and ``route_key``; within a route a node appears once.
    """
    parsed = _parse_route_rows(routes, local_crs, _RouteInputs(ref2speed))
    if len(parsed["node_id"]) == 0 and len(parsed["edge_u"]) == 0:
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()
    labels, keys = parsed["labels"], parsed["keys"]
    types = np.asarray([loc.transport_type for loc in parsed["rows"]], dtype=object)
    extras = [extract_needed(loc, needed_tags) for loc in parsed["rows"]]

    node_route = parsed["node_route"]
    node_types = types[node_route]
    node_types[parsed["node_platform"]] = "platform"
    nodes_gdf = gpd.GeoDataFrame(
        {
            "node_id": parsed["node_id"],
            "geometry": parsed["node_points"],
            "type": node_types.tolist(),
            "route": labels[node_route].tolist(),
            "route_key": keys[node_route].tolist(),
            "extra_data": _node_attributes(parsed["rows"], node_route, parsed["node_id"], member_tags or {}),
        },
        columns=GROUND_NODE_COLUMNS,
        geometry="geometry",
        crs=local_crs,
    )

    edge_route, boarding = parsed["edge_route"], parsed["edge_boarding"]
    edge_types = types[edge_route]
    edge_types[boarding] = "boarding"
    extra = np.asarray([np.nan if value is None else value for value in extras] + [np.nan], dtype=object)
    edges_gdf = gpd.GeoDataFrame(
        {
            "u": parsed["edge_u"],
            "v": parsed["edge_v"],
            "type": edge_types.tolist(),
            "geometry": parsed["edge_lines"],
            "oneway": (~boarding).tolist(),
            "extra_data": extra[np.where(boarding, len(extras), edge_route)].tolist(),
            "speed_m_min": parsed["edge_speed"],
            "route": labels[edge_route].tolist(),
            "route_key": keys[edge_route].tolist(),
        },
        columns=GROUND_EDGE_COLUMNS,
        geometry="geometry",
        crs=local_crs,
    )
    return edges_gdf, nodes_gdf


def osm_route_key(relation_id, fallback) -> str:
    """Key of an OSM route variant: ``osm:r<relation id>``; ``osm:row<n>`` when the id is unknown."""
    try:
        if relation_id is not None and not pd.isna(relation_id):
            return f"osm:r{int(relation_id)}"
    except (TypeError, ValueError):
        pass
    return f"osm:row{fallback}"


def merge_on_ids(left: pd.DataFrame, right: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Outer merge on node-id keys that mix OSM ids (int) and synthetic ids (str, ``"from_<id>"``).

    When every key on one side happens to be a string, pandas infers a string dtype
    and refuses to merge it with the integer keys of the other side; Turin's subway
    hit this. The keys are compared as Python objects, so ``123`` and ``"123"`` stay distinct.
    """
    left, right = left.copy(), right.copy()
    for frame in (left, right):
        for column in on:
            frame[column] = frame[column].astype(object)
    return pd.merge(left, right, how="outer", on=on)


def _stop_area_platforms(add_edges: pd.DataFrame, add_nodes: pd.DataFrame):
    """For a subway stop: the platform its stop area links it to by a boarding edge, ``(node id, (x, y) or None)``.

    Of several, the first among the stop-area nodes; None when the stop links to no platform.
    """
    neighbours: dict = defaultdict(list)
    if {"u", "v", "type"} <= set(add_edges.columns):
        boarding = add_edges[add_edges["type"] == "boarding"]
        for u, v in zip(boarding["u"], boarding["v"]):
            neighbours[u].append(v)
            neighbours[v].append(u)
    first: dict = {}
    if {"node_id", "type", "geometry"} <= set(add_nodes.columns):
        platforms = add_nodes[add_nodes["type"] == "subway_platform"]
        for position, (node_id, geometry) in enumerate(zip(platforms["node_id"], platforms["geometry"])):
            first.setdefault(node_id, (position, node_id, geometry))

    def stand_in(ref):
        found = [first[node] for node in neighbours.get(ref, ()) if node in first]
        if not found:
            return None
        _, node_id, geometry = min(found, key=lambda item: item[0])
        return node_id, (None if geometry is None or pd.isna(geometry) else (geometry.x, geometry.y))

    return stand_in


def overpass_subway2edgenode(
    subway_data: pd.DataFrame, local_crs, known_members: dict | None = None
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Convert subway Overpass route and stop-area data into edge and node tables.

    ``known_members``: stop-area member tags and coordinates that arrived with the
    route response; see :func:`patch_members_roles_inplace`.
    """

    transport_type = "subway"

    for column in ["is_stop_area", "is_stop_area_group", "is_station"]:
        subway_data[column] = subway_data[column].astype("boolean").fillna(False)
    routes_data = subway_data[
        ~((subway_data["is_stop_area"]) | (subway_data["is_stop_area_group"]) | (subway_data["is_station"]))
    ].copy()
    stop_areas = subway_data[subway_data["is_stop_area"]].copy()
    stop_areas_group = subway_data[subway_data["is_stop_area_group"]].copy()
    stations_data = subway_data[subway_data["is_station"]].copy()

    add_edges, add_nodes = parse_overpass_subway_data(
        stop_areas, stop_areas_group, stations_data, local_crs, known_members=known_members
    )  # Entrances, exits, transfers and similar edges.
    add_edges = add_edges.rename(columns={"u_ref": "u", "v_ref": "v"})
    add_nodes = add_nodes.rename(columns={"ref_id": "node_id"})

    # One kernel call for all subway routes, as for the ground ones; a stop without a platform of its route
    # takes the platform its stop area links it to.
    parsed = _parse_route_rows(
        routes_data, local_crs, _RouteInputs(stand_in=_stop_area_platforms(add_edges, add_nodes)), kinds_apart=True
    )
    labels, keys = parsed["labels"], parsed["keys"]
    node_route, edge_route, boarding = parsed["node_route"], parsed["edge_route"], parsed["edge_boarding"]
    nodes_data = {
        "node_id": parsed["node_id"],
        "geometry": parsed["node_points"],
        "type": np.where(parsed["node_platform"], "subway_platform", transport_type).tolist(),
        "route": labels[node_route].tolist(),
        "route_key": keys[node_route].tolist(),
        "extra_data": _node_attributes(
            parsed["rows"],
            node_route,
            parsed["node_id"],
            {key: payload.get("tags") for key, payload in (known_members or {}).items()},
        ),
    }
    edges_data = {
        "u": parsed["edge_u"],
        "v": parsed["edge_v"],
        "type": np.where(boarding, "boarding", transport_type).tolist(),
        "geometry": parsed["edge_lines"],
        "oneway": (~boarding).tolist(),
        "route": labels[edge_route].tolist(),
        "route_key": keys[edge_route].tolist(),
    }

    nodes_gdf = gpd.GeoDataFrame(
        nodes_data,
        columns=["node_id", "geometry", "type", "route", "route_key", "extra_data"],
        geometry="geometry",
        crs=local_crs,
    ).drop_duplicates(subset=["node_id", "route_key", "type"])

    merged_nodes = merge_on_ids(add_nodes, nodes_gdf, ["node_id"])

    base_cols = {c[:-2] for c in merged_nodes.columns if c.endswith("_x")}
    for base in base_cols:
        col_x = f"{base}_x"
        col_y = f"{base}_y"
        if merged_nodes[col_x].isna().all():
            merged_nodes[base] = merged_nodes[col_y]
        elif merged_nodes[col_y].isna().all():
            merged_nodes[base] = merged_nodes[col_x]
        else:
            merged_nodes[base] = merged_nodes[col_x].combine_first(merged_nodes[col_y])
        merged_nodes = merged_nodes.drop(columns=[col_x, col_y])

    merged_nodes = gpd.GeoDataFrame(merged_nodes, geometry="geometry", crs=local_crs)

    edges_gdf = gpd.GeoDataFrame(
        edges_data,
        columns=["u", "v", "type", "geometry", "oneway", "route", "route_key"],
        geometry="geometry",
        crs=local_crs,
    )

    merged_edges = merge_on_ids(add_edges, edges_gdf, ["u", "v", "type"])
    base_cols = {c[:-2] for c in merged_edges.columns if c.endswith("_x")}
    for base in base_cols:
        col_x = f"{base}_x"
        col_y = f"{base}_y"

        if merged_edges[col_x].isna().all():
            merged_edges[base] = merged_edges[col_y]
        elif merged_edges[col_y].isna().all():
            merged_edges[base] = merged_edges[col_x]
        else:
            merged_edges[base] = merged_edges[col_x].combine_first(merged_edges[col_y])
        merged_edges = merged_edges.drop(columns=[col_x, col_y])

    merged_edges = gpd.GeoDataFrame(merged_edges, geometry="geometry", crs=local_crs)

    platforms_ids = merged_nodes[merged_nodes["type"] == "subway_platform"]["node_id"].unique()
    platform_link_edges = merged_edges[
        (merged_edges["type"] != "boarding")
        & (merged_edges["u"].isin(platforms_ids) | merged_edges["v"].isin(platforms_ids))
    ]
    platforms_with_station = pd.concat(
        [
            platform_link_edges.loc[platform_link_edges["u"].isin(platforms_ids), "u"],
            platform_link_edges.loc[platform_link_edges["v"].isin(platforms_ids), "v"],
        ],
        ignore_index=True,
    ).unique()

    platforms_wout_station = (
        pd.Index(platforms_ids.astype(str)).difference(platforms_with_station.astype(str)).to_numpy()
    )

    if len(platforms_wout_station) > 0:
        merged_nodes.loc[merged_nodes["node_id"].astype(str).isin(platforms_wout_station), "type"] = "platform"

    return merged_edges, merged_nodes


def infer_role_from_tags(tags: dict) -> str:
    """Infer a public-transport member role from OSM tags."""
    _TRUE = {"yes", "true", "1"}

    def _is_true(v):
        return str(v).lower() in _TRUE

    if not tags:
        return ""

    pt = (tags.get("public_transport", "")).lower()
    rail = (tags.get("railway", "")).lower()
    stat = (tags.get("station", "")).lower()
    entr = (tags.get("entrance", "")).lower()
    entry = (tags.get("entry", "")).lower()
    exit_ = (tags.get("exit", "")).lower()

    if pt == "station" or rail == "station" or stat == "subway":
        return "station"

    if pt == "platform" or rail == "platform":
        return "platform"

    if pt in {"stop_position", "stop"} or rail in {"stop", "halt"}:
        return "stop"

    if rail == "subway_entrance" or entr in {"yes", "main", "service", "entrance", "entry", "exit"}:
        if entr == "entry" or (_is_true(entry) and not _is_true(exit_)):
            return "entry_only"
        if entr == "exit" or (_is_true(exit_) and not _is_true(entry)):
            return "exit_only"
        return "entrance"
    return ""


#: Node type of a subway stop-area member by its role; a member with another role gets one from its tags.
SUBWAY_ROLE_TYPES = {
    "station": "subway_station",
    "stop": "subway",
    "platform": "subway_platform",
    "entrance": "subway_entry_exit",
    "subway_entrance": "subway_entry_exit",
    "entrance_yes": "subway_entry_exit",
    "entrance_main": "subway_entry_exit",
    "entry_only": "subway_entry",
    "entrance_entry_only": "subway_entry",
    "entry": "subway_entry",
    "exit_only": "subway_exit",
    "entrance_exit_only": "subway_exit",
    "exit": "subway_exit",
}


def _role_unknown(member: dict) -> bool:
    """A stop-area member whose role the subway parser does not read: none, or one such as ``entrence``."""
    return (member.get("role") or "").strip().lower() not in SUBWAY_ROLE_TYPES


def patch_members_roles_inplace(stop_areas_df, known_members: dict | None = None):
    """Fill missing or unknown subway stop-area member roles and missing geometry in place.

    A member whose role is empty or not one the parser reads (OSM has ``entrence`` and the like) takes the
    role its tags give, see :func:`infer_role_from_tags`; a station whose entrances carry a misspelt role
    would otherwise have no entrance and be joined to walking as a platform.

    ``known_members`` are the member tags and coordinates that arrived with the
    route response (see ``split_stop_area_members``). Only members absent from it
    are fetched with a separate request; with ``None`` -- a response that carried no
    member section -- every member needing a role or a position is fetched, as
    before.
    """
    missing = []
    for _, r in stop_areas_df.iterrows():
        for m in r.get("members") or []:
            role_missing = _role_unknown(m)

            has_latlon = "lat" in m and "lon" in m and m["lat"] is not None and m["lon"] is not None
            has_geom = "geometry" in m and m["geometry"]
            geo_missing = (not has_latlon) and (not has_geom)

            if role_missing or geo_missing:
                missing.append({"type": m["type"], "ref": int(m["ref"])})

    if not missing:
        return

    known = known_members or {}
    unknown = [m for m in missing if (m["type"], m["ref"]) not in known]
    tags_map = dict(known)
    if unknown:
        tags_map.update(fetch_member_tags(unknown))

    for _, r in stop_areas_df.iterrows():
        for m in r.get("members") or []:
            key = (m["type"], int(m["ref"]))
            payload = tags_map.get(key, {})

            if _role_unknown(m):
                tags = payload.get("tags", {}) or {}
                role = infer_role_from_tags(tags)
                if role:
                    m["role"] = role

            has_latlon = "lat" in m and "lon" in m and m["lat"] is not None and m["lon"] is not None
            if not has_latlon:
                if "__latlon__" in payload:
                    lon, lat = payload["__latlon__"]
                    m["lon"], m["lat"] = lon, lat
                elif "__center__" in payload:
                    lon, lat = payload["__center__"]
                    m["lon"], m["lat"] = lon, lat

            has_geom = "geometry" in m and m["geometry"]
            if not has_geom and "__geometry__" in payload:
                m["geometry"] = payload["__geometry__"]


def parse_overpass_subway_data(
    stop_areas, stop_areas_group, stations_data, to_crs, known_members: dict | None = None
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Parse detailed subway stop-area data into edge and node tables."""
    graph_nodes = []
    graph_edges = []

    transformer = _to_local(to_crs)

    station_parent_ref = {}
    osm_roles_to_iduedu = SUBWAY_ROLE_TYPES

    def _collect_by_roles(members, parent_id):
        roles = defaultdict(list)
        for m in members or []:
            role = (m.get("role") or "").lower()
            roles[role].append(m)

        stations = roles.get("station", [])
        for s in stations:
            station_parent_ref[parent_id] = s["ref"]
        stops = roles.get("stop", [])
        platforms = roles.get("platform", [])

        entrances = (
            roles.get("entrance", [])
            + roles.get("subway_entrance", [])
            + roles.get("entrance_yes", [])
            + roles.get("entrance_main", [])
        )
        entry_only = roles.get("entry_only", []) + roles.get("entrance_entry_only", []) + roles.get("entry", [])
        exit_only = roles.get("exit_only", []) + roles.get("entrance_exit_only", []) + roles.get("exit", [])

        return stations, stops, platforms, entrances, entry_only, exit_only

    def add_node(ref_id, x, y, node_type, kind=None):
        """Add a graph node record to the local accumulator."""
        if x is None or y is None:
            return
        point = Point(transformer.transform(x, y))
        tags = (known_members or {}).get((kind, int(ref_id)), {}).get("tags")
        graph_nodes.append(
            {
                "ref_id": int(ref_id),
                "geometry": point,
                "type": osm_roles_to_iduedu.get(node_type),
                "extra_data": osm_node_attributes(kind, ref_id, tags),
            }
        )

    def add_edge(u, v, edge_type, oneway=True):
        """Add a graph edge record to the local accumulator."""
        graph_edges.append({"u_ref": int(u), "v_ref": int(v), "type": edge_type, "oneway": bool(oneway)})

    patch_members_roles_inplace(stop_areas, known_members)

    for _, stop_area in stop_areas.iterrows():

        members = stop_area["members"]
        all_nodes = _collect_by_roles(members, stop_area["id"])
        start_idx = len(graph_nodes)

        for node in chain.from_iterable(all_nodes):
            if "geometry" in node:
                new_lon, new_lat = LineString((xy["lon"], xy["lat"]) for xy in node["geometry"]).centroid.xy
                node["lon"], node["lat"] = new_lon[0], new_lat[0]
            if "lon" not in node:
                logger.debug(f"Skipping stop area member without coordinates: {node}")
                continue

            add_node(node["ref"], node["lon"], node["lat"], node["role"], node.get("type"))

        ref2idx = {graph_nodes[i]["ref_id"]: i for i in range(start_idx, len(graph_nodes))}

        stations, stops, platforms, entrances, entry_only, exit_only = all_nodes

        for stop in stops:
            for platform in platforms:
                add_edge(stop["ref"], platform["ref"], "boarding", oneway=False)

        has_station = len(stations) > 0

        if has_station:
            for platform in platforms:
                for station in stations:
                    add_edge(station["ref"], platform["ref"], "subway_station", oneway=False)

        targets = [s["ref"] for s in stations] if has_station else [p["ref"] for p in platforms]

        has_entrance = (len(entrances) + len(entry_only)) > 0
        has_exit = (len(entrances) + len(exit_only)) > 0

        for entrance in entrances:
            for t in targets:
                add_edge(entrance["ref"], t, "subway_entrance")
                add_edge(t, entrance["ref"], "subway_exit")

        for entrance in entry_only:
            for t in targets:
                add_edge(entrance["ref"], t, "subway_entrance")

        for entrance in exit_only:
            for t in targets:
                add_edge(t, entrance["ref"], "subway_exit")

        if not (has_entrance and has_exit):
            for t in targets:
                idx = ref2idx.get(int(t))
                if idx is not None:
                    graph_nodes[idx]["type"] = "platform"

    for _, stop_area_group in stop_areas_group.iterrows():
        members = stop_area_group["members"]

        connect_refs = [station_parent_ref[mem["ref"]] for mem in members if mem["ref"] in station_parent_ref]

        pairs = list(combinations(connect_refs, 2))
        for pair in pairs:
            if pair[0] != pair[1]:
                add_edge(pair[0], pair[1], "subway_transfer", oneway=False)

    nodes_gdf = gpd.GeoDataFrame(
        graph_nodes, columns=["ref_id", "geometry", "type", "extra_data"], geometry="geometry", crs=to_crs
    )
    edges_df = pd.DataFrame(graph_edges, columns=["u_ref", "v_ref", "type", "oneway"])

    if len(nodes_gdf) == 0:
        return _empty_subway_edges(to_crs), nodes_gdf

    nodes_gdf = nodes_gdf.dropna(subset=["geometry"]).drop_duplicates(subset=["ref_id"])

    for _, station_data in stations_data.iterrows():
        tags = station_data["tags"]
        ref_id = station_data["id"]
        station_ind = nodes_gdf[nodes_gdf["ref_id"] == int(ref_id)].index
        extra_data = osm_node_attributes(station_data.get("type"), ref_id, tags)
        extra_data["depth"] = tags.get("depth", 0)
        nodes_gdf.loc[station_ind, "extra_data"] = [dict(extra_data) for _ in station_ind]

    edges_df = edges_df.drop_duplicates(subset=["u_ref", "v_ref", "type"])

    def _depth_from_extra(d):
        if isinstance(d, dict):
            v = d.get("depth", 0)
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    if len(edges_df) == 0:
        return _empty_subway_edges(to_crs), nodes_gdf

    nodes_info = nodes_gdf[["ref_id", "geometry", "type", "extra_data"]].copy()
    nodes_info["depth_m"] = nodes_info["extra_data"].apply(_depth_from_extra)

    nodes_lut = nodes_info.set_index("ref_id")[["geometry", "type", "depth_m"]]

    edges_nodes = (
        edges_df.join(nodes_lut.add_prefix("u_"), on="u_ref")
        .join(nodes_lut.add_prefix("v_"), on="v_ref")
        .drop_duplicates(subset=["u_ref", "v_ref", "type"])
    )

    def _horiz_len(pu, pv):
        if pu is None or pv is None:
            return np.nan
        try:
            return LineString([pu, pv]).length
        except Exception:
            return np.nan

    missing_endpoint = edges_nodes["u_geometry"].isna() | edges_nodes["v_geometry"].isna()
    edges_nodes = edges_nodes.loc[~missing_endpoint].copy()
    if len(edges_nodes) == 0:
        return _empty_subway_edges(to_crs), nodes_gdf

    def _straight_geom(pu, pv):
        if pu is None or pv is None:
            return None
        try:
            return LineString([pu, pv])
        except Exception:
            return None

    edges_nodes["geometry"] = edges_nodes.apply(lambda r: _straight_geom(r["u_geometry"], r["v_geometry"]), axis=1)
    edges_nodes = edges_nodes.dropna(subset=["geometry"]).copy()
    edges_nodes["length_meter"] = 0.0
    edges_nodes["time_min"] = 0.0
    edges_nodes["horiz_m"] = edges_nodes.apply(lambda r: _horiz_len(r["u_geometry"], r["v_geometry"]), axis=1)

    edges_nodes["station_depth_m"] = np.where(
        edges_nodes["u_type"].eq("subway_station"),
        edges_nodes["u_depth_m"].fillna(0.0),
        np.where(edges_nodes["v_type"].eq("subway_station"), edges_nodes["v_depth_m"].fillna(0.0), 0.0),
    ).astype(float)

    ESCALATOR_SPEED_MPMIN = 0.75 * 60
    WALK_SPEED_MPMIN = 5 * 1000 / 60
    TRANSFER_PATH_FACTOR = 1.5

    mask_escal = edges_nodes["type"].isin(["subway_entrance", "subway_exit"]) & edges_nodes["horiz_m"].notna()
    edges_nodes.loc[mask_escal, "length_meter"] = np.hypot(
        edges_nodes.loc[mask_escal, "horiz_m"], edges_nodes.loc[mask_escal, "station_depth_m"]
    )
    edges_nodes.loc[mask_escal, "time_min"] = edges_nodes.loc[mask_escal, "length_meter"] / ESCALATOR_SPEED_MPMIN

    mask_transfer = edges_nodes["type"].eq("subway_transfer") & edges_nodes["horiz_m"].notna()
    edges_nodes.loc[mask_transfer, "length_meter"] = TRANSFER_PATH_FACTOR * edges_nodes.loc[
        mask_transfer, "horiz_m"
    ].round(1)
    edges_nodes.loc[mask_transfer, "time_min"] = (
        edges_nodes.loc[mask_transfer, "length_meter"] / WALK_SPEED_MPMIN
    ).round(1)

    edges_gdf = gpd.GeoDataFrame(
        edges_nodes[["u_ref", "v_ref", "type", "length_meter", "time_min", "geometry", "oneway"]].copy(),
        geometry="geometry",
        crs=to_crs,
    )

    return edges_gdf, nodes_gdf
