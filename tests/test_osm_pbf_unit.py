# pylint: disable=redefined-outer-name, protected-access, unused-argument, too-many-lines
"""Offline OSM source: Overpass tag filters, PbfSource element shapes, Overpass/pbf graph equivalence,
per-city splitting and snapshot timestamps.

Every OSM file here is synthetic and written into ``tmp_path``; no network is used. The Overpass side of
the equivalence tests is a mocked ``_overpass_request`` whose answers are assembled from the same data with
the element sets written out by hand below, independently of the pbf reader's own selection.
"""

import datetime as dt
from pathlib import Path

import pytest

osmium = pytest.importorskip("osmium")

from geopandas.testing import assert_geodataframe_equal  # noqa: E402  pylint: disable=wrong-import-position
from shapely import Polygon, box  # noqa: E402  pylint: disable=wrong-import-position

from iduedu import (  # noqa: E402  pylint: disable=wrong-import-position
    PbfSource,
    config,
    get_4326_boundary,
    get_drive_graph,
    get_public_transport_graph,
    get_walk_graph,
    snapshot_time,
    split_by_polygons,
)
from iduedu.constants.network_enums import Network  # noqa: E402  pylint: disable=wrong-import-position
from iduedu.osm_pbf.filters import (  # noqa: E402  pylint: disable=wrong-import-position
    UnsupportedFilterError,
    parse_way_filter,
)
from iduedu.overpass import downloaders  # noqa: E402  pylint: disable=wrong-import-position

pytestmark = pytest.mark.unit

POLY = box(30.300, 59.930, 30.320, 59.940)
TIMESTAMP = "2026-10-01T20:21:02Z"
PT_TYPES = ["bus", "trolleybus", "tram", "subway"]


# ---------------------------------------------------------------------------
# Synthetic OSM data
# ---------------------------------------------------------------------------

_TYPE_NAME = {"n": "node", "w": "way", "r": "relation"}
_TYPE_ORDER = {"n": 0, "w": 1, "r": 2}


def _opl_escape(text: str) -> str:
    return "".join(c if (c.isascii() and (c.isalnum() or c in "_-.:")) else f"%{ord(c):x}%" for c in str(text))


class OsmData:
    """Nodes, ways and relations held in Python, written to OPL/PBF and rendered as Overpass JSON."""

    def __init__(self):
        self.nodes = {}  # id -> (lon, lat, tags)
        self.ways = {}  # id -> (refs, tags)
        self.rels = {}  # id -> (members [(type, ref, role)], tags)

    def node(self, nid, lon, lat, tags=None):
        self.nodes[nid] = (round(lon, 7), round(lat, 7), dict(tags or {}))
        return nid

    def way(self, wid, refs, tags=None):
        self.ways[wid] = (list(refs), dict(tags or {}))
        return wid

    def rel(self, rid, members, tags=None):
        self.rels[rid] = (list(members), dict(tags or {}))
        return rid

    # ----- files -----
    @staticmethod
    def _tags(tags):
        return ",".join(f"{_opl_escape(k)}={_opl_escape(v)}" for k, v in tags.items())

    def write_opl(self, path: Path, exclude=()) -> Path:
        lines = []
        for nid in sorted(self.nodes):
            if ("n", nid) in exclude:
                continue
            lon, lat, tags = self.nodes[nid]
            lines.append(f"n{nid} v1 x{lon:.7f} y{lat:.7f} T{self._tags(tags)}")
        for wid in sorted(self.ways):
            if ("w", wid) in exclude:
                continue
            refs, tags = self.ways[wid]
            lines.append(f"w{wid} v1 T{self._tags(tags)} N{','.join(f'n{r}' for r in refs)}")
        for rid in sorted(self.rels):
            if ("r", rid) in exclude:
                continue
            members, tags = self.rels[rid]
            mem = ",".join(f"{t}{ref}@{_opl_escape(role)}" for t, ref, role in members)
            lines.append(f"r{rid} v1 T{self._tags(tags)} M{mem}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_pbf(self, path: Path, timestamp: str | None = TIMESTAMP) -> Path:
        opl = self.write_opl(path.with_suffix(".src.opl"))
        header = osmium.io.Header()
        if timestamp:
            header.set("osmosis_replication_timestamp", timestamp)
        with osmium.SimpleWriter(str(path), header=header, overwrite=True) as writer:
            for obj in osmium.FileProcessor(str(opl)):
                writer.add(obj)
        return path

    # ----- Overpass JSON, as the Overpass API prints it -----
    def _coords(self, kind, eid):
        if kind == "n":
            lon, lat, _ = self.nodes[eid]
            return [(lon, lat)]
        if kind == "w":
            return [self.nodes[r][:2] for r in self.ways[eid][0]]
        coords = []
        for t, ref, _ in self.rels[eid][0]:
            if t in ("n", "w"):
                coords.extend(self._coords(t, ref))
        return coords

    @staticmethod
    def _bounds(coords):
        lons, lats = [c[0] for c in coords], [c[1] for c in coords]
        return {"minlat": min(lats), "minlon": min(lons), "maxlat": max(lats), "maxlon": max(lons)}

    def _center(self, coords):
        b = self._bounds(coords)
        return {
            "lat": float(f"{(b['minlat'] + b['maxlat']) / 2:.7f}"),
            "lon": float(f"{(b['minlon'] + b['maxlon']) / 2:.7f}"),
        }

    def tags_of(self, kind, eid):
        return {"n": self.nodes, "w": self.ways, "r": self.rels}[kind][eid][-1]

    @staticmethod
    def _tagged(element, tags):
        if tags:
            element["tags"] = dict(tags)
        return element

    def ov_way_geom(self, wid):
        refs, tags = self.ways[wid]
        coords = self._coords("w", wid)
        element = {
            "type": "way",
            "id": wid,
            "bounds": self._bounds(coords),
            "nodes": refs,
            "geometry": [{"lat": lat, "lon": lon} for lon, lat in coords],
        }
        return self._tagged(element, tags)

    def ov_tags(self, kind, eid):
        return self._tagged({"type": _TYPE_NAME[kind], "id": eid}, self.tags_of(kind, eid))

    def ov_rel_geom(self, rid):
        members, tags = self.rels[rid]
        out = []
        for t, ref, role in members:
            m = {"type": _TYPE_NAME[t], "ref": ref, "role": role}
            if t == "n":
                m["lat"], m["lon"] = self.nodes[ref][1], self.nodes[ref][0]
            elif t == "w":
                m["geometry"] = [{"lat": lat, "lon": lon} for lon, lat in self._coords("w", ref)]
            out.append(m)
        element = {"type": "relation", "id": rid, "bounds": self._bounds(self._coords("r", rid)), "members": out}
        return self._tagged(element, tags)

    def ov_rel_body(self, rid):
        members, tags = self.rels[rid]
        element = {
            "type": "relation",
            "id": rid,
            "members": [{"type": _TYPE_NAME[t], "ref": ref, "role": role} for t, ref, role in members],
        }
        return self._tagged(element, tags)

    def ov_center(self, kind, eid):
        element = {"type": _TYPE_NAME[kind], "id": eid}
        if kind == "n":
            element["lat"], element["lon"] = self.nodes[eid][1], self.nodes[eid][0]
        elif kind == "w":
            element["center"] = self._center(self._coords("w", eid))
            element["nodes"] = self.ways[eid][0]
        else:
            element["center"] = self._center(self._coords("r", eid))
            element["members"] = self.ov_rel_body(eid)["members"]
        return self._tagged(element, self.tags_of(kind, eid))

    def ov_routes(self, ground, stop_area_like, groups):
        """The answer to ``get_routes_by_poly``'s query, statement by statement."""
        out = []
        ways = sorted({ref for r in ground for t, ref, _ in self.rels[r][0] if t == "w"})
        out += [self.ov_tags("w", w) for w in ways]
        out += [self.ov_rel_geom(r) for r in ground]
        out += [self.ov_rel_geom(r) for r in stop_area_like]
        out += [self.ov_rel_body(g) for g in groups]
        members = sorted(
            {(t, ref) for r in stop_area_like for t, ref, _ in self.rels[r][0]},
            key=lambda m: (_TYPE_ORDER[m[0]], m[1]),
        )
        out += [self.ov_tags(t, ref) for t, ref in members if self.tags_of(t, ref).get("public_transport") == "station"]
        out.append({"type": "section", "id": 1, "tags": {"name": downloaders.STOP_AREA_MEMBERS_SECTION}})
        out += [self.ov_center(t, ref) for t, ref in members]
        return out


COLS = [30.302, 30.306, 30.310, 30.314, 30.318]
ROWS = [59.932, 59.934, 59.936, 59.938]


def _grid_node(row, col):
    return 10 * (row + 1) + col  # 10..14, 20..24, 30..34, 40..44


def build_city() -> OsmData:
    """A small city in Saint Petersburg's coordinates with streets, ground routes and a subway line."""
    d = OsmData()
    for r, lat in enumerate(ROWS):
        for c, lon in enumerate(COLS):
            d.node(_grid_node(r, c), lon, lat)

    # Rows: residential / residential sidewalk=separate / primary oneway / tertiary foot=no.
    d.way(101, [_grid_node(0, c) for c in range(5)], {"highway": "residential", "name": "Row 0"})
    d.way(102, [_grid_node(1, c) for c in range(5)], {"highway": "residential", "sidewalk": "separate"})
    d.way(103, [_grid_node(2, c) for c in range(5)], {"highway": "primary", "oneway": "yes", "maxspeed": "60"})
    d.way(104, [_grid_node(3, c) for c in range(5)], {"highway": "tertiary", "foot": "no", "name": "Row 3"})
    # Columns.
    d.way(111, [_grid_node(r, 0) for r in range(4)], {"highway": "footway"})
    d.way(112, [_grid_node(r, 1) for r in range(4)], {"highway": "service", "service": "parking_aisle"})
    d.way(113, [_grid_node(r, 2) for r in range(4)], {"highway": "residential", "access": "private"})
    d.way(114, [_grid_node(r, 3) for r in range(4)], {"highway": "steps"})
    d.way(115, [_grid_node(r, 4) for r in range(4)], {"highway": "cycleway"})
    d.way(116, [_grid_node(0, 4), _grid_node(1, 3)], {"highway": "motorway_link", "oneway": "yes"})
    d.way(117, [_grid_node(1, 0), _grid_node(2, 1), _grid_node(1, 1)], {"highway": "pedestrian", "area": "yes"})
    d.way(118, [_grid_node(2, 0), _grid_node(3, 1)], {"highway": "pedestrian", "area": "no"})
    d.way(120, [_grid_node(0, 2), _grid_node(1, 2)], {"highway": "unclassified", "motor_vehicle": "no"})
    # A street crossing a corner of the polygon with no node inside it.
    d.node(51, 30.2995, 59.9380)
    d.node(52, 30.3020, 59.9405)
    d.way(121, [51, 52], {"highway": "residential", "name": "Corner"})
    # A street outside the polygon.
    d.node(61, 30.330, 59.950)
    d.node(62, 30.335, 59.950)
    d.node(63, 30.340, 59.950)
    d.way(119, [61, 62, 63], {"highway": "residential", "name": "Far"})
    # Not a street at all.
    d.node(71, 30.3045, 59.9325)
    d.node(72, 30.3047, 59.9325)
    d.node(73, 30.3047, 59.9327)
    d.way(180, [71, 72, 73, 71], {"building": "yes"})

    # Bus 1 along row 0: stops on the street, platforms 11 m north of it.
    members = []
    for i, c in enumerate((0, 2, 4)):
        stop = _grid_node(0, c)
        d.nodes[stop][2].update({"public_transport": "stop_position", "bus": "yes"})
        platform = d.node(200 + i, COLS[c], ROWS[0] + 0.0001, {"highway": "bus_stop", "public_transport": "platform"})
        members += [("n", stop, "stop"), ("n", platform, "platform")]
    d.rel(1001, members + [("w", 101, "")], {"type": "route", "route": "bus", "ref": "1", "name": "Bus 1"})

    # Bus 2 along row 2 and on, out of the polygon, to a terminus far east.
    d.node(81, 30.325, 59.936)
    d.node(82, 30.330, 59.936)
    d.node(83, 30.340, 59.936, {"public_transport": "stop_position", "bus": "yes"})
    d.way(130, [_grid_node(2, 4), 81, 82], {"highway": "primary", "name": "East"})
    d.way(131, [82, 83], {"highway": "primary", "name": "Far East"})
    d.node(210, 30.340, 59.9361, {"highway": "bus_stop", "public_transport": "platform"})
    d.node(211, COLS[0], ROWS[2] - 0.0001, {"highway": "bus_stop", "public_transport": "platform"})
    d.node(212, COLS[2], ROWS[2] - 0.0001, {"highway": "bus_stop", "public_transport": "platform"})
    d.rel(
        1002,
        [
            ("n", _grid_node(2, 0), "stop"),
            ("n", 211, "platform"),
            ("n", _grid_node(2, 2), "stop"),
            ("n", 212, "platform"),
            ("n", 83, "stop_exit_only"),
            ("n", 210, "platform_exit_only"),
            ("w", 103, ""),
            ("w", 130, ""),
            ("w", 131, ""),
        ],
        {"type": "route", "route": "bus", "ref": "2"},
    )

    # Tram tagged light_rail on its own tracks.
    d.node(301, 30.302, 59.937, {"railway": "tram_stop", "public_transport": "stop_position"})
    d.node(302, 30.310, 59.937)
    d.node(303, 30.318, 59.937, {"railway": "tram_stop", "public_transport": "stop_position"})
    d.way(140, [301, 302, 303], {"railway": "tram"})
    d.node(311, 30.302, 59.9371, {"public_transport": "platform", "railway": "platform"})
    d.node(313, 30.318, 59.9371, {"public_transport": "platform", "railway": "platform"})
    d.rel(
        1003,
        [("n", 301, "stop"), ("n", 311, "platform"), ("n", 303, "stop"), ("n", 313, "platform"), ("w", 140, "")],
        {"type": "route", "route": "light_rail", "ref": "T"},
    )

    # A bus route entirely outside, and a trolleybus selected only through a stop node inside.
    d.node(64, 30.335, 59.9501, {"highway": "bus_stop", "public_transport": "platform"})
    d.rel(1004, [("n", 62, "stop"), ("n", 64, "platform"), ("w", 119, "")], {"type": "route", "route": "bus"})
    d.rel(
        1005,
        [("n", _grid_node(3, 4), "stop"), ("w", 119, "")],
        {"type": "route", "route": "trolleybus", "ref": "Tb"},
    )
    # Outside too, and read after 1005 has pulled way 119 into the city file: must still be left out.
    d.rel(1006, [("n", 63, "stop"), ("w", 119, "")], {"type": "route", "route": "bus", "ref": "6"})

    # Subway: two stations with stop areas, entrances and an interchange group.
    d.node(401, 30.303, 59.933)
    d.node(402, 30.309, 59.933, {"railway": "stop", "public_transport": "stop_position", "subway": "yes"})
    d.node(403, 30.312, 59.933)
    d.node(404, 30.315, 59.933, {"railway": "stop", "public_transport": "stop_position", "subway": "yes"})
    d.node(405, 30.318, 59.933)
    d.way(150, [401, 402, 403], {"railway": "subway"})
    d.way(151, [403, 404, 405], {"railway": "subway"})
    d.node(412, 30.309, 59.93305, {"railway": "platform", "public_transport": "platform"})
    d.node(414, 30.315, 59.93305, {"railway": "platform", "public_transport": "platform"})
    d.node(
        422,
        30.3090,
        59.9332,
        {"public_transport": "station", "railway": "station", "station": "subway", "name": "Alpha", "depth": "30"},
    )
    d.node(
        424,
        30.3150,
        59.9332,
        {"public_transport": "station", "railway": "station", "station": "subway", "name": "Beta"},
    )
    d.node(432, 30.3088, 59.9338, {"railway": "subway_entrance"})
    d.node(434, 30.3152, 59.9338, {"railway": "subway_entrance", "entrance": "main"})
    d.rel(
        1010,
        [("n", 402, "stop"), ("n", 412, "platform"), ("n", 404, "stop"), ("n", 414, "platform")]
        + [("w", 150, ""), ("w", 151, "")],
        {"type": "route", "route": "subway", "ref": "M1", "name": "Line 1"},
    )
    # Station and entrance of Alpha carry no role: their roles come from the member tags.
    d.rel(
        1011,
        [("n", 402, "stop"), ("n", 412, "platform"), ("n", 422, ""), ("n", 432, "")],
        {"type": "public_transport", "public_transport": "stop_area", "name": "Alpha"},
    )
    d.rel(
        1012,
        [("n", 404, "stop"), ("n", 414, "platform"), ("n", 424, "station"), ("n", 434, "entrance")],
        {"type": "public_transport", "public_transport": "stop_area", "name": "Beta"},
    )
    d.rel(
        1013,
        [("r", 1011, ""), ("r", 1012, "")],
        {"type": "public_transport", "public_transport": "stop_area_group", "name": "Alpha-Beta"},
    )

    # The city boundary: one closed outer way.
    corners = [(30.299, 59.929), (30.321, 59.929), (30.321, 59.941), (30.299, 59.941)]
    ring = [d.node(501 + i, lon, lat) for i, (lon, lat) in enumerate(corners)]
    d.way(160, ring + [ring[0]])
    d.rel(1020, [("w", 160, "outer")], {"type": "boundary", "boundary": "administrative", "name": "City"})
    return d


# What Overpass selects from POLY, written out by hand.
WALK_WAYS = [101, 103, 111, 112, 114, 118, 120, 121, 130]
DRIVE_WAYS = [101, 102, 103, 104, 116, 121, 130]
GROUND_ROUTES = [1001, 1002, 1003, 1005]
STOP_AREA_LIKE = [1010, 1011, 1012]  # rel(bn.route_nodes): the subway route itself and both stop areas
GROUPS = [1013]


@pytest.fixture(scope="module")
def city():
    return build_city()


@pytest.fixture(scope="module")
def city_pbf(city, tmp_path_factory):
    return city.write_pbf(tmp_path_factory.mktemp("osm") / "city.osm.pbf")


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch):
    monkeypatch.setattr(config, "osm_source", "overpass")
    monkeypatch.setattr(config, "pbf_path", None)
    monkeypatch.setattr(config, "overpass_date", None)
    monkeypatch.setattr(config, "enable_tqdm_bar", False)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def mocked_overpass(city, monkeypatch):
    """Overpass answering from ``city``, with the cache bypassed; returns the list of queries sent."""
    queries = []
    answers = {
        "walk": [city.ov_way_geom(w) for w in WALK_WAYS],
        "drive": [city.ov_way_geom(w) for w in DRIVE_WAYS],
        "routes": city.ov_routes(GROUND_ROUTES, STOP_AREA_LIKE, GROUPS),
        "boundary": [city.ov_rel_geom(1020)],
    }

    def _fake_request(method, overpass_url, params=None, data=None, timeout=None, **kwargs):
        query = (data or params)["data"]
        queries.append(query)
        if "rel(poly:" in query:
            return _FakeResponse({"elements": answers["routes"]})
        if Network.WALK.filter in query:
            return _FakeResponse({"elements": answers["walk"]})
        if Network.DRIVE.filter in query:
            return _FakeResponse({"elements": answers["drive"]})
        if "relation(1020)" in query:
            return _FakeResponse({"elements": answers["boundary"]})
        raise AssertionError(f"unexpected Overpass query: {query}")

    monkeypatch.setattr(downloaders, "_overpass_request", _fake_request)
    monkeypatch.setattr(downloaders, "cache_load", lambda prefix, key_src: None)
    monkeypatch.setattr(downloaders, "cache_save_async", lambda prefix, key_src, obj: None)
    return queries


@pytest.fixture
def no_overpass(monkeypatch):
    """Fail on any use of Overpass or of its cache."""

    def _fail(*args, **kwargs):
        raise AssertionError("Overpass or its cache must not be used in pbf mode")

    monkeypatch.setattr(downloaders, "_overpass_request", _fail)
    monkeypatch.setattr(downloaders, "cache_load", _fail)
    monkeypatch.setattr(downloaders, "cache_save_async", _fail)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, tags, expected",
    [
        ('["highway"]', {"highway": "x"}, True),
        ('["highway"]', {}, False),
        ('[!"highway"]', {}, True),
        ('[!"highway"]', {"highway": "x"}, False),
        ('["highway"="primary"]', {"highway": "primary"}, True),
        ('["highway"="primary"]', {"highway": "primary_link"}, False),
        ('["highway"="primary"]', {}, False),
        ('["access"!="private"]', {}, True),
        ('["access"!="private"]', {"access": "yes"}, True),
        ('["access"!="private"]', {"access": "private"}, False),
        ('["highway"~"primary|secondary"]', {"highway": "primary_link"}, True),  # unanchored
        ('["highway"~"^(primary|secondary)$"]', {"highway": "primary_link"}, False),
        ('["highway"~"primary"]', {}, False),
        ('["highway"~"primary"]', {"highway": "Primary"}, False),  # case-sensitive
        ('["highway"~"primary",i]', {"highway": "Primary"}, True),
        ('["area"!~"yes"]', {}, True),  # absent key passes a negated regex
        ('["area"!~"yes"]', {"area": "no"}, True),
        ('["area"!~"yes"]', {"area": "yes"}, False),
        ('["highway"!~"motor"]', {"highway": "motorway_link"}, False),
        ("[highway=primary]", {"highway": "primary"}, True),  # unquoted
        ("['name'='O\\'Neil']", {"name": "O'Neil"}, True),  # single quotes and escapes
        ('["ref"~"^\\d+$"]', {"ref": "42"}, True),  # regex escapes survive
        ('["sidewalk:both"!~"separate"]', {"sidewalk:both": "separate"}, False),
        ('  ["a"] ["b"="c"]  ', {"a": "1", "b": "c"}, True),
        ("", {}, True),
    ],
)
def test_way_filter_semantics(text, tags, expected):
    assert parse_way_filter(text)(tags) is expected


@pytest.mark.parametrize(
    "text",
    [
        '[~"high"~"way"]',  # key regex
        '["highway"](if: t["a"] > 1)',
        '["highway"',
        '["highway"="primary",i]',
        '["highway"?"x"]',
        '["highway"~"("]',  # invalid regex
        '(poly:"1 2 3 4")',
    ],
)
def test_way_filter_rejects_unsupported_syntax(text):
    with pytest.raises(UnsupportedFilterError):
        parse_way_filter(text)


def test_way_filter_required_keys_and_network_presets():
    walk = parse_way_filter(Network.WALK.filter)
    drive = parse_way_filter(Network.DRIVE.filter)
    assert walk.required_keys == ("highway",)
    assert walk({"highway": "footway"}) and not drive({"highway": "footway"})
    assert drive({"highway": "primary", "oneway": "yes"}) and walk({"highway": "primary"})
    assert not walk({"highway": "residential", "sidewalk": "separate"})
    assert not drive({"highway": "unclassified", "motor_vehicle": "no"})
    assert not walk({"highway": "motorway"}) and drive({"highway": "motorway"})
    assert parse_way_filter(Network.DRIVE_SERVICE.filter)({"highway": "service"})


def test_way_filter_selection_matches_the_hand_written_sets(city):
    walk = parse_way_filter(Network.WALK.filter)
    drive = parse_way_filter(Network.DRIVE.filter)
    inside = {101, 102, 103, 104, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121, 130, 140, 150, 151, 160, 180}
    assert sorted(w for w in inside if walk(city.ways[w][1])) == WALK_WAYS
    assert sorted(w for w in inside if drive(city.ways[w][1])) == DRIVE_WAYS


# ---------------------------------------------------------------------------
# Element shapes
# ---------------------------------------------------------------------------


def test_ways_by_filter_returns_overpass_out_geom_ways(city, city_pbf):
    src = PbfSource(city_pbf)
    assert src.ways_by_filter(POLY, Network.WALK.filter) == [city.ov_way_geom(w) for w in WALK_WAYS]
    assert src.ways_by_filter(POLY, Network.DRIVE.filter) == [city.ov_way_geom(w) for w in DRIVE_WAYS]


def test_way_shape_written_out_literally(city_pbf):
    (way,) = PbfSource(city_pbf).ways_by_filter(POLY, '["name"="Corner"]')
    assert way == {
        "type": "way",
        "id": 121,
        "bounds": {"minlat": 59.938, "minlon": 30.2995, "maxlat": 59.9405, "maxlon": 30.302},
        "nodes": [51, 52],
        "geometry": [{"lat": 59.938, "lon": 30.2995}, {"lat": 59.9405, "lon": 30.302}],
        "tags": {"highway": "residential", "name": "Corner"},
    }


def test_poly_semantics_for_ways(tmp_path):
    d = OsmData()
    d.node(1, 0.5, 0.5)  # inside
    d.node(2, 2.0, 0.5)
    d.node(3, -1.0, 0.5)
    d.node(4, 2.0, 2.0)
    d.node(5, 3.0, 3.0)
    d.node(6, 1.0, 0.5)  # on the boundary
    d.way(10, [1, 2], {"highway": "a"})  # node inside
    d.way(11, [3, 2], {"highway": "b"})  # no node inside, segment crosses
    d.way(12, [4, 5], {"highway": "c"})  # outside
    d.way(13, [6], {"highway": "d"})  # a single node on the boundary
    d.way(14, [2, 6, 4], {"highway": "e"})  # touches the boundary
    path = d.write_opl(tmp_path / "poly.opl")
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], holes=[[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]])
    ids = [w["id"] for w in PbfSource(path).ways_by_filter(square, '["highway"]')]
    # Holes are ignored, as the Overpass downloaders send only the exterior ring.
    assert ids == [10, 11, 13, 14]


def test_routes_by_poly_reproduces_the_overpass_statements(city, city_pbf):
    got = PbfSource(city_pbf).routes_by_poly(POLY, PT_TYPES)
    assert got == city.ov_routes(GROUND_ROUTES, STOP_AREA_LIKE, GROUPS)


def test_routes_by_poly_ground_only_has_no_member_section(city, city_pbf):
    got = PbfSource(city_pbf).routes_by_poly(POLY, ["bus"])
    expected_routes = [1001, 1002]
    ways = sorted({ref for r in expected_routes for t, ref, _ in city.rels[r][0] if t == "w"})
    assert got == [city.ov_tags("w", w) for w in ways] + [city.ov_rel_geom(r) for r in expected_routes]
    assert not any(e["type"] == "section" for e in got)


def test_route_leaving_the_polygon_keeps_its_full_geometry(city_pbf):
    route = next(e for e in PbfSource(city_pbf).routes_by_poly(POLY, ["bus"]) if e.get("id") == 1002)
    far = next(m for m in route["members"] if m["ref"] == 131)
    assert far["geometry"] == [{"lat": 59.936, "lon": 30.33}, {"lat": 59.936, "lon": 30.34}]
    terminus = next(m for m in route["members"] if m["ref"] == 83)
    assert (terminus["lat"], terminus["lon"], terminus["role"]) == (59.936, 30.34, "stop_exit_only")
    assert [m["ref"] for m in route["members"]] == [20 + 10, 211, 32, 212, 83, 210, 103, 130, 131]


def test_routes_by_poly_alias_and_empty_types(city_pbf):
    src = PbfSource(city_pbf)
    tram = [e for e in src.routes_by_poly(POLY, ["tram"]) if e["type"] == "relation"]
    assert [e["id"] for e in tram] == [1003]
    assert src.routes_by_poly(POLY, []) == []


def test_member_tags_match_fetch_member_tags_shapes(city, city_pbf):
    members = [{"type": "node", "ref": 422}, {"type": "way", "ref": 150}, {"type": "relation", "ref": 1011}]
    got = PbfSource(city_pbf).member_tags(members)
    expected = downloaders.member_payloads(
        [city.ov_center("n", 422), city.ov_center("w", 150), city.ov_center("r", 1011)]
    )
    assert got == expected
    assert got[("node", 422)]["__latlon__"] == (30.309, 59.9332)
    assert "__center__" in got[("way", 150)] and "__center__" in got[("relation", 1011)]


def test_boundary_returns_relation_out_geom(city, city_pbf):
    src = PbfSource(city_pbf)
    assert src.boundary(1020) == [city.ov_rel_geom(1020)]
    assert src.boundary(999999) == []


def test_pbf_source_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        PbfSource(tmp_path / "missing.osm.pbf")


# ---------------------------------------------------------------------------
# Equivalence: the same city built from Overpass JSON and from the pbf file
# ---------------------------------------------------------------------------


def _assert_same_graph(a, b):
    assert (a.is_directed, a.is_multigraph, a.type) == (b.is_directed, b.is_multigraph, b.type)
    assert a.crs == b.crs
    assert_geodataframe_equal(a.nodes_gdf, b.nodes_gdf)
    assert_geodataframe_equal(a.edges_gdf, b.edges_gdf)


def _build_all():
    return {
        "walk": get_walk_graph(territory=POLY),
        "drive": get_drive_graph(territory=POLY),
        "pt": get_public_transport_graph(territory=POLY, transport_types=PT_TYPES),
        "boundary": get_4326_boundary(osm_id=1020),
    }


def test_pbf_and_overpass_build_identical_graphs(city_pbf, mocked_overpass, monkeypatch):
    from_overpass = _build_all()
    assert len(mocked_overpass) == 4  # walk, drive, routes, boundary: no member fallback request

    def _fail(*args, **kwargs):
        raise AssertionError("Overpass or its cache must not be used in pbf mode")

    monkeypatch.setattr(downloaders, "_overpass_request", _fail)
    monkeypatch.setattr(downloaders, "cache_load", _fail)
    with config.use_pbf(city_pbf):
        from_pbf = _build_all()
    assert config.osm_source == "overpass"

    for kind in ("walk", "drive", "pt"):
        _assert_same_graph(from_overpass[kind], from_pbf[kind])
    assert from_pbf["boundary"].equals(from_overpass["boundary"])

    pt_types = set(from_pbf["pt"].edges_gdf["type"])
    assert {"bus", "tram", "subway", "boarding", "subway_transfer", "subway_entrance", "subway_exit"} <= pt_types
    assert len(from_pbf["walk"].edges_gdf) > 0 and len(from_pbf["drive"].edges_gdf) > 0


def test_pbf_mode_ignores_overpass_date(city_pbf, no_overpass, monkeypatch):
    monkeypatch.setattr(config, "overpass_date", "2020-01-01T00:00:00Z")
    with config.use_pbf(city_pbf):
        graph = get_public_transport_graph(territory=POLY, transport_types=["subway"])
    assert "subway_transfer" in set(graph.edges_gdf["type"])


def test_member_fallback_reads_the_file(city, city_pbf, no_overpass):
    with config.use_pbf(city_pbf):
        got = downloaders.fetch_member_tags(iter([{"type": "node", "ref": 432}, {"type": "relation", "ref": 1012}]))
    assert got == downloaders.member_payloads([city.ov_center("n", 432), city.ov_center("r", 1012)])


# ---------------------------------------------------------------------------
# split_by_polygons
# ---------------------------------------------------------------------------


def _ids(path):
    out = {"n": set(), "w": set(), "r": set()}
    for obj in osmium.FileProcessor(str(path)):
        out[obj.type_str()].add(obj.id)
    return out


@pytest.fixture(scope="module")
def region_pbf(city, tmp_path_factory):
    d = build_city()
    # Things far from the city, in the same regional file.
    d.node(901, 31.000, 60.500)
    d.node(902, 31.010, 60.500, {"public_transport": "stop_position", "bus": "yes"})
    d.node(903, 31.010, 60.5001, {"highway": "bus_stop"})
    d.way(910, [901, 902], {"highway": "residential"})
    d.rel(1090, [("n", 902, "stop"), ("n", 903, "platform"), ("w", 910, "")], {"type": "route", "route": "bus"})
    return d.write_pbf(tmp_path_factory.mktemp("region") / "region.osm.pbf")


@pytest.fixture(scope="module")
def split_city(region_pbf, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("cities")
    paths = split_by_polygons(
        region_pbf,
        {"city": POLY},
        out_dir,
        location_storage="sparse_file_array",
        keep_relations={"city": [1020]},
    )
    return paths["city"]


def test_split_writes_compact_complete_city_files(split_city, region_pbf):
    assert split_city.name == "city.osm.pbf"
    assert not list(split_city.parent.glob("*.idx"))  # the temporary node index is removed
    ids = _ids(split_city)
    # Bus 2 leaves the polygon: the relation, its far way, the way's nodes and its far stops are all there.
    assert {1002, 1001, 1003, 1005, 1010, 1011, 1012, 1013, 1020} <= ids["r"]
    assert {131, 130, 160} <= ids["w"]
    assert {82, 83, 210} <= ids["n"]
    # Way 119 is outside but is a member of trolleybus 1005 (which has a stop inside); bus 1004 is outside.
    assert 119 in ids["w"] and 1004 not in ids["r"] and 1006 not in ids["r"]
    # Nothing from far away, no buildings.
    assert 1090 not in ids["r"] and 910 not in ids["w"] and 901 not in ids["n"]
    assert 180 not in ids["w"] and 71 not in ids["n"]
    # Referential completeness: every way node and every node/way member is in the file.
    nodes, ways = ids["n"], ids["w"]
    for obj in osmium.FileProcessor(str(split_city)):
        if obj.is_way():
            assert all(n.ref in nodes for n in obj.nodes)
        elif obj.is_relation() and obj.id != 1013:
            for m in obj.members:
                assert (m.type == "n" and m.ref in nodes) or (m.type == "w" and m.ref in ways) or m.type == "r"


def test_split_city_file_answers_like_the_region(split_city, region_pbf, no_overpass):
    with config.use_pbf(region_pbf):
        from_region = _build_all()
    with config.use_pbf(split_city):
        from_city = _build_all()
    for kind in ("walk", "drive", "pt"):
        _assert_same_graph(from_region[kind], from_city[kind])
    assert from_city["boundary"].equals(from_region["boundary"])
    region, small = PbfSource(region_pbf), PbfSource(split_city)
    assert small.routes_by_poly(POLY, PT_TYPES) == region.routes_by_poly(POLY, PT_TYPES)
    assert small.ways_by_filter(POLY, Network.WALK.filter) == region.ways_by_filter(POLY, Network.WALK.filter)


def test_split_keeps_the_snapshot_time(split_city, region_pbf):
    assert snapshot_time(split_city) == snapshot_time(region_pbf) == dt.datetime(2026, 10, 1, 20, 21, 2, tzinfo=dt.UTC)


def test_split_validates_names_and_existing_files(region_pbf, tmp_path):
    with pytest.raises(ValueError):
        split_by_polygons(region_pbf, {"a/b": POLY}, tmp_path)
    with pytest.raises(ValueError):
        split_by_polygons(region_pbf, {"a": POLY}, tmp_path, keep_relations={"b": [1]})
    (tmp_path / "a.osm.pbf").write_bytes(b"")
    with pytest.raises(FileExistsError):
        split_by_polygons(region_pbf, {"a": POLY}, tmp_path)
    assert split_by_polygons(region_pbf, {}, tmp_path) == {}


def test_split_several_cities_in_one_pass(region_pbf, tmp_path):
    west = box(30.300, 59.930, 30.307, 59.940)
    far = box(30.990, 60.490, 31.020, 60.510)
    paths = split_by_polygons(region_pbf, {"west": west, "far": far}, tmp_path)
    west_ids, far_ids = _ids(paths["west"]), _ids(paths["far"])
    assert 1090 in far_ids["r"] and 910 in far_ids["w"] and 1001 not in far_ids["r"]
    assert 1090 not in west_ids["r"] and 1001 in west_ids["r"]
    assert 113 not in west_ids["w"]  # column 2 lies east of the western polygon


# ---------------------------------------------------------------------------
# snapshot_time
# ---------------------------------------------------------------------------


def test_snapshot_time_from_header(tmp_path):
    d = OsmData()
    d.node(1, 0.0, 0.0)
    with_ts = d.write_pbf(tmp_path / "a.osm.pbf", timestamp="2026-09-30T23:59:59Z")
    without = d.write_pbf(tmp_path / "b.osm.pbf", timestamp=None)
    assert snapshot_time(with_ts) == dt.datetime(2026, 9, 30, 23, 59, 59, tzinfo=dt.UTC)
    assert PbfSource(with_ts).snapshot_time() == snapshot_time(with_ts)
    assert snapshot_time(without) is None
    assert snapshot_time(d.write_opl(tmp_path / "c.opl")) is None
