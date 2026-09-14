import math

import pandas as pd
import pytest

from iduedu.overpass.parsers import (
    _link_unconnected,
    infer_role_from_tags,
    overpass_ground_transport2edgenode,
    overpass_routes_to_df,
    overpass_subway2edgenode,
    parse_maxspeed_to_m_per_min,
    parse_overpass_subway_data,
)

pytestmark = pytest.mark.unit

LOCAL_CRS = "EPSG:32636"


# ---------------------------------------------------------------------------
# parse_maxspeed_to_m_per_min
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected_kmh",
    [
        (50, 50.0),
        ("50", 50.0),
        ("50 km/h", 50.0),
        ("50kmh", 50.0),
        ("60kph", 60.0),
        ("30 mph", 30 * 1.60934),
    ],
)
def test_parse_maxspeed_known_formats(raw, expected_kmh):
    result = parse_maxspeed_to_m_per_min(raw)
    assert result == pytest.approx(expected_kmh * 1000.0 / 60.0)


@pytest.mark.parametrize("raw", [None, "", "   ", "walk", "fast", "50 knots"])
def test_parse_maxspeed_unparseable_returns_none(raw):
    assert parse_maxspeed_to_m_per_min(raw) is None


# ---------------------------------------------------------------------------
# infer_role_from_tags
# ---------------------------------------------------------------------------


def test_infer_role_empty_tags():
    assert infer_role_from_tags({}) == ""


@pytest.mark.parametrize(
    "tags, expected",
    [
        ({"public_transport": "station"}, "station"),
        ({"railway": "station"}, "station"),
        ({"station": "subway"}, "station"),
        ({"public_transport": "platform"}, "platform"),
        ({"railway": "platform"}, "platform"),
        ({"public_transport": "stop_position"}, "stop"),
        ({"railway": "halt"}, "stop"),
        ({"railway": "subway_entrance"}, "entrance"),
        ({"entrance": "yes"}, "entrance"),
        ({"entrance": "entry"}, "entry_only"),
        ({"entrance": "exit"}, "exit_only"),
        ({"entrance": "yes", "entry": "yes"}, "entry_only"),
        ({"entrance": "yes", "exit": "yes"}, "exit_only"),
        ({"amenity": "cafe"}, ""),
    ],
)
def test_infer_role_from_tags(tags, expected):
    assert infer_role_from_tags(tags) == expected


# ---------------------------------------------------------------------------
# overpass_routes_to_df
# ---------------------------------------------------------------------------


def test_overpass_routes_to_df_empty_input_has_bool_flag_columns():
    df = overpass_routes_to_df([], enable_subway_details=True)
    assert df.empty
    for col in ("is_stop_area", "is_stop_area_group", "is_station"):
        assert col in df.columns


def test_overpass_routes_to_df_flags_way_data_and_speed():
    routes = [
        {"type": "relation", "id": 1, "tags": {"route": "bus"}},
        {"type": "way", "id": 2, "tags": {"highway": "primary", "maxspeed": "60"}},
    ]
    df = overpass_routes_to_df(routes, enable_subway_details=False)

    way_row = df[df["id"] == 2].iloc[0]
    assert bool(way_row["is_way_data"]) is True
    assert way_row["way_speed_m_per_min"] == pytest.approx(60 * 1000.0 / 60.0)

    rel_row = df[df["id"] == 1].iloc[0]
    assert rel_row["transport_type"] == "bus"


def test_overpass_routes_to_df_marks_subway_station_details():
    routes = [
        {"type": "relation", "id": 10, "tags": {"public_transport": "stop_area"}},
        {"type": "relation", "id": 11, "tags": {"public_transport": "station"}},
    ]
    df = overpass_routes_to_df(routes, enable_subway_details=True)

    assert bool(df[df["id"] == 10].iloc[0]["is_stop_area"]) is True
    assert bool(df[df["id"] == 11].iloc[0]["is_station"]) is True
    # station-context relations are re-labelled as subway
    assert set(df["transport_type"]) == {"subway"}


# ---------------------------------------------------------------------------
# parse_overpass_subway_data / overpass_subway2edgenode: surface access typing
#
# Node type carries the surface-access contract consumed by the intermodal builder:
# "platform" means the object is reachable from the street and must be projected onto the
# walk graph, "subway_platform" means it is reachable only through a station or an entrance.
# ---------------------------------------------------------------------------


def _member(ref, role, lat, lon):
    return {"type": "node", "ref": ref, "role": role, "lat": lat, "lon": lon}


def _stop_area_row(area_id, members, **flags):
    row = {
        "id": area_id,
        "tags": {},
        "members": members,
        "is_stop_area": True,
        "is_stop_area_group": False,
        "is_station": False,
    }
    row.update(flags)
    return row


def _parse_stop_area(members):
    stop_areas = pd.DataFrame([_stop_area_row(10, members)])
    empty = pd.DataFrame(columns=["id", "tags", "members"])
    edges, nodes = parse_overpass_subway_data(stop_areas, empty, empty, LOCAL_CRS)
    return edges, dict(zip(nodes["ref_id"], nodes["type"]))


def _stop_area_members(*, station=True, entrance=True):
    # Refs: 1 station, 2 platform, 3 stop, 4 entrance.
    members = []
    if station:
        members.append(_member(1, "station", 59.9000, 30.3000))
    members.append(_member(2, "platform", 59.9001, 30.3001))
    members.append(_member(3, "stop", 59.9001, 30.30011))
    if entrance:
        members.append(_member(4, "entrance", 59.9002, 30.3002))
    return members


def test_parse_subway_stop_area_with_entrance_keeps_platform_underground():
    edges, types = _parse_stop_area(_stop_area_members())

    assert types[1] == "subway_station"
    assert types[2] == "subway_platform"
    assert types[4] == "subway_entry_exit"
    assert {"subway_entrance", "subway_exit", "subway_station", "boarding"} <= set(edges["type"])


def test_parse_subway_stop_area_without_entrance_marks_station_as_surface_platform():
    _, types = _parse_stop_area(_stop_area_members(entrance=False))

    # Without entrances the station itself becomes the surface access point.
    assert types[1] == "platform"
    assert types[2] == "subway_platform"


def test_parse_subway_stop_area_without_station_and_entrance_marks_platform_as_surface():
    _, types = _parse_stop_area(_stop_area_members(station=False, entrance=False))

    assert types[2] == "platform"


def test_parse_subway_stop_area_without_station_links_entrances_to_platform():
    edges, types = _parse_stop_area(_stop_area_members(station=False))

    assert types[2] == "subway_platform"
    entrance_edges = edges[edges["type"] == "subway_entrance"]
    assert set(zip(entrance_edges["u_ref"], entrance_edges["v_ref"])) == {(4, 2)}


def _route_members(*, platforms=True):
    # Two stops along one way, each with a platform 8 m aside unless platforms are disabled.
    members = [_member(301, "stop", 59.9100, 30.3102), _member(303, "stop", 59.9100, 30.3140)]
    if platforms:
        members += [_member(302, "platform", 59.91007, 30.3102), _member(304, "platform", 59.91007, 30.3140)]
    members.append(
        {
            "type": "way",
            "ref": 305,
            "role": "",
            "geometry": [{"lat": 59.9100, "lon": 30.3095}, {"lat": 59.9100, "lon": 30.3145}],
        }
    )
    return members


def _route_row(route_id, members):
    return {
        "id": route_id,
        "tags": {"ref": "M1"},
        "members": members,
        "is_stop_area": False,
        "is_stop_area_group": False,
        "is_station": False,
    }


def test_overpass_subway2edgenode_marks_route_platforms_without_station_as_surface():
    subway_data = pd.DataFrame([_stop_area_row(10, _stop_area_members()), _route_row(20, _route_members())])

    _, nodes = overpass_subway2edgenode(subway_data, LOCAL_CRS)
    types = dict(zip(nodes["node_id"], nodes["type"]))

    # Route platforms belong to no stop area, so nothing links them to a station or an entrance.
    assert types[302] == "platform"
    assert types[304] == "platform"
    # The stop-area platform is reachable through its station and stays underground.
    assert types[1] == "subway_station"
    assert types[2] == "subway_platform"


def test_overpass_subway2edgenode_parses_routes_without_stop_areas():
    # Territories where subway stop areas are not mapped produce no stop-area edges at all.
    subway_data = pd.DataFrame([_route_row(20, _route_members())])

    edges, nodes = overpass_subway2edgenode(subway_data, LOCAL_CRS)
    types = dict(zip(nodes["node_id"], nodes["type"]))

    assert types[301] == "subway"
    assert types[302] == "platform"
    assert {"boarding", "subway"} <= set(edges["type"])


def test_overpass_subway2edgenode_builds_platforms_for_stops_without_platform_members():
    # Stops without a platform are paired with a generated one, which reads the stop-area edges.
    subway_data = pd.DataFrame([_route_row(20, _route_members(platforms=False))])

    edges, nodes = overpass_subway2edgenode(subway_data, LOCAL_CRS)
    types = dict(zip(nodes["node_id"], nodes["type"]))

    assert types["from_301"] == "platform"
    assert types["from_303"] == "platform"
    assert ("301", "from_301") in {(str(u), str(v)) for u, v in zip(edges["u"], edges["v"])}


@pytest.mark.parametrize(
    "members",
    [
        pytest.param([], id="no_members"),
        pytest.param([{"type": "node", "role": "stop", "lat": 34.26, "lon": -6.58}], id="member_without_ref"),
    ],
)
def test_overpass_ground_transport2edgenode_survives_members_without_ref(members):
    """A relation whose members carry no ``ref`` must yield nothing, not raise.

    Such relations exist: OSM allows a route relation with no members at all.
    One of them in Kenitra raised ``KeyError: 'ref'`` and left the city with no
    graph, because the column every consumer indexes by name was never created.
    """
    route = pd.Series(
        {"tags": {"ref": "12"}, "transport_type": "bus", "members": members},
        name=99,
    )

    edges, nodes = overpass_ground_transport2edgenode(route, LOCAL_CRS, {}, [])

    assert len(nodes) == 0
    assert len(edges) == 0


# ---------------------------------------------------------------------------
# Route stitching: pieces of a relation that share no vertex
# ---------------------------------------------------------------------------


def _longest_step(coords):
    return max(math.dist(a, b) for a, b in zip(coords, coords[1:]))


def test_link_unconnected_puts_a_detour_between_the_pieces_around_it():
    """A piece that belongs between two others goes there, not onto the far end of the route.

    Delhi's "(+) TMS" leaves its corridor for a loop into a terminal whose closing ways are missing
    from the relation. The corridor before and after the loop meet at one vertex, so the greedy
    chain joined them first and could only glue the loop onto an end: a 14 km line across the city.
    """
    before = {"coords": [(-5000.0, 0.0), (0.0, 0.0)], "speeds": [1.0, 1.0]}
    loop = {"coords": [(100.0, 300.0), (50.0, 60.0)], "speeds": [2.0, 2.0]}
    after = {"coords": [(0.0, 0.0), (5000.0, 0.0)], "speeds": [3.0, 3.0]}

    [linked] = _link_unconnected([before, loop, after])

    # The loop leaves from and returns to the same vertex, so either way round it costs the same.
    assert linked["coords"][:2] == [(-5000.0, 0.0), (0.0, 0.0)]
    assert sorted(linked["coords"][2:4]) == [(50.0, 60.0), (100.0, 300.0)]
    assert linked["coords"][4:] == [(0.0, 0.0), (5000.0, 0.0)]
    assert _longest_step(linked["coords"][1:5]) < 400
    # One speed per vertex, each for the segment leaving it; a straight join keeps the road it leaves.
    assert linked["speeds"] == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_link_unconnected_runs_in_member_order():
    first = {"coords": [(0.0, 0.0), (1000.0, 0.0)]}
    second = {"coords": [(1050.0, 0.0), (2000.0, 0.0)]}

    [linked] = _link_unconnected([second, first])

    # The joins are the same either way round; the members decide the direction.
    assert linked["coords"] == [(2000.0, 0.0), (1050.0, 0.0), (1000.0, 0.0), (0.0, 0.0)]
    assert linked["speeds"] is None


def test_overpass_ground_transport2edgenode_follows_members_when_the_first_way_is_drawn_backwards():
    """The route runs in member order even when its first way is drawn against the traffic.

    Every member after such a way used to be prepended to it, so the whole route came out reversed
    and its one-way edges pointed the wrong way.
    """
    members = [
        _member(401, "stop", 59.9100, 30.3020),
        _member(402, "stop", 59.9100, 30.3180),
        {
            "type": "way",
            "ref": 403,
            "role": "",
            "geometry": [{"lat": 59.9100, "lon": 30.3100}, {"lat": 59.9100, "lon": 30.3000}],
        },
        {
            "type": "way",
            "ref": 404,
            "role": "",
            "geometry": [{"lat": 59.9100, "lon": 30.3100}, {"lat": 59.9100, "lon": 30.3200}],
        },
    ]
    route = pd.Series({"tags": {"ref": "7"}, "transport_type": "bus", "members": members}, name=77)

    edges, _ = overpass_ground_transport2edgenode(route, LOCAL_CRS, {}, [])

    ride = edges[edges["type"] == "bus"]
    assert [(str(u), str(v)) for u, v in zip(ride["u"], ride["v"])] == [("401", "402")]


def test_link_unconnected_cuts_the_route_at_a_hole():
    """A join longer than a kilometre is a hole in the relation: the route is cut there, not bridged."""
    near = {"coords": [(0.0, 0.0), (1000.0, 0.0)]}
    close = {"coords": [(1900.0, 0.0), (3000.0, 0.0)]}
    far = {"coords": [(4100.0, 0.0), (5000.0, 0.0)]}

    parts = _link_unconnected([near, close, far])

    # 900 m is still a way OSM missed; 1100 m is not.
    assert [part["coords"] for part in parts] == [
        [(0.0, 0.0), (1000.0, 0.0), (1900.0, 0.0), (3000.0, 0.0)],
        [(4100.0, 0.0), (5000.0, 0.0)],
    ]


def _members_with_a_hole():
    # One route mapped in two stretches 5 km apart, with two stops on each and nothing listed between them.
    members = [
        _member(501, "stop", 59.9100, 30.3020),
        _member(502, "stop", 59.9100, 30.3080),
        _member(503, "stop", 59.9100, 30.4020),
        _member(504, "stop", 59.9100, 30.4080),
    ]
    for ref, (start, end) in ((505, (30.3000, 30.3100)), (506, (30.4000, 30.4100))):
        members.append(
            {
                "type": "way",
                "ref": ref,
                "role": "",
                "geometry": [{"lat": 59.9100, "lon": start}, {"lat": 59.9100, "lon": end}],
            }
        )
    return members


def _rides(edges, mode):
    ride = edges[edges["type"] == mode]
    return {(str(u), str(v)) for u, v in zip(ride["u"], ride["v"])}


def test_overpass_ground_transport2edgenode_does_not_bridge_a_hole_in_the_relation():
    route = pd.Series({"tags": {"ref": "405A"}, "transport_type": "bus", "members": _members_with_a_hole()}, name=88)

    edges, nodes = overpass_ground_transport2edgenode(route, LOCAL_CRS, {}, [])

    assert _rides(edges, "bus") == {("501", "502"), ("503", "504")}
    assert {"501", "502", "503", "504"} <= set(nodes["node_id"].astype(str))


def test_overpass_subway2edgenode_does_not_bridge_a_hole_in_the_relation():
    subway_data = pd.DataFrame([_route_row(20, _members_with_a_hole())])

    edges, _ = overpass_subway2edgenode(subway_data, LOCAL_CRS)

    assert _rides(edges, "subway") == {("501", "502"), ("503", "504")}


def test_overpass_ground_transport2edgenode_follows_the_stop_order_when_ways_are_listed_backwards():
    """Stops, not ways, decide the direction: PTv2 lists them in travel order whichever way the ways run.

    A relation listing its ways from the far end came out backwards, and every one-way edge with it.
    """
    members = [
        _member(601, "stop", 59.9100, 30.3020),
        _member(602, "stop", 59.9100, 30.3080),
        _member(603, "stop", 59.9100, 30.3180),
        {
            "type": "way",
            "ref": 604,
            "role": "",
            "geometry": [{"lat": 59.9100, "lon": 30.3100}, {"lat": 59.9100, "lon": 30.3200}],
        },
        {
            "type": "way",
            "ref": 605,
            "role": "",
            "geometry": [{"lat": 59.9100, "lon": 30.3000}, {"lat": 59.9100, "lon": 30.3100}],
        },
    ]
    route = pd.Series({"tags": {"ref": "8"}, "transport_type": "bus", "members": members}, name=78)

    edges, _ = overpass_ground_transport2edgenode(route, LOCAL_CRS, {}, [])

    assert _rides(edges, "bus") == {("601", "602"), ("602", "603")}


def test_overpass_ground_transport2edgenode_keeps_the_ways_direction_against_too_few_stops():
    """Two stops listed backwards do not reverse a route whose ways are in order.

    Stop lists are sometimes jumbled or hold a couple of platforms for a whole route; Delhi's 740Extra was
    reversed by one such pair while its ways and every one-way street it drives agreed on the direction.
    """
    members = [
        _member(702, "stop", 59.9100, 30.3180),
        _member(701, "stop", 59.9100, 30.3020),
        {
            "type": "way",
            "ref": 703,
            "role": "",
            "geometry": [{"lat": 59.9100, "lon": 30.3000}, {"lat": 59.9100, "lon": 30.3100}],
        },
        {
            "type": "way",
            "ref": 704,
            "role": "",
            "geometry": [{"lat": 59.9100, "lon": 30.3100}, {"lat": 59.9100, "lon": 30.3200}],
        },
    ]
    route = pd.Series({"tags": {"ref": "9"}, "transport_type": "bus", "members": members}, name=79)

    edges, _ = overpass_ground_transport2edgenode(route, LOCAL_CRS, {}, [])

    assert _rides(edges, "bus") == {("701", "702")}


def test_overpass_ground_transport2edgenode_leaves_out_a_route_without_stops():
    """A relation listing no stops or platforms has nowhere to board and yields nothing.

    Such routes used to get two synthetic stops at the ends of their path, counted as mapped stops.
    """
    members = [
        {
            "type": "way",
            "ref": 801,
            "role": "",
            "geometry": [{"lat": 59.9100, "lon": 30.3000}, {"lat": 59.9100, "lon": 30.3100}],
        }
    ]
    route = pd.Series({"tags": {"ref": "10"}, "transport_type": "bus", "members": members}, name=80)

    edges, nodes = overpass_ground_transport2edgenode(route, LOCAL_CRS, {}, [])

    assert len(edges) == 0
    assert len(nodes) == 0
