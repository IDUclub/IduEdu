from pathlib import Path

import pandas as pd
import pytest

from iduedu import get_gtfs_public_transport_graph, merge_gtfs_feeds
from iduedu.gtfs.reader import read_gtfs_feed
from tests.factories import write_gtfs_feed

pytestmark = pytest.mark.unit

# Saint Petersburg: far enough north that a degree of longitude is about 55.8 km,
# which is what the offsets below are sized against.
LAT = 59.9300
LON = 30.3100
TEN_METRES_LON = 0.00018

BASE_TABLES = {
    "agency.txt": "agency_id,agency_name,agency_url,agency_timezone\na,Agency,https://example.com,Europe/Moscow\n",
    "calendar.txt": (
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
        "svc,1,1,1,1,1,1,1,20250101,20251231\n"
    ),
}


def _feed(directory: Path, *, lon_offset: float = 0.0, location_type: str = "0", short_name: str = "R") -> Path:
    """A minimal one-route feed. Every feed uses the same identifiers on purpose:
    that collision is what merging has to survive. ``short_name`` is the one field
    left distinguishable, because the built graph labels edges with it."""
    tables = {
        **BASE_TABLES,
        "stops.txt": (
            "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station\n"
            f"s1,First,{LAT},{LON + lon_offset},{location_type},\n"
            f"s2,Second,{LAT},{LON + lon_offset + 0.01},{location_type},\n"
        ),
        "routes.txt": f"route_id,agency_id,route_short_name,route_type\n1,a,{short_name},3\n",
        "trips.txt": "route_id,service_id,trip_id,shape_id\n1,svc,t1,\n1,svc,t2,\n",
        "stop_times.txt": (
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
            "t1,08:00:00,08:00:00,s1,1\n"
            "t1,08:10:00,08:10:00,s2,2\n"
            "t2,08:30:00,08:30:00,s1,1\n"
            "t2,08:40:00,08:40:00,s2,2\n"
        ),
    }
    return write_gtfs_feed(directory, tables)


def test_single_feed_is_returned_unchanged(tmp_path):
    """The property the rest of the library depends on: one feed in, that feed out.

    Published results were produced through the single-feed path, so merging must
    not quietly rename anything when there is nothing to disambiguate against.
    """
    source = _feed(tmp_path / "only")
    merged = merge_gtfs_feeds([source])
    direct = read_gtfs_feed(source)

    assert set(merged.tables) == set(direct.tables)
    for name, frame in direct.tables.items():
        assert merged[name].equals(frame), name


def test_colliding_identifiers_are_separated(tmp_path):
    merged = merge_gtfs_feeds([_feed(tmp_path / "subway"), _feed(tmp_path / "buses")])

    assert sorted(merged["routes"]["route_id"]) == ["buses:1", "subway:1"]
    assert sorted(merged["trips"]["trip_id"]) == ["buses:t1", "buses:t2", "subway:t1", "subway:t2"]
    assert sorted(merged["stops"]["stop_id"]) == ["buses:s1", "buses:s2", "subway:s1", "subway:s2"]


def test_references_follow_their_definitions(tmp_path):
    """A reference rewritten out of step with its definition is the failure mode
    that would silently attach one agency's trips to another agency's routes."""
    merged = merge_gtfs_feeds([_feed(tmp_path / "subway"), _feed(tmp_path / "buses")])

    for _, trip in merged["trips"].iterrows():
        assert trip["route_id"].split(":")[0] == trip["trip_id"].split(":")[0]
    for _, call in merged["stop_times"].iterrows():
        assert call["stop_id"].split(":")[0] == call["trip_id"].split(":")[0]


def test_blank_optional_references_stay_blank(tmp_path):
    """An empty parent_station means "no parent". Prefixing it would invent one."""
    merged = merge_gtfs_feeds([_feed(tmp_path / "subway"), _feed(tmp_path / "buses")])

    assert (merged["stops"]["parent_station"] == "").all()
    assert (merged["trips"]["shape_id"] == "").all()


def test_prefix_collision_is_disambiguated(tmp_path):
    """Two agencies can ship archives of the same name; the prefixes must differ
    anyway, or the merge reintroduces the collision it exists to remove."""
    merged = merge_gtfs_feeds([_feed(tmp_path / "a" / "gtfs"), _feed(tmp_path / "b" / "gtfs")])

    prefixes = {value.split(":")[0] for value in merged["routes"]["route_id"]}
    assert prefixes == {"gtfs", "gtfs2"}


def test_explicit_prefixes_are_used(tmp_path):
    merged = merge_gtfs_feeds([_feed(tmp_path / "one"), _feed(tmp_path / "two")], prefixes=["nyct", "mtabc"])
    assert sorted(merged["routes"]["route_id"]) == ["mtabc:1", "nyct:1"]


def test_nearby_stops_merge_across_feeds(tmp_path):
    """Two agencies publishing the same kerb should become one node, so that a
    transfer between them exists inside the graph."""
    merged = merge_gtfs_feeds(
        [_feed(tmp_path / "subway"), _feed(tmp_path / "buses", lon_offset=TEN_METRES_LON)],
        merge_stops_within=25.0,
    )

    surviving = set(merged["stops"]["stop_id"])
    assert len(surviving) == 2, surviving
    # Both feeds' calls now reference stops that exist.
    assert set(merged["stop_times"]["stop_id"]) <= surviving


def test_distant_stops_are_left_alone(tmp_path):
    merged = merge_gtfs_feeds(
        [_feed(tmp_path / "subway"), _feed(tmp_path / "buses", lon_offset=TEN_METRES_LON)],
        merge_stops_within=5.0,
    )
    assert len(set(merged["stops"]["stop_id"])) == 4


def test_merge_is_independent_of_source_order(tmp_path):
    """Which identifier survives must be a property of the data, not of the order
    the feeds were passed in."""
    subway = _feed(tmp_path / "subway")
    buses = _feed(tmp_path / "buses", lon_offset=TEN_METRES_LON)

    forward = merge_gtfs_feeds([subway, buses], merge_stops_within=25.0)
    backward = merge_gtfs_feeds([buses, subway], merge_stops_within=25.0)

    assert sorted(forward["stops"]["stop_id"]) == sorted(backward["stops"]["stop_id"])


def test_stations_are_not_fused(tmp_path):
    """location_type=1 is a station: structure rather than a place a vehicle calls
    at, and fusing two of them would restructure the hierarchy."""
    merged = merge_gtfs_feeds(
        [
            _feed(tmp_path / "subway", location_type="1"),
            _feed(tmp_path / "buses", lon_offset=TEN_METRES_LON, location_type="1"),
        ],
        merge_stops_within=25.0,
    )
    assert len(set(merged["stops"]["stop_id"])) == 4


def test_merged_feed_passes_validation(tmp_path):
    """The merge is checked against the library's own contract rather than against
    what the merge happens to produce."""
    merged = merge_gtfs_feeds([_feed(tmp_path / "subway"), _feed(tmp_path / "buses")], validate=True)
    assert not merged["stops"]["stop_id"].duplicated().any()


def test_prefix_count_must_match_sources(tmp_path):
    with pytest.raises(ValueError, match="prefixes"):
        merge_gtfs_feeds([_feed(tmp_path / "one"), _feed(tmp_path / "two")], prefixes=["only-one"])


def test_no_sources_is_an_error():
    with pytest.raises(ValueError, match="at least one source"):
        merge_gtfs_feeds([])


def test_builder_accepts_several_feeds(tmp_path):
    """The whole point: a city that publishes many feeds builds one graph."""
    graph = get_gtfs_public_transport_graph(
        [
            _feed(tmp_path / "subway", short_name="S"),
            _feed(tmp_path / "buses", short_name="B"),
        ]
    )

    assert not graph.nodes_gdf.empty
    routes = {value for value in graph.edges_gdf.get("route", []) if isinstance(value, str)}
    assert {"S", "B"} <= routes, routes


def test_builder_still_accepts_one_feed(tmp_path):
    source = _feed(tmp_path / "only")
    from_path = get_gtfs_public_transport_graph(source)
    from_list = get_gtfs_public_transport_graph([source])

    assert len(from_path.nodes_gdf) == len(from_list.nodes_gdf)
    assert len(from_path.edges_gdf) == len(from_list.edges_gdf)


# ---------------------------------------------------------------------------
# deduplicate: the same service published by two sources, counted once
# ---------------------------------------------------------------------------


def _timed_feed(directory: Path, *, times: tuple[str, str, str, str], lon_offset: float = 0.0) -> Path:
    """``_feed`` with its four stop times replaced (t1 at s1, s2; t2 at s1, s2)."""
    _feed(directory, lon_offset=lon_offset)
    a, b, c, d = times
    (directory / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        f"t1,{a},{a},s1,1\nt1,{b},{b},s2,2\nt2,{c},{c},s1,1\nt2,{d},{d},s2,2\n",
        encoding="utf-8",
    )
    return directory


def test_identical_feeds_are_counted_once(tmp_path):
    """A feed and its mirror: one set of platforms, one set of trips, a valid feed."""
    merged = merge_gtfs_feeds(
        [_feed(tmp_path / "original"), _feed(tmp_path / "mirror")], prefixes=["a", "b"], deduplicate=True
    )
    assert sorted(merged["stops"]["stop_id"]) == ["a:s1", "a:s2"]
    assert sorted(merged["trips"]["trip_id"]) == ["a:t1", "a:t2"]
    assert set(merged["stop_times"]["stop_id"]) <= set(merged["stops"]["stop_id"])
    graph = get_gtfs_public_transport_graph(merged)
    assert len(graph.edges_gdf) > 0


def test_deduplication_is_off_by_default(tmp_path):
    merged = merge_gtfs_feeds([_feed(tmp_path / "original"), _feed(tmp_path / "mirror")], prefixes=["a", "b"])
    assert len(merged["stops"]) == 4 and len(merged["trips"]) == 4


def test_same_platforms_different_timetable_keeps_every_trip(tmp_path):
    """Two operators calling at the same kerb at other times: one platform, all trips."""
    merged = merge_gtfs_feeds(
        [
            _feed(tmp_path / "day"),
            _timed_feed(tmp_path / "night", times=("23:00:00", "23:10:00", "23:30:00", "23:40:00")),
        ],
        prefixes=["day", "night"],
        deduplicate=True,
    )
    assert len(merged["stops"]) == 2
    assert len(merged["trips"]) == 4


def test_platforms_ten_metres_apart_stay_distinct(tmp_path):
    """Beyond the 0.5 m tolerance two platforms are two places, and trips calling
    at them are different trips even at identical times."""
    merged = merge_gtfs_feeds(
        [_feed(tmp_path / "a"), _feed(tmp_path / "b", lon_offset=TEN_METRES_LON)],
        prefixes=["a", "b"],
        deduplicate=True,
    )
    assert len(merged["stops"]) == 4
    assert len(merged["trips"]) == 4


def test_time_notation_does_not_hide_a_copy(tmp_path):
    """8:00:00 and 08:00:00 are one departure."""
    merged = merge_gtfs_feeds(
        [
            _feed(tmp_path / "padded"),
            _timed_feed(tmp_path / "unpadded", times=("8:00:00", "8:10:00", "8:30:00", "8:40:00")),
        ],
        prefixes=["a", "b"],
        deduplicate=True,
    )
    assert sorted(merged["trips"]["trip_id"]) == ["a:t1", "a:t2"]


def test_deduplication_is_independent_of_source_order(tmp_path):
    first, second = _feed(tmp_path / "first"), _feed(tmp_path / "second")
    forward = merge_gtfs_feeds([first, second], prefixes=["a", "b"], deduplicate=True)
    backward = merge_gtfs_feeds([second, first], prefixes=["b", "a"], deduplicate=True)
    assert sorted(forward["trips"]["trip_id"]) == sorted(backward["trips"]["trip_id"])
    assert sorted(forward["stops"]["stop_id"]) == sorted(backward["stops"]["stop_id"])


def test_one_source_is_never_deduplicated_against_itself(tmp_path):
    """Two platforms or two identical trips inside one feed are that feed's decision."""
    from iduedu import deduplicate_across_feeds

    feed = read_gtfs_feed(_feed(tmp_path / "only"))
    tables = {name: frame.copy() for name, frame in feed.tables.items()}
    for name, columns in (("stops", ["stop_id"]), ("stop_times", ["stop_id", "trip_id"]), ("trips", ["trip_id"])):
        for column in columns:
            tables[name][column] = "a:" + tables[name][column].astype(str)
    # a second platform on top of s1 and a second, identical run of t1
    stops = tables["stops"]
    tables["stops"] = pd.concat([stops, stops.iloc[[0]].assign(stop_id="a:s1bis")], ignore_index=True)
    trips = tables["trips"]
    tables["trips"] = pd.concat([trips, trips.iloc[[0]].assign(trip_id="a:t1bis")], ignore_index=True)
    times = tables["stop_times"]
    tables["stop_times"] = pd.concat(
        [times, times[times["trip_id"] == "a:t1"].assign(trip_id="a:t1bis")], ignore_index=True
    )

    counts = deduplicate_across_feeds(tables, ["a"])

    assert counts == {"platforms_fused": 0, "trips_dropped": 0}
    assert len(tables["stops"]) == 3 and len(tables["trips"]) == 3


def test_stations_are_not_fused_by_deduplication(tmp_path):
    merged = merge_gtfs_feeds(
        [_feed(tmp_path / "a", location_type="1"), _feed(tmp_path / "b", location_type="1")],
        prefixes=["a", "b"],
        deduplicate=True,
        validate=False,
    )
    assert len(merged["stops"]) == 4
