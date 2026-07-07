import pytest

from iduedu.overpass.parsers import (
    infer_role_from_tags,
    overpass_routes_to_df,
    parse_maxspeed_to_m_per_min,
)

pytestmark = pytest.mark.unit


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
