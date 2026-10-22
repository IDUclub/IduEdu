import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from iduedu.overpass import cache, downloaders
from iduedu.overpass.cache import cache_load, cache_save, get_cache_dir
from iduedu.overpass.downloaders import (
    _poly_to_overpass,
    get_4326_boundary,
    get_boundary_by_osm_id,
    get_network_by_filters,
    get_routes_by_poly,
)

pytestmark = pytest.mark.unit


SQUARE = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _disable_cache(monkeypatch):
    """Force every downloader test through the live (mocked) request path, not the disk cache."""
    monkeypatch.setattr(downloaders, "cache_load", lambda prefix, key_src: None)
    monkeypatch.setattr(downloaders, "cache_save_async", lambda prefix, key_src, obj: None)


def _mock_request(monkeypatch, payload):
    calls = {}

    def _fake_request(method, overpass_url, params=None, data=None, timeout=None, **kwargs):
        calls["method"] = method
        calls["params"] = params
        calls["data"] = data
        return _FakeResponse(payload)

    monkeypatch.setattr(downloaders, "_overpass_request", _fake_request)
    return calls


def test_poly_to_overpass_emits_lat_lon_pairs_without_closing_vertex():
    result = _poly_to_overpass(SQUARE)
    # exterior has 5 coords (closed ring); the closing vertex is dropped -> 4 "lat lon" pairs
    assert result == "0.0 0.0 0.0 1.0 1.0 1.0 1.0 0.0"


def test_get_4326_boundary_requires_input():
    with pytest.raises(ValueError, match="Either osm_id or territory"):
        get_4326_boundary()


def test_get_4326_boundary_returns_polygon_unchanged():
    assert get_4326_boundary(territory=SQUARE) is SQUARE


def test_get_4326_boundary_converts_multipolygon_to_convex_hull():
    multi = MultiPolygon([SQUARE, Polygon([(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)])])
    result = get_4326_boundary(territory=multi)
    assert isinstance(result, Polygon)
    assert result.contains(SQUARE.centroid)


def test_get_4326_boundary_accepts_geodataframe():
    gdf = gpd.GeoDataFrame(geometry=[SQUARE], crs=4326)
    result = get_4326_boundary(territory=gdf)
    assert isinstance(result, Polygon)
    assert result.equals(SQUARE)


def test_get_4326_boundary_accepts_geoseries():
    gs = gpd.GeoSeries([SQUARE], crs=4326)
    result = get_4326_boundary(territory=gs)
    assert isinstance(result, Polygon)
    assert result.equals(SQUARE)


def test_get_4326_boundary_rejects_unsupported_type():
    with pytest.raises(TypeError, match="territory must be one of"):
        get_4326_boundary(territory="not a geometry")


def test_get_4326_boundary_fetches_from_osm_id(monkeypatch):
    monkeypatch.setattr(downloaders, "get_boundary_by_osm_id", lambda osm_id: SQUARE)
    assert get_4326_boundary(osm_id=42) is SQUARE


def test_get_boundary_by_osm_id_builds_polygon_with_hole(monkeypatch):
    payload = {
        "elements": [
            {
                "members": [
                    {
                        "role": "outer",
                        "geometry": [
                            {"lon": 0.0, "lat": 0.0},
                            {"lon": 10.0, "lat": 0.0},
                            {"lon": 10.0, "lat": 10.0},
                            {"lon": 0.0, "lat": 10.0},
                            {"lon": 0.0, "lat": 0.0},
                        ],
                    },
                    {
                        "role": "inner",
                        "geometry": [
                            {"lon": 4.0, "lat": 4.0},
                            {"lon": 6.0, "lat": 4.0},
                            {"lon": 6.0, "lat": 6.0},
                            {"lon": 4.0, "lat": 6.0},
                            {"lon": 4.0, "lat": 4.0},
                        ],
                    },
                ]
            }
        ]
    }
    _mock_request(monkeypatch, payload)

    boundary = get_boundary_by_osm_id(123)

    assert boundary.area == pytest.approx(100.0 - 4.0)
    # a point inside the hole must not be contained
    assert not boundary.contains(Polygon([(4.5, 4.5), (5.5, 4.5), (5.5, 5.5), (4.5, 5.5)]).centroid)


def test_get_network_by_filters_returns_elements_dataframe(monkeypatch):
    payload = {"elements": [{"type": "way", "id": 1}, {"type": "way", "id": 2}]}
    calls = _mock_request(monkeypatch, payload)

    df = get_network_by_filters(SQUARE, '["highway"]')

    assert len(df) == 2
    assert calls["method"] == "POST"
    assert set(df["id"]) == {1, 2}


def test_get_network_by_filters_handles_empty_elements(monkeypatch):
    _mock_request(monkeypatch, {"elements": []})
    df = get_network_by_filters(SQUARE, '["highway"]')
    assert df.empty


def test_get_routes_by_poly_returns_empty_for_no_types():
    # No transport types -> short-circuits before building any query
    result = get_routes_by_poly(SQUARE, [])
    assert len(result) == 0


def test_get_routes_by_poly_single_type_builds_query(monkeypatch):
    calls = _mock_request(monkeypatch, {"elements": [{"type": "relation", "id": 7}]})

    result = get_routes_by_poly(SQUARE, ["bus"])

    assert result == [{"type": "relation", "id": 7}]
    assert '["route"="bus"]' in calls["data"]["data"]


def test_get_routes_by_poly_multiple_types_use_regex(monkeypatch):
    calls = _mock_request(monkeypatch, {"elements": []})

    get_routes_by_poly(SQUARE, ["bus", "tram"])

    query = calls["data"]["data"]
    assert '["route"~"^(bus|light_rail|tram)$"]' in query


def test_get_routes_by_poly_asks_for_the_tags_osm_actually_uses(monkeypatch):
    """A request for a mode must fetch the other tags OSM uses for it.

    Fourteen cities in the paper sample tag their trams ``light_rail`` and nine tag
    shared taxis ``share_taxi``; asking for the registry name alone left both out
    of the graph entirely.
    """
    calls = _mock_request(monkeypatch, {"elements": []})

    get_routes_by_poly(SQUARE, ["tram"])
    assert '["route"~"^(light_rail|tram)$"]' in calls["data"]["data"]

    calls = _mock_request(monkeypatch, {"elements": []})
    get_routes_by_poly(SQUARE, ["taxi"])
    assert '["route"~"^(share_taxi|taxi)$"]' in calls["data"]["data"]

    calls = _mock_request(monkeypatch, {"elements": []})
    get_routes_by_poly(SQUARE, ["bus"])
    assert '["route"="bus"]' in calls["data"]["data"], "a mode without synonyms keeps the simple filter"


def test_get_routes_by_poly_subway_adds_station_details(monkeypatch):
    monkeypatch.setattr(downloaders.config, "overpass_date", None, raising=False)
    calls = _mock_request(monkeypatch, {"elements": []})

    get_routes_by_poly(SQUARE, ["subway"])

    query = calls["data"]["data"]
    assert '["route"="subway"]' in query
    assert "stop_area" in query


def test_subway_query_carries_stop_area_members_after_a_marker(monkeypatch):
    monkeypatch.setattr(downloaders.config, "overpass_date", None, raising=False)
    calls = _mock_request(monkeypatch, {"elements": []})

    get_routes_by_poly(SQUARE, ["subway"])

    query = calls["data"]["data"]
    assert "nwr(r.stop_areas)->.stop_area_members;" in query
    marker = f'make section name="{downloaders.STOP_AREA_MEMBERS_SECTION}";'
    assert marker in query
    # members are printed after the marker, so nothing before it is mistaken for one
    assert query.index(marker) < query.index(".stop_area_members out center qt;")


def test_ground_only_query_has_no_member_section(monkeypatch):
    calls = _mock_request(monkeypatch, {"elements": []})
    get_routes_by_poly(SQUARE, ["bus"])
    assert "stop_area_members" not in calls["data"]["data"]


def test_split_stop_area_members_separates_the_section():
    route = {"type": "relation", "id": 7, "tags": {"route": "subway"}}
    marker = {"type": "section", "id": 1, "tags": {"name": downloaders.STOP_AREA_MEMBERS_SECTION}}
    platform = {"type": "node", "id": 5, "lat": 1.5, "lon": 2.5, "tags": {"public_transport": "platform"}}
    entrance = {"type": "way", "id": 6, "center": {"lat": 3.0, "lon": 4.0}, "tags": {"entrance": "yes"}}

    routes, members = downloaders.split_stop_area_members([route, marker, platform, entrance])

    assert routes == [route]
    assert members[("node", 5)]["__latlon__"] == (2.5, 1.5)
    assert members[("way", 6)]["__center__"] == (4.0, 3.0)
    assert members[("way", 6)]["tags"] == {"entrance": "yes"}


def test_split_stop_area_members_without_section_returns_none():
    elements = [{"type": "relation", "id": 7}, object()]
    routes, members = downloaders.split_stop_area_members(elements)
    assert routes == elements
    assert members is None


def test_request_does_not_probe_status_before_an_attempt(monkeypatch):
    """``/status`` is asked only after a 429; a plain request is one HTTP call."""

    class _Ok:
        status_code = 200

    probes = []
    monkeypatch.setattr(downloaders, "_get_overpass_pause", lambda *a, **k: probes.append(1) or 0)
    monkeypatch.setattr(downloaders, "_overpass_http", lambda *a, **k: _Ok())

    downloaders._overpass_request("POST", "https://example.org/api/interpreter", data={"data": "x"})

    assert probes == []


@pytest.fixture
def cache_config(monkeypatch, tmp_path):
    monkeypatch.setattr(cache.config, "overpass_cache_enabled", True, raising=False)
    monkeypatch.setattr(cache.config, "overpass_cache_dir", str(tmp_path), raising=False)
    return tmp_path


def test_get_cache_dir_none_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(cache.config, "overpass_cache_enabled", False, raising=False)
    monkeypatch.setattr(cache.config, "overpass_cache_dir", str(tmp_path), raising=False)
    assert get_cache_dir() is None


def test_get_cache_dir_none_when_dir_empty(monkeypatch):
    monkeypatch.setattr(cache.config, "overpass_cache_enabled", True, raising=False)
    monkeypatch.setattr(cache.config, "overpass_cache_dir", "", raising=False)
    assert get_cache_dir() is None


def test_cache_roundtrip_miss_and_disabling(monkeypatch, cache_config):
    assert cache_load("routes", "missing") is None
    payload = {"elements": [{"type": "way", "id": 1}], "meta": "ok"}
    cache_save("network", "key", payload)
    assert cache_load("network", "key") == payload
    monkeypatch.setattr(cache.config, "overpass_cache_enabled", False)
    assert cache_load("network", "key") is None
