# pylint: disable=redefined-outer-name, protected-access, unused-argument
"""The downloader dispatch to the pbf source."""


import pandas as pd
import pytest
from shapely import box

from iduedu import config
from iduedu.overpass import downloaders

pytestmark = pytest.mark.unit

SQUARE = box(30.0, 59.0, 30.01, 59.01)


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch):
    monkeypatch.setattr(config, "osm_source", "overpass")
    monkeypatch.setattr(config, "pbf_path", None)
    monkeypatch.setattr(config, "overpass_date", None)


class _FakeSource:
    def __init__(self):
        self.calls = []

    def ways_by_filter(self, polygon, way_filter):
        self.calls.append(("ways", way_filter))
        return [{"type": "way", "id": 1, "nodes": [1, 2], "geometry": [], "tags": {"highway": "x"}}]

    def ways_table(self, polygon, way_filter):
        return pd.DataFrame(self.ways_by_filter(polygon, way_filter))

    def routes_by_poly(self, polygon, types):
        self.calls.append(("routes", tuple(types)))
        return [{"type": "relation", "id": 7}]

    def member_tags(self, members):
        self.calls.append(("members", sorted((m["type"], m["ref"]) for m in members)))
        return {("node", 5): {"tags": {}}}

    def boundary(self, osm_id):
        self.calls.append(("boundary", osm_id))
        ring = [{"lat": 0.0, "lon": 0.0}, {"lat": 0.0, "lon": 1.0}, {"lat": 1.0, "lon": 1.0}, {"lat": 0.0, "lon": 0.0}]
        return [
            {
                "type": "relation",
                "id": osm_id,
                "members": [{"type": "way", "ref": 1, "role": "outer", "geometry": ring}],
            }
        ]


@pytest.fixture
def fake_pbf(monkeypatch, tmp_path):
    source = _FakeSource()
    path = tmp_path / "x.osm.pbf"
    path.write_bytes(b"")

    def _fail(*args, **kwargs):
        raise AssertionError("Overpass or its cache must not be used in pbf mode")

    monkeypatch.setattr(downloaders, "_pbf_source", lambda: source)
    monkeypatch.setattr(downloaders, "_overpass_request", _fail)
    monkeypatch.setattr(downloaders, "cache_load", _fail)
    monkeypatch.setattr(downloaders, "cache_save_async", _fail)
    monkeypatch.setattr(config, "osm_source", "pbf")
    monkeypatch.setattr(config, "pbf_path", str(path))
    return source


def test_downloaders_dispatch_to_the_pbf_source(fake_pbf):
    frame = downloaders.get_network_by_filters(SQUARE, '["highway"]')
    assert isinstance(frame, pd.DataFrame) and frame["id"].tolist() == [1]
    assert downloaders.get_routes_by_poly(SQUARE, ["tram", "bus", "bus"]) == [{"type": "relation", "id": 7}]
    assert downloaders.fetch_member_tags(iter([{"type": "node", "ref": 5}])) == {("node", 5): {"tags": {}}}
    boundary = downloaders.get_boundary_by_osm_id(42)
    assert boundary.area == pytest.approx(0.5)
    assert fake_pbf.calls == [
        ("ways", '["highway"]'),
        ("routes", ("bus", "tram")),
        ("members", [("node", 5)]),
        ("boundary", 42),
    ]


def test_pbf_mode_treats_overpass_date_as_empty(fake_pbf, monkeypatch):
    monkeypatch.setattr(config, "overpass_date", "2020-01-01T00:00:00Z")
    assert downloaders.active_overpass_date() is None
    assert downloaders.osm_source_label().startswith("PBF file ")
    monkeypatch.setattr(config, "osm_source", "overpass")
    assert downloaders.active_overpass_date() == "2020-01-01T00:00:00Z"
    assert downloaders.osm_source_label() == "Overpass"


def test_pbf_mode_without_a_path_is_an_error(monkeypatch):
    monkeypatch.setattr(config, "osm_source", "pbf")
    with pytest.raises(ValueError, match="pbf_path"):
        downloaders.get_network_by_filters(SQUARE, '["highway"]')


def test_overpass_mode_is_unchanged(monkeypatch):
    sent = []

    class _Response:
        def json(self):
            return {"elements": [{"type": "way", "id": 3}]}

    monkeypatch.setattr(downloaders, "cache_load", lambda prefix, key_src: None)
    monkeypatch.setattr(downloaders, "cache_save_async", lambda prefix, key_src, obj: None)
    monkeypatch.setattr(downloaders, "_overpass_request", lambda *a, **k: sent.append(k) or _Response())
    monkeypatch.setattr(downloaders, "_pbf_source", lambda: pytest.fail("pbf source used in overpass mode"))
    assert downloaders.get_network_by_filters(SQUARE, '["highway"]')["id"].tolist() == [3]
    assert 'way["highway"](poly:' in sent[0]["data"]["data"]
