# pylint: disable=redefined-outer-name, protected-access, unused-argument
"""The pbf source without pyosmium, and the downloader dispatch, tested without reading any OSM file."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest
from shapely import box

from iduedu import config
from iduedu.overpass import downloaders

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
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


_NO_OSMIUM_SCRIPT = textwrap.dedent(
    """
    import json, sys
    sys.modules["osmium"] = None  # every "import osmium" now raises ModuleNotFoundError(name="osmium")

    import iduedu
    from iduedu import config, get_walk_graph
    from iduedu.overpass import downloaders
    from shapely import box

    result = {}
    for name, call in [
        ("PbfSource", lambda: iduedu.PbfSource(sys.argv[1])),
        ("split_by_polygons", lambda: iduedu.split_by_polygons(sys.argv[1], {"a": box(0, 0, 1, 1)}, ".")),
        ("snapshot_time", lambda: iduedu.snapshot_time(sys.argv[1])),
        ("set_osm_source", lambda: config.set_osm_source("pbf", path=sys.argv[1])),
    ]:
        try:
            call()
            result[name] = "no error"
        except ImportError as exc:
            result[name] = str(exc)

    config.osm_source, config.pbf_path = "pbf", sys.argv[1]
    try:
        downloaders.get_network_by_filters(box(0, 0, 1, 1), '["highway"]')
        result["pbf_mode"] = "no error"
    except ImportError as exc:
        result["pbf_mode"] = str(exc)
    config.osm_source = "overpass"

    class _Response:
        def json(self):
            return {"elements": [
                {"type": "way", "id": 1, "nodes": [1, 2, 3], "tags": {"highway": "residential"},
                 "geometry": [{"lat": 59.0, "lon": 30.0}, {"lat": 59.0, "lon": 30.001}, {"lat": 59.001, "lon": 30.001}]}
            ]}

    downloaders.cache_load = lambda prefix, key_src: None
    downloaders.cache_save_async = lambda prefix, key_src, obj: None
    downloaders._overpass_request = lambda *a, **k: _Response()
    graph = get_walk_graph(territory=box(29.99, 58.99, 30.01, 59.01))
    result["walk_edges"] = len(graph.edges_gdf)
    result["osmium_loaded"] = sys.modules.get("osmium") is not None
    print(json.dumps(result))
    """
)


def test_iduedu_works_without_osmium(tmp_path):
    osm_file = tmp_path / "city.osm.pbf"
    osm_file.write_bytes(b"")
    proc = subprocess.run(
        [sys.executable, "-c", _NO_OSMIUM_SCRIPT, str(osm_file)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
        check=False,
        # This subprocess checks optional imports and OSM building, not routing.
        # Avoid compiling unused Dijkstra kernels in a second interpreter;
        # the main routing/Numba tests still exercise actual JIT compilation.
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "ENABLE_TQDM": "0", "NUMBA_DISABLE_JIT": "1"},
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    for name in ("PbfSource", "split_by_polygons", "snapshot_time", "set_osm_source", "pbf_mode"):
        assert "pip install 'iduedu[pbf]'" in result[name], (name, result[name])
    assert result["walk_edges"] > 0
    assert result["osmium_loaded"] is False
