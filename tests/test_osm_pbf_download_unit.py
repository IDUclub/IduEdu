# pylint: disable=redefined-outer-name
"""Geofabrik downloads against a local HTTP server that mimics download.geofabrik.de: index, extract, md5."""

import hashlib
import http.server
import json
import threading
from functools import partial

import pytest

osmium = pytest.importorskip("osmium")

from shapely import box  # noqa: E402  pylint: disable=wrong-import-position

from iduedu import (  # noqa: E402  pylint: disable=wrong-import-position
    config,
    download_osm_pbf,
    find_geofabrik_region,
    get_walk_graph,
    use_pbf,
)
from iduedu.osm_pbf import download  # noqa: E402  pylint: disable=wrong-import-position

pytestmark = pytest.mark.unit

CITY = box(30.300, 59.930, 30.320, 59.940)
REL_ID = 77


def _region_opl() -> str:
    """A walkable street crossing the city, a street far away and the city boundary relation."""
    lines = [
        "n1 v1 x30.3050000 y59.9350000",
        "n2 v1 x30.3150000 y59.9350000",
        "n3 v1 x30.4000000 y59.9000000",
        "n4 v1 x30.4100000 y59.9000000",
        "n11 v1 x30.3000000 y59.9300000",
        "n12 v1 x30.3200000 y59.9300000",
        "n13 v1 x30.3200000 y59.9400000",
        "n14 v1 x30.3000000 y59.9400000",
        "w1 v1 Thighway=residential,name=Inner Nn1,n2",
        "w2 v1 Thighway=residential,name=Outer Nn3,n4",
        "w11 v1 T Nn11,n12,n13,n14,n11",
        f"r{REL_ID} v1 Ttype=boundary,boundary=administrative,name=City Mw11@outer",
    ]
    return "\n".join(lines) + "\n"


@pytest.fixture()
def geofabrik(tmp_path, monkeypatch):
    """Serve a fake Geofabrik and point the downloader at it; yields (server dir, request log, cache dir)."""
    root = tmp_path / "server"
    root.mkdir()
    opl = tmp_path / "region.opl"
    opl.write_text(_region_opl(), encoding="utf-8")
    pbf = root / "city-region-latest.osm.pbf"
    with osmium.SimpleWriter(str(pbf)) as writer:
        for obj in osmium.FileProcessor(str(opl)):
            writer.add(obj)
    (root / "city-region-latest.osm.pbf.md5").write_text(
        f"{hashlib.md5(pbf.read_bytes(), usedforsecurity=False).hexdigest()}  {pbf.name}\n", encoding="utf-8"
    )

    log = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # silence the server
            pass

        def do_GET(self):  # pylint: disable=invalid-name
            log.append(self.path)
            super().do_GET()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(root)))
    base = f"http://127.0.0.1:{server.server_address[1]}"
    regions = [
        ("country", None, box(29, 59, 32, 61)),
        ("city-region", "country", box(30.2, 59.85, 30.5, 60.0)),
        ("elsewhere", "country", box(31, 60, 31.5, 60.5)),
    ]
    index = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": rid,
                    "parent": parent,
                    "name": rid.title(),
                    "urls": {"pbf": f"{base}/{rid}-latest.osm.pbf"},
                },
                "geometry": json.loads(json.dumps(geom.__geo_interface__)),
            }
            for rid, parent, geom in regions
        ],
    }
    (root / "index-v1.json").write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setattr(download, "GEOFABRIK_INDEX_URL", f"{base}/index-v1.json")
    monkeypatch.setattr(config, "enable_tqdm_bar", False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, log, tmp_path / "cache"
    server.shutdown()


def test_smallest_covering_region(geofabrik):
    _, _, cache = geofabrik
    assert find_geofabrik_region(CITY, out_dir=cache) == "city-region"
    assert find_geofabrik_region(box(30.9, 60.1, 31.1, 60.2), out_dir=cache) == "country"
    with pytest.raises(ValueError, match="No Geofabrik extract"):
        find_geofabrik_region(box(10, 10, 11, 11), out_dir=cache)


def test_download_cut_and_build_a_graph(geofabrik):
    _, log, cache = geofabrik
    # a stale partial download from an interrupted run is replaced, the checksum is verified
    (cache).mkdir(parents=True, exist_ok=True)
    (cache / "city-region-latest.osm.pbf.part").write_bytes(b"garbage")

    path = download_osm_pbf(osm_id=REL_ID, region="city-region", out_dir=cache)

    assert path.name == f"osm_{REL_ID}.osm.pbf"
    assert (cache / "city-region-latest.osm.pbf").exists()
    with config.use_pbf(path):
        walk = get_walk_graph(osm_id=REL_ID)
    assert set(walk.edges_gdf["name"]) == {"Inner"}

    requests_made = len(log)
    assert download_osm_pbf(osm_id=REL_ID, region="city-region", out_dir=cache) == path
    with use_pbf(osm_id=REL_ID, region="city-region", out_dir=cache):
        assert config.pbf_path == str(path)
    assert len(log) == requests_made  # the cut file is reused without touching the network
    assert config.osm_source == "overpass"


def test_use_pbf_downloads_a_territory_and_picks_the_region(geofabrik):
    _, _, cache = geofabrik
    with use_pbf(territory=CITY, out_dir=cache):
        walk = get_walk_graph(territory=CITY)
    assert set(walk.edges_gdf["name"]) == {"Inner"}
    assert config.osm_source == "overpass"


def test_checksum_mismatch_removes_the_download(geofabrik):
    root, _, cache = geofabrik
    (root / "city-region-latest.osm.pbf.md5").write_text("0" * 32, encoding="utf-8")
    with pytest.raises(OSError, match="Checksum mismatch"):
        download.download_geofabrik_region("city-region", out_dir=cache)
    assert not list(cache.glob("city-region-latest.osm.pbf*"))
