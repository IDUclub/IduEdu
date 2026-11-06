import os
import sys
from importlib.metadata import version
from pathlib import Path

import pytest


def _coverage_is_active() -> bool:
    if any(arg == "--cov" or arg.startswith("--cov=") or arg.startswith("--cov-report") for arg in sys.argv):
        return True
    try:
        import coverage
    except ImportError:
        return False
    return coverage.Coverage.current() is not None


coverage_active = _coverage_is_active()
if coverage_active:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_JIT_COVERAGE", "1")
if "NUMBA_CACHE_DIR" not in os.environ:
    # Numba invalidates entries when source/signatures change. A PID-specific
    # directory discarded that cache on every pytest invocation.
    cache_version = f"{sys.implementation.cache_tag}-numba-{version('numba')}"
    numba_cache_dir = Path(__file__).resolve().parents[1] / ".pytest_cache" / "numba" / cache_version
    numba_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache_dir)

# The early plugin in tests/__init__.py preloads numerical extensions for
# coverage runs. Ordinary runs import Numba here with JIT enabled.
import numba  # noqa: E402  pylint: disable=wrong-import-position

NETWORK_ONLY_TEST_MODULES = {
    "test_graph_builders_network.py",
    "test_matrix_network.py",
}

# The Geofabrik extract that covers the test territory, read when Overpass is unavailable.
TERRITORY_OSM_ID = 1114252
TERRITORY_GEOFABRIK_REGION = "northwestern-fed-district"


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that may call Overpass or depend on downloaded graph data",
    )
    parser.addoption(
        "--osm-source",
        choices=("auto", "overpass", "pbf"),
        default="auto",
        help="OSM data for network tests: Overpass, the Geofabrik extract of the test territory, "
        "or Overpass with the extract as a fallback when Overpass does not answer (auto, the default)",
    )


def pytest_ignore_collect(collection_path, config):
    if config.getoption("--run-network"):
        return False
    return collection_path.name in NETWORK_ONLY_TEST_MODULES


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return

    skip_network = pytest.mark.skip(reason="need --run-network option to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture(scope="session", autouse=True)
def _numba_threads():
    """Tiny synthetic graphs need few workers; respect explicit user settings."""
    previous = numba.get_num_threads()
    if "NUMBA_NUM_THREADS" not in os.environ:
        numba.set_num_threads(min(2, previous))
    yield
    numba.set_num_threads(previous)


def _overpass_available() -> bool:
    """One tiny query: True when the configured Overpass instance answers it."""
    import requests

    from iduedu import config

    try:
        response = requests.post(
            config.overpass_url,
            data={"data": "[out:json][timeout:20];node(1);out ids;"},
            timeout=30,
            headers={"User-Agent": config.user_agent},
            proxies=config.proxies,
            verify=config.verify_ssl,
        )
    except requests.RequestException:
        return False
    return response.status_code == 200


@pytest.fixture(scope="session")
def territory_osm_id():
    return TERRITORY_OSM_ID  # OSM ID for https://www.openstreetmap.org/relation/1114252


@pytest.fixture(scope="session")
def osm_source(request, territory_osm_id):
    """Where network tests read OSM data: Overpass, or the territory cut out of its Geofabrik extract.

    With ``--osm-source=auto`` the extract is used only when Overpass does not answer. An OSM file selected
    through ``IDUEDU_OSM_SOURCE=pbf`` is kept as it is.
    """
    import requests

    from iduedu import config, download_osm_pbf, use_pbf

    choice = request.config.getoption("--osm-source")
    if config.osm_source == "pbf":
        print(f"\n OSM data from {config.pbf_path} \n")
        yield config.pbf_path
        return
    if choice == "overpass" or (choice == "auto" and _overpass_available()):
        print(f"\n OSM data from Overpass {config.overpass_url} \n")
        yield "overpass"
        return

    pytest.importorskip("osmium", reason="the PBF fallback needs pyosmium: pip install 'iduedu[pbf]'")
    print(f"\n Overpass is not used; OSM data from the Geofabrik extract {TERRITORY_GEOFABRIK_REGION} \n")
    try:
        path = download_osm_pbf(osm_id=territory_osm_id, region=TERRITORY_GEOFABRIK_REGION)
    except requests.RequestException as exc:
        pytest.skip(f"Overpass is not used and the Geofabrik extract could not be downloaded: {exc}")
    with use_pbf(path):
        yield str(path)


@pytest.fixture(scope="session")
def bounds(territory_osm_id, osm_source):  # pylint: disable=unused-argument
    from iduedu import get_4326_boundary

    print(f"\n Downloading boundary {territory_osm_id} \n")
    return get_4326_boundary(osm_id=territory_osm_id)


@pytest.fixture(scope="session")
def intermodal_graph(bounds):
    from iduedu import get_intermodal_graph

    print("\n Downloading intermodal graph for bounds \n")
    return get_intermodal_graph(territory=bounds)
