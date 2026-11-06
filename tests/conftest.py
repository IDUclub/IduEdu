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


_OSM_SOURCE = pytest.StashKey[tuple[str, str | None]]()


def _network_osm_source(pytest_config) -> tuple[str, str | None]:
    """Decide once per run where network tests read OSM data.

    Returns ``("overpass", None)``, ``("pbf", path)`` or ``("skip", reason)``. With ``--osm-source=auto`` the
    test territory is cut out of its Geofabrik extract only when Overpass does not answer. An OSM file selected
    through ``IDUEDU_OSM_SOURCE=pbf`` is kept as it is.
    """
    if _OSM_SOURCE in pytest_config.stash:
        return pytest_config.stash[_OSM_SOURCE]

    import importlib.util

    import requests

    from iduedu import config, download_osm_pbf

    choice = pytest_config.getoption("--osm-source")
    if config.osm_source == "pbf":
        decision = ("pbf", config.pbf_path)
        print(f"\n OSM data from {config.pbf_path} \n")
    elif choice == "overpass" or (choice == "auto" and _overpass_available()):
        decision = ("overpass", None)
        print(f"\n OSM data from Overpass {config.overpass_url} \n")
    elif importlib.util.find_spec("osmium") is None:
        decision = ("skip", "Overpass is not used and the PBF fallback needs pyosmium: pip install 'iduedu[pbf]'")
    else:
        print(f"\n Overpass is not used; OSM data from the Geofabrik extract {TERRITORY_GEOFABRIK_REGION} \n")
        try:
            path = download_osm_pbf(osm_id=TERRITORY_OSM_ID, region=TERRITORY_GEOFABRIK_REGION)
            decision = ("pbf", str(path))
        except requests.RequestException as exc:
            decision = ("skip", f"Overpass is not used and the Geofabrik extract could not be downloaded: {exc}")
    pytest_config.stash[_OSM_SOURCE] = decision
    return decision


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):  # pylint: disable=unused-argument
    """Run each network test, with its fixture setup and teardown, on the chosen OSM source.

    The switch to the PBF file covers only network tests: unit tests in the same run keep the default source
    and their mocked Overpass responses.
    """
    if "network" not in item.keywords or item.get_closest_marker("skip"):
        yield
        return
    from iduedu import config, use_pbf

    kind, value = _network_osm_source(item.config)
    if kind == "skip":
        item.add_marker(pytest.mark.skip(reason=value))
    if kind != "pbf" or config.osm_source == "pbf":
        yield
        return
    with use_pbf(value):
        yield


@pytest.fixture(scope="session")
def osm_source(request):
    """``"overpass"`` or the path of the OSM file the network tests read."""
    kind, value = _network_osm_source(request.config)
    return value if kind == "pbf" else kind


@pytest.fixture(scope="session")
def bounds(territory_osm_id):
    from iduedu import get_4326_boundary

    print(f"\n Downloading boundary {territory_osm_id} \n")
    return get_4326_boundary(osm_id=territory_osm_id)


@pytest.fixture(scope="session")
def intermodal_graph(bounds):
    from iduedu import get_intermodal_graph

    print("\n Downloading intermodal graph for bounds \n")
    return get_intermodal_graph(territory=bounds)
