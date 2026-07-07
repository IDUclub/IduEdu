import os
import sys
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


if _coverage_is_active():
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_JIT_COVERAGE", "1")
numba_cache_dir = Path(__file__).resolve().parents[1] / ".pytest_cache" / "numba" / str(os.getpid())
numba_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))

# Import numba (which imports numpy internals) before any test module imports geopandas.
# Under coverage tracing, letting geopandas trigger the first numpy import causes numpy 2.x to
# fail with "cannot load module more than once per process". Forcing numba's numpy import path
# to run first — after the NUMBA_* env vars above are set — avoids that double-load crash.
import numba  # noqa: E402,F401  pylint: disable=wrong-import-position,unused-import

NETWORK_ONLY_TEST_MODULES = {
    "test_downloaders_network.py",
    "test_graph_builders_network.py",
    "test_matrix_network.py",
    "test_nx_utils_network.py",
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that may call Overpass or depend on downloaded graph data",
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


@pytest.fixture(scope="session")
def territory_osm_id():
    return 1114252  # OSM ID for https://www.openstreetmap.org/relation/1114252


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
