import pytest

from iduedu import get_boundary, get_walk_graph, get_intermodal_graph


@pytest.fixture(scope="session")
def bounds():
    return get_boundary(osm_id=1114252)  # OSM ID for https://www.openstreetmap.org/relation/1114252


@pytest.fixture(scope="session")
def walk_graph(bounds):
    return get_walk_graph(polygon=bounds)


@pytest.fixture(scope="session")
def intermodal_graph(bounds):
    return get_intermodal_graph(polygon=bounds)
