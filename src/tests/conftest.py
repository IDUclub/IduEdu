import pytest

from iduedu import get_4326_boundary, get_intermodal_graph


@pytest.fixture(scope="session")
def bounds():
    print("\n Downloading boundary 1114252 \n")
    return get_4326_boundary(osm_id=1114252)  # OSM ID for https://www.openstreetmap.org/relation/1114252


@pytest.fixture(scope="session")
def intermodal_graph(bounds):
    print("\n Downloading intermodal graph for bounds \n")
    return get_intermodal_graph(territory=bounds)
