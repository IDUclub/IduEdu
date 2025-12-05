import pytest

from iduedu import get_4326_boundary, get_intermodal_graph


@pytest.fixture(scope="session")
def territory_osm_id():
    return 1114252  # OSM ID for https://www.openstreetmap.org/relation/1114252


@pytest.fixture(scope="session")
def bounds(territory_osm_id):
    print(f"\n Downloading boundary {territory_osm_id} \n")
    return get_4326_boundary(osm_id=territory_osm_id)


@pytest.fixture(scope="session")
def intermodal_graph(bounds):
    print("\n Downloading intermodal graph for bounds \n")
    return get_intermodal_graph(territory=bounds)
