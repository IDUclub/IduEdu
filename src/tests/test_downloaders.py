import pytest
import geopandas as gpd
from iduedu import (
    config,
    get_boundary,
)
from iduedu.modules.downloaders import get_routes_by_poly, RequestError

config.change_logger_lvl("DEBUG")


@pytest.fixture(scope="session")
def bounds():
    return get_boundary(osm_id=1114252)  # OSM ID for https://www.openstreetmap.org/relation/1114252


def test_get_boundary_by_osm_id(bounds):
    assert bounds is not None

def test_get_boundary_by_name():
    bounds = get_boundary(territory_name="Василеостровский район")
    assert bounds is not None


def test_routes_by_wrong_poly_crs(bounds):
    wrong_poly = gpd.GeoDataFrame(geometry=[bounds],crs=4326).to_crs(32636).union_all()
    with pytest.raises(RequestError) as _:
        get_routes_by_poly(wrong_poly, public_transport_type='bus')
