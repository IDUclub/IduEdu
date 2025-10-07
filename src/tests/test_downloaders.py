# pylint: disable=redefined-outer-name

from iduedu import (
    config,
)

config.configure_logging("DEBUG")


def test_get_boundary_by_osm_id(bounds):
    assert bounds is not None
