# pylint: disable=redefined-outer-name

import pytest

from iduedu import (
    config,
)

config.configure_logging("DEBUG")

pytestmark = [pytest.mark.network, pytest.mark.slow]


def test_get_boundary_by_osm_id(bounds):
    assert bounds is not None
