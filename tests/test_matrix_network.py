# pylint: disable=redefined-outer-name

import os

import geopandas as gpd
import numpy as np
import pytest

from iduedu import od_matrix

pytestmark = [pytest.mark.network, pytest.mark.slow]


@pytest.fixture(scope="module")
def sample_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../docs/examples/data/spb_buildings.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File {data_path} not found!")
    return gpd.read_parquet(data_path).head(8)


@pytest.mark.parametrize(
    "origins,destinations",
    [(slice(None), slice(None)), (slice(0, 1), slice(1, 2)), (slice(None), slice(0, 4)), (slice(0, 4), slice(None))],
    ids=["square", "single", "many-to-less", "less-to-many"],
)
def test_od_matrix_on_intermodal_graph(sample_data, intermodal_graph, origins, destinations):
    gdf_origins, gdf_destinations = sample_data.iloc[origins], sample_data.iloc[destinations]

    matrix = od_matrix(
        intermodal_graph,
        gdf_origins=gdf_origins,
        gdf_destinations=gdf_destinations,
        weight="time_min",
        dtype=np.float32,
    )

    assert matrix.shape == (len(gdf_origins), len(gdf_destinations))
    assert np.array_equal(matrix.index, gdf_origins.index)
    assert np.array_equal(matrix.columns, gdf_destinations.index)
    assert np.isfinite(matrix.to_numpy()).any()
