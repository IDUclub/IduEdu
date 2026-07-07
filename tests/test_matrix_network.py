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


def test_get_square_od_matrix(sample_data, intermodal_graph):
    matrix = od_matrix(
        intermodal_graph,
        gdf_origins=sample_data,
        gdf_destinations=sample_data,
        weight="time_min",
        threshold=15,
    )

    assert matrix is not None
    assert matrix.shape == (len(sample_data), len(sample_data))
    assert np.array_equal(matrix.index, sample_data.index)
    assert np.array_equal(matrix.columns, sample_data.index)


def test_get_single_od_matrix(sample_data, intermodal_graph):
    single_from = sample_data.iloc[[0]]
    single_to = sample_data.iloc[[1]]

    matrix = od_matrix(
        intermodal_graph,
        gdf_origins=single_from,
        gdf_destinations=single_to,
        weight="time_min",
        dtype=np.float32,
    )

    assert matrix is not None
    assert matrix.shape == (1, 1)
    assert matrix.index[0] == single_from.index[0]
    assert matrix.columns[0] == single_to.index[0]


def test_get_many_to_less_od_matrix(sample_data, intermodal_graph):
    smaller_data = sample_data.iloc[: len(sample_data) // 2]

    matrix = od_matrix(
        intermodal_graph,
        gdf_origins=sample_data,
        gdf_destinations=smaller_data,
        weight="time_min",
        dtype=np.float32,
    )

    assert matrix is not None
    assert matrix.shape == (len(sample_data), len(smaller_data))
    assert np.array_equal(matrix.index, sample_data.index)
    assert np.array_equal(matrix.columns, smaller_data.index)


def test_get_less_to_many_od_matrix(sample_data, intermodal_graph):
    smaller_data = sample_data.iloc[: len(sample_data) // 2]

    matrix = od_matrix(
        intermodal_graph,
        gdf_origins=smaller_data,
        gdf_destinations=sample_data,
        weight="time_min",
        dtype=np.float32,
    )

    assert matrix is not None
    assert matrix.shape == (len(smaller_data), len(sample_data))
    assert np.array_equal(matrix.index, smaller_data.index)
    assert np.array_equal(matrix.columns, sample_data.index)
