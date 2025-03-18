# pylint: disable=redefined-outer-name

import os
import random

import geopandas as gpd
import numpy as np
import pytest

from iduedu import get_adj_matrix_gdf_to_gdf, get_closest_nodes, get_intermodal_graph


@pytest.fixture(scope="module")
def sample_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../examples/data/spb_buildings.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл {data_path} не найден!")
    return gpd.read_parquet(data_path)


@pytest.fixture(scope="module")
def intermodal_graph(sample_data):
    polygon = sample_data.to_crs(4326).union_all().convex_hull.buffer(0.001)
    return get_intermodal_graph(polygon=polygon)


def test_get_square_adj_matrix(sample_data, intermodal_graph):
    matrix = get_adj_matrix_gdf_to_gdf(sample_data, sample_data, intermodal_graph, weight="time_min", threshold=15)

    assert matrix is not None

    assert matrix.shape[0] == len(sample_data)
    assert matrix.shape[1] == len(sample_data)

    assert np.array_equal(matrix.index, sample_data.index)
    assert np.array_equal(matrix.columns, sample_data.index)


def test_get_single_adj_matrix(sample_data, intermodal_graph):
    random_index_1 = random.choice(sample_data.index)
    random_index_2 = random.choice(sample_data.index)

    single_from = sample_data.loc[[random_index_1]]
    single_to = sample_data.loc[[random_index_2]]

    matrix = get_adj_matrix_gdf_to_gdf(single_from, single_to, intermodal_graph, weight="time_min", dtype=np.float32)

    assert matrix is not None

    assert matrix.shape[0] == 1
    assert matrix.shape[1] == 1

    assert matrix.index[0] == random_index_1
    assert matrix.columns[0] == random_index_2


def test_get_many_to_less_adj_matrix(sample_data, intermodal_graph):
    random_indices = random.sample(list(sample_data.index), k=len(sample_data) // 2)
    smaller_data = sample_data.loc[random_indices]

    matrix = get_adj_matrix_gdf_to_gdf(sample_data, smaller_data, intermodal_graph, weight="time_min", dtype=np.float32)

    assert matrix is not None

    assert matrix.shape[0] == len(sample_data)
    assert matrix.shape[1] == len(smaller_data)

    assert np.array_equal(matrix.index, sample_data.index)
    assert np.array_equal(matrix.columns, smaller_data.index)


def test_get_less_to_many_adj_matrix(sample_data, intermodal_graph):
    random_indices = random.sample(list(sample_data.index), k=len(sample_data) // 2)
    smaller_data = sample_data.loc[random_indices]

    matrix = get_adj_matrix_gdf_to_gdf(smaller_data, sample_data, intermodal_graph, weight="time_min", dtype=np.float32)

    assert matrix is not None

    assert matrix.shape[0] == len(smaller_data)
    assert matrix.shape[1] == len(sample_data)

    assert np.array_equal(matrix.index, smaller_data.index)
    assert np.array_equal(matrix.columns, sample_data.index)


def test_get_nearest_graph_node(sample_data, intermodal_graph):
    closest_nodes, dist = get_closest_nodes(sample_data, intermodal_graph)
    assert len(closest_nodes) == len(sample_data)
