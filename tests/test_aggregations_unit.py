import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from iduedu import UrbanGraph, aggregate_objects, diversity, nearest_objects, od_matrix
from tests.factories import CRS, _edges, _nodes, directed_oneway_graph, undirected_line_graph

pytestmark = pytest.mark.unit


def random_directed_graph(size=6, seed=3) -> UrbanGraph:
    """A grid whose streets are one-way at random, with random travel times."""
    rng = np.random.default_rng(seed)
    coords = {i * size + j: (j * 100.0, i * 100.0) for i in range(size) for j in range(size)}
    rows = []
    for i in range(size):
        for j in range(size):
            node = i * size + j
            for other in ([node + 1] if j + 1 < size else []) + ([node + size] if i + 1 < size else []):
                u, v = (node, other) if rng.random() < 0.5 else (other, node)
                rows.append({"u": u, "v": v, "oneway": bool(rng.random() < 0.4), "time_min": float(rng.uniform(1, 3))})
    return UrbanGraph(
        _nodes(coords),
        _edges(coords, rows),
        is_multigraph=False,
        is_directed=True,
        edge_direction_column="oneway",
        crs=CRS,
    )


def test_nearest_objects_ranks_by_cost_and_keeps_table_order():
    graph = undirected_line_graph()  # 0 - 1 - 2 - 3 with times 1, 2, 4
    objects = pd.Series([3, 1, 1], index=["a", "b", "c"])
    kwargs = {"origins_nodes": [0, 2], "gdf_objects": objects.rename("graph_node_id").to_frame()}
    found = nearest_objects(graph, k=2, **kwargs)
    assert found.index.names == ["origin", "rank"]
    assert found.index.tolist() == [(0, 1), (0, 2), (2, 1), (2, 2)]
    assert found["object"].tolist() == ["b", "c", "b", "c"]
    assert found["dist"].tolist() == pytest.approx([1.0, 1.0, 2.0, 2.0])
    three = nearest_objects(graph, k=3, **kwargs)
    assert three.loc[0, "object"].tolist() == ["b", "c", "a"] and three.loc[(0, 3), "dist"] == pytest.approx(7.0)
    assert three.loc[2, "object"].tolist() == ["b", "c", "a"]
    assert nearest_objects(graph, k=3, cutoff=5, **kwargs).loc[0, "object"].tolist() == ["b", "c"]
    by_length = nearest_objects(graph, origins_nodes=[3, 3], object_nodes=[0, 2], weight="length_meter")
    assert by_length["object"].tolist() == [2, 2] and by_length["dist"].tolist() == pytest.approx([10.0, 10.0])
    with pytest.raises(ValueError, match="k must"):
        nearest_objects(graph, k=0, **kwargs)


def test_nearest_objects_follow_the_direction():
    graph = directed_oneway_graph()  # 0 -> 1 <-> 2 -> 3
    assert nearest_objects(graph, origins_nodes=[3], object_nodes=[0]).empty
    back = nearest_objects(graph, origins_nodes=[3], object_nodes=[0], direction="to")
    assert back["dist"].tolist() == pytest.approx([7.0])
    with pytest.raises(ValueError, match="direction"):
        nearest_objects(graph, origins_nodes=[3], object_nodes=[0], direction="both")


def _reference(graph, origins, objects, values, cutoff, direction, decay=None, beta=None):
    """Sums by brute force from the full cost matrix."""
    if direction == "from":
        cost = od_matrix(graph, origins_nodes=origins, destination_nodes=objects).sparse.to_dense().to_numpy()
    else:
        cost = od_matrix(graph, origins_nodes=objects, destination_nodes=origins).sparse.to_dense().to_numpy().T
    weights = (cost <= cutoff).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if decay == "linear":
            weights = weights * (1 - cost / cutoff)
        elif decay == "exp":
            weights = weights * np.exp(-(beta if beta is not None else 4 / cutoff) * cost)
        elif decay == "inverse":
            weights = np.where((cost > 0) & (cost <= cutoff), 1 / cost, 0.0)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0)
    return weights.sum(axis=1), weights @ values


@pytest.mark.parametrize("direction", ["from", "to"])
@pytest.mark.parametrize("decay", [None, "linear", "exp", "inverse"])
@pytest.mark.parametrize("many_origins", [True, False])  # search from the objects, or from the origins
def test_aggregate_objects_match_the_cost_matrix(direction, decay, many_origins):
    graph = random_directed_graph()
    rng = np.random.default_rng(7)
    nodes = graph.nodes_gdf.index.to_numpy()
    origins = nodes if many_origins else rng.choice(nodes, 5, replace=False)
    object_nodes = rng.choice(nodes, 8 if many_origins else 30)
    objects = pd.DataFrame({"graph_node_id": object_nodes, "area": rng.uniform(1, 9, len(object_nodes))})
    result = aggregate_objects(
        graph, 6.0, origins_nodes=origins, gdf_objects=objects, columns=["area"], direction=direction, decay=decay
    )
    count, total = _reference(graph, origins, object_nodes, objects["area"].to_numpy(), 6.0, direction, decay)
    assert result.index.tolist() == list(origins)
    assert result["count_6"].to_numpy() == pytest.approx(count, rel=1e-5, abs=1e-6)
    assert result["area_sum_6"].to_numpy() == pytest.approx(total, rel=1e-5, abs=1e-5)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    assert result["area_mean_6"].to_numpy() == pytest.approx(mean, rel=1e-5, abs=1e-5, nan_ok=True)


def test_aggregate_objects_columns_cutoffs_and_missing_values():
    graph = undirected_line_graph()  # times 1, 2, 4
    objects = pd.DataFrame({"graph_node_id": [1, 1, 3], "seats": [10.0, np.nan, 5.0]}, index=["a", "b", "c"])
    table = aggregate_objects(graph, [1, 7], origins_nodes=[0, 3], gdf_objects=objects, columns=["seats"])
    assert table.columns.tolist() == [
        "count_1",
        "seats_sum_1",
        "seats_mean_1",
        "count_7",
        "seats_sum_7",
        "seats_mean_7",
    ]
    assert table.loc[0].tolist() == pytest.approx([2, 10, 10, 3, 15, 7.5])
    assert table.loc[3].tolist() == pytest.approx([1, 5, 5, 3, 15, 7.5])
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy([1.0, 29.0], [0.0, 0.0]), index=["p", "q"], crs=CRS)
    near = aggregate_objects(graph, 1, gdf_origins=points, object_nodes=[1, 2])
    assert near["count_1"].tolist() == [1.0, 0.0] and near.index.tolist() == ["p", "q"]
    empty = aggregate_objects(graph, 0.5, origins_nodes=[0], gdf_objects=objects, columns=["seats"])
    assert empty["count_0.5"].iloc[0] == 0 and np.isnan(empty["seats_mean_0.5"].iloc[0])
    custom = aggregate_objects(graph, 3, origins_nodes=[0], object_nodes=[2], decay="exp", beta=0.5)
    assert custom["count_3"].iloc[0] == pytest.approx(np.exp(-1.5))
    with pytest.raises(ValueError, match="columns need"):
        aggregate_objects(graph, 1, origins_nodes=[0], object_nodes=[1], columns=["seats"])
    with pytest.raises(KeyError):
        aggregate_objects(graph, 1, origins_nodes=[0], gdf_objects=objects, columns=["beds"])
    with pytest.raises(ValueError, match="decay"):
        aggregate_objects(graph, 1, origins_nodes=[0], object_nodes=[1], decay="cubic")
    with pytest.raises(ValueError, match="beta"):
        aggregate_objects(graph, None, origins_nodes=[0], object_nodes=[1], decay="exp")


def test_diversity_of_categories():
    graph = undirected_line_graph()
    objects = pd.DataFrame(
        {
            "graph_node_id": [1, 1, 2, 2, 3, 3],
            "kind": ["shop", "shop", "cafe", "cafe", "park", None],
            "area": [30.0, 30.0, 10.0, 10.0, 99.0, 99.0],
        }
    )
    table = diversity(graph, [1, 3, 7], "kind", origins_nodes=[0, 3], gdf_objects=objects)
    assert table.loc[0, "richness_1"] == 1 and table.loc[0, "shannon_1"] == pytest.approx(0.0)
    assert table.loc[0, "hill_1"] == pytest.approx(1.0)
    assert table.loc[0, "richness_3"] == 2 and table.loc[0, "shannon_3"] == pytest.approx(np.log(2))
    assert table.loc[0, "hill_3"] == pytest.approx(2.0)
    shares = np.array([2, 2, 1]) / 5
    assert table.loc[0, "richness_7"] == 3 and table.loc[0, "hill_7"] == pytest.approx(1 / (shares**2).sum())
    assert table.loc[3, "richness_1"] == 1  # the park; the object without a category is left out
    by_area = diversity(graph, 3, "kind", origins_nodes=[0], gdf_objects=objects, values="area", hill_order=1)
    shares = np.array([60, 20]) / 80
    entropy = -(shares * np.log(shares)).sum()
    assert by_area["shannon_3"].iloc[0] == pytest.approx(entropy) and by_area["hill_3"].iloc[0] == pytest.approx(
        np.exp(entropy)
    )
    assert diversity(graph, 3, "kind", origins_nodes=[0], gdf_objects=objects, hill_order=0)["hill_3"].iloc[0] == 2
    nothing = diversity(graph, 0.5, "kind", origins_nodes=[0], gdf_objects=objects)
    assert nothing["richness_0.5"].iloc[0] == 0 and np.isnan(nothing["shannon_0.5"].iloc[0])
    with pytest.raises(KeyError):
        diversity(graph, 1, "colour", origins_nodes=[0], gdf_objects=objects)
