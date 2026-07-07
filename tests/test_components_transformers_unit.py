import pytest

from iduedu.graph.components import (
    connected_components,
    largest_component,
    strongly_connected_components,
    weakly_connected_components,
)
from iduedu.graph.transformers import keep_largest_connected_component, simplify_multiedges, to_directed, to_undirected
from tests.factories import (
    directed_oneway_graph,
    disconnected_graph,
    multigraph_with_parallel_edges,
    undirected_line_graph,
)

pytestmark = pytest.mark.unit


def test_connected_components_on_disconnected_undirected_graph_are_sorted_by_size():
    graph = disconnected_graph()

    assert connected_components(graph) == [{0, 1, 2}, {10, 11}]
    assert largest_component(graph) == {0, 1, 2}


def test_connected_components_rejects_directed_graph():
    graph = directed_oneway_graph()

    with pytest.raises(ValueError, match="directed"):
        connected_components(graph)


def test_weak_and_strong_components_respect_direction():
    graph = directed_oneway_graph()

    assert weakly_connected_components(graph) == [{0, 1, 2, 3}]
    assert strongly_connected_components(graph) == [{1, 2}, {0}, {3}]
    assert largest_component(graph) == {1, 2}


def test_keep_largest_connected_component_preserves_graph_contract():
    graph = disconnected_graph()
    largest = keep_largest_connected_component(graph)

    assert set(largest.nodes_gdf.index) == {0, 1, 2}
    assert set(largest.edges_gdf["u"]) <= {0, 1, 2}
    assert set(largest.edges_gdf["v"]) <= {0, 1, 2}
    assert largest.crs == graph.crs


def test_to_directed_adds_default_direction_column_and_to_undirected_ignores_it():
    graph = undirected_line_graph()
    directed = to_directed(graph, edge_direction_column="oneway", default_direction_value=False)

    assert directed.is_directed
    assert directed.edge_direction_column == "oneway"
    assert directed.edges_gdf["oneway"].tolist() == [False, False, False]

    undirected = to_undirected(directed)
    assert not undirected.is_directed
    assert undirected.edge_direction_column is None
    assert "oneway" in undirected.edges_gdf.columns


def test_simplify_multiedges_selects_weighted_edge_and_drops_key():
    graph = multigraph_with_parallel_edges()

    simplified = simplify_multiedges(graph, weight="time_min", rule="min")

    assert not simplified.is_multigraph
    assert "k" not in simplified.edges_gdf.columns
    edge_01 = simplified.edges_gdf[(simplified.edges_gdf["u"] == 0) & (simplified.edges_gdf["v"] == 1)].iloc[0]
    assert edge_01["time_min"] == 1.0
