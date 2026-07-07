import pytest
from shapely.geometry import LineString

from iduedu.graph.adapters import nx_graph2urban_graph, urban_graph2nx_graph
from tests.factories import undirected_line_graph

nx = pytest.importorskip("networkx")

pytestmark = pytest.mark.unit

CRS = "EPSG:3857"


def _nx_line_graph(graph_cls=nx.Graph, *, with_oneway=False):
    graph = graph_cls()
    graph.graph["crs"] = CRS
    graph.add_node(0, x=0.0, y=0.0)
    graph.add_node(1, x=10.0, y=0.0)
    edge_attrs = {
        "geometry": LineString([(0.0, 0.0), (10.0, 0.0)]),
        "length_meter": 10.0,
        "time_min": 1.0,
    }
    if with_oneway:
        edge_attrs["oneway"] = True
    if graph.is_multigraph():
        graph.add_edge(0, 1, key=0, **edge_attrs)
    else:
        graph.add_edge(0, 1, **edge_attrs)
    return graph


def test_urban_graph2nx_graph_preserves_nodes_edges_and_attrs():
    graph = undirected_line_graph()

    nx_graph = urban_graph2nx_graph(graph)

    assert isinstance(nx_graph, nx.Graph) and not nx_graph.is_directed()
    assert nx_graph.number_of_nodes() == 4
    assert nx_graph.number_of_edges() == 3
    assert nx_graph.nodes[0]["x"] == 0.0
    assert nx_graph[0][1]["time_min"] == 1.0
    assert nx_graph.graph["crs"] is not None


def test_round_trip_nx_preserves_structure_and_weights():
    graph = undirected_line_graph()

    restored = nx_graph2urban_graph(urban_graph2nx_graph(graph))

    assert len(restored.nodes_gdf) == 4
    assert len(restored.edges_gdf) == 3
    assert not restored.is_multigraph
    assert not restored.is_directed
    edge_01 = restored.edges_gdf[(restored.edges_gdf["u"] == 0) & (restored.edges_gdf["v"] == 1)].iloc[0]
    assert edge_01["time_min"] == 1.0


def test_oneway_column_makes_graph_directed():
    nx_graph = _nx_line_graph(nx.MultiGraph, with_oneway=True)

    urban = nx_graph2urban_graph(nx_graph)

    assert urban.edge_direction_column == "oneway"
    assert urban.is_directed


def test_directed_nx_graph_emits_warning():
    nx_graph = _nx_line_graph(nx.DiGraph)

    with pytest.warns(UserWarning, match="Directed NetworkX graphs"):
        nx_graph2urban_graph(nx_graph)


def test_nx_graph2urban_graph_rejects_non_graph():
    with pytest.raises(TypeError, match="nx.Graph"):
        nx_graph2urban_graph("not-a-graph")


def test_nx_graph2urban_graph_requires_crs():
    graph = nx.Graph()  # no crs attribute set

    with pytest.raises(ValueError, match="crs"):
        nx_graph2urban_graph(graph)


def test_nx_graph2urban_graph_requires_nodes_and_edges():
    no_nodes = nx.Graph()
    no_nodes.graph["crs"] = CRS
    with pytest.raises(ValueError, match="no nodes"):
        nx_graph2urban_graph(no_nodes)

    no_edges = nx.Graph()
    no_edges.graph["crs"] = CRS
    no_edges.add_node(0, x=0.0, y=0.0)
    with pytest.raises(ValueError, match="no edges"):
        nx_graph2urban_graph(no_edges)


def test_nx_graph2urban_graph_requires_node_coordinates():
    graph = nx.Graph()
    graph.graph["crs"] = CRS
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, geometry=LineString([(0, 0), (1, 0)]), length_meter=1.0, time_min=1.0)

    with pytest.raises(ValueError, match="required 'x'"):
        nx_graph2urban_graph(graph)
