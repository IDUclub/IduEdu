import numpy as np
import pytest
from shapely.geometry import LineString

from iduedu import UrbanGraph, simplify_graph
from tests.factories import CRS, _edges, _nodes

pytestmark = pytest.mark.unit

LINE = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (20.0, 0.0), 3: (30.0, 0.0)}


def _graph(coords, rows, *, directed=False, multigraph=True, direction="oneway") -> UrbanGraph:
    for i, row in enumerate(rows):
        row.setdefault("k", 0) if multigraph else None
        row.setdefault("name", "a")
        row.setdefault("time_min", 1.0 + i)
    edges = _edges(coords, rows)
    if directed and direction:
        edges[direction] = edges.get(direction, False)
    return UrbanGraph(
        _nodes(coords),
        edges,
        is_multigraph=multigraph,
        is_directed=directed,
        edge_direction_column=direction if directed else None,
        crs=CRS,
    )


def _ends(graph: UrbanGraph) -> list[tuple]:
    points = graph.nodes_gdf.geometry
    rows = []
    for u, v, geometry in zip(graph.edges_gdf["u"], graph.edges_gdf["v"], graph.edges_gdf.geometry):
        assert geometry.coords[0] == points[u].coords[0] and geometry.coords[-1] == points[v].coords[0]
        rows.append((u, v))
    return rows


def test_undirected_chain_drawn_head_to_head_is_merged_and_turned():
    graph = _graph(LINE, [{"u": 0, "v": 1}, {"u": 2, "v": 1}, {"u": 2, "v": 3}])
    result = simplify_graph(graph)
    assert _ends(result) == [(0, 3)]
    edge = result.edges_gdf.iloc[0]
    assert edge["length_meter"] == 30.0 and edge["time_min"] == 6.0
    assert list(edge.geometry.coords) == [(0, 0), (10, 0), (20, 0), (30, 0)]
    assert list(result.nodes_gdf.index) == [0, 3]


def test_a_change_of_key_breaks_the_chain():
    graph = _graph(LINE, [{"u": 0, "v": 1}, {"u": 1, "v": 2, "name": "b"}, {"u": 2, "v": 3, "name": "b"}])
    assert sorted(_ends(simplify_graph(graph))) == [(0, 1), (1, 3)]
    assert _ends(simplify_graph(graph, keys=[])) == [(0, 3)]


def test_directed_two_way_edges_are_turned_and_one_way_edges_only_join_head_to_tail():
    two_way = _graph(LINE, [{"u": 1, "v": 0}, {"u": 1, "v": 2}, {"u": 3, "v": 2}], directed=True)
    assert _ends(simplify_graph(two_way)) == [(0, 3)] or _ends(simplify_graph(two_way)) == [(3, 0)]
    rows = [{"u": 0, "v": 1, "oneway": True}, {"u": 1, "v": 2, "oneway": True}, {"u": 3, "v": 2, "oneway": True}]
    one_way = simplify_graph(_graph(LINE, rows, directed=True))
    assert sorted(_ends(one_way)) == [(0, 2), (3, 2)]
    mixed = [{"u": 0, "v": 1, "oneway": True}, {"u": 1, "v": 2, "oneway": False}]
    assert sorted(_ends(simplify_graph(_graph(LINE, mixed, directed=True)))) == [(0, 1), (1, 2)]


def test_a_ring_becomes_a_loop_and_kept_nodes_stay():
    square = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (10.0, 10.0), 3: (0.0, 10.0)}
    rows = [{"u": 0, "v": 1}, {"u": 2, "v": 1}, {"u": 2, "v": 3}, {"u": 3, "v": 0}]
    ring = simplify_graph(_graph(square, rows))
    assert _ends(ring) == [(0, 0)] and ring.edges_gdf["length_meter"].item() == 40.0
    kept = simplify_graph(_graph(LINE, [{"u": 0, "v": 1}, {"u": 1, "v": 2}, {"u": 2, "v": 3}]), keep_nodes=[2])
    assert sorted(_ends(kept)) == [(0, 2), (2, 3)]


def test_junctions_stay_and_costs_with_nan_are_summed():
    coords = {**LINE, 4: (10.0, 10.0)}
    rows = [{"u": 0, "v": 1, "time_walk_min": np.nan}, {"u": 1, "v": 2, "time_walk_min": np.nan}, {"u": 1, "v": 4}]
    result = simplify_graph(_graph(coords, rows))
    assert sorted(_ends(result)) == [(0, 1), (1, 2), (1, 4)]
    chain = simplify_graph(
        _graph(LINE, [{"u": 0, "v": 1, "time_walk_min": np.nan}, {"u": 1, "v": 2, "time_walk_min": np.nan}])
    )
    assert np.isnan(chain.edges_gdf["time_walk_min"].item()) and chain.edges_gdf["time_min"].item() == 3.0


def test_a_simple_graph_keeps_parallel_chains_split():
    # Two routes between the junctions 0 and 2, each through one node of degree two.
    coords = {0: (0.0, 0.0), 1: (10.0, 5.0), 2: (20.0, 0.0), 3: (10.0, -5.0), 4: (-10.0, 0.0), 5: (30.0, 0.0)}
    rows = [{"u": 0, "v": 1}, {"u": 1, "v": 2}, {"u": 0, "v": 3}, {"u": 3, "v": 2}, {"u": 4, "v": 0}, {"u": 2, "v": 5}]
    pairs = _ends(simplify_graph(_graph(coords, rows, multigraph=False)))
    assert len(pairs) == 5 and len(set(pairs)) == 5 and (0, 2) in pairs
    multi = simplify_graph(_graph(coords, rows))
    assert sorted(_ends(multi)) == [(0, 2), (0, 2), (2, 5), (4, 0)]
    assert sorted(multi.edges_gdf.loc[multi.edges_gdf["u"].eq(0), "k"]) == [0, 1]


def test_unknown_columns_raise():
    graph = _graph(LINE, [{"u": 0, "v": 1}])
    with pytest.raises(KeyError, match="keys"):
        simplify_graph(graph, keys=["missing"])
    with pytest.raises(KeyError, match="sum_columns"):
        simplify_graph(graph, sum_columns=["missing"])


def test_geometry_of_turned_edges_keeps_inner_points():
    coords = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (20.0, 0.0)}
    rows = [{"u": 0, "v": 1}, {"u": 2, "v": 1, "geometry": LineString([(20, 0), (15, 3), (10, 0)])}]
    result = simplify_graph(_graph(coords, rows))
    assert list(result.edges_gdf.geometry.item().coords) == [(0, 0), (10, 0), (15, 3), (20, 0)]


def test_nodes_walking_reaches_transport_at_stay():
    """A subway station without entrances is a platform between two platform links; merged away, no walk join
    could reach the station."""
    graph = _graph(LINE, [{"u": 1, "v": 0}, {"u": 1, "v": 2}, {"u": 2, "v": 3}])
    graph.nodes_gdf["type"] = ["subway_platform", "platform", "subway", "subway"]
    result = simplify_graph(graph)
    assert 1 in result.nodes_gdf.index and 2 not in result.nodes_gdf.index
