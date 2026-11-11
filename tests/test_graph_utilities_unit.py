import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from iduedu import UrbanGraph, consolidate_intersections, nearest_edges, unique_segments
from tests.factories import CRS, _edges, _nodes, undirected_line_graph

pytestmark = pytest.mark.unit


def _points(*xy, crs=CRS, index=None):
    return gpd.GeoDataFrame(geometry=[Point(p) for p in xy], crs=crs, index=index)


# --------------------------------------------------------------------------- nearest_edges


def test_nearest_edges_returns_edge_distance_and_position():
    graph = undirected_line_graph()
    result = nearest_edges(graph, _points((12.0, 3.0), (29.0, -4.0), index=["a", "b"]))
    assert result.index.tolist() == ["a", "b"]
    assert result[["u", "v"]].values.tolist() == [[1, 2], [2, 3]]
    assert result["dist"].tolist() == pytest.approx([3.0, 4.0])
    assert result["along"].tolist() == pytest.approx([2.0, 9.0])
    assert str(result["u"].dtype) == "Int64"


def test_nearest_edges_leaves_far_objects_missing_and_reprojects():
    graph = undirected_line_graph()
    far = nearest_edges(graph, _points((15.0, 1.0), (15.0, 50.0)), max_dist=5)
    assert far["u"].tolist()[0] == 1 and far.iloc[1].isna().all()
    lonlat = _points((15.0, 1.0)).to_crs(4326)
    assert nearest_edges(graph, lonlat)["u"].iloc[0] == 1


def test_nearest_edges_checks_inputs():
    graph = undirected_line_graph()
    with pytest.raises(TypeError):
        nearest_edges(graph, pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="max_dist"):
        nearest_edges(graph, _points((0.0, 0.0)), max_dist=-1)
    with pytest.raises(ValueError, match="CRS"):
        nearest_edges(graph, gpd.GeoDataFrame(geometry=[Point(0, 0)]))


# --------------------------------------------------------------------------- unique_segments


def test_unique_segments_count_shared_pieces_once():
    coords = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (20.0, 0.0), 3: (10.0, 5.0)}
    there = LineString([(0, 0), (10, 0), (20, 0)])
    back = LineString([(20, 0), (10, 0), (10, 5), (0, 0)])  # shares only 10-20 with "there"
    separate = LineString([(0, 1), (20, 1)])  # its own points: a separate carriageway
    edges = _edges(
        coords,
        [
            {"u": 0, "v": 2, "k": 0, "geometry": there},
            {"u": 2, "v": 0, "k": 0, "geometry": back},
            {"u": 0, "v": 2, "k": 1, "geometry": separate},
            {"u": 0, "v": 2, "k": 2, "geometry": LineString([(0, 0), (0, 0), (10, 0)])},
        ],
    )
    graph = UrbanGraph(_nodes(coords), edges, is_multigraph=True, is_directed=True, crs=CRS)
    segments = unique_segments(graph)
    expected = 10 + 10 + np.hypot(10, 5) + 20  # 0-10, 10-20, 10,5-0,0 and the separate line
    assert segments["length"].sum() == pytest.approx(expected + 5)  # plus the 10,0-10,5 piece of "back"
    # 0-10 is in "there" and in the zero-length edge, 10-20 in "there" and "back"; the rest once.
    assert sorted(segments["edge_count"]) == [1, 1, 1, 2, 2]
    assert set(segments["edge"]) <= set(edges.index)
    assert segments.crs == graph.crs


def test_unique_segments_rejects_geographic_crs_and_bad_precision():
    graph = undirected_line_graph()
    with pytest.raises(ValueError, match="precision"):
        unique_segments(graph, precision=0)
    graph.edges_gdf = graph.edges_gdf.to_crs(4326)
    graph.nodes_gdf = graph.nodes_gdf.to_crs(4326)
    with pytest.raises(ValueError, match="projected"):
        unique_segments(graph)


# --------------------------------------------------------------------------- consolidate_intersections


def _junction_graph():
    # Four nodes of one junction (0-3) within 6 m, two roads leaving it, and a bridge node (9) 4 m above node 4
    # that no edge inside the cluster connects to it.
    coords = {
        0: (0.0, 0.0),
        1: (6.0, 0.0),
        2: (6.0, 6.0),
        3: (0.0, 6.0),
        4: (100.0, 0.0),
        5: (-100.0, 6.0),
        9: (100.0, 4.0),
        10: (100.0, 200.0),
    }
    rows = [
        {"u": 0, "v": 1},
        {"u": 1, "v": 2},
        {"u": 2, "v": 3},
        {"u": 3, "v": 0},
        {"u": 1, "v": 4},
        {"u": 5, "v": 3},
        {"u": 9, "v": 10},
    ]
    edges = _edges(
        coords, [{**row, "time_min": 2.0 * LineString([coords[row["u"]], coords[row["v"]]]).length} for row in rows]
    )
    return UrbanGraph(_nodes(coords), edges, is_multigraph=False, is_directed=False, crs=CRS, graph_type="walk")


def test_consolidate_merges_a_junction_and_keeps_a_bridge_apart():
    graph = _junction_graph()
    result, mapping = consolidate_intersections(graph, 8.0, return_mapping=True)
    assert mapping.loc[[0, 1, 2, 3]].tolist() == [0, 0, 0, 0]
    assert mapping.loc[[4, 9]].tolist() == [4, 9]  # 4 m apart but not joined by an edge inside the cluster
    assert sorted(result.nodes_gdf.index) == [0, 4, 5, 9, 10]
    assert result.nodes_gdf.geometry.loc[0].coords[0] == pytest.approx((3.0, 3.0))
    assert sorted(map(tuple, result.edges_gdf[["u", "v"]].to_numpy().tolist())) == [(0, 4), (5, 0), (9, 10)]
    road = result.edges_gdf.set_index(["u", "v"]).loc[(0, 4)]
    assert road.geometry.coords[0] == pytest.approx((3.0, 3.0))
    assert road["length_meter"] == pytest.approx(np.hypot(97, 3))
    assert road["time_min"] == pytest.approx(2.0 * np.hypot(97, 3))  # scaled with the new length
    assert not result.is_multigraph and result.type == "walk"


def test_consolidate_respects_max_span_and_tolerance_zero():
    graph = _junction_graph()
    untouched = consolidate_intersections(graph, 8.0, max_span=5.0)
    assert len(untouched.nodes_gdf) == len(graph.nodes_gdf)
    assert len(consolidate_intersections(graph, 0.0).nodes_gdf) == len(graph.nodes_gdf)


def test_consolidate_keeps_lighter_parallel_edge_of_a_simple_graph():
    coords = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (50.0, 0.0)}
    edges = _edges(coords, [{"u": 0, "v": 1}, {"u": 0, "v": 2, "time_min": 99.0}, {"u": 1, "v": 2, "time_min": 1.0}])
    graph = UrbanGraph(_nodes(coords), edges, is_multigraph=False, is_directed=False, crs=CRS)
    result = consolidate_intersections(graph, 5.0)
    assert len(result.edges_gdf) == 1 and not result.is_multigraph
    assert result.edges_gdf["time_min"].iloc[0] < 2.0


def test_consolidate_rejects_bad_input():
    graph = _junction_graph()
    with pytest.raises(TypeError):
        consolidate_intersections("graph")
    with pytest.raises(ValueError, match="tolerance"):
        consolidate_intersections(graph, -1)
