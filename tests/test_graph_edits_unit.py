import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from iduedu import (
    GraphEdits,
    UrbanGraph,
    UrbanGraphChanges,
    apply_urban_graph_changes,
    relabel_urban_graph,
    resolve_graph_edits,
)
from iduedu.graph.editors import coordinate_node_ids
from tests.factories import (
    CRS,
    _edges,
    _nodes,
    directed_oneway_graph,
    multigraph_with_parallel_edges,
    undirected_line_graph,
)

pytestmark = pytest.mark.unit


def _lines(*coords, crs=CRS, **columns) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns, geometry=[LineString(c) for c in coords], crs=crs)


def _apply(graph: UrbanGraph, edits: GraphEdits, **kwargs) -> tuple[UrbanGraph, pd.DataFrame]:
    changes, report = resolve_graph_edits(graph, edits, **kwargs)
    return apply_urban_graph_changes(graph, changes), report


def _edge(graph: UrbanGraph, u, v) -> pd.DataFrame:
    edges = graph.edges_gdf
    return edges[(edges["u"] == u) & (edges["v"] == v)]


def _ends(graph: UrbanGraph) -> set[tuple[tuple[float, float], tuple[float, float]]]:
    return {(g.coords[0], g.coords[-1]) for g in graph.edges_gdf.geometry}


# ---------------------------------------------------------------------------
# edges_to_update
# ---------------------------------------------------------------------------


def test_edges_to_update_sets_values_keeps_missing_and_adds_columns():
    graph = undirected_line_graph()
    updates = pd.DataFrame({"u": [1, 2], "v": [2, 3], "time_min": [9.0, np.nan], "closed": [True, np.nan]})
    result = apply_urban_graph_changes(graph, UrbanGraphChanges(edges_to_update=updates))
    assert _edge(result, 1, 2)[["time_min", "closed"]].values.tolist() == [[9.0, True]]
    assert _edge(result, 2, 3)["time_min"].item() == 4.0 and pd.isna(_edge(result, 2, 3)["closed"].item())
    assert _edge(result, 0, 1)["time_min"].item() == 1.0
    assert len(result.edges_gdf) == 3 and "closed" not in graph.edges_gdf.columns


def test_edges_to_update_by_multigraph_key():
    graph = multigraph_with_parallel_edges()
    updates = pd.DataFrame({"u": [0], "v": [1], "k": [1], "time_min": [7.0]})
    result = apply_urban_graph_changes(graph, UrbanGraphChanges(edges_to_update=updates, is_multigraph=True))
    assert sorted(_edge(result, 0, 1)["time_min"]) == [5.0, 7.0]


def test_edges_to_update_rejects_missing_and_deleted_keys():
    graph = undirected_line_graph()
    with pytest.raises(ValueError, match="not present"):
        apply_urban_graph_changes(
            graph, UrbanGraphChanges(edges_to_update=pd.DataFrame({"u": [5], "v": [6], "time_min": [1.0]}))
        )
    with pytest.raises(ValueError, match="both deleted and updated"):
        apply_urban_graph_changes(
            graph,
            UrbanGraphChanges(
                edges_to_delete=pd.DataFrame({"u": [0], "v": [1]}),
                edges_to_update=pd.DataFrame({"u": [0], "v": [1], "time_min": [1.0]}),
            ),
        )


@pytest.mark.parametrize(
    "table, message",
    [
        (pd.DataFrame({"u": [0, 0], "v": [1, 1], "time_min": [1.0, 2.0]}), "unique"),
        (pd.DataFrame({"u": [0], "v": [1]}), "no value columns"),
        (pd.DataFrame({"u": [0], "v": [np.nan], "time_min": [1.0]}), "NaN"),
        (pd.DataFrame({"u": [0], "time_min": [1.0]}), "missing required"),
        (pd.DataFrame({"u": [0], "v": [1], "geometry": [None]}), "cannot change"),
    ],
)
def test_edges_to_update_validation(table, message):
    with pytest.raises(ValueError, match=message):
        UrbanGraphChanges(edges_to_update=table)


# ---------------------------------------------------------------------------
# relabel by coordinates
# ---------------------------------------------------------------------------


def test_relabel_by_coordinates_gives_equal_ids_for_equal_points():
    graph = relabel_urban_graph(undirected_line_graph(), by="coordinates")
    other = relabel_urban_graph(directed_oneway_graph(), by="coordinates")
    assert graph.nodes_gdf.index.tolist() == other.nodes_gdf.index.tolist()
    assert graph.nodes_gdf.index[0] == 90 * 10**7 * (360 * 10**7 + 1) + 180 * 10**7  # (0, 0)
    assert set(graph.edges_gdf["u"]) | set(graph.edges_gdf["v"]) == set(graph.nodes_gdf.index)
    lat, lon = np.divmod(graph.nodes_gdf.index.to_numpy(), 360 * 10**7 + 1)
    lonlat = graph.nodes_gdf.to_crs(4326).geometry
    assert np.allclose((lon - 180 * 10**7) / 1e7, lonlat.x, atol=1e-7)
    assert np.allclose((lat - 90 * 10**7) / 1e7, lonlat.y, atol=1e-7)


def test_relabel_by_coordinates_rejects_coincident_nodes():
    coords = {0: (0.0, 0.0), 1: (0.001, 0.0), 2: (10.0, 0.0)}
    edges = _edges(coords, [{"u": 0, "v": 2}, {"u": 1, "v": 2}])
    graph = UrbanGraph(_nodes(coords), edges, is_multigraph=False, is_directed=False, crs=CRS)
    with pytest.raises(ValueError, match="share a coordinate id"):
        relabel_urban_graph(graph, by="coordinates")
    with pytest.raises(ValueError, match="by must be"):
        relabel_urban_graph(graph, by="osm")


# ---------------------------------------------------------------------------
# resolve_graph_edits: anchors
# ---------------------------------------------------------------------------


def test_remove_part_of_an_edge_splits_it_at_the_anchor_ends():
    graph, report = _apply(undirected_line_graph(), GraphEdits(remove=_lines([(12, 1), (18, 1)])), tolerance=1.5)
    assert report.empty
    assert _ends(graph) == {((0, 0), (10, 0)), ((10, 0), (12, 0)), ((18, 0), (20, 0)), ((20, 0), (30, 0))}
    piece = graph.edges_gdf[graph.edges_gdf["u"] == 1]
    assert piece["length_meter"].item() == pytest.approx(2.0) and piece["time_min"].item() == pytest.approx(0.4)


def test_anchor_over_several_edges_selects_all_of_them():
    graph, _ = _apply(undirected_line_graph(), GraphEdits(remove=_lines([(0, 0.5), (20, 0.5)])), tolerance=1.0)
    assert _ends(graph) == {((20, 0), (30, 0))}


def test_update_by_speed_sets_travel_time_from_length():
    edits = GraphEdits(update=_lines([(0, 0), (10, 0)], speed_kmh=[6.0]))
    changes, _ = resolve_graph_edits(undirected_line_graph(), edits, tolerance=1.0)
    assert changes.edges_gdf is None and changes.edges_to_delete.empty
    assert changes.edges_to_update.to_dict("records") == [{"u": 0, "v": 1, "time_min": 0.1}]


def test_update_keeps_old_values_where_the_edit_has_none():
    edits = GraphEdits(update=_lines([(0, 0), (10, 0)], [(20, 0), (30, 0)], highway=["primary", None], lanes=[None, 2]))
    graph, _ = _apply(undirected_line_graph(), edits, tolerance=1.0)
    assert _edge(graph, 0, 1)["highway"].item() == "primary" and pd.isna(_edge(graph, 0, 1)["lanes"].item())
    assert _edge(graph, 2, 3)["lanes"].item() == 2 and pd.isna(_edge(graph, 2, 3)["highway"].item())


def test_crossing_street_is_not_selected():
    coords = {0: (0.0, 0.0), 1: (20.0, 0.0), 2: (10.0, -10.0), 3: (10.0, 10.0)}
    edges = _edges(coords, [{"u": 0, "v": 1}, {"u": 2, "v": 3}])
    graph = UrbanGraph(_nodes(coords), edges, is_multigraph=False, is_directed=False, crs=CRS)
    result, report = _apply(graph, GraphEdits(remove=_lines([(0, 0), (20, 0)])), tolerance=2.0)
    assert report.empty and _ends(result) == {((10, -10), (10, 10))}


def test_unbound_anchors_are_reported():
    edits = GraphEdits(
        remove=_lines([(0, 50), (10, 50)], [(25, 0), (60, 0)]),
        update=_lines([(0, 0), (10, 0)], time_min=[np.nan]),  # no values: skipped silently
    )
    changes, report = resolve_graph_edits(undirected_line_graph(), edits, tolerance=1.0)
    assert report[["kind", "edit", "reason"]].values.tolist() == [
        ["remove", 0, "no edges along the anchor"],
        ["remove", 1, "anchor not covered by the graph"],
    ]
    assert report["coverage"].tolist() == [0.0, pytest.approx(5 / 35, abs=1e-3)]
    assert changes.edges_gdf is None and changes.edges_to_delete.empty and changes.edges_to_update is None


def test_min_coverage_accepts_a_partly_covered_anchor():
    _, report = _apply(undirected_line_graph(), GraphEdits(remove=_lines([(25, 0), (60, 0)])), min_coverage=0.1)
    assert report.empty


# ---------------------------------------------------------------------------
# resolve_graph_edits: directions
# ---------------------------------------------------------------------------


def test_remove_one_direction_of_a_two_way_edge():
    edits = GraphEdits(remove=_lines([(20, 0), (10, 0)], direction=["along"]))
    graph, _ = _apply(directed_oneway_graph(), edits, tolerance=1.0)
    row = _edge(graph, 1, 2)
    assert len(row) == 1 and bool(row["oneway"].item()) and _edge(graph, 2, 1).empty


def test_update_one_direction_of_part_of_a_two_way_edge():
    edits = GraphEdits(update=_lines([(12, 0), (20, 0)], direction=["along"], time_min=[9.0]))
    graph, _ = _apply(directed_oneway_graph(), edits, tolerance=1.0)
    new = graph.nodes_gdf.index.difference(directed_oneway_graph().nodes_gdf.index).item()
    assert _edge(graph, 1, new)["oneway"].item() is False or not _edge(graph, 1, new)["oneway"].item()
    assert _edge(graph, new, 2)[["time_min", "oneway"]].values.tolist() == [[9.0, True]]
    assert _edge(graph, 2, new)[["time_min", "oneway"]].values.tolist() == [[pytest.approx(1.6), True]]
    assert _edge(graph, 2, new).geometry.item().coords[0] == (20.0, 0.0)


def test_along_anchor_against_a_one_way_edge_changes_nothing():
    edits = GraphEdits(remove=_lines([(3, 0), (0, 0)], direction=["along"]))
    changes, report = resolve_graph_edits(directed_oneway_graph(), edits, tolerance=1.0)
    assert report["reason"].tolist() == ["no edges along the anchor"]
    assert changes.edges_gdf is None and changes.nodes_gdf is None


def test_both_directions_of_a_directed_pair_split_at_one_node():
    coords = {0: (0.0, 0.0), 1: (10.0, 0.0)}
    edges = _edges(coords, [{"u": 0, "v": 1}, {"u": 1, "v": 0}])
    graph = UrbanGraph(_nodes(coords), edges, is_multigraph=False, is_directed=True, crs=CRS)
    result, _ = _apply(graph, GraphEdits(remove=_lines([(4, 0), (10, 0)])), tolerance=1.0)
    assert len(result.nodes_gdf) == 3
    assert _ends(result) == {((0, 0), (4, 0)), ((4, 0), (0, 0))}


def test_parallel_edges_of_a_multigraph_split_with_unique_keys():
    edits = GraphEdits(update=_lines([(0, 0), (5, 0)], closed=[True]))
    graph, _ = _apply(multigraph_with_parallel_edges(), edits, tolerance=1.0)
    assert not graph.edges_gdf[["u", "v", "k"]].duplicated().any()
    closed = graph.edges_gdf[graph.edges_gdf["closed"].eq(True)]
    assert len(closed) == 2 and sorted(closed["time_min"]) == [0.5, 2.5]


# ---------------------------------------------------------------------------
# resolve_graph_edits: added lines
# ---------------------------------------------------------------------------


def test_added_line_attaches_to_a_node_and_splits_an_edge():
    edits = GraphEdits(add=_lines([(10.5, 0.5), (15, 10)], [(25, 1), (15, 10)], highway=["path", "path"]))
    graph, report = _apply(undirected_line_graph(), edits, tolerance=1.5, speed_kmh=6)
    assert report.empty
    assert len(graph.nodes_gdf) == 6  # the shared far end and the split point at x = 25
    added = graph.edges_gdf[graph.edges_gdf["highway"].eq("path")]
    assert {g.coords[0] for g in added.geometry} == {(10.0, 0.0), (25.0, 0.0)}
    assert added["time_min"].tolist() == pytest.approx(list(added["length_meter"] / 100.0), abs=1e-3)
    assert _ends(graph) >= {((20, 0), (25, 0)), ((25, 0), (30, 0))}


def test_added_line_takes_given_time_and_one_way_direction():
    edits = GraphEdits(add=_lines([(5, 0), (5, 10)], oneway=[True], time_min=[3.0]))
    graph, _ = _apply(directed_oneway_graph(), edits, tolerance=1.0)
    added = graph.edges_gdf[graph.edges_gdf["time_min"].eq(3.0)]
    assert len(added) == 1 and bool(added["oneway"].item()) and added["length_meter"].item() == 10.0


def test_two_way_line_in_a_directed_graph_without_direction_column_gives_two_edges():
    coords = {0: (0.0, 0.0), 1: (10.0, 0.0)}
    graph = UrbanGraph(_nodes(coords), _edges(coords, [{"u": 0, "v": 1}]), is_multigraph=False, is_directed=True)
    graph = UrbanGraph(graph.nodes_gdf, graph.edges_gdf, False, True, crs=CRS)
    result, _ = _apply(graph, GraphEdits(add=_lines([(10, 0), (10, 10)])), tolerance=1.0, speed_kmh=6)
    assert {((10, 0), (10, 10)), ((10, 10), (10, 0))} <= _ends(result)


def test_unbound_added_lines_are_reported():
    edits = GraphEdits(add=_lines([(0, 0), (0, 50)], [(0, 0.5), (0.5, 0.5)], [(0, 0), (10, 0)]))
    changes, report = resolve_graph_edits(undirected_line_graph(), edits, tolerance=1.0)
    assert report["reason"].tolist() == ["no time_min or speed", "no time_min or speed", "no time_min or speed"]
    _, report = resolve_graph_edits(undirected_line_graph(), edits, tolerance=1.0, speed_kmh=5)
    assert report[["edit", "reason"]].values.tolist() == [[1, "both ends at one node"], [2, "edge already exists"]]
    assert changes.edges_gdf is None


def test_edits_in_another_crs_are_reprojected():
    remove = _lines([(12, 0), (18, 0)]).to_crs(4326)
    graph, report = _apply(undirected_line_graph(), GraphEdits(remove=remove), tolerance=1.0)
    assert report.empty and ((10, 0), (12, 0)) in {
        (tuple(map(round, a)), tuple(map(round, b))) for a, b in _ends(graph)
    }


# ---------------------------------------------------------------------------
# node ids
# ---------------------------------------------------------------------------


def test_new_nodes_by_coordinates_match_a_rebuilt_graph():
    graph = relabel_urban_graph(undirected_line_graph(), by="coordinates")
    changes, _ = resolve_graph_edits(
        graph, GraphEdits(remove=_lines([(12, 0), (18, 0)])), tolerance=1.0, node_ids="coordinates"
    )
    expected = coordinate_node_ids(gpd.GeoSeries([Point(12, 0), Point(18, 0)], crs=CRS))
    assert sorted(changes.nodes_gdf.index) == sorted(expected)
    apply_urban_graph_changes(graph, changes)


def test_new_node_id_collision_raises():
    graph = undirected_line_graph()
    taken = int(coordinate_node_ids(gpd.GeoSeries([Point(15, 40)], crs=CRS))[0])
    graph = UrbanGraph(
        graph.nodes_gdf.rename(index={3: taken}),
        graph.edges_gdf.replace({"u": {3: taken}, "v": {3: taken}}),
        False,
        False,
        crs=CRS,
    )
    edits = GraphEdits(add=_lines([(0, 0), (15, 40)]))
    with pytest.raises(ValueError, match="collide"):
        resolve_graph_edits(graph, edits, speed_kmh=5, node_ids="coordinates")


def test_next_ids_need_integer_node_ids():
    graph = undirected_line_graph()
    named = UrbanGraph(
        graph.nodes_gdf.rename(index=lambda i: f"n{i}"),
        graph.edges_gdf.assign(u=lambda e: "n" + e["u"].astype(str), v=lambda e: "n" + e["v"].astype(str)),
        False,
        False,
        crs=CRS,
    )
    with pytest.raises(ValueError, match="integer node ids"):
        resolve_graph_edits(named, GraphEdits(remove=_lines([(12, 0), (18, 0)])), tolerance=1.0)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"remove": pd.DataFrame({"geometry": [LineString([(0, 0), (1, 0)])]})}, TypeError, "GeoDataFrame"),
        ({"remove": gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])])}, ValueError, "CRS"),
        ({"add": gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=CRS)}, ValueError, "LineString"),
        ({"add": _lines([(0, 0), (1, 0)], u=[1])}, ValueError, "cannot set"),
        ({"remove": _lines([(0, 0), (1, 0)], direction=["left"])}, ValueError, "direction"),
        ({"update": _lines([(0, 0), (1, 0)])}, ValueError, "no value columns"),
        ({"remove": _lines([(0, 0), (1, 0)], [(1, 0), (2, 0)]).set_axis([0, 0])}, ValueError, "unique"),
    ],
)
def test_graph_edits_validation(kwargs, error, message):
    with pytest.raises(error, match=message):
        GraphEdits(**kwargs)


def test_resolve_needs_a_projected_graph_and_valid_parameters():
    edits = GraphEdits(remove=_lines([(0, 0), (1, 0)]))
    geographic = undirected_line_graph()
    geographic = UrbanGraph(geographic.nodes_gdf.to_crs(4326), geographic.edges_gdf.to_crs(4326), False, False)
    with pytest.raises(ValueError, match="projected CRS"):
        resolve_graph_edits(geographic, edits)
    for kwargs in ({"tolerance": 0}, {"speed_kmh": -1}, {"min_coverage": 0}, {"node_ids": "osm"}):
        with pytest.raises(ValueError):
            resolve_graph_edits(undirected_line_graph(), edits, **kwargs)


def test_edge_types_keep_other_layers_along_the_anchor():
    coords = {0: (0.0, 0.0), 1: (20.0, 0.0), 2: (0.0, 1.0), 3: (20.0, 1.0)}
    edges = _edges(coords, [{"u": 0, "v": 1, "type": "walk"}, {"u": 2, "v": 3, "type": "bus"}])
    graph = UrbanGraph(_nodes(coords), edges, is_multigraph=False, is_directed=False, crs=CRS)
    result, _ = _apply(graph, GraphEdits(remove=_lines([(0, 0.5), (20, 0.5)])), tolerance=2.0, edge_types=["walk"])
    assert result.edges_gdf["type"].tolist() == ["bus"]
    added, _ = _apply(graph, GraphEdits(add=_lines([(20, 0.8), (20, 30)])), speed_kmh=5, edge_types=["walk"])
    assert _edge(added, 1, 4).shape[0] == 1  # attached to the walk node, not to the nearer bus node
    with pytest.raises(KeyError, match="type"):
        resolve_graph_edits(undirected_line_graph(), GraphEdits(remove=_lines([(0, 0), (1, 0)])), edge_types=["walk"])
