import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from iduedu.graph.editors import (
    UrbanGraphChanges,
    apply_urban_graph_changes,
    clip_urban_graph,
    join_urban_graphs,
    project_objects2urban_graph,
    relabel_urban_graph,
    subgraph_by_nodes,
)
from iduedu.graph.urban_graph import UrbanGraph
from tests.factories import CRS, _edges, _nodes, disconnected_graph, undirected_line_graph

pytestmark = pytest.mark.unit

WALK_SPEED_M_PER_MIN = 100.0


def _two_node_graph(coords, edge_rows, *, crs=CRS, tag=None) -> UrbanGraph:
    nodes = _nodes(coords)
    if tag is not None:
        nodes["tag"] = tag
    edges = _edges(coords, edge_rows, crs=crs)
    return UrbanGraph(nodes, edges, is_multigraph=False, is_directed=False, crs=crs, graph_type="walk")


# ---------------------------------------------------------------------------
# relabel / subgraph / clip
# ---------------------------------------------------------------------------


def test_relabel_urban_graph_remaps_sparse_node_ids_to_range_index():
    graph = disconnected_graph()  # node ids: 0, 1, 2, 10, 11

    relabeled = relabel_urban_graph(graph)

    assert relabeled.nodes_gdf.index.tolist() == [0, 1, 2, 3, 4]
    edge_pairs = set(zip(relabeled.edges_gdf["u"], relabeled.edges_gdf["v"]))
    assert (3, 4) in edge_pairs  # old (10, 11) edge
    # Source graph is left untouched.
    assert graph.nodes_gdf.index.tolist() == [0, 1, 2, 10, 11]


def test_subgraph_by_nodes_keeps_only_internal_edges():
    graph = undirected_line_graph()

    sub = subgraph_by_nodes(graph, [0, 1, 2])

    assert set(sub.nodes_gdf.index) == {0, 1, 2}
    edge_pairs = set(zip(sub.edges_gdf["u"], sub.edges_gdf["v"]))
    assert edge_pairs == {(0, 1), (1, 2)}  # (2, 3) dropped


def test_subgraph_by_nodes_rejects_unknown_node():
    graph = undirected_line_graph()

    with pytest.raises(ValueError, match="absent in graph"):
        subgraph_by_nodes(graph, [0, 999])


def test_clip_urban_graph_keeps_nodes_inside_polygon():
    graph = undirected_line_graph()  # nodes at x = 0, 10, 20, 30 (y = 0)
    polygon = box(-5.0, -5.0, 15.0, 5.0)

    clipped = clip_urban_graph(graph, polygon)

    assert set(clipped.nodes_gdf.index) == {0, 1}
    edge_pairs = set(zip(clipped.edges_gdf["u"], clipped.edges_gdf["v"]))
    assert edge_pairs == {(0, 1)}


def test_clip_urban_graph_rejects_non_geometry_polygon():
    graph = undirected_line_graph()

    with pytest.raises(TypeError, match="shapely geometry"):
        clip_urban_graph(graph, "not-a-polygon")


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------


def test_join_urban_graphs_merges_nodes_and_edges():
    left = _two_node_graph({0: (0.0, 0.0), 1: (10.0, 0.0)}, [{"u": 0, "v": 1}])
    right = _two_node_graph({1: (10.0, 0.0), 2: (20.0, 0.0)}, [{"u": 1, "v": 2}])

    joined = join_urban_graphs(left, right, graph_type="merged")

    assert set(joined.nodes_gdf.index) == {0, 1, 2}
    assert set(zip(joined.edges_gdf["u"], joined.edges_gdf["v"])) == {(0, 1), (1, 2)}
    assert joined.type == "merged"


def test_join_urban_graphs_node_conflict_selects_winning_side():
    left = _two_node_graph({0: (0.0, 0.0), 1: (10.0, 0.0)}, [{"u": 0, "v": 1}], tag="L")
    right = _two_node_graph({1: (10.0, 0.0), 2: (20.0, 0.0)}, [{"u": 1, "v": 2}], tag="R")

    keep_left = join_urban_graphs(left, right, node_conflict="left")
    keep_right = join_urban_graphs(left, right, node_conflict="right")

    assert keep_left.nodes_gdf.loc[1, "tag"] == "L"
    assert keep_right.nodes_gdf.loc[1, "tag"] == "R"


def test_join_urban_graphs_rejects_duplicate_edge_keys():
    left = _two_node_graph({0: (0.0, 0.0), 1: (10.0, 0.0)}, [{"u": 0, "v": 1}])
    right = _two_node_graph({0: (0.0, 0.0), 1: (10.0, 0.0)}, [{"u": 0, "v": 1}])

    with pytest.raises(ValueError, match="Duplicate edge keys"):
        join_urban_graphs(left, right)


def test_join_urban_graphs_rejects_crs_mismatch():
    left = _two_node_graph({0: (0.0, 0.0), 1: (10.0, 0.0)}, [{"u": 0, "v": 1}])
    right_coords = {2: (0.0, 0.0), 3: (1.0, 0.0)}
    right_nodes = gpd.GeoDataFrame(
        geometry=[Point(xy) for xy in right_coords.values()],
        index=list(right_coords),
        crs="EPSG:4326",
    )
    right_edges = gpd.GeoDataFrame(
        [{"u": 2, "v": 3, "geometry": LineString([(0, 0), (1, 0)]), "length_meter": 1.0, "time_min": 1.0}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    right = UrbanGraph(right_nodes, right_edges, is_multigraph=False, is_directed=False, crs="EPSG:4326")

    with pytest.raises(ValueError, match="CRS mismatch"):
        join_urban_graphs(left, right)


# ---------------------------------------------------------------------------
# project_objects2urban_graph + apply_urban_graph_changes
# ---------------------------------------------------------------------------


def test_project_objects_splits_nearest_edge_and_connects_object():
    graph = undirected_line_graph()
    objects = gpd.GeoDataFrame(index=[1000], geometry=[Point(15.0, 5.0)], crs=CRS)

    changes, object2node = project_objects2urban_graph(graph, objects, WALK_SPEED_M_PER_MIN)

    assert object2node.index.tolist() == [1000]
    assert changes.nodes_gdf is not None and not changes.nodes_gdf.empty
    assert changes.edges_gdf is not None and not changes.edges_gdf.empty
    # Nearest edge (1, 2) is scheduled for replacement by the split segments.
    deleted = set(zip(changes.edges_to_delete["u"], changes.edges_to_delete["v"]))
    assert (1, 2) in deleted

    new_graph = apply_urban_graph_changes(graph, changes)
    obj_node = object2node.loc[1000]
    assert obj_node in new_graph.nodes_gdf.index
    assert len(new_graph.nodes_gdf) > len(graph.nodes_gdf)
    remaining = set(zip(new_graph.edges_gdf["u"], new_graph.edges_gdf["v"]))
    assert (1, 2) not in remaining  # original edge replaced by split parts


def test_project_objects_snaps_to_existing_endpoint_without_splitting():
    graph = undirected_line_graph()
    objects = gpd.GeoDataFrame(index=[1000], geometry=[Point(0.0, 3.0)], crs=CRS)

    changes, object2node = project_objects2urban_graph(graph, objects, WALK_SPEED_M_PER_MIN)

    assert changes.edges_to_delete.empty  # endpoint snap does not split any edge
    new_graph = apply_urban_graph_changes(graph, changes)
    # Original edge (0, 1) is preserved.
    assert ((new_graph.edges_gdf["u"] == 0) & (new_graph.edges_gdf["v"] == 1)).any()
    assert object2node.loc[1000] in new_graph.nodes_gdf.index


def test_project_objects_returns_empty_changes_when_outside_max_dist():
    graph = undirected_line_graph()
    objects = gpd.GeoDataFrame(index=[1000], geometry=[Point(15.0, 1000.0)], crs=CRS)

    changes, object2node = project_objects2urban_graph(graph, objects, WALK_SPEED_M_PER_MIN, max_dist=10.0)

    assert object2node.empty
    assert changes.nodes_gdf is None
    assert changes.edges_gdf is None


@pytest.mark.parametrize(
    "objects, kwargs, exc, match",
    [
        (pd.DataFrame({"geometry": [Point(0, 0)]}), {}, TypeError, "GeoDataFrame"),
        (gpd.GeoDataFrame(geometry=[], crs=CRS), {}, ValueError, "empty"),
        (
            gpd.GeoDataFrame(index=["a", "a"], geometry=[Point(0, 0), Point(1, 0)], crs=CRS),
            {},
            ValueError,
            "unique",
        ),
    ],
)
def test_project_objects_validates_objects(objects, kwargs, exc, match):
    graph = undirected_line_graph()
    with pytest.raises(exc, match=match):
        project_objects2urban_graph(graph, objects, WALK_SPEED_M_PER_MIN, **kwargs)


@pytest.mark.parametrize("speed", [0.0, -5.0])
def test_project_objects_rejects_non_positive_speed(speed):
    graph = undirected_line_graph()
    objects = gpd.GeoDataFrame(index=["obj"], geometry=[Point(15.0, 5.0)], crs=CRS)

    with pytest.raises(ValueError, match="must be > 0"):
        project_objects2urban_graph(graph, objects, speed)


def test_apply_changes_rejects_multigraph_flag_mismatch():
    graph = undirected_line_graph()  # is_multigraph=False
    changes = UrbanGraphChanges(is_multigraph=True, is_directed=False)

    with pytest.raises(ValueError, match="is_multigraph"):
        apply_urban_graph_changes(graph, changes)


def test_apply_changes_rejects_deleting_absent_edge():
    graph = undirected_line_graph()
    changes = UrbanGraphChanges(
        edges_to_delete=pd.DataFrame([{"u": 0, "v": 99}]),
        is_multigraph=False,
        is_directed=False,
    )

    with pytest.raises(ValueError, match="not present"):
        apply_urban_graph_changes(graph, changes)


def test_apply_changes_rejects_node_id_collision():
    graph = undirected_line_graph()
    colliding = gpd.GeoDataFrame(index=[0], geometry=[Point(0.0, 0.0)], crs=CRS)
    changes = UrbanGraphChanges(nodes_gdf=colliding, is_multigraph=False, is_directed=False)

    with pytest.raises(ValueError, match="collision"):
        apply_urban_graph_changes(graph, changes)


# ---------------------------------------------------------------------------
# UrbanGraphChanges validation
# ---------------------------------------------------------------------------


def test_changes_rejects_non_geodataframe_nodes():
    with pytest.raises(TypeError, match="nodes_gdf must be GeoDataFrame"):
        UrbanGraphChanges(nodes_gdf=pd.DataFrame({"geometry": [Point(0, 0)]}))


def test_changes_rejects_edges_missing_required_columns():
    bad_edges = gpd.GeoDataFrame(
        [{"u": 0, "geometry": LineString([(0, 0), (1, 0)])}],
        geometry="geometry",
        crs=CRS,
    )
    with pytest.raises(ValueError, match="missing required columns"):
        UrbanGraphChanges(edges_gdf=bad_edges, is_multigraph=False)
