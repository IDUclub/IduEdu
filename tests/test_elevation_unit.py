import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from iduedu import UrbanGraph, add_elevation, apply_grade_speed, graph_stats
from iduedu.graph.elevation import cycling, tobler
from tests.factories import CRS, _edges, _nodes, undirected_line_graph

pytestmark = pytest.mark.unit


def slope(x, y):
    """A plane rising 10 cm per metre to the east."""
    return 0.1 * np.asarray(x)


def line_graph(rows=None, *, directed=False, step=100.0) -> UrbanGraph:
    coords = {i: (i * step, 0.0) for i in range(4)}
    rows = rows or [{"u": 0, "v": 1}, {"u": 1, "v": 2}, {"u": 2, "v": 3}]
    edges = _edges(coords, rows)
    edges["time_min"] = edges["length_meter"] / 100.0
    return UrbanGraph(
        _nodes(coords),
        edges,
        is_multigraph=False,
        is_directed=directed,
        edge_direction_column="oneway" if directed else None,
        crs=CRS,
        graph_type="walk",
    )


def test_heights_and_grades_on_a_plane():
    backwards = LineString([(200.0, 0.0), (100.0, 0.0)])  # the geometry of edge 1 -> 2 runs from 2 to 1
    graph = line_graph([{"u": 0, "v": 1}, {"u": 1, "v": 2, "geometry": backwards}, {"u": 3, "v": 2}])
    result = add_elevation(graph, slope)
    assert result is not graph and "elevation" not in graph.nodes_gdf
    assert result.nodes_gdf["elevation"].tolist() == [0.0, 10.0, 20.0, 30.0]
    edges = result.edges_gdf
    assert edges["grade"].tolist() == pytest.approx([0.1, 0.1, -0.1])
    assert edges["grade_max"].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert edges["ascent"].tolist() == pytest.approx([10.0, 10.0, 0.0])
    assert edges["descent"].tolist() == pytest.approx([0.0, 0.0, 10.0])


def test_a_hill_inside_an_edge():
    def hill(x, y):
        return 5.0 - np.abs(np.asarray(x) - 50.0) / 10.0

    coords = {0: (0.0, 0.0), 1: (100.0, 0.0)}
    graph = UrbanGraph(
        _nodes(coords), _edges(coords, [{"u": 0, "v": 1}]), is_multigraph=False, is_directed=False, crs=CRS
    )
    edge = add_elevation(graph, hill, sample_distance=10).edges_gdf.iloc[0]
    assert edge["grade"] == 0 and edge["grade_max"] == pytest.approx(0.1)
    assert edge["ascent"] == pytest.approx(5.0) and edge["descent"] == pytest.approx(5.0)
    coarse = add_elevation(graph, hill, sample_distance=200).edges_gdf.iloc[0]
    assert coarse["ascent"] == 0 and coarse["grade_max"] == 0  # one step sees only the ends


def test_bridges_are_interpolated_between_their_ends():
    def valley(x, y):
        return np.where((np.asarray(x) > 50) & (np.asarray(x) < 250), 0.0, 10.0)

    plain = add_elevation(line_graph(), valley)
    assert plain.nodes_gdf["elevation"].tolist() == [10.0, 0.0, 0.0, 10.0]
    assert plain.edges_gdf["descent"].iloc[0] == pytest.approx(10.0)

    rows = [{"u": 0, "v": 1, "bridge": "yes"}, {"u": 1, "v": 2, "bridge": "viaduct"}, {"u": 2, "v": 3, "bridge": "no"}]
    bridged = add_elevation(line_graph(rows), valley)
    # Node 1 is inside the bridge and is lifted to the line between nodes 0 and 2; node 2 stands on the ground.
    assert bridged.nodes_gdf["elevation"].tolist() == [10.0, 5.0, 0.0, 10.0]
    assert bridged.edges_gdf["descent"].tolist() == pytest.approx([5.0, 5.0, 0.0])
    assert bridged.edges_gdf["grade_max"].tolist()[:2] == pytest.approx([0.05, 0.05])  # straight spans
    assert bridged.edges_gdf["ascent"].iloc[2] == pytest.approx(10.0)
    tunnel = add_elevation(line_graph([{"u": 0, "v": 1}, {"u": 1, "v": 2, "tunnel": "yes"}, {"u": 2, "v": 3}]), slope)
    assert tunnel.edges_gdf["grade"].tolist() == pytest.approx([0.1, 0.1, 0.1])


def _spot_heights():
    xs, ys = np.meshgrid(np.arange(-50.0, 400.0, 50.0), np.arange(-50.0, 100.0, 50.0))
    return gpd.GeoDataFrame(
        {"h": slope(xs.ravel(), None)}, geometry=gpd.points_from_xy(xs.ravel(), ys.ravel()), crs=CRS
    )


@pytest.mark.parametrize("method, exact", [("linear", True), ("idw", False), ("nearest", False)])
def test_heights_from_points(method, exact):
    result = add_elevation(line_graph(step=90.0), _spot_heights(), column="h", method=method)
    heights = result.nodes_gdf["elevation"].to_numpy()
    assert heights[0] == 0.0  # a node on a data point
    if exact:
        assert heights == pytest.approx([0.0, 9.0, 18.0, 27.0])
    else:
        assert np.all(np.diff(heights) > 0) and heights == pytest.approx([0.0, 9.0, 18.0, 27.0], abs=2.6)
    reprojected = add_elevation(line_graph(step=90.0), _spot_heights().to_crs(4326), column="h")
    assert reprojected.nodes_gdf["elevation"].to_numpy() == pytest.approx([0.0, 9.0, 18.0, 27.0], abs=0.05)


def test_heights_from_contours_and_bad_sources():
    contours = gpd.GeoDataFrame(
        {"elevation": [0.0, 20.0, 40.0]},
        geometry=[LineString([(x, -500.0), (x, 500.0)]) for x in (0.0, 200.0, 400.0)],
        crs=CRS,
    )
    result = add_elevation(line_graph(), contours)
    assert result.nodes_gdf["elevation"].to_numpy() == pytest.approx([0.0, 10.0, 20.0, 30.0], abs=0.01)
    with pytest.raises(KeyError):
        add_elevation(line_graph(), contours, column="height")
    with pytest.raises(TypeError):
        add_elevation(line_graph(), 42)
    with pytest.raises(ValueError, match="method"):
        add_elevation(line_graph(), _spot_heights(), column="h", method="cubic")
    with pytest.raises(ValueError, match="sample_distance"):
        add_elevation(line_graph(), slope, sample_distance=0)
    geographic = undirected_line_graph()
    geographic.nodes_gdf = geographic.nodes_gdf.to_crs(4326)
    geographic.edges_gdf = geographic.edges_gdf.to_crs(4326)
    with pytest.raises(ValueError, match="projected"):
        add_elevation(geographic, slope)


def test_heights_from_a_raster(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin  # pylint: disable=import-outside-toplevel

    # Cells of 10 m; the centre of column j is at x = -95 + 10 j, where the plane has the height 0.1 x.
    cols, rows = 60, 30
    centres = -100.0 + 10.0 * np.arange(cols) + 5.0
    data = np.tile(slope(centres, None), (rows, 1)).astype("float32")
    data[:, -1] = -9999.0
    path = tmp_path / "dem.tif"
    profile = {
        "driver": "GTiff",
        "width": cols,
        "height": rows,
        "count": 1,
        "dtype": "float32",
        "crs": CRS,
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", transform=from_origin(-100.0, 150.0, 10.0, 10.0), **profile) as target:
        target.write(data, 1)
    result = add_elevation(line_graph(), path)
    assert result.nodes_gdf["elevation"].to_numpy() == pytest.approx([0.0, 10.0, 20.0, 30.0], abs=1e-3)
    assert result.edges_gdf["grade"].tolist() == pytest.approx([0.1, 0.1, 0.1], abs=1e-4)
    assert result.edges_gdf["ascent"].tolist() == pytest.approx([10.0, 10.0, 10.0], abs=1e-2)
    # A node outside the raster gets no height, and neither do the edges at it.
    coords = {0: (0.0, 0.0), 1: (100.0, 0.0), 2: (900.0, 0.0)}
    far = UrbanGraph(
        _nodes(coords),
        _edges(coords, [{"u": 0, "v": 1}, {"u": 1, "v": 2}]),
        is_multigraph=False,
        is_directed=False,
        crs=CRS,
    )
    partial = add_elevation(far, path)
    assert np.isnan(partial.nodes_gdf["elevation"].iloc[2]) and np.isnan(partial.edges_gdf["grade"].iloc[1])
    assert partial.edges_gdf["grade"].iloc[0] == pytest.approx(0.1, abs=1e-4)


def test_speed_models():
    assert tobler(0.0) == pytest.approx(1.0) and tobler(-0.05) == pytest.approx(np.exp(0.175))
    assert tobler(0.2) < tobler(-0.2) < 1.0
    assert cycling(np.array([0.0, 0.05, 0.5, -0.05, -0.5])).tolist() == pytest.approx([1.0, 0.7, 0.25, 1.2, 1.5])


def test_grade_speed_rescales_travel_times():
    graph = add_elevation(line_graph(), slope)  # undirected, every edge climbs 10 %
    walked = apply_grade_speed(graph)
    both_ways = (1 / tobler(0.1) + 1 / tobler(-0.1)) / 2
    assert walked.edges_gdf["time_min"].tolist() == pytest.approx([both_ways] * 3, abs=1e-3)
    assert walked.edges_gdf["time_min_flat"].tolist() == [1.0, 1.0, 1.0]
    assert graph.edges_gdf["time_min"].tolist() == [1.0, 1.0, 1.0] and walked.adjacency_matrix is None
    again = apply_grade_speed(walked)  # starts from the flat times, does not compound
    assert again.edges_gdf["time_min"].tolist() == walked.edges_gdf["time_min"].tolist()

    rows = [{"u": 0, "v": 1, "oneway": True}, {"u": 2, "v": 1, "oneway": True}, {"u": 2, "v": 3, "oneway": False}]
    directed = apply_grade_speed(add_elevation(line_graph(rows, directed=True), slope), "cycling")
    assert directed.edges_gdf["time_min"].tolist() == pytest.approx(
        [1 / 0.4, 1 / 1.4, (1 / 0.4 + 1 / 1.4) / 2], abs=1e-3
    )
    halved = apply_grade_speed(graph, lambda grade: np.full_like(grade, 0.5))
    assert halved.edges_gdf["time_min"].tolist() == [2.0, 2.0, 2.0]

    typed = graph.copy()
    typed.edges_gdf["type"] = ["walk", "bus", "walk"]
    partly = apply_grade_speed(typed, edge_types=["walk"]).edges_gdf["time_min"].tolist()
    assert partly[1] == 1.0 and partly[0] == pytest.approx(both_ways, abs=1e-3)
    with pytest.raises(KeyError, match="add_elevation"):
        apply_grade_speed(line_graph())
    with pytest.raises(ValueError, match="model"):
        apply_grade_speed(graph, "skiing")
    with pytest.raises(ValueError, match="positive"):
        apply_grade_speed(graph, lambda grade: np.zeros_like(grade))


def test_relief_in_graph_stats():
    stats = graph_stats(add_elevation(line_graph(), slope))
    assert (stats["elevation_min"], stats["elevation_max"], stats["elevation_mean"]) == (0.0, 30.0, 15.0)
    assert stats["grade_mean"] == pytest.approx(0.1) and stats["grade_max"] == pytest.approx(0.1)
    assert "grade_mean" not in graph_stats(line_graph())
    assert Point(0, 0).equals(line_graph().nodes_gdf.geometry.iloc[0])
