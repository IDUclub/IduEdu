import numpy as np
import pytest
import shapely
from shapely import LineString, Point
from shapely.ops import substring

from iduedu.graph.lines import cut_line, vertex_distances

pytestmark = pytest.mark.unit


def _as_line(result):
    return LineString([result, result]) if isinstance(result, Point) else result


@pytest.mark.parametrize("seed", range(5))
def test_cut_line_matches_shapely_substring(seed):
    rng = np.random.default_rng(seed)
    line = LineString(np.cumsum(rng.normal(size=(40, 2)) * 10, axis=0))
    coords = np.asarray(line.coords)
    distances = vertex_distances(coords)
    # Random cuts, cuts exactly at vertices, at both ends, reversed and empty ones.
    cuts = list(rng.uniform(0, line.length, size=(30, 2)))
    cuts += [(line.project(Point(coords[i])), line.project(Point(coords[j]))) for i, j in [(3, 17), (25, 4), (0, 39)]]
    cuts += [(0.0, line.length), (line.length, 0.0), (5.0, 5.0)]
    for start, end in cuts:
        expected = _as_line(substring(line, start, end))
        result = cut_line(coords, distances, start, end, line.interpolate(start), line.interpolate(end))
        assert shapely.equals_exact(result, expected, tolerance=0), (start, end)
