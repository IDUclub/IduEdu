"""Cutting line geometries at distances along them, as ``shapely.ops.substring`` does, in linear time per line."""

import numpy as np
from shapely import LineString


def vertex_distances(coords: np.ndarray) -> np.ndarray:
    """Distance of every vertex from the start of the line, summed the way ``shapely.ops.substring`` sums it.

    ``** 0.5`` per segment as in shapely, not ``np.sqrt``: the two can differ in the last bit, which decides
    whether a vertex lying exactly at a cut point is kept.
    """
    xy = np.asarray(coords)[:, :2].tolist()
    lengths = [((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 for (x1, y1), (x2, y2) in zip(xy, xy[1:])]
    return np.concatenate([[0.0], np.cumsum(lengths)])


def cut_line(coords: np.ndarray, vertex_dist: np.ndarray, start: float, end: float, start_point, end_point):
    """``shapely.ops.substring(line, start, end)`` for ``0 <= start, end <= line.length``.

    ``coords`` and ``vertex_dist`` (from :func:`vertex_distances`) describe the line once, so many cuts of one
    line cost a binary search each instead of a walk over the whole line. ``start_point`` and ``end_point`` are
    ``line.interpolate(start)`` and ``line.interpolate(end)``, the end points shapely uses. The result keeps the
    vertices strictly between the two distances, runs from ``start`` to ``end`` (backwards when
    ``start > end``) and is a two-point line when the distances are equal (where shapely returns a Point).
    """
    first_xy = (start_point.x, start_point.y)
    if start == end:
        return LineString([first_xy, first_xy])
    low, high = (start, end) if start < end else (end, start)
    first = np.searchsorted(vertex_dist, low, side="right")
    last = min(np.searchsorted(vertex_dist, high, side="left"), len(coords) - 1)
    inner = np.asarray(coords)[first:last, :2]
    if start > end:
        inner = inner[::-1]
    return LineString(np.vstack([[first_xy], inner, [(end_point.x, end_point.y)]]))
