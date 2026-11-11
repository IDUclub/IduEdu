"""Elevation of a graph: node heights, edge grades and travel times that depend on the grade.

:func:`add_elevation` reads heights from a raster (a digital elevation model), from a GeoDataFrame of
spot heights or contour lines, or from any function of the coordinates, and stores the height of every
node and, for every edge, its signed grade, the steepest grade along its geometry and the total ascent and
descent. Bridges and tunnels do not follow the ground: their heights are interpolated between their ends.
:func:`apply_grade_speed` then rescales the travel times with a speed model such as Tobler's hiking
function.
"""

import os
from pathlib import Path
from typing import Callable, Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy import sparse
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from iduedu.graph.graph_inputs import projected_crs
from iduedu.graph.urban_graph import UrbanGraph

RASTER_HINT = (
    "Reading elevation from a raster file needs the optional dependency rasterio. "
    "Install it with: pip install 'iduedu[elevation]'"
)
Sampler = Callable[[np.ndarray], np.ndarray]


def _raster_sampler(path: Path, crs, bounds, band: int) -> Sampler:
    """Bilinear heights from the part of a raster that covers ``bounds`` (graph CRS)."""
    try:
        import rasterio  # pylint: disable=import-outside-toplevel
        from rasterio.windows import Window, from_bounds  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        if exc.name != "rasterio":
            raise
        raise ImportError(RASTER_HINT) from exc

    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError(f"raster {path} has no CRS")
        to_raster = Transformer.from_crs(crs, source.crs, always_xy=True)
        minx, miny, maxx, maxy = bounds
        corners_x, corners_y = to_raster.transform([minx, minx, maxx, maxx], [miny, maxy, miny, maxy])
        window = from_bounds(min(corners_x), min(corners_y), max(corners_x), max(corners_y), source.transform)
        # A margin of two cells for the interpolation, clipped to the raster.
        col0 = max(int(np.floor(window.col_off)) - 2, 0)
        row0 = max(int(np.floor(window.row_off)) - 2, 0)
        col1 = min(int(np.ceil(window.col_off + window.width)) + 2, source.width)
        row1 = min(int(np.ceil(window.row_off + window.height)) + 2, source.height)
        if col1 <= col0 or row1 <= row0:
            raise ValueError(f"raster {path} does not cover the graph")
        data = source.read(band, window=Window(col0, row0, col1 - col0, row1 - row0)).astype(float)
        if source.nodata is not None:
            data[data == source.nodata] = np.nan
        inverse = ~source.transform

    def sample(xy: np.ndarray) -> np.ndarray:
        x, y = to_raster.transform(xy[:, 0], xy[:, 1])
        x, y = np.asarray(x), np.asarray(y)
        cols = inverse.a * x + inverse.b * y + inverse.c
        rows = inverse.d * x + inverse.e * y + inverse.f
        # Cell centres are at half-integer pixel coordinates.
        rows, cols = rows - 0.5 - row0, cols - 0.5 - col0
        values = map_coordinates(data, [rows, cols], order=1, mode="nearest")
        outside = (rows < -0.5) | (cols < -0.5) | (rows > data.shape[0] - 0.5) | (cols > data.shape[1] - 0.5)
        values[outside] = np.nan
        return values

    return sample


def _table_sampler(table: gpd.GeoDataFrame, column: str, crs, method: str, spacing: float) -> Sampler:
    """Heights interpolated from points or contour lines with a height column."""
    if column not in table.columns:
        raise KeyError(f"elevation table has no column {column!r}")
    if table.crs is None:
        raise ValueError("elevation table must have a CRS")
    table = table[[column, table.geometry.name]].dropna()
    if table.crs != crs:
        table = table.to_crs(crs)
    geometry = table.geometry.values
    heights = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
    lines = np.isin(shapely.get_type_id(geometry), [1, 2, 5])  # LineString, LinearRing, MultiLineString
    if lines.any():  # contours: their vertices and points between them
        geometry = geometry.copy()
        geometry[lines] = shapely.segmentize(geometry[lines], spacing)
    xy, owner = shapely.get_coordinates(geometry, return_index=True)
    z = heights[owner]
    known = np.isfinite(z)
    xy, z = xy[known], z[known]
    xy, first = np.unique(xy, axis=0, return_index=True)
    z = z[first]
    if len(xy) == 0:
        raise ValueError("elevation table has no heights")
    tree = cKDTree(xy)

    def nearest(points):
        return z[tree.query(points)[1]]

    if method == "nearest" or len(xy) < 3:
        return nearest
    if method == "idw":

        def weighted(points):
            dist, index = tree.query(points, k=min(8, len(xy)))
            dist, index = dist.reshape(len(points), -1), index.reshape(len(points), -1)
            exact = dist[:, 0] == 0
            weights = 1.0 / np.maximum(dist, 1e-12) ** 2
            values = (weights * z[index]).sum(axis=1) / weights.sum(axis=1)
            values[exact] = z[index[exact, 0]]
            return values

        return weighted
    if method != "linear":
        raise ValueError("method must be 'linear', 'nearest' or 'idw'")
    triangles = LinearNDInterpolator(xy, z)

    def linear(points):
        values = triangles(points)
        outside = np.isnan(values)
        if outside.any():  # outside the hull of the data: the nearest height
            values[outside] = nearest(points[outside])
        return values

    return linear


def _sampler(elevation, crs, bounds, column, method, spacing, band) -> Sampler:
    if isinstance(elevation, (str, os.PathLike)):
        return _raster_sampler(Path(elevation), crs, bounds, band)
    if isinstance(elevation, gpd.GeoDataFrame):
        return _table_sampler(elevation, column, crs, method, spacing)
    if callable(elevation):
        return lambda xy: np.asarray(elevation(xy[:, 0], xy[:, 1]), dtype=float)
    raise TypeError(
        "elevation must be a raster path, a GeoDataFrame of points or contour lines, or a function of x and y, "
        f"got {type(elevation).__name__}"
    )


def _structures(edges: gpd.GeoDataFrame, columns: Iterable[str]) -> np.ndarray:
    """Edges on a bridge or in a tunnel: a structure column with a value other than ``no``."""
    mask = np.zeros(len(edges), dtype=bool)
    for column in columns:
        if column in edges.columns:
            value = edges[column].astype("string").str.strip().str.lower()
            mask |= (value.notna() & ~value.isin(["", "no", "false", "0", "none", "nan"])).to_numpy(dtype=bool)
    return mask


def _span_heights(z: np.ndarray, u: np.ndarray, v: np.ndarray, length: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Heights of nodes inside bridges and tunnels, interpolated between the ends of the structure.

    A node between structure edges only does not stand on the ground. Its height solves the Laplace
    equation on the structure edges with weights ``1 / length``: linear along a chain, a weighted mean at a
    junction.
    """
    n = len(z)
    grounded = np.zeros(n, dtype=bool)
    grounded[u[~structure]] = grounded[v[~structure]] = True
    spans = np.bincount(u[structure], minlength=n) + np.bincount(v[structure], minlength=n)
    touched = spans > 0
    # A node with a single structure edge and nothing else is the end of the structure: it stands on the ground.
    grounded |= spans == 1
    inner = np.flatnonzero(touched & ~grounded)
    if len(inner) == 0:
        return z
    a, b = u[structure], v[structure]
    keep = a != b
    a, b, w = a[keep], b[keep], 1.0 / np.maximum(length[structure][keep], 1e-6)
    adjacency = sparse.coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    # Structures with no end on the ground keep their sampled heights.
    _, label = sparse.csgraph.connected_components(adjacency, directed=False)
    anchored = np.zeros(label.max() + 1, dtype=bool)
    anchors = touched & grounded & np.isfinite(z)
    anchored[label[anchors]] = True
    inner = inner[anchored[label[inner]]]
    if len(inner) == 0:
        return z
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = sparse.diags(degree) - adjacency
    fixed = np.flatnonzero(anchors)
    solved = spsolve(laplacian[inner][:, inner].tocsc(), -laplacian[inner][:, fixed] @ z[fixed])
    z = z.copy()
    z[inner] = solved
    return z


def add_elevation(  # pylint: disable=too-many-locals
    urban_graph: UrbanGraph,
    elevation: str | os.PathLike | gpd.GeoDataFrame | Callable,
    *,
    column: str = "elevation",
    method: Literal["linear", "nearest", "idw"] = "linear",
    sample_distance: float = 20.0,
    structure_columns: Iterable[str] = ("bridge", "tunnel"),
    band: int = 1,
) -> UrbanGraph:
    """Add node heights and edge grades to a graph.

    Nodes get ``elevation`` (metres). Every edge is sampled along its geometry at steps of at most
    ``sample_distance`` and gets, in the direction from ``u`` to ``v``:

    - ``grade``: rise over length between its ends, positive uphill;
    - ``grade_max``: the steepest absolute grade between two neighbouring samples;
    - ``ascent`` and ``descent``: metres climbed and descended along the edge, both non-negative.

    Edges on a bridge or in a tunnel (a ``structure_columns`` tag other than ``no``) get a straight profile
    between their ends, and the nodes inside such structures get heights interpolated between the ends of
    the structure instead of the ground below or above them.

    Args:
        urban_graph: Graph in a projected (metric) CRS.
        elevation: Source of heights: a raster file (read with rasterio, bilinear interpolation, any CRS),
            a GeoDataFrame of points or contour lines with heights in ``column``, or a function
            ``f(x, y) -> z`` of coordinate arrays in the graph CRS.
        column: Height column of an elevation GeoDataFrame.
        method: Interpolation between the heights of a GeoDataFrame: ``"linear"`` on a triangulation (the
            nearest height outside the data), ``"nearest"`` or ``"idw"`` (inverse squared distance to the
            eight nearest points).
        sample_distance: Largest step, in metres, between samples along an edge, and between points taken
            from contour lines.
        structure_columns: Edge columns that mark bridges and tunnels.
        band: Band of a raster file.

    Returns:
        A new ``UrbanGraph`` with the added columns. Nodes without data have ``NaN`` and so do the edges
        that touch them.

    Raises:
        ImportError: If ``elevation`` is a raster path and rasterio is not installed.
        ValueError: If the graph CRS is missing or geographic, or the source does not cover the graph.
    """
    crs = projected_crs(urban_graph, "add_elevation", required=True)
    if not np.isfinite(sample_distance) or sample_distance <= 0:
        raise ValueError("sample_distance must be a positive number")
    result = urban_graph.copy()
    nodes, edges = result.nodes_gdf, result.edges_gdf
    xy = np.column_stack([nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()]).reshape(-1, 2)
    bounds = edges.total_bounds if len(edges) else nodes.total_bounds
    sample = _sampler(elevation, crs, bounds, column, method, sample_distance, band)
    z = np.asarray(sample(xy), dtype=float) if len(xy) else np.zeros(0)

    index = pd.Index(nodes.index)
    u, v = index.get_indexer(edges["u"]), index.get_indexer(edges["v"])
    geoms = edges.geometry.values
    length = shapely.length(geoms) if len(edges) else np.zeros(0)
    structure = _structures(edges, structure_columns)
    if structure.any():
        z = _span_heights(z, u, v, length, structure)
    nodes["elevation"] = np.round(z, 2)

    if len(edges):
        # Samples at equal steps along every edge; the ends take the node heights.
        steps = np.maximum(np.ceil(length / sample_distance), 1).astype(np.int64)
        owner = np.repeat(np.arange(len(edges)), steps + 1)
        first = np.concatenate([[0], np.cumsum(steps + 1)[:-1]])
        position = np.arange(len(owner)) - first[owner]
        fraction = position / steps[owner]
        coords = shapely.get_coordinates(shapely.line_interpolate_point(geoms[owner], fraction, normalized=True))
        start = shapely.get_coordinates(shapely.get_point(geoms, 0))
        from_u = np.hypot(*(start - xy[u]).T) <= np.hypot(*(start - xy[v]).T)  # the geometry may run from v to u
        z_start, z_end = np.where(from_u, z[u], z[v]), np.where(from_u, z[v], z[u])
        inner = (position > 0) & (position < steps[owner]) & ~structure[owner]
        profile = z_start[owner] + (z_end - z_start)[owner] * fraction  # straight line: structures and ends
        if inner.any():
            profile[inner] = sample(coords[inner])
        rise = np.diff(profile)[position[1:] > 0]  # one value per step, in geometry direction
        step_owner = owner[1:][position[1:] > 0]
        step_length = (length / steps)[step_owner]
        up = np.bincount(step_owner, weights=np.maximum(rise, 0), minlength=len(edges))
        down = np.bincount(step_owner, weights=np.maximum(-rise, 0), minlength=len(edges))
        missing = np.bincount(step_owner, weights=np.isnan(rise), minlength=len(edges)) > 0
        steepest = np.zeros(len(edges))
        with np.errstate(divide="ignore", invalid="ignore"):
            np.maximum.at(steepest, step_owner, np.abs(np.where(step_length > 0, rise / step_length, 0.0)))
            grade = np.where(length > 0, (z[v] - z[u]) / length, 0.0)
        up, down, steepest = (np.where(missing, np.nan, values) for values in (up, down, steepest))
        edges["grade"] = np.round(grade, 4)
        edges["grade_max"] = np.round(steepest, 4)
        edges["ascent"] = np.round(np.where(from_u, up, down), 2)
        edges["descent"] = np.round(np.where(from_u, down, up), 2)
    else:
        for name in ("grade", "grade_max", "ascent", "descent"):
            edges[name] = np.zeros(0)
    return result


def tobler(grade: np.ndarray) -> np.ndarray:
    """Tobler's hiking function as a speed relative to flat ground: ``exp(-3.5 |g + 0.05|) / exp(-0.175)``.

    The fastest walk is on a gentle descent of 5 %, where the value is about 1.19.
    """
    return np.exp(-3.5 * np.abs(np.asarray(grade, dtype=float) + 0.05)) / np.exp(-3.5 * 0.05)


def cycling(
    grade: np.ndarray, *, uphill: float = 6.0, downhill: float = 4.0, slowest: float = 0.25, fastest: float = 1.5
):
    """A linear cycling speed model relative to flat ground.

    Speed falls by ``uphill`` per cent for each per cent of climb, not below ``slowest`` of the flat speed,
    and grows by ``downhill`` per cent for each per cent of descent, not above ``fastest``.
    """
    grade = np.asarray(grade, dtype=float)
    return np.where(grade >= 0, np.maximum(1.0 - uphill * grade, slowest), np.minimum(1.0 + downhill * -grade, fastest))


_MODELS = {"tobler": tobler, "cycling": cycling}


def apply_grade_speed(
    urban_graph: UrbanGraph,
    model: Literal["tobler", "cycling"] | Callable[[np.ndarray], np.ndarray] = "tobler",
    *,
    edge_types: Iterable[str] | None = None,
) -> UrbanGraph:
    """Rescale the travel times of a graph with elevation by a speed model that depends on the grade.

    The time of an edge is divided by the relative speed the model gives for its grade. An edge that goes
    up and down is taken as a climb of ``ascent`` followed by a drop of ``descent``, both at the mean
    absolute grade ``(ascent + descent) / length``. A row that serves both directions (an edge of an
    undirected graph, or a two-way edge of a directed one) gets the mean of the two directions.

    The times before the change are kept in ``time_min_flat``, and a repeated call starts from them.

    Args:
        urban_graph: Graph with ``ascent`` and ``descent`` on its edges (see :func:`add_elevation`).
        model: ``"tobler"`` (:func:`tobler`, walking), ``"cycling"`` (:func:`cycling`) or a function that
            maps an array of signed grades to speeds relative to flat ground.
        edge_types: Change only edges whose ``type`` is one of these, for example ``["walk"]`` in an
            intermodal graph.

    Returns:
        A new ``UrbanGraph`` with changed ``time_min``.

    Raises:
        KeyError: If the edges have no ``ascent`` and ``descent``.
        ValueError: If the model is unknown or returns speeds that are not positive.
    """
    if not isinstance(urban_graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(urban_graph).__name__}")
    speed = _MODELS.get(model) if isinstance(model, str) else model
    if speed is None or not callable(speed):
        raise ValueError(f"model must be one of {sorted(_MODELS)} or a function, got {model!r}")
    result = urban_graph.copy()
    edges = result.edges_gdf
    missing = [c for c in ("ascent", "descent", "length_meter", "time_min") if c not in edges.columns]
    if missing:
        raise KeyError(f"edges_gdf has no columns {missing}; run add_elevation first")
    if "time_min_flat" not in edges.columns:
        edges["time_min_flat"] = edges["time_min"]
    flat = edges["time_min_flat"].to_numpy(dtype=float)
    up = np.nan_to_num(edges["ascent"].to_numpy(dtype=float))
    down = np.nan_to_num(edges["descent"].to_numpy(dtype=float))
    length = edges["length_meter"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        grade = np.where(length > 0, (up + down) / length, 0.0)
        climbing = np.where(up + down > 0, up / (up + down), 0.0)
    uphill, downhill = np.asarray(speed(grade), dtype=float), np.asarray(speed(-grade), dtype=float)
    if not (np.isfinite(uphill).all() and np.isfinite(downhill).all() and (uphill > 0).all() and (downhill > 0).all()):
        raise ValueError("the speed model must return positive finite speeds")
    forward = climbing / uphill + (1.0 - climbing) / downhill
    backward = climbing / downhill + (1.0 - climbing) / uphill
    direction = result.edge_direction_column
    if not result.is_directed:
        both = np.ones(len(edges), dtype=bool)
    elif direction is not None and direction in edges.columns:
        both = ~edges[direction].to_numpy(dtype=bool)
    else:
        both = np.zeros(len(edges), dtype=bool)
    factor = np.where(both, (forward + backward) / 2.0, forward)
    if edge_types is not None:
        if "type" not in edges.columns:
            raise KeyError("edge_types needs a 'type' column in edges_gdf")
        factor = np.where(edges["type"].isin(list(edge_types)).to_numpy(), factor, 1.0)
    edges["time_min"] = np.round(flat * factor, 3)
    result.adjacency_matrix = None
    return result
