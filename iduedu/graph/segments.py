"""Elementary segments of the graph geometry, each counted once."""

import geopandas as gpd
import numpy as np
import shapely

from iduedu.graph.urban_graph import UrbanGraph


def unique_segments(urban_graph: UrbanGraph, *, precision: float = 0.01) -> gpd.GeoDataFrame:
    """Return the straight segments the edge geometries are made of, each physical segment once.

    Every pair of consecutive coordinates of an edge geometry is one segment. Segments are compared without
    direction and after rounding their end points to ``precision``, so the two directions of a two-way
    street, parallel edges drawn over the same points, and stretches that a there-and-back pair of edges
    shares only in part are all counted once. Separately mapped carriageways, which have their own points,
    stay separate. The sum of ``length`` is the length of the network on the ground.

    Args:
        urban_graph: Graph with line geometries in a projected (metric) CRS.
        precision: Grid, in CRS units, that end points are rounded to before comparing segments.

    Returns:
        GeoDataFrame in the graph CRS with one two-point LineString per unique segment and columns
        ``length``, ``edge`` (label of the first edge in ``edges_gdf`` that contains the segment) and
        ``edge_count`` (how many edge pieces share it). Zero-length segments are left out.

    Raises:
        ValueError: If ``precision`` is not positive or the graph CRS is geographic.
    """
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError("precision must be a positive number")
    edges = urban_graph.edges_gdf
    crs = getattr(edges, "crs", None) or urban_graph.crs
    if crs is not None and getattr(crs, "is_geographic", False):
        raise ValueError("unique_segments needs a projected CRS; reproject the graph first")
    columns = ["geometry", "length", "edge", "edge_count"]
    if edges.empty:
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)

    coords, owner = shapely.get_coordinates(edges.geometry.values, return_index=True)
    same_edge = owner[1:] == owner[:-1]
    start, end, owner = coords[:-1][same_edge], coords[1:][same_edge], owner[:-1][same_edge]

    # Grid integers make equal points compare equal; each segment is stored from its lower end point.
    a = np.round(start / precision).astype(np.int64)
    b = np.round(end / precision).astype(np.int64)
    swap = (b[:, 0] < a[:, 0]) | ((b[:, 0] == a[:, 0]) & (b[:, 1] < a[:, 1]))
    keys = np.where(swap[:, None], np.hstack([b, a]), np.hstack([a, b]))
    nonzero = np.any(keys[:, :2] != keys[:, 2:], axis=1)
    keys, start, end, owner = keys[nonzero], start[nonzero], end[nonzero], owner[nonzero]

    _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    order = np.argsort(first)  # keep the edge order of the graph
    first, counts = first[order], counts[order]
    geometry = shapely.linestrings(np.stack([start[first], end[first]], axis=1))
    return gpd.GeoDataFrame(
        {
            "length": shapely.length(geometry),
            "edge": edges.index.to_numpy()[owner[first]],
            "edge_count": counts,
        },
        geometry=geometry,
        crs=crs,
    )[columns]
