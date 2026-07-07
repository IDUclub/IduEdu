import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from iduedu.graph.urban_graph import UrbanGraph

CRS = "EPSG:3857"


def _nodes(coords: dict[int, tuple[float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"x": [xy[0] for xy in coords.values()], "y": [xy[1] for xy in coords.values()]},
        geometry=[Point(xy) for xy in coords.values()],
        index=pd.Index(coords.keys(), name="node"),
        crs=CRS,
    )


def _edge_geom(coords: dict[int, tuple[float, float]], u: int, v: int) -> LineString:
    return LineString([coords[u], coords[v]])


def _edges(
    coords: dict[int, tuple[float, float]],
    rows: list[dict],
    *,
    crs: str = CRS,
) -> gpd.GeoDataFrame:
    edge_rows = []
    for row in rows:
        item = dict(row)
        item.setdefault("geometry", _edge_geom(coords, int(item["u"]), int(item["v"])))
        item.setdefault("length_meter", float(item["geometry"].length))
        item.setdefault("time_min", float(item["length_meter"]))
        edge_rows.append(item)
    return gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=crs)


def undirected_line_graph() -> UrbanGraph:
    coords = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (20.0, 0.0), 3: (30.0, 0.0)}
    edges = _edges(
        coords,
        [
            {"u": 0, "v": 1, "length_meter": 10.0, "time_min": 1.0},
            {"u": 1, "v": 2, "length_meter": 10.0, "time_min": 2.0},
            {"u": 2, "v": 3, "length_meter": 10.0, "time_min": 4.0},
        ],
    )
    return UrbanGraph(_nodes(coords), edges, is_multigraph=False, is_directed=False, crs=CRS, graph_type="walk")


def directed_oneway_graph() -> UrbanGraph:
    coords = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (20.0, 0.0), 3: (30.0, 0.0)}
    edges = _edges(
        coords,
        [
            {"u": 0, "v": 1, "length_meter": 10.0, "time_min": 1.0, "oneway": True},
            {"u": 1, "v": 2, "length_meter": 10.0, "time_min": 2.0, "oneway": False},
            {"u": 2, "v": 3, "length_meter": 10.0, "time_min": 4.0, "oneway": True},
        ],
    )
    return UrbanGraph(
        _nodes(coords),
        edges,
        is_multigraph=False,
        is_directed=True,
        edge_direction_column="oneway",
        crs=CRS,
        graph_type="walk",
    )


def multigraph_with_parallel_edges() -> UrbanGraph:
    coords = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (20.0, 0.0)}
    edges = _edges(
        coords,
        [
            {"u": 0, "v": 1, "k": 0, "length_meter": 10.0, "time_min": 5.0},
            {"u": 0, "v": 1, "k": 1, "length_meter": 10.0, "time_min": 1.0},
            {"u": 1, "v": 2, "k": 0, "length_meter": 10.0, "time_min": 2.0},
        ],
    )
    return UrbanGraph(_nodes(coords), edges, is_multigraph=True, is_directed=False, crs=CRS, graph_type="walk")


def disconnected_graph() -> UrbanGraph:
    coords = {
        0: (0.0, 0.0),
        1: (10.0, 0.0),
        2: (20.0, 0.0),
        10: (100.0, 0.0),
        11: (110.0, 0.0),
    }
    edges = _edges(
        coords,
        [
            {"u": 0, "v": 1, "length_meter": 10.0, "time_min": 1.0},
            {"u": 1, "v": 2, "length_meter": 10.0, "time_min": 1.0},
            {"u": 10, "v": 11, "length_meter": 10.0, "time_min": 1.0},
        ],
    )
    return UrbanGraph(_nodes(coords), edges, is_multigraph=False, is_directed=False, crs=CRS, graph_type="walk")


def tiny_walk_graph() -> UrbanGraph:
    coords = {100: (0.0, 0.0), 101: (20.0, 0.0)}
    edges = _edges(
        coords,
        [{"u": 100, "v": 101, "k": 0, "length_meter": 20.0, "time_min": 4.0, "oneway": False, "type": "walk"}],
    )
    return UrbanGraph(
        _nodes(coords),
        edges,
        is_multigraph=True,
        is_directed=False,
        edge_direction_column=None,
        crs=CRS,
        graph_type="walk",
    )


def tiny_public_transport_graph_with_sequence_attrs() -> UrbanGraph:
    coords = {200: (8.0, 2.0), 201: (40.0, 0.0)}
    nodes = _nodes(coords)
    nodes["type"] = ["platform", "stop"]
    nodes["route_refs"] = [["A", "B"], ["C"]]
    nodes["route_names"] = [["Alpha"], ["Gamma", "Delta"]]
    edges = _edges(
        coords,
        [
            {
                "u": 200,
                "v": 201,
                "k": 0,
                "length_meter": 32.0,
                "time_min": 3.0,
                "type": "subway",
            }
        ],
    )
    return UrbanGraph(nodes, edges, is_multigraph=True, is_directed=True, crs=CRS, graph_type="public_transport")
