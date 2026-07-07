"""
Shared harness for the IduEdu paper benchmarks.

Design goals (same as the v1 scripts, generalized):
- Resume-safe: every measurement is appended to CSV immediately; a run is skipped
  if its key is already present. Safe to interrupt and re-run, and safe to merge
  results produced from different environments (e.g. a separate pyrosm venv).
- One row = one measured attempt. Aggregation (median) happens in the figures
  notebook, never in the benchmark scripts.
- Deterministic representation-size helpers for graph objects.
"""

from __future__ import annotations

import gc
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PYROSM_CACHE_DIR = str(Path(__file__).resolve().parents[1] / "pbf_cache")


# ----------------------------
# Resume-safe CSV
# ----------------------------


def load_existing_keys(csv_path: Path, key_columns: list[str]) -> set[tuple]:
    """Return the set of key tuples already present in the CSV (all values as str)."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in key_columns if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} exists but misses key columns {missing}; delete or fix it")
    return {tuple(row) for row in df[key_columns].itertuples(index=False, name=None)}


def make_key(values: Iterable[Any]) -> tuple:
    """Normalize key values to strings, encoding None as 'NA' (CSV round-trip safe)."""
    return tuple("NA" if v is None else str(v) for v in values)


def append_row(csv_path: Path, row: dict) -> None:
    """Append one row, upgrading the CSV schema in place when new columns appear.

    Rows are dicts; if an existing CSV lacks some of the row's keys (e.g. a newly
    added metric column), the whole file is rewritten once with the union header
    so old rows keep aligning and resume/merge across environments stays safe.
    """
    if not csv_path.exists():
        pd.DataFrame([row]).to_csv(csv_path, index=False)
        return

    header = pd.read_csv(csv_path, nrows=0).columns.to_list()
    new_columns = [c for c in row if c not in header]
    if new_columns:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        for column in new_columns:
            df[column] = ""
        df.to_csv(csv_path, index=False)
        header = header + new_columns

    aligned = {column: row.get(column, "") for column in header}
    pd.DataFrame([aligned]).to_csv(csv_path, mode="a", header=False, index=False)


# ----------------------------
# Timing
# ----------------------------


@dataclass
class Measurement:
    time_sec: float
    result: Any


def measure(func: Callable, *args, **kwargs) -> Measurement:
    """Run func, returning wall time and the result."""
    gc.collect()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return Measurement(time_sec=elapsed, result=result)


def force_cleanup(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()


# ----------------------------
# AOI from PBF header (same protocol as v1: one bbox for every library)
# ----------------------------


@dataclass
class BoundsInfo:
    polygon_4326: Any
    bbox: tuple[float, float, float, float]
    pbf_path: str


def bbox_from_pbf(pbf_path: str) -> BoundsInfo:
    """Read the bounding box from a PBF header (fast, no parsing of the body)."""
    import osmium
    from shapely.geometry import box

    reader = osmium.io.Reader(pbf_path)
    header = reader.header()
    reader.close()
    if header.box is None:
        raise RuntimeError(f"No bbox in PBF header: {pbf_path}")
    bl = header.box().bottom_left
    tr = header.box().top_right
    bbox = (bl.lon, bl.lat, tr.lon, tr.lat)
    return BoundsInfo(polygon_4326=box(*bbox), bbox=bbox, pbf_path=pbf_path)


# BBBike city extracts (same source pyrosm uses for these keys). Direct URLs keep
# the harness usable when pyrosm cannot be installed (it needs MSVC on Windows).
BBBIKE_CITIES: dict[str, str] = {
    "Helsinki": "Helsinki",
    "Saint Petersburg": "SanktPetersburg",
    "Moscow": "Moskau",
    "London": "London",
    "Seoul": "Seoul",
    "New York": "NewYork",
}


def resolve_area_pbf(area_label: str) -> str:
    """Return a local PBF path for the area: pyrosm cache if importable, else direct BBBike download."""
    cache_dir = Path(PYROSM_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    bbbike_name = BBBIKE_CITIES[area_label]
    local_path = cache_dir / f"{bbbike_name}.osm.pbf"
    if local_path.exists():
        return str(local_path)

    try:
        from pyrosm import get_data

        return get_data(PYROSM_DATASET_KEYS[area_label], directory=str(cache_dir))
    except ImportError:
        pass

    import urllib.request

    url = f"https://download.bbbike.org/osm/bbbike/{bbbike_name}/{bbbike_name}.osm.pbf"
    print(f"[pbf] downloading {url} -> {local_path}")
    tmp_path = local_path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp_path)  # nosec - fixed benchmark source
    tmp_path.replace(local_path)
    return str(local_path)


# pyrosm get_data dataset keys (used only when pyrosm is available).
PYROSM_DATASET_KEYS: dict[str, str] = {
    "Helsinki": "helsinki",
    "Saint Petersburg": "sanktpetersburg",
    "Moscow": "moscow",
    "London": "london",
    "Seoul": "Seoul",
    "New York": "NewYorkCity",
}

# City presets shared by build/intermodal benchmarks, small to large.
AREAS: list[str] = list(BBBIKE_CITIES)


# ----------------------------
# Graph size metric
# ----------------------------


def urban_graph_memory_mb(graph) -> float:
    """Deterministic in-memory representation size of an UrbanGraph's tables, in MB.

    pandas deep memory of both tables plus the actual shapely coordinate buffers
    (``memory_usage(deep=True)`` only counts object pointers for geometry columns).
    Deterministic by construction, comparable across runs and
    across simplify on/off variants of the same graph.
    """
    import shapely

    total = float(graph.nodes_gdf.memory_usage(deep=True).sum())
    total += float(graph.edges_gdf.memory_usage(deep=True).sum())
    for frame in (graph.nodes_gdf, graph.edges_gdf):
        geometry = getattr(frame, "geometry", None)
        if geometry is not None:
            total += shapely.get_coordinates(geometry.values).nbytes
    return total / 2**20


def csr_memory_mb(matrix) -> float:
    """Deterministic in-memory representation size of a scipy CSR matrix, in MB."""
    return (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / 2**20


def networkx_graph_memory_mb(graph) -> float:
    """Deterministic deep size of a NetworkX graph object and its attributes, in MB."""
    from sys import getsizeof

    import shapely
    from shapely.geometry.base import BaseGeometry

    seen: set[int] = set()
    total = 0

    def add(obj) -> None:
        nonlocal total
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        try:
            total += getsizeof(obj)
        except TypeError:
            return
        if isinstance(obj, BaseGeometry):
            total += shapely.get_coordinates(obj).nbytes
        elif isinstance(obj, np.ndarray):
            total += obj.nbytes

    def add_mapping(mapping) -> None:
        add(mapping)
        for key, value in mapping.items():
            add(key)
            add(value)

    add(graph)
    add_mapping(getattr(graph, "graph", {}))

    nodes = getattr(graph, "_node", {})
    add(nodes)
    for node, attrs in nodes.items():
        add(node)
        add_mapping(attrs)

    # NetworkX MultiDiGraph stores nested adjacency dictionaries. Count the dict
    # shells for both successors and predecessors, and count edge attribute
    # dictionaries once from successors (pred usually points at the same attrs).
    for adjacency_name, include_edge_attrs in (("_succ", True), ("_pred", False)):
        adjacency = getattr(graph, adjacency_name, {})
        add(adjacency)
        for u, nbrs in adjacency.items():
            add(u)
            add(nbrs)
            for v, keydict in nbrs.items():
                add(v)
                add(keydict)
                for key, attrs in keydict.items():
                    add(key)
                    if include_edge_attrs:
                        add_mapping(attrs)

    return total / 2**20


# ----------------------------
# Competitor representations — honest fairness protocol
#
# Before UrbanGraph, the ecosystem pipeline was osmnx -> networkx MultiDiGraph ->
# (optionally) convert to networkit/igraph for speed. So the fair starting point
# for every competitor is a NetworkX graph, NOT UrbanGraph's CSR. We build that
# NetworkX graph once from the same UrbanGraph (identical topology and weights,
# oneway expanded into directed edges, node x/y kept for spatial snapping) and
# let each competitor pay the real costs it would pay in practice:
#   - convert: networkx -> its own graph format (min-collapse of parallel edges);
#   - snap:    build a spatial index over node coordinates + query the points.
# IduEdu instead consumes its own UrbanGraph and snaps internally via the
# persistent GeoDataFrame R-tree, so for it convert == 0 and snapping is folded
# into the routing call.
# ----------------------------


def urban_to_networkx(urban_graph, weight: str = "time_min"):
    """Directed MultiDiGraph equivalent of ``urban_graph`` (the competitors' input).

    Oneway edges become a single u->v arc; two-way edges are expanded to both
    directions, exactly as the UrbanGraph adjacency does. Node ``x``/``y`` are
    kept so competitors can build their own spatial index.
    """
    import networkx as nx

    nodes = urban_graph.nodes_gdf
    edges = urban_graph.edges_gdf

    graph = nx.MultiDiGraph()
    node_ids = nodes.index.to_numpy()
    xs = nodes.geometry.x.to_numpy()
    ys = nodes.geometry.y.to_numpy()
    graph.add_nodes_from((int(nid), {"x": float(x), "y": float(y)}) for nid, x, y in zip(node_ids, xs, ys))

    u = edges["u"].to_numpy()
    v = edges["v"].to_numpy()
    w = edges[weight].to_numpy()
    if "oneway" in edges.columns:
        oneway = edges["oneway"].to_numpy().astype(bool)
    else:
        oneway = np.ones(len(edges), dtype=bool)

    ebunch = []
    for i in range(len(u)):
        ui, vi, wi = int(u[i]), int(v[i]), float(w[i])
        ebunch.append((ui, vi, {"weight": wi}))
        if not oneway[i]:
            ebunch.append((vi, ui, {"weight": wi}))
    graph.add_edges_from(ebunch)
    return graph


def nx_min_edges_relabel(nx_graph, weight: str = "weight"):
    """Collapse parallel edges to the minimum weight and relabel nodes to 0..n-1.

    Returns ``(n, node_to_pos, src, dst, w)`` — the cost a networkit/igraph user
    pays to turn a NetworkX graph into a compact indexed edge list.
    """
    nodes = list(nx_graph.nodes())
    node_to_pos = {node: i for i, node in enumerate(nodes)}
    min_w: dict[tuple[int, int], float] = {}
    for a, b, data in nx_graph.edges(data=True):
        wv = data.get(weight)
        if wv is None:
            continue
        key = (node_to_pos[a], node_to_pos[b])
        old = min_w.get(key)
        if old is None or wv < old:
            min_w[key] = float(wv)

    m = len(min_w)
    src = np.empty(m, dtype=np.int64)
    dst = np.empty(m, dtype=np.int64)
    w = np.empty(m, dtype=np.float64)
    for i, ((a, b), wv) in enumerate(min_w.items()):
        src[i], dst[i], w[i] = a, b, wv
    return len(nodes), node_to_pos, src, dst, w


def nx_to_networkit(nx_graph, weight: str = "weight"):
    import networkit as nk

    n, node_to_pos, src, dst, w = nx_min_edges_relabel(nx_graph, weight)
    graph = nk.Graph(n, weighted=True, directed=True)
    add_edge = graph.addEdge
    for a, b, wv in zip(src.tolist(), dst.tolist(), w.tolist()):
        add_edge(a, b, wv, addMissing=False)
    return graph, node_to_pos


def nx_to_igraph(nx_graph, weight: str = "weight"):
    import igraph as ig

    n, node_to_pos, src, dst, w = nx_min_edges_relabel(nx_graph, weight)
    graph = ig.Graph(n=n, edges=list(zip(src.tolist(), dst.tolist())), directed=True)
    graph.es["weight"] = w.tolist()
    return graph, node_to_pos


def node_coords(nx_graph):
    """Node id array + Nx2 coordinate array from a competitor NetworkX graph."""
    ids = np.fromiter((int(n) for n in nx_graph.nodes()), dtype=np.int64, count=nx_graph.number_of_nodes())
    xy = np.array([(nx_graph.nodes[n]["x"], nx_graph.nodes[n]["y"]) for n in ids], dtype=float)
    return ids, xy


def gdf_points_xy(gdf, crs=None) -> np.ndarray:
    """Representative point coordinates of ``gdf`` geometries, in ``crs``.

    Uses ``representative_point()`` so polygon inputs (e.g. buildings) yield a
    valid interior point, and reprojects to ``crs`` (the graph CRS) first — this
    mirrors what IduEdu's internal snapping does, keeping the competitor snap fair.
    """
    geom = gdf.geometry
    if crs is not None and geom.crs is not None and geom.crs != crs:
        geom = geom.to_crs(crs)
    geom = geom.representative_point()
    return np.column_stack([geom.x.to_numpy(), geom.y.to_numpy()])


# ----------------------------
# Environment fingerprint (goes into results/env_*.json once per script run)
# ----------------------------


def dump_environment(tag: str) -> None:
    import importlib.metadata as md

    packages = {}
    for name in (
        "iduedu",
        "osmnx",
        "pyrosm",
        "networkx",
        "networkit",
        "igraph",
        "numba",
        "numpy",
        "pandas",
        "geopandas",
        "shapely",
        "scipy",
    ):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    info = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "packages": packages,
    }
    out = RESULTS_DIR / f"env_{tag}.json"
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[env] recorded -> {out}")
