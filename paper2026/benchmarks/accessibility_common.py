"""Shared data, graph-cache, and metric helpers for the B5/B6 experiments."""

import hashlib
import importlib.metadata as metadata
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from bench_common import RESULTS_DIR, measure
from scipy import sparse

HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parent
ROOT_DIR = PAPER_DIR.parent

ACCESSIBILITY_DIR = RESULTS_DIR / "accessibility"
GRAPH_CACHE_DIR = RESULTS_DIR / "accessibility_cache"
NETWORK_CACHE_DIR = HERE / ".iduedu_cache"
SCENARIO_DETAIL_DIR = ACCESSIBILITY_DIR / "scenario_origins"

MANIFEST_PATH = ACCESSIBILITY_DIR / "manifest.json"
ORIGINS_OUTPUT = ACCESSIBILITY_DIR / "origins_accessibility.parquet"
SCHOOLS_OUTPUT = ACCESSIBILITY_DIR / "schools.parquet"
SUMMARY_OUTPUT = ACCESSIBILITY_DIR / "accessibility_summary.csv"
TIMINGS_OUTPUT = ACCESSIBILITY_DIR / "accessibility_timings.csv"
ISOCHRONES_OUTPUT = ACCESSIBILITY_DIR / "isochrones.parquet"
STEPPED_ISOCHRONES_OUTPUT = ACCESSIBILITY_DIR / "stepped_isochrones.parquet"
STEPPED_COVERAGE_OUTPUT = ACCESSIBILITY_DIR / "stepped_coverage.parquet"
CATCHMENTS_OUTPUT = ACCESSIBILITY_DIR / "school_catchments.parquet"
MAP_WALK_EDGES_OUTPUT = ACCESSIBILITY_DIR / "map_walk_edges.parquet"
MAP_PT_EDGES_OUTPUT = ACCESSIBILITY_DIR / "map_pt_edges.parquet"
SELECTED_ORIGINS_OUTPUT = ACCESSIBILITY_DIR / "selected_origins.parquet"
VISUAL_BOUNDARY_OSM_ID = 421007
VISUAL_BOUNDARY_OUTPUT = ACCESSIBILITY_DIR / f"visual_boundary_{VISUAL_BOUNDARY_OSM_ID}.parquet"

SCENARIO_SUMMARY_OUTPUT = ACCESSIBILITY_DIR / "scenario_summary.csv"
ROUTE_CRITICALITY_OUTPUT = ACCESSIBILITY_DIR / "route_criticality.csv"
TOP_ROUTES_OUTPUT = ACCESSIBILITY_DIR / "top_routes.json"

DEFAULT_ORIGINS_PATH = PAPER_DIR / "buildings_spb.parquet"
FALLBACK_ORIGINS_PATH = ROOT_DIR / "docs" / "examples" / "data" / "spb_buildings.parquet"
DEFAULT_SCHOOLS_PATH = PAPER_DIR / "schools_spb.parquet"

THRESHOLDS_MIN = (15.0, 30.0, 45.0, 60.0)
WEIGHT = "time_min"
SEED = 42


def ensure_directories() -> None:
    """Create derived-output directories without touching the network cache contents."""
    ACCESSIBILITY_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    NETWORK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIO_DETAIL_DIR.mkdir(parents=True, exist_ok=True)


def configure_network_cache() -> None:
    """Pin IduEdu's relative Overpass cache to the benchmark directory."""
    from iduedu import config

    ensure_directories()
    config.configure_logging(level="DEBUG")
    config.set_overpass_cache(enabled=True, cache_dir=str(NETWORK_CACHE_DIR))


def load_or_download_visual_boundary() -> gpd.GeoDataFrame:
    """Load the cached historical Saint Petersburg boundary used only for map extents."""
    if VISUAL_BOUNDARY_OUTPUT.exists():
        return gpd.read_parquet(VISUAL_BOUNDARY_OUTPUT)

    from iduedu import get_4326_boundary

    configure_network_cache()
    geometry = get_4326_boundary(osm_id=VISUAL_BOUNDARY_OSM_ID)
    boundary = gpd.GeoDataFrame(
        {"osm_id": [VISUAL_BOUNDARY_OSM_ID]},
        geometry=[geometry],
        crs=4326,
    )
    boundary.to_parquet(VISUAL_BOUNDARY_OUTPUT, index=False)
    return boundary


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def stable_hash(*values: Any, length: int = 16) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, bytes):
            payload = value
        else:
            payload = str(value).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()[:length]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing {MANIFEST_PATH}; run bench_accessibility.py first")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def resolve_origins_path(path: str | None = None) -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved
    if DEFAULT_ORIGINS_PATH.exists():
        return DEFAULT_ORIGINS_PATH
    if FALLBACK_ORIGINS_PATH.exists():
        return FALLBACK_ORIGINS_PATH
    raise FileNotFoundError("No origins file found; pass --origins-path")


def load_origins(path: Path, *, smoke: bool, max_origins: int | None) -> gpd.GeoDataFrame:
    origins = gpd.read_parquet(path) if path.suffix.lower() == ".parquet" else gpd.read_file(path)
    if origins.crs is None:
        raise ValueError(f"Origins have no CRS: {path}")
    origins = origins.loc[origins.geometry.notna() & ~origins.geometry.is_empty].copy()

    living_mask = pd.Series(False, index=origins.index)
    if "is_living" in origins.columns:
        living_mask |= origins["is_living"].fillna(False).astype(bool)
    if "resident_number" in origins.columns:
        living_mask |= pd.to_numeric(origins["resident_number"], errors="coerce").fillna(0).gt(0)
    if living_mask.any():
        origins = origins.loc[living_mask].copy()

    origins = origins.reset_index(drop=False).rename(columns={"index": "source_index"})
    if "origin_id" not in origins.columns:
        id_column = "building_id" if "building_id" in origins.columns else "source_index"
        origins["origin_id"] = origins[id_column].astype(str)
    origins.index = pd.RangeIndex(len(origins), name="origin_row")

    limit = 80 if smoke else max_origins
    if limit is not None and len(origins) > limit:
        origins = origins.sample(limit, random_state=SEED).sort_index().copy()
        origins.index = pd.RangeIndex(len(origins), name="origin_row")
    if origins.empty:
        raise ValueError("No residential origins remain after filtering")
    return origins


def build_analysis_zone(origins: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    local_crs = origins.estimate_utm_crs()
    if local_crs is None:
        raise ValueError("Could not estimate a projected CRS for origins")
    local = origins.to_crs(local_crs)
    geom = local.geometry.representative_point().union_all().convex_hull.buffer(float(buffer_m))
    return gpd.GeoDataFrame({"zone_id": ["analysis_aoi"]}, geometry=[geom], crs=local_crs).to_crs(4326)


def load_schools(
    zone: gpd.GeoDataFrame,
    *,
    school_path: str | None,
    smoke: bool,
    max_schools: int | None,
) -> tuple[gpd.GeoDataFrame, str]:
    ensure_directories()
    candidate = Path(school_path).expanduser().resolve() if school_path else DEFAULT_SCHOOLS_PATH
    if not candidate.exists():
        raise FileNotFoundError(f"School input is required: {candidate}")
    stat = candidate.stat()
    cache_hash = stable_hash(
        zone.to_crs(4326).geometry.iloc[0].wkb,
        candidate,
        stat.st_size,
        stat.st_mtime_ns,
    )
    cached_path = GRAPH_CACHE_DIR / f"schools_{cache_hash}.parquet"
    if cached_path.exists():
        schools = gpd.read_parquet(cached_path)
        source = f"{candidate} (cache:{cached_path.name})"
    else:
        schools = gpd.read_parquet(candidate) if candidate.suffix.lower() == ".parquet" else gpd.read_file(candidate)
        schools = schools.to_crs(4326)
        schools = schools.loc[schools.geometry.intersects(zone.to_crs(4326).geometry.iloc[0])].copy()
        source = str(candidate)
        schools.geometry = schools.geometry.representative_point()
        if "school_id" not in schools.columns:
            if {"element_type", "osmid"}.issubset(schools.columns):
                schools["school_id"] = schools["element_type"].astype(str) + ":" + schools["osmid"].astype(str)
            else:
                schools["school_id"] = schools.index.astype(str)
        schools = schools.set_index("school_id", drop=False).rename_axis("school_row").sort_index()
        schools.to_parquet(cached_path, index=True)

    limit = 12 if smoke else max_schools
    if limit is not None and len(schools) > limit:
        schools = schools.sample(limit, random_state=SEED).sort_index().copy()
    if schools.empty:
        raise ValueError("No schools intersect the analysis AOI")
    return schools, source


def _walk_graph_from_intermodal(intermodal):
    from iduedu import UrbanGraph

    edges = intermodal.edges_gdf
    if "type" not in edges.columns:
        raise KeyError("Intermodal graph has no edge 'type' column")
    walk_edges = edges.loc[edges["type"].astype(str).eq("walk")].copy()
    node_ids = pd.Index(walk_edges["u"]).append(pd.Index(walk_edges["v"])).unique()
    walk_nodes = intermodal.nodes_gdf.loc[node_ids].copy()
    return UrbanGraph(
        nodes_gdf=walk_nodes,
        edges_gdf=walk_edges,
        is_multigraph=intermodal.is_multigraph,
        is_directed=intermodal.is_directed,
        edge_direction_column=intermodal.edge_direction_column,
        adjacency_weight=WEIGHT,
        crs=intermodal.crs,
        graph_type="walk",
    )


def load_or_build_graphs(zone: gpd.GeoDataFrame, *, smoke: bool) -> tuple[Any, Any, dict]:
    from iduedu import get_intermodal_graph, read_urban_graph, write_urban_graph

    ensure_directories()
    configure_network_cache()
    graph_hash = stable_hash(
        zone.to_crs(4326).geometry.iloc[0].wkb,
        f"simplify={not smoke}",
        "clip=True",
        "boarding=1.0",
        package_version("iduedu"),
    )
    intermodal_path = GRAPH_CACHE_DIR / f"intermodal_{graph_hash}.urbangraph"
    walk_path = GRAPH_CACHE_DIR / f"walk_{graph_hash}.urbangraph"
    cache_metadata_path = GRAPH_CACHE_DIR / f"graph_{graph_hash}.json"
    timings: dict[str, Any] = {"graph_hash": graph_hash}

    if intermodal_path.exists() and walk_path.exists():
        print(f"[graph] loading intermodal cache without validation: {intermodal_path.name}", flush=True)
        m_intermodal = measure(read_urban_graph, intermodal_path, validate=False)
        print(f"[graph] intermodal loaded in {m_intermodal.time_sec:.2f}s", flush=True)
        print(f"[graph] loading walk cache without validation: {walk_path.name}", flush=True)
        m_walk = measure(read_urban_graph, walk_path, validate=False)
        print(f"[graph] walk loaded in {m_walk.time_sec:.2f}s", flush=True)
        intermodal, walk = m_intermodal.result, m_walk.result
        cache_metadata = (
            json.loads(cache_metadata_path.read_text(encoding="utf-8")) if cache_metadata_path.exists() else {}
        )
        timings.update(
            source="warm_load",
            intermodal_load_sec=m_intermodal.time_sec,
            walk_load_sec=m_walk.time_sec,
            build_sec=0.0,
            write_sec=0.0,
            cold_build_sec=cache_metadata.get("cold_build_sec"),
            cold_write_sec=cache_metadata.get("cold_write_sec"),
        )
        print(f"[graph] loaded cached graphs ({graph_hash})", flush=True)
    else:
        print("[graph] building intermodal graph; .iduedu_cache is preserved")
        m_build = measure(
            get_intermodal_graph,
            territory=zone.to_crs(4326).geometry.iloc[0],
            clip_by_territory=True,
            keep_largest_subgraph=True,
            walk_kwargs={"simplify": not smoke, "keep_largest_subgraph": False},
            pt_kwargs={"avg_boarding_time_min": 1.0},
        )
        intermodal = m_build.result
        walk = _walk_graph_from_intermodal(intermodal)
        intermodal.update_adjacency_matrix(weight=WEIGHT)
        walk.update_adjacency_matrix(weight=WEIGHT)
        t0 = time.perf_counter()
        write_urban_graph(intermodal, intermodal_path, include_adjacency=True)
        write_urban_graph(walk, walk_path, include_adjacency=True)
        write_sec = time.perf_counter() - t0
        timings.update(
            source="cold_build",
            intermodal_load_sec=0.0,
            walk_load_sec=0.0,
            build_sec=m_build.time_sec,
            write_sec=write_sec,
            cold_build_sec=m_build.time_sec,
            cold_write_sec=write_sec,
        )
        write_json(
            cache_metadata_path,
            {
                "graph_hash": graph_hash,
                "cold_build_sec": m_build.time_sec,
                "cold_write_sec": write_sec,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
        print(f"[graph] cached graphs ({graph_hash})")

    timings.update(
        intermodal_path=str(intermodal_path),
        walk_path=str(walk_path),
        intermodal_nodes=len(intermodal.nodes_gdf),
        intermodal_edges=len(intermodal.edges_gdf),
        walk_nodes=len(walk.nodes_gdf),
        walk_edges=len(walk.edges_gdf),
    )
    return walk, intermodal, timings


def nearest_school_metrics(
    graph,
    origins: gpd.GeoDataFrame,
    schools: gpd.GeoDataFrame,
    *,
    origin_node_column: str | None = None,
    school_node_column: str | None = None,
) -> tuple[pd.DataFrame, float]:
    from iduedu import multi_source_dijkstra_nearest_source

    origin_nodes = (
        origins[origin_node_column]
        if origin_node_column is not None and origin_node_column in origins.columns
        else graph.nearest_nodes(origins)
    )
    school_nodes = (
        schools[school_node_column]
        if school_node_column is not None and school_node_column in schools.columns
        else graph.nearest_nodes(schools)
    )
    unique_school_nodes = school_nodes.loc[~school_nodes.duplicated(keep="first")]

    m_routing = measure(
        multi_source_dijkstra_nearest_source,
        graph,
        source_nodes=unique_school_nodes.to_list(),
        weight=WEIGHT,
        reverse=True,
    )
    nearest = m_routing.result
    distances = nearest["dist"].sparse.to_dense()
    node_to_school = {node: school_index for school_index, node in unique_school_nodes.items()}

    node_values = origin_nodes.to_numpy()
    result = pd.DataFrame(index=origins.index)
    result["graph_node_id"] = node_values
    result["nearest_time_min"] = distances.reindex(node_values, fill_value=np.inf).to_numpy(dtype=float)
    nearest_source_nodes = nearest["source_node"].reindex(node_values)
    result["nearest_school_id"] = nearest_source_nodes.map(node_to_school).to_numpy()
    return result, m_routing.time_sec


def school_opportunity_counts(
    graph,
    origins: pd.DataFrame,
    schools: pd.DataFrame,
    *,
    origin_node_column: str,
    school_node_column: str,
    thresholds: tuple[float, ...] = THRESHOLDS_MIN,
) -> tuple[pd.DataFrame, float]:
    """Count reachable schools from one complete origin-destination matrix."""
    from iduedu import od_matrix

    origin_nodes = pd.DataFrame(
        {"graph_node_id": origins[origin_node_column].to_numpy()},
        index=origins.index,
    )
    school_nodes = pd.DataFrame(
        {"graph_node_id": schools[school_node_column].to_numpy()},
        index=schools.index,
    )
    measurement = measure(
        od_matrix,
        graph,
        gdf_origins=origin_nodes,
        gdf_destinations=school_nodes,
        weight=WEIGHT,
        threshold=max(thresholds),
    )
    matrix = measurement.result
    counts_by_threshold = {threshold: np.zeros(len(matrix), dtype=np.int32) for threshold in thresholds}
    for column in matrix.columns:
        sparse_array = matrix[column].array
        row_positions = np.asarray(sparse_array.sp_index.indices, dtype=np.int64)
        values = np.asarray(sparse_array.sp_values)
        finite = np.isfinite(values)
        row_positions = row_positions[finite]
        values = values[finite]
        for threshold in thresholds:
            reached_rows = row_positions[values <= threshold]
            counts_by_threshold[threshold][reached_rows] += 1

    counts = pd.DataFrame(
        {f"schools_within_{int(threshold)}": counts_by_threshold[threshold] for threshold in thresholds},
        index=origins.index,
    )
    return counts, measurement.time_sec


def od_school_counts(
    graph,
    origins: gpd.GeoDataFrame,
    schools: gpd.GeoDataFrame,
    *,
    label: str,
    origin_node_column: str,
    school_node_column: str,
) -> tuple[pd.DataFrame, float, Path]:
    counts, routing_sec = school_opportunity_counts(
        graph,
        origins,
        schools,
        origin_node_column=origin_node_column,
        school_node_column=school_node_column,
    )
    counts_path = ACCESSIBILITY_DIR / f"od_{label}_counts.parquet"
    counts.to_parquet(counts_path, index=True)
    return counts, routing_sec, counts_path


def save_sparse_frame(frame: pd.DataFrame, path: Path) -> None:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for column_i, column in enumerate(frame.columns):
        array = frame[column].array
        row_positions = np.asarray(array.sp_index.indices, dtype=np.int64)
        finite_values = np.asarray(array.sp_values, dtype=np.float32)
        finite = np.isfinite(finite_values)
        if finite.any():
            rows.append(row_positions[finite])
            cols.append(np.full(int(finite.sum()), column_i, dtype=np.int64))
            values.append(finite_values[finite])
    if rows:
        matrix = sparse.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=frame.shape,
        ).tocsr()
    else:
        matrix = sparse.csr_matrix(frame.shape, dtype=np.float32)
    sparse.save_npz(path, matrix)
    write_json(
        path.with_suffix(".index.json"),
        {"index": frame.index.astype(str).tolist(), "columns": frame.columns.astype(str).tolist()},
    )


def slugify(value: Any, max_length: int = 80) -> str:
    text = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", str(value)).strip("_")
    if not text:
        text = "unnamed"
    if len(text) > max_length:
        text = f"{text[: max_length - 9]}_{stable_hash(text, length=8)}"
    return text


def route_equals(series: pd.Series, route: Any) -> pd.Series:
    route_string = str(route)

    def matches(value: Any) -> bool:
        if isinstance(value, (list, tuple, set)):
            return any(str(item) == route_string for item in value)
        return pd.notna(value) and str(value) == route_string

    return series.map(matches)


def finite_percentile(values: Iterable[float], percentile: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, percentile)) if len(array) else None
