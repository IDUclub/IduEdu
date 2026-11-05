"""Accessibility from OpenStreetMap against accessibility from the timetable.

Cumulative opportunities: 1 500 points drawn once per city from its walk network
(seed fixed), the first 100 of them origins; each origin counts the points it reaches
within 15, 30 and 45 minutes. Every variant joins its transit graph to the **same**
walk graph and keeps every component, so the variants differ only in their transit:

======  ========  ==============================  ====================
variant transit   waiting time                    run time
======  ========  ==============================  ====================
D       timetable the timetable's, per stop       the timetable's
D1      timetable the city's mean per mode        the timetable's
A_w     OSM       the city's mean per mode        the registry's model
A       OSM       the registry's (world median)   the registry's model
======  ========  ==============================  ====================

A/D is the study's comparison; its logarithm is exactly the sum of the three steps
(D->D1 averaging waits, D1->A_w the data, A_w->A transferred waits). A city with no
OSM transit takes the walk network alone.

* ``graphs`` -- OSM transit graphs with the city's own waits (``A_w``).
* ``measure`` -> ``accessibility_variants.csv``, ``accessibility.csv`` (A against D),
  ``accessibility_variants_summary.csv``.
* ``decompose`` -> ``accessibility_decomposition.csv``: bias (mean log per step) and
  share of the spread (cov(step, total) / var(total), shares sum to one).

    python pipeline/access.py graphs | measure | decompose
"""

import argparse
import copy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from common import (
    ACCESSIBILITY_CSV,
    ALL_OSM_MODES,
    RESULTS_DIR,
    THRESHOLDS_MIN,
    VARIANT_GRAPH_DIR,
    VARIANTS_CSV,
    WAITS_CSV,
    boundary,
    city_pbf_path,
    gtfs_graph_path,
    last_rows,
    measurable,
    osm_graph_path,
    restrict,
    setup,
    with_gtfs_graph,
    with_osm_graph,
)
from scipy.spatial import cKDTree

from iduedu import DEFAULT_REGISTRY_W_TRAIN, TransportRegistry, read_urban_graph

N_DESTINATIONS, N_ORIGINS, SEED = 1500, 100, 20260811
ORDER = ("D", "D1", "A_w", "A")
STEPS = [("D", "D1", "averaging waits"), ("D1", "A_w", "data (OSM vs timetable)"), ("A_w", "A", "transferred waits")]
STRUCTURE_TYPES = {"platform", "station", "entrance", "exit", "boarding_area", "node"}


# --- measuring -------------------------------------------------------------------


def sample_points(walk_nodes) -> np.ndarray:
    chosen = np.random.default_rng(SEED).choice(len(walk_nodes), min(N_DESTINATIONS, len(walk_nodes)), replace=False)
    geometry = walk_nodes.geometry.iloc[chosen]
    return np.column_stack([geometry.x.to_numpy(), geometry.y.to_numpy()])


def _nearest_nodes(graph, points: np.ndarray) -> np.ndarray:
    nodes = graph.nodes_gdf
    _, indices = cKDTree(np.column_stack([nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()])).query(points)
    return nodes.index.to_numpy()[indices]


def mean_reach(transit, walk, points: np.ndarray) -> dict[int, float]:
    """Mean number of points reached per origin within each threshold."""
    from iduedu.graph.shortest_paths import single_source_dijkstra_path_length
    from iduedu.graph_builders.intermodal_builders import join_pt_walk_graph

    graph = (
        walk
        if transit is None or transit.nodes_gdf.empty
        else join_pt_walk_graph(transit, walk, keep_largest_subgraph=False)
    )
    destinations = pd.Index(_nearest_nodes(graph, points))
    counts = {threshold: [] for threshold in THRESHOLDS_MIN}
    for origin in _nearest_nodes(graph, points[:N_ORIGINS]):
        try:
            lengths = single_source_dijkstra_path_length(
                graph, source_node=origin, weight="time_min", cutoff=float(max(THRESHOLDS_MIN))
            )
            reached = lengths.reindex(destinations).to_numpy(dtype=float)
        except Exception:  # noqa: BLE001 - an unreachable origin reaches nothing
            reached = np.full(len(destinations), np.inf)
        for threshold in THRESHOLDS_MIN:
            counts[threshold].append(int(np.count_nonzero(reached <= threshold)))
    return {threshold: round(float(np.mean(values)), 2) for threshold, values in counts.items()}


# --- variant graphs ----------------------------------------------------------------


def variant_path(city_key: str):
    return VARIANT_GRAPH_DIR / "w" / f"{city_key}__pt.urbangraph"


def city_waits() -> dict[str, dict[str, float]]:
    """Each city's timetable wait per mode (``waits.py measure``): city -> mode -> minutes."""
    waits = last_rows(WAITS_CSV, ["city_key", "mode"], keep_default_na=False)
    waits = waits[(waits["status"] == "ok") & (waits["mode"] != "*")]
    waits["wait_harmonic_min"] = pd.to_numeric(waits["wait_harmonic_min"], errors="coerce")
    return {key: dict(zip(group["mode"], group["wait_harmonic_min"])) for key, group in waits.groupby("city_key")}


def city_registry(waits: dict[str, float]) -> TransportRegistry:
    """The registry with the city's own wait for every mode it has one for."""
    registry = copy.deepcopy(DEFAULT_REGISTRY_W_TRAIN)
    for mode, wait in waits.items():
        if registry.try_get(mode) is not None and wait == wait and wait > 0:
            registry.update(mode, avg_wait_time_min=float(wait))
    return registry


def build_transit(city_key: str, registry, destination) -> dict:
    """The city's OSM transit graph under the given registry, as ``osm.py`` builds it."""
    from iduedu import config, get_public_transport_graph, write_urban_graph

    row = {"city_key": city_key, "status": "failed"}
    with config.use_pbf(city_pbf_path(city_key)):
        graph = get_public_transport_graph(
            territory=boundary(city_key),
            transport_types=ALL_OSM_MODES,
            clip_by_territory=True,
            transport_registry=registry,
        )
    if graph.nodes_gdf.empty:
        row["status"] = "empty"
        return row
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_urban_graph(graph, destination)
    row.update(status="ok", n_nodes=len(graph.nodes_gdf), n_edges=len(graph.edges_gdf))
    return row


def _build_variant(city_key: str, waits: dict[str, float]) -> dict:
    return {"variant": "A_w", **build_transit(city_key, city_registry(waits), variant_path(city_key))}


def run_parallel(logger, jobs: list[tuple], function, workers: int) -> list[dict]:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, *job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
            except Exception as error:  # noqa: BLE001 - one city must not stop the sweep
                row = {
                    "city_key": futures[future][0],
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}"[:300],
                }
            rows.append(row)
            logger.info("[%d/%d] %s: %s", index, len(jobs), " ".join(map(str, futures[future])), row.get("status"))
    return rows


# --- four variants -----------------------------------------------------------------


def averaged_waits(gtfs, waits: dict[str, float]):
    """The timetable graph with every boarding edge at the city's mean wait of its mode."""
    edges = gtfs.edges_gdf.copy()
    modes = edges.loc[edges["type"].eq("boarding"), "v"].map(gtfs.nodes_gdf["type"])
    values = modes.map(waits)
    replace = values.notna() & ~modes.isin(STRUCTURE_TYPES)
    edges.loc[values[replace].index, "time_min"] = values[replace].astype(float)
    gtfs.edges_gdf = edges
    return gtfs


def measure_variants(city_key: str, waits: dict[str, float]) -> dict:
    paths = {"walk": osm_graph_path(city_key, "walk"), "A": osm_graph_path(city_key, "pt"),
             "A_w": variant_path(city_key), "D": gtfs_graph_path(city_key)}  # fmt: skip
    missing = [name for name, path in paths.items() if not path.exists()]
    if "walk" in missing or "D" in missing:
        return {"city_key": city_key, "status": "missing_graph:" + "+".join(missing)}
    started = time.perf_counter()
    walk = read_urban_graph(paths["walk"])
    transit = {name: read_urban_graph(paths[name]) for name in ("A", "A_w", "D") if name not in missing}
    transit["D1"] = averaged_waits(read_urban_graph(paths["D"]), waits)
    points = sample_points(walk.nodes_gdf)
    row = {"city_key": city_key, "n_origins": min(N_ORIGINS, len(points)), "n_destinations": len(points)}
    for name in ORDER:
        graph = transit.get(name)
        if name == "A_w" and graph is None and "A" not in missing:
            row[f"{name}_transit"] = None  # the variant was not built: no estimate, not "walk only"
            continue
        row.update({f"{name}_{t}": value for t, value in mean_reach(graph, walk, points).items()})
        row[f"{name}_transit"] = graph is not None and not graph.nodes_gdf.empty
    row.update(elapsed_s=round(time.perf_counter() - started, 1), status="ok")
    return row


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    ok, rows = table[table["status"] == "ok"], []
    for threshold in THRESHOLDS_MIN:
        for before, after in [(a, b) for a, b, _ in STEPS] + [("D", "A")]:
            a, b = ok[f"{before}_{threshold}"].astype(float), ok[f"{after}_{threshold}"].astype(float)
            ratio = (b / a).replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({"threshold_min": threshold, "step": f"{before}->{after}", "cities": int(ratio.size),
                         "median_ratio": round(float(ratio.median()), 4), "p25": round(float(ratio.quantile(0.25)), 4),
                         "p75": round(float(ratio.quantile(0.75)), 4),
                         "within_10pct": round(float(((ratio - 1).abs() <= 0.1).mean()), 3),
                         "spearman": round(float(a.corr(b, method="spearman")), 3)})  # fmt: skip
    return pd.DataFrame(rows)


def run_measure(logger, keys: list[str], workers: int) -> None:
    by_city = city_waits()
    rows = run_parallel(logger, [(key, by_city.get(key, {})) for key in keys], measure_variants, workers)
    if VARIANTS_CSV.exists():
        earlier = pd.read_csv(VARIANTS_CSV, keep_default_na=False)
        rows = earlier[~earlier["city_key"].isin(keys)].to_dict("records") + rows
    table = pd.DataFrame(rows).sort_values("city_key")
    table.to_csv(VARIANTS_CSV, index=False)
    comparison = table[["city_key", "status"]].copy()  # A against D: the table the other stages read
    for t in THRESHOLDS_MIN:
        a, d = pd.to_numeric(table.get(f"A_{t}"), errors="coerce"), pd.to_numeric(table.get(f"D_{t}"), errors="coerce")
        comparison[f"osm_mean_{t}"], comparison[f"gtfs_mean_{t}"], comparison[f"ratio_{t}"] = a, d, (a / d).round(4)
    comparison.to_csv(ACCESSIBILITY_CSV, index=False)
    summarise(table).to_csv(RESULTS_DIR / "accessibility_variants_summary.csv", index=False)


# --- decomposition -----------------------------------------------------------------


def decompose(table: pd.DataFrame) -> pd.DataFrame:
    ok, rows = table[table["status"] == "ok"], []
    for threshold in THRESHOLDS_MIN:
        values = {name: ok[f"{name}_{threshold}"].astype(float) for name in ORDER}
        usable = np.logical_and.reduce([v > 0 for v in values.values()])
        total = np.log(values["A"][usable] / values["D"][usable])
        variance = float(total.var(ddof=0))
        base = {"threshold_min": threshold, "cities": int(usable.sum()), "cities_without_log": int((~usable).sum())}
        for a, b, label in STEPS:
            step = np.log(values[b][usable] / values[a][usable])
            rows.append({**base, "step": f"{a}->{b}", "meaning": label, "mean_log": round(float(step.mean()), 4),
                         "geometric_change_pct": round(float(np.expm1(step.mean()) * 100), 2),
                         "share_of_spread": round(float(np.cov(step, total, ddof=0)[0, 1] / variance), 3) if variance else None})  # fmt: skip
        rows.append({**base, "step": "D->A", "meaning": "total", "mean_log": round(float(total.mean()), 4),
                     "geometric_change_pct": round(float(np.expm1(total.mean()) * 100), 2), "share_of_spread": 1.0})  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["graphs", "measure", "decompose"])
    parser.add_argument("--cities", nargs="*")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    logger = setup("access")
    timed = restrict(sorted(set(measurable()) & with_gtfs_graph()), arguments.cities)
    if arguments.step == "graphs":
        waits = city_waits()
        keys = [key for key in timed if key in with_osm_graph() and (arguments.force or not variant_path(key).exists())]
        rows = run_parallel(logger, [(key, waits.get(key, {})) for key in keys], _build_variant, arguments.workers)
        if rows:
            pd.DataFrame(rows).to_csv(RESULTS_DIR / "osm_graphs_variants.csv", index=False)
    elif arguments.step == "measure":
        if VARIANTS_CSV.exists() and not arguments.force:
            done = pd.read_csv(VARIANTS_CSV, keep_default_na=False)
            timed = [key for key in timed if key not in set(done.loc[done["status"] == "ok", "city_key"])]
        run_measure(logger, timed, arguments.workers)
    else:
        result = decompose(pd.read_csv(VARIANTS_CSV, keep_default_na=False))
        result.to_csv(RESULTS_DIR / "accessibility_decomposition.csv", index=False)
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
