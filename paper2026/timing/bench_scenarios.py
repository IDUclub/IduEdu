#!/usr/bin/env python3
"""B6 — P0 waiting-time sensitivity and P1 leave-one-route-out screening.

Prerequisite: run ``bench_accessibility.py``. The script reuses the cached
intermodal graph and baseline origin metrics. Each scenario only edits a copy of
``edges_gdf``, rebuilds the CSR, and repeats nearest-school and school-opportunity routing.
"""

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from accessibility_common import (
    ORIGINS_OUTPUT,
    ROUTE_CRITICALITY_OUTPUT,
    SCENARIO_DETAIL_DIR,
    SCENARIO_SUMMARY_OUTPUT,
    SCHOOLS_OUTPUT,
    TOP_ROUTES_OUTPUT,
    WEIGHT,
    ensure_directories,
    finite_percentile,
    nearest_school_metrics,
    read_manifest,
    route_equals,
    school_opportunity_counts,
    slugify,
    stable_hash,
    write_json,
)
from bench_common import append_row, dump_environment, measure

PT_TYPES = ("bus", "tram", "trolleybus", "subway", "train")
SCENARIO_THRESHOLDS_MIN = (15.0, 30.0)
P1_MAP_MODES = ("subway", "bus", "tram")


def normalized_route_frame(edges: gpd.GeoDataFrame) -> pd.DataFrame:
    if "route" not in edges.columns or "type" not in edges.columns:
        raise KeyError("Scenario screening requires edge columns 'route' and 'type'")
    routes = edges.loc[edges["type"].astype(str).isin(PT_TYPES), ["type", "route", "length_meter"]].copy()
    routes = routes.explode("route")
    routes = routes.loc[routes["route"].notna()].copy()
    routes["mode"] = routes["type"].astype(str)
    routes["route"] = routes["route"].astype(str)
    routes = routes.loc[~routes["route"].isin({"", "nan", "None", "<NA>"})]
    return routes


def route_catalog(edges: gpd.GeoDataFrame) -> pd.DataFrame:
    routes = normalized_route_frame(edges)
    catalog = (
        routes.groupby(["mode", "route"], as_index=False)
        .agg(n_edges=("route", "size"), length_km=("length_meter", lambda values: values.sum() / 1000.0))
        .sort_values(["length_km", "n_edges"], ascending=False)
    )
    mode_count = catalog.groupby("route")["mode"].nunique().rename("n_modes_for_route")
    return catalog.join(mode_count, on="route")


def choose_mode_and_route(
    catalog: pd.DataFrame,
    requested_mode: str,
    requested_route: str,
) -> tuple[str, str]:
    if catalog.empty:
        raise RuntimeError("Intermodal graph contains no route-labelled PT travel edges")
    if requested_route != "auto":
        candidates = catalog.loc[catalog["route"].astype(str).eq(str(requested_route))]
        if requested_mode != "auto":
            candidates = candidates.loc[candidates["mode"].eq(requested_mode)]
        if candidates.empty:
            raise ValueError(f"Route {requested_route!r} is absent for mode {requested_mode!r}")
        selected = candidates.sort_values("length_km", ascending=False).iloc[0]
        return str(selected["mode"]), str(selected["route"])
    if requested_mode == "auto":
        mode = catalog.groupby("mode")["length_km"].sum().sort_values(ascending=False).index[0]
    else:
        mode = requested_mode
        if mode not in set(catalog["mode"]):
            raise ValueError(f"Mode {mode!r} is absent; available={sorted(catalog['mode'].unique())}")
    candidates = catalog.loc[(catalog["mode"] == mode) & (catalog["n_modes_for_route"] == 1)]
    if candidates.empty:
        candidates = catalog.loc[catalog["mode"] == mode]
    return mode, str(candidates.iloc[0]["route"])


def boarding_mask(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    mode: str,
    route: str | None = None,
) -> pd.Series:
    """Select boarding edges by the PT mode stored on their destination nodes."""
    is_boarding = edges["type"].astype(str).eq("boarding")
    destination_mode = edges["v"].map(nodes["type"]).astype(str)
    mask = is_boarding & destination_mode.eq(mode)
    if route is not None:
        mask &= route_equals(edges["route"], route)
    return mask


def scenario_metrics(
    baseline_time: np.ndarray,
    scenario_time: np.ndarray,
    baseline_counts: pd.DataFrame,
    scenario_counts: pd.DataFrame,
    *,
    n_origins: int,
) -> dict:
    both_finite = np.isfinite(baseline_time) & np.isfinite(scenario_time)
    increase = np.full(len(baseline_time), np.nan, dtype=float)
    increase[both_finite] = scenario_time[both_finite] - baseline_time[both_finite]
    finite_increase = increase[np.isfinite(increase)]
    row = {
        "n_origins": n_origins,
        "n_finite_baseline": int(np.isfinite(baseline_time).sum()),
        "n_finite_scenario": int(np.isfinite(scenario_time).sum()),
        "n_affected_gt_1min": int(np.sum(increase > 1.0)),
        "affected_gt_1min_share": float(np.mean(increase > 1.0)),
        "mean_increase_min": float(np.mean(finite_increase)) if len(finite_increase) else None,
        "median_increase_min": finite_percentile(finite_increase, 50),
        "p90_increase_min": finite_percentile(finite_increase, 90),
        "max_increase_min": float(np.max(finite_increase)) if len(finite_increase) else None,
    }
    for threshold in SCENARIO_THRESHOLDS_MIN:
        threshold = int(threshold)
        baseline = baseline_counts[f"intermodal_schools_{threshold}"].to_numpy(dtype=int)
        scenario = scenario_counts[f"schools_within_{threshold}"].to_numpy(dtype=int)
        opportunity_loss = np.maximum(baseline - scenario, 0)
        origins_losing = opportunity_loss > 0
        newly_unreachable = (baseline > 0) & (scenario == 0)
        row[f"n_origins_losing_opportunities_{threshold}"] = int(origins_losing.sum())
        row[f"origins_losing_opportunities_{threshold}_share"] = float(origins_losing.mean())
        row[f"total_school_opportunities_lost_{threshold}"] = int(opportunity_loss.sum())
        row[f"mean_school_opportunities_lost_{threshold}"] = float(opportunity_loss.mean())
        row[f"max_school_opportunities_lost_{threshold}"] = int(opportunity_loss.max(initial=0))
        row[f"n_newly_unreachable_{threshold}"] = int(newly_unreachable.sum())
        row[f"newly_unreachable_{threshold}_share"] = float(newly_unreachable.mean())
    return row


def save_scenario_detail(
    origins: gpd.GeoDataFrame,
    scenario_time: np.ndarray,
    scenario_counts: pd.DataFrame,
    *,
    scenario_id: str,
) -> Path:
    detail = origins.copy()
    baseline = detail["intermodal_time_min"].to_numpy(dtype=float)
    both_finite = np.isfinite(baseline) & np.isfinite(scenario_time)
    increase = np.full(len(detail), np.nan, dtype=float)
    increase[both_finite] = scenario_time[both_finite] - baseline[both_finite]
    detail["scenario_id"] = scenario_id
    detail["scenario_time_min"] = scenario_time
    detail["scenario_increase_min"] = increase
    for threshold in SCENARIO_THRESHOLDS_MIN:
        threshold = int(threshold)
        baseline_count = detail[f"intermodal_schools_{threshold}"].to_numpy(dtype=int)
        scenario_count = scenario_counts[f"schools_within_{threshold}"].to_numpy(dtype=int)
        detail[f"scenario_schools_{threshold}"] = scenario_count
        detail[f"lost_school_opportunities_{threshold}"] = np.maximum(baseline_count - scenario_count, 0)
        detail[f"newly_unreachable_{threshold}"] = (baseline_count > 0) & (scenario_count == 0)
    keep = [
        column
        for column in (
            "origin_id",
            "population",
            "gain_class",
            "intermodal_time_min",
            "scenario_id",
            "scenario_time_min",
            "scenario_increase_min",
            "intermodal_schools_15",
            "intermodal_schools_30",
            "scenario_schools_15",
            "scenario_schools_30",
            "lost_school_opportunities_15",
            "lost_school_opportunities_30",
            "newly_unreachable_15",
            "newly_unreachable_30",
            "geometry",
        )
        if column in detail.columns
    ]
    path = SCENARIO_DETAIL_DIR / f"{slugify(scenario_id)}.parquet"
    detail[keep].to_parquet(path, index=True)
    return path


def execute_scenario(
    base_graph,
    origins: gpd.GeoDataFrame,
    schools: gpd.GeoDataFrame,
    *,
    scenario_id: str,
    edge_mask: pd.Series,
    operation: str,
    extra_wait_min: float | None,
    metadata: dict,
    experiment_hash: str,
    save_detail: bool,
) -> tuple[dict, np.ndarray]:
    graph = base_graph.copy()
    t0 = time.perf_counter()
    if operation == "add_wait":
        graph.edges_gdf.loc[edge_mask, WEIGHT] = graph.edges_gdf.loc[edge_mask, WEIGHT].astype(float) + float(
            extra_wait_min
        )
    elif operation == "remove_edges":
        graph.edges_gdf = graph.edges_gdf.loc[~edge_mask].copy()
    else:
        raise ValueError(operation)
    edit_sec = time.perf_counter() - t0

    m_csr = measure(graph.update_adjacency_matrix, weight=WEIGHT)
    nearest, nearest_routing_sec = nearest_school_metrics(
        graph,
        origins,
        schools,
        origin_node_column="intermodal_node_id",
        school_node_column="intermodal_node_id",
    )
    scenario_counts, opportunity_routing_sec = school_opportunity_counts(
        graph,
        origins,
        schools,
        origin_node_column="intermodal_node_id",
        school_node_column="intermodal_node_id",
        thresholds=SCENARIO_THRESHOLDS_MIN,
    )
    scenario_time = nearest["nearest_time_min"].to_numpy(dtype=float)
    baseline_time = origins["intermodal_time_min"].to_numpy(dtype=float)
    row = {
        "experiment_hash": experiment_hash,
        "scenario_id": scenario_id,
        "scenario_group": metadata["scenario_group"],
        "mode": metadata.get("mode"),
        "route": metadata.get("route"),
        "operation": operation,
        "extra_wait_min": extra_wait_min,
        "n_edges_changed": int(edge_mask.sum()),
        "time_edit_sec": edit_sec,
        "time_csr_sec": m_csr.time_sec,
        "time_nearest_routing_sec": nearest_routing_sec,
        "time_opportunity_routing_sec": opportunity_routing_sec,
        "time_routing_sec": nearest_routing_sec + opportunity_routing_sec,
        "time_total_sec": edit_sec + m_csr.time_sec + nearest_routing_sec + opportunity_routing_sec,
        **scenario_metrics(
            baseline_time,
            scenario_time,
            origins,
            scenario_counts,
            n_origins=len(origins),
        ),
    }
    if save_detail:
        row["detail_path"] = str(save_scenario_detail(origins, scenario_time, scenario_counts, scenario_id=scenario_id))
    del graph, nearest, scenario_counts, m_csr
    return row, scenario_time


def load_existing_ids(experiment_hash: str) -> set[str]:
    if not SCENARIO_SUMMARY_OUTPUT.exists():
        return set()
    frame = pd.read_csv(SCENARIO_SUMMARY_OUTPUT, dtype=str, keep_default_na=False)
    if "experiment_hash" not in frame.columns or "scenario_id" not in frame.columns:
        return set()
    return set(frame.loc[frame["experiment_hash"].eq(experiment_hash), "scenario_id"])


def current_summary(experiment_hash: str) -> pd.DataFrame:
    if not SCENARIO_SUMMARY_OUTPUT.exists():
        return pd.DataFrame()
    frame = pd.read_csv(SCENARIO_SUMMARY_OUTPUT)
    return frame.loc[frame["experiment_hash"].astype(str).eq(experiment_hash)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mode", default="auto", choices=["auto", *PT_TYPES])
    parser.add_argument("--route", default="auto", help="P0 route ref; auto selects the longest route for the mode")
    parser.add_argument("--extra-wait", default="5,10", help="comma-separated P0 penalties in minutes")
    parser.add_argument("--max-routes", type=int, default=0, help="P1 routes by network length; 0 means all")
    parser.add_argument("--skip-p1", action="store_true", help="run P0 scenarios without repeating P1 screening")
    args = parser.parse_args()

    ensure_directories()
    dump_environment("scenarios")
    manifest = read_manifest()

    from iduedu import read_urban_graph

    graph_path = Path(manifest["graph"]["intermodal_path"])
    graph = read_urban_graph(graph_path)
    origins = gpd.read_parquet(ORIGINS_OUTPUT)
    schools = gpd.read_parquet(SCHOOLS_OUTPUT)
    catalog = route_catalog(graph.edges_gdf)
    if catalog.empty:
        raise RuntimeError("No PT routes found in the cached graph")
    mode, selected_route = choose_mode_and_route(catalog, args.mode, args.route)
    print(f"[scenario] P0 mode={mode!r}, route={selected_route!r}")

    extras = [float(value.strip()) for value in args.extra_wait.split(",") if value.strip()]
    if args.smoke:
        extras = extras[:1]
    max_routes = 3 if args.smoke else args.max_routes
    experiment_hash = stable_hash(
        "school-opportunity-v4-cutoff30-mode-aware-boarding",
        manifest["graph"]["graph_hash"],
        len(origins),
        len(schools),
        extras,
        max_routes,
        mode,
        selected_route,
        SCENARIO_THRESHOLDS_MIN,
    )
    existing = load_existing_ids(experiment_hash)

    mode_boarding_mask = boarding_mask(graph.edges_gdf, graph.nodes_gdf, mode=mode)
    route_boarding_mask = boarding_mask(
        graph.edges_gdf,
        graph.nodes_gdf,
        mode=mode,
        route=selected_route,
    )
    p0_specs = [
        ("mode", mode, mode_boarding_mask, {"scenario_group": "P0_mode_wait", "mode": mode, "route": None}),
        (
            "route",
            selected_route,
            route_boarding_mask,
            {"scenario_group": "P0_route_wait", "mode": mode, "route": selected_route},
        ),
    ]
    for scope, value, mask, metadata_row in p0_specs:
        if not mask.any():
            print(f"[warn] no boarding edges for P0 {scope}={value!r}; skipping")
            continue
        for extra in extras:
            scenario_id = f"p0_{scope}_{slugify(value)}_wait_plus_{extra:g}"
            if scenario_id in existing:
                print(f"[skip] {scenario_id}")
                continue
            print(f"[run ] {scenario_id} edges={int(mask.sum()):,}")
            row, _ = execute_scenario(
                graph,
                origins,
                schools,
                scenario_id=scenario_id,
                edge_mask=mask,
                operation="add_wait",
                extra_wait_min=extra,
                metadata=metadata_row,
                experiment_hash=experiment_hash,
                save_detail=True,
            )
            append_row(SCENARIO_SUMMARY_OUTPUT, row)
            existing.add(scenario_id)

    p1_catalog = catalog.sort_values(["length_km", "n_edges"], ascending=False)
    if args.skip_p1:
        p1_catalog = p1_catalog.iloc[0:0]
    if max_routes > 0:
        p1_catalog = p1_catalog.head(max_routes)
    for route_row in p1_catalog.itertuples(index=False):
        scenario_id = f"p1_remove_{slugify(route_row.mode)}_{slugify(route_row.route)}"
        if scenario_id in existing:
            print(f"[skip] {scenario_id}")
            continue
        travel_mask = graph.edges_gdf["type"].astype(str).eq(str(route_row.mode)) & route_equals(
            graph.edges_gdf["route"], route_row.route
        )
        if not travel_mask.any():
            continue
        print(f"[run ] {scenario_id} edges={int(travel_mask.sum()):,}")
        row, _ = execute_scenario(
            graph,
            origins,
            schools,
            scenario_id=scenario_id,
            edge_mask=travel_mask,
            operation="remove_edges",
            extra_wait_min=None,
            metadata={"scenario_group": "P1_leave_one_route_out", "mode": route_row.mode, "route": route_row.route},
            experiment_hash=experiment_hash,
            save_detail=False,
        )
        row["route_length_km"] = float(route_row.length_km)
        append_row(SCENARIO_SUMMARY_OUTPUT, row)
        existing.add(scenario_id)

    summary = current_summary(experiment_hash)
    p1 = summary.loc[summary["scenario_group"].eq("P1_leave_one_route_out")].copy()
    if not p1.empty:
        p1 = p1.sort_values(
            [
                "total_school_opportunities_lost_30",
                "n_origins_losing_opportunities_30",
                "mean_increase_min",
            ],
            ascending=False,
        )
        p1.to_csv(ROUTE_CRITICALITY_OUTPUT, index=False)
        top_by_mode = []
        for map_mode in P1_MAP_MODES:
            mode_rows = p1.loc[p1["mode"].astype(str).eq(map_mode)]
            if not mode_rows.empty:
                top_by_mode.append(mode_rows.iloc[[0]])
        top = pd.concat(top_by_mode, ignore_index=True) if top_by_mode else p1.head(3)
        top_payload = top[
            [
                "scenario_id",
                "mode",
                "route",
                "total_school_opportunities_lost_30",
                "n_origins_losing_opportunities_30",
            ]
        ].to_dict(orient="records")
        write_json(TOP_ROUTES_OUTPUT, {"experiment_hash": experiment_hash, "routes": top_payload})

        for row in top.itertuples(index=False):
            travel_mask = graph.edges_gdf["type"].astype(str).eq(str(row.mode)) & route_equals(
                graph.edges_gdf["route"], row.route
            )
            print(f"[detail] {row.scenario_id}")
            execute_scenario(
                graph,
                origins,
                schools,
                scenario_id=row.scenario_id,
                edge_mask=travel_mask,
                operation="remove_edges",
                extra_wait_min=None,
                metadata={"scenario_group": "P1_leave_one_route_out", "mode": row.mode, "route": row.route},
                experiment_hash=experiment_hash,
                save_detail=True,
            )

    run_info = {
        "experiment_hash": experiment_hash,
        "graph_path": str(graph_path),
        "mode": mode,
        "selected_route": selected_route,
        "extra_wait_min": extras,
        "scenario_thresholds_min": list(SCENARIO_THRESHOLDS_MIN),
        "p1_routes": len(p1_catalog),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(SCENARIO_SUMMARY_OUTPUT.with_suffix(".json"), run_info)
    print(f"[done] scenarios -> {SCENARIO_SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
