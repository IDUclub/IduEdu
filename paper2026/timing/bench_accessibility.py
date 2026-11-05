"""B5 — school accessibility, ObjectNat geometries, and the spatial value of public transport.

The script is graph-cache aware and safe to rerun. It never clears
``benchmarks/.iduedu_cache``. If ``results/accessibility_cache/*.urbangraph``
exists, the graph is loaded with its CSR; otherwise it is built once and saved.

Outputs live under ``results/accessibility/`` and are consumed by
``figures/accessibility_figures.ipynb``.
"""

import argparse
import time

import geopandas as gpd
import numpy as np
import objectnat
import pandas as pd
from accessibility_common import (
    ACCESSIBILITY_DIR,
    CATCHMENTS_OUTPUT,
    ISOCHRONES_OUTPUT,
    MANIFEST_PATH,
    MAP_PT_EDGES_OUTPUT,
    MAP_WALK_EDGES_OUTPUT,
    ORIGINS_OUTPUT,
    SCHOOLS_OUTPUT,
    SELECTED_ORIGINS_OUTPUT,
    STEPPED_COVERAGE_OUTPUT,
    STEPPED_ISOCHRONES_OUTPUT,
    SUMMARY_OUTPUT,
    THRESHOLDS_MIN,
    TIMINGS_OUTPUT,
    VISUAL_BOUNDARY_OSM_ID,
    VISUAL_BOUNDARY_OUTPUT,
    WEIGHT,
    build_analysis_zone,
    ensure_directories,
    finite_percentile,
    load_or_build_graphs,
    load_or_download_visual_boundary,
    load_origins,
    load_schools,
    nearest_school_metrics,
    od_school_counts,
    package_version,
    resolve_origins_path,
    stable_hash,
    write_json,
)
from bench_common import dump_environment, measure

ANALYSIS_ZONE_OUTPUT = ACCESSIBILITY_DIR / "analysis_zone.parquet"
PT_TYPES = {"bus", "tram", "trolleybus", "subway", "train"}


def add_metric(
    rows: list[dict],
    *,
    metric: str,
    value,
    graph: str = "comparison",
    threshold_min: float | None = None,
    unit: str = "value",
) -> None:
    rows.append(
        {
            "graph": graph,
            "metric": metric,
            "threshold_min": threshold_min,
            "value": value,
            "unit": unit,
        }
    )


def population_weights(origins: gpd.GeoDataFrame) -> pd.Series:
    if "population" not in origins.columns:
        return pd.Series(0.0, index=origins.index)
    return pd.to_numeric(origins["population"], errors="coerce").fillna(0.0).clip(lower=0.0)


def summarize_graph(
    rows: list[dict],
    origins: gpd.GeoDataFrame,
    *,
    graph_label: str,
) -> None:
    times = pd.to_numeric(origins[f"{graph_label}_time_min"], errors="coerce")
    finite = np.isfinite(times)
    add_metric(rows, graph=graph_label, metric="finite_nearest_share", value=float(finite.mean()), unit="share")
    add_metric(
        rows,
        graph=graph_label,
        metric="nearest_time_median",
        value=finite_percentile(times, 50),
        unit="min",
    )
    add_metric(
        rows,
        graph=graph_label,
        metric="nearest_time_p90",
        value=finite_percentile(times, 90),
        unit="min",
    )
    for threshold in THRESHOLDS_MIN:
        reachable = times <= threshold
        add_metric(
            rows,
            graph=graph_label,
            metric="origins_reachable",
            threshold_min=threshold,
            value=float(reachable.mean()),
            unit="share",
        )
        counts = origins[f"{graph_label}_schools_{int(threshold)}"]
        add_metric(
            rows,
            graph=graph_label,
            metric="reachable_schools_mean",
            threshold_min=threshold,
            value=float(counts.mean()),
            unit="schools",
        )
        add_metric(
            rows,
            graph=graph_label,
            metric="reachable_schools_median",
            threshold_min=threshold,
            value=float(counts.median()),
            unit="schools",
        )


def summarize_population_gain(
    rows: list[dict],
    origins: gpd.GeoDataFrame,
    population: pd.Series,
) -> None:
    """Add only the population-weighted walk-to-intermodal coverage difference."""
    total_population = float(population.sum())
    if total_population <= 0:
        return
    for threshold in THRESHOLDS_MIN:
        walk_reachable = pd.to_numeric(origins["walk_time_min"], errors="coerce") <= threshold
        intermodal_reachable = pd.to_numeric(origins["intermodal_time_min"], errors="coerce") <= threshold
        walk_share = float(population.loc[walk_reachable].sum() / total_population)
        intermodal_share = float(population.loc[intermodal_reachable].sum() / total_population)
        add_metric(
            rows,
            metric="population_reachable_gain",
            threshold_min=threshold,
            value=(intermodal_share - walk_share) * 100.0,
            unit="percentage_points",
        )


def classify_gain(origins: gpd.GeoDataFrame) -> pd.Series:
    walk = origins["walk_schools_15"].to_numpy(dtype=int)
    intermodal = origins["intermodal_schools_15"].to_numpy(dtype=int)
    gain = origins["opportunity_gain_15"].to_numpy(dtype=int)
    relative = origins["relative_opportunity_gain_15"].to_numpy(dtype=float)
    labels = np.full(len(origins), "moderate", dtype=object)
    labels[(walk == 0) & (intermodal == 0)] = "unreachable"
    labels[(walk == 0) & (intermodal > 0)] = "newly_reachable"
    labels[(walk > 0) & (gain <= 0)] = "no_effect"
    substantial = (walk > 0) & ((gain >= 5) | (relative >= 0.25))
    labels[substantial] = "substantial"
    return pd.Series(labels, index=origins.index, name="gain_class")


def select_map_origins(origins: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    finite = origins[np.isfinite(origins["opportunity_gain_15"])].copy()
    selected: list[tuple[str, int]] = []
    if not finite.empty:
        selected.append(("highest_gain", int(finite["opportunity_gain_15"].idxmax())))
        positive = finite.loc[finite["opportunity_gain_15"] > 0]
        if not positive.empty:
            median_delta = float(positive["opportunity_gain_15"].median())
            typical_idx = (positive["opportunity_gain_15"] - median_delta).abs().idxmin()
            selected.append(("typical_gain", int(typical_idx)))
        no_effect = finite.loc[finite["opportunity_gain_15"] <= 0]
        if not no_effect.empty:
            selected.append(("no_effect", int(no_effect["opportunity_gain_15"].abs().idxmin())))
    if not selected:
        selected.append(("example", int(origins.index[0])))

    unique = []
    used = set()
    for label, index in selected:
        if index not in used:
            unique.append((label, index))
            used.add(index)
    result = origins.loc[[index for _, index in unique]].copy()
    result["selection"] = [label for label, _ in unique]
    return result


def save_map_layers(intermodal, *, include_walk: bool) -> None:
    edges = intermodal.edges_gdf
    edge_types = edges["type"].astype(str) if "type" in edges.columns else pd.Series("unknown", index=edges.index)
    walk = edges.loc[edge_types.eq("walk"), ["geometry"]].copy() if include_walk else edges.iloc[0:0][["geometry"]]
    walk.to_parquet(MAP_WALK_EDGES_OUTPUT, index=False)

    pt = edges.loc[edge_types.isin(PT_TYPES)].copy()
    keep = [column for column in ("type", "route", "geometry") if column in pt.columns]
    pt = pt[keep]
    if "route" in pt.columns:
        pt["route"] = pt["route"].map(lambda value: None if value is None or value is pd.NA else str(value))
    pt.to_parquet(MAP_PT_EDGES_OUTPUT, index=False)


def run_objectnat_outputs(
    walk,
    intermodal,
    origins: gpd.GeoDataFrame,
    schools: gpd.GeoDataFrame,
    zone: gpd.GeoDataFrame,
    timings: list[dict],
) -> dict:
    selected = select_map_origins(origins)
    selected.to_parquet(SELECTED_ORIGINS_OUTPUT, index=True)

    iso_frames = []
    stepped_iso_frames = []
    coverage_frames = []
    catchment_frames = []
    highest = selected.loc[selected["selection"].eq("highest_gain")]
    if highest.empty:
        highest = selected.iloc[[0]]

    for graph_label, graph in (("walk", walk), ("intermodal", intermodal)):
        origin_node_column = f"{graph_label}_node_id"
        school_node_column = f"{graph_label}_node_id"
        selected_graph = selected[["geometry", origin_node_column]].rename(
            columns={origin_node_column: "graph_node_id"}
        )
        highest_graph = highest[["geometry", origin_node_column]].rename(columns={origin_node_column: "graph_node_id"})
        schools_graph = schools[["geometry", school_node_column]].rename(columns={school_node_column: "graph_node_id"})
        m_iso = measure(
            objectnat.get_graph_isochrones,
            graph,
            gdf_origins=selected_graph,
            weight_value_cutoff=30.0,
            weight_type=WEIGHT,
            geometry_type="ways",
            zone=zone,
        )
        iso = m_iso.result
        iso["graph"] = graph_label
        iso["cutoff_min"] = 30.0
        iso["selection"] = selected["selection"].reindex(iso.index).to_numpy()
        iso_frames.append(iso)
        timings.append({"stage": "objectnat_isochrones", "graph": graph_label, "time_sec": m_iso.time_sec})

        m_stepped_iso = measure(
            objectnat.get_stepped_graph_isochrones,
            graph,
            gdf_origins=highest_graph,
            weight_value_cutoff=30.0,
            step=5.0,
            weight_type=WEIGHT,
            geometry_type="radius",
            zone=zone,
        )
        stepped_iso = m_stepped_iso.result
        stepped_iso["graph"] = graph_label
        stepped_iso_frames.append(stepped_iso)
        timings.append(
            {"stage": "objectnat_stepped_isochrone", "graph": graph_label, "time_sec": m_stepped_iso.time_sec}
        )

        m_coverage = measure(
            objectnat.get_stepped_graph_coverage,
            graph,
            gdf_destinations=schools_graph,
            weight_value_cutoff=30.0,
            step=5.0,
            weight_type=WEIGHT,
            geometry_type="radius",
            zone=zone,
        )
        coverage = m_coverage.result
        coverage["graph"] = graph_label
        coverage_frames.append(coverage)
        timings.append({"stage": "objectnat_stepped_coverage", "graph": graph_label, "time_sec": m_coverage.time_sec})

        m_catchments = measure(
            objectnat.get_graph_coverage,
            graph,
            gdf_destinations=schools_graph,
            weight_value_cutoff=30.0,
            weight_type=WEIGHT,
            geometry_type="radius",
            zone=zone,
        )
        catchments = m_catchments.result
        catchments["graph"] = graph_label
        catchments["school_id"] = catchments.index.astype(str)
        catchment_frames.append(catchments)
        timings.append({"stage": "objectnat_catchments", "graph": graph_label, "time_sec": m_catchments.time_sec})

    gpd.GeoDataFrame(pd.concat(iso_frames), geometry="geometry", crs=iso_frames[0].crs).to_parquet(
        ISOCHRONES_OUTPUT, index=True
    )
    gpd.GeoDataFrame(pd.concat(stepped_iso_frames), geometry="geometry", crs=stepped_iso_frames[0].crs).to_parquet(
        STEPPED_ISOCHRONES_OUTPUT, index=True
    )
    gpd.GeoDataFrame(pd.concat(coverage_frames), geometry="geometry", crs=coverage_frames[0].crs).to_parquet(
        STEPPED_COVERAGE_OUTPUT, index=True
    )
    gpd.GeoDataFrame(pd.concat(catchment_frames), geometry="geometry", crs=catchment_frames[0].crs).to_parquet(
        CATCHMENTS_OUTPUT, index=True
    )
    return {
        "version": getattr(objectnat, "__version__", package_version("objectnat")),
        "selected_origins": len(selected),
        "isochrone_geometry_type": "ways",
        "stepped_isochrone_geometry_type": "radius",
        "coverage_geometry_type": "radius",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="small AOI and sampled objects")
    parser.add_argument("--origins-path", default=None)
    parser.add_argument("--schools-path", default=None)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--max-schools", type=int, default=None)
    parser.add_argument("--aoi-buffer-m", type=float, default=1200.0)
    parser.add_argument("--skip-objectnat", action="store_true")
    args = parser.parse_args()

    ensure_directories()
    dump_environment("accessibility")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    origins_path = resolve_origins_path(args.origins_path)
    print(f"[data] loading origins: {origins_path}", flush=True)
    data_started = time.perf_counter()
    origins = load_origins(origins_path, smoke=args.smoke, max_origins=args.max_origins)
    print(f"[data] origins loaded in {time.perf_counter() - data_started:.2f}s", flush=True)
    buffer_m = 400.0 if args.smoke else args.aoi_buffer_m
    zone_started = time.perf_counter()
    zone = build_analysis_zone(origins, buffer_m)
    print(f"[data] analysis zone built in {time.perf_counter() - zone_started:.2f}s", flush=True)
    schools_started = time.perf_counter()
    schools, schools_source = load_schools(
        zone,
        school_path=args.schools_path,
        smoke=args.smoke,
        max_schools=args.max_schools,
    )
    print(f"[data] schools loaded in {time.perf_counter() - schools_started:.2f}s", flush=True)
    print(
        f"[data] origins={len(origins):,} schools={len(schools):,} source={schools_source}",
        flush=True,
    )
    visual_boundary = load_or_download_visual_boundary()

    zone.to_parquet(ANALYSIS_ZONE_OUTPUT, index=False)
    walk, intermodal, graph_info = load_or_build_graphs(zone, smoke=args.smoke)
    timings = [
        {"stage": "graph_build", "graph": "intermodal", "time_sec": graph_info["build_sec"]},
        {"stage": "graph_write", "graph": "both", "time_sec": graph_info["write_sec"]},
        {"stage": "graph_load", "graph": "intermodal", "time_sec": graph_info["intermodal_load_sec"]},
        {"stage": "graph_load", "graph": "walk", "time_sec": graph_info["walk_load_sec"]},
    ]

    output = origins.copy()
    for graph_label, graph in (("walk", walk), ("intermodal", intermodal)):
        origin_node_column = f"{graph_label}_node_id"
        school_node_column = f"{graph_label}_node_id"
        print(f"[snap] {graph_label}: {len(origins):,} origins", flush=True)
        m_origin_snap = measure(graph.nearest_nodes, origins)
        print(f"[snap] {graph_label}: {len(schools):,} schools", flush=True)
        m_school_snap = measure(graph.nearest_nodes, schools)
        output[origin_node_column] = m_origin_snap.result.to_numpy()
        schools[school_node_column] = m_school_snap.result.to_numpy()
        nearest, nearest_sec = nearest_school_metrics(
            graph,
            output,
            schools,
            origin_node_column=origin_node_column,
            school_node_column=school_node_column,
        )
        print(f"[od] {graph_label}: full matrix {len(origins):,} x {len(schools):,}", flush=True)
        counts, od_sec, od_path = od_school_counts(
            graph,
            output,
            schools,
            label=graph_label,
            origin_node_column=origin_node_column,
            school_node_column=school_node_column,
        )
        print(f"[od] {graph_label}: completed in {od_sec:.2f}s", flush=True)
        output[f"{graph_label}_time_min"] = nearest["nearest_time_min"]
        output[f"{graph_label}_school_id"] = nearest["nearest_school_id"].astype("string")
        for threshold in THRESHOLDS_MIN:
            source_column = f"schools_within_{int(threshold)}"
            output[f"{graph_label}_schools_{int(threshold)}"] = counts[source_column]
        timings.extend(
            [
                {"stage": "nearest_school", "graph": graph_label, "time_sec": nearest_sec},
                {"stage": "snap_origins", "graph": graph_label, "time_sec": m_origin_snap.time_sec},
                {"stage": "snap_schools", "graph": graph_label, "time_sec": m_school_snap.time_sec},
                {"stage": "od_school_counts", "graph": graph_label, "time_sec": od_sec, "artifact": str(od_path)},
            ]
        )

    walk_time = output["walk_time_min"].to_numpy(dtype=float)
    intermodal_time = output["intermodal_time_min"].to_numpy(dtype=float)
    both_finite = np.isfinite(walk_time) & np.isfinite(intermodal_time)
    delta = np.full(len(output), np.nan, dtype=float)
    delta[both_finite] = walk_time[both_finite] - intermodal_time[both_finite]
    relative = np.full(len(output), np.nan, dtype=float)
    valid_walk = both_finite & (walk_time > 0)
    relative[valid_walk] = delta[valid_walk] / walk_time[valid_walk]
    output["delta_time_min"] = delta
    output["relative_gain"] = relative
    for threshold in THRESHOLDS_MIN:
        threshold = int(threshold)
        walk_count = output[f"walk_schools_{threshold}"].to_numpy(dtype=int)
        intermodal_count = output[f"intermodal_schools_{threshold}"].to_numpy(dtype=int)
        opportunity_gain = intermodal_count - walk_count
        output[f"opportunity_gain_{threshold}"] = opportunity_gain
        output[f"relative_opportunity_gain_{threshold}"] = opportunity_gain / np.maximum(walk_count, 1)
        output[f"newly_reachable_{threshold}"] = (walk_count == 0) & (intermodal_count > 0)
    output["gain_class"] = classify_gain(output)
    output.to_parquet(ORIGINS_OUTPUT, index=True)
    schools.to_parquet(SCHOOLS_OUTPUT, index=True)
    include_walk_map = args.smoke or len(output) <= 5_000
    save_map_layers(intermodal, include_walk=include_walk_map)

    population = population_weights(output)
    summary_rows: list[dict] = []
    summarize_graph(summary_rows, output, graph_label="walk")
    summarize_graph(summary_rows, output, graph_label="intermodal")
    summarize_population_gain(summary_rows, output, population)
    positive_delta = output.loc[np.isfinite(output["delta_time_min"]), "delta_time_min"].clip(lower=0.0)
    add_metric(summary_rows, metric="delta_time_median", value=finite_percentile(positive_delta, 50), unit="min")
    add_metric(summary_rows, metric="delta_time_p90", value=finite_percentile(positive_delta, 90), unit="min")
    for threshold in THRESHOLDS_MIN:
        gain = output[f"opportunity_gain_{int(threshold)}"]
        add_metric(
            summary_rows,
            metric="school_opportunity_gain_mean",
            threshold_min=threshold,
            value=float(gain.mean()),
            unit="schools",
        )
    add_metric(
        summary_rows,
        metric="substantial_gain_share",
        value=float(output["gain_class"].eq("substantial").mean()),
        unit="share",
    )
    add_metric(
        summary_rows,
        metric="no_effect_share",
        value=float(output["gain_class"].eq("no_effect").mean()),
        unit="share",
    )
    for threshold in THRESHOLDS_MIN:
        add_metric(
            summary_rows,
            metric="newly_reachable_share",
            threshold_min=threshold,
            value=float(output[f"newly_reachable_{int(threshold)}"].mean()),
            unit="share",
        )
    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUTPUT, index=False)

    objectnat_info = None
    if not args.skip_objectnat:
        objectnat_info = run_objectnat_outputs(walk, intermodal, output, schools, zone, timings)

    timing_frame = pd.DataFrame(timings)
    timing_frame.insert(0, "run_timestamp", started)
    timing_frame.to_csv(TIMINGS_OUTPUT, index=False)

    input_scope = "city_input" if origins_path.name == "buildings_spb.parquet" else "docs_sample"
    manifest = {
        "started_at": started,
        "smoke": args.smoke,
        "scope": input_scope,
        "origins_path": str(origins_path),
        "schools_source": schools_source,
        "n_origins": len(output),
        "n_schools": len(schools),
        "aoi_buffer_m": buffer_m,
        "aoi_bounds_4326": zone.total_bounds.tolist(),
        "aoi_hash": stable_hash(zone.geometry.iloc[0].wkb),
        "visual_boundary": {
            "osm_id": VISUAL_BOUNDARY_OSM_ID,
            "path": str(VISUAL_BOUNDARY_OUTPUT),
            "bounds_4326": visual_boundary.total_bounds.tolist(),
        },
        "graph": graph_info,
        "thresholds_min": list(THRESHOLDS_MIN),
        "objectnat": objectnat_info,
        "map_walk_network": include_walk_map,
        "outputs": {
            "origins": str(ORIGINS_OUTPUT),
            "schools": str(SCHOOLS_OUTPUT),
            "summary": str(SUMMARY_OUTPUT),
            "timings": str(TIMINGS_OUTPUT),
        },
    }
    write_json(MANIFEST_PATH, manifest)
    print(f"[done] metrics -> {SUMMARY_OUTPUT}")
    print(f"[done] manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
