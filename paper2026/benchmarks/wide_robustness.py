"""Does the conclusion depend on which point of the fitted valley we picked?

The fit is flat. Three defensible settings for the bus -- 37.0 km/h with a 0.450
dwell, 41.0 with 0.475, 38.0 with 0.450 -- score 0.2678, 0.2669 and 0.2622 on the
same 102,704 matched segments, a spread of under two per cent. A reviewer is
entitled to ask whether the headline correlation survives a move inside that
valley, and the honest answer is a rerun, not an argument.

So each alternative setting gets its own transit graphs and its own accessibility
sweep, and the Spearman correlation is recomputed from them. Everything else is
held fixed: the pedestrian graphs carry no transit constants, the schedule graphs
take their times from the timetable, and the sampled points come from the same
seed. The only thing that differs between a variant and the standing result is
the two numbers under test.

Variants write to their own graph directory and their own CSV, so the canonical
graphs are never overwritten -- otherwise checking robustness would cost a third
rebuild to put the study back the way it was.

    python wide_robustness.py --list
    python wide_robustness.py --variant bus_37_0450 --run
    python wide_robustness.py --report
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from bench_common import already_done, sweep
from scipy.stats import spearmanr
from wide_access import N_DESTINATIONS, N_ORIGINS, SEED, THRESHOLDS_MIN, _nearest_nodes, _reach, _sample_points
from wide_cohort import measurable, restrict, with_gtfs_graph
from wide_paths import (
    ALL_OSM_MODES,
    COVERAGE_CSV,
    OSM_GRAPH_DIR,
    PAPER_DIR,
    WIDE_DIR,
    gtfs_graph_path,
    osm_graph_path,
    use_paper_cache,
)

from iduedu import (
    DEFAULT_REGISTRY_W_TRAIN,
    TransportRegistry,
    config,
    get_public_transport_graph,
    join_pt_walk_graph,
    read_urban_graph,
    write_urban_graph,
)

logger = logging.getLogger(__name__)

#: The three settings that score within two per cent of each other. The middle one
#: is what the registry holds, so it is not rerun -- its result is the standing one.
VARIANTS = {
    "bus_37_0450": {"bus": {"base_speed_kmh": 37.0, "dwell_min": 0.450}},
    "bus_38_0450": {"bus": {"base_speed_kmh": 38.0, "dwell_min": 0.450}},
}


def variant_registry(variant: str) -> TransportRegistry:
    """A copy of the widest registry with one mode's constants replaced.

    A copy, not the global: mutating ``DEFAULT_REGISTRY_W_TRAIN`` would leave the
    process holding altered constants for anything else it touched afterwards.
    """
    base = DEFAULT_REGISTRY_W_TRAIN
    registry = TransportRegistry({name: base.get(name) for name in base.list_types()})
    for mode, fields in VARIANTS[variant].items():
        registry.update(mode, **fields)
    return registry


def variant_graph_dir(variant: str) -> Path:
    return PAPER_DIR / f"osm_graphs_robustness_{variant}"


def variant_csv(variant: str) -> Path:
    return WIDE_DIR / f"robustness_{variant}.csv"


def build_pt(city_key: str, variant: str) -> Path | None:
    """The city's transit graph under the variant constants, built if absent."""
    destination = variant_graph_dir(variant) / f"{city_key}__pt.urbangraph"
    if destination.exists():
        return destination
    canonical = osm_graph_path(city_key, "pt", ALL_OSM_MODES)
    if not canonical.exists():
        return None
    # The osm_id is not carried in the graph, so it comes from the same table the
    # canonical build used.
    cities = pd.read_csv(WIDE_DIR / "cities.csv").fillna("")
    row = cities.loc[cities["city_key"].astype(str) == city_key]
    if row.empty:
        return None
    osm_id = int(float(row.iloc[0]["osm_id"]))

    graph = get_public_transport_graph(
        osm_id=osm_id,
        transport_types=ALL_OSM_MODES,
        clip_by_territory=True,
        transport_registry=variant_registry(variant),
    )
    if graph.nodes_gdf.empty:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_urban_graph(graph, destination)
    return destination


def measure_city(city_key: str, variant: str) -> dict:
    """Same measurement as the standing one, on the variant's transit graph."""
    row = {"city_key": city_key, "variant": variant}
    walk_path = osm_graph_path(city_key, "walk")
    gtfs_path = gtfs_graph_path(city_key)
    if not (walk_path.exists() and gtfs_path.exists()):
        row["status"] = "missing_graph"
        return row

    started = time.perf_counter()
    pt_path = build_pt(city_key, variant)
    if pt_path is None:
        row["status"] = "no_osm_transit"
        row["elapsed_s"] = round(time.perf_counter() - started, 1)
        return row

    walk = read_urban_graph(walk_path)
    osm_pt = read_urban_graph(pt_path)
    gtfs_pt = read_urban_graph(gtfs_path)
    if walk.nodes_gdf.empty or osm_pt.nodes_gdf.empty or gtfs_pt.nodes_gdf.empty:
        row["status"] = "empty_graph"
        return row

    points = _sample_points(walk.nodes_gdf, N_DESTINATIONS, SEED)
    origins = points[:N_ORIGINS]
    variants = {}
    for name, transit in (("A", osm_pt), ("D", gtfs_pt)):
        joined = join_pt_walk_graph(transit, walk, keep_largest_subgraph=False)
        variants[name] = _reach(joined, _nearest_nodes(joined, origins), _nearest_nodes(joined, points))

    for threshold in THRESHOLDS_MIN:
        osm_reach = np.array(variants["A"][threshold], dtype=float)
        reference = np.array(variants["D"][threshold], dtype=float)
        row[f"osm_mean_{threshold}"] = round(float(osm_reach.mean()), 2)
        row[f"gtfs_mean_{threshold}"] = round(float(reference.mean()), 2)
        row[f"ratio_{threshold}"] = round(float(osm_reach.mean() / reference.mean()), 4) if reference.mean() else None
    row["elapsed_s"] = round(time.perf_counter() - started, 1)
    row["status"] = "ok"
    return row


def correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """The same predictor table the standing results report, on a variant's ratios."""
    coverage = pd.read_csv(COVERAGE_CSV)
    city_level = coverage.loc[coverage["mode"].eq("*") & coverage["status"].eq("ok")]
    city_level = city_level.drop_duplicates("city_key", keep="last")
    # The result tables are append-only, so a city measured under two runs appears
    # twice; the metrics script keeps the last row per city and so must this, or a
    # variant would be compared against a mixture of runs.
    frame = frame.drop_duplicates("city_key", keep="last")
    frame = frame.loc[frame["city_key"].map(str).isin(set(measurable()))]
    table = frame.loc[frame["status"].eq("ok")].merge(
        city_level[["city_key", "feed_to_osm_100", "osm_to_feed_100"]], on="city_key", how="inner"
    )
    table["relative_completeness"] = table["feed_to_osm_100"] - table["osm_to_feed_100"]
    rows = []
    for threshold in THRESHOLDS_MIN:
        subset = table[[f"ratio_{threshold}", "relative_completeness"]].dropna()
        rho, p = spearmanr(subset["relative_completeness"], subset[f"ratio_{threshold}"])
        rows.append(
            {
                "threshold": f"{threshold} min",
                "cities": len(subset),
                "median_ratio": round(table[f"ratio_{threshold}"].median(), 3),
                "rho": round(rho, 3),
                "p": f"{p:.2e}",
            }
        )
    return pd.DataFrame(rows)


def report() -> None:
    """Every variant that has finished, beside the standing result."""
    from wide_paths import ACCESSIBILITY_CSV

    frames = {"standing (41.0 / 0.475)": pd.read_csv(ACCESSIBILITY_CSV)}
    for variant in VARIANTS:
        path = variant_csv(variant)
        if path.exists():
            speed = VARIANTS[variant]["bus"]["base_speed_kmh"]
            dwell = VARIANTS[variant]["bus"]["dwell_min"]
            frames[f"{variant} ({speed} / {dwell})"] = pd.read_csv(path)

    for label, frame in frames.items():
        table = correlations(frame)
        table.insert(0, "setting", label)
        print(table.to_string(index=False))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--cities", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run", action="store_true", help="without it, only the plan is printed")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", action="store_true", help="print the comparison and exit")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logger.setLevel(logging.INFO)

    if arguments.report:
        report()
        return
    if not arguments.variant:
        raise SystemExit("--variant is required unless --report is given")

    use_paper_cache()
    # A hundred progress bars in a log file are a hundred thousand lines of
    # carriage returns; the sweep already logs one line per city.
    config.set_enable_tqdm(False)
    keys = restrict(sorted(set(measurable()) & with_gtfs_graph()), arguments.cities)
    csv_path = variant_csv(arguments.variant)
    if arguments.limit:
        done = set() if arguments.force else already_done(csv_path)
        keys = [key for key in keys if key not in done][: arguments.limit]

    if not arguments.run:
        done = already_done(csv_path)
        print(f"variant {arguments.variant}: {VARIANTS[arguments.variant]}")
        print(f"{len(keys)} cities in the cohort, {len(done & set(keys))} already measured")
        print(f"graphs go to {variant_graph_dir(arguments.variant)}")
        print(f"canonical graphs in {OSM_GRAPH_DIR} are not touched")
        print("Nothing was computed. Re-run with --run.")
        return

    sweep(
        keys,
        lambda key: measure_city(key, arguments.variant),
        csv_path,
        force=arguments.force,
        logger=logger,
        describe=lambda rows: f"{rows[0].get('status')} ratio 30 min {rows[0].get('ratio_30')}",
    )
    print()
    print(correlations(pd.read_csv(csv_path)).to_string(index=False))


if __name__ == "__main__":
    main()
