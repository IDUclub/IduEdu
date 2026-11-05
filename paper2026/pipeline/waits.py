"""Waiting times per city and mode, as the timetable graphs carry them.

* ``measure`` -- read off the boarding edges of each city's GTFS graph: the wait the
  accessibility model uses, not a formula re-derived here. A city's value for a mode
  is the harmonic mean over its line-stops (= the mean wait per departure); stops
  with a single departure (the 60-minute fallback) and waits over 60 minutes
  (headway over 2 h: irregular service) are left out. -> ``indicators_waits.csv``
* ``table`` -- the distribution over cities per mode, and how far the registry's
  constant misses each city. -> ``waits_by_city_mode.csv``, ``waits_by_mode.csv``

    python pipeline/waits.py measure | table
"""

import argparse

import numpy as np
import pandas as pd
from common import (
    RESULTS_DIR,
    WAITS_CSV,
    gtfs_graph_path,
    measurable,
    restrict,
    setup,
    sweep,
    with_gtfs_graph,
)
from gtfs import SINGLE_DEPARTURE_WAIT_MIN

from iduedu import read_urban_graph
from iduedu.constants.transport_specs import DEFAULT_REGISTRY_W_TRAIN

REGULAR_WAIT_MAX_MIN = 60.0
STRUCTURE_TYPES = {"platform", "station", "entrance", "exit", "boarding_area", "node"}
QUANTILES = ((0.0, "min"), (0.1, "p10"), (0.25, "p25"), (0.5, "median"), (0.75, "p75"), (0.9, "p90"), (1.0, "max"))


def harmonic(values: pd.Series) -> float | None:
    positive = values[values > 0]
    return float(len(positive) / np.sum(1.0 / positive)) if len(positive) else None


def _rounded(value, digits=3):
    return None if value is None else round(value, digits)


# --- measure -----------------------------------------------------------------------


def measure_city(city_key: str) -> list[dict]:
    graph = read_urban_graph(gtfs_graph_path(city_key))
    edges, nodes = graph.edges_gdf, graph.nodes_gdf
    boarding = edges.loc[edges["type"].eq("boarding"), ["u", "v", "route_id", "time_min"]].copy()
    # a boarding edge ends at the vehicle: the mode is that of the node it arrives at
    boarding["mode"] = boarding["v"].map(nodes["type"])
    boarding = boarding.loc[boarding["mode"].notna() & ~boarding["mode"].isin(STRUCTURE_TYPES)]
    if boarding.empty:
        return [{"city_key": city_key, "mode": "*", "status": "no_boarding_edges"}]
    fallback = pd.Series(np.isclose(boarding["time_min"], SINGLE_DEPARTURE_WAIT_MIN), index=boarding.index)
    irregular = boarding["time_min"].gt(REGULAR_WAIT_MAX_MIN)
    rows = []
    for mode, group in list(boarding.groupby("mode")) + [("*", boarding)]:
        single, too_long = fallback.loc[group.index], irregular.loc[group.index]
        regular = group.loc[~single & ~too_long, "time_min"]
        rows.append({
            "city_key": city_key, "mode": str(mode), "n_boarding": len(group),
            "stops": int(group["u"].nunique()), "routes": int(group["route_id"].nunique()),
            "n_single_departure": int(single.sum()), "single_departure_share": round(float(single.mean()), 4),
            "n_irregular": int(too_long.sum()),
            "wait_harmonic_min": _rounded(harmonic(regular)),
            "wait_median_min": None if regular.empty else round(float(regular.median()), 3),
            "wait_harmonic_uncapped_min": _rounded(harmonic(group.loc[~single, "time_min"])),
            "status": "ok",
        })  # fmt: skip
    return rows


# --- table -------------------------------------------------------------------------


def table() -> pd.DataFrame:
    waits = pd.read_csv(WAITS_CSV, keep_default_na=False)
    waits = waits[(waits["status"] == "ok") & (waits["mode"] != "*") & waits["city_key"].isin(measurable())].copy()
    waits["wait_min"] = pd.to_numeric(waits["wait_harmonic_min"], errors="coerce")
    by_city = waits[["city_key", "mode", "wait_min", "n_boarding", "single_departure_share", "stops", "routes"]].copy()
    constants = {mode: DEFAULT_REGISTRY_W_TRAIN.try_get(mode) for mode in by_city["mode"].unique()}
    by_city["library_constant_min"] = by_city["mode"].map(
        lambda m: constants[m].avg_wait_time_min if constants[m] else np.nan
    )
    by_city["constant_over_own"] = (by_city["library_constant_min"] / by_city["wait_min"]).round(3)
    by_city.to_csv(RESULTS_DIR / "waits_by_city_mode.csv", index=False)

    rows = []
    for mode, group in by_city.groupby("mode"):
        values, ratio = group["wait_min"].dropna(), group["constant_over_own"].dropna()
        row = {"mode": mode, "cities": len(values), "library_constant_min": group["library_constant_min"].iloc[0]}
        row.update(
            {f"wait_{name}": round(float(values.quantile(q)), 2) if len(values) else None for q, name in QUANTILES}
        )
        row.update(
            {f"ratio_{name}": round(float(ratio.quantile(q)), 3) if len(ratio) else None for q, name in QUANTILES[1:-1]}
        )
        row["ratio_within_25pct"] = round(float(((ratio - 1).abs() <= 0.25).mean()), 3) if len(ratio) else None
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("cities", ascending=False)
    summary.to_csv(RESULTS_DIR / "waits_by_mode.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["measure", "table"])
    parser.add_argument("cities", nargs="*", help="measure: restrict to these")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    logger = setup("waits")
    if arguments.step == "measure":
        # a graph without run times still has its departures, so its waits count
        keys = restrict(sorted(set(measurable()) & with_gtfs_graph(include_untimed=True)), arguments.cities)
        sweep(keys, measure_city, WAITS_CSV, force=arguments.force, row_keys=["city_key", "mode"], logger=logger,
              describe=lambda rows: f"wait {rows[-1].get('wait_harmonic_min')} min")  # fmt: skip
    else:
        print(table().to_string(index=False))


if __name__ == "__main__":
    main()
