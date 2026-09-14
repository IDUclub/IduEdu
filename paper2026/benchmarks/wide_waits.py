"""Waiting times per mode, read off the reference graphs.

The number is taken from the ``boarding`` edges of each city's GTFS graph rather
than recomputed from headways here. That is deliberate: the wait a rider
experiences in this study's accessibility model *is* the boarding edge weight, so
measuring anything else would report a quantity the results do not use.

Aggregation is harmonic within a city and median across cities. Harmonic because
averaging waits directly is averaging the wrong thing -- what adds across a
network is service frequency, and the harmonic mean of waits is the reciprocal of
the mean frequency. A city with one bus an hour and one every two minutes runs a
two-minute service for almost everyone using it.

Pattern-stops with a single scheduled departure carry the fallback constant, not
a measured headway, and are excluded from the headline figure.

⚠ ``single_departure_share`` is **not** the share of stops served once a day, and
must not be reported as one. A pattern-stop has one departure when any of three
different things is true, and the feeds in this sample show all three:

* service really is that rare;
* every trip carries its own stop sequence, so each trip is its own pattern --
  Blue Downs has 236 trips and 236 distinct sequences, Kumasi 343 and 326;
* one timetable is repeated across several ``service_id`` values with identical
  clock times, which deduplication collapses to a single departure -- Bar has 22
  trips over 4 sequences and Vijayawada 433 over 46, yet both come out at 1.00.

Separating them needs the feed, not the graph. Until that is done the column is a
diagnostic for excluding rows from the wait statistic, nothing more.
"""

import argparse
import logging

import numpy as np
import pandas as pd
from bench_common import sweep
from wide_cohort import measurable, restrict, with_gtfs_graph
from wide_gtfs import SINGLE_DEPARTURE_WAIT_MIN
from wide_paths import WAITS_CSV, gtfs_graph_path

from iduedu import read_urban_graph

logger = logging.getLogger(__name__)

#: Node types that are structure rather than service.
STRUCTURE_TYPES = {"platform", "station", "entrance", "exit", "boarding_area", "node"}


REGULAR_WAIT_MAX_MIN = 30.0


def _harmonic_mean(values: pd.Series) -> float | None:
    positive = values[values > 0]
    if positive.empty:
        return None
    return float(len(positive) / np.sum(1.0 / positive))


def measure_city(city_key: str) -> list[dict]:
    path = gtfs_graph_path(city_key)
    if not path.exists():
        return [{"city_key": city_key, "mode": "*", "status": "missing_graph"}]

    graph = read_urban_graph(path)
    edges, nodes = graph.edges_gdf, graph.nodes_gdf
    if edges.empty or "type" not in edges:
        return [{"city_key": city_key, "mode": "*", "status": "empty_graph"}]

    boarding = edges.loc[edges["type"].eq("boarding"), ["v", "time_min"]].copy()
    if boarding.empty:
        return [{"city_key": city_key, "mode": "*", "status": "no_boarding_edges"}]

    # A boarding edge ends at the vehicle, so the mode is the mode of the node it
    # arrives at -- reading it off the platform it left would give "platform".
    boarding["mode"] = boarding["v"].map(nodes["type"])
    boarding = boarding.loc[~boarding["mode"].isin(STRUCTURE_TYPES) & boarding["mode"].notna()]
    if boarding.empty:
        return [{"city_key": city_key, "mode": "*", "status": "no_mode_on_boarding"}]

    fallback = pd.Series(np.isclose(boarding["time_min"], SINGLE_DEPARTURE_WAIT_MIN), index=boarding.index)
    irregular = boarding["time_min"].gt(REGULAR_WAIT_MAX_MIN)
    rows: list[dict] = []
    for mode, group in boarding.groupby("mode"):
        keep = ~fallback.loc[group.index] & ~irregular.loc[group.index]
        measured = group.loc[keep, "time_min"]
        # Kept alongside: the difference between the two columns is what irregular trips add.
        all_measured = group.loc[~fallback.loc[group.index], "time_min"]
        rows.append(
            {
                "city_key": city_key,
                "mode": str(mode),
                "n_boarding": len(group),
                "n_single_departure": int(fallback.loc[group.index].sum()),
                "single_departure_share": round(float(fallback.loc[group.index].mean()), 4),
                "n_irregular": int(irregular.loc[group.index].sum()),
                "wait_harmonic_min": None if measured.empty else round(_harmonic_mean(measured), 3),
                "wait_median_min": None if measured.empty else round(float(measured.median()), 3),
                "wait_harmonic_uncapped_min": (None if all_measured.empty else round(_harmonic_mean(all_measured), 3)),
                "status": "ok",
            }
        )

    measured_all = boarding.loc[~fallback & ~irregular, "time_min"]
    rows.append(
        {
            "city_key": city_key,
            "mode": "*",
            "n_boarding": len(boarding),
            "n_single_departure": int(fallback.sum()),
            "single_departure_share": round(float(fallback.mean()), 4),
            "wait_harmonic_min": None if measured_all.empty else round(_harmonic_mean(measured_all), 3),
            "wait_median_min": None if measured_all.empty else round(float(measured_all.median()), 3),
            "modes": ";".join(sorted(boarding["mode"].map(str).unique())),
            "status": "ok",
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="*")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def describe(rows: list[dict]) -> str:
        headline = next((row for row in rows if row.get("mode") == "*"), rows[0])
        return (
            f"{headline.get('status')} wait {headline.get('wait_harmonic_min')} min, "
            f"single-departure {headline.get('single_departure_share')}"
        )

    # A graph with no usable run times still has real headways: its stops and its
    # departures are intact, only the time between stops is missing.
    keys = sorted(set(measurable()) & with_gtfs_graph(include_untimed=True))
    sweep(
        restrict(keys, arguments.cities),
        measure_city,
        WAITS_CSV,
        force=arguments.force,
        logger=logger,
        describe=describe,
        failure_extra={"mode": "*"},
    )


if __name__ == "__main__":
    main()
