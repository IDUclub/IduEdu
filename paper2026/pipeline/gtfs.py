"""The timetable side: one graph per city archive, the reference OSM is compared against.

The graph carries every pattern that runs on the averaged days, and its boarding
edges carry the **average day's** waits (IduEdu 2.3): each day is computed on
its own trips and every day weighs the same, 06:00-22:00. The days are the reference
week 12-18.10.2026 (the week of the OSM snapshot); a source not in service that week
adds its own busiest week, so a feed of 2018 merged with a current one is measured
on its own calendar. A wait is half the mean gap between departures of one **line**
(route and next stop) at the stop. A stop with a single departure gets a 60-minute
wait rather than no boarding edge, so once-a-day service stays in the graph.

The graph adopts the CRS of the city's OSM walk graph, so the two can be joined.

    python pipeline/gtfs.py [--cities ...] [--force]
"""

import argparse
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
from common import (
    GTFS_GRAPHS_CSV,
    boundary,
    city_feed_path,
    gtfs_graph_path,
    measurable,
    osm_graph_path,
    restrict,
    setup,
    sweep,
)

from iduedu import get_gtfs_public_transport_graph, read_urban_graph, write_urban_graph
from iduedu.gtfs.reader import read_gtfs_feed

SINGLE_DEPARTURE_WAIT_MIN = 60.0
REFERENCE_WEEK = (date(2026, 10, 12), date(2026, 10, 18))
DAY_START, DAY_END = "06:00", "22:00"
#: Edge types that are service, not access.
TRAVEL_TYPES = {"bus", "trolleybus", "tram", "subway", "train", "ferry", "monorail", "funicular", "coach"}


def _busiest_week(days: pd.DatetimeIndex, trips_per_day: np.ndarray) -> list[date]:
    weekly = pd.Series(trips_per_day, index=days).groupby(days - pd.to_timedelta(days.dayofweek, unit="D")).sum()
    monday = weekly.idxmax().date()
    return [monday + timedelta(days=i) for i in range(7)]


def _own_span(feed, services: set[str]) -> tuple[date | None, date | None]:
    """First and last day of these services alone; removals are not days of service."""
    bounds = []
    calendar = feed.get("calendar")
    if calendar is not None and not calendar.empty:
        rows = calendar[calendar["service_id"].astype(str).isin(services)]
        for column in ("start_date", "end_date"):
            bounds += list(pd.to_datetime(rows[column], format="%Y%m%d", errors="coerce"))
    exceptions = feed.get("calendar_dates")
    if exceptions is not None and not exceptions.empty:
        added = exceptions["service_id"].astype(str).isin(services) & exceptions["exception_type"].astype(str).eq("1")
        bounds += list(pd.to_datetime(exceptions.loc[added, "date"], format="%Y%m%d", errors="coerce"))
    bounds = [bound for bound in bounds if pd.notna(bound)]
    return (min(bounds).date(), max(bounds).date()) if bounds else (None, None)


def wait_days(feed) -> tuple[list[date] | None, str]:
    """The days the waits are averaged over, and the rule that chose them."""
    from iduedu.gtfs.service_days import service_days

    trips_per_service = feed["trips"]["service_id"].astype(str).value_counts()
    reference = [REFERENCE_WEEK[0] + timedelta(days=i) for i in range(7)]
    found = service_days(feed, set(trips_per_service.index), dates=reference)
    if found is None:
        return reference, "reference week (feed has no calendar)"
    _, names, matrix = found
    active = {name for name, row in zip(names, matrix) if row.any()}
    source_of = pd.Series(trips_per_service.index.str.split(":").str[0], index=trips_per_service.index)
    days, rules = (list(reference), ["reference week"]) if active else ([], [])
    for source, group in trips_per_service.groupby(source_of):
        if set(group.index) & active:
            continue
        first, last = _own_span(feed, set(group.index))
        if first is None:
            continue
        whole = service_days(feed, set(group.index), first, min(last, first + timedelta(days=365)))
        if whole is None or not whole[2].any():
            continue
        span, own, own_matrix = whole
        week = _busiest_week(span, group.reindex(own).fillna(0).to_numpy() @ own_matrix)
        days += week
        rules.append(f"{source}: {week[0].isoformat()}")
    if not days:
        return None, "no service day in the calendar: all trips as one day"
    return sorted(set(days)), "; ".join(rules)


def city_crs(city_key: str):
    """The walk graph's CRS (else the transit graph's, else the OSM builders' estimate)."""
    from iduedu.graph.transformers import estimate_crs_for_bounds

    for path in (osm_graph_path(city_key, "walk"), osm_graph_path(city_key, "pt")):
        if path.exists():
            return read_urban_graph(path).nodes_gdf.crs
    return estimate_crs_for_bounds(*boundary(city_key).bounds)


def _summary(graph) -> dict:
    nodes, edges = graph.nodes_gdf, graph.edges_gdf
    row = {"n_nodes": len(nodes), "n_edges": len(edges)}
    if not edges.empty and "type" in edges:
        row["edge_types"] = ";".join(f"{k}:{v}" for k, v in sorted(edges["type"].value_counts().items()))
        boarding = edges.loc[edges["type"].eq("boarding"), "time_min"]
        if not boarding.empty:
            row["wait_median_min"] = round(float(boarding.median()), 3)
        travel = edges.loc[edges["type"].isin(TRAVEL_TYPES), "time_min"]
        if not travel.empty:  # a zero-time segment hands the network free travel
            zero = int((travel <= 0).sum())
            row.update(n_travel_edges=len(travel), zero_time_edges=zero, zero_time_share=round(zero / len(travel), 4))
    if not nodes.empty and "type" in nodes:
        modes = sorted(set(nodes["type"].dropna().map(str)) - {"platform", "station", "entrance", "boarding_area"})
        row["node_types"] = ";".join(modes)
    return row


def build_city(city_key: str) -> dict:
    row = {"city_key": city_key, "path": str(gtfs_graph_path(city_key)), "status": "failed"}
    started = time.perf_counter()
    feed = read_gtfs_feed(city_feed_path(city_key))
    days, rule = wait_days(feed)
    row.update(wait_days=len(days) if days else 0, wait_period_rule=rule[:300])
    graph = get_gtfs_public_transport_graph(
        feed,
        crs=city_crs(city_key),
        single_departure_wait_min=SINGLE_DEPARTURE_WAIT_MIN,
        service_dates=days,
        start_time=DAY_START,
        end_time=DAY_END,
    )
    row["elapsed_s"] = round(time.perf_counter() - started, 1)
    if graph.nodes_gdf.empty:
        row["status"] = "empty"
        return row
    write_urban_graph(graph, gtfs_graph_path(city_key))
    row.update(_summary(graph), crs=str(graph.nodes_gdf.crs))
    # every segment instantaneous: stops and headways are kept, run times are not usable
    row["status"] = "no_usable_times" if row.get("zero_time_share") == 1.0 else "ok"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cities", nargs="*")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    logger = setup("gtfs")
    keys = restrict(measurable(), arguments.cities)
    done = None if arguments.force else {key for key in keys if gtfs_graph_path(key).exists()}
    sweep(
        keys,
        build_city,
        GTFS_GRAPHS_CSV,
        force=arguments.force,
        done=done,
        describe=lambda rows: f"{rows[0]['status']} {rows[0].get('n_nodes', 0)} nodes, {rows[0].get('wait_days', 0)} days",
        logger=logger,
    )


if __name__ == "__main__":
    main()
