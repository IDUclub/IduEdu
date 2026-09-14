"""Coverage in both directions: how much of each source the other one knows.

The metric this replaced counted route relations, and the previous run concluded
it "means nothing at all": OSM splits a route's variants and directions into
separate relations while a feed keeps one ``route_id``, so the ratio measures
mapping convention, not content. Spatial matching is immune to that.

Two shares, per mode, per city:

* **feed to OSM** -- share of the feed's stops that have an OSM node of the same
  mode within R;
* **OSM to feed** -- the reverse.

Keeping them apart is the point, because they diagnose different defects:

    Sofia     0.96 / 0.94   both sides complete
    Tokyo     0.82 / 0.41   the *feed* is partial -- one operator of many
    Curitiba  0.33 / 0.96   OSM is a strict subset: correct, but a third of it
    Bogota    0.17 / 0.57   OSM has the trunk BRT only

A single "completeness" number would have called Tokyo and Curitiba equally
incomplete while they are opposite situations.

Three radii are computed rather than one. The tolerance that matched segments in
the previous run was 60 m, but a platform pair across a street is routinely
80--100 m apart in one source and a single node in the other, so a single radius
would be a hidden parameter of the headline result. The ladder makes the result's
sensitivity to it visible, and 100 m is reported as primary.

Both sides are clipped to the same OSM relation, so a difference here is a
difference in content and not in extent.
"""

import argparse
import logging
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from bench_common import sweep
from scipy.spatial import cKDTree
from wide_cohort import measurable, restrict
from wide_osm import MODE_ALIASES
from wide_paths import ALL_OSM_MODES, COVERAGE_CSV, city_feed_path, osm_graph_path

from iduedu import read_urban_graph
from iduedu.graph_builders.gtfs_builders import _transport_type
from iduedu.gtfs.reader import read_gtfs_feed

logger = logging.getLogger(__name__)

#: Radii in metres. The middle one is the headline; the others show its influence.
MATCH_RADII_M = (50, 100, 150)
PRIMARY_RADIUS_M = 100

#: Positions closer than this are the same stop. Both sources describe a stop once
#: per route, so without this the denominators count route-stops and a city with
#: many overlapping routes looks better covered than it is.
SAME_STOP_M = 1.0

#: Node types that are structure, not service: they carry no mode and must not
#: enter either denominator.
STRUCTURE_TYPES = {"platform", "station", "entrance", "exit", "boarding_area", "node"}


def feed_stops(city_key: str) -> gpd.GeoDataFrame:
    """The city's stops with the modes calling at them, one row per stop and mode.

    Modes come from ``stop_times -> trips -> routes``, not from the feed's
    self-description: a stop's ``location_type`` says nothing about what serves it.
    """
    feed = read_gtfs_feed(city_feed_path(city_key))
    routes = feed["routes"].copy()
    if "transport_type" not in routes:
        routes["transport_type"] = ""
    routes["mode"] = [
        _transport_type(route_type, custom)
        for route_type, custom in zip(routes["route_type"], routes["transport_type"])
    ]
    trips = feed["trips"].merge(routes[["route_id", "mode"]], on="route_id", how="left", validate="many_to_one")
    calls = feed["stop_times"][["trip_id", "stop_id"]].merge(
        trips[["trip_id", "mode"]], on="trip_id", how="inner", validate="many_to_one"
    )
    pairs = calls[["stop_id", "mode"]].drop_duplicates()

    stops = feed["stops"].copy()
    if "parent_station" not in stops:
        stops["parent_station"] = ""
    stops["_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    coordinates = stops.set_index("stop_id")[["_lat", "_lon"]]
    missing = coordinates["_lat"].isna() | coordinates["_lon"].isna()
    if missing.any():  # a platform without coordinates inherits its station's
        parents = stops.set_index("stop_id")["parent_station"].reindex(coordinates.index[missing])
        inherited = coordinates.reindex(parents.to_numpy())
        coordinates.loc[missing, ["_lat", "_lon"]] = inherited.to_numpy()

    pairs = pairs.join(coordinates, on="stop_id").dropna(subset=["_lat", "_lon"])
    return gpd.GeoDataFrame(
        pairs[["stop_id", "mode"]],
        geometry=gpd.points_from_xy(pairs["_lon"], pairs["_lat"]),
        crs="EPSG:4326",
    )


def osm_stops(city_key: str) -> gpd.GeoDataFrame | None:
    """Service nodes of the city's OSM public-transport graph, by mode."""
    path = osm_graph_path(city_key, "pt", ALL_OSM_MODES)
    if not path.exists():
        return None
    nodes = read_urban_graph(path).nodes_gdf
    if nodes.empty or "type" not in nodes:
        return nodes.iloc[0:0] if not nodes.empty else gpd.GeoDataFrame(geometry=[], crs=None)
    modes = nodes["type"].map(str).map(lambda value: MODE_ALIASES.get(value, value))
    service = nodes.loc[modes.isin(ALL_OSM_MODES)].copy()
    service["mode"] = modes.loc[service.index]
    return service[["mode", "geometry"]]


def _positions(frame: gpd.GeoDataFrame, mode: str) -> np.ndarray:
    """Distinct stop positions of one mode, as an (n, 2) array of metres."""
    subset = frame.loc[frame["mode"].eq(mode)]
    if subset.empty:
        return np.empty((0, 2))
    points = np.column_stack([subset.geometry.x.to_numpy(), subset.geometry.y.to_numpy()])
    if len(points) < 2:
        return points
    # Collapse route-stops onto stops: cluster by a grid whose cell is SAME_STOP_M.
    keys = np.round(points / SAME_STOP_M).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def _share_within(source: np.ndarray, target: np.ndarray, radius: float) -> float | None:
    """Share of ``source`` positions with a ``target`` position within ``radius``."""
    if len(source) == 0:
        return None
    if len(target) == 0:
        return 0.0
    distances, _ = cKDTree(target).query(source, distance_upper_bound=radius)
    return float(np.isfinite(distances).mean())


#: Beyond this, the nearest counterpart is not a missed match but another part of
#: town. Redland City's feed describes the eastern islands while its OSM data
#: describes the western mainland: both inside one boundary, 11 km apart, and a
#: coverage of zero that means something entirely different from a sparse map.
ELSEWHERE_M = 1000.0


def _distance_profile(source: np.ndarray, target: np.ndarray) -> dict:
    """How far the nearest counterpart is -- absence and displacement look alike
    in a share and completely different here."""
    if len(source) == 0 or len(target) == 0:
        return {}
    distances, _ = cKDTree(target).query(source)
    return {
        "nearest_median_m": round(float(np.median(distances)), 1),
        "nearest_p90_m": round(float(np.quantile(distances, 0.9)), 1),
        "share_elsewhere": round(float((distances > ELSEWHERE_M).mean()), 4),
    }


def measure_city(city_key: str) -> list[dict]:
    """Both shares for every mode either source has, plus a pooled city row."""
    started = time.perf_counter()
    osm = osm_stops(city_key)
    if osm is None:
        return [{"city_key": city_key, "mode": "*", "status": "osm_missing"}]

    feed = feed_stops(city_key)
    if osm is not None and len(osm) and osm.crs is not None:
        feed = feed.to_crs(osm.crs)
    else:  # an empty OSM graph carries no CRS; metres still needed for the feed
        feed = feed.to_crs(feed.estimate_utm_crs())

    modes = sorted(set(feed["mode"]) | (set(osm["mode"]) if len(osm) else set()))
    rows: list[dict] = []
    for mode in modes:
        feed_points = _positions(feed, mode)
        osm_points = _positions(osm, mode) if len(osm) else np.empty((0, 2))
        # A mode OSM cannot be asked for is not evidence about OSM. It is recorded
        # so the analysis can exclude it, never counted as a zero.
        fetchable = mode in ALL_OSM_MODES
        row = {
            "city_key": city_key,
            "mode": mode,
            "fetchable_from_osm": fetchable,
            "n_feed_stops": len(feed_points),
            "n_osm_stops": len(osm_points),
            "status": "ok",
        }
        for radius in MATCH_RADII_M:
            forward = _share_within(feed_points, osm_points, radius) if fetchable else None
            backward = _share_within(osm_points, feed_points, radius)
            row[f"feed_to_osm_{radius}"] = None if forward is None else round(forward, 4)
            row[f"osm_to_feed_{radius}"] = None if backward is None else round(backward, 4)
        if fetchable:
            row.update(_distance_profile(feed_points, osm_points))
        rows.append(row)

    pooled = [row for row in rows if row["fetchable_from_osm"]]
    summary = {
        "city_key": city_key,
        "mode": "*",
        "fetchable_from_osm": True,
        "n_feed_stops": sum(row["n_feed_stops"] for row in pooled),
        "n_osm_stops": sum(row["n_osm_stops"] for row in pooled),
        "modes_feed": ";".join(sorted({row["mode"] for row in rows if row["n_feed_stops"]})),
        "modes_osm": ";".join(sorted({row["mode"] for row in rows if row["n_osm_stops"]})),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "status": "ok",
    }
    # The pooled share is weighted by stops, so a city is not summarised by its
    # rarest mode. Modes are pooled only where OSM could have been asked.
    for radius in MATCH_RADII_M:
        for direction, denominator in (("feed_to_osm", "n_feed_stops"), ("osm_to_feed", "n_osm_stops")):
            total = sum(row[denominator] for row in pooled)
            matched = sum(
                row[denominator] * row[f"{direction}_{radius}"]
                for row in pooled
                if row[f"{direction}_{radius}"] is not None
            )
            summary[f"{direction}_{radius}"] = round(matched / total, 4) if total else None
    rows.append(summary)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="*", help="city keys; default is every city with a boundary")
    parser.add_argument("--force", action="store_true", help="recompute cities already recorded")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def describe(rows: list[dict]) -> str:
        headline = next((row for row in rows if row.get("mode") == "*"), rows[0])
        return (
            f"{headline.get('status')} "
            f"feed->osm {headline.get(f'feed_to_osm_{PRIMARY_RADIUS_M}')} "
            f"osm->feed {headline.get(f'osm_to_feed_{PRIMARY_RADIUS_M}')}"
        )

    sweep(
        restrict(measurable(), arguments.cities),
        measure_city,
        COVERAGE_CSV,
        force=arguments.force,
        logger=logger,
        describe=describe,
        failure_extra={"mode": "*"},
    )


if __name__ == "__main__":
    main()
