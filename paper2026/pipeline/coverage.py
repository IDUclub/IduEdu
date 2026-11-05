"""The two networks compared directly: stops, their matches, and the feeds' geometry.

* ``stops`` -- coverage in both directions, per mode and pooled: the share of the
  feed's stops with an OSM stop of the same mode within 50/100/150 m, and the
  reverse. Positions closer than 1 m are one stop (both sources describe a stop once
  per route). Nothing is chosen between candidates; many-to-one is allowed.
  -> ``indicators_coverage.csv``
* ``provenance`` -- share of the feed's shape vertices within 10 cm of an OSM vertex;
  above 0.30 the feed's geometry was traced from OSM. -> ``feed_provenance.csv``
* ``displacement`` -- distance between the two bus-stop clouds' centres over their
  spread: below one, one lies inside the other; above one, they sit apart.
  -> ``displacement.csv``

    python pipeline/coverage.py all | stops | provenance | displacement
"""

import argparse
import io
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from common import (
    ALL_OSM_MODES,
    COVERAGE_CSV,
    PROVENANCE_CSV,
    RESULTS_DIR,
    city_feed_path,
    gtfs_graph_path,
    last_rows,
    measurable,
    osm_graph_path,
    restrict,
    setup,
    sweep,
    with_gtfs_graph,
)
from scipy.spatial import cKDTree

from iduedu import read_urban_graph
from iduedu.graph_builders.gtfs_builders import _transport_type
from iduedu.gtfs.reader import read_gtfs_feed

RADII_M = (50, 100, 150)
PRIMARY_RADIUS_M = 100
SAME_STOP_M = 1.0
#: Beyond this the nearest counterpart is another part of town, not a missed match.
ELSEWHERE_M = 1000.0
#: OSM node types that name a mode differently from the registry.
MODE_ALIASES = {"rail": "train", "metro": "subway", "trolley": "trolleybus", "light_rail": "tram"}
#: A copied coordinate is exact to ~1 cm; 10 cm allows for reprojection, not for GPS.
COINCIDENCE_M, LOOSE_M = 0.10, 2.0
MAX_VERTICES = 60000
DISPLACEMENT_CSV = RESULTS_DIR / "displacement.csv"


# --- stops of each side ----------------------------------------------------------


def feed_stops(city_key: str) -> gpd.GeoDataFrame:
    """One row per stop and mode calling at it (modes from stop_times -> trips -> routes)."""
    feed = read_gtfs_feed(city_feed_path(city_key))
    routes = feed["routes"].copy()
    if "transport_type" not in routes:
        routes["transport_type"] = ""
    routes["mode"] = [_transport_type(t, custom) for t, custom in zip(routes["route_type"], routes["transport_type"])]
    trips = feed["trips"].merge(routes[["route_id", "mode"]], on="route_id", how="left", validate="many_to_one")
    calls = feed["stop_times"][["trip_id", "stop_id"]].merge(
        trips[["trip_id", "mode"]], on="trip_id", validate="many_to_one"
    )
    pairs = calls[["stop_id", "mode"]].drop_duplicates()

    stops = feed["stops"].copy()
    if "parent_station" not in stops:
        stops["parent_station"] = ""
    stops["_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    coordinates = stops.set_index("stop_id")[["_lat", "_lon"]]
    missing = coordinates["_lat"].isna() | coordinates["_lon"].isna()
    if missing.any():  # a platform without coordinates takes its station's
        parents = stops.set_index("stop_id")["parent_station"].reindex(coordinates.index[missing])
        coordinates.loc[missing, ["_lat", "_lon"]] = coordinates.reindex(parents.to_numpy()).to_numpy()
    pairs = pairs.join(coordinates, on="stop_id").dropna(subset=["_lat", "_lon"])
    return gpd.GeoDataFrame(
        pairs[["stop_id", "mode"]], geometry=gpd.points_from_xy(pairs["_lon"], pairs["_lat"]), crs=4326
    )


def osm_stops(city_key: str) -> gpd.GeoDataFrame | None:
    """Service nodes of the city's OSM transit graph, by mode."""
    path = osm_graph_path(city_key, "pt")
    if not path.exists():
        return None
    nodes = read_urban_graph(path).nodes_gdf
    if nodes.empty or "type" not in nodes:
        return nodes.iloc[0:0] if not nodes.empty else gpd.GeoDataFrame(geometry=[], crs=None)
    modes = nodes["type"].map(str).map(lambda value: MODE_ALIASES.get(value, value))
    service = nodes.loc[modes.isin(ALL_OSM_MODES)].copy()
    service["mode"] = modes.loc[service.index]
    return service[["mode", "geometry"]]


def positions(frame: gpd.GeoDataFrame, mode: str) -> np.ndarray:
    """Distinct stop positions of one mode, in metres (route-stops within 1 m are one stop)."""
    subset = frame.loc[frame["mode"].eq(mode)]
    points = (
        np.column_stack([subset.geometry.x.to_numpy(), subset.geometry.y.to_numpy()])
        if len(subset)
        else np.empty((0, 2))
    )
    if len(points) < 2:
        return points
    _, first = np.unique(np.round(points / SAME_STOP_M).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(first)]


def both_sides(city_key: str):
    osm = osm_stops(city_key)
    if osm is None:
        return None, None
    feed = feed_stops(city_key)
    feed = feed.to_crs(osm.crs) if len(osm) and osm.crs is not None else feed.to_crs(feed.estimate_utm_crs())
    return feed, osm


# --- stops: coverage in both directions ------------------------------------------


def _share_within(source: np.ndarray, target: np.ndarray, radius: float) -> float | None:
    if len(source) == 0:
        return None
    if len(target) == 0:
        return 0.0
    distances, _ = cKDTree(target).query(source, distance_upper_bound=radius)
    return float(np.isfinite(distances).mean())


def coverage_city(city_key: str) -> list[dict]:
    feed, osm = both_sides(city_key)
    if osm is None:
        return [{"city_key": city_key, "mode": "*", "status": "osm_missing"}]
    rows = []
    for mode in sorted(set(feed["mode"]) | (set(osm["mode"]) if len(osm) else set())):
        feed_points = positions(feed, mode)
        osm_points = positions(osm, mode) if len(osm) else np.empty((0, 2))
        fetchable = mode in ALL_OSM_MODES  # a mode OSM was never asked for is no evidence about OSM
        row = {"city_key": city_key, "mode": mode, "fetchable_from_osm": fetchable,
               "n_feed_stops": len(feed_points), "n_osm_stops": len(osm_points), "status": "ok"}  # fmt: skip
        for radius in RADII_M:
            forward = _share_within(feed_points, osm_points, radius) if fetchable else None
            backward = _share_within(osm_points, feed_points, radius)
            row[f"feed_to_osm_{radius}"] = None if forward is None else round(forward, 4)
            row[f"osm_to_feed_{radius}"] = None if backward is None else round(backward, 4)
        if fetchable and len(feed_points) and len(osm_points):
            distances, _ = cKDTree(osm_points).query(feed_points)
            row.update(
                nearest_median_m=round(float(np.median(distances)), 1),
                nearest_p90_m=round(float(np.quantile(distances, 0.9)), 1),
                share_elsewhere=round(float((distances > ELSEWHERE_M).mean()), 4),
            )
        rows.append(row)
    pooled = [row for row in rows if row["fetchable_from_osm"]]
    summary = {
        "city_key": city_key, "mode": "*", "fetchable_from_osm": True,
        "n_feed_stops": sum(r["n_feed_stops"] for r in pooled), "n_osm_stops": sum(r["n_osm_stops"] for r in pooled),
        "modes_feed": ";".join(sorted({r["mode"] for r in rows if r["n_feed_stops"]})),
        "modes_osm": ";".join(sorted({r["mode"] for r in rows if r["n_osm_stops"]})), "status": "ok",
    }  # fmt: skip
    for radius in RADII_M:  # pooled over modes, weighted by stops
        for direction, denominator in (("feed_to_osm", "n_feed_stops"), ("osm_to_feed", "n_osm_stops")):
            total = sum(r[denominator] for r in pooled)
            matched = sum(
                r[denominator] * r[f"{direction}_{radius}"] for r in pooled if r[f"{direction}_{radius}"] is not None
            )
            summary[f"{direction}_{radius}"] = round(matched / total, 4) if total else None
    return rows + [summary]


# --- provenance -------------------------------------------------------------------


def _feed_vertices(city_key: str, crs) -> np.ndarray:
    with zipfile.ZipFile(city_feed_path(city_key)) as handle:
        if "shapes.txt" not in handle.namelist():
            return np.empty((0, 2))
        shapes = pd.read_csv(io.BytesIO(handle.read("shapes.txt")), dtype=str)
    latitude = pd.to_numeric(shapes.get("shape_pt_lat"), errors="coerce")
    longitude = pd.to_numeric(shapes.get("shape_pt_lon"), errors="coerce")
    valid = latitude.notna() & longitude.notna()
    if not valid.any():
        return np.empty((0, 2))
    points = gpd.GeoSeries(gpd.points_from_xy(longitude[valid], latitude[valid]), crs=4326).to_crs(crs)
    coordinates = np.column_stack([points.x.to_numpy(), points.y.to_numpy()])
    if len(coordinates) > MAX_VERTICES:
        coordinates = coordinates[np.random.default_rng(0).choice(len(coordinates), MAX_VERTICES, replace=False)]
    return coordinates


def provenance_city(city_key: str) -> dict:
    row = {"city_key": city_key}
    path = osm_graph_path(city_key, "pt")
    edges = read_urban_graph(path).edges_gdf if path.exists() else None
    if edges is None or edges.empty or edges.geometry.isna().all():
        row.update(n_osm_vertices=0, status="no_osm_geometry")
        return row
    osm_points = np.concatenate([np.asarray(g.coords) for g in edges.geometry.dropna() if not g.is_empty])[:, :2]
    feed_points = _feed_vertices(city_key, edges.crs)
    row.update(n_osm_vertices=len(osm_points), n_feed_vertices=len(feed_points))
    if len(feed_points) == 0:
        row["status"] = "no_feed_shapes"
        return row
    distances, _ = cKDTree(osm_points).query(feed_points)
    row.update(
        coincident_share=round(float((distances <= COINCIDENCE_M).mean()), 4),
        near_share=round(float((distances <= LOOSE_M).mean()), 4),
        median_distance_m=round(float(np.median(distances)), 3),
        status="ok",
    )
    return row


# --- displacement -----------------------------------------------------------------


def _bus_cloud(nodes):
    if "type" in nodes.columns:
        nodes = nodes.loc[nodes["type"].astype(str).eq("bus")]
    return np.column_stack([nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()]) if len(nodes) >= 3 else None


def displacement_city(city_key: str) -> dict:
    row = {"city_key": city_key}
    if not (osm_graph_path(city_key, "pt").exists() and gtfs_graph_path(city_key).exists()):
        row["status"] = "missing_graph"
        return row
    osm = _bus_cloud(read_urban_graph(osm_graph_path(city_key, "pt")).nodes_gdf)
    gtfs = _bus_cloud(read_urban_graph(gtfs_graph_path(city_key)).nodes_gdf)
    if osm is None or gtfs is None:
        row["status"] = "too_few_stops"
        return row
    osm_centre, gtfs_centre = osm.mean(axis=0), gtfs.mean(axis=0)
    separation = float(np.linalg.norm(osm_centre - gtfs_centre))
    spread = float(
        np.mean([np.sqrt(np.mean(np.sum((c - m) ** 2, axis=1))) for c, m in ((osm, osm_centre), (gtfs, gtfs_centre))])
    )
    row.update(n_osm_bus=len(osm), n_gtfs_bus=len(gtfs), separation_m=round(separation, 1), spread_m=round(spread, 1),
               ratio=round(separation / spread, 3) if spread else None, status="ok")  # fmt: skip
    return row


# --- bus cases ---------------------------------------------------------------------

COVERED, SPARSE = "covered", "sparse in the same area"
ELSEWHERE, NO_NETWORK = "sources describe different areas", "no bus network in OpenStreetMap"
CASE_ORDER = [COVERED, SPARSE, ELSEWHERE, NO_NETWORK]
#: Below both, OSM holds a handful of stops rather than a network.
NO_NETWORK_STOPS, NO_NETWORK_SHARE = 10, 0.05
COVERED_SHARE, ELSEWHERE_SHARE, DISPLACED_RATIO = 0.5, 0.5, 1.0


def classify_bus(per_mode: pd.DataFrame) -> pd.DataFrame:
    """A ``case`` per city for its bus coverage; substance is tested before position.

    A city is "elsewhere" only when most feed stops are a kilometre from any OSM stop
    **and** the two clouds sit further apart than their own size (Pune: OSM covers the
    centre of a feed that covers the agglomeration -- far, but nested)."""
    bus = per_mode.loc[per_mode["mode"].eq("bus") & per_mode["n_feed_stops"].gt(0)].copy()
    ratios = {}
    if DISPLACEMENT_CSV.exists():
        displacement = last_rows(DISPLACEMENT_CSV, ["city_key"])
        ratios = dict(displacement.loc[displacement["status"] == "ok", ["city_key", "ratio"]].itertuples(index=False))

    def case(row) -> str:
        if row["n_osm_stops"] < NO_NETWORK_STOPS and row["n_osm_stops"] < NO_NETWORK_SHARE * row["n_feed_stops"]:
            return NO_NETWORK
        if row["feed_to_osm_100"] >= COVERED_SHARE:
            return COVERED
        if row.get("share_elsewhere", 0) >= ELSEWHERE_SHARE:
            ratio = ratios.get(row["city_key"])
            if ratio is None or ratio > DISPLACED_RATIO:
                return ELSEWHERE
        return SPARSE

    bus["case"] = bus.apply(case, axis=1)
    return bus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["all", "stops", "provenance", "displacement"])
    parser.add_argument("--cities", nargs="*")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    logger = setup("coverage")
    keys = restrict(measurable(), arguments.cities)
    steps = ["stops", "provenance", "displacement"] if arguments.step == "all" else [arguments.step]
    if "stops" in steps:
        sweep(keys, coverage_city, COVERAGE_CSV, force=arguments.force, row_keys=["city_key", "mode"], logger=logger,
              describe=lambda rows: f"feed->osm {rows[-1].get(f'feed_to_osm_{PRIMARY_RADIUS_M}')} "
                                    f"osm->feed {rows[-1].get(f'osm_to_feed_{PRIMARY_RADIUS_M}')}")  # fmt: skip
    if "provenance" in steps:
        sweep(keys, provenance_city, PROVENANCE_CSV, force=arguments.force, logger=logger,
              describe=lambda rows: f"{rows[0].get('status')} coincident {rows[0].get('coincident_share')}")  # fmt: skip
    if "displacement" in steps:
        timed = sorted(set(keys) & with_gtfs_graph(include_untimed=True))
        sweep(timed, displacement_city, DISPLACEMENT_CSV, force=arguments.force, logger=logger,
              describe=lambda rows: f"{rows[0].get('status')} ratio {rows[0].get('ratio')}")  # fmt: skip


if __name__ == "__main__":
    main()
