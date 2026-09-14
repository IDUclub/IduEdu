"""Pair OSM segments with GTFS segments, and recover the road limit OSM used.

The kinematic constants in ``DEFAULT_REGISTRY`` were fitted in a run whose code
did not survive. This module rebuilds the evidence they need: for every segment
that exists in both sources, how long the schedule says the vehicle takes, and
how long the model thinks it takes.

A pair is two segments whose **both** endpoints are within
``ENDPOINT_TOLERANCE_M`` and whose lengths agree within ``LENGTH_TOLERANCE``.
Requiring both ends is what makes the pair a pair: a single shared stop is served
by many routes going elsewhere.

**The road limit is recovered rather than fetched.** An OSM edge already carries
``length_meter`` and ``time_min``, and the model that produced the time is
invertible, so the limit that went into it can be read straight back out. No
Overpass traffic, and the recovered value is by construction the one the model
actually used -- refetching ``maxspeed`` would answer a different question.

Recovery fails in two honest ways, both recorded rather than guessed: the free
speed can be pinned at the mode's technical maximum (the limit was above it, so
its value left no trace) or at the floor. Those rows carry no limit and must not
be fitted against one.
"""

import argparse
import logging
from math import sqrt

import numpy as np
import pandas as pd
from bench_common import sweep
from scipy.spatial import cKDTree
from wide_cohort import measurable, restrict, with_gtfs_graph
from wide_paths import (
    ALL_OSM_MODES,
    PAIRS_CSV,
)
from wide_paths import PAIRS_SUMMARY_CSV as SUMMARY_CSV
from wide_paths import (
    SPEED_BY_LENGTH_CSV,
    gtfs_graph_path,
    osm_graph_path,
)

from iduedu import read_urban_graph
from iduedu.constants.transport_specs import DEFAULT_REGISTRY_W_TRAIN, TransportSpec

logger = logging.getLogger(__name__)

#: A segment pairs with its counterpart when both endpoints fall within this
#: distance and the two lengths agree to within ``LENGTH_TOLERANCE``.
ENDPOINT_TOLERANCE_M = 60.0
LENGTH_TOLERANCE = 0.25

MIN_SPEED_MPM = 60.0  # the builder's own floor: 1 m/s

#: Length classes for the descriptive question "does speed grow with distance?".
#: It does, roughly doubling from the shortest class to the longest without ever
#: flattening, which is why a single free-flow speed per mode cannot work.
LENGTH_BINS = [0, 125, 250, 500, 950, 1350, 2500, 5000, np.inf]


def _travel_edges(edges, nodes, modes: set[str]) -> pd.DataFrame:
    """Travel edges with their endpoint coordinates, in the nodes' metres.

    Nodes are passed separately because the two graphs pick their own local UTM
    zone and the GTFS side has to be reprojected onto the OSM one first.
    """
    if edges.empty or "type" not in edges:
        return pd.DataFrame()
    frame = edges.loc[edges["type"].isin(modes), ["u", "v", "type", "length_meter", "time_min"]].copy()
    if frame.empty:
        return frame
    positions = pd.DataFrame(
        {"x": nodes.geometry.x.to_numpy(), "y": nodes.geometry.y.to_numpy()},
        index=nodes.index,
    )
    for end in ("u", "v"):
        frame[f"{end}x"] = frame[end].map(positions["x"])
        frame[f"{end}y"] = frame[end].map(positions["y"])
    frame = frame.dropna(subset=["ux", "uy", "vx", "vy", "length_meter", "time_min"])
    return frame.loc[frame["length_meter"].gt(0) & frame["time_min"].gt(0)]


def recover_limit_kmh(spec: TransportSpec, length_m: float, time_min: float) -> tuple[float | None, str]:
    """Invert the travel-time model to get the road limit that produced the time."""
    span = max(spec.accel_dist_m, 0.0) + max(spec.brake_dist_m, 0.0)
    moving = time_min - spec.dwell_min
    if moving <= 0:
        return None, "dwell_exceeds_time"
    if span > 1e-9 and length_m < span:
        velocity = 2.0 * sqrt(length_m * span) / moving
    else:
        velocity = (length_m + span) / moving

    vmax = spec.vmax_tech_kmh * 1000.0 / 60.0
    if velocity >= vmax - 1e-6:
        return None, "capped_at_vmax"
    if velocity <= MIN_SPEED_MPM + 1e-6:
        return None, "floored"

    # Free speed is ``min(base, limit)``, so the limit leaves a trace only where it
    # binds. Where the mode runs at its own free speed the tag could have been
    # anything at or above it, and reporting a number would be inventing one.
    base = spec.base_speed_kmh * 1000.0 / 60.0
    if velocity >= base - 1e-6:
        return None, "limit_above_free_speed"
    return velocity * 60.0 / 1000.0, "ok"


def match_city(city_key: str) -> tuple[pd.DataFrame, dict]:
    """Every segment the two sources agree exists, with both their run times."""
    osm_path, gtfs_path = osm_graph_path(city_key, "pt", ALL_OSM_MODES), gtfs_graph_path(city_key)
    if not osm_path.exists() or not gtfs_path.exists():
        return pd.DataFrame(), {"city_key": city_key, "status": "missing_graph"}

    osm_graph, gtfs_graph = read_urban_graph(osm_path), read_urban_graph(gtfs_path)
    if osm_graph.nodes_gdf.empty or gtfs_graph.nodes_gdf.empty:
        return pd.DataFrame(), {"city_key": city_key, "status": "empty_graph"}
    osm_nodes = osm_graph.nodes_gdf
    gtfs_nodes = gtfs_graph.nodes_gdf.to_crs(osm_nodes.crs)

    modes = set(ALL_OSM_MODES)
    osm_edges = _travel_edges(osm_graph.edges_gdf, osm_nodes, modes)
    gtfs_edges = _travel_edges(gtfs_graph.edges_gdf, gtfs_nodes, modes)
    if osm_edges.empty or gtfs_edges.empty:
        return pd.DataFrame(), {"city_key": city_key, "status": "no_travel_edges"}

    pairs: list[dict] = []
    for mode in sorted(set(osm_edges["type"]) & set(gtfs_edges["type"])):
        spec = DEFAULT_REGISTRY_W_TRAIN.try_get(mode)
        left = gtfs_edges.loc[gtfs_edges["type"].eq(mode)]
        right = osm_edges.loc[osm_edges["type"].eq(mode)]
        if left.empty or right.empty:
            continue
        starts = np.column_stack([right["ux"].to_numpy(), right["uy"].to_numpy()])
        tree = cKDTree(starts)
        candidates = tree.query_ball_point(
            np.column_stack([left["ux"].to_numpy(), left["uy"].to_numpy()]), r=ENDPOINT_TOLERANCE_M
        )
        right_values = right[["vx", "vy", "length_meter", "time_min"]].to_numpy()
        left_values = left[["ux", "uy", "vx", "vy", "length_meter", "time_min"]].to_numpy()

        for index, options in enumerate(candidates):
            if not options:
                continue
            ux, uy, vx, vy, length, time_min = left_values[index]
            best, best_distance = None, np.inf
            for option in options:
                other_vx, other_vy, other_length, other_time = right_values[option]
                end_distance = float(np.hypot(vx - other_vx, vy - other_vy))
                if end_distance > ENDPOINT_TOLERANCE_M:
                    continue
                if abs(other_length - length) > LENGTH_TOLERANCE * length:
                    continue
                start_distance = float(np.hypot(ux - starts[option][0], uy - starts[option][1]))
                if start_distance + end_distance < best_distance:
                    best, best_distance = option, start_distance + end_distance
            if best is None:
                continue
            other_vx, other_vy, other_length, other_time = right_values[best]
            limit_kmh, recovery = (
                recover_limit_kmh(spec, float(other_length), float(other_time)) if spec else (None, "unknown_mode")
            )
            pairs.append(
                {
                    "city_key": city_key,
                    "mode": mode,
                    "gtfs_length_m": round(float(length), 1),
                    "gtfs_time_min": round(float(time_min), 4),
                    "osm_length_m": round(float(other_length), 1),
                    "osm_time_min": round(float(other_time), 4),
                    "osm_limit_kmh": None if limit_kmh is None else round(limit_kmh, 1),
                    "limit_recovery": recovery,
                    "endpoint_error_m": round(best_distance, 1),
                }
            )

    frame = pd.DataFrame(pairs)
    summary = {
        "city_key": city_key,
        "status": "ok",
        "n_gtfs_edges": len(gtfs_edges),
        "n_osm_edges": len(osm_edges),
        "n_matched": len(frame),
        "match_share_gtfs": round(len(frame) / len(gtfs_edges), 4) if len(gtfs_edges) else 0.0,
    }
    if not frame.empty:
        summary["n_with_limit"] = int(frame["osm_limit_kmh"].notna().sum())
        summary["modes"] = ";".join(sorted(frame["mode"].unique()))
    return frame, summary


def speed_by_length(pairs: pd.DataFrame) -> pd.DataFrame:
    """Observed speed against segment length -- the shape the model must reproduce."""
    frame = pairs.copy()
    frame["speed_kmh"] = frame["gtfs_length_m"] / frame["gtfs_time_min"] * 60.0 / 1000.0
    frame["length_class"] = pd.cut(frame["gtfs_length_m"], bins=LENGTH_BINS)
    grouped = frame.groupby(["mode", "length_class"], observed=True)["speed_kmh"]
    result = grouped.agg(["count", "median"]).reset_index()
    result["median"] = result["median"].round(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="*", help="city keys; default is every city with both graphs")
    parser.add_argument("--force", action="store_true", help="rematch cities already recorded")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    collected: list[pd.DataFrame] = []

    # Pairs are appended as they are produced, which is what makes an interrupted
    # sweep resumable -- and what silently doubled the file the first time a run
    # used --force, because every city was matched again and appended again. A
    # forced run therefore starts from an empty file, exactly as it starts from an
    # empty summary.
    if arguments.force and PAIRS_CSV.exists():
        PAIRS_CSV.unlink()
        logger.info("--force: cleared %s so re-matched cities do not duplicate", PAIRS_CSV.name)

    def work(city_key: str) -> dict:
        """The pairs go to their own file; only the summary is a sweep row."""
        frame, summary = match_city(city_key)
        if not frame.empty:
            frame.to_csv(PAIRS_CSV, mode="a", header=not PAIRS_CSV.exists(), index=False, encoding="utf-8")
            collected.append(frame)
        return summary

    # Matching compares run times, so a schedule that carries none is no use here.
    sweep(
        restrict(sorted(set(measurable()) & with_gtfs_graph()), arguments.cities),
        work,
        SUMMARY_CSV,
        force=arguments.force,
        logger=logger,
        describe=lambda rows: f"{rows[0].get('status')} matched {rows[0].get('n_matched', 0)}",
    )

    # From the whole pairs file, never from this run's additions. A resumed sweep
    # collects only the cities it matched, and writing the summary from those alone
    # replaced a table over 74 cities with one over the handful just added.
    if collected and PAIRS_CSV.exists():
        speed_by_length(pd.read_csv(PAIRS_CSV)).to_csv(SPEED_BY_LENGTH_CSV, index=False)
        logger.info(f"written {SPEED_BY_LENGTH_CSV} from all pairs on disk")


if __name__ == "__main__":
    main()
