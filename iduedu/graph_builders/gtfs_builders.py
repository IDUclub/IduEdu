from datetime import date, datetime, time
from math import sqrt
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely import LineString
from shapely.ops import substring

from iduedu import config
from iduedu.graph.transformers import estimate_crs_for_bounds
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.gtfs.merge import merge_gtfs_feeds
from iduedu.gtfs.reader import GTFSFeed, read_gtfs_feed
from iduedu.gtfs.validation import GTFSValidationError, validate_gtfs_feed

logger = config.logger

DEFAULT_WALK_SPEED_M_PER_MIN = 5 * 1000 / 60
GTFS_ROUTE_TYPES = {
    0: "tram",
    1: "subway",
    2: "train",
    3: "bus",
    4: "ferry",
    5: "cable_tram",
    6: "aerial_lift",
    7: "funicular",
    11: "trolleybus",
    12: "monorail",
}

# Hierarchical codes come from Google's Extended Route Types, which the GTFS reference itself
# does not cover -- it documents 0-12 only. Ranges missing from that table (300, 500, 600, 1600
# and 1700 "Miscellaneous") are deliberately left unmapped: a feed using them gets an honest
# ``public_transport`` rather than a guess based on neighbouring codes.
GTFS_EXTENDED_ROUTE_TYPES: tuple[tuple[int, int, str], ...] = (
    (100, 117, "train"),
    (200, 209, "coach"),  # intercity/long-distance, kept apart from city buses
    (400, 404, "subway"),
    (405, 405, "monorail"),
    (406, 499, "subway"),
    (700, 716, "bus"),
    (800, 899, "trolleybus"),
    (900, 906, "tram"),
    (1000, 1000, "ferry"),
    (1200, 1200, "ferry"),
    (1300, 1300, "aerial_lift"),
    (1400, 1400, "funicular"),
    (1500, 1500, "taxi"),
)


def _parse_service_date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"service_date must be YYYY-MM-DD or YYYYMMDD, got {value!r}")


def _parse_time_bound(value: time | str | int | float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, time):
        return float(value.hour * 3600 + value.minute * 60 + value.second)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return float(value)
    parts = str(value).strip().split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"{name} must use HH:MM or HH:MM:SS, got {value!r}")
    try:
        hours, minutes = int(parts[0]), int(parts[1])
        seconds = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise ValueError(f"{name} must use HH:MM or HH:MM:SS, got {value!r}") from exc
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"Invalid {name}: {value!r}")
    return float(hours * 3600 + minutes * 60 + seconds)


def _time_seconds(values: pd.Series, field: str) -> pd.Series:
    text = values.astype(str).str.strip()
    result = pd.to_timedelta(text.where(text.ne("")), errors="coerce").dt.total_seconds()
    invalid = text.ne("") & result.isna()
    if invalid.any():
        examples = text.loc[invalid].drop_duplicates().head(5).tolist()
        raise GTFSValidationError(f"{field} contains invalid GTFS times: {examples}")
    return result


def _window_mask(values: pd.Series, start: float | None, end: float | None) -> pd.Series:
    if start is None and end is None:
        return values.notna()
    adjusted = values.copy()
    adjusted_end = end
    if start is not None and end is not None and end < start:
        adjusted_end = end + 24 * 3600
        adjusted = adjusted.where(adjusted >= start, adjusted + 24 * 3600)
    mask = adjusted.notna()
    if start is not None:
        mask &= adjusted >= start
    if adjusted_end is not None:
        mask &= adjusted < adjusted_end
    return mask


def _active_service_ids(feed: GTFSFeed, service_date: date) -> set[str]:
    yyyymmdd = service_date.strftime("%Y%m%d")
    weekday = service_date.strftime("%A").lower()
    active: set[str] = set()

    calendar = feed.get("calendar")
    if calendar is not None and not calendar.empty:
        valid_range = calendar["start_date"].le(calendar["end_date"])
        mask = valid_range & calendar["start_date"].le(yyyymmdd) & calendar["end_date"].ge(yyyymmdd)
        if weekday in calendar.columns:
            mask &= calendar[weekday].eq("1")
        active.update(calendar.loc[mask, "service_id"])

    exceptions = feed.get("calendar_dates")
    if exceptions is not None and not exceptions.empty:
        today = exceptions.loc[exceptions["date"].eq(yyyymmdd)]
        active.update(today.loc[today["exception_type"].eq("1"), "service_id"])
        active.difference_update(today.loc[today["exception_type"].eq("2"), "service_id"])
    return active


def _transport_type(route_type: str, custom_type: str = "") -> str:
    custom = str(custom_type).strip().lower()
    if custom:
        aliases = {"trolley": "trolleybus", "metro": "subway", "rail": "train"}
        return aliases.get(custom, custom)
    try:
        numeric = int(route_type)
    except (TypeError, ValueError):
        return "public_transport"
    if numeric in GTFS_ROUTE_TYPES:
        return GTFS_ROUTE_TYPES[numeric]
    for low, high, name in GTFS_EXTENDED_ROUTE_TYPES:
        if low <= numeric <= high:
            return name
    return "public_transport"


def _prepare_trips(feed: GTFSFeed, service_date: date | None) -> pd.DataFrame:
    trips = feed["trips"].copy()
    for column in ("direction_id", "shape_id"):
        if column not in trips:
            trips[column] = ""
    if service_date is not None:
        active = _active_service_ids(feed, service_date)
        trips = trips.loc[trips["service_id"].isin(active)].copy()
    routes = feed["routes"].copy()
    if "transport_type" not in routes:
        routes["transport_type"] = ""
    for column in ("route_short_name", "route_long_name"):
        if column not in routes:
            routes[column] = ""
    route_columns = [
        column
        for column in ("route_id", "route_type", "transport_type", "route_short_name", "route_long_name")
        if column in routes
    ]
    trips = trips.merge(routes[route_columns], on="route_id", how="left", validate="many_to_one")
    trips["type"] = [
        _transport_type(route_type, custom) for route_type, custom in zip(trips["route_type"], trips["transport_type"])
    ]
    return trips


def _prepare_stop_times(feed: GTFSFeed, trips: pd.DataFrame) -> pd.DataFrame:
    stop_times = feed["stop_times"].copy()
    stop_times = stop_times.loc[stop_times["trip_id"].isin(trips["trip_id"])].copy()
    if stop_times.empty:
        return stop_times
    for column, default in (("pickup_type", "0"), ("drop_off_type", "0"), ("shape_dist_traveled", "")):
        if column not in stop_times:
            stop_times[column] = default
    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="raise").astype(np.int64)
    stop_times["arrival_seconds"] = _time_seconds(stop_times["arrival_time"], "stop_times.arrival_time")
    stop_times["departure_seconds"] = _time_seconds(stop_times["departure_time"], "stop_times.departure_time")

    arrival_missing = stop_times["arrival_seconds"].isna() & stop_times["departure_seconds"].notna()
    departure_missing = stop_times["departure_seconds"].isna() & stop_times["arrival_seconds"].notna()
    stop_times.loc[arrival_missing, "arrival_seconds"] = stop_times.loc[arrival_missing, "departure_seconds"]
    stop_times.loc[departure_missing, "departure_seconds"] = stop_times.loc[departure_missing, "arrival_seconds"]

    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"], kind="stable").reset_index(drop=True)
    both_missing = stop_times[["arrival_seconds", "departure_seconds"]].isna().all(axis=1)
    if both_missing.any():
        logger.warning(f"Interpolating times for {int(both_missing.sum())} stop_times rows without arrival/departure")
        for column in ("arrival_seconds", "departure_seconds"):
            stop_times[column] = stop_times.groupby("trip_id", sort=False)[column].transform(
                lambda values: values.interpolate(limit_area="inside")
            )
    return stop_times


def _assign_patterns(trips: pd.DataFrame, stop_times: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    signature_tokens = list(zip(stop_times["stop_id"], stop_times["pickup_type"], stop_times["drop_off_type"]))
    stop_times = stop_times.copy()
    stop_times["_signature_token"] = signature_tokens
    signatures = (
        stop_times.groupby("trip_id", sort=False)["_signature_token"].agg(tuple).rename("stop_signature").reset_index()
    )
    trips = trips.merge(signatures, on="trip_id", how="inner", validate="one_to_one")
    keys = list(zip(trips["route_id"], trips["direction_id"], trips["shape_id"], trips["stop_signature"]))
    trips["pattern_id"] = pd.factorize(pd.Series(keys, dtype=object), sort=False)[0].astype(np.int64)
    stop_times = stop_times.drop(columns="_signature_token").merge(
        trips[["trip_id", "pattern_id"]], on="trip_id", how="inner", validate="many_to_one"
    )
    stop_times["stop_position"] = stop_times.groupby("trip_id", sort=False).cumcount().astype(np.int64)
    return trips, stop_times


def _segment_times(stop_times: pd.DataFrame) -> pd.Series:
    next_arrival = stop_times.groupby("trip_id", sort=False)["arrival_seconds"].shift(-1)
    duration = (next_arrival - stop_times["departure_seconds"]) / 60
    duration = duration.where(duration >= 0)
    valid = duration.notna()
    result = (
        stop_times.loc[valid, ["pattern_id", "stop_position"]]
        .assign(time_min=duration.loc[valid])
        .groupby(["pattern_id", "stop_position"], sort=False)["time_min"]
        .median()
    )
    return result


def _scheduled_wait_times(
    stop_times: pd.DataFrame,
    frequency_trip_ids: set[str],
    start: float | None,
    end: float | None,
) -> tuple[pd.Series, pd.MultiIndex]:
    boardable = ~stop_times["pickup_type"].eq("1")
    if frequency_trip_ids:
        boardable &= ~stop_times["trip_id"].isin(frequency_trip_ids)
    boardable &= _window_mask(stop_times["departure_seconds"], start, end)
    departures = stop_times.loc[boardable, ["pattern_id", "stop_position", "departure_seconds"]].drop_duplicates()
    if departures.empty:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["pattern_id", "stop_position"])
        return pd.Series(index=empty_index, dtype=float), empty_index
    departures = departures.sort_values(["pattern_id", "stop_position", "departure_seconds"])
    groups = departures.groupby(["pattern_id", "stop_position"], sort=False)["departure_seconds"]
    counts = groups.size()
    single = counts[counts == 1].index
    gaps = groups.diff()
    waits = (
        departures.assign(gap=gaps)
        .dropna(subset=["gap"])
        .groupby(["pattern_id", "stop_position"], sort=False)["gap"]
        .mean()
        / 120
    )
    return waits, single


def _frequency_wait_times(
    feed: GTFSFeed, trips: pd.DataFrame, start: float | None, end: float | None
) -> tuple[pd.Series, set[str]]:
    frequencies = feed.get("frequencies")
    if frequencies is None or frequencies.empty:
        return pd.Series(dtype=float), set()
    frequencies = frequencies.loc[frequencies["trip_id"].isin(trips["trip_id"])].copy()
    if frequencies.empty:
        return pd.Series(dtype=float), set()
    frequency_trip_ids = set(frequencies["trip_id"])
    frequencies = frequencies.merge(trips[["trip_id", "pattern_id"]], on="trip_id", how="inner", validate="many_to_one")
    frequencies["_start"] = _time_seconds(frequencies["start_time"], "frequencies.start_time")
    frequencies["_end"] = _time_seconds(frequencies["end_time"], "frequencies.end_time")
    frequencies["_headway"] = pd.to_numeric(frequencies["headway_secs"], errors="coerce")
    invalid = (
        frequencies["_start"].isna()
        | frequencies["_end"].isna()
        | frequencies["_headway"].isna()
        | frequencies["_headway"].le(0)
        | frequencies["_end"].le(frequencies["_start"])
    )
    if invalid.any():
        raise GTFSValidationError(f"frequencies.txt has {int(invalid.sum())} invalid rows")

    adjusted_end = end
    if start is not None and end is not None and end < start:
        adjusted_end = end + 24 * 3600
        frequencies["_start"] = frequencies["_start"].where(
            frequencies["_start"] >= start, frequencies["_start"] + 24 * 3600
        )
        frequencies["_end"] = frequencies["_end"].where(frequencies["_end"] > start, frequencies["_end"] + 24 * 3600)
    overlap_start = frequencies["_start"] if start is None else frequencies["_start"].clip(lower=start)
    overlap_end = frequencies["_end"] if adjusted_end is None else frequencies["_end"].clip(upper=adjusted_end)
    frequencies["_duration"] = (overlap_end - overlap_start).clip(lower=0)
    frequencies = frequencies.loc[frequencies["_duration"].gt(0)].copy()
    if frequencies.empty:
        return pd.Series(dtype=float), frequency_trip_ids
    frequencies["_weighted_wait"] = frequencies["_duration"] * frequencies["_headway"] / 120
    grouped = frequencies.groupby("pattern_id", sort=False)
    waits = grouped["_weighted_wait"].sum() / grouped["_duration"].sum()
    return waits, frequency_trip_ids


def _boarding_wait_times(
    feed: GTFSFeed,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    start: float | None,
    end: float | None,
    single_departure_wait_min: float | None,
) -> pd.Series:
    frequency_wait, frequency_trip_ids = _frequency_wait_times(feed, trips, start, end)
    scheduled_wait, single = _scheduled_wait_times(stop_times, frequency_trip_ids, start, end)
    waits = scheduled_wait.copy()

    if not frequency_wait.empty:
        pattern_positions = stop_times.loc[
            ~stop_times["pickup_type"].eq("1") & stop_times["pattern_id"].isin(frequency_wait.index),
            ["pattern_id", "stop_position"],
        ].drop_duplicates()
        freq_index = pd.MultiIndex.from_frame(pattern_positions)
        freq_values = pd.Series(
            freq_index.get_level_values("pattern_id").map(frequency_wait), index=freq_index, dtype=float
        )
        waits = pd.concat([waits.loc[~waits.index.isin(freq_index)], freq_values])

    if single_departure_wait_min is not None and len(single) > 0:
        fallback = pd.Series(float(single_departure_wait_min), index=single)
        waits = pd.concat([waits, fallback.loc[~fallback.index.isin(waits.index)]])
    elif len(single) > 0:
        logger.warning(
            f"Omitting boarding edges for {len(single)} pattern stops with only one departure; "
            "set single_departure_wait_min to keep them"
        )
    if not isinstance(waits.index, pd.MultiIndex):
        waits.index = pd.MultiIndex.from_tuples(waits.index, names=["pattern_id", "stop_position"])
    else:
        waits.index = waits.index.set_names(["pattern_id", "stop_position"])
    return waits


def _resolved_stop_coordinates(stops: pd.DataFrame, needed_ids: set[str]) -> pd.DataFrame:
    stops = stops.copy().set_index("stop_id", drop=False)
    for column in ("location_type", "parent_station", "stop_access"):
        if column not in stops:
            stops[column] = ""
    stops["_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")

    # Generic pathway nodes and boarding areas may omit coordinates. Their parent
    # station/platform gives a stable Point fallback required by UrbanGraph.
    for _ in range(3):
        missing = stops["_lat"].isna() | stops["_lon"].isna()
        parents = stops.loc[missing, "parent_station"]
        known_parent = parents.isin(stops.index)
        if not known_parent.any():
            break
        rows = parents.index[known_parent]
        parent_rows = stops.loc[parents.loc[rows]]
        stops.loc[rows, "_lat"] = parent_rows["_lat"].to_numpy()
        stops.loc[rows, "_lon"] = parent_rows["_lon"].to_numpy()

    missing_needed = pd.Index(needed_ids).difference(stops.index)
    if len(missing_needed) > 0:
        raise GTFSValidationError(f"Graph references missing stops: {missing_needed[:10].tolist()}")
    needed = stops.loc[sorted(needed_ids)].copy()
    missing_geometry = needed["_lat"].isna() | needed["_lon"].isna()
    if missing_geometry.any():
        ids = needed.loc[missing_geometry, "stop_id"].head(10).tolist()
        raise GTFSValidationError(f"Cannot infer coordinates for GTFS stops: {ids}")
    return needed


def _target_crs(stops: pd.DataFrame, crs: Any | None) -> CRS:
    if crs is None:
        result = estimate_crs_for_bounds(
            float(stops["_lon"].min()),
            float(stops["_lat"].min()),
            float(stops["_lon"].max()),
            float(stops["_lat"].max()),
        )
    else:
        result = CRS.from_user_input(crs)
    if not result.is_projected:
        raise ValueError(f"GTFS UrbanGraph CRS must be projected in metres, got {result.to_string()}")
    return result


def _shape_geometries(feed: GTFSFeed, shape_ids: set[str], crs: CRS) -> dict[str, tuple[LineString, Any]]:
    shapes = feed.get("shapes")
    if shapes is None or shapes.empty or not shape_ids:
        return {}
    shapes = shapes.loc[shapes["shape_id"].isin(shape_ids)].copy()
    if shapes.empty:
        return {}
    shapes["shape_pt_sequence"] = pd.to_numeric(shapes["shape_pt_sequence"], errors="raise")
    shapes["_lat"] = pd.to_numeric(shapes["shape_pt_lat"], errors="coerce")
    shapes["_lon"] = pd.to_numeric(shapes["shape_pt_lon"], errors="coerce")
    invalid = shapes[["_lat", "_lon"]].isna().any(axis=1)
    if invalid.any():
        raise GTFSValidationError(f"shapes.txt has {int(invalid.sum())} invalid coordinates")
    transformer = Transformer.from_crs(4326, crs, always_xy=True)
    shapes["_x"], shapes["_y"] = transformer.transform(shapes["_lon"].to_numpy(), shapes["_lat"].to_numpy())
    shapes["_feed_dist"] = (
        pd.to_numeric(shapes["shape_dist_traveled"], errors="coerce") if "shape_dist_traveled" in shapes else np.nan
    )

    result: dict[str, tuple[LineString, Any]] = {}
    for shape_id, group in shapes.groupby("shape_id", sort=False):
        group = group.sort_values("shape_pt_sequence", kind="stable")
        coords = list(zip(group["_x"], group["_y"]))
        if len(coords) < 2:
            continue
        line = LineString(coords)
        feed_dist = group["_feed_dist"].to_numpy(dtype=float)
        if np.isfinite(feed_dist).all() and np.all(np.diff(feed_dist) >= 0) and feed_dist[-1] > feed_dist[0]:
            xy = np.asarray(coords)
            cumulative = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))])

            def locate(value, source=feed_dist, target=cumulative):
                return float(np.interp(float(value), source, target))

        else:
            locate = None
        result[str(shape_id)] = (line, locate)
    return result


def _canonical_pattern_stops(trips: pd.DataFrame, stop_times: pd.DataFrame) -> pd.DataFrame:
    canonical = trips.groupby("pattern_id", sort=False)["trip_id"].first()
    rows = stop_times.loc[stop_times["trip_id"].isin(canonical)].copy()
    # Some producers add a non-standard shape_id to stop_times.txt. The trip-level
    # shapes reference is authoritative for pattern geometry and avoids merge suffixes.
    rows = rows.drop(columns=["shape_id"], errors="ignore")
    rows = rows.merge(
        trips[
            [
                "trip_id",
                "pattern_id",
                "route_id",
                "direction_id",
                "shape_id",
                "type",
                "route_short_name",
                "route_long_name",
            ]
        ],
        on=["trip_id", "pattern_id"],
        how="left",
        validate="many_to_one",
    )
    return rows.sort_values(["pattern_id", "stop_position"], kind="stable")


def _empty_gtfs_graph(crs: Any | None = None) -> UrbanGraph:
    return UrbanGraph.empty(
        crs=crs, is_multigraph=True, is_directed=True, edge_direction_column="oneway", graph_type="public_transport"
    )


def _build_graph_tables(
    feed: GTFSFeed,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    waits: pd.Series,
    segment_times: pd.Series,
    crs: Any | None,
    walk_speed_m_per_min: float,
) -> UrbanGraph:
    pattern_stops = _canonical_pattern_stops(trips, stop_times)
    served_ids = set(pattern_stops["stop_id"])
    pathways = feed.get("pathways")
    pathway_ids: set[str] = set()
    if pathways is not None and not pathways.empty:
        pathway_ids = set(pathways["from_stop_id"]) | set(pathways["to_stop_id"])
    needed_ids = served_ids | pathway_ids
    stops = _resolved_stop_coordinates(feed["stops"], needed_ids)
    local_crs = _target_crs(stops, crs)
    stop_points = gpd.GeoSeries(gpd.points_from_xy(stops["_lon"], stops["_lat"]), index=stops.index, crs=4326).to_crs(
        local_crs
    )

    stations_with_pathways: set[str] = set()
    if pathway_ids:
        all_stops = feed["stops"].copy().set_index("stop_id", drop=False)
        for column in ("location_type", "parent_station"):
            if column not in all_stops:
                all_stops[column] = ""
        for endpoint in pathway_ids:
            current = str(endpoint)
            for _ in range(4):
                if current not in all_stops.index:
                    break
                row = all_stops.loc[current]
                if str(row["location_type"]) == "1":
                    stations_with_pathways.add(current)
                    break
                parent = str(row["parent_station"])
                if not parent:
                    break
                current = parent

    def node_type(row) -> str:
        location_type = str(row["location_type"] or "0")
        parent = str(row["parent_station"])
        if row["stop_id"] in served_ids:
            if parent in stations_with_pathways and row["stop_access"] != "1":
                return "station_platform"
            return "platform"
        if location_type == "2":
            return "station_entry_exit"
        if location_type == "3":
            return "station_node"
        if location_type == "4":
            return "station_boarding_area"
        if location_type == "1":
            return "station"
        return "station_platform"

    node_records: list[dict[str, Any]] = []
    stop_to_node: dict[str, int] = {}
    for stop_id, row in stops.iterrows():
        node_id = len(node_records)
        stop_to_node[str(stop_id)] = node_id
        node_records.append(
            {
                "node_id": node_id,
                "type": node_type(row),
                "gtfs_stop_id": str(stop_id),
                "stop_name": row.get("stop_name", ""),
                "parent_station": row.get("parent_station", ""),
                "stop_access": row.get("stop_access", ""),
                "geometry": stop_points.loc[stop_id],
            }
        )

    shape_ids = set(pattern_stops.loc[pattern_stops["shape_id"].ne(""), "shape_id"])
    shapes = _shape_geometries(feed, shape_ids, local_crs)
    route_stop_node: dict[tuple[int, int], int] = {}
    route_stop_distance: dict[tuple[int, int], float | None] = {}

    for pattern_id, group in pattern_stops.groupby("pattern_id", sort=False):
        group = group.sort_values("stop_position", kind="stable")
        shape_id = str(group["shape_id"].iloc[0])
        shape_data = shapes.get(shape_id)
        previous_distance = 0.0
        for _, row in group.iterrows():
            key = (int(pattern_id), int(row["stop_position"]))
            platform_point = node_records[stop_to_node[str(row["stop_id"])]]["geometry"]
            distance = None
            route_point = platform_point
            if shape_data is not None:
                line, locate = shape_data
                feed_distance = pd.to_numeric(pd.Series([row["shape_dist_traveled"]]), errors="coerce").iloc[0]
                if locate is not None and pd.notna(feed_distance):
                    distance = locate(feed_distance)
                else:
                    distance = max(previous_distance, float(line.project(platform_point)))
                distance = min(max(float(distance), 0.0), line.length)
                previous_distance = distance
                route_point = line.interpolate(distance)
            node_id = len(node_records)
            route_stop_node[key] = node_id
            route_stop_distance[key] = distance
            node_records.append(
                {
                    "node_id": node_id,
                    "type": row["type"],
                    "gtfs_stop_id": str(row["stop_id"]),
                    "pattern_id": int(pattern_id),
                    "stop_position": int(row["stop_position"]),
                    "stop_sequence": int(row["stop_sequence"]),
                    "route_id": str(row["route_id"]),
                    "direction_id": str(row["direction_id"]),
                    "shape_id": shape_id,
                    "route": row.get("route_short_name", ""),
                    "geometry": route_point,
                }
            )

    edge_records: list[dict[str, Any]] = []

    def add_edge(u: int, v: int, geometry: LineString, edge_type: str, length: float, time_min: float, **attrs) -> None:
        record = {
            "u": u,
            "v": v,
            "geometry": geometry,
            "type": edge_type,
            "length_meter": float(length),
            "time_min": float(time_min),
            "oneway": True,
        }
        record.update(attrs)
        edge_records.append(record)

    for pattern_id, group in pattern_stops.groupby("pattern_id", sort=False):
        group = group.sort_values("stop_position", kind="stable")
        rows = list(group.to_dict("records"))
        for position, row in enumerate(rows):
            key = (int(pattern_id), int(row["stop_position"]))
            platform_node = stop_to_node[str(row["stop_id"])]
            route_node = route_stop_node[key]
            platform_point = node_records[platform_node]["geometry"]
            route_point = node_records[route_node]["geometry"]
            connector = LineString([platform_point, route_point])
            edge_attrs = {
                "route_id": str(row["route_id"]),
                "pattern_id": int(pattern_id),
                "stop_position": int(row["stop_position"]),
                "route": row.get("route_short_name", ""),
            }
            wait = waits.get(key, np.nan)
            if row["pickup_type"] != "1" and pd.notna(wait):
                add_edge(platform_node, route_node, connector, "boarding", 0.0, float(wait), **edge_attrs)
            if row["drop_off_type"] != "1":
                add_edge(
                    route_node,
                    platform_node,
                    LineString(list(connector.coords)[::-1]),
                    "alighting",
                    0.0,
                    0.0,
                    **edge_attrs,
                )

            if position == len(rows) - 1:
                continue
            next_row = rows[position + 1]
            next_key = (int(pattern_id), int(next_row["stop_position"]))
            next_node = route_stop_node[next_key]
            start_point = node_records[route_node]["geometry"]
            end_point = node_records[next_node]["geometry"]
            shape_data = shapes.get(str(row["shape_id"]))
            start_distance = route_stop_distance[key]
            end_distance = route_stop_distance[next_key]
            geometry = LineString([start_point, end_point])
            length_meter = sqrt(2) * geometry.length
            if (
                shape_data is not None
                and start_distance is not None
                and end_distance is not None
                and end_distance > start_distance
            ):
                candidate = substring(shape_data[0], start_distance, end_distance)
                if isinstance(candidate, LineString):
                    geometry = candidate
                    length_meter = geometry.length
            travel_time = segment_times.get(key, np.nan)
            if pd.isna(travel_time):
                raise GTFSValidationError(
                    f"Cannot calculate travel time for pattern={pattern_id}, stop_position={row['stop_position']}"
                )
            add_edge(
                route_node,
                next_node,
                geometry,
                str(row["type"]),
                length_meter,
                float(travel_time),
                route_id=str(row["route_id"]),
                pattern_id=int(pattern_id),
                route=row.get("route_short_name", ""),
            )

    if pathways is not None and not pathways.empty:
        for _, row in pathways.iterrows():
            from_node = stop_to_node[str(row["from_stop_id"])]
            to_node = stop_to_node[str(row["to_stop_id"])]
            from_point = node_records[from_node]["geometry"]
            to_point = node_records[to_node]["geometry"]
            geometry = LineString([from_point, to_point])
            explicit_length = pd.to_numeric(pd.Series([row.get("length", "")]), errors="coerce").iloc[0]
            length = float(explicit_length) if pd.notna(explicit_length) else geometry.length
            traversal = pd.to_numeric(pd.Series([row.get("traversal_time", "")]), errors="coerce").iloc[0]
            time_min = float(traversal) / 60 if pd.notna(traversal) else length / walk_speed_m_per_min
            add_edge(
                from_node,
                to_node,
                geometry,
                "pathway",
                length,
                time_min,
                pathway_id=str(row["pathway_id"]),
                pathway_mode=str(row["pathway_mode"]),
            )
            edge_records[-1]["oneway"] = str(row["is_bidirectional"]) != "1"

    # Materialise the parent_station hierarchy. Feeds that describe a station down to boarding
    # areas (location_type=4) route their pathways to those areas and never to the platforms
    # that stop_times references, so the platforms -- and with them the whole metro component --
    # end up disconnected from everything an intermodal join can project onto the walk layer.
    # Linking a child to its parent closes the chain (entrance -> pathway -> boarding area ->
    # station_link -> platform -> boarding) without projecting platforms directly, which would
    # bypass the concourse the pathways exist to describe.
    # These edges are free: the hierarchy states containment, and the feed gives no traversal
    # time for it. Only already existing nodes are linked; no node is invented.
    for stop_id, node_id in stop_to_node.items():
        parent = str(node_records[node_id].get("parent_station", "") or "")
        parent_node = stop_to_node.get(parent)
        if not parent or parent_node is None or parent_node == node_id:
            continue
        child_point = node_records[node_id]["geometry"]
        parent_point = node_records[parent_node]["geometry"]
        link = LineString([child_point, parent_point])
        add_edge(
            node_id,
            parent_node,
            link,
            "station_link",
            link.length,
            0.0,
            gtfs_stop_id=str(stop_id),
            parent_station=parent,
        )
        edge_records[-1]["oneway"] = False

    nodes = gpd.GeoDataFrame(node_records, geometry="geometry", crs=local_crs).set_index("node_id")
    if not edge_records:
        return UrbanGraph(
            nodes_gdf=nodes,
            edges_gdf=gpd.GeoDataFrame(
                columns=["u", "v", "k", "geometry", "length_meter", "time_min", "oneway"],
                geometry="geometry",
                crs=local_crs,
            ),
            is_multigraph=True,
            is_directed=True,
            edge_direction_column="oneway",
            crs=local_crs,
            graph_type="public_transport",
        )
    edges = gpd.GeoDataFrame(edge_records, geometry="geometry", crs=local_crs)
    edges["k"] = edges.groupby(["u", "v"], sort=False).cumcount()
    ordered = ["u", "v", "k"] + [column for column in edges.columns if column not in {"u", "v", "k"}]
    edges = edges[ordered]
    return UrbanGraph(
        nodes_gdf=nodes,
        edges_gdf=edges,
        is_multigraph=True,
        is_directed=True,
        edge_direction_column="oneway",
        crs=local_crs,
        graph_type="public_transport",
    )


def _resolve_feed(feed: Any) -> GTFSFeed:
    """Accept one feed or several, and return the tables to build from."""
    if isinstance(feed, GTFSFeed):
        return feed
    if isinstance(feed, (str, Path)):
        return read_gtfs_feed(feed)
    sources = list(feed)
    if not sources:
        raise ValueError("no GTFS sources given")
    if len(sources) == 1:
        return _resolve_feed(sources[0])
    return merge_gtfs_feeds(sources)


def get_gtfs_public_transport_graph(
    feed: str | Path | GTFSFeed | Sequence[str | Path | GTFSFeed],
    *,
    service_date: date | datetime | str | None = None,
    start_time: time | str | int | float | None = None,
    end_time: time | str | int | float | None = None,
    crs: Any | None = None,
    single_departure_wait_min: float | None = None,
    walk_speed_m_per_min: float = DEFAULT_WALK_SPEED_M_PER_MIN,
) -> UrbanGraph:
    """Build a static public-transport ``UrbanGraph`` from a local GTFS feed.

    Boarding weights use the first-iteration half-headway heuristic. Scheduled
    services use half the mean gap between consecutive departures of the same
    route pattern at the same stop. ``frequencies.txt`` uses
    ``headway_secs / 120``.

    ``None`` date/time values are intentionally unbounded: with all three
    filters omitted, every trip in the feed participates. This is an aggregate
    all-service graph, not the timetable for a particular real-world day.

    Args:
        feed: Directory containing GTFS text files, a GTFS ZIP archive, an
            already-read :class:`GTFSFeed`, or a sequence of any of those. A
            sequence is merged with :func:`iduedu.merge_gtfs_feeds` first, which
            qualifies identifiers by source; pass the merged feed yourself when
            you need to control prefixes or fuse stops across feeds.
        service_date: Optional service date used with ``calendar.txt`` and
            ``calendar_dates.txt``. Accepts ``date``, ``YYYY-MM-DD`` or
            ``YYYYMMDD``.
        start_time: Inclusive departure-time bound. ``None`` means unbounded.
        end_time: Exclusive departure-time bound. ``None`` means unbounded.
        crs: Projected target CRS. If omitted, a local UTM CRS is estimated.
        single_departure_wait_min: Explicit wait fallback for a pattern-stop
            with one scheduled departure. If ``None``, its boarding edge is
            omitted.
        walk_speed_m_per_min: Fallback speed for pathways without an explicit
            traversal time.
    """

    if single_departure_wait_min is not None and single_departure_wait_min < 0:
        raise ValueError("single_departure_wait_min must be non-negative or None")
    if walk_speed_m_per_min <= 0:
        raise ValueError("walk_speed_m_per_min must be positive")

    parsed_date = _parse_service_date(service_date)
    parsed_start = _parse_time_bound(start_time, "start_time")
    parsed_end = _parse_time_bound(end_time, "end_time")
    source = _resolve_feed(feed)
    validate_gtfs_feed(source)
    trips = _prepare_trips(source, parsed_date)
    if trips.empty:
        logger.warning("No GTFS trips match the requested service date")
        return _empty_gtfs_graph(crs)
    stop_times = _prepare_stop_times(source, trips)
    if stop_times.empty:
        logger.warning("No GTFS stop times match the requested service filters")
        return _empty_gtfs_graph(crs)
    trips, stop_times = _assign_patterns(trips, stop_times)
    segment_times = _segment_times(stop_times)
    waits = _boarding_wait_times(source, trips, stop_times, parsed_start, parsed_end, single_departure_wait_min)
    return _build_graph_tables(source, trips, stop_times, waits, segment_times, crs, walk_speed_m_per_min)
