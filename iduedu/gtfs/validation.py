from collections.abc import Iterable

import pandas as pd

from iduedu import config
from iduedu.gtfs.reader import REQUIRED_GTFS_TABLES, GTFSFeed

logger = config.logger


class GTFSValidationError(ValueError):
    """Raised when a GTFS feed cannot be converted into an UrbanGraph."""


REQUIRED_COLUMNS = {
    "agency": {"agency_name", "agency_url", "agency_timezone"},
    "stops": {"stop_id", "stop_lat", "stop_lon"},
    "routes": {"route_id", "route_type"},
    "trips": {"route_id", "service_id", "trip_id"},
    "stop_times": {"trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"},
    "calendar": {
        "service_id",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "start_date",
        "end_date",
    },
    "calendar_dates": {"service_id", "date", "exception_type"},
    "frequencies": {"trip_id", "start_time", "end_time", "headway_secs"},
    "shapes": {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"},
    "pathways": {"pathway_id", "from_stop_id", "to_stop_id", "pathway_mode", "is_bidirectional"},
}


def _require_unique(frame: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    columns = list(columns)
    if frame.duplicated(columns).any():
        raise GTFSValidationError(f"{table}.txt has duplicate primary key {columns}")


def _require_references(values: pd.Series, targets: pd.Series, field: str) -> None:
    missing = pd.Index(values[values.ne("")].unique()).difference(pd.Index(targets.unique()))
    if len(missing) > 0:
        raise GTFSValidationError(f"{field} references missing ids: {missing[:10].tolist()}")


def validate_gtfs_feed(feed: GTFSFeed) -> None:
    """Validate the structural GTFS contracts required by the graph builder."""

    missing_tables = set(REQUIRED_GTFS_TABLES) - set(feed.tables)
    if missing_tables:
        raise GTFSValidationError(f"GTFS feed is missing required tables: {sorted(missing_tables)}")
    if "calendar" not in feed and "calendar_dates" not in feed:
        raise GTFSValidationError("GTFS feed must contain calendar.txt, calendar_dates.txt, or both")

    for table, frame in feed.tables.items():
        required = REQUIRED_COLUMNS.get(table, set())
        missing = required - set(frame.columns)
        if missing:
            raise GTFSValidationError(f"{table}.txt is missing required columns: {sorted(missing)}")

    _require_unique(feed["stops"], ["stop_id"], "stops")
    _require_unique(feed["routes"], ["route_id"], "routes")
    _require_unique(feed["trips"], ["trip_id"], "trips")
    _require_unique(feed["stop_times"], ["trip_id", "stop_sequence"], "stop_times")

    _require_references(feed["trips"]["route_id"], feed["routes"]["route_id"], "trips.route_id")
    service_ids = []
    if "calendar" in feed:
        service_ids.append(feed["calendar"]["service_id"])
    if "calendar_dates" in feed:
        service_ids.append(feed["calendar_dates"]["service_id"])
    _require_references(
        feed["trips"]["service_id"],
        pd.concat(service_ids, ignore_index=True),
        "trips.service_id",
    )
    _require_references(feed["stop_times"]["trip_id"], feed["trips"]["trip_id"], "stop_times.trip_id")
    _require_references(feed["stop_times"]["stop_id"], feed["stops"]["stop_id"], "stop_times.stop_id")

    if "frequencies" in feed:
        _require_unique(feed["frequencies"], ["trip_id", "start_time"], "frequencies")
        _require_references(feed["frequencies"]["trip_id"], feed["trips"]["trip_id"], "frequencies.trip_id")
    if "shapes" in feed:
        _require_unique(feed["shapes"], ["shape_id", "shape_pt_sequence"], "shapes")
    if "calendar" in feed:
        _require_unique(feed["calendar"], ["service_id"], "calendar")
    if "calendar_dates" in feed:
        _require_unique(feed["calendar_dates"], ["service_id", "date"], "calendar_dates")
    if "pathways" in feed:
        _require_unique(feed["pathways"], ["pathway_id"], "pathways")
        _require_references(feed["pathways"]["from_stop_id"], feed["stops"]["stop_id"], "pathways.from_stop_id")
        _require_references(feed["pathways"]["to_stop_id"], feed["stops"]["stop_id"], "pathways.to_stop_id")

    if "calendar" in feed:
        calendar = feed["calendar"]
        invalid = calendar["start_date"].gt(calendar["end_date"])
        if invalid.any():
            logger.warning(
                f"Ignoring {int(invalid.sum())} calendar.txt rows whose start_date is after end_date; "
                "calendar_dates.txt exceptions can still activate services"
            )
