"""Profile every downloaded GTFS archive: where it is, how big it is, what it contains.

The catalogs cannot answer any of these questions reliably -- a third of the
entries name no city and 1376 of them name no country -- but the archive always
can, because a feed without stop coordinates is not a feed. Each archive is
therefore described from its own contents:

* **Where.** The median of stop latitudes and longitudes. The median rather than
  the mean, because one forgotten depot or a single intercity terminal drags an
  average away from the network it is supposed to locate.
* **How large.** The distance from that centre containing 90% of the stops. This
  is what separates a city network from a regional or national dump, and it is
  the check that was missing when regional feeds inflated waiting times in the
  previous run.
* **What is inside.** Stops, routes by transport type, and trips -- the numbers
  that expose a stub: an archive that is a valid zip with a ``stops.txt`` and
  almost nothing else. Vancouver arrived as 9.7 KB.
* **When it is valid.** The service window, so the gap between the feed and today
  can be carried as a covariate instead of silently comparing a 2020 timetable
  with a 2026 map.

Written to ``results/wide_tier/feed_profile.csv``, one row per archive.
"""

import argparse
import logging
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from wide_feeds import FEED_CACHE
from wide_paths import DOWNLOADS_CSV, PAPER_DIR, PROFILE_CSV, WIDE_DIR

from iduedu.graph_builders.gtfs_builders import _transport_type

logger = logging.getLogger(__name__)


EARTH_RADIUS_KM = 6371.0088


def _read_member(archive: zipfile.ZipFile, name: str) -> pd.DataFrame | None:
    """Read a GTFS table by name, tolerating folder nesting and case."""
    wanted = name.lower()
    for member in archive.namelist():
        if Path(member).name.lower() != wanted:
            continue
        try:
            with archive.open(member) as handle:
                return pd.read_csv(handle, dtype=str, encoding="utf-8-sig", low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            logger.debug("%s: cannot read %s (%s)", archive.filename, member, type(exc).__name__)
            return None
    return None


def _spread_km(latitudes: np.ndarray, longitudes: np.ndarray, centre_lat: float, centre_lon: float) -> float:
    """Distance from the centre containing 90% of stops, in kilometres."""
    lat1, lon1 = math.radians(centre_lat), math.radians(centre_lon)
    lat2, lon2 = np.radians(latitudes), np.radians(longitudes)
    haversine = np.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    distances = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    return float(np.percentile(distances, 90))


def _service_window(archive: zipfile.ZipFile) -> tuple[str, str]:
    """Earliest and latest date the feed claims to describe."""
    dates: list[str] = []
    calendar = _read_member(archive, "calendar.txt")
    if calendar is not None:
        for column in ("start_date", "end_date"):
            if column in calendar:
                dates += [value for value in calendar[column].dropna().astype(str) if value.isdigit()]
    exceptions = _read_member(archive, "calendar_dates.txt")
    if exceptions is not None and "date" in exceptions:
        dates += [value for value in exceptions["date"].dropna().astype(str) if value.isdigit()]
    if not dates:
        return "", ""
    return min(dates), max(dates)


def profile(path: Path) -> dict:
    """Describe one archive. ``ok`` is false when the archive cannot be profiled."""
    row: dict = {"feed_id": path.stem, "ok": False, "problem": ""}
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        row["problem"] = "corrupt zip archive"
        return row

    with archive:
        stops = _read_member(archive, "stops.txt")
        if stops is None or "stop_lat" not in stops or "stop_lon" not in stops:
            row["problem"] = "stops.txt missing or without coordinates"
            return row

        latitudes = pd.to_numeric(stops["stop_lat"], errors="coerce")
        longitudes = pd.to_numeric(stops["stop_lon"], errors="coerce")
        usable = latitudes.between(-90, 90) & longitudes.between(-180, 180) & latitudes.notna() & longitudes.notna()
        # A stop at exactly (0, 0) is a missing coordinate, not a stop in the Gulf of Guinea.
        usable &= ~(latitudes.eq(0) & longitudes.eq(0))
        if not usable.any():
            row["problem"] = "no usable stop coordinates"
            return row

        lat_values = latitudes[usable].to_numpy()
        lon_values = longitudes[usable].to_numpy()
        centre_lat = float(np.median(lat_values))
        centre_lon = float(np.median(lon_values))

        routes = _read_member(archive, "routes.txt")
        trips = _read_member(archive, "trips.txt")
        modes: dict[str, int] = {}
        if routes is not None and "route_type" in routes:
            for route_type in routes["route_type"].fillna(""):
                mode = _transport_type(str(route_type))
                modes[mode] = modes.get(mode, 0) + 1

        first_date, last_date = _service_window(archive)

        row.update(
            {
                "ok": True,
                "n_stops": int(usable.sum()),
                "n_stops_unusable": int((~usable).sum()),
                "n_routes": 0 if routes is None else len(routes),
                "n_trips": 0 if trips is None else len(trips),
                "centre_lat": round(centre_lat, 6),
                "centre_lon": round(centre_lon, 6),
                "spread_km": round(_spread_km(lat_values, lon_values, centre_lat, centre_lon), 2),
                "bbox_km": round(
                    _spread_km(
                        np.array([lat_values.min(), lat_values.max()]),
                        np.array([lon_values.min(), lon_values.max()]),
                        centre_lat,
                        centre_lon,
                    ),
                    2,
                ),
                "modes": ";".join(f"{mode}:{count}" for mode, count in sorted(modes.items())),
                "service_from": first_date,
                "service_to": last_date,
                "size_bytes": path.stat().st_size,
            }
        )
    return row


def run() -> pd.DataFrame:
    log = pd.read_csv(DOWNLOADS_CSV).drop_duplicates(subset=["feed_id"], keep="last")
    stored = log[log["status"] == "ok"]["feed_id"].astype(str).tolist()

    rows = []
    for index, feed_id in enumerate(stored, start=1):
        path = FEED_CACHE / f"{feed_id}.zip"
        if not path.exists():
            continue
        row = profile(path)
        rows.append(row)
        if index % 20 == 0 or not row["ok"]:
            logger.info("[%d/%d] %s %s", index, len(stored), feed_id, row.get("problem") or "ok")

    frame = pd.DataFrame(rows)
    WIDE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROFILE_CSV, index=False, encoding="utf-8")
    return frame


def _report(frame: pd.DataFrame) -> None:
    good = frame[frame["ok"]]
    print(f"profiled {len(good)} archives, {len(frame) - len(good)} unreadable\n")

    print("size of network (stops):")
    print(good["n_stops"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(0).to_string())

    print("\nspatial spread, km (90% of stops within):")
    print(good["spread_km"].describe(percentiles=[0.1, 0.5, 0.9]).round(1).to_string())

    print("\nlikely stubs — fewer than 50 stops or no trips:")
    stubs = good[(good["n_stops"] < 50) | (good["n_trips"] == 0)]
    print(
        stubs[["feed_id", "n_stops", "n_routes", "n_trips", "spread_km", "size_bytes"]].to_string(index=False)
        if not stubs.empty
        else "  none"
    )

    print("\nlikely regional or national rather than city (90% spread over 150 km):")
    wide = good[good["spread_km"] > 150].sort_values("spread_km", ascending=False)
    print(
        wide[["feed_id", "n_stops", "spread_km", "modes"]].head(20).to_string(index=False)
        if not wide.empty
        else "  none"
    )

    expired = good[(good["service_to"] != "") & (good["service_to"].astype(str) < "20260810")]
    print(f"\nfeeds whose service window ended before today: {len(expired)}")
    if not expired.empty:
        oldest = expired.sort_values("service_to").head(10)
        print(oldest[["feed_id", "service_from", "service_to", "n_stops"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true", help="read the existing profile instead of rebuilding")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if arguments.report_only and PROFILE_CSV.exists():
        frame = pd.read_csv(PROFILE_CSV)
    else:
        frame = run()
    _report(frame)
    print(f"\nwritten: {PROFILE_CSV.relative_to(PAPER_DIR)}")


if __name__ == "__main__":
    main()
