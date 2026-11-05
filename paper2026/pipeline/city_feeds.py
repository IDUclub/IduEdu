"""One GTFS archive per city: the city's feeds merged, clipped to its boundary, repaired.

* **Merge.** Every feed serving the city (``inputs/city_members.csv``) is read with its
  identifiers prefixed by the feed id, so ``route_id = 1`` of two feeds stays two
  routes; route types a source codes by a scheme of its own are recoded
  (``inputs/route_type_fixes.csv``).
* **Clip.** A segment is kept when both its stops lie inside the boundary, as
  ``clip_by_territory`` keeps OSM edges; a trip that leaves the city and comes back
  splits into one trip per run of in-boundary stops. Shapes are carried untouched,
  like OSM edge geometry.
* **Deduplicate.** What two sources publish twice is counted once
  (``iduedu.deduplicate_across_feeds``: platforms of different sources within 0.5 m
  fuse, identical trips of another source drop).
* **Repair.** The archive is made to satisfy GTFS keys and references; every repair
  is counted in ``city_feeds.csv``, because a feed that breaks its own specification
  is a result.

    python pipeline/city_feeds.py [--fetch] [--cities ...] [--force]
"""

import argparse
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from common import (
    CITY_FEEDS_CSV,
    FEED_CACHE_DIR,
    FEEDS_CSV,
    ROUTE_TYPE_FIXES_CSV,
    USER_AGENT,
    boundary,
    cities,
    city_feed_path,
    members,
    restrict,
    setup,
    sweep,
)
from shapely.geometry import Point

from iduedu import deduplicate_across_feeds

#: Tables carried into the city archive (fare tables are dropped).
TABLES = [
    "agency.txt", "stops.txt", "routes.txt", "trips.txt", "stop_times.txt", "calendar.txt",
    "calendar_dates.txt", "frequencies.txt", "pathways.txt", "transfers.txt", "shapes.txt",
]  # fmt: skip

#: Identifier columns prefixed with the feed id before merging.
ID_COLUMNS = {
    "stops.txt": ["stop_id", "parent_station"],
    "routes.txt": ["route_id", "agency_id"],
    "trips.txt": ["route_id", "service_id", "trip_id", "shape_id"],
    "stop_times.txt": ["trip_id", "stop_id"],
    "calendar.txt": ["service_id"],
    "calendar_dates.txt": ["service_id"],
    "agency.txt": ["agency_id"],
    "frequencies.txt": ["trip_id"],
    "pathways.txt": ["pathway_id", "from_stop_id", "to_stop_id"],
    "transfers.txt": ["from_stop_id", "to_stop_id"],
    "shapes.txt": ["shape_id"],
}

#: Uniqueness ``validate_gtfs_feed`` enforces.
PRIMARY_KEYS = {
    "stops.txt": ["stop_id"],
    "routes.txt": ["route_id"],
    "trips.txt": ["trip_id"],
    "stop_times.txt": ["trip_id", "stop_sequence"],
    "frequencies.txt": ["trip_id", "start_time"],
    "shapes.txt": ["shape_id", "shape_pt_sequence"],
    "calendar.txt": ["service_id"],
    "calendar_dates.txt": ["service_id", "date"],
    "pathways.txt": ["pathway_id"],
}

#: References ``validate_gtfs_feed`` enforces, ordered so resolving one cannot orphan the next.
REFERENCES = [
    ("trips.txt", "route_id", ("routes.txt",), "route_id"),
    ("trips.txt", "service_id", ("calendar.txt", "calendar_dates.txt"), "service_id"),
    ("stop_times.txt", "trip_id", ("trips.txt",), "trip_id"),
    ("stop_times.txt", "stop_id", ("stops.txt",), "stop_id"),
    ("frequencies.txt", "trip_id", ("trips.txt",), "trip_id"),
    ("pathways.txt", "from_stop_id", ("stops.txt",), "stop_id"),
    ("pathways.txt", "to_stop_id", ("stops.txt",), "stop_id"),
]

#: Suffix of a trip split at a boundary crossing ("#" occurs in real trip ids).
SPLIT_MARK = "~run"


# --- reading -------------------------------------------------------------------


def read_member(archive: zipfile.ZipFile, name: str) -> pd.DataFrame | None:
    """A GTFS table by name, tolerating folder nesting and case; ``None`` if unreadable."""
    for member in archive.namelist():
        if Path(member).name.lower() != name.lower():
            continue
        try:
            with archive.open(member) as handle:
                return pd.read_csv(handle, dtype=str, encoding="utf-8-sig", low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return None
    return None


def _prefixed(frame: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            values = out[column].fillna("").astype(str)
            out[column] = values.where(values.eq(""), prefix + ":" + values)
    return out


def recode_route_types(tables: dict[str, pd.DataFrame], feed_id: str, fixes: pd.DataFrame) -> None:
    """Apply ``inputs/route_type_fixes.csv``; a route recoded to nothing goes with its trips."""
    rules = fixes[fixes["feed_id"] == feed_id]
    routes = tables.get("routes.txt")
    if rules.empty or routes is None:
        return
    mapping = dict(zip(rules["published"], rules["gtfs"]))
    published = routes["route_type"].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    recoded = published.map(lambda code: mapping.get(code, code))
    dropped = set(routes.loc[recoded.eq(""), "route_id"])
    tables["routes.txt"] = routes.assign(route_type=recoded)[~routes["route_id"].isin(dropped)]
    if dropped and "trips.txt" in tables:
        trips = tables["trips.txt"]
        gone = set(trips.loc[trips["route_id"].isin(dropped), "trip_id"])
        tables["trips.txt"] = trips[~trips["trip_id"].isin(gone)]
        for name in ("stop_times.txt", "frequencies.txt"):
            if name in tables:
                tables[name] = tables[name][~tables[name]["trip_id"].isin(gone)]


def read_feed(feed_id: str, fixes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    with zipfile.ZipFile(FEED_CACHE_DIR / f"{feed_id}.zip") as archive:
        for name in TABLES:
            frame = read_member(archive, name)
            if frame is not None and not frame.empty:
                tables[name] = _prefixed(frame, ID_COLUMNS.get(name, []), feed_id)
    recode_route_types(tables, feed_id, fixes)
    return tables


# --- clipping ------------------------------------------------------------------


def _clip_stop_times(times: pd.DataFrame, kept_ids: set) -> tuple[pd.DataFrame, int, int]:
    """Keep the runs of consecutive in-boundary stops; a trip with several runs splits."""
    order = times.columns.tolist()
    frame = times.copy()
    frame["_seq"] = pd.to_numeric(frame.get("stop_sequence"), errors="coerce")
    frame = frame.sort_values(["trip_id", "_seq"], kind="stable")
    frame["_inside"] = frame["stop_id"].isin(kept_ids)
    new_trip = frame["trip_id"] != frame["trip_id"].shift()
    broke = (~frame["_inside"]) | (~frame["_inside"].shift(fill_value=False))
    frame["_run"] = (new_trip | broke).cumsum()

    segments_before = int((frame["trip_id"] == frame["trip_id"].shift(-1)).sum())
    inside = frame[frame["_inside"]].copy()
    inside = inside[inside.groupby("_run")["stop_id"].transform("size") >= 2]
    segments_after = int((inside["_run"] == inside["_run"].shift(-1)).sum())

    runs_per_trip = inside.groupby("trip_id")["_run"].transform("nunique")
    run_index = inside.groupby("trip_id")["_run"].rank(method="dense").astype(int)
    inside["trip_id"] = inside["trip_id"].where(
        runs_per_trip == 1, inside["trip_id"] + SPLIT_MARK + run_index.astype(str)
    )
    return inside[order].reset_index(drop=True), segments_before, segments_after


def _expand_split_trips(frame: pd.DataFrame, kept_trips: set) -> pd.DataFrame:
    """One row per run for a trip that was split."""
    mapping: dict[str, list[str]] = {}
    for trip_id in map(str, kept_trips):
        mapping.setdefault(trip_id.rsplit(SPLIT_MARK, 1)[0] if SPLIT_MARK in trip_id else trip_id, []).append(trip_id)
    base = frame.assign(_new=frame["trip_id"].astype(str).map(mapping))
    base = base[base["_new"].notna()].explode("_new")
    return base.assign(trip_id=base["_new"]).drop(columns=["_new"]).reset_index(drop=True)


def clip_to_boundary(tables: dict[str, pd.DataFrame], polygon) -> tuple[dict[str, pd.DataFrame], dict]:
    stops = tables.get("stops.txt")
    stats = {"stops_before": 0 if stops is None else len(stops), "stops_after": 0}
    if stops is None:
        return tables, stats
    latitudes = pd.to_numeric(stops["stop_lat"], errors="coerce")
    longitudes = pd.to_numeric(stops["stop_lon"], errors="coerce")
    points = gpd.GeoSeries([Point(x, y) for x, y in zip(longitudes, latitudes)], crs=4326)
    inside = points.within(polygon).to_numpy() & latitudes.notna().to_numpy()
    kept = stops.loc[inside].copy()
    stats["stops_after"] = len(kept)
    if kept.empty:
        return {}, stats

    kept_ids = set(kept["stop_id"])
    if "parent_station" in stops.columns:  # a kept platform keeps its station, wherever it is
        parents = set(stops.loc[inside, "parent_station"].dropna()) - {""}
        extra = stops[stops["stop_id"].isin(parents) & ~inside]
        if not extra.empty:
            kept = pd.concat([kept, extra], ignore_index=True)
            kept_ids |= set(extra["stop_id"])
    tables["stops.txt"] = kept

    kept_trips: set = set()
    times = tables.get("stop_times.txt")
    if times is not None:
        times, stats["segments_before"], stats["segments_after"] = _clip_stop_times(times, kept_ids)
        tables["stop_times.txt"] = times
        kept_trips = set(times["trip_id"])
        for name in ("trips.txt", "frequencies.txt"):
            if name in tables and "trip_id" in tables[name].columns:
                tables[name] = _expand_split_trips(tables[name], kept_trips)

    trips = tables.get("trips.txt")
    if trips is not None:
        trips = tables["trips.txt"] = trips[trips["trip_id"].isin(kept_trips)]
        routes, services = set(trips.get("route_id", [])), set(trips.get("service_id", []))
        for name, column, keep in (
            ("routes.txt", "route_id", routes),
            ("calendar.txt", "service_id", services),
            ("calendar_dates.txt", "service_id", services),
            ("frequencies.txt", "trip_id", kept_trips),
        ):
            if name in tables and column in tables[name].columns:
                tables[name] = tables[name][tables[name][column].isin(keep)]
    for name in ("pathways.txt", "transfers.txt"):  # both ends must have survived
        frame = tables.get(name)
        if frame is not None and {"from_stop_id", "to_stop_id"} <= set(frame.columns):
            tables[name] = frame[frame["from_stop_id"].isin(kept_ids) & frame["to_stop_id"].isin(kept_ids)]
    shapes = tables.get("shapes.txt")
    if shapes is not None and trips is not None and "shape_id" in trips.columns:
        tables["shapes.txt"] = shapes[shapes["shape_id"].isin(set(trips["shape_id"]))]
    return tables, stats


# --- merging and repairs -------------------------------------------------------


def merge(parts: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    merged = {}
    for name in TABLES:
        frames = [part[name] for part in parts if name in part and not part[name].empty]
        if frames:
            merged[name] = pd.concat(frames, ignore_index=True, sort=False)
    return merged


def deduplicate(tables: dict[str, pd.DataFrame], prefixes: list[str]) -> dict[str, int]:
    bare = {name[: -len(".txt")]: frame.fillna("") for name, frame in tables.items()}
    counts = deduplicate_across_feeds(bare, prefixes)
    for name, frame in bare.items():
        tables[f"{name}.txt"] = frame
    return {
        "platforms_fused_across_feeds": counts["platforms_fused"],
        "trips_dropped_as_copies": counts["trips_dropped"],
    }


def _seconds(values: pd.Series) -> pd.Series:
    parts = values.astype(str).str.strip().str.split(":", expand=True)
    if parts.shape[1] < 3:
        return pd.Series(float("nan"), index=values.index)
    numeric = parts.iloc[:, :3].apply(pd.to_numeric, errors="coerce")
    return numeric.iloc[:, 0] * 3600 + numeric.iloc[:, 1] * 60 + numeric.iloc[:, 2]


def _interpolate_inside(values: pd.Series, groups: pd.Series) -> pd.Series:
    """``interpolate(limit_area="inside")`` within each group, vectorised."""
    position = values.groupby(groups, sort=False).cumcount().astype(float)
    known_position = position.where(values.notna())
    by = groups.to_numpy()
    previous_value, next_value = values.groupby(by, sort=False).ffill(), values.groupby(by, sort=False).bfill()
    previous_position = known_position.groupby(by, sort=False).ffill()
    span = known_position.groupby(by, sort=False).bfill() - previous_position
    filled = previous_value + (next_value - previous_value) * (position - previous_position) / span.where(span > 0)
    return values.where(values.notna(), filled)


def _drop_unusable_frequencies(tables: dict[str, pd.DataFrame]) -> int:
    """Rows with an empty window or a zero headway describe no frequency."""
    frame = tables.get("frequencies.txt")
    if frame is None or frame.empty or not {"start_time", "end_time", "headway_secs"} <= set(frame.columns):
        return 0
    start, end = _seconds(frame["start_time"]), _seconds(frame["end_time"])
    headway = pd.to_numeric(frame["headway_secs"], errors="coerce")
    unusable = start.isna() | end.isna() | headway.isna() | headway.le(0) | end.le(start)
    tables["frequencies.txt"] = frame.loc[~unusable]
    return int(unusable.sum())


def _drop_untimeable_trips(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Trips whose times run backwards, or stay unknown after interpolating between timepoints."""
    stop_times = tables.get("stop_times.txt")
    if stop_times is None or stop_times.empty or "stop_sequence" not in stop_times:
        return {"trips_dropped_bad_times": 0, "trips_kept": 0}
    frame = stop_times.copy()
    frame["_sequence"] = pd.to_numeric(frame["stop_sequence"], errors="coerce")
    for column, source in (("_arrival", "arrival_time"), ("_departure", "departure_time")):
        frame[column] = _seconds(frame[source]) if source in frame.columns else float("nan")
    frame["_arrival"] = frame["_arrival"].fillna(frame["_departure"])
    frame["_departure"] = frame["_departure"].fillna(frame["_arrival"])
    frame = frame.sort_values(["trip_id", "_sequence"], kind="stable")
    for column in ("_arrival", "_departure"):
        frame[column] = _interpolate_inside(frame[column], frame["trip_id"])
    unknown = frame["_arrival"].isna() | frame["_departure"].isna()
    backwards = frame.groupby("trip_id", sort=False)["_arrival"].shift(-1).lt(frame["_departure"])
    bad = set(frame.loc[unknown | backwards, "trip_id"])
    if bad:
        for name in ("stop_times.txt", "trips.txt", "frequencies.txt"):
            if name in tables and "trip_id" in tables[name].columns:
                tables[name] = tables[name].loc[~tables[name]["trip_id"].isin(bad)]
    return {
        "trips_dropped_bad_times": len(bad),
        "trips_kept": int(frame.loc[~frame["trip_id"].isin(bad), "trip_id"].nunique()),
    }


def _drop_orphan_references(tables: dict[str, pd.DataFrame]) -> int:
    """Rows pointing at identifiers no table defines (Moscow: 7 416 undeclared trips)."""
    dropped = 0
    for table, column, targets, target_column in REFERENCES:
        frame = tables.get(table)
        if frame is None or frame.empty or column not in frame.columns:
            continue
        known = set()
        for target in targets:
            if target in tables and target_column in tables[target].columns:
                known |= set(tables[target][target_column].dropna().map(str))
        if not known:
            continue
        values = frame[column].fillna("").map(str)
        orphan = ~values.isin(known) & values.ne("")
        dropped += int(orphan.sum())
        tables[table] = frame.loc[~orphan]
    return dropped


def repair(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Make the archive satisfy GTFS keys and references, counting every repair.

    A repeated row is identical in every column and costs nothing; a colliding row
    reuses an identifier for different content (Addis Ababa: one ``stop_id`` for stops
    2 km apart) and the first is kept.
    """
    counts = {"repeated_rows_dropped": 0, "colliding_rows_dropped": 0}
    for name, keys in PRIMARY_KEYS.items():
        frame = tables.get(name)
        if frame is None or frame.empty or not set(keys) <= set(frame.columns):
            continue
        before = len(frame)
        frame = frame.drop_duplicates()
        counts["repeated_rows_dropped"] += before - len(frame)
        before = len(frame)
        tables[name] = frame.drop_duplicates(subset=keys, keep="first")
        counts["colliding_rows_dropped"] += before - len(tables[name])
    counts["unusable_frequencies_dropped"] = _drop_unusable_frequencies(tables)
    counts.update(_drop_untimeable_trips(tables))
    counts["orphan_rows_dropped"] = _drop_orphan_references(tables)
    return counts


def write_archive(tables: dict[str, pd.DataFrame], destination: Path) -> None:
    """Write through a ``.part`` file; the rename is retried (Windows scanners hold new files)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".zip.part")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in tables.items():
            archive.writestr(name, frame.to_csv(index=False, encoding="utf-8"))
    for attempt in range(6):
        try:
            partial.replace(destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.4 * (attempt + 1))


# --- one city ------------------------------------------------------------------


def build_city(city_key: str, fixes: pd.DataFrame) -> dict:
    city = cities().loc[city_key]
    feeds = members().loc[members()["city_key"] == city_key, "feed_id"].astype(str).tolist()
    row = {
        "city_key": city_key,
        "city_name": city["city_name"],
        "osm_id": city["osm_id"],
        "n_feeds_merged": len(feeds),
        "feeds": "+".join(feeds),
        "carved_from_regional": city["carved_from_regional"],
        "status": "failed",
    }
    polygon = boundary(city_key)
    parts, totals = [], {"stops_before": 0, "stops_after": 0, "segments_before": 0, "segments_after": 0}
    missing = []
    for feed_id in feeds:
        try:
            tables = read_feed(feed_id, fixes)
        except (zipfile.BadZipFile, FileNotFoundError):  # an archive not obtained is noted, the city goes on
            missing.append(feed_id)
            continue
        clipped, stats = clip_to_boundary(tables, polygon)
        for key in totals:
            totals[key] += stats.get(key, 0)
        if clipped.get("stops.txt") is not None and not clipped["stops.txt"].empty:
            parts.append(clipped)
    row.update(totals, feeds_missing="+".join(missing))
    row["stop_share_kept"] = round(totals["stops_after"] / totals["stops_before"], 4) if totals["stops_before"] else 0.0
    row["segment_share_kept"] = (
        round(totals["segments_after"] / totals["segments_before"], 4) if totals["segments_before"] else 0.0
    )
    if not parts:
        row["reason"] = "no stop survived the boundary"
        return row
    merged = merge(parts)
    row.update(deduplicate(merged, feeds))
    row.update(repair(merged))
    write_archive(merged, city_feed_path(city_key))
    row.update(
        status="ok",
        n_stops=len(merged.get("stops.txt", [])),
        n_routes=len(merged.get("routes.txt", [])),
        n_trips=len(merged.get("trips.txt", [])),
    )
    return row


def fetch_missing(logger, keys: list[str]) -> None:
    """Download the member feeds not in ``feed_cache/`` and check them against ``inputs/feeds.csv``.

    A source may have moved on since October 2026: an archive whose sha256 differs is
    kept as ``<feed_id>.zip.changed`` and reported, never used in its place.
    """
    import hashlib

    import requests

    feeds = pd.read_csv(FEEDS_CSV, keep_default_na=False)
    feeds = feeds[feeds["feed_id"].isin(members().loc[members()["city_key"].isin(keys), "feed_id"])]
    missing = feeds[~feeds["feed_id"].map(lambda feed_id: (FEED_CACHE_DIR / f"{feed_id}.zip").exists())]
    logger.info("%d of %d feeds missing from %s", len(missing), len(feeds), FEED_CACHE_DIR)
    FEED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for row in missing.itertuples():
        try:
            response = requests.get(row.url, headers={"User-Agent": USER_AGENT}, timeout=300)
            response.raise_for_status()
        except requests.RequestException as error:
            logger.warning("%s: %s", row.feed_id, error)
            continue
        same = hashlib.sha256(response.content).hexdigest() == row.sha256
        target = FEED_CACHE_DIR / (f"{row.feed_id}.zip" if same else f"{row.feed_id}.zip.changed")
        target.write_bytes(response.content)
        logger.info("%s: %s", row.feed_id, "ok" if same else "the source has changed since October 2026")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cities", nargs="*")
    parser.add_argument("--force", action="store_true", help="rebuild archives already on disk")
    parser.add_argument("--fetch", action="store_true", help="first download the feeds missing from feed_cache/")
    arguments = parser.parse_args()
    logger = setup("city_feeds")
    keys = restrict(cities().index, arguments.cities)
    if arguments.fetch:
        fetch_missing(logger, keys)
    fixes = pd.read_csv(ROUTE_TYPE_FIXES_CSV, dtype=str, keep_default_na=False)
    done = None if arguments.force else {key for key in keys if city_feed_path(key).exists()}
    sweep(
        keys,
        lambda key: build_city(key, fixes),
        CITY_FEEDS_CSV,
        force=arguments.force,
        done=done,
        describe=lambda rows: f"{rows[0]['status']} {rows[0].get('n_stops', 0)} stops, kept {rows[0].get('stop_share_kept', 0):.0%}",
        logger=logger,
    )


if __name__ == "__main__":
    main()
