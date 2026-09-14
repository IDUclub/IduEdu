"""Build one GTFS archive per city: merged across sources, clipped to the OSM boundary.

This is the artefact everything else consumes. A city is rarely one feed and is
often part of a larger one, so two things happen here and both are recorded,
because both are findings in their own right:

* **Merging.** Singapore arrives as four archives, Toronto and Los Angeles as
  three. Identifiers collide across sources -- every feed has a ``route_id`` of
  ``1`` -- so each source is prefixed before merging. ``n_feeds_merged`` says how
  fragmented a city's open data is.
* **Clipping.** A national feed is not a city network: the Norwegian archive
  carries 142 349 stops across 939 km. Stops outside the boundary are dropped and
  trips left with fewer than two stops go with them. ``stop_share_kept`` says how
  much of the source belonged to the city, and ``carved_from_regional`` marks the
  cities where this had to be done at all.

Clipping is by stop, not by route: a route running out of town still serves the
in-town stops, and cutting whole routes would delete service the city really has.

Boundaries come from the same OSM relation that later clips the OSM graph, so
both sides of the comparison are cut by one geometry rather than two.
"""

import argparse
import logging
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from bench_common import append_row
from shapely.geometry import LineString, MultiLineString, Point
from wide_feeds import FEED_CACHE
from wide_paths import (
    CITIES_CSV,
    CITY_FEED_DIR,
)
from wide_paths import CITY_FEEDS_CSV as MANIFEST_CSV
from wide_paths import CITY_MEMBERS_CSV as MEMBERS_CSV
from wide_places import use_benchmark_cache
from wide_profile import _read_member

from iduedu import get_4326_boundary

logger = logging.getLogger(__name__)

#: Tables carried into the merged archive. Only fare tables are dropped.
#:
#: ``shapes.txt`` is kept and clipped rather than discarded: the OSM side is
#: downloaded with ``clip_by_territory=True``, so unless the feed geometry is cut
#: by the same boundary the two sides are not geometrically comparable and any
#: geometric metric measures the difference in extent instead of in content.
#: ``frequencies.txt`` carries the headways of frequency-based feeds -- without it
#: Santiago's 418 bus routes have no service interval at all. ``pathways.txt``
#: carries station structure, which is what connects a metro platform to the
#: street; dropping it would reintroduce the disconnected-metro defect by hand.
TABLES = [
    "agency.txt",
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "frequencies.txt",
    "pathways.txt",
    "transfers.txt",
    "shapes.txt",
]

#: Columns holding identifiers that must not collide when sources are merged.
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


def parse_osm_id(value) -> int | None:
    """Read a boundary id from a CSV cell.

    The column holds blanks, so pandas types it as float and the ids arrive as
    ``19910275.0``. A plain ``isdigit`` check rejects every one of them, which is
    how Singapore, Hong Kong and Bogota all came back as "no usable boundary".
    """
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _clip_stop_times(times: pd.DataFrame, kept_ids: set) -> tuple[pd.DataFrame, int, int]:
    """Keep a segment only when **both** of its ends are inside the boundary.

    This mirrors ``clip_by_territory`` on the OSM side, which clips nodes and then
    keeps whole edges whose both endpoints survived. Dropping single stops instead
    would leave a trip with a segment jumping across the part of the route that
    ran outside the city -- an edge the OSM side would have removed.

    A trip that leaves the city and comes back splits into separate trips, one per
    run of consecutive in-boundary stops, because a single trip cannot express a
    gap and joining the runs would recreate exactly that phantom segment.
    """
    order = times.columns.tolist()
    frame = times.copy()
    frame["_seq"] = pd.to_numeric(frame.get("stop_sequence"), errors="coerce")
    frame = frame.sort_values(["trip_id", "_seq"], kind="stable")
    frame["_inside"] = frame["stop_id"].isin(kept_ids)

    # A new run starts at the first stop of a trip and after every gap.
    new_trip = frame["trip_id"] != frame["trip_id"].shift()
    broke = (~frame["_inside"]) | (~frame["_inside"].shift(fill_value=False))
    frame["_run"] = (new_trip | broke).cumsum()

    segments_before = int((frame["trip_id"] == frame["trip_id"].shift(-1)).sum())
    inside = frame[frame["_inside"]].copy()
    sizes = inside.groupby("_run")["stop_id"].transform("size")
    inside = inside[sizes >= 2]
    segments_after = int((inside["_run"] == inside["_run"].shift(-1)).sum())

    # Only a split trip needs a new id; an untouched trip keeps its own.
    runs_per_trip = inside.groupby("trip_id")["_run"].transform("nunique")
    run_index = inside.groupby("trip_id")["_run"].transform(lambda values: values.rank(method="dense").astype(int))
    inside["trip_id"] = inside["trip_id"].where(runs_per_trip == 1, inside["trip_id"] + "#" + run_index.astype(str))

    return inside[order].reset_index(drop=True), segments_before, segments_after


def _expand_split_trips(frame: pd.DataFrame, kept_trips: set) -> pd.DataFrame:
    """Duplicate a trip row for every run its trip was split into."""
    base = frame.copy()
    mapping: dict[str, list[str]] = {}
    for trip_id in kept_trips:
        mapping.setdefault(str(trip_id).split("#", 1)[0], []).append(str(trip_id))
    base["_new"] = base["trip_id"].astype(str).map(mapping)
    base = base[base["_new"].notna()].explode("_new")
    base["trip_id"] = base["_new"]
    return base.drop(columns=["_new"]).reset_index(drop=True)


def clip_shapes(shapes: pd.DataFrame, boundary) -> tuple[pd.DataFrame, float, float]:
    """Cut shape polylines to the boundary, rebuilding the point sequences.

    A route that leaves the city and comes back clips into several pieces; GTFS
    has no way to express a shape with a gap, so the longest piece is kept and the
    loss is reported. ``shape_dist_traveled`` is recomputed from the surviving
    geometry -- the original values measure the uncut line and would place stops
    at the wrong offsets.
    """
    if shapes is None or shapes.empty or boundary is None:
        return shapes, 0.0, 0.0

    latitudes = pd.to_numeric(shapes["shape_pt_lat"], errors="coerce")
    longitudes = pd.to_numeric(shapes["shape_pt_lon"], errors="coerce")
    sequence = pd.to_numeric(shapes.get("shape_pt_sequence"), errors="coerce")
    frame = pd.DataFrame({"shape_id": shapes["shape_id"], "lat": latitudes, "lon": longitudes, "seq": sequence}).dropna(
        subset=["lat", "lon"]
    )

    records: list[dict] = []
    length_before = length_after = 0.0
    for shape_id, group in frame.groupby("shape_id", sort=False):
        group = group.sort_values("seq", kind="stable") if group["seq"].notna().all() else group
        if len(group) < 2:
            continue
        line = LineString(list(zip(group["lon"], group["lat"])))
        length_before += line.length
        try:
            piece = line.intersection(boundary)
        except Exception:  # noqa: BLE001 - invalid geometry must not stop a city
            continue
        if piece.is_empty:
            continue
        if isinstance(piece, MultiLineString):
            piece = max(piece.geoms, key=lambda part: part.length)
        if not isinstance(piece, LineString) or len(piece.coords) < 2:
            continue
        length_after += piece.length

        travelled = 0.0
        previous = None
        for order, (longitude, latitude) in enumerate(piece.coords, start=1):
            if previous is not None:
                travelled += Point(previous).distance(Point((longitude, latitude)))
            previous = (longitude, latitude)
            records.append(
                {
                    "shape_id": shape_id,
                    "shape_pt_lat": round(latitude, 7),
                    "shape_pt_lon": round(longitude, 7),
                    "shape_pt_sequence": order,
                    "shape_dist_traveled": round(travelled, 6),
                }
            )
    return pd.DataFrame.from_records(records), length_before, length_after


def _prefixed(frame: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            values = out[column].fillna("").astype(str)
            out[column] = values.where(values.eq(""), prefix + ":" + values)
    return out


def read_feed(feed_id: str) -> dict[str, pd.DataFrame]:
    path = FEED_CACHE / f"{feed_id}.zip"
    tables: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as archive:
        for name in TABLES:
            frame = _read_member(archive, name)
            if frame is not None and not frame.empty:
                tables[name] = _prefixed(frame, ID_COLUMNS.get(name, []), feed_id)
    return tables


def clip_to_boundary(tables: dict[str, pd.DataFrame], boundary) -> tuple[dict[str, pd.DataFrame], dict]:
    """Drop everything outside the boundary and report how much survived."""
    stops = tables.get("stops.txt")
    stats = {"stops_before": 0 if stops is None else len(stops), "stops_after": 0}
    if stops is None or boundary is None:
        stats["stops_after"] = stats["stops_before"]
        return tables, stats

    latitudes = pd.to_numeric(stops["stop_lat"], errors="coerce")
    longitudes = pd.to_numeric(stops["stop_lon"], errors="coerce")
    points = gpd.GeoSeries([Point(x, y) for x, y in zip(longitudes, latitudes)], crs=4326)
    inside = points.within(boundary).to_numpy() & latitudes.notna().to_numpy()

    kept = stops.loc[inside].copy()
    stats["stops_after"] = len(kept)
    if kept.empty:
        return {}, stats

    kept_ids = set(kept["stop_id"])  # noqa: F841 - kept for the pathway/transfer filters below
    # Parent stations of kept platforms stay even if the parent point is outside.
    if "parent_station" in stops.columns:
        parents = set(stops.loc[inside, "parent_station"].dropna()) - {""}
        extra = stops[stops["stop_id"].isin(parents) & ~inside]
        if not extra.empty:
            kept = pd.concat([kept, extra], ignore_index=True)
            kept_ids |= set(extra["stop_id"])
    tables["stops.txt"] = kept

    times = tables.get("stop_times.txt")
    if times is not None:
        times, segments_before, segments_after = _clip_stop_times(times, kept_ids)
        stats["segments_before"], stats["segments_after"] = segments_before, segments_after
        tables["stop_times.txt"] = times
        kept_trips = set(times["trip_id"])
        # Splitting a trip at a boundary crossing creates new trip ids that
        # trips.txt and frequencies.txt have to learn about.
        for name in ("trips.txt", "frequencies.txt"):
            frame = tables.get(name)
            if frame is not None and "trip_id" in frame.columns:
                tables[name] = _expand_split_trips(frame, kept_trips)
    else:
        kept_trips = set()

    trips = tables.get("trips.txt")
    if trips is not None:
        trips = trips[trips["trip_id"].isin(kept_trips)]
        tables["trips.txt"] = trips
        kept_routes, kept_services = set(trips.get("route_id", [])), set(trips.get("service_id", []))
        for name, column, keep in (
            ("routes.txt", "route_id", kept_routes),
            ("calendar.txt", "service_id", kept_services),
            ("calendar_dates.txt", "service_id", kept_services),
            ("frequencies.txt", "trip_id", kept_trips),
        ):
            frame = tables.get(name)
            if frame is not None and column in frame.columns:
                tables[name] = frame[frame[column].isin(keep)]

    # Station structure and transfers survive only where both ends did.
    for name in ("pathways.txt", "transfers.txt"):
        frame = tables.get(name)
        if frame is not None and {"from_stop_id", "to_stop_id"} <= set(frame.columns):
            tables[name] = frame[frame["from_stop_id"].isin(kept_ids) & frame["to_stop_id"].isin(kept_ids)]

    # Geometry is carried over untouched. ``clip_by_territory`` on the OSM side
    # clips *nodes* and then keeps whole edges whose both endpoints survived
    # (editors.py:187) -- edge geometry is never cut. Cutting the feed's shapes
    # would therefore make the two sides incomparable in the opposite direction,
    # and it is also what made shape retention diverge from stop retention.
    shapes = tables.get("shapes.txt")
    if shapes is not None and trips is not None and "shape_id" in trips.columns:
        tables["shapes.txt"] = shapes[shapes["shape_id"].isin(set(trips["shape_id"]))]
    return tables, stats


def merge(parts: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    merged: dict[str, pd.DataFrame] = {}
    for name in TABLES:
        frames = [part[name] for part in parts if name in part and not part[name].empty]
        if frames:
            merged[name] = pd.concat(frames, ignore_index=True, sort=False)
    return merged


#: The keys ``validate_gtfs_feed`` enforces. Mirrored rather than imported so a
#: change in the library shows up here as a build failure instead of silently
#: changing what the dataset contains.
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


def enforce_primary_keys(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Make the merged archive satisfy GTFS uniqueness, counting what it cost.

    Identifiers are prefixed per source before merging, so a duplicate key here
    always came in with a source feed. Two kinds occur and they are not equally
    harmless:

    * **Repeated rows** -- identical in every column. Dropping them loses nothing;
      Bangkok has one such pair in ``frequencies.txt``.
    * **Colliding rows** -- one identifier, different content. Addis Ababa reuses
      ``stop_id`` for stops two kilometres apart. Nothing can recover which one a
      given trip meant, so the first is kept and the rest dropped; the trips
      pointing at the others silently move to the first location.

    The second kind is a defect of the published feed and is counted separately,
    because "the feed does not satisfy its own specification" is a result this
    study reports rather than a nuisance it hides.
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
        frame = frame.drop_duplicates(subset=keys, keep="first")
        counts["colliding_rows_dropped"] += before - len(frame)
        tables[name] = frame
    counts["unusable_frequencies_dropped"] = _drop_unusable_frequencies(tables)
    counts.update(_drop_untimeable_trips(tables))
    counts["orphan_rows_dropped"] = _drop_orphan_references(tables)
    return counts


#: Every reference ``validate_gtfs_feed`` enforces, in an order where resolving
#: one cannot orphan the next: trips lose their unknown routes and services
#: before stop_times is checked against what is left of trips.
REFERENCES = [
    ("trips.txt", "route_id", ("routes.txt",), "route_id"),
    ("trips.txt", "service_id", ("calendar.txt", "calendar_dates.txt"), "service_id"),
    ("stop_times.txt", "trip_id", ("trips.txt",), "trip_id"),
    ("stop_times.txt", "stop_id", ("stops.txt",), "stop_id"),
    ("frequencies.txt", "trip_id", ("trips.txt",), "trip_id"),
    ("pathways.txt", "from_stop_id", ("stops.txt",), "stop_id"),
    ("pathways.txt", "to_stop_id", ("stops.txt",), "stop_id"),
]


def _drop_orphan_references(tables: dict[str, pd.DataFrame]) -> int:
    """Drop rows pointing at identifiers no table defines.

    Moscow's published feed schedules 7416 trips that ``trips.txt`` never
    declares -- three percent of everything it references. A dangling reference
    aborts the build, and there is nothing to recover from it: a trip with no
    route, service or direction is a schedule for a service the feed does not
    describe.
    """
    dropped = 0
    for table, column, target_tables, target_column in REFERENCES:
        frame = tables.get(table)
        if frame is None or frame.empty or column not in frame.columns:
            continue
        known: set[str] = set()
        for target in target_tables:
            other = tables.get(target)
            if other is not None and target_column in other.columns:
                known |= set(other[target_column].dropna().map(str))
        if not known:
            continue
        values = frame[column].fillna("").map(str)
        orphan = ~values.isin(known) & values.ne("")
        if orphan.any():
            dropped += int(orphan.sum())
            tables[table] = frame.loc[~orphan]
    return dropped


def _seconds(values: pd.Series) -> pd.Series:
    parts = values.astype(str).str.strip().str.split(":", expand=True)
    if parts.shape[1] < 3:
        return pd.Series(float("nan"), index=values.index)
    numeric = parts.iloc[:, :3].apply(pd.to_numeric, errors="coerce")
    return numeric.iloc[:, 0] * 3600 + numeric.iloc[:, 1] * 60 + numeric.iloc[:, 2]


def _drop_unusable_frequencies(tables: dict[str, pd.DataFrame]) -> int:
    """Remove ``frequencies.txt`` rows that describe no frequency.

    Bangkok has fifteen rows whose window is empty and whose headway is zero --
    a single departure written in the frequency table rather than left to
    ``stop_times``. Kept, they abort the build; dropped, the trip falls back to
    its scheduled departure, which is what the row meant in the first place.
    """
    frame = tables.get("frequencies.txt")
    if frame is None or frame.empty or not {"start_time", "end_time", "headway_secs"} <= set(frame.columns):
        return 0
    start, end = _seconds(frame["start_time"]), _seconds(frame["end_time"])
    headway = pd.to_numeric(frame["headway_secs"], errors="coerce")
    unusable = start.isna() | end.isna() | headway.isna() | headway.le(0) | end.le(start)
    if not unusable.any():
        return 0
    tables["frequencies.txt"] = frame.loc[~unusable]
    return int(unusable.sum())


def _drop_untimeable_trips(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Remove trips whose schedule cannot yield a run time.

    Two defects, both fatal to the whole city when left in, because a segment
    with no usable time aborts the build rather than the trip:

    * **Times running backwards.** Kigali has a trip arriving at its tenth stop
      fifteen minutes before it left its ninth.
    * **A tail that never closes.** GTFS allows blank times between timepoints
      and expects the consumer to interpolate, which needs a known time on both
      sides. Curitiba and Kumasi have trips whose last stop carries no time, so
      everything after the first timepoint stays unknown.

    Interpolation here mirrors ``_prepare_stop_times`` so that a trip kept is a
    trip the builder can time. Dropped trips are counted, not hidden: the share
    of a feed that cannot be timed is a quality measure in its own right.
    """
    stop_times, trips = tables.get("stop_times.txt"), tables.get("trips.txt")
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
        frame[column] = frame.groupby("trip_id", sort=False)[column].transform(
            lambda values: values.interpolate(limit_area="inside")
        )

    unknown = frame["_arrival"].isna() | frame["_departure"].isna()
    backwards = frame.groupby("trip_id", sort=False)["_arrival"].shift(-1).lt(frame["_departure"])
    bad_trips = set(frame.loc[unknown | backwards, "trip_id"])
    if not bad_trips:
        return {"trips_dropped_bad_times": 0, "trips_kept": int(frame["trip_id"].nunique())}

    tables["stop_times.txt"] = stop_times.loc[~stop_times["trip_id"].isin(bad_trips)]
    if trips is not None and "trip_id" in trips.columns:
        tables["trips.txt"] = trips.loc[~trips["trip_id"].isin(bad_trips)]
    frequencies = tables.get("frequencies.txt")
    if frequencies is not None and "trip_id" in frequencies.columns:
        tables["frequencies.txt"] = frequencies.loc[~frequencies["trip_id"].isin(bad_trips)]
    kept = int(frame.loc[~frame["trip_id"].isin(bad_trips), "trip_id"].nunique())
    return {"trips_dropped_bad_times": len(bad_trips), "trips_kept": kept}


def write_archive(tables: dict[str, pd.DataFrame], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".zip.part")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in tables.items():
            archive.writestr(name, frame.to_csv(index=False, encoding="utf-8"))

    # Windows refuses to rename over a file another process still holds -- an
    # indexer or virus scanner touching a freshly written archive is enough, and
    # one such refusal killed a 114-city rebuild on its second city.
    for attempt in range(6):
        try:
            partial.replace(destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.4 * (attempt + 1))
            try:
                destination.unlink(missing_ok=True)
            except PermissionError:
                pass


def build_city(city: pd.Series, members: pd.DataFrame, force: bool = False) -> dict:
    city_key = str(city["city_key"])
    destination = CITY_FEED_DIR / f"{city_key}.zip"
    feeds = members.loc[members["city_key"] == city_key, "feed_id"].astype(str).tolist()
    row = {
        "city_key": city_key,
        "city_name": city.get("city_name", ""),
        "osm_id": city.get("osm_id", ""),
        "n_feeds_merged": len(feeds),
        "feeds": "+".join(feeds),
        "carved_from_regional": bool(city.get("carved_from_regional", False)),
        "stops_before": 0,
        "stops_after": 0,
        "stop_share_kept": 0.0,
        "status": "failed",
        "reason": "",
    }
    if destination.exists() and not force:
        row["status"] = "skipped"
        row["reason"] = "already built"
        return row
    osm_id = parse_osm_id(city.get("osm_id", ""))
    if osm_id is None:
        row["reason"] = "no usable boundary"
        return row

    try:
        boundary = get_4326_boundary(osm_id=osm_id)
    except Exception as exc:  # noqa: BLE001
        row["reason"] = f"boundary: {type(exc).__name__}: {exc}"[:200]
        return row

    parts, before, after = [], 0, 0
    segments_before = segments_after = 0
    for feed_id in feeds:
        try:
            tables = read_feed(feed_id)
        except (zipfile.BadZipFile, FileNotFoundError) as exc:
            logger.warning("%s: %s unreadable (%s)", city_key, feed_id, type(exc).__name__)
            continue
        clipped, stats = clip_to_boundary(tables, boundary)
        before += stats["stops_before"]
        after += stats["stops_after"]
        segments_before += stats.get("segments_before", 0)
        segments_after += stats.get("segments_after", 0)
        if clipped.get("stops.txt") is not None and not clipped["stops.txt"].empty:
            parts.append(clipped)

    row["stops_before"], row["stops_after"] = before, after
    row["stop_share_kept"] = round(after / before, 4) if before else 0.0
    row["segments_before"], row["segments_after"] = segments_before, segments_after
    row["segment_share_kept"] = round(segments_after / segments_before, 4) if segments_before else 0.0
    if not parts:
        row["reason"] = "no stop survived the boundary"
        return row

    merged = merge(parts)
    row.update(enforce_primary_keys(merged))
    write_archive(merged, destination)
    row["status"] = "ok"
    row["n_stops"] = len(merged.get("stops.txt", []))
    row["n_routes"] = len(merged.get("routes.txt", []))
    row["n_trips"] = len(merged.get("trips.txt", []))
    return row


def run(force: bool = False, only: list[str] | None = None) -> pd.DataFrame:
    use_benchmark_cache()
    cities = pd.read_csv(CITIES_CSV).fillna("")
    members = pd.read_csv(MEMBERS_CSV).fillna("")
    if only:
        cities = cities[cities["city_key"].astype(str).isin(only)]

    rows = []
    for index, (_, city) in enumerate(cities.iterrows(), start=1):
        row = build_city(city, members, force)
        if row["status"] != "skipped":
            append_row(MANIFEST_CSV, row)
        rows.append(row)
        logger.info(
            "[%d/%d] %-9s %-24s feeds=%d kept=%.0f%% %s",
            index,
            len(cities),
            row["status"],
            str(row["city_name"])[:24],
            row["n_feeds_merged"],
            row["stop_share_kept"] * 100,
            row["reason"],
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cities", nargs="*", default=None)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    result = run(arguments.force, arguments.cities)
    print("\n" + str(result["status"].value_counts().to_dict()))
    good = result[result["status"] == "ok"]
    if not good.empty:
        print(f"built {len(good)} city feeds, {good['n_stops'].sum():,} stops total")
        carved = good[good["carved_from_regional"]]
        print(f"  carved out of a regional feed: {len(carved)}")
        print(f"  merged from several sources:   {int((good['n_feeds_merged'] > 1).sum())}")


if __name__ == "__main__":
    main()
