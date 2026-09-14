from pathlib import Path
from typing import Any, Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy.spatial import cKDTree

from iduedu import config
from iduedu.graph.transformers import estimate_crs_for_bounds

from .reader import GTFSFeed, read_gtfs_feed
from .validation import validate_gtfs_feed

logger = config.logger

#: Column groups sharing one identifier space, as ``table -> columns``. Both the
#: definition and every reference are listed, because they must be rewritten
#: together or the feed stops validating.

ID_COLUMNS: dict[str, dict[str, tuple[str, ...]]] = {
    "agency_id": {"agency": ("agency_id",), "routes": ("agency_id",)},
    "stop_id": {
        "stops": ("stop_id", "parent_station"),
        "stop_times": ("stop_id",),
        "pathways": ("from_stop_id", "to_stop_id"),
    },
    "route_id": {"routes": ("route_id",), "trips": ("route_id",)},
    "trip_id": {"trips": ("trip_id",), "stop_times": ("trip_id",), "frequencies": ("trip_id",)},
    "service_id": {"trips": ("service_id",), "calendar": ("service_id",), "calendar_dates": ("service_id",)},
    "shape_id": {"shapes": ("shape_id",), "trips": ("shape_id",)},
    "block_id": {"trips": ("block_id",)},
    "zone_id": {"stops": ("zone_id",)},
    "level_id": {"levels": ("level_id",), "stops": ("level_id",)},
    "pathway_id": {"pathways": ("pathway_id",)},
}

#: ``location_type`` values that denote a place a vehicle actually calls at.
#: Stations, entrances, generic nodes and boarding areas are structure rather
#: than service and are never fused by coordinate.
_STOP_LOCATION_TYPES = {"", "0"}


def _feed_prefix(source: Any, index: int) -> str:
    """A short, stable name for a source, used to qualify its identifiers."""
    if isinstance(source, GTFSFeed):
        source = source.source
    stem = Path(str(source)).stem.strip()
    return stem or f"feed{index}"


def _unique_prefixes(sources: Sequence[Any]) -> list[str]:
    """One prefix per source, disambiguated when two sources share a name."""
    seen: dict[str, int] = {}
    prefixes: list[str] = []
    for index, source in enumerate(sources):
        base = _feed_prefix(source, index)
        count = seen.get(base, 0)
        seen[base] = count + 1
        prefixes.append(base if count == 0 else f"{base}{count + 1}")
    return prefixes


def _prefix_frame(frame: pd.DataFrame, columns: Iterable[str], prefix: str) -> pd.DataFrame:
    """Qualify the given columns with ``prefix``, leaving blanks blank.

    An empty ``parent_station`` means "this stop has no parent"; turning it into
    ``"nyct_subway:"`` would invent one, and the reference would not resolve.
    """
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].astype(str)
        filled = values.str.len() > 0
        frame.loc[filled, column] = prefix + ":" + values[filled]
    return frame


def _apply_prefix(feed: GTFSFeed, prefix: str) -> dict[str, pd.DataFrame]:
    tables = {name: frame.copy() for name, frame in feed.tables.items()}
    for spaces in ID_COLUMNS.values():
        for table, columns in spaces.items():
            if table in tables:
                tables[table] = _prefix_frame(tables[table], columns, prefix)
    return tables


def _concat(parts: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """Stack the tables of every source, unioning columns.

    Feeds differ in which optional columns they carry, and a column absent from
    one source must read as empty there rather than as NaN: every value in a
    GTFSFeed is a string.
    """
    names: list[str] = []
    for part in parts:
        for name in part:
            if name not in names:
                names.append(name)

    merged: dict[str, pd.DataFrame] = {}
    for name in names:
        frames = [part[name] for part in parts if name in part]
        columns: list[str] = []
        for frame in frames:
            for column in frame.columns:
                if column not in columns:
                    columns.append(column)
        aligned = [frame.reindex(columns=columns, fill_value="").fillna("") for frame in frames]
        merged[name] = pd.concat(aligned, ignore_index=True)
    return merged


def _stop_points(stops: pd.DataFrame) -> pd.DataFrame:
    """Stop rows that carry usable coordinates, with numeric lat/lon attached."""
    frame = stops.copy()
    frame["_lat"] = pd.to_numeric(frame.get("stop_lat"), errors="coerce")
    frame["_lon"] = pd.to_numeric(frame.get("stop_lon"), errors="coerce")
    eligible = frame["_lat"].notna() & frame["_lon"].notna()
    if "location_type" in frame.columns:
        eligible &= frame["location_type"].astype(str).isin(_STOP_LOCATION_TYPES)
    return frame.loc[eligible]


def _stop_merge_map(stops: pd.DataFrame, tolerance: float, crs: Any | None) -> dict[str, str]:
    """Map every fused stop_id onto the representative that survives.

    The intent is the one the OSM public-transport builder applies to platforms
    -- points close enough together are one place -- but not its mechanism. That
    builder snaps to a one-metre grid, which is adequate when the coordinates
    come from a single survey and platforms all but coincide. Across agencies the
    same kerb is published tens of metres apart, and at that tolerance a grid
    stops describing distance: whether two stops fuse would depend on where the
    cell boundary happens to fall rather than on how far apart they are.

    So the rule here is an explicit radius, assigned greedily in ``stop_id``
    order: each stop either joins the first representative within ``tolerance``
    or becomes one itself. Two properties follow, and both matter. Every fused
    stop really is within the tolerance of the stop it was fused into. And the
    result cannot chain -- under a transitive rule a line of stops each just
    inside the tolerance of its neighbour would collapse end to end, however long
    the line.
    """
    points = _stop_points(stops)
    if points.empty:
        return {}

    target = (
        CRS.from_user_input(crs)
        if crs is not None
        else estimate_crs_for_bounds(
            float(points["_lon"].min()),
            float(points["_lat"].min()),
            float(points["_lon"].max()),
            float(points["_lat"].max()),
        )
    )
    if not target.is_projected:
        raise ValueError(f"merge_stops_within needs a projected CRS in metres, got {target.to_string()}")

    projected = gpd.GeoSeries(gpd.points_from_xy(points["_lon"], points["_lat"]), crs="EPSG:4326").to_crs(target)

    # Sorting makes the surviving identifier a property of the data rather than
    # of the order the feeds happened to be passed in.
    order = np.argsort(points["stop_id"].astype(str).to_numpy(), kind="stable")
    stop_ids = points["stop_id"].astype(str).to_numpy()[order]
    coords = np.column_stack([projected.x.to_numpy()[order], projected.y.to_numpy()[order]])

    tree = cKDTree(coords)
    assigned = np.full(len(stop_ids), -1, dtype="int64")
    mapping: dict[str, str] = {}
    for index in range(len(stop_ids)):
        if assigned[index] != -1:
            continue
        assigned[index] = index
        for neighbour in tree.query_ball_point(coords[index], tolerance):
            if assigned[neighbour] == -1:
                assigned[neighbour] = index
                mapping[stop_ids[neighbour]] = stop_ids[index]
    return mapping


def _apply_stop_merge(tables: dict[str, pd.DataFrame], mapping: dict[str, str]) -> None:
    if not mapping:
        return

    # Drop the fused rows before rewriting anything: once ``stop_id`` has been
    # replaced by its representative, a fused row is indistinguishable from the
    # row that survives, and both would remain.
    stops = tables["stops"]
    tables["stops"] = stops.loc[~stops["stop_id"].astype(str).isin(mapping)].reset_index(drop=True)

    for table, columns in ID_COLUMNS["stop_id"].items():
        if table not in tables:
            continue
        frame = tables[table]
        for column in columns:
            # ``stops.stop_id`` now holds only survivors and must not be rewritten.
            if column in frame.columns and not (table == "stops" and column == "stop_id"):
                # ``map`` and not ``replace``: on a dictionary of thousands of keys
                # ``Series.replace`` falls off its fast path, and ``stop_times`` runs
                # to millions of rows in a city that publishes several feeds.
                values = frame[column].astype(str)
                frame[column] = values.map(mapping).fillna(values)

    # A pathway between two stops that are now one stop describes nothing.
    if "pathways" in tables:
        pathways = tables["pathways"]
        loops = pathways["from_stop_id"].astype(str) == pathways["to_stop_id"].astype(str)
        if loops.any():
            logger.info(f"Dropping {int(loops.sum())} pathways whose endpoints were merged into one stop")
            tables["pathways"] = pathways.loc[~loops].reset_index(drop=True)


def merge_gtfs_feeds(
    sources: Sequence[str | Path | GTFSFeed],
    *,
    prefixes: Sequence[str] | None = None,
    merge_stops_within: float | None = None,
    crs: Any | None = None,
    validate: bool = True,
) -> GTFSFeed:
    """Combine several GTFS feeds into one that the graph builder can consume.

    Identifiers are qualified by their source, so a ``route_id`` of ``1`` in two
    feeds becomes two distinct routes. Definitions and references are rewritten
    together, and blank optional references stay blank.

    Args:
        sources: Paths to GTFS directories or ZIP archives, or already-read
            feeds. A single source is returned unchanged unless stop merging is
            requested, so this function is a no-op where merging is not needed.
        prefixes: One prefix per source. Defaults to each source's file name,
            with a numeric suffix where two sources share one.
        merge_stops_within: Distance in metres below which stops published by
            different feeds are treated as the same place and fused into one
            node. ``None`` (the default) keeps every stop distinct: fusing stops
            changes the topology of the network, and doing it silently would make
            an inferred transfer indistinguishable from a published one. Values
            around 15--25 m suit feeds from different agencies; the 1 m used for
            OSM platforms is far too tight here, because two agencies survey the
            same kerb independently.
        crs: Projected CRS in metres used for the distance test. Estimated from
            the stops when omitted.
        validate: Run :func:`validate_gtfs_feed` on the result. Leave this on:
            it is what makes the merge verifiable rather than merely plausible.

    Returns:
        A :class:`GTFSFeed` whose ``source`` names the merged inputs.

    Raises:
        ValueError: If no sources are given, or the number of prefixes does not
            match the number of sources.
        GTFSValidationError: If the merged feed breaks a GTFS contract.
    """

    sources = list(sources)
    if not sources:
        raise ValueError("merge_gtfs_feeds needs at least one source")
    if prefixes is not None and len(prefixes) != len(sources):
        raise ValueError(f"got {len(prefixes)} prefixes for {len(sources)} sources")

    feeds = [source if isinstance(source, GTFSFeed) else read_gtfs_feed(source) for source in sources]

    # One feed needs no qualification: returning it untouched keeps single-feed
    # behaviour identical to reading it directly, which the rest of the library
    # and its published results depend on.
    if len(feeds) == 1 and prefixes is None and merge_stops_within is None:
        return feeds[0]

    names = list(prefixes) if prefixes is not None else _unique_prefixes(sources)

    if len(feeds) == 1 and prefixes is None:
        # Nothing to disambiguate against: qualifying the only feed's identifiers
        # would rename them for no reason.
        parts = [{name: frame.copy() for name, frame in feeds[0].tables.items()}]
    else:
        parts = [_apply_prefix(feed, prefix) for feed, prefix in zip(feeds, names)]
    tables = _concat(parts)

    if merge_stops_within is not None:
        if merge_stops_within <= 0:
            raise ValueError("merge_stops_within must be positive")
        mapping = _stop_merge_map(tables["stops"], float(merge_stops_within), crs)
        _apply_stop_merge(tables, mapping)
        logger.info(f"Merged {len(mapping)} stops into shared nodes at {merge_stops_within} m")

    merged = GTFSFeed(source=Path(" + ".join(names)), tables=tables)
    if validate:
        validate_gtfs_feed(merged)
    return merged
