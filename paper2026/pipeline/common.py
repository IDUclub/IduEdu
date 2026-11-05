"""What every stage shares: paths, the frozen inputs, the cohort, resumable sweeps.

Stages never declare paths of their own; the layout of the data is decided here.
``PAPER_RUN=<name>`` sends every output (tables, city feeds, graphs) to sibling
directories suffixed ``_<name>``, so a test run never touches the paper's results.
The inputs -- feed archives, city extracts, the tables in ``inputs/`` -- are shared.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from iduedu import DEFAULT_REGISTRY_W_TRAIN

# --- directories -------------------------------------------------------------

PAPER_DIR = Path(__file__).resolve().parents[1]
PAPER_RUN = os.environ.get("PAPER_RUN", "").strip()
_SUFFIX = f"_{PAPER_RUN}" if PAPER_RUN else ""

#: Frozen inputs: the cities, the feeds behind them, boundaries, regions.
INPUTS_DIR = PAPER_DIR / "inputs"
#: Downloaded GTFS archives, one ``<feed_id>.zip`` each (sha256 in ``inputs/feeds.csv``).
FEED_CACHE_DIR = PAPER_DIR / "feed_cache"
#: Geofabrik extracts: regional files, and ``cities/`` cut from them per city.
PBF_DIR = Path(os.environ.get("PAPER_PBF_DIR", PAPER_DIR / "osm"))
CITY_PBF_DIR = PBF_DIR / "cities"

RESULTS_DIR = PAPER_DIR / "results" / f"wide_tier{_SUFFIX}"
LOG_DIR = RESULTS_DIR / "logs"
CITY_FEED_DIR = PAPER_DIR / f"city_feeds{_SUFFIX}"
OSM_GRAPH_DIR = PAPER_DIR / f"osm_graphs{_SUFFIX}"
GTFS_GRAPH_DIR = PAPER_DIR / f"gtfs_graphs{_SUFFIX}"
VARIANT_GRAPH_DIR = PAPER_DIR / f"osm_graphs_variants{_SUFFIX}"

# --- inputs ------------------------------------------------------------------

CITIES_CSV = INPUTS_DIR / "cities.csv"
MEMBERS_CSV = INPUTS_DIR / "city_members.csv"
FEEDS_CSV = INPUTS_DIR / "feeds.csv"
BOUNDARIES_GEOJSON = INPUTS_DIR / "boundaries.geojson"
REGIONS_CSV = INPUTS_DIR / "geofabrik_regions.csv"
ROUTE_TYPE_FIXES_CSV = INPUTS_DIR / "route_type_fixes.csv"

# --- outputs, in the order the stages write them -----------------------------

CITY_FEEDS_CSV = RESULTS_DIR / "city_feeds.csv"
OSM_GRAPHS_CSV = RESULTS_DIR / "osm_graphs.csv"
GTFS_GRAPHS_CSV = RESULTS_DIR / "gtfs_graphs.csv"
COVERAGE_CSV = RESULTS_DIR / "indicators_coverage.csv"
WAITS_CSV = RESULTS_DIR / "indicators_waits.csv"
PROVENANCE_CSV = RESULTS_DIR / "feed_provenance.csv"
VARIANTS_CSV = RESULTS_DIR / "accessibility_variants.csv"
ACCESSIBILITY_CSV = RESULTS_DIR / "accessibility.csv"
ROUTE_STOPS_CSV = RESULTS_DIR / "route_stops.csv"

#: Every mode the OSM builder can supply. The OSM side is always built for all of
#: them, whatever the feed holds: a mode only OSM has must be able to show up.
ALL_OSM_MODES = sorted(DEFAULT_REGISTRY_W_TRAIN.list_types())
#: The modes both sources describe and the study compares.
TRANSIT_MODES = ("bus", "trolleybus", "tram", "subway", "train", "monorail")
THRESHOLDS_MIN = (15, 30, 45)
#: Every download names the study, as the October 2026 downloads did: the Transitous
#: mirror answers the default User-Agent of ``requests`` with HTTP 403.
USER_AGENT = "IduEdu-research/1.0 (+https://github.com/IduClub/IduEdu)"


def setup(name: str | None = None) -> logging.Logger:
    """UTF-8 console (the Windows default is not) and one log format for every stage."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    os.environ.setdefault("ENABLE_TQDM", "0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    from iduedu import config

    config.set_enable_tqdm(False)
    for directory in (RESULTS_DIR, LOG_DIR, CITY_FEED_DIR, OSM_GRAPH_DIR, GTFS_GRAPH_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    return logging.getLogger(name or "pipeline")


# --- paths of per-city artefacts ---------------------------------------------


def city_feed_path(city_key: str) -> Path:
    return CITY_FEED_DIR / f"{city_key}.zip"


def city_pbf_path(city_key: str) -> Path:
    return CITY_PBF_DIR / f"{city_key}.osm.pbf"


def region_pbf_path(region: str) -> Path:
    return PBF_DIR / f"{region.replace('/', '_')}.osm.pbf"


def osm_graph_path(city_key: str, kind: str) -> Path:
    """``kind`` is ``pt`` (every mode in ``ALL_OSM_MODES``) or ``walk``."""
    if kind == "pt":
        return OSM_GRAPH_DIR / f"{city_key}__{'+'.join(ALL_OSM_MODES)}__pt.urbangraph"
    return OSM_GRAPH_DIR / f"{city_key}__{kind}.urbangraph"


def gtfs_graph_path(city_key: str) -> Path:
    return GTFS_GRAPH_DIR / f"{city_key}__gtfs.urbangraph"


# --- graph metadata ----------------------------------------------------------


def registry_fingerprint(registry) -> str:
    """Digest of every registry constant that changes a travel time.

    Written beside each transit graph: a graph built under other constants is stale,
    and its file name cannot say so.
    """
    parts = []
    for mode in sorted(registry.list_types()):
        spec = registry.get(mode)
        part = (
            f"{mode}:{spec.avg_wait_time_min},{spec.base_speed_kmh},{spec.dwell_min},"
            f"{spec.accel_dist_m},{spec.brake_dist_m},{spec.vmax_tech_kmh}"
        )
        if spec.long_speed_kmh is not None:
            part += f",{spec.long_speed_kmh},{spec.speed_half_length_m}"
        parts.append(part)
    return hashlib.sha256(";".join(parts).encode("utf-8")).hexdigest()[:12]


def read_graph_meta(graph_path: Path) -> dict:
    path = graph_path.with_suffix(graph_path.suffix + ".meta.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_graph_meta(graph_path: Path, **fields) -> None:
    fields["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = graph_path.with_suffix(graph_path.suffix + ".meta.json")
    path.write_text(json.dumps(fields, indent=2), encoding="utf-8")


# --- frozen inputs -----------------------------------------------------------


@lru_cache(maxsize=1)
def cities() -> pd.DataFrame:
    """The study cities, one row each, indexed by ``city_key``."""
    return pd.read_csv(CITIES_CSV, keep_default_na=False).set_index("city_key")


@lru_cache(maxsize=1)
def members() -> pd.DataFrame:
    """Which feed serves which city (``feed_id``, ``city_key``)."""
    return pd.read_csv(MEMBERS_CSV, keep_default_na=False)


@lru_cache(maxsize=1)
def _boundaries():
    import geopandas as gpd

    return gpd.read_file(BOUNDARIES_GEOJSON).set_index("city_key")


def boundary(city_key: str):
    """The city's boundary polygon in EPSG:4326: it clips both sources."""
    return _boundaries().at[city_key, "geometry"]


@lru_cache(maxsize=1)
def regions() -> pd.DataFrame:
    """Geofabrik region of every city (``city_key`` -> ``region``, ``url``, snapshot)."""
    return pd.read_csv(REGIONS_CSV, keep_default_na=False).set_index("city_key")


# --- cohort ------------------------------------------------------------------


def _built(path: Path, statuses: tuple[str, ...]) -> set[str]:
    if not path.exists():
        return set()
    frame = last_rows(path, ["city_key"] + (["kind"] if path == OSM_GRAPHS_CSV else []))
    if path == OSM_GRAPHS_CSV:
        frame = frame[frame["kind"] == "pt"]
    return {str(key) for key in frame.loc[frame["status"].isin(statuses), "city_key"]}


def measurable() -> list[str]:
    """Cities every indicator is computed for: a frozen input row and a city feed on disk.

    The graphs are not required: a city whose OSM side is empty stays in the
    denominator, because "OSM has nothing here" is a result. With no city feed on disk
    at all (the deposit carries none) the cities whose feed was built, as
    ``city_feeds.csv`` records them, stand in, so the result tables can still be reported.
    """
    on_disk = sorted(key for key in cities().index if city_feed_path(key).exists())
    if on_disk or not CITY_FEEDS_CSV.exists():
        return on_disk
    built = last_rows(CITY_FEEDS_CSV, ["city_key"])
    return sorted(set(built.loc[built["status"] == "ok", "city_key"]) & set(cities().index))


def with_osm_graph() -> set[str]:
    return _built(OSM_GRAPHS_CSV, ("ok",))


def with_gtfs_graph(include_untimed: bool = False) -> set[str]:
    """Cities with a timetable graph; ``include_untimed`` adds those whose schedule has
    stops and departures but no run times (their waits are still measurable)."""
    return _built(GTFS_GRAPHS_CSV, ("ok", "no_usable_times") if include_untimed else ("ok",))


def restrict(keys: Iterable[str], wanted: Iterable[str] | None) -> list[str]:
    """Apply a ``--cities`` argument without letting it add cities."""
    keys = list(keys)
    if not wanted:
        return keys
    chosen = set(wanted)
    return [key for key in keys if key in chosen]


# --- tables ------------------------------------------------------------------


def last_rows(path: Path, keys: list[str], **read_kwargs) -> pd.DataFrame:
    """A result table with one row per key: the last one written."""
    frame = pd.read_csv(path, **read_kwargs)
    return frame.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append rows; a new column rewrites the file once with the union header.

    Retried, because on Windows a table open in a spreadsheet denies the append.
    """
    if not rows:
        return
    for attempt in range(6):
        try:
            if not path.exists():
                pd.DataFrame(rows).to_csv(path, index=False)
                return
            header = pd.read_csv(path, nrows=0).columns.to_list()
            new_columns = [c for c in dict.fromkeys(k for row in rows for k in row) if c not in header]
            if new_columns:
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                for column in new_columns:
                    frame[column] = ""
                frame.to_csv(path, index=False)
                header += new_columns
            pd.DataFrame([{c: row.get(c, "") for c in header} for row in rows]).to_csv(
                path, mode="a", header=False, index=False
            )
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.5 * (attempt + 1))


def sweep(
    keys: Iterable[str],
    work: Callable[[str], dict | list[dict]],
    path: Path,
    *,
    force: bool = False,
    done: set[str] | None = None,
    row_keys: list[str] | None = None,
    describe: Callable[[list[dict]], str] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Run ``work`` per city, append its rows as they come, then keep one row per key.

    A city already in the table is skipped unless ``force`` (``done`` overrides what
    counts as finished). An exception in one city becomes a ``failed`` row; the
    sweep goes on. Rows are appended immediately, so an interrupted run resumes;
    at the end the table is compacted to the last row per ``row_keys``.
    """
    logger = logger or logging.getLogger("pipeline")
    keys = list(keys)
    if done is None:
        done = (
            set() if force or not path.exists() else set(pd.read_csv(path, usecols=["city_key"])["city_key"].map(str))
        )
    pending = [key for key in keys if key not in done]
    logger.info("%s: %d cities, %d recorded, %d to do", path.name, len(keys), len(keys) - len(pending), len(pending))
    for index, key in enumerate(pending, start=1):
        try:
            produced = work(key)
        except Exception as error:  # noqa: BLE001 - one bad city must not end the sweep
            produced = {"city_key": key, "status": "failed", "reason": f"{type(error).__name__}: {error}"[:300]}
            logger.warning("%s: %s", key, produced["reason"])
        rows = produced if isinstance(produced, list) else [produced]
        append_rows(path, rows)
        headline = describe(rows) if describe and rows else (rows[0].get("status") if rows else "no rows")
        logger.info("[%d/%d] %s: %s", index, len(pending), key, headline)
    if path.exists():
        last_rows(path, row_keys or ["city_key"]).to_csv(path, index=False)
