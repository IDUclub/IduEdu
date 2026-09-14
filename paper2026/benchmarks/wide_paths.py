"""Every path the wide-tier pipeline writes to or reads from, in one place.

Three reasons this module exists rather than each stage declaring its own paths.

**One name, one file.** ``MANIFEST_CSV`` used to mean ``osm_graphs.csv`` in one
module, ``gtfs_graphs.csv`` in three others and ``city_feeds.csv`` in a fifth,
while those modules import constants from each other. Names here say which file
they are.

**No import cycle.** ``wide_gtfs`` needs the walk graph's path to settle a city's
CRS, and ``wide_osm`` needs the GTFS manifest, so the two imported each other and
one of them had to do it inside a function body. Both now take paths from here.

**One place to move data.** The archive layout is decided by this file alone.
"""

from pathlib import Path

from iduedu import DEFAULT_REGISTRY_W_TRAIN, config

# --- directories -----------------------------------------------------------

PAPER_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PAPER_DIR / "benchmarks"
WIDE_DIR = PAPER_DIR / "results" / "wide_tier"
FEED_CACHE_DIR = PAPER_DIR / "feed_cache"
CATALOG_CACHE_DIR = PAPER_DIR / "catalog_cache"
CITY_FEED_DIR = PAPER_DIR / "city_feeds"
OSM_GRAPH_DIR = PAPER_DIR / "osm_graphs"
GTFS_GRAPH_DIR = PAPER_DIR / "gtfs_graphs"
#: Feeds the timing benchmarks build from, kept apart from the study corpus so a
#: sweep cannot change what a benchmark measured. Copies, not links.
BENCH_FEED_DIR = PAPER_DIR / "bench_feeds"
#: One Overpass cache for the whole paper. There used to be two -- the study
#: pipeline wrote here, the timing benchmarks wrote to ``benchmarks/.iduedu_cache``
#: -- so the same response could be fetched twice and two gigabytes of responses
#: sat inside the source directory.
OVERPASS_CACHE_DIR = PAPER_DIR / ".iduedu_cache"


def use_paper_cache() -> None:
    """Point IduEdu at the paper's own Overpass cache.

    Every stage that touches Overpass calls this first. Without it a run uses the
    library's default cache, which lives outside the project and would not be
    part of the archive.
    """
    config.set_overpass_cache(enabled=True, cache_dir=str(OVERPASS_CACHE_DIR))


# --- tables, in the order the pipeline writes them -------------------------

SELECTION_CSV = WIDE_DIR / "selection.csv"
DOWNLOADS_CSV = WIDE_DIR / "downloads.csv"
PROFILE_CSV = WIDE_DIR / "feed_profile.csv"
PLACES_REVIEW_CSV = WIDE_DIR / "places_review.csv"
CITIES_CSV = WIDE_DIR / "cities.csv"
CITY_MEMBERS_CSV = WIDE_DIR / "city_members.csv"
CITY_FEEDS_CSV = WIDE_DIR / "city_feeds.csv"
CITY_COMPLETION_CSV = WIDE_DIR / "city_completion.csv"

OSM_GRAPHS_CSV = WIDE_DIR / "osm_graphs.csv"
GTFS_GRAPHS_CSV = WIDE_DIR / "gtfs_graphs.csv"
OSM_SANITY_CSV = WIDE_DIR / "osm_sanity.csv"

COVERAGE_CSV = WIDE_DIR / "indicators_coverage.csv"
WAITS_CSV = WIDE_DIR / "indicators_waits.csv"
PROVENANCE_CSV = WIDE_DIR / "feed_provenance.csv"
PAIRS_CSV = WIDE_DIR / "matched_segments.csv"
PAIRS_SUMMARY_CSV = WIDE_DIR / "matched_segments_summary.csv"
SPEED_BY_LENGTH_CSV = WIDE_DIR / "speed_by_length.csv"
KINEMATICS_FIT_CSV = WIDE_DIR / "kinematics_fit.csv"
KINEMATICS_BY_CITY_CSV = WIDE_DIR / "kinematics_by_city.csv"
ACCESSIBILITY_CSV = WIDE_DIR / "accessibility.csv"

# --- graphs ----------------------------------------------------------------

#: The OSM side is fetched for **every** mode the builder can supply, not for the
#: modes a city's feed happens to contain: completeness is measured in both
#: directions, and a mode OSM has while the feed lacks it could otherwise never
#: appear.
#:
#: The comparison narrows the graph afterwards -- Saint Petersburg's feed has no
#: metro, and leaving metro in the OSM graph when computing accessibility would
#: hand the city its fastest mode for free. A superset can be narrowed; a graph
#: fetched narrow cannot be widened. Fixing the set also stabilises the cache key:
#: adding a feed with a new mode no longer invalidates a city's graph.
ALL_OSM_MODES = sorted(DEFAULT_REGISTRY_W_TRAIN.list_types())


def osm_graph_path(city_key: str, kind: str, modes: list[str] | None = None) -> Path:
    """Cache path. The mode set is in the name so a wider request cannot reuse a narrower graph."""
    if kind == "pt" and modes:
        return OSM_GRAPH_DIR / f"{city_key}__{'+'.join(modes)}__pt.urbangraph"
    return OSM_GRAPH_DIR / f"{city_key}__{kind}.urbangraph"


def gtfs_graph_path(city_key: str) -> Path:
    return GTFS_GRAPH_DIR / f"{city_key}__gtfs.urbangraph"


def city_feed_path(city_key: str) -> Path:
    return CITY_FEED_DIR / f"{city_key}.zip"


def registry_fingerprint(registry) -> str:
    """A short digest of everything in a registry that changes a travel time.

    A cached graph is stale the moment the constants that produced its edge times
    change, and the file name cannot say so: it carries the city and the modes,
    both of which stay the same. Writing this fingerprint beside the graph lets a
    build notice. The accessibility cache has keyed on the registry for a while;
    the transit graphs did not, so changing the bus speed required remembering to
    pass ``--force``, twice, from memory.
    """
    import hashlib

    parts = []
    for mode in sorted(registry.list_types()):
        spec = registry.get(mode)
        parts.append(
            f"{mode}:{spec.avg_wait_time_min},{spec.base_speed_kmh},{spec.dwell_min},"
            f"{spec.accel_dist_m},{spec.brake_dist_m},{spec.vmax_tech_kmh}"
        )
    return hashlib.sha256(";".join(parts).encode("utf-8")).hexdigest()[:12]


def graph_metadata_path(graph_path: Path) -> Path:
    """Where a graph's build fingerprint lives."""
    return graph_path.with_suffix(graph_path.suffix + ".meta.json")
