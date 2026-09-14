#!/usr/bin/env python3
"""
B1 — Graph construction benchmark: IduEdu vs OSMnx vs Pyrosm vs cityseer.

Measures wall time, node/edge counts, and resulting in-memory graph
representation size for walk and drive graphs over the same AOI (bbox from the
PBF header), with the simplify on/off ablation for IduEdu, OSMnx and cityseer
(pyrosm has no comparable switch).

The four are not interchangeable and the table should not pretend they are:
IduEdu, OSMnx and cityseer fetch from Overpass while pyrosm reads a local PBF,
and cityseer builds pedestrian networks only, so it is charged on walk alone.

One CSV row per attempt; resume-safe. Overpass responses are cached by both
IduEdu and OSMnx after the first (warm-up) call, so measured times reflect
parsing/graph assembly rather than network I/O — stated explicitly in the paper.

Usage:
    python bench_build.py                 # all cities, all libraries available
    python bench_build.py --smoke         # tiny cached territory, quick harness check
    python bench_build.py --areas "Helsinki,London" --networks walk
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from bench_common import (
    AREAS,
    CITYSEER_CACHE_DIR,
    RESULTS_DIR,
    bbox_from_pbf,
    dump_environment,
    force_cleanup,
    make_key,
    measure,
    networkx_graph_memory_mb,
    resolve_area_pbf,
    urban_graph_memory_mb,
)
from wide_paths import use_paper_cache

OUT_CSV = RESULTS_DIR / "build_benchmark.csv"
KEY_COLUMNS = ["library", "area", "network", "simplify", "attempt"]

ATTEMPTS = 3
SLEEP_SEC = 0.5
SIMPLIFY_SETTINGS = [True, False]
NETWORKS = ["walk", "drive"]

SMOKE_OSM_ID = 1114252  # small SPb district used by the test suite (cached)

#: Which Overpass instance every arm fetches from. IduEdu reads ``OVERPASS_URL``
#: itself; OSMnx and cityseer have to be told, and this is the only place that
#: tells them, so the three cannot silently end up on different servers.
#:
#: Needed because the main instance stopped completing our TLS handshake after a
#: day of large queries. A mirror serves the same planet with a different
#: replication lag -- hours at most, against the 0.4% node drift we measure over
#: six weeks -- so a city fetched from one is comparable with a city fetched from
#: another. Which instance filled the cache is recorded in ``env_build.json``.
OVERPASS_URL = os.getenv("OVERPASS_URL")


def _row_key(row) -> tuple:
    return make_key(row[column] for column in KEY_COLUMNS)


def _needs_graph_memory(row) -> bool:
    """Rows written before representation_size_mb existed should be re-measured."""
    return row.get("representation_size_mb", "") == ""


def normalize_build_csv_schema(csv_path: Path) -> pd.DataFrame:
    """Load B1 CSV and migrate old memory columns to the current schema."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    changed = False
    if "representation_size_mb" not in df.columns:
        if "graph_memory_mb" in df.columns:
            df = df.rename(columns={"graph_memory_mb": "representation_size_mb"})
        else:
            df["representation_size_mb"] = ""
        changed = True
    if "graph_memory_mb" in df.columns:
        df = df.drop(columns=["graph_memory_mb"])
        changed = True
    columns = [*KEY_COLUMNS, "time_sec", "n_nodes", "n_edges", "representation_size_mb"]
    extra_columns = [column for column in df.columns if column not in columns]
    if extra_columns:
        df = df.drop(columns=extra_columns)
        changed = True
    if changed:
        df.to_csv(csv_path, index=False)
    return df


def load_existing_build_keys(csv_path: Path) -> set[tuple]:
    """Return complete B1 keys, treating rows without representation size as pending."""
    if not csv_path.exists():
        return set()
    df = normalize_build_csv_schema(csv_path)
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} exists but misses key columns {missing}; delete or fix it")
    return {_row_key(row) for _, row in df.iterrows() if not _needs_graph_memory(row)}


def append_or_replace_build_row(csv_path: Path, row: dict) -> None:
    """Append a B1 row, replacing a stale row with the same resume key when present."""
    if not csv_path.exists():
        pd.DataFrame([row]).to_csv(csv_path, index=False)
        return

    df = normalize_build_csv_schema(csv_path)
    for column in row:
        if column not in df.columns:
            df[column] = ""

    key = make_key(row[column] for column in KEY_COLUMNS)
    key_mask = pd.Series(True, index=df.index)
    for column, value in zip(KEY_COLUMNS, key):
        key_mask &= df[column].astype(str) == value

    aligned = {column: row.get(column, "") for column in df.columns}
    if key_mask.any():
        first = key_mask[key_mask].index[0]
        df.loc[first, :] = aligned
        df = df.loc[~(key_mask & (df.index != first))]
        df.to_csv(csv_path, index=False)
        return

    pd.DataFrame([aligned]).to_csv(csv_path, mode="a", header=False, index=False)


# ----------------------------
# Builders: each returns (n_nodes, n_edges, graph_or_none).
# The graph is returned (not cleaned up) only when a post-build metric needs it;
# run_one computes that metric OUTSIDE the timed region and cleans up.
# ----------------------------


def build_iduedu(polygon, network: str, simplify: bool):
    from iduedu import get_drive_graph, get_walk_graph

    builder = get_walk_graph if network == "walk" else get_drive_graph
    graph = builder(territory=polygon, simplify=simplify, keep_largest_subgraph=True)
    return len(graph.nodes_gdf), len(graph.edges_gdf), graph


def build_osmnx(polygon, network: str, simplify: bool):
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.log_console = False
    if OVERPASS_URL:
        ox.settings.overpass_url = OVERPASS_URL
    graph = ox.graph_from_polygon(
        polygon=polygon,
        network_type=network,
        simplify=simplify,
        retain_all=False,
        truncate_by_edge=False,
    )
    return graph.number_of_nodes(), graph.number_of_edges(), graph


#: cityseer's own request, with one addition: a server-side time limit.
#:
#: Its default is ``[out:json];`` with no ``[timeout:]``, so the public API applies
#: its own 180-second budget, gives up on a city the size of London and closes the
#: connection mid-response -- which arrives as an SSL EOF that no amount of
#: retrying fixes. OSMnx does not hit this because it sends ``[timeout:180]`` and
#: splits a large polygon into tiles; IduEdu because it retries with backoff on a
#: query the server does answer. Reporting "cityseer cannot fetch London" while
#: the other two are asked politely and it is not would be a benchmark artefact,
#: not a property of the library.
#:
#: The filter body below is copied verbatim from cityseer 5.8.0
#: (``tools/io.osm_graph_from_poly``) with its default flags -- cycleways kept,
#: busways dropped -- so the response is the same set of ways its own request
#: would return. Check it against the installed version before upgrading cityseer.
CITYSEER_REQUEST = """
        /* https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL */
        [out:json][timeout:900];
        (way["highway"]
            ["highway"!~"bus_guideway|busway|escape|raceway|proposed|planned|abandoned|platform|emergency_bay|
                rest_area|disused|corridor|ladder|bus_stop|elevator|services"]
            ["area"!="yes"]
            ["footway"!="sidewalk"]
            ["amenity"!~"charging_station|parking|fuel|motorcycle_parking|parking_entrance|parking_space"]
            ["indoor"!="yes"]
            ["level"!="-2"]
            ["level"!="-3"]
            ["level"!="-4"]
            ["level"!="-5"](poly:"{geom_osm}");
        );
        out body;
        >;
        out qt;
        """


def cityseer_cache_path(area: str) -> Path:
    """One cached Overpass response per area, shared by both simplify settings.

    The request depends on the polygon, not on simplification, so the two ablation
    arms read the same bytes -- which is the point: the difference between them is
    then the cleaning cost and nothing else.
    """
    return CITYSEER_CACHE_DIR / f"{area.replace(' ', '_')}_walk.json"


def build_cityseer(polygon, network: str, simplify: bool, cache_path: Path):
    """Pedestrian networks only, which is what the library is for.

    cityseer fetches from Overpass like IduEdu and OSMnx, so the three are directly
    comparable. It has no driving mode: its request keeps cycleways and drops
    busways by default because it exists for pedestrian morphology, and running it
    over a drive network would report a tool doing something it does not claim.

    ``cache_path`` is what makes the arm comparable at all: IduEdu and OSMnx keep
    their own Overpass caches, so after the warm-up their measured runs parse a
    local response. cityseer caches nothing by itself and would otherwise be timed
    downloading the city again on every attempt.
    """
    import osmnx as ox
    from cityseer.tools import io

    # An interrupted or failed write leaves a truncated file that cityseer would
    # then happily read as a cached response. An empty one is no cache at all.
    if cache_path.exists() and cache_path.stat().st_size == 0:
        cache_path.unlink()

    # Simplification is not one request but four: after the network, cityseer asks
    # OSMnx for parks, plazas and parking so it can label footways inside green
    # areas. ``cache_path`` covers only the first, and the other three carry
    # OSMnx's default 180-second server budget, which London is too large to meet
    # -- the server gives up and drops the connection. Raise the budget for the
    # duration of this call only: leaving it raised would change the query string
    # of our own OSMnx arm, invalidating its cache and, worse, measuring it under
    # settings it does not ship with.
    previous_timeout = ox.settings.requests_timeout
    ox.settings.requests_timeout = 900
    try:
        graph = io.osm_graph_from_poly(
            polygon,
            poly_crs_code=4326,
            simplify=simplify,
            cache_path=cache_path,
            custom_request=CITYSEER_REQUEST,
            timeout=960,
            overpass_url=OVERPASS_URL,
        )
    finally:
        ox.settings.requests_timeout = previous_timeout
    return graph.number_of_nodes(), graph.number_of_edges(), graph


def cityseer_available() -> bool:
    try:
        import cityseer  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - an optional competitor must not end the run
        return False


def build_pyrosm(pbf_path: str, network: str):
    from pyrosm import OSM

    network_type = "walking" if network == "walk" else "driving"
    osm = OSM(pbf_path)
    nodes, edges = osm.get_network(nodes=True, network_type=network_type)
    graph = osm.to_graph(nodes, edges, graph_type="networkx", network_type=network_type)
    force_cleanup(osm, nodes, edges)
    return graph.number_of_nodes(), graph.number_of_edges(), graph


def pyrosm_available() -> bool:
    try:
        import pyrosm  # noqa: F401

        return True
    except ImportError:
        return False


# ----------------------------
# Main
# ----------------------------


def warm_up(label: str, build_fn, *args, tries: int = 6) -> None:
    """Populate a library's Overpass cache, surviving a server that drops us.

    Six tries at 60, 120, 240, 480 and 960 seconds is half an hour of patience,
    which is what a public instance needs: it refuses a large query outright while
    the slot of the previous one is still busy, and the arms are serialised, so
    each arm arrives right behind the last arm's heaviest fetch.

    Overpass answers a large city with an SSL EOF when it is loaded, and OSMnx
    does not retry: London, Seoul and New York were all lost that way, after the
    same fetches had succeeded for IduEdu, whose client backs off on its own. The
    retry lives here, in the warm-up, and nowhere near a measured call -- a retry
    inside a timed region would quietly turn one network stall into a benchmark
    result.
    """
    for attempt in range(1, tries + 1):
        try:
            force_cleanup(build_fn(*args)[2])
            return
        except Exception as error:  # noqa: BLE001 - any transport failure is retryable here
            if attempt == tries:
                raise
            delay = 60 * 2 ** (attempt - 1)
            print(
                f"  [warm] {label}: {type(error).__name__}, retrying in {delay}s (attempt {attempt}/{tries})",
                flush=True,
            )
            time.sleep(delay)


#: The three feature layers cityseer 5.8.0 asks OSMnx for while simplifying, so
#: it can label footways inside green areas. Copied from its ``_auto_clean_network``.
CITYSEER_AUX_TAGS = (
    ("parks", {"landuse": ["cemetery", "forest"], "leisure": ["park", "garden", "sports_centre"]}),
    ("plazas", {"highway": ["pedestrian"]}),
    ("parking", {"amenity": ["parking"]}),
)


def warm_cityseer_auxiliaries(polygon, tries: int = 7) -> None:
    """Fetch cityseer's three auxiliary layers into the OSMnx cache, one at a time.

    Without this, every retry of the cityseer warm-up re-reads and re-parses its
    cached network response -- hundreds of megabytes, minutes of local work --
    before reaching the auxiliary query that is actually failing, so six retries
    spend hours redoing what already succeeded. Warming each layer separately
    costs one request each and leaves exactly the cache state a successful
    cityseer run would have left.

    Worth being stubborn here rather than anywhere else: an uncached layer is
    re-requested by *every* measured attempt, so one refusal in six loses the
    city. Seoul recorded exactly one attempt of six that way -- the parking layer
    went through on the first build and not on the second.

    Failures are swallowed: this is an optimisation of the warm-up, and the
    cityseer warm-up that follows will report the problem properly.
    """
    import osmnx as ox

    previous_timeout = ox.settings.requests_timeout
    ox.settings.requests_timeout = 900
    try:
        for name, tags in CITYSEER_AUX_TAGS:
            for attempt in range(1, tries + 1):
                try:
                    ox.features_from_polygon(polygon, tags=tags)
                    print(f"  [warm] cityseer/{name}: cached", flush=True)
                    break
                except Exception as error:  # noqa: BLE001 - InsufficientResponseError included
                    if type(error).__name__ == "InsufficientResponseError":
                        print(f"  [warm] cityseer/{name}: empty response, which cityseer handles", flush=True)
                        break
                    if attempt == tries:
                        print(f"  [warm] cityseer/{name}: giving up ({type(error).__name__})", flush=True)
                        break
                    delay = 60 * 2 ** (attempt - 1)
                    print(
                        f"  [warm] cityseer/{name}: {type(error).__name__}, retrying in {delay}s "
                        f"(attempt {attempt}/{tries})",
                        flush=True,
                    )
                    time.sleep(delay)
    finally:
        ox.settings.requests_timeout = previous_timeout


def try_warm(label: str, build_fn, *args) -> bool:
    """Warm one arm up, returning whether it is safe to measure.

    A city is four independent arms, and until this existed a warm-up that ran
    out of retries took the other three down with it: Overpass refused cityseer
    on London and the IduEdu, OSMnx and pyrosm rows for that city -- already
    fetched and ready to measure -- were never written. An arm that could not be
    warmed is skipped rather than measured, because its first measured call would
    then be timing the download.
    """
    try:
        warm_up(label, build_fn, *args)
        return True
    except Exception as error:  # noqa: BLE001 - the other arms still have work to do
        print(
            f"  [skip] {label}: warm-up failed after retries ({type(error).__name__}); "
            f"leaving this arm for a later run",
            flush=True,
        )
        return False


def has_pending(existing: set, library: str, area: str, network: str, simplifies: list) -> bool:
    """True if any (simplify, attempt) measurement for this library is still missing."""
    return any(
        make_key([library, area, network, simplify, attempt]) not in existing
        for simplify in simplifies
        for attempt in range(1, ATTEMPTS + 1)
    )


def run_one(
    existing: set, library: str, area: str, network: str, simplify: bool | None, attempt: int, build_fn, *args
) -> None:
    key = make_key([library, area, network, simplify, attempt])
    if key in existing:
        print(f"  [skip] {key}")
        return
    print(f"  [run ] {library} {area} {network} simplify={simplify} attempt={attempt}")
    m = measure(build_fn, *args)
    n_nodes, n_edges, graph = m.result

    # Deterministic size of the resulting graph representation, computed outside
    # the timed region so it does not pollute time_sec.
    representation_size_mb = ""
    if graph is not None:
        if library == "iduedu":
            representation_size_mb = round(urban_graph_memory_mb(graph), 2)
        else:
            representation_size_mb = round(networkx_graph_memory_mb(graph), 2)
        force_cleanup(graph)

    append_or_replace_build_row(
        OUT_CSV,
        dict(
            library=library,
            area=area,
            network=network,
            simplify="NA" if simplify is None else simplify,
            attempt=attempt,
            time_sec=round(m.time_sec, 3),
            n_nodes=n_nodes,
            n_edges=n_edges,
            representation_size_mb=representation_size_mb,
        ),
    )
    existing.add(key)
    time.sleep(SLEEP_SEC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny cached territory, iduedu+osmnx only")
    parser.add_argument("--areas", default=None, help="comma-separated subset of areas")
    parser.add_argument("--networks", default=",".join(NETWORKS))
    parser.add_argument(
        "--libraries",
        default=None,
        help=(
            "comma-separated subset of arms to measure (iduedu, osmnx, cityseer, pyrosm). "
            "pyrosm reads a local PBF and needs no Overpass, so it is the one arm that can be "
            "banked while the API is refusing large queries; the others can then be filled in "
            "when it recovers, since every row is resume-safe."
        ),
    )
    args = parser.parse_args()

    # Without this the library falls back to a cache path relative to the working
    # directory, so the benchmarks quietly built their own 1.7 GB store inside
    # ``benchmarks/`` while the study's cache went unused. Sharing one cache also
    # keeps the comparison honest: IduEdu and OSMnx both read Overpass through it.
    use_paper_cache()

    if OVERPASS_URL:
        print(f"[overpass] every arm fetching from {OVERPASS_URL}")
    dump_environment("build")
    existing = load_existing_build_keys(OUT_CSV)
    if existing:
        print(f"[resume] {len(existing)} measurements already in {OUT_CSV}")

    networks = [n.strip() for n in args.networks.split(",") if n.strip()]

    if args.smoke:
        from iduedu import get_4326_boundary

        polygon = get_4326_boundary(osm_id=SMOKE_OSM_ID)
        for network in networks:
            for simplify in SIMPLIFY_SETTINGS:
                run_one(existing, "iduedu", "smoke", network, simplify, 1, build_iduedu, polygon, network, simplify)
                run_one(existing, "osmnx", "smoke", network, simplify, 1, build_osmnx, polygon, network, simplify)
        print(f"[done] smoke results -> {OUT_CSV}")
        return

    areas = [a.strip() for a in args.areas.split(",")] if args.areas else AREAS
    wanted = {lib.strip() for lib in args.libraries.split(",")} if args.libraries else None
    if wanted is not None:
        unknown = wanted - {"iduedu", "osmnx", "cityseer", "pyrosm"}
        if unknown:
            parser.error(f"unknown library/libraries: {sorted(unknown)}")
        print(f"[arms] measuring {sorted(wanted)} only")
    has_pyrosm = pyrosm_available() and (wanted is None or "pyrosm" in wanted)
    has_cityseer = cityseer_available() and (wanted is None or "cityseer" in wanted)
    want_iduedu = wanted is None or "iduedu" in wanted
    want_osmnx = wanted is None or "osmnx" in wanted
    if not has_cityseer:
        print("[warn] cityseer not importable in this environment; its walk rows will be missing")
    if not has_pyrosm:
        print("[warn] pyrosm not importable in this environment; run its rows from a conda env later")

    for area in areas:
        try:
            # Work out what is still pending up front, so a fully-recorded area needs
            # neither a PBF download (hundreds of MB) nor a warm-up.
            idu_pending = {
                n: want_iduedu and has_pending(existing, "iduedu", area, n, SIMPLIFY_SETTINGS) for n in networks
            }
            osm_pending = {
                n: want_osmnx and has_pending(existing, "osmnx", area, n, SIMPLIFY_SETTINGS) for n in networks
            }
            pyr_pending = {n: has_pyrosm and has_pending(existing, "pyrosm", area, n, [None]) for n in networks}
            # cityseer is a walk-only arm, so it is pending on the walk network alone.
            cs_pending = {
                n: has_cityseer and n == "walk" and has_pending(existing, "cityseer", area, n, SIMPLIFY_SETTINGS)
                for n in networks
            }

            # Merging these with ** collapses them by key -- all three are keyed by
            # network name -- so only pyrosm survived the merge. With pyrosm absent or
            # already recorded, every area reported "already recorded" and skipped the
            # iduedu and OSMnx arms entirely, in zero seconds, over a file that did not
            # exist. Chain the values instead of merging the dicts.
            if not any(
                list(idu_pending.values())
                + list(osm_pending.values())
                + list(pyr_pending.values())
                + list(cs_pending.values())
            ):
                print(f"\n=== {area}: all measurements already recorded, skipping ===")
                continue

            # An arm is measured only when its warm-up left a usable cache behind.
            idu_warm = dict.fromkeys(networks, True)
            osm_warm = dict.fromkeys(networks, True)
            cs_warm = dict.fromkeys(networks, True)

            pbf_path = resolve_area_pbf(area)
            bounds = bbox_from_pbf(pbf_path)
            print(f"\n=== {area} | bbox={bounds.bbox} ===")

            # Warm-up populates the Overpass cache for iduedu/osmnx so the first
            # recorded run excludes the download. Skip per library+network when
            # nothing is pending there.
            for network in networks:
                if idu_pending[network]:
                    print(f"  [warm] iduedu {network}")
                    idu_warm[network] = try_warm(f"iduedu {network}", build_iduedu, bounds.polygon_4326, network, True)
                if osm_pending[network]:
                    print(f"  [warm] osmnx {network}")
                    osm_warm[network] = try_warm(f"osmnx {network}", build_osmnx, bounds.polygon_4326, network, True)
                if cs_pending[network]:
                    print(f"  [warm] cityseer {network}")
                    warm_cityseer_auxiliaries(bounds.polygon_4326)
                    cs_warm[network] = try_warm(
                        f"cityseer {network}",
                        build_cityseer,
                        bounds.polygon_4326,
                        network,
                        True,
                        cityseer_cache_path(area),
                    )

            for network in networks:
                if has_pyrosm:
                    for attempt in range(1, ATTEMPTS + 1):
                        run_one(existing, "pyrosm", area, network, None, attempt, build_pyrosm, pbf_path, network)
                for simplify in SIMPLIFY_SETTINGS:
                    for attempt in range(1, ATTEMPTS + 1) if idu_warm[network] and want_iduedu else ():
                        run_one(
                            existing,
                            "iduedu",
                            area,
                            network,
                            simplify,
                            attempt,
                            build_iduedu,
                            bounds.polygon_4326,
                            network,
                            simplify,
                        )
                    for attempt in range(1, ATTEMPTS + 1) if osm_warm[network] and want_osmnx else ():
                        run_one(
                            existing,
                            "osmnx",
                            area,
                            network,
                            simplify,
                            attempt,
                            build_osmnx,
                            bounds.polygon_4326,
                            network,
                            simplify,
                        )
                    # Walk only: cityseer has no drive mode to charge.
                    if has_cityseer and network == "walk" and cs_warm[network]:
                        for attempt in range(1, ATTEMPTS + 1):
                            run_one(
                                existing,
                                "cityseer",
                                area,
                                network,
                                simplify,
                                attempt,
                                build_cityseer,
                                bounds.polygon_4326,
                                network,
                                simplify,
                                cityseer_cache_path(area),
                            )

        except Exception as error:  # noqa: BLE001 - one city must not end the sweep
            # Overpass answers 504 under load and its retries can run out, and a
            # mirror can refuse a PBF. Every stage here is resume-safe, so the
            # honest response is to name the city that failed and go on to the
            # next one rather than lose the cities that would have followed.
            print(f"[fail] {area}: {type(error).__name__}: {error}", flush=True)

    print(f"\n[done] results -> {OUT_CSV}")


if __name__ == "__main__":
    main()
