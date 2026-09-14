#!/usr/bin/env python3
"""
B7 — Transit-layer construction from a GTFS feed: IduEdu vs peartree.

B2 measures what it costs to build the transit layer from OSM route relations.
This stage measures the same thing from the other open source -- a published
timetable -- so the two can be put side by side.

The two tools are not interchangeable and the table should not pretend they are.
IduEdu returns an ``UrbanGraph`` with separate boarding, alighting, travel and
station-link edges, carrying geometry and ready to be joined to a pedestrian
network. peartree returns a NetworkX ``MultiDiGraph`` of stops and headway-derived
costs, with no walk layer to join to; its ``join`` column is therefore empty by
construction, not by omission.

Service selection is made comparable rather than left to each tool's default:
both arms are given the same departure window (``--window``), because IduEdu
defaults to every trip in the feed and peartree requires a window.

Feeds come from ``bench_feeds/`` -- copies, kept apart from the study corpus in
``city_feeds/`` so that rerunning a sweep cannot change what a benchmark measured.
Cities whose feed is absent are reported with a reason and skipped; Seoul has no
GTFS feed in either public catalogue at all and is not expected here.

One CSV row per (area, library, attempt); resume-safe.

Usage:
    python bench_gtfs.py
    python bench_gtfs.py --smoke
    python bench_gtfs.py --areas "London,Moscow" --window 07:00-09:00
    python bench_gtfs.py --no-join            # transit layer only, skip the walk join
"""

import argparse
import time
from pathlib import Path

from bench_common import (
    RESULTS_DIR,
    append_row,
    dump_environment,
    force_cleanup,
    load_existing_keys,
    make_key,
    measure,
    networkx_graph_memory_mb,
    urban_graph_memory_mb,
)

try:
    from wide_paths import BENCH_FEED_DIR, use_paper_cache
except ImportError:  # pragma: no cover - the peartree arm runs without IduEdu
    # peartree 0.6.4 needs pandas 1.x (it calls Series.iteritems and reads the
    # grouping column inside groupby.apply, both gone since pandas 2.0), while
    # IduEdu requires pandas >= 3. The two cannot be installed together, so the
    # competitor arm is run from its own interpreter with --only-peartree, and
    # this module must import there -- where wide_paths, which imports IduEdu at
    # module level, does not.
    BENCH_FEED_DIR = Path(__file__).resolve().parents[1] / "bench_feeds"

    def use_paper_cache() -> None:
        """No Overpass traffic in the peartree arm: it never builds a walk layer."""


OUT_CSV = RESULTS_DIR / "gtfs_benchmark.csv"
KEY_COLUMNS = ["area", "library", "attempt"]

ATTEMPTS = 3
SLEEP_SEC = 0.5
DEFAULT_WINDOW = "07:00-09:00"

#: Area label -> feed file in ``bench_feeds/``. The labels match ``bench_common.AREAS``
#: so a row here lines up with the same city's row in B1 and B2.
#:
#: Moscow, London and Saint Petersburg are copies from the study corpus, already
#: merged and clipped to the city. Helsinki is HSL's own archive, which covers the
#: wider region rather than the municipality. New York is listed as its nine raw
#: archives and merged at build time.
#:
#: Missing for good: Seoul publishes no GTFS in the Mobility Database or the
#: Transitland Atlas, so the GTFS arm covers five of the six B1 cities.
#:
#: ⚠ The extents therefore differ: three cities are boundary-clipped, two are not.
#: For build cost against graph size that is legible, but any cross-city
#: comparison must read the node and edge counts alongside the seconds.
CITY_FEEDS: dict[str, str | list[str]] = {
    "Helsinki": "hsl.zip",
    "Saint Petersburg": "sankt_peterburg.zip",
    "Moscow": "moscow.zip",
    "London": "london.zip",
    # New York publishes nine archives and no merged one. Listed here rather than
    # pre-merged so the run exercises the library's own merge; identifiers collide
    # across all nine (every feed numbers a route "1"), which is what it is for.
    # Kept under the MTA's own file names, so an archive can be traced back to the
    # page it was downloaded from without a rename table.
    "New York": [
        "nyc_feeds/gtfs_subway.zip",  # NYCT subway
        "nyc_feeds/gtfs_bx.zip",  # NYCT bus, Bronx
        "nyc_feeds/gtfs_b.zip",  # NYCT bus, Brooklyn
        "nyc_feeds/gtfs_m.zip",  # NYCT bus, Manhattan
        "nyc_feeds/gtfs_q.zip",  # NYCT bus, Queens
        "nyc_feeds/gtfs_si.zip",  # NYCT bus, Staten Island
        "nyc_feeds/gtfs_busco.zip",  # MTA Bus Company
        "nyc_feeds/gtfslirr.zip",  # Long Island Rail Road
        "nyc_feeds/gtfsmnr.zip",  # Metro-North Railroad
    ],
}

NO_FEED_PUBLISHED = {
    "Seoul": "no GTFS feed in the Mobility Database or the Transitland Atlas",
}


def parse_window(text: str) -> tuple[str, str]:
    """Split ``HH:MM-HH:MM`` into its two bounds."""
    start, _, end = text.partition("-")
    start, end = start.strip(), end.strip()
    if not start or not end:
        raise ValueError(f"window must look like 07:00-09:00, got {text!r}")
    return start, end


def seconds_since_midnight(hhmm: str) -> int:
    hours, _, minutes = hhmm.partition(":")
    return int(hours) * 3600 + int(minutes) * 60


def feed_paths(area: str) -> list[Path] | None:
    """Return an area's feeds, or None with a printed reason.

    A city may publish one archive or nine; the builder merges a list itself, so
    both cases return a list and only the length differs.
    """
    if area in NO_FEED_PUBLISHED:
        print(f"  [skip] {area}: {NO_FEED_PUBLISHED[area]}")
        return None
    declared = CITY_FEEDS.get(area)
    if declared is None:
        print(f"  [skip] {area}: no feed declared in CITY_FEEDS")
        return None
    names = [declared] if isinstance(declared, str) else list(declared)
    paths = [BENCH_FEED_DIR / name for name in names]
    missing = [path for path in paths if not path.exists()]
    if missing:
        print(f"  [skip] {area}: {len(missing)} of {len(paths)} archives absent, first {missing[0]}")
        return None
    return paths


def walk_polygon_for(pt_graph):
    """A 4326 polygon covering the feed's stops, for the pedestrian layer.

    The walk network is derived from the feed's own extent rather than from the
    PBF bbox used in B1/B2: the feed is already clipped to its city, and building
    the walk layer over a larger extract would charge the join for streets no
    service touches.
    """
    from shapely.geometry import box

    bounds = pt_graph.nodes_gdf.to_crs(4326).total_bounds
    return box(*bounds)


def run_iduedu(existing: set, area: str, feeds: list[Path], window: tuple[str, str], do_join: bool) -> None:
    from iduedu import get_gtfs_public_transport_graph, get_walk_graph, join_pt_walk_graph

    start, end = window
    for attempt in range(1, ATTEMPTS + 1):
        key = make_key([area, "iduedu", attempt])
        if key in existing:
            print(f"  [skip] {key}")
            continue
        print(f"  [run ] {area} iduedu attempt={attempt}")

        # A list is merged inside the builder: identifiers are qualified by source.
        m_pt = measure(get_gtfs_public_transport_graph, feeds, start_time=start, end_time=end)
        pt_g = m_pt.result

        if pt_g.nodes_gdf.empty:
            print(f"  [warn] {area}: feed yielded an empty graph in {start}-{end}")
            append_row(
                OUT_CSV,
                dict(
                    area=area,
                    library="iduedu",
                    attempt=attempt,
                    window=f"{start}-{end}",
                    status="empty_graph",
                    time_pt_sec=round(m_pt.time_sec, 3),
                ),
            )
            existing.add(key)
            continue

        join_time = ""
        n_nodes_join = ""
        n_edges_join = ""
        if do_join:
            walk_g = get_walk_graph(territory=walk_polygon_for(pt_g), simplify=True, keep_largest_subgraph=False)
            m_join = measure(join_pt_walk_graph, pt_g, walk_g, keep_largest_subgraph=True)
            join_time = round(m_join.time_sec, 3)
            n_nodes_join = len(m_join.result.nodes_gdf)
            n_edges_join = len(m_join.result.edges_gdf)
            force_cleanup(walk_g, m_join.result)

        append_row(
            OUT_CSV,
            dict(
                area=area,
                library="iduedu",
                attempt=attempt,
                window=f"{start}-{end}",
                status="ok",
                time_pt_sec=round(m_pt.time_sec, 3),
                time_join_sec=join_time,
                n_nodes_pt=len(pt_g.nodes_gdf),
                n_edges_pt=len(pt_g.edges_gdf),
                n_nodes_intermodal=n_nodes_join,
                n_edges_intermodal=n_edges_join,
                representation_size_mb=round(urban_graph_memory_mb(pt_g), 2),
            ),
        )
        existing.add(key)
        force_cleanup(pt_g)
        time.sleep(SLEEP_SEC)


def run_peartree(existing: set, area: str, feeds: list[Path], window: tuple[str, str]) -> None:
    """Best-effort arm.

    peartree 0.6.4 was released around 2020 and is effectively unmaintained; it
    may not import on a current Python. A failure is recorded as a row with a
    status rather than left as a gap, because "the only direct competitor no
    longer runs" is itself a result worth reporting.
    """
    start_s = seconds_since_midnight(window[0])
    end_s = seconds_since_midnight(window[1])

    try:
        import peartree as pt
    except Exception as exc:  # ImportError, but a dead package can fail in other ways
        print(f"  [warn] peartree unavailable: {exc!r}")
        key = make_key([area, "peartree", 1])
        if key not in existing:
            append_row(
                OUT_CSV,
                dict(
                    area=area,
                    library="peartree",
                    attempt=1,
                    window=f"{window[0]}-{window[1]}",
                    status=f"unavailable: {type(exc).__name__}",
                ),
            )
            existing.add(key)
        return

    for attempt in range(1, ATTEMPTS + 1):
        key = make_key([area, "peartree", attempt])
        if key in existing:
            print(f"  [skip] {key}")
            continue
        print(f"  [run ] {area} peartree attempt={attempt}")

        try:
            # peartree reads one archive; a multi-feed city is beyond it, which is
            # itself worth recording rather than working around.
            if len(feeds) > 1:
                raise NotImplementedError(f"peartree cannot merge {len(feeds)} feeds")
            # Reading the archive is charged to both arms: IduEdu's builder is
            # handed paths and reads them inside the measured call, so timing
            # peartree from an already-parsed feed would compare unlike work.
            m_read = measure(pt.get_representative_feed, str(feeds[0]))
            feed_obj = m_read.result
            m = measure(pt.load_feed_as_graph, feed_obj, start_s, end_s)
        except Exception as exc:
            print(f"  [warn] {area}: peartree failed: {exc!r}")
            append_row(
                OUT_CSV,
                dict(
                    area=area,
                    library="peartree",
                    attempt=attempt,
                    window=f"{window[0]}-{window[1]}",
                    status=f"failed: {type(exc).__name__}",
                ),
            )
            existing.add(key)
            continue

        graph = m.result
        append_row(
            OUT_CSV,
            dict(
                area=area,
                library="peartree",
                attempt=attempt,
                window=f"{window[0]}-{window[1]}",
                status="ok",
                time_pt_sec=round(m_read.time_sec + m.time_sec, 3),
                time_read_sec=round(m_read.time_sec, 3),
                time_build_sec=round(m.time_sec, 3),
                # peartree builds no pedestrian layer, so there is nothing to join.
                time_join_sec="",
                n_nodes_pt=graph.number_of_nodes(),
                n_edges_pt=graph.number_of_edges(),
                representation_size_mb=round(networkx_graph_memory_mb(graph), 2),
            ),
        )
        existing.add(key)
        force_cleanup(graph)
        time.sleep(SLEEP_SEC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="smallest feed only, no walk join")
    parser.add_argument("--areas", default=None, help="comma-separated subset of area labels")
    parser.add_argument("--window", default=DEFAULT_WINDOW, help=f"departure window, default {DEFAULT_WINDOW}")
    parser.add_argument("--no-join", dest="join", action="store_false", help="skip the pedestrian join")
    parser.add_argument("--no-peartree", dest="peartree", action="store_false", help="skip the competitor arm")
    parser.add_argument(
        "--only-peartree",
        action="store_true",
        help="competitor arm only; run this from the pandas-1.x environment peartree needs",
    )
    args = parser.parse_args()

    window = parse_window(args.window)

    use_paper_cache()  # one Overpass cache for the paper; the walk join fetches through it
    dump_environment("gtfs_peartree" if args.only_peartree else "gtfs")
    existing = load_existing_keys(OUT_CSV, KEY_COLUMNS)
    if existing:
        print(f"[resume] {len(existing)} measurements already in {OUT_CSV}")

    if args.smoke:
        feeds = feed_paths("London")
        if feeds is None:
            return
        run_iduedu(existing, "smoke", feeds, window, do_join=False)
        print(f"[done] smoke results -> {OUT_CSV}")
        return

    areas = [a.strip() for a in args.areas.split(",")] if args.areas else list(CITY_FEEDS) + list(NO_FEED_PUBLISHED)
    for area in areas:
        print(f"\n=== {area} ===")
        feeds = feed_paths(area)
        if feeds is None:
            continue
        if not args.only_peartree:
            run_iduedu(existing, area, feeds, window, do_join=args.join)
        if args.peartree:
            run_peartree(existing, area, feeds, window)

    print(f"\n[done] results -> {OUT_CSV}")


if __name__ == "__main__":
    main()
