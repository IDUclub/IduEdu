"""Which cities a stage runs over, decided once instead of five times.

Every measuring stage used to pick its own set. ``wide_provenance`` globbed the
city-feed archives, ``wide_indicators`` read ``cities.csv`` and filtered on a
usable boundary, ``wide_access`` and ``wide_waits`` took whatever the GTFS
manifest called built, and ``wide_sanity`` took the OSM manifest. The four sets
differ, so the denominators of the result tables differed too -- by code rather
than by data, which is the one reason a reader cannot check.

The concrete case that exposed it: Kangar and Nuuk have a city feed but no
boundary in ``cities.csv``, so their feeds were never clipped and no OSM graph
was ever built for them. They still reached the waiting-time and provenance
tables, which therefore covered 115 cities while coverage and matching covered
113 -- and nothing in either table said why.

**The rule.** A city enters the study when it has a boundary, because the
boundary is what makes the two sides comparable: the same relation clips the feed
and the OSM extract. Cities without one are not silently dropped -- they are
returned by :func:`excluded` with a reason, and the funnel reports them.
"""

import pandas as pd
from wide_paths import (
    CITIES_CSV,
    CITY_FEED_DIR,
    GTFS_GRAPHS_CSV,
    OSM_GRAPHS_CSV,
    city_feed_path,
)

#: A manifest row counts as built under these statuses; anything else is a failure
#: that the funnel reports rather than hides.
BUILT = ("ok", "cached")

#: A merged city feed with fewer stops than this is one operator's shuttle, not a
#: city network, and comparing a whole OSM extract against it recreates the
#: partial-reference defect: the reverse share collapses and the accessibility
#: ratio explodes. All eight cities below the line claimed OSM overestimates --
#: Vancouver with 8 stops against 4,293 OSM transit nodes claimed 1.86.
#:
#: The test uses the feed's own size and never compares it with OSM, so it cannot
#: select cities by the outcome being measured. Moving the line to 50 stops
#: changes the headline correlation by 0.006; the number is recorded here so the
#: sensitivity can be rerun.
STUB_STOPS = 20


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1"])


def _manifest(path, statuses: tuple[str, ...] = BUILT) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path).drop_duplicates(subset=["city_key"], keep="last")
    return {str(key) for key in frame.loc[frame["status"].isin(statuses), "city_key"]}


def resolved() -> pd.DataFrame:
    """Every city the resolver produced, boundary or not."""
    return pd.read_csv(CITIES_CSV).fillna("")


def with_boundary() -> set[str]:
    """Cities with a boundary that can clip both sides. The study's population."""
    table = resolved()
    return {str(key) for key in table.loc[_truthy(table["has_usable_boundary"]), "city_key"]}


def with_city_feed() -> set[str]:
    """Cities whose feeds were merged into one archive, as it exists on disk.

    The archive rather than the manifest: a manifest row can outlive the file it
    describes, and every stage downstream reads the file.
    """
    return {path.stem for path in CITY_FEED_DIR.glob("*.zip")}


def with_osm_graph() -> set[str]:
    return _manifest(OSM_GRAPHS_CSV)


def with_gtfs_graph(include_untimed: bool = False) -> set[str]:
    """Cities with a reference graph.

    ``include_untimed`` also takes cities whose schedule carries no run times:
    their stops and departures are intact, so headways are measurable even though
    travel times are not. Curitiba is the whole of that set.
    """
    statuses = BUILT + ("no_usable_times",) if include_untimed else BUILT
    return _manifest(GTFS_GRAPHS_CSV, statuses)


def feed_sizes() -> dict[str, float]:
    """Stops in each city's merged feed, as the resolver recorded them."""
    table = resolved()
    sizes = pd.to_numeric(table["n_stops"], errors="coerce").fillna(0)
    return {str(key): float(value) for key, value in zip(table["city_key"], sizes)}


def measurable() -> list[str]:
    """Cities every indicator can be computed for: a boundary, a feed archive, and
    a feed large enough to be a city network.

    This is the cohort for coverage, waiting times, provenance and matching. It
    deliberately does not require the graphs to exist -- a city whose OSM graph
    failed still belongs in the denominator, because "OSM has nothing here" is a
    finding rather than a missing measurement.
    """
    sizes = feed_sizes()
    return sorted(key for key in with_boundary() if city_feed_path(key).exists() and sizes.get(key, 0) >= STUB_STOPS)


def comparable() -> list[str]:
    """Cities where both sides exist, so an accessibility ratio can be formed."""
    return sorted(set(measurable()) & with_osm_graph() & with_gtfs_graph())


def excluded() -> pd.DataFrame:
    """Cities left out of :func:`measurable`, one row each, with the reason.

    Kept as data rather than prose so the funnel figure and the manuscript can
    both be generated from it.
    """
    table = resolved()
    boundary = with_boundary()
    sizes = feed_sizes()
    rows = []
    for key in (str(k) for k in table["city_key"]):
        has_feed = city_feed_path(key).exists()
        if key in boundary and has_feed and sizes.get(key, 0) >= STUB_STOPS:
            continue
        if key not in boundary and has_feed:
            reason = "no boundary that holds the network"
        elif key not in boundary:
            reason = "no boundary and no city feed"
        elif not has_feed:
            reason = "no city feed archive"
        else:
            reason = f"feed too small to be a city network ({int(sizes.get(key, 0))} stops)"
        rows.append({"city_key": key, "reason": reason})
    return pd.DataFrame(rows, columns=["city_key", "reason"])


def restrict(keys: list[str], wanted: list[str] | None) -> list[str]:
    """Apply a ``--cities`` argument without letting it invent cities."""
    if not wanted:
        return keys
    chosen = set(wanted)
    return [key for key in keys if key in chosen]


def main() -> None:
    print(f"resolved            {len(resolved()):>4}")
    print(f"with a boundary     {len(with_boundary()):>4}")
    print(f"with a city feed    {len(with_city_feed()):>4}")
    print(f"measurable          {len(measurable()):>4}")
    print(f"with an OSM graph   {len(with_osm_graph()):>4}")
    print(f"with a GTFS graph   {len(with_gtfs_graph()):>4}")
    print(f"comparable          {len(comparable()):>4}")
    left_out = excluded()
    if not left_out.empty:
        print("\nexcluded:")
        print(left_out.to_string(index=False))


if __name__ == "__main__":
    main()
