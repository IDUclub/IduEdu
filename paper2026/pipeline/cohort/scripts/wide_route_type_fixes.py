"""Route types a source codes by a scheme of its own (author, 19.10.2026).

Korea's national feed (KTDB, ``trs-kr_korea-ad1d68``), the only GTFS behind the five
Korean study cities, numbers its modes in its own order, shifted against the GTFS
reference. The route names leave no doubt:

====  ==================================================  ===========
code  routes behind it                                    GTFS code
====  ==================================================  ===========
0     city and village buses (24,581 routes, "마을버스")    3  bus
1     metro and light metro ("서울6호선", "의정부경전철")       1  subway
2     island ferries ("청산도-완도")                          4  ferry
3     intercity and express coaches ("제천-동대구")           200 coach
4     conventional rail (Saemaeul, Mugunghwa, commuter)     2  train
5     airport limousine buses ("공항출발")                   200 coach
6     high-speed rail (KTX, SRT)                            2  train
7     domestic flights ("김포-제주")                         dropped
====  ==================================================  ===========

Read as published, Korean city buses entered the study as trams, ferries as trains
and flights as funiculars. The fix is applied where a source is read for a city feed
(``wide_city_feed.read_feed``); ``python wide_route_type_fixes.py`` applies the same
rule to city feeds already on disk and writes ``route_type_fixes.csv``.
"""

import argparse
import logging
import zipfile

import pandas as pd
from wide_paths import CITY_MEMBERS_CSV, WIDE_DIR, city_feed_path

logger = logging.getLogger(__name__)

#: Per source: published ``route_type`` -> GTFS ``route_type``; ``None`` drops the route.
ROUTE_TYPE_FIXES: dict[str, dict[str, str | None]] = {
    "trs-kr_korea-ad1d68": {"0": "3", "1": "1", "2": "4", "3": "200", "4": "2", "5": "200", "6": "2", "7": None},
}

REPORT_CSV = WIDE_DIR / "route_type_fixes.csv"
#: Written into a fixed archive, one row per source fixed: the rules swap codes (2 and 4),
#: so a second pass would undo the first.
MARKER = "route_type_fixes.txt"


def _route_type(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str).str.strip()
    return text.str.replace(r"\.0$", "", regex=True)


def fix_tables(tables: dict[str, pd.DataFrame], feed_id: str, prefix: str = "") -> list[dict]:
    """Recode one source's routes in place; drop the routes coded ``None`` with their trips.

    ``prefix`` selects the source's routes in a merged city feed (``"<feed_id>:"``);
    empty, every route is the source's.
    """
    rules = ROUTE_TYPE_FIXES.get(feed_id)
    routes = tables.get("routes.txt")
    marker = tables.get(MARKER, pd.DataFrame({"feed_id": []}))
    if not rules or routes is None or routes.empty or feed_id in set(marker["feed_id"].astype(str)):
        return []
    tables[MARKER] = pd.concat([marker, pd.DataFrame({"feed_id": [feed_id]})], ignore_index=True)
    mine = routes["route_id"].astype(str).str.startswith(prefix) if prefix else pd.Series(True, index=routes.index)
    published = _route_type(routes["route_type"])
    report = [
        {"feed_id": feed_id, "published": code, "recoded": rules.get(code, code) or "dropped", "routes": int(n)}
        for code, n in published[mine].value_counts().sort_index().items()
    ]
    recoded = published.map(lambda code: rules.get(code, code))
    dropped_routes = set(routes.loc[mine & recoded.isna(), "route_id"].astype(str))
    routes = routes.copy()
    routes.loc[mine, "route_type"] = recoded[mine]
    tables["routes.txt"] = routes[~routes["route_id"].astype(str).isin(dropped_routes)]
    if dropped_routes and "trips.txt" in tables:
        trips = tables["trips.txt"]
        dropped_trips = set(trips.loc[trips["route_id"].astype(str).isin(dropped_routes), "trip_id"].astype(str))
        tables["trips.txt"] = trips[~trips["trip_id"].astype(str).isin(dropped_trips)]
        for name in ("stop_times.txt", "frequencies.txt"):
            if name in tables:
                tables[name] = tables[name][~tables[name]["trip_id"].astype(str).isin(dropped_trips)]
    return report


def fix_city_archives(cities: list[str] | None = None) -> pd.DataFrame:
    """Apply the rules to city feeds built before them; an archive already fixed is left as is."""
    from wide_city_feed import TABLES, write_archive
    from wide_profile import _read_member

    members = pd.read_csv(CITY_MEMBERS_CSV, dtype=str, keep_default_na=False)
    members = members[members["feed_id"].isin(ROUTE_TYPE_FIXES)]
    if cities:
        members = members[members["city_key"].isin(cities)]
    rows = []
    for city_key, group in members.groupby("city_key"):
        path = city_feed_path(city_key)
        if not path.exists():
            logger.warning("%s: no city feed on disk", city_key)
            continue
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            tables = {name: _read_member(archive, name) for name in TABLES + [MARKER] if name in names}
        tables = {name: frame for name, frame in tables.items() if frame is not None}
        changed = False
        for feed_id in group["feed_id"]:
            report = fix_tables(tables, feed_id, prefix=f"{feed_id}:")
            changed |= bool(report)
            rows += [{"city_key": city_key, **row} for row in report]
        if changed:
            write_archive(tables, path)
            logger.info("%s: recoded", city_key)
    report = pd.DataFrame(rows)
    if not report.empty:
        report.to_csv(REPORT_CSV, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cities", nargs="*")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    print(fix_city_archives(arguments.cities).to_string(index=False))


if __name__ == "__main__":
    main()
