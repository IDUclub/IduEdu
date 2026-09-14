"""Cities whose feed never arrived: does OpenStreetMap have their transit anyway?

The sample is bounded by what the catalogues could deliver. Forty-four selected
archives never became usable data -- a catalogue entry serving an HTML page, a
host that stopped resolving, a zip without ``stops.txt`` -- and every one of them
is a city the study cannot speak about.

The question this module answers is whether those cities are missing from the
study because nobody mapped them or because nobody published a schedule for
them. If OSM holds route relations in a city whose feed could not be retrieved,
the limit is publication, and the completeness figures elsewhere in the paper are
measured on the subset of the world that publishes.

Names, not stops, are all a failed download leaves behind, so each city is looked
up by name inside its country and the answer is recorded with the boundary that
was actually used. Cities the lookup cannot place are reported as such rather
than dropped: an unplaced city is a limit of this check, not evidence of absence.

    python wide_unpublished.py            # what it would query
    python wide_unpublished.py --run
"""

import argparse
import logging

import pandas as pd
from bench_common import sweep
from wide_city_feed import parse_osm_id
from wide_paths import CITIES_CSV, DOWNLOADS_CSV, SELECTION_CSV, WIDE_DIR, use_paper_cache
from wide_sanity import OSM_ROUTE_VALUES

from iduedu import config
from iduedu.overpass.downloaders import _overpass_request

logger = logging.getLogger(__name__)

REPORT_CSV = WIDE_DIR / "unpublished_cities.csv"

#: Levels a city can sit at, matching the resolver used for the main sample.
CITY_ADMIN_LEVELS = "4|5|6|7|8|9|10"


def failed_entries() -> pd.DataFrame:
    """Catalogue rows whose archive never became usable data, with a place name."""
    downloads = pd.read_csv(DOWNLOADS_CSV).drop_duplicates(subset=["feed_id"], keep="last")
    selection = pd.read_csv(SELECTION_CSV).drop_duplicates(subset=["feed_id"], keep="last")
    failed = downloads.loc[~downloads["status"].eq("ok"), ["feed_id", "status", "reason"]]
    frame = failed.merge(selection, on="feed_id", how="left")
    frame["place"] = frame["municipality"].fillna("").astype(str).str.strip()
    named = frame.loc[frame["place"].str.len() > 1].copy()
    logger.info(
        "%d entries failed, %d of them name a municipality the lookup can try",
        len(frame),
        len(named),
    )
    return named


def _boundary_area(place: str, country_code: str) -> dict | None:
    """The smallest administrative area with this name inside this country."""
    country = (country_code or "").strip().upper()
    filter_country = f'["ISO3166-1"="{country}"][admin_level=2]' if country else ""
    query = (
        f"[out:json][timeout:{config.timeout}];"
        + (f"area{filter_country}->.c;" if filter_country else "")
        + f'relation["boundary"="administrative"]["admin_level"~"^({CITY_ADMIN_LEVELS})$"]'
        + f'["name"="{place}"]'
        + (("(area.c);") if filter_country else ";")
        + "out tags;"
    )
    response = _overpass_request("POST", config.overpass_url, data={"data": query})
    elements = response.json().get("elements", [])
    if not elements:
        return None
    # The deepest level is the most city-like of the matches.
    best = max(elements, key=lambda element: int(element.get("tags", {}).get("admin_level", 0) or 0))
    return {"osm_id": best["id"], "admin_level": best.get("tags", {}).get("admin_level", "")}


def _route_relations(osm_id: int) -> dict[str, int]:
    """Transit route relations inside the boundary, counted straight from Overpass."""
    values = "|".join(sorted(OSM_ROUTE_VALUES))
    query = (
        f"[out:json][timeout:{config.timeout}];"
        f"area({3_600_000_000 + int(osm_id)})->.a;"
        f'relation(area.a)["route"~"^({values})$"];out tags;'
    )
    response = _overpass_request("POST", config.overpass_url, data={"data": query})
    counts: dict[str, int] = {}
    for element in response.json().get("elements", []):
        route = (element.get("tags", {}) or {}).get("route", "")
        counts[route] = counts.get(route, 0) + 1
    return counts


def sampled_boundaries() -> set[int]:
    """Boundary ids already in the study, so a second archive for a city we have
    is not mistaken for a city we lack."""
    cities = pd.read_csv(CITIES_CSV).fillna("")
    ids: set[int] = set()
    for value in cities["osm_id"]:
        parsed = parse_osm_id(value)
        if parsed is not None:
            ids.add(int(parsed))
    return ids


def measure(feed_id: str, row: pd.Series, in_sample: set[int]) -> dict:
    out = {
        "feed_id": feed_id,
        "place": row["place"],
        "country_code": row.get("country_code", ""),
        "download_status": row["status"],
        "download_reason": str(row["reason"])[:80],
    }
    area = _boundary_area(row["place"], str(row.get("country_code", "")))
    if area is None:
        out["status"] = "not_placed"
        return out
    out.update({"osm_id": area["osm_id"], "admin_level": area["admin_level"]})
    if int(area["osm_id"]) in in_sample:
        # A failed archive for a city another feed already covers says nothing
        # about cities the study is missing.
        out["status"] = "already_in_sample"
        return out
    counts = _route_relations(area["osm_id"])
    out["n_routes_osm"] = sum(counts.values())
    out["routes_by_mode"] = ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))
    out["status"] = "ok"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="query Overpass; without it, only report the plan")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    entries = failed_entries()
    if arguments.limit:
        entries = entries.head(arguments.limit)
    if not arguments.run:
        print(entries[["feed_id", "place", "country_code", "status", "reason"]].to_string(index=False))
        print(f"\n{len(entries)} cities would be checked; pass --run to query Overpass")
        return

    use_paper_cache()
    in_sample = sampled_boundaries()
    lookup = entries.set_index("feed_id")
    sweep(
        [str(key) for key in lookup.index],
        lambda feed_id: measure(feed_id, lookup.loc[feed_id], in_sample),
        REPORT_CSV,
        force=arguments.force,
        logger=logger,
        key_column="feed_id",
        describe=lambda rows: f"{rows[0].get('status')} routes {rows[0].get('n_routes_osm', '-')}",
    )

    report = pd.read_csv(REPORT_CSV).drop_duplicates(subset=["feed_id"], keep="last")
    print("\n" + report["status"].value_counts().to_string())
    placed = report.loc[report["status"].eq("ok")]
    with_transit = placed.loc[placed["n_routes_osm"].fillna(0) > 0]
    print(
        f"\nplaced and outside the sample: {len(placed)}; "
        f"{len(with_transit)} of them have transit routes in OSM without a usable feed"
    )
    if not with_transit.empty:
        print(
            with_transit.nlargest(15, "n_routes_osm")[
                ["place", "country_code", "n_routes_osm", "download_reason"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
