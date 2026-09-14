"""Sampling frame for the wide tier: which feeds we try to download.

The unit of selection is a **feed**, not a city. A city is often described by
several feeds -- New York needs seven borough archives, and many cities split by
operator -- and picking one archive per city is exactly what produced the
"partial reference" defect in the previous run, where a metro-only archive was
compared against an OSM graph containing every operator. Cities are therefore
assembled *after* download, by clustering feeds on the centroid of their
``stops.txt``, and the number of feeds that had to be merged into one city is
kept as an indicator in its own right: it measures how fragmented a city's open
data is, which is part of what this study is about.

Selection maximises geographic and institutional spread rather than sampling
proportionally: proportional sampling of the catalogs would return a study of
Japanese and American operators. Regions take turns, then countries within a
region take turns, so countries represented by a single feed are never crowded
out by countries represented by a thousand.

Run this module directly to write ``results/wide_tier/selection.csv``.
"""

import argparse
import json
import logging

import pandas as pd
import requests
from wide_catalogs import CATALOG_CACHE, USER_AGENT, merged_catalog
from wide_paths import SELECTION_CSV
from wide_paths import WIDE_DIR as SELECTION_DIR

logger = logging.getLogger(__name__)

WORLD_BANK_URL = "https://api.worldbank.org/v2/country?format=json&per_page=400"

DEFAULT_TARGET = 140


def merge_into_selection(frame: pd.DataFrame) -> int:
    """Add feeds to the sampling frame, skipping any already in it.

    Two later stages extend the frame -- ``wide_dt4a`` with the African and Latin
    American repositories, ``wide_complete`` with the other feeds serving a city
    already chosen -- and both had grown their own copy of this. Selection order
    continues from the end so the frame stays a record of what was picked when.
    """
    if frame.empty:
        return 0
    selection = pd.read_csv(SELECTION_CSV)
    fresh = frame[~frame["feed_id"].isin(set(selection["feed_id"].astype(str)))].copy()
    if fresh.empty:
        return 0
    fresh["selection_order"] = range(len(selection) + 1, len(selection) + len(fresh) + 1)
    combined = pd.concat([selection, fresh.reindex(columns=selection.columns)], ignore_index=True)
    combined.to_csv(SELECTION_CSV, index=False, encoding="utf-8")
    return len(fresh)


#: Feeds are ordered inside a country by how much we already know about them:
#: a Mobility Database row naming a municipality is likely to be city scale and
#: comes with a licence, whereas an Atlas row may be a national or intercity feed.
def _feed_rank(row: pd.Series) -> tuple:
    has_municipality = 0 if str(row.get("municipality", "")).strip() else 1
    from_catalog = 0 if row.get("catalog") == "mobility_database" else 1
    located = 0 if pd.notna(row.get("lat")) else 1
    return has_municipality, from_catalog, located, str(row.get("feed_id", ""))


def world_bank_countries(force: bool = False) -> pd.DataFrame:
    """ISO-2 country code to World Bank region and income group.

    Region and income are the covariates the analysis groups by, so they are
    taken from an authoritative source rather than typed by hand -- the previous
    run found four errors in a hand-written income column.
    """
    path = CATALOG_CACHE / "world_bank_countries.json"
    if force or not path.exists():
        logger.info("downloading World Bank country list")
        response = requests.get(WORLD_BANK_URL, headers={"User-Agent": USER_AGENT}, timeout=120)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    records = []
    for entry in rows:
        region = (entry.get("region") or {}).get("value", "")
        if not region or region == "Aggregates":  # the list also contains groupings such as "World"
            continue
        records.append(
            {
                "country_code": str(entry.get("iso2Code", "")).strip().upper(),
                "country_name": entry.get("name", ""),
                "region": region,
                "income_group": (entry.get("incomeLevel") or {}).get("value", ""),
            }
        )
    return pd.DataFrame.from_records(records).drop_duplicates(subset=["country_code"])


def build_frame(force: bool = False) -> pd.DataFrame:
    """Merged catalog joined to region and income, ready for selection."""
    catalog = merged_catalog(force)
    countries = world_bank_countries(force)
    frame = catalog.merge(countries, on="country_code", how="left")
    frame["region"] = frame["region"].fillna("")
    frame["income_group"] = frame["income_group"].fillna("")
    frame.loc[frame["country_code"] == "", "region"] = "Unknown"
    frame.loc[frame["region"] == "", "region"] = "Unknown"
    frame.loc[frame["income_group"] == "", "income_group"] = "Unknown"
    frame["country_name"] = frame["country_name"].fillna("")
    return frame


#: Feeds whose country cannot be guessed before download form a single large bucket
#: (roughly a third of the union). Letting it take a region's turn would hand it an
#: eighth of the sample on the strength of a missing field, so its share is fixed.
UNKNOWN_REGION = "Unknown"
UNKNOWN_SHARE = 0.1


def select(
    frame: pd.DataFrame,
    target: int = DEFAULT_TARGET,
    per_country_cap: int | None = None,
    unknown_share: float = UNKNOWN_SHARE,
) -> pd.DataFrame:
    """Round-robin over regions, then over countries, until ``target`` feeds are chosen.

    The cap is a safety net rather than the mechanism: the round robin already
    keeps any single country from dominating, because a country only gets a
    second feed once every other country in its region has had a first.
    """
    ranked = frame.copy()
    ranked["_rank"] = [_feed_rank(row) for _, row in ranked.iterrows()]
    ranked = ranked.sort_values("_rank", kind="stable")

    # A country with few feeds must not be crowded out, so countries take turns
    # ordered by how little they offer.
    country_sizes = ranked.groupby("country_code", dropna=False).size()

    queues: dict[str, dict[str, list[int]]] = {}
    for region, region_rows in ranked.groupby("region", sort=True):
        by_country: dict[str, list[int]] = {}
        for country, country_rows in region_rows.groupby("country_code", sort=True):
            by_country[country] = list(country_rows.index)
        queues[region] = dict(sorted(by_country.items(), key=lambda item: (country_sizes.get(item[0], 0), item[0])))

    chosen: list[int] = []
    taken_per_country: dict[str, int] = {}
    regions = sorted(queues)
    exhausted: set[str] = set()
    unknown_budget = int(round(target * unknown_share))
    unknown_taken = 0

    while len(chosen) < target and len(exhausted) < len(regions):
        progressed = False
        for region in regions:
            if len(chosen) >= target or region in exhausted:
                continue
            if region == UNKNOWN_REGION and unknown_taken >= unknown_budget:
                exhausted.add(region)
                continue
            countries = queues[region]
            picked = False
            for country, indices in countries.items():
                if not indices:
                    continue
                if per_country_cap is not None and taken_per_country.get(country, 0) >= per_country_cap:
                    continue
                chosen.append(indices.pop(0))
                taken_per_country[country] = taken_per_country.get(country, 0) + 1
                if region == UNKNOWN_REGION:
                    unknown_taken += 1
                picked = progressed = True
                break
            if not picked:
                exhausted.add(region)
            else:
                # Move the country that has just been served to the back of its region's queue.
                queues[region] = {**{k: v for k, v in countries.items() if k != country}, country: countries[country]}
        if not progressed:
            break

    selection = frame.loc[chosen].copy()
    selection.insert(0, "selection_order", range(1, len(selection) + 1))
    return selection.reset_index(drop=True)


def _report(frame: pd.DataFrame, selection: pd.DataFrame) -> None:
    print(f"frame: {len(frame)} feeds, {frame['country_code'].replace('', pd.NA).nunique()} countries")
    print(f"selected: {len(selection)} feeds, {selection['country_code'].replace('', pd.NA).nunique()} countries\n")
    print("by region:")
    for region, count in selection["region"].value_counts().items():
        available = int((frame["region"] == region).sum())
        countries = selection.loc[selection["region"] == region, "country_code"].nunique()
        print(f"  {region:<30} {count:>4}  ({countries} countries, {available} available)")
    print("\nby income group:")
    for group, count in selection["income_group"].value_counts().items():
        print(f"  {group:<30} {count:>4}")
    print("\nby catalog: " + str(selection["catalog"].value_counts().to_dict()))
    top = selection["country_code"].value_counts()
    print(
        f"largest country share: {top.iloc[0]} feeds ({top.index[0]}), "
        f"countries with a single feed: {int((top == 1).sum())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="how many feeds to select")
    parser.add_argument("--per-country-cap", type=int, default=None, help="hard cap of feeds per country")
    parser.add_argument("--refresh", action="store_true", help="re-download catalogs and the World Bank list")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frame = build_frame(arguments.refresh)
    selection = select(frame, arguments.target, arguments.per_country_cap)

    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    destination = SELECTION_DIR / "selection.csv"
    selection.drop(columns=["_rank"], errors="ignore").to_csv(destination, index=False, encoding="utf-8")
    _report(frame, selection)
    print(f"\nwritten: {destination}")


if __name__ == "__main__":
    main()
