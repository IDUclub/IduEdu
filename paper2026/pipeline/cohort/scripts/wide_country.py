"""Where each feed's country comes from: the catalogue, the URL, or the feed itself.

Brief, task 9. Before download the country of a record is the catalogue's field,
the host's country-code domain where that is empty, and nothing otherwise -- 65 of
the 229 records drawn from the Mobility Database, the Atlas and DT4A had none, and
selection kept them apart in an "Unknown" bucket. Once a feed is downloaded its country is
not a guess -- ``wide_places`` asks OSM which areas contain the feed's anchor, and
the admin level 2 among them carries ``ISO3166-1``. This module puts the three
side by side for every record of the run and says where they disagree.

Writes ``country_sources.csv`` (one row per record) and ``country_sources_summary.csv``.
"""

import pandas as pd
from wide_paths import PLACES_REVIEW_CSV, SELECTION_CSV, WIDE_DIR

DETAIL_CSV = WIDE_DIR / "country_sources.csv"
SUMMARY_CSV = WIDE_DIR / "country_sources_summary.csv"

SOURCE_NAMES = {
    "catalog": "catalogue field",
    "url_tld": "country-code domain of the URL",
    "transitous_file": "Transitous feed file (feeds/<country>.json)",
    "": "none before download",
}


def country_from_stack(stack_json: str) -> str:
    """The deepest ISO 3166-1 code among the areas containing the feed's anchor.

    ``wide_places`` keeps the first code it meets, which is the country level, so
    Hong Kong came back as CN. Hong Kong, Macau, Puerto Rico and similar carry their
    own ``ISO3166-1`` one level deeper, and the World Bank lists them as economies
    of their own; the deepest code is the one the regions table can use.
    """
    import json

    try:
        stack = json.loads(stack_json) if stack_json else []
    except ValueError:
        return ""
    codes = [(int(item.get("level", 0)), str(item.get("country", "")).upper()) for item in stack]
    codes = [entry for entry in codes if entry[1]]
    if not codes:
        return ""
    # Réunion, Guadeloupe, Martinique, French Guiana and Mayotte carry their own
    # ISO code in OSM but are part of France for the World Bank, which then has no
    # region for them; New Caledonia, Hong Kong, Macau, Puerto Rico are economies of
    # their own. So: the deepest code the World Bank lists, else the deepest at all.
    listed = [entry for entry in codes if entry[1] in _world_bank_codes()]
    return max(listed or codes)[1]


_WB_CODES: set[str] | None = None


def _world_bank_codes() -> set[str]:
    global _WB_CODES
    if _WB_CODES is None:
        from wide_selection import world_bank_countries

        _WB_CODES = set(world_bank_countries()["country_code"])
    return _WB_CODES


def feed_countries() -> pd.DataFrame:
    """feed_id -> OSM country, from the places table."""
    places = pd.read_csv(PLACES_REVIEW_CSV, keep_default_na=False, encoding="utf-8-sig")
    places["osm_country"] = places["admin_stack_json"].map(country_from_stack)
    return places[["feed_id", "osm_country"]]


def build() -> pd.DataFrame:
    selection = pd.read_csv(SELECTION_CSV, keep_default_na=False)
    places = feed_countries()
    table = selection[["feed_id", "catalog", "country_code", "country_source"]].merge(places, on="feed_id", how="left")
    table["osm_country"] = table["osm_country"].fillna("").astype(str).str.upper().str.strip()
    table["country_code"] = table["country_code"].astype(str).str.upper().str.strip()
    table["located"] = table["feed_id"].isin(set(places["feed_id"]))
    table["country"] = table["osm_country"].where(table["osm_country"] != "", table["country_code"])
    table["country_from"] = "feed coordinates (OSM)"
    table.loc[table["osm_country"] == "", "country_from"] = table["country_source"].map(SOURCE_NAMES)
    table.loc[(table["osm_country"] == "") & (table["country_code"] == ""), "country_from"] = "unknown"
    both = (table["osm_country"] != "") & (table["country_code"] != "")
    table["agrees"] = pd.NA
    table.loc[both, "agrees"] = table.loc[both, "osm_country"] == table.loc[both, "country_code"]
    return table


def summarise(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    located = table[table["located"]]
    rows = [
        ("records in the run's selection", len(table)),
        ("  located by OSM (downloaded and profiled)", len(located)),
        ("before download: country from the catalogue field", (table["country_source"] == "catalog").sum()),
        ("before download: from the URL's country-code domain", (table["country_source"] == "url_tld").sum()),
        ("before download: from the Transitous feed file", (table["country_source"] == "transitous_file").sum()),
        ("before download: none", (table["country_code"] == "").sum()),
        ("after download: country from the feed's coordinates (OSM)", (located["osm_country"] != "").sum()),
        ("  of which had no country before", ((located["osm_country"] != "") & (located["country_code"] == "")).sum()),
        ("  where both exist, OSM agrees with the earlier country", int(located["agrees"].eq(True).sum())),
        ("  where both exist, OSM disagrees", int(located["agrees"].eq(False).sum())),
        ("after download: still unknown", (located["country"] == "").sum()),
    ]
    summary = pd.DataFrame(rows, columns=["measure", "count"])
    by_catalog = (
        table.groupby(["catalog", "country_from"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["catalog", "count"])
    )
    return summary, by_catalog


def main() -> None:
    table = build()
    table.to_csv(DETAIL_CSV, index=False, encoding="utf-8")
    summary, by_catalog = summarise(table)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nby catalogue:\n" + by_catalog.to_string(index=False))
    disagree = table[table["agrees"].eq(True) == False].dropna(subset=["agrees"])  # noqa: E712
    if len(disagree):
        print(
            "\ndisagreements:\n"
            + disagree[["feed_id", "catalog", "country_code", "country_source", "osm_country"]].to_string(index=False)
        )


if __name__ == "__main__":
    main()


def boundary_facts() -> pd.DataFrame:
    """Per boundary relation: its OSM ``population`` tag and the country it lies in.

    Read from the run's ``is_in`` answers (no request): every answer lists the areas
    around one point, so a relation's country is the deepest World Bank economy among
    the areas of an answer that contains it. Used for a city's own population (not
    its nearest large settlement's: Hoboken had been given New York's) and for the
    country of a carved city (Geneva's densest cell lies in France, its canton does not).
    """
    import glob
    import json

    from wide_paths import OVERPASS_CACHE_DIR

    offset = 3_600_000_000
    rows: dict[int, dict] = {}
    for path in glob.glob(str(OVERPASS_CACHE_DIR / "is_in_*.json")):
        try:
            elements = json.loads(open(path, encoding="utf-8").read()).get("elements", [])
        except (OSError, ValueError):
            continue
        stack = []
        for element in elements:
            tags = element.get("tags", {}) or {}
            level = str(tags.get("admin_level", ""))
            stack.append({"level": int(level) if level.isdigit() else 0, "country": tags.get("ISO3166-1", "") or ""})
        country = country_from_stack(json.dumps(stack))
        for element in elements:
            area_id = int(element.get("id", 0))
            if area_id <= offset:
                continue
            tags = element.get("tags", {}) or {}
            entry = rows.setdefault(area_id - offset, {"population": None, "country": ""})
            if tags.get("population") and entry["population"] is None:
                entry["population"] = pd.to_numeric(
                    str(tags["population"]).replace(" ", "").replace(",", ""), errors="coerce"
                )
            if country and not entry["country"]:
                entry["country"] = country
    frame = pd.DataFrame.from_dict(rows, orient="index").rename_axis("osm_id").reset_index()
    frame["osm_id"] = frame["osm_id"].astype(str)
    return frame
