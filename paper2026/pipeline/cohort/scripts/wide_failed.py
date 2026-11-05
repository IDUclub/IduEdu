"""Which feeds could not be obtained, by country, region and income group.

A feed counts as failed when no attempt to download it gave a valid GTFS archive
(``downloads.csv``: ``failed`` -- no archive -- or ``invalid`` -- an archive without
the required tables). Its country is, in order: the country of its stops as
located in OSM (``country_sources.csv``), the catalog's country code, and for the
few rows that carry neither, the city the catalog names (``HAND_COUNTRY``). Region
and income group come from the World Bank list, with Taiwan added by hand as in
the city table.

Writes ``feed_outcomes.csv`` (one row per selected feed) and
``feed_outcomes_by_income.csv`` / ``feed_outcomes_by_region.csv``.
"""

import pandas as pd
from wide_paths import SELECTION_CSV, WIDE_DIR
from wide_selection import world_bank_countries

#: Rows with no country anywhere in the catalogs, named by their city: DT4A archives
#: carry only the repository (city) name, and one Transitland feed only a geohash.
HAND_COUNTRY = {
    "Abidjan": "CI",
    "Accra": "GH",
    "Addis Ababa": "ET",
    "Alexandria": "EG",
    "Bamako": "ML",
    "Cairo": "EG",
    "Cochabamba": "BO",
    "Douala": "CM",
    "Duitama": "CO",
    "Freetown": "SL",
    "Harare": "ZW",
    "Kampala": "UG",
    "Kigali": "RW",
    "Kinshasa": "CD",
    "Kumasi": "GH",
    "Lagos": "NG",
    "Lusaka": "ZM",
    "Mali": "ML",
    "Nairobi": "KE",
    "Santiago de los Caballeros": "DO",
    "Stellenbosch": "ZA",
    "Tetouan": "MA",
    "f-9mu-mts": "US",  # San Diego MTS, Transitland geohash 9mu
    "f-9wvk-mountainmetropolitantransit": "US",  # Colorado Springs
    "f-djwq-chathamareatransit": "US",  # Savannah, Georgia
    "f-djwq-chathamareatransit~ferry": "US",
    # stops located at Tokelau: placeholder coordinates; the operator is the
    # Mar Chiquita partido, Buenos Aires province (Transitland geohash 6d6)
    "f-6d6-marchiquitasrl": "AR",
}


def world_bank() -> pd.DataFrame:
    countries = world_bank_countries()[["country_code", "country_name", "region", "income_group"]].copy()
    for column in ("country_name", "region", "income_group"):
        countries[column] = countries[column].str.strip()
    taiwan = {
        "country_code": "TW",
        "country_name": "Taiwan",
        "region": "East Asia & Pacific",
        "income_group": "High income",
    }
    return pd.concat([countries[countries["country_code"] != "TW"], pd.DataFrame([taiwan])], ignore_index=True)


def outcomes() -> pd.DataFrame:
    selection = pd.read_csv(SELECTION_CSV, keep_default_na=False)
    downloads = pd.read_csv(WIDE_DIR / "downloads.csv", keep_default_na=False)
    ok = set(downloads.loc[downloads["status"] == "ok", "feed_id"])
    last = downloads.drop_duplicates(subset=["feed_id"], keep="last").set_index("feed_id")
    located = pd.read_csv(WIDE_DIR / "country_sources.csv", keep_default_na=False).set_index("feed_id")

    frame = selection[["feed_id", "catalog", "provider", "name", "country_code"]].copy()
    frame["outcome"] = [
        "ok" if feed in ok else (last.at[feed, "status"] if feed in last.index else "not attempted")
        for feed in frame["feed_id"]
    ]
    frame["reason"] = [
        "" if feed in ok or feed not in last.index else last.at[feed, "reason"] for feed in frame["feed_id"]
    ]
    country, source = [], []
    for row in frame.itertuples():
        stops = located.at[row.feed_id, "country"] if row.feed_id in located.index else ""
        hand = HAND_COUNTRY.get(row.feed_id) or (HAND_COUNTRY.get(row.provider) if row.catalog == "dt4a" else None)
        if row.feed_id in HAND_COUNTRY:
            country.append(hand), source.append("hand (city named in catalog)")
        elif stops:
            country.append(stops), source.append("stops (OSM)")
        elif row.country_code:
            country.append(row.country_code), source.append("catalog")
        elif hand:
            country.append(hand), source.append("hand (city named in catalog)")
        else:
            country.append(""), source.append("")
    frame["country_code"] = country
    frame["country_from"] = source
    frame = frame.merge(world_bank(), on="country_code", how="left")
    for column in ("region", "income_group"):
        frame[column] = frame[column].fillna("Unknown")
    frame["country_name"] = frame["country_name"].fillna("")
    return frame


def summary(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    attempted = frame[frame["outcome"] != "not attempted"]
    table = attempted.groupby(by).agg(
        feeds=("feed_id", "size"),
        ok=("outcome", lambda s: int((s == "ok").sum())),
        failed=("outcome", lambda s: int((s == "failed").sum())),
        invalid=("outcome", lambda s: int((s == "invalid").sum())),
    )
    table["not_obtained_share"] = ((table["failed"] + table["invalid"]) / table["feeds"]).round(3)
    return table.reset_index()


def main() -> None:
    frame = outcomes()
    frame.to_csv(WIDE_DIR / "feed_outcomes.csv", index=False)
    income = summary(frame, "income_group")
    region = summary(frame, "region")
    income.to_csv(WIDE_DIR / "feed_outcomes_by_income.csv", index=False)
    region.to_csv(WIDE_DIR / "feed_outcomes_by_region.csv", index=False)
    bad = frame[frame["outcome"].isin(["failed", "invalid"])]
    print(
        f"not obtained: {len(bad)}; with country {int((bad['country_code'] != '').sum())}; "
        f"with income group {int((bad['income_group'] != 'Unknown').sum())}"
    )
    print(bad["country_from"].value_counts().to_string())
    print(income.to_string(index=False))
    print(region.to_string(index=False))


if __name__ == "__main__":
    main()
