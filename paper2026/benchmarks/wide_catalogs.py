"""Public GTFS catalogs, merged into one sampling frame.

Two catalogs are used, and they barely overlap:

* **Mobility Database** -- a CSV export, no key required. Carries country,
  subdivision, municipality, licence and a bounding box, but those fields are
  often empty: requiring a non-empty ``municipality`` *and* a bounding box cuts
  the frame from 1531 usable feeds to under 900 and from 72 countries to 34.
  That filter measures how well the catalog is filled in, not whether a country
  publishes schedules, so it is deliberately **not** applied here. Geography for
  the analysis is taken later from the centroid of ``stops.txt``.
* **Transitland Atlas** -- a public git repository of DMFR files, no key
  required. It has no country or city columns at all, but its feed identifiers
  embed a geohash (``f-<geohash>-<slug>``) that is good enough to assign a feed
  to a region before anything is downloaded.

Run this module directly to refresh the caches and print the funnel.
"""

import argparse
import io
import json
import logging
import zipfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests
from wide_paths import CATALOG_CACHE_DIR as CATALOG_CACHE

logger = logging.getLogger(__name__)

MOBILITY_DATABASE_URL = "https://storage.googleapis.com/storage/v1/b/mdb-csv/o/sources.csv?alt=media"
TRANSITLAND_ATLAS_URL = "https://github.com/transitland/transitland-atlas/archive/refs/heads/master.zip"

USER_AGENT = "IduEdu-research/1.0 (+https://github.com/DDonnyy/IduEdu)"

#: Statuses that mean the feed is not worth trying.
DEAD_STATUSES = {"deprecated", "inactive", "development"}

COLUMNS = [
    "catalog",
    "feed_id",
    "provider",
    "name",
    "country_code",
    "country_source",
    "subdivision",
    "municipality",
    "url",
    "url_key",
    "license",
    "lat",
    "lon",
    "min_lat",
    "max_lat",
    "min_lon",
    "max_lon",
]

_GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"

#: Country-code top level domains that do not match their ISO 3166-1 alpha-2 code,
#: plus the ones that carry no country at all and must not be read as one.
_TLD_OVERRIDES = {"uk": "GB", "eu": "", "su": "", "tv": "", "me": "", "io": "", "ai": "", "co": ""}

#: Two-letter geohash cells span more than a thousand kilometres, and strings such as
#: ``f-us-flixbus`` or ``f-de-flixbus`` decode to a valid but meaningless cell. Only
#: identifiers with a longer geohash are trusted.
MIN_GEOHASH_LENGTH = 3


def country_from_host(url: str) -> str:
    """ISO 3166-1 alpha-2 guessed from the host's country-code TLD, or an empty string."""
    try:
        host = urlsplit(str(url)).netloc.lower().split(":")[0]
    except ValueError:
        return ""
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    if len(tld) != 2 or not tld.isalpha():
        return ""
    return _TLD_OVERRIDES.get(tld, tld.upper())


def decode_geohash(value: str) -> tuple[float, float] | None:
    """Return the centre of a geohash cell, or ``None`` if the string is not one."""
    if not value or len(value) < MIN_GEOHASH_LENGTH:
        return None
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    even = True
    for char in value.lower():
        index = _GEOHASH_ALPHABET.find(char)
        if index < 0:
            return None
        for mask in (16, 8, 4, 2, 1):
            target = lon_range if even else lat_range
            middle = (target[0] + target[1]) / 2
            if index & mask:
                target[0] = middle
            else:
                target[1] = middle
            even = not even
    return (lat_range[0] + lat_range[1]) / 2, (lon_range[0] + lon_range[1]) / 2


def normalise_url(url: str) -> str:
    """Key used to recognise the same feed listed by both catalogs."""
    try:
        parts = urlsplit(str(url).strip())
    except ValueError:
        return str(url).strip().lower()
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, "")).lower()


def _download(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        logger.debug("using cached %s", destination.name)
        return destination
    logger.info("downloading %s", url)
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=180)
    response.raise_for_status()
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(destination)
    return destination


def mobility_database(force: bool = False) -> pd.DataFrame:
    """Usable static GTFS feeds from the Mobility Database, normalised."""
    path = _download(MOBILITY_DATABASE_URL, CATALOG_CACHE / "mobility_database.csv", force)
    raw = pd.read_csv(path, low_memory=False)

    frame = raw[raw["data_type"].astype(str).str.lower() == "gtfs"].copy()
    status = frame["status"].astype(str).str.lower()
    frame = frame[~status.isin(DEAD_STATUSES)]
    frame = frame[frame["urls.authentication_type"].fillna(0).astype(float) == 0]
    frame = frame[frame["urls.direct_download"].notna()]

    out = pd.DataFrame(
        {
            "catalog": "mobility_database",
            "feed_id": "mdb-" + frame["mdb_source_id"].astype(str),
            "provider": frame["provider"].fillna(""),
            "name": frame["name"].fillna(""),
            "country_code": frame["location.country_code"].fillna(""),
            "subdivision": frame["location.subdivision_name"].fillna(""),
            "municipality": frame["location.municipality"].fillna(""),
            "url": frame["urls.direct_download"].astype(str),
            "license": frame["urls.license"].fillna(""),
            "min_lat": pd.to_numeric(frame["location.bounding_box.minimum_latitude"], errors="coerce"),
            "max_lat": pd.to_numeric(frame["location.bounding_box.maximum_latitude"], errors="coerce"),
            "min_lon": pd.to_numeric(frame["location.bounding_box.minimum_longitude"], errors="coerce"),
            "max_lon": pd.to_numeric(frame["location.bounding_box.maximum_longitude"], errors="coerce"),
        }
    )
    out["lat"] = (out["min_lat"] + out["max_lat"]) / 2
    out["lon"] = (out["min_lon"] + out["max_lon"]) / 2
    out["url_key"] = out["url"].map(normalise_url)

    out["country_source"] = out["country_code"].map(lambda code: "catalog" if code else "")
    missing = out["country_code"] == ""
    out.loc[missing, "country_code"] = out.loc[missing, "url"].map(country_from_host)
    out.loc[missing & (out["country_code"] != ""), "country_source"] = "url_tld"
    return out[COLUMNS].reset_index(drop=True)


def transitland_atlas(force: bool = False) -> pd.DataFrame:
    """Usable static GTFS feeds from the Transitland Atlas, normalised."""
    path = _download(TRANSITLAND_ATLAS_URL, CATALOG_CACHE / "transitland_atlas.zip", force)
    archive = zipfile.ZipFile(io.BytesIO(path.read_bytes()))

    records: list[dict] = []
    unparsable = 0
    for name in archive.namelist():
        if not name.endswith(".dmfr.json"):
            continue
        try:
            payload = json.loads(archive.read(name))
        except (json.JSONDecodeError, UnicodeDecodeError):
            unparsable += 1
            continue
        for feed in payload.get("feeds", []):
            if str(feed.get("spec", "")).lower() != "gtfs":
                continue
            if feed.get("authorization"):
                continue
            url = (feed.get("urls") or {}).get("static_current")
            if not url:
                continue
            feed_id = str(feed.get("id", ""))
            operators = feed.get("operators") or []
            parts = feed_id.split("-")
            centre = decode_geohash(parts[1]) if len(parts) > 2 else None
            country = country_from_host(url)
            records.append(
                {
                    "catalog": "transitland_atlas",
                    "feed_id": feed_id,
                    "provider": (operators[0].get("name", "") if operators else ""),
                    "name": feed.get("name", "") or (operators[0].get("name", "") if operators else ""),
                    "country_code": country,
                    "country_source": "url_tld" if country else "",
                    "subdivision": "",
                    "municipality": "",
                    "url": str(url),
                    "license": json.dumps(feed.get("license") or {}, ensure_ascii=False),
                    "lat": centre[0] if centre else None,
                    "lon": centre[1] if centre else None,
                    "min_lat": None,
                    "max_lat": None,
                    "min_lon": None,
                    "max_lon": None,
                }
            )
    if unparsable:
        logger.warning("%d DMFR files could not be parsed and were skipped", unparsable)

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return pd.DataFrame(columns=COLUMNS)
    out["url_key"] = out["url"].map(normalise_url)
    return out.reindex(columns=COLUMNS).reset_index(drop=True)


def merged_catalog(force: bool = False) -> pd.DataFrame:
    """Union of both catalogs, keyed by normalised URL.

    Mobility Database rows win on conflicts because they carry country and city;
    the Atlas contributes what the Mobility Database has never heard of.
    """
    mdb = mobility_database(force)
    atlas = transitland_atlas(force)
    combined = pd.concat([mdb, atlas], ignore_index=True)
    combined = combined.drop_duplicates(subset=["url_key"], keep="first").reset_index(drop=True)
    # Catalog values arrive padded often enough that "US" and "US " become two countries.
    for column in ("country_code", "subdivision", "municipality"):
        combined[column] = combined[column].fillna("").astype(str).str.strip()
    combined["country_code"] = combined["country_code"].str.upper()
    return combined


def _funnel() -> None:
    mdb = mobility_database()
    atlas = transitland_atlas()
    merged = merged_catalog()
    overlap = len(mdb) + len(atlas) - len(merged)

    print(f"Mobility Database, usable static GTFS   {len(mdb):>5}")
    print(f"  with municipality                     {(mdb['municipality'] != '').sum():>5}")
    print(f"  with a bounding box                   {mdb['lat'].notna().sum():>5}")
    print(f"  distinct countries                    {mdb['country_code'].replace('', pd.NA).nunique():>5}")
    print(f"Transitland Atlas, usable static GTFS   {len(atlas):>5}")
    print(f"  with a decodable geohash              {atlas['lat'].notna().sum():>5}")
    print(f"Union by normalised URL                 {len(merged):>5}   (overlap {overlap})")
    print(f"  of which locatable at all             {merged['lat'].notna().sum():>5}")
    print(f"  with a country (catalog or TLD)       {(merged['country_code'] != '').sum():>5}")
    print(f"  distinct countries in the union       {merged['country_code'].replace('', pd.NA).nunique():>5}")
    print("  country signal: " + str(merged["country_source"].replace("", "none").value_counts().to_dict()))

    counts = merged["country_code"].replace("", pd.NA).value_counts()
    print("\nUnion, top countries: " + ", ".join(f"{code} {count}" for code, count in counts.head(12).items()))
    print(f"countries with a single feed: {(counts == 1).sum()}")
    print(
        "outside US/CA/EU-ish top: " + ", ".join(f"{code} {count}" for code, count in counts.head(40).tail(20).items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download the catalogs instead of using the cache")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if arguments.refresh:
        mobility_database(force=True)
        transitland_atlas(force=True)
    _funnel()


if __name__ == "__main__":
    main()
