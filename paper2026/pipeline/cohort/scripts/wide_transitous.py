"""Transitous as a catalogue: what it shares with the other three, and its feeds.

The other catalogues are the Mobility Database, the Transitland Atlas and
DigitalTransport4Africa. Transitous lists feeds from all over and refers to many of
them by those catalogues' identifiers, so this stage first measures what it shares
with them, then fetches the feeds it alone carries.

Steps, each resume-safe and run in this order (``--all`` runs the first four):

``--overlap``
    Every enabled static source Transitous lists, matched to the other catalogues
    by catalogue identifier and by normalised download URL. A Transitous feed (the
    sources it merges under one name, one mirror file) counts as shared if any of
    its sources is. Writes ``transitous_sources.csv``.
``--download``
    Every feed not shared is fetched from Transitous's own mirror, which republishes
    every feed it imports at a stable address -- the only uniform way to reach the
    sources it names by an identifier the public Mobility Database export does not
    carry (``tld-``, ``jbda-``). Same checks and SHA-256 log as ``wide_feeds``;
    ``transitous_downloads.csv``.
``--profile``
    ``wide_profile.profile`` on each archive, plus the agency names in
    ``agency.txt`` (overlap by operator); ``transitous_profile.csv``.
``--locate``
    Each feed's anchor as ``wide_resolve`` will take it (median stop, or densest
    cell for a regional feed) and the overlap summary,
    ``transitous_overlap_summary.csv``. Overlap by city is not decided here: once
    every feed has its city, ``wide_spread`` reports which cities draw on which
    catalogues.
``--select 0``
    Adds every profiled feed with at least one stop to ``selection.csv``; which
    cities are studied is decided later, for all catalogues alike, by ``wide_spread``.
"""

import argparse
import json
import logging
import re
import time
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1

import pandas as pd
from bench_common import append_row
from wide_catalogs import TRANSITOUS_MIRROR_URL, merged_catalog, normalise_url, transitous
from wide_feeds import FEED_CACHE, fetch
from wide_paths import PAPER_DIR, SELECTION_CSV, WIDE_DIR
from wide_profile import _read_member, profile

logger = logging.getLogger(__name__)

SOURCES_CSV = WIDE_DIR / "transitous_sources.csv"
DOWNLOADS_CSV = WIDE_DIR / "transitous_downloads.csv"
PROFILE_CSV = WIDE_DIR / "transitous_profile.csv"
LOCATED_CSV = WIDE_DIR / "transitous_located.csv"
SUMMARY_CSV = WIDE_DIR / "transitous_overlap_summary.csv"

CURRENT_CATALOGS = PAPER_DIR / "catalog_cache" / "transitous" / "current"

#: A feed with no stop is not a feed (brief, task 9); a regional feed is one whose
#: stops spread beyond ``wide_places.CITY_SPREAD_KM``.
STUB_STOPS = 1
CITY_SPREAD_KM = 60.0


def feed_id_for(mirror_file: str) -> str:
    """A file-system-safe id. Mirror names can be Japanese or carry spaces."""
    stem = mirror_file[: -len(".gtfs.zip")]
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    ascii_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", ascii_stem).strip("-")[:60]
    return f"trs-{ascii_stem}-{sha1(stem.encode('utf-8')).hexdigest()[:6]}"


# --- overlap ------------------------------------------------------------------


def _other_catalogues() -> pd.DataFrame:
    """The other three catalogues: the merged Mobility Database / Atlas catalogue,
    plus the DigitalTransport4Africa and manual records of the selection."""
    merged = merged_catalog()
    selection = pd.read_csv(SELECTION_CSV, keep_default_na=False)
    selection = selection[selection["catalog"] != "transitous"]
    extra = selection[~selection["feed_id"].isin(set(merged["feed_id"]))]
    return pd.concat([merged, extra.reindex(columns=merged.columns)], ignore_index=True)


def _current_catalog_urls(snapshot: str) -> tuple[dict, dict]:
    """Identifier -> URL in today's catalogues, for sources Transitous names by id."""
    mdb = pd.read_csv(CURRENT_CATALOGS / f"mobility_database_{snapshot}.csv", low_memory=False)
    mdb_url = dict(zip("mdb-" + mdb["mdb_source_id"].astype(str), mdb["urls.direct_download"].astype(str)))
    atlas_url = {}
    with zipfile.ZipFile(CURRENT_CATALOGS / f"transitland_atlas_{snapshot}.zip") as archive:
        for name in archive.namelist():
            if not name.endswith(".dmfr.json"):
                continue
            try:
                payload = json.loads(archive.read(name))
            except ValueError:
                continue
            for feed in payload.get("feeds", []):
                url = (feed.get("urls") or {}).get("static_current")
                if url:
                    atlas_url[str(feed.get("id"))] = url
    return mdb_url, atlas_url


def overlap() -> pd.DataFrame:
    sources = transitous()
    snapshot = sources["snapshot"].iloc[0]
    mdb_url, atlas_url = _current_catalog_urls(snapshot)
    frame = _other_catalogues()
    by_id = set(frame["feed_id"].astype(str))
    by_url = dict(zip(frame["url_key"].astype(str), frame["feed_id"].astype(str)))

    def resolved(row) -> str:
        if row.source_type in ("http", "ftp"):
            return row.source_url
        if row.source_type == "mobility-database":
            return mdb_url.get(row.mdb_ref, "")
        return atlas_url.get(row.atlas_ref, "")

    sources["resolved_url"] = [resolved(row) for row in sources.itertuples(index=False)]
    sources["url_key"] = sources["resolved_url"].map(lambda url: normalise_url(url) if url else "")
    sources["ref_id"] = sources["mdb_ref"].where(sources["mdb_ref"] != "", sources["atlas_ref"])

    matched_how, matched_id = [], []
    for row in sources.itertuples(index=False):
        if row.ref_id and row.ref_id in by_id:
            matched_how.append("catalogue id")
            matched_id.append(row.ref_id)
        elif row.url_key and row.url_key in by_url:
            matched_how.append("download url")
            matched_id.append(by_url[row.url_key])
        else:
            matched_how.append("")
            matched_id.append("")
    sources["shared_by"] = matched_how
    sources["shared_with_feed_id"] = matched_id
    sources["feed_id"] = sources["mirror_file"].map(feed_id_for)
    sources["ref_resolvable"] = (sources["source_type"].isin(["http", "ftp"])) | (sources["resolved_url"] != "")
    # Transitous merges sources sharing a name into one feed and one mirror file
    # (Translink-BOW is two Atlas entries). The feed is the unit downloaded and
    # compared; it counts as shared if any of its sources is.
    first_match = sources[sources["shared_by"] != ""].groupby("feed_id")["shared_by"].first()
    sources["feed_shared_by"] = sources["feed_id"].map(first_match).fillna("")
    sources["feed_n_sources"] = sources.groupby("feed_id")["feed_id"].transform("size")
    WIDE_DIR.mkdir(parents=True, exist_ok=True)
    sources.to_csv(SOURCES_CSV, index=False, encoding="utf-8")
    logger.info(
        "%d Transitous static sources, %d shared with the other catalogues",
        len(sources),
        (sources.shared_by != "").sum(),
    )
    return sources


# --- download -------------------------------------------------------------------


def _no_copy_in_run(sources: pd.DataFrame) -> pd.Series:
    """Per source row: whether no source of its feed has another catalog's copy in the run.

    A feed that shares a source with another catalog is skipped only when that
    catalog's copy is in the run. The run's selection took a few feeds per country,
    so most shared feeds -- New York, Chicago, Boston among them -- had no copy at all
    (741 of 839, found 14.10.2026).
    """
    selected = set(pd.read_csv(SELECTION_CSV, keep_default_na=False)["feed_id"].astype(str))
    selected -= set(sources["feed_id"].astype(str))  # the Transitous feeds themselves are no copy
    copied = sources["shared_with_feed_id"].map(lambda ids: any(i in selected for i in str(ids).split(";") if i))
    return ~copied.groupby(sources["feed_id"]).transform("any")


def download(workers: int = 3, retry_failed: bool = False) -> None:
    sources = pd.read_csv(SOURCES_CSV, keep_default_na=False)
    wanted = sources[_no_copy_in_run(sources) & (sources["mirror_href"] != "")]
    wanted = wanted.drop_duplicates(subset=["feed_id"])
    done = set()
    if DOWNLOADS_CSV.exists():
        log = pd.read_csv(DOWNLOADS_CSV).drop_duplicates(subset=["feed_id"], keep="last")
        for row in log.itertuples(index=False):
            if row.status == "ok" and (FEED_CACHE / f"{row.feed_id}.zip").exists():
                done.add(row.feed_id)
            elif not retry_failed:
                done.add(row.feed_id)
    pending = wanted[~wanted["feed_id"].isin(done)]
    logger.info("%d mirror archives to fetch (%d recorded)", len(pending), len(done))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch, row.feed_id, TRANSITOUS_MIRROR_URL + row.mirror_href): row
            for row in pending.itertuples(index=False)
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            row["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            append_row(DOWNLOADS_CSV, row)
            if index % 25 == 0 or row["status"] != "ok":
                logger.info(
                    "[%d/%d] %-8s %-40s %s",
                    index,
                    len(pending),
                    row["status"],
                    row["feed_id"][:40],
                    row["reason"] or f"{row['bytes'] / 1e6:.1f} MB",
                )


# --- profile --------------------------------------------------------------------


def _agencies(path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            agency = _read_member(archive, "agency.txt")
    except zipfile.BadZipFile:
        return ""
    if agency is None or "agency_name" not in agency:
        return ""
    return " | ".join(sorted(set(agency["agency_name"].dropna().astype(str).str.strip())))[:2000]


def profile_all() -> pd.DataFrame:
    log = pd.read_csv(DOWNLOADS_CSV).drop_duplicates(subset=["feed_id"], keep="last")
    stored = log[log["status"] == "ok"]["feed_id"].astype(str)
    rows = []
    for index, feed_id in enumerate(stored, start=1):
        path = FEED_CACHE / f"{feed_id}.zip"
        if not path.exists():
            continue
        row = profile(path)
        row["agencies"] = _agencies(path)
        rows.append(row)
        if index % 100 == 0:
            logger.info("profiled %d/%d", index, len(stored))
    frame = pd.DataFrame(rows)
    frame.to_csv(PROFILE_CSV, index=False, encoding="utf-8")
    return frame


# --- locate ---------------------------------------------------------------------


def _normalise_operator(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def locate() -> pd.DataFrame:
    from wide_selection import world_bank_countries

    sources = pd.read_csv(SOURCES_CSV, keep_default_na=False)
    prof = pd.read_csv(PROFILE_CSV, keep_default_na=False)
    prof = prof[prof["ok"].astype(str).str.lower().isin({"true", "1"})]
    feed_level = sources[_no_copy_in_run(sources)].drop_duplicates(subset=["feed_id"])
    feeds = feed_level.merge(prof, on="feed_id", how="inner")
    for column in ("n_stops", "spread_km", "centre_lat", "centre_lon"):
        feeds[column] = pd.to_numeric(feeds[column], errors="coerce")

    # Operator overlap: any agency in the archive named like a provider in the other catalogues.
    frame = merged_catalog()
    providers = set(frame["provider"].map(_normalise_operator)) - {""}
    feeds["operator_in_other_catalogues"] = feeds["agencies"].map(
        lambda names: any(_normalise_operator(n) in providers for n in str(names).split(" | ") if n.strip())
    )

    # Where a feed is anchored, as wide_resolve anchors it: the median stop for a
    # city-scale feed, the densest 0.05-degree cell for a regional one (the median of
    # a national feed is a point in the countryside).
    from wide_resolve import dense_core

    feeds["scale"] = feeds["spread_km"].map(lambda km: "city" if km <= CITY_SPREAD_KM else "regional")
    feeds["anchor_lat"], feeds["anchor_lon"] = feeds["centre_lat"], feeds["centre_lon"]
    for index in feeds.index[feeds["scale"] == "regional"]:
        core = dense_core(feeds.at[index, "feed_id"])
        if core:
            feeds.at[index, "anchor_lat"], feeds.at[index, "anchor_lon"] = core

    countries = world_bank_countries()
    feeds = feeds.merge(countries, on="country_code", how="left")
    feeds["stub"] = feeds["n_stops"] < STUB_STOPS
    # Every feed with a stop enters the run; regional feeds too -- wide_resolve carves a
    # city out of them -- and feeds in a city other catalogues also describe, which
    # wide_spread merges into one city.
    feeds["enters_run"] = ~feeds["stub"]
    feeds.to_csv(LOCATED_CSV, index=False, encoding="utf-8")
    _summaries(sources, feeds)
    return feeds


def _summaries(sources: pd.DataFrame, feeds: pd.DataFrame) -> None:
    unique = sources.drop_duplicates(subset=["feed_id"])
    not_shared = unique[unique["feed_shared_by"] == ""]
    rows = [
        ("static GTFS sources listed by Transitous", len(sources)),
        ("  in another catalogue by catalogue id", (sources["shared_by"] == "catalogue id").sum()),
        ("  in another catalogue by download URL", (sources["shared_by"] == "download url").sum()),
        ("  in no other catalogue", (sources["shared_by"] == "").sum()),
        ("Transitous feeds (sources merged by name, one mirror file each)", len(unique)),
        ("  with at least one source in another catalogue", (unique["feed_shared_by"] != "").sum()),
        ("  carried by Transitous alone", len(not_shared)),
        ("    of which with a mirror copy", (not_shared["mirror_href"] != "").sum()),
        ("    downloaded and profiled", len(feeds)),
        ("    operator named in another catalogue", int(feeds["operator_in_other_catalogues"].sum())),
        ("    regional or national (spread > 60 km)", int((feeds["scale"] == "regional").sum())),
        ("    no stop", int(feeds["stub"].sum())),
        ("    entering the run (at least one stop)", int(feeds["enters_run"].sum())),
    ]
    pd.DataFrame(rows, columns=["measure", "count"]).to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    for measure, count in rows:
        print(f"{measure:<66} {count:>6}")


# --- select ---------------------------------------------------------------------


def as_catalog(feeds: pd.DataFrame) -> pd.DataFrame:
    """Transitous feeds in the catalogue's own columns. The position is the median of
    the archive's stops, which no other catalogue row has; ``country_source`` says
    the country came from the Transitous file name."""
    out = pd.DataFrame(
        {
            "catalog": "transitous",
            "feed_id": feeds["feed_id"],
            "provider": feeds["agencies"].astype(str).str.split(" | ", regex=False).str[0].fillna(""),
            "name": feeds["transitous_name"],
            "country_code": feeds["country_code"],
            "country_source": "transitous_file",
            "subdivision": feeds["region_file"],
            "municipality": "",
            "url": TRANSITOUS_MIRROR_URL + feeds["mirror_href"],
            "license": feeds["license"],
            "lat": feeds["anchor_lat"] if "anchor_lat" in feeds else feeds["centre_lat"],
            "lon": feeds["anchor_lon"] if "anchor_lon" in feeds else feeds["centre_lon"],
            "country_name": feeds.get("country_name", ""),
            "region": feeds.get("region", "").fillna("Unknown").replace("", "Unknown"),
            "income_group": feeds.get("income_group", "").fillna("Unknown").replace("", "Unknown"),
        }
    )
    out["url_key"] = out["url"].map(normalise_url)
    return out


def select_new(target: int) -> pd.DataFrame:
    """Add Transitous feeds to the run's selection.

    Candidates are all profiled feeds with at least one stop (``enters_run``); with a
    TARGET, ``wide_selection.select`` takes turns over World Bank regions
    and, inside a region, over countries.
    Their archives are already in the run's feed cache, so their download rows are
    copied into ``downloads.csv`` and ``wide_profile`` picks them up from there.
    """
    from wide_paths import DOWNLOADS_CSV as RUN_DOWNLOADS_CSV
    from wide_selection import merge_into_selection, select

    feeds = pd.read_csv(LOCATED_CSV, keep_default_na=False)
    feeds = feeds[feeds["enters_run"].astype(str).str.lower().isin({"true", "1"})]
    for column in ("centre_lat", "centre_lon", "anchor_lat", "anchor_lon"):
        feeds[column] = pd.to_numeric(feeds[column], errors="coerce")
    frame = as_catalog(feeds.reset_index(drop=True))
    # target 0 = every candidate (author, 12.10.2026: collect all new cities first,
    # decide on a cap afterwards), with no regional quota or round robin.
    chosen = select(frame, target, unknown_share=0.0) if target else frame
    chosen = chosen.drop(columns=["selection_order", "_rank"], errors="ignore")
    before = len(pd.read_csv(SELECTION_CSV))
    added = merge_into_selection(chosen)

    log = pd.read_csv(DOWNLOADS_CSV).drop_duplicates(subset=["feed_id"], keep="last")
    present = set(pd.read_csv(RUN_DOWNLOADS_CSV)["feed_id"].astype(str)) if RUN_DOWNLOADS_CSV.exists() else set()
    for row in log[log["feed_id"].isin(set(chosen["feed_id"])) & ~log["feed_id"].isin(present)].to_dict("records"):
        append_row(RUN_DOWNLOADS_CSV, row)
    print(f"selection: {before} -> {before + added} feeds ({added} from Transitous)")
    print(chosen.groupby("region").size().to_string())
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", metavar="YYYYMMDD", help="download a new Transitous snapshot under this date")
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--locate", action="store_true")
    parser.add_argument("--all", action="store_true", help="overlap, download, profile, locate")
    parser.add_argument("--workers", type=int, default=3, help="parallel downloads from the mirror")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--select", type=int, metavar="TARGET", help="add TARGET new-city feeds to the run's selection.csv; 0 adds all"
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if arguments.refresh:
        from wide_catalogs import refresh_transitous

        refresh_transitous(arguments.refresh)
    if arguments.overlap or arguments.all:
        overlap()
    if arguments.download or arguments.all:
        download(arguments.workers, arguments.retry_failed)
    if arguments.profile or arguments.all:
        profile_all()
    if arguments.locate or arguments.all:
        locate()
    if arguments.select is not None:
        select_new(arguments.select)


if __name__ == "__main__":
    main()
