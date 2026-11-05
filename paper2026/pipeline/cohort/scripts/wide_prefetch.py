"""Fetch the candidate boundaries for every feed in a few batched Overpass requests.

``wide_resolve`` tries a feed's administrative units one by one and fetches each
polygon with its own request (``iduedu.get_4326_boundary``). Over ~1,500 feeds that
is thousands of requests against a server that refuses half of them, and the
refusals, not the data, set the pace. Overpass returns any number of relations to
one query, ``rel(id:a,b,c); out geom;``, so this stage asks for the boundaries in
batches and writes each relation into the library's cache under exactly the key a
single-relation request would have used. ``wide_resolve`` then runs unchanged and
finds them there; the result is byte-for-byte what it would have fetched itself.

Candidates are taken from ``places_review.csv`` the way ``choose_boundary`` takes
them: relations in the feed's administrative stack at a usable level, or whose
name matches the settlement. Regional feeds are re-described at their densest
cell by ``wide_resolve`` and get their candidates there; they are not prefetched.
"""

import argparse
import json
import logging

import pandas as pd
from wide_paths import CITIES_CSV, PLACES_REVIEW_CSV, use_paper_cache
from wide_places import CITY_SPREAD_KM
from wide_resolve import MIN_USABLE_ADMIN_LEVEL, name_matches

from iduedu import config
from iduedu.overpass.cache import cache_load, cache_save
from iduedu.overpass.downloaders import _overpass_request

logger = logging.getLogger(__name__)

BATCH = 10


def single_key(osm_id: int) -> str:
    """The cache key ``get_boundary_by_osm_id`` uses -- reproduced verbatim."""
    header = config.overpass_header
    overpass_query = f"""
                    {header}
                    (relation({osm_id}););
                    out geom;
    """
    return f"{config.overpass_url}\nGET\n{overpass_query}"


def candidates() -> list[int]:
    places = pd.read_csv(PLACES_REVIEW_CSV, keep_default_na=False, encoding="utf-8-sig")
    ids: list[int] = []
    for record in places.to_dict("records"):
        if float(record.get("spread_km") or 0) > CITY_SPREAD_KM:
            continue
        try:
            stack = json.loads(record.get("admin_stack_json") or "[]")
        except ValueError:
            continue
        for item in stack:
            osm = str(item.get("osm", ""))
            if not osm.startswith("relation/"):
                continue
            if item.get("level", 0) >= MIN_USABLE_ADMIN_LEVEL or name_matches(record.get("settlement", ""), item):
                ids.append(int(osm.split("/")[1]))
    return list(dict.fromkeys(ids))


def prefetch(ids: list[int], batch: int = BATCH) -> dict[str, int]:
    missing = [osm_id for osm_id in ids if cache_load("boundary", single_key(osm_id)) is None]
    logger.info("%d candidate relations, %d not cached, %d requests", len(ids), len(missing), -(-len(missing) // batch))
    done = failed = 0
    for start in range(0, len(missing), batch):
        chunk = missing[start : start + batch]
        query = f"{config.overpass_header}rel(id:{','.join(map(str, chunk))});out geom;"
        try:
            payload = _overpass_request("POST", config.overpass_url, data={"data": query}).json()
        except Exception as exc:  # noqa: BLE001 - leave the chunk to wide_resolve's own requests
            failed += len(chunk)
            logger.warning("batch of %d failed (%s); resolve will fetch them singly", len(chunk), exc)
            continue
        by_id = {element.get("id"): element for element in payload.get("elements", [])}
        for osm_id in chunk:
            element = by_id.get(osm_id)
            if element is None:
                # not cached: an empty answer here had stood in for Prague's boundary
                failed += 1
                continue
            entry = {key: value for key, value in payload.items() if key != "elements"}
            entry["elements"] = [element]
            cache_save("boundary", single_key(osm_id), entry)
            done += 1
        logger.info("[%d/%d] cached", done, len(missing))
    return {"candidates": len(ids), "missing": len(missing), "cached_now": done, "left_to_resolve": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--dry-run", action="store_true", help="only count what is missing")
    parser.add_argument("--cities", action="store_true", help="the boundaries of the cities in cities.csv instead")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    use_paper_cache()
    if arguments.cities:
        osm = pd.to_numeric(pd.read_csv(CITIES_CSV, keep_default_na=False)["osm_id"], errors="coerce").dropna()
        ids = sorted({int(value) for value in osm})
    else:
        ids = candidates()
    if arguments.dry_run:
        missing = [i for i in ids if cache_load("boundary", single_key(i)) is None]
        print(f"{len(ids)} candidate relations, {len(missing)} not cached")
        return
    print(prefetch(ids, arguments.batch))


if __name__ == "__main__":
    main()
