"""Do the route relations OSM holds actually carry stops?

The sanity check flags cities where OSM has route relations and our transit graph
has few nodes. Two explanations fit, and they point in opposite directions: the
builder is losing routes, or the routes carry no stops to find. The second is
common in OpenStreetMap -- a route relation may hold only the roadway ways, with
no member tagged as a stop or a platform -- and it is a finding about the data
rather than about the pipeline, because a route without stops cannot enter an
accessibility calculation from either source.

    python wide_routestops.py --cities madison sibenik
"""

import argparse
import logging

import pandas as pd
from bench_common import already_done, append_row
from wide_city_feed import parse_osm_id
from wide_cohort import measurable
from wide_paths import CITIES_CSV, WIDE_DIR, use_paper_cache
from wide_places import RELATION_AREA_OFFSET
from wide_sanity import OSM_ROUTE_VALUES

from iduedu import config
from iduedu.overpass.downloaders import _overpass_request

logger = logging.getLogger(__name__)

REPORT_CSV = WIDE_DIR / "route_stops.csv"

#: Member roles the public-transport schemes use for a boarding point. Both the
#: modern ``public_transport`` scheme and the older one are covered.
STOP_ROLES = {"stop", "stop_entry_only", "stop_exit_only", "platform", "platform_entry_only", "platform_exit_only"}


def relations_with_members(osm_id: int) -> list[dict]:
    values = "|".join(sorted(OSM_ROUTE_VALUES))
    query = (
        f"[out:json][timeout:{config.timeout}];"
        f"area({RELATION_AREA_OFFSET + int(osm_id)})->.a;"
        f'relation(area.a)["route"~"^({values})$"];out body;'
    )
    response = _overpass_request("POST", config.overpass_url, data={"data": query})
    return response.json().get("elements", [])


#: What a transit stop looks like in OSM, under both tagging schemes.
STOP_NODE_FILTERS = (
    '["highway"="bus_stop"]',
    '["public_transport"~"^(stop_position|platform)$"]',
    '["railway"~"^(station|halt|tram_stop)$"]',
)


def count_stop_nodes(osm_id: int) -> int:
    """Stop nodes inside the boundary itself.

    Counting the members of a route relation overstates the city: a regional route
    with one member inside the boundary drags its whole line into the count, and
    Madison's 77 relations carry 2,344 stop members of which most lie elsewhere.
    This counts the nodes a builder could place inside the boundary, which is the
    number our graph should be near.
    """
    area = f"area({RELATION_AREA_OFFSET + int(osm_id)})->.a;"
    unions = "".join(f"node(area.a){filt};" for filt in STOP_NODE_FILTERS)
    query = f"[out:json][timeout:{config.timeout}];{area}({unions});out ids;"
    response = _overpass_request("POST", config.overpass_url, data={"data": query})
    return len(response.json().get("elements", []))


def measure_city(city_key: str, osm_id: int) -> dict:
    rows = []
    for element in relations_with_members(osm_id):
        tags = element.get("tags", {}) or {}
        members = element.get("members", []) or []
        stops = sum(1 for member in members if (member.get("role") or "") in STOP_ROLES)
        rows.append({"route": tags.get("route", ""), "members": len(members), "stops": stops})
    frame = pd.DataFrame(rows)
    row = {"city_key": city_key, "n_relations": len(frame), "stop_nodes_in_boundary": count_stop_nodes(osm_id)}
    if frame.empty:
        row["status"] = "no relations"
        return row
    row["status"] = "ok"
    row["n_with_stops"] = int((frame["stops"] > 0).sum())
    row["n_without_stops"] = int((frame["stops"] == 0).sum())
    row["stops_total"] = int(frame["stops"].sum())
    for mode in sorted(frame["route"].unique()):
        sub = frame.loc[frame["route"].eq(mode)]
        row[f"{mode}_relations"] = len(sub)
        row[f"{mode}_with_stops"] = int((sub["stops"] > 0).sum())
        row[f"{mode}_stops"] = int(sub["stops"].sum())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="*", help="default is the whole cohort")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    use_paper_cache()

    cities = pd.read_csv(CITIES_CSV).fillna("").set_index("city_key")
    keys = arguments.cities or measurable()
    done = already_done(REPORT_CSV) if REPORT_CSV.exists() else set()
    keys = [key for key in keys if key not in done]
    logger.info("%d cities to measure, %d already recorded", len(keys), len(done))

    rows = []
    for key in keys:
        osm_id = parse_osm_id(cities.loc[key, "osm_id"]) if key in cities.index else None
        if osm_id is None:
            logger.warning("%s: no boundary", key)
            continue
        try:
            row = measure_city(key, int(osm_id))
        except Exception as error:  # noqa: BLE001 - one unreachable city must not end the sweep
            logger.warning("%s: %s", key, error)
            continue
        rows.append(row)
        append_row(REPORT_CSV, row)
        logger.info(
            "%-14s %3d relations | %3d carry stops | %5d stop members | %5d stop nodes inside the boundary",
            key,
            row.get("n_relations", 0),
            row.get("n_with_stops", 0),
            row.get("stops_total", 0),
            row.get("stop_nodes_in_boundary", 0),
        )

    if not rows:
        raise SystemExit("nothing measured; the report was left untouched")
    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwritten: {REPORT_CSV}")


if __name__ == "__main__":
    main()
