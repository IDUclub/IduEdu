"""Is a two-node OSM graph a finding or a defect?

Douala's feed describes 3025 stops and 740 routes; its OSM graph has two nodes.
Kinshasa: 4282 against 22. Either OSM genuinely holds almost nothing about
transit in these cities -- which is the strongest observation this study can
make -- or the graph builder lost something on the way, which would be the
second time an aggregate was believed without checking the edges.

So the count is taken twice, from different places:

* how many route relations Overpass returns inside the boundary, before any
  parsing;
* how many survive into the graph.

A large gap between them is a defect of ours. Agreement means the city really is
empty in OSM, and that goes in the paper.
"""

import argparse
import logging

import pandas as pd
from wide_city_feed import parse_osm_id
from wide_paths import CITIES_CSV, OSM_GRAPHS_CSV
from wide_paths import OSM_SANITY_CSV as REPORT_CSV
from wide_places import RELATION_AREA_OFFSET, use_benchmark_cache

from iduedu import config, get_4326_boundary
from iduedu.overpass.downloaders import _overpass_request

logger = logging.getLogger(__name__)

#: Relation types the builder asks OSM for, in OSM's own vocabulary.
OSM_ROUTE_VALUES = {"bus", "subway", "tram", "trolleybus", "train", "light_rail", "monorail", "share_taxi"}


def _count(query: str) -> dict[str, int]:
    response = _overpass_request("POST", config.overpass_url, data={"data": query})
    counts: dict[str, int] = {}
    for element in response.json().get("elements", []):
        route = (element.get("tags", {}) or {}).get("route", "")
        counts[route] = counts.get(route, 0) + 1
    return counts


def count_relations(polygon) -> dict[str, int]:
    """Route relations in the boundary's bounding box: an upper bound, no more."""
    bounds = polygon.bounds
    bbox = f"{bounds[1]},{bounds[0]},{bounds[3]},{bounds[2]}"
    values = "|".join(sorted(OSM_ROUTE_VALUES))
    return _count(f'[out:json][timeout:{config.timeout}];relation({bbox})["route"~"^({values})$"];out tags;')


def count_relations_in_boundary(osm_id: int) -> dict[str, int]:
    """Route relations with a member inside the boundary itself.

    The bounding box answers one question only -- whether OSM is empty -- because a
    box around an irregular city contains a great deal that the city does not. A
    route counted there may run past the city without entering it, and 26 cities
    were flagged as "check the builder" on exactly that basis. Counting inside the
    administrative area is the comparison that means something.
    """
    values = "|".join(sorted(OSM_ROUTE_VALUES))
    query = (
        f"[out:json][timeout:{config.timeout}];"
        f"area({RELATION_AREA_OFFSET + int(osm_id)})->.a;"
        f'relation(area.a)["route"~"^({values})$"];out tags;'
    )
    return _count(query)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nodes", type=int, default=150, help="check cities whose graph is smaller than this")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    use_benchmark_cache()

    graphs = pd.read_csv(OSM_GRAPHS_CSV).drop_duplicates(subset=["city_key"], keep="last")
    cities = pd.read_csv(CITIES_CSV).fillna("").set_index("city_key")
    suspect = graphs[(graphs["status"] == "empty") | (graphs["n_nodes"].fillna(0) < arguments.max_nodes)]
    logger.info("checking %d cities", len(suspect))

    rows: list[dict] = []
    failures: list[str] = []
    for _, graph in suspect.iterrows():
        key = str(graph["city_key"])
        if key not in cities.index:
            continue
        osm_id = parse_osm_id(cities.loc[key, "osm_id"])
        if osm_id is None:
            continue
        try:
            counts = count_relations(get_4326_boundary(osm_id=osm_id))
            inside = count_relations_in_boundary(osm_id)
        except Exception as exc:  # noqa: BLE001 - one unreachable city must not end the check
            logger.warning("%s: %s", key, exc)
            failures.append(key)
            continue
        # The bbox is an upper bound and settles emptiness; the area count is what
        # decides whether a small graph means a sparse city or a lossy extraction.
        total = sum(counts.values())
        total_inside = sum(inside.values())
        nodes = float(graph.get("n_nodes", 0) or 0)
        if total_inside == 0:
            verdict = "OSM is empty"
        elif nodes >= 2 * total_inside:
            verdict = "graph matches OSM"
        else:
            verdict = "check the builder"
        rows.append(
            {
                "city_key": key,
                "city_name": graph.get("city_name", ""),
                "graph_nodes": graph.get("n_nodes", 0),
                "relations_in_bbox": total,
                "relations_in_boundary": total_inside,
                "by_route": ";".join(f"{name}:{value}" for name, value in sorted(inside.items())),
                "verdict": verdict,
            }
        )
        logger.info(
            "%-26s graph %6s nodes | %4d in bbox | %4d inside the boundary -> %s",
            key[:26],
            graph.get("n_nodes", 0),
            total,
            total_inside,
            verdict,
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        # A run where every city raised is a broken run, not a clean sheet. Writing
        # it out would replace a verified report with nothing, which is how the
        # previous report was lost to a missing import.
        raise SystemExit(
            f"every one of the {len(failures)} cities failed; the report was left untouched. "
            f"First failures: {failures[:5]}"
        )
    frame.to_csv(REPORT_CSV, index=False, encoding="utf-8")
    print(f"\n{frame['verdict'].value_counts().to_dict() if not frame.empty else 'nothing to check'}")
    if not frame.empty:
        print(frame.sort_values("relations_in_boundary", ascending=False).to_string(index=False))
    print(f"\nwritten: {REPORT_CSV}")


if __name__ == "__main__":
    main()
