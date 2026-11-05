"""What OpenStreetMap holds without a timetable: route relations, their stops, mapped stops.

* ``census`` -- per city, from the .pbf snapshot: route relations touching the
  boundary (as Overpass ``relation(area)`` counts them), the share that list at least
  one stop or platform, stop nodes mapped inside the boundary, and the share of them
  some route lists. Relations come from the city file (it holds every route relation
  touching the boundary, all members complete); stop nodes from the regional extract
  in one node pass per region, because a stop on no route is not in the city file.
  -> ``route_stops.csv``
* ``proxy`` -- the share of a city's route relations that list a stop, beside the
  discrepancy. -> ``proxy_without_schedule.csv``

    python pipeline/osm_relations.py census | proxy
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import shapely
from common import (
    ACCESSIBILITY_CSV,
    RESULTS_DIR,
    ROUTE_STOPS_CSV,
    boundary,
    cities,
    city_pbf_path,
    measurable,
    region_pbf_path,
    regions,
    restrict,
    setup,
)

from iduedu.constants.transport_specs import canonical_transport_type

OSM_ROUTE_VALUES = ("bus", "light_rail", "monorail", "share_taxi", "subway", "train", "tram", "trolleybus")
#: Route values the library reads as another mode (``OSM_ROUTE_ALIASES``): counted as tagged.
ALIASED_ROUTE_VALUES = ("light_rail", "share_taxi")
STOP_ROLES = {"stop", "stop_entry_only", "stop_exit_only", "platform", "platform_entry_only", "platform_exit_only"}
STOP_NODE_TAGS = (("highway", "bus_stop"), ("public_transport", "stop_position"), ("public_transport", "platform"),
                  ("railway", "station"), ("railway", "halt"), ("railway", "tram_stop"))  # fmt: skip
PROXY_CSV = RESULTS_DIR / "proxy_without_schedule.csv"


# --- census --------------------------------------------------------------------------


def relations_of(city_key: str) -> tuple[pd.DataFrame, set[int]]:
    """Route relations touching the boundary (one row each) and every node they list."""
    from iduedu.osm_pbf.reader import PbfSource, _selection_polygon, _Store

    source, store = PbfSource(city_pbf_path(city_key)), _Store()
    # pylint: disable=protected-access
    candidates = source._read_relations(store, tag_pairs=[("route", value) for value in OSM_ROUTE_VALUES])
    source._read_members(store, candidates)
    rows, listed = [], set()
    for rid in source._selected_relations(store, _selection_polygon(boundary(city_key)), candidates):
        members, tags = store.rels[rid]
        listed.update(ref for kind, ref, _ in members if kind == "n")
        stops = sum(1 for _, _, role in members if (role or "") in STOP_ROLES)
        rows.append({"relation": rid, "route": tags.get("route", ""),
                     "mode": canonical_transport_type(tags.get("route", "")), "stops": stops})  # fmt: skip
    return pd.DataFrame(rows, columns=["relation", "route", "mode", "stops"]), listed


def stop_nodes_by_city(region: str, keys: list[str]) -> dict[str, np.ndarray]:
    """Stop-tagged node ids inside each city's boundary, from one pass over the region."""
    import osmium

    ids, lons, lats = [], [], []
    processor = osmium.FileProcessor(str(region_pbf_path(region)), osmium.osm.NODE)
    processor.with_filter(osmium.filter.TagFilter(*STOP_NODE_TAGS))
    for node in processor:
        if node.location.valid():
            ids.append(node.id)
            lons.append(node.location.lon)
            lats.append(node.location.lat)
    ids, lons, lats = np.array(ids, dtype=np.int64), np.array(lons), np.array(lats)
    inside = {}
    for key in keys:
        polygon = boundary(key)
        minx, miny, maxx, maxy = polygon.bounds
        box = (lons >= minx) & (lons <= maxx) & (lats >= miny) & (lats <= maxy)
        hit = np.zeros(len(ids), dtype=bool)
        hit[box] = shapely.contains_xy(polygon, lons[box], lats[box])
        inside[key] = ids[hit]
    return inside


def census_city(city_key: str, stop_nodes: np.ndarray) -> dict:
    relations, listed = relations_of(city_key)
    on_route = int(np.isin(stop_nodes, np.fromiter(listed, np.int64, len(listed))).sum())
    row = {"city_key": city_key, "n_relations": len(relations), "stop_nodes_in_boundary": len(stop_nodes),
           "stop_nodes_on_a_route": on_route,
           "share_stops_on_a_route": round(on_route / len(stop_nodes), 4) if len(stop_nodes) else None}  # fmt: skip
    if relations.empty:
        return {**row, "status": "no relations"}
    row.update(status="ok", n_with_stops=int((relations["stops"] > 0).sum()), n_without_stops=int((relations["stops"] == 0).sum()),
               share_relations_with_stops=round(float((relations["stops"] > 0).mean()), 4), stops_total=int(relations["stops"].sum()))  # fmt: skip
    row.update({f"{value}_tagged": int((relations["route"] == value).sum()) for value in ALIASED_ROUTE_VALUES})
    for mode, group in relations.groupby("mode"):
        row.update({f"{mode}_relations": len(group), f"{mode}_with_stops": int((group["stops"] > 0).sum()),
                    f"{mode}_stops": int(group["stops"].sum())})  # fmt: skip
    return row


def census(logger, keys: list[str]) -> pd.DataFrame:
    by_region = defaultdict(list)
    for key in keys:
        if city_pbf_path(key).exists():
            by_region[regions().at[key, "region"]].append(key)
    # the stop nodes come from the regional extract; where it is absent, as in the deposit, a city keeps its row
    kept = pd.read_csv(ROUTE_STOPS_CSV, dtype=str, keep_default_na=False) if ROUTE_STOPS_CSV.exists() else None
    rows, without_extract = [], 0
    for index, (region, region_keys) in enumerate(sorted(by_region.items()), start=1):
        if not region_pbf_path(region).exists():
            if kept is not None:
                rows += kept[kept["city_key"].isin(region_keys)].to_dict("records")
            without_extract += 1
            continue
        stops = stop_nodes_by_city(region, region_keys)
        for key in region_keys:
            try:
                rows.append(census_city(key, stops[key]))
            except Exception as error:  # noqa: BLE001 - one unreadable city must not end the sweep
                rows.append({"city_key": key, "status": "failed", "reason": f"{type(error).__name__}: {error}"[:300]})
        logger.info("[%d/%d] %s: %d cities", index, len(by_region), region, len(region_keys))
    if without_extract:
        logger.info("%d regions without an extract: their cities keep their rows", without_extract)
    return pd.DataFrame(rows)


# --- proxy ---------------------------------------------------------------------------


def proxy_table() -> pd.DataFrame:
    access = pd.read_csv(ACCESSIBILITY_CSV)
    frame = pd.read_csv(ROUTE_STOPS_CSV).merge(
        access.loc[access["status"] == "ok", ["city_key", "ratio_15", "ratio_30", "ratio_45"]],
        on="city_key",
        how="left",
    )
    frame = frame[frame["city_key"].isin(measurable())]
    frame["relations_with_stops"] = frame["n_with_stops"] / frame["n_relations"].replace(0, np.nan)
    return frame[
        ["city_key", "n_relations", "n_with_stops", "relations_with_stops", "ratio_15", "ratio_30", "ratio_45"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["census", "proxy"])
    parser.add_argument("--cities", nargs="*")
    arguments = parser.parse_args()
    logger = setup("osm_relations")
    if arguments.step == "census":
        table = census(logger, restrict(cities().index, arguments.cities))
        table.to_csv(ROUTE_STOPS_CSV, index=False)
        logger.info("%s", table["status"].value_counts().to_dict())
    else:
        proxy_table().to_csv(PROXY_CSV, index=False)


if __name__ == "__main__":
    main()
