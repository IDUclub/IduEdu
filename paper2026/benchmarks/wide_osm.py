"""Build the OSM side of the comparison: public transport, walking, intermodal.

Nothing here runs by accident. Without ``--run`` the module only prints what it
would fetch, because a full sweep is hours of Overpass traffic against a public
service.

Four failures from the previous run are designed out rather than commented on:

* **Modes are taken from the union of a city's feeds, never from one of them.**
  New York's first archive is the subway, and asking OSM for ``subway`` alone
  produced a graph with no buses that was then compared against 253 bus routes.
* **The mode set is part of the cache file name.** Otherwise a later, wider
  request silently reuses the narrower graph already on disk.
* **Requesting a mode the registry does not know raises**, and the exception
  takes the whole city with it -- that is how five cities, New York and Tokyo
  among them, were lost when ``train`` was requested with a registry that has no
  train. The registry is chosen from the modes actually requested.
* **Modes OSM cannot supply are recorded as unavailable, not as zero.** A ferry
  route missing from an OSM graph we never asked for a ferry about is not
  evidence about OSM.

Cities come from ``results/wide_tier/cities.csv`` (produced once feeds are
resolved to boundaries): ``city_key``, ``city_name``, ``osm_id``, ``gtfs_modes``.
"""

import argparse
import json
import logging
import time

import pandas as pd
from bench_common import append_row
from wide_city_feed import parse_osm_id
from wide_paths import (
    ALL_OSM_MODES,
    CITIES_CSV,
    OSM_GRAPH_DIR,
    OSM_GRAPHS_CSV,
    PAPER_DIR,
    graph_metadata_path,
    osm_graph_path,
    registry_fingerprint,
)
from wide_places import use_benchmark_cache

from iduedu import (
    DEFAULT_REGISTRY,
    DEFAULT_REGISTRY_W_TRAIN,
    UrbanGraph,
    get_intermodal_graph,
    get_public_transport_graph,
    get_walk_graph,
    write_urban_graph,
)

logger = logging.getLogger(__name__)

#: GTFS and the OSM builder disagree on two names. The deep tier had this mapping;
#: the wide tier did not, so commuter rail was never requested from OSM at all.
MODE_ALIASES = {"rail": "train", "metro": "subway", "trolley": "trolleybus", "light_rail": "tram"}

#: Everything the OSM public-transport builder can be asked for, which is exactly
#: what the widest registry knows.
FETCHABLE_MODES = set(DEFAULT_REGISTRY_W_TRAIN.list_types())


def normalise_modes(gtfs_modes: str | list[str]) -> tuple[list[str], list[str]]:
    """Split a city's GTFS modes into what OSM can be asked for and what it cannot.

    Returns ``(requestable, unavailable)``. The second list matters: those modes
    must be excluded from the comparison rather than counted as missing from OSM.
    """
    if isinstance(gtfs_modes, str):
        raw = [part.split(":")[0] for part in gtfs_modes.split(";") if part.strip()]
    else:
        raw = list(gtfs_modes)
    mapped = {MODE_ALIASES.get(mode.strip().lower(), mode.strip().lower()) for mode in raw if str(mode).strip()}
    requestable = sorted(mapped & FETCHABLE_MODES)
    unavailable = sorted(mapped - FETCHABLE_MODES)
    return requestable, unavailable


def registry_for(modes: list[str]):
    """The registry must contain every requested mode or the build raises."""
    return DEFAULT_REGISTRY_W_TRAIN if "train" in modes else DEFAULT_REGISTRY


def _summarise(graph: UrbanGraph) -> dict:
    edges = graph.edges_gdf
    nodes = graph.nodes_gdf
    summary = {"n_nodes": len(nodes), "n_edges": len(edges)}
    if not edges.empty and "type" in edges.columns:
        counts = edges["type"].value_counts().to_dict()
        summary["edge_types"] = ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))
    if not nodes.empty and "route" in nodes.columns:
        routes = nodes["route"].dropna()
        summary["n_routes"] = int(routes.map(lambda value: str(value)).nunique())
    return summary


def _fingerprint_on_disk(destination) -> str | None:
    """What the registry looked like when this graph was written, if it says."""
    meta = graph_metadata_path(destination)
    if not meta.exists():
        return None
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("registry_fingerprint")
    except (OSError, ValueError):
        return None


def _write_fingerprint(destination, fingerprint: str, modes: list[str]) -> None:
    graph_metadata_path(destination).write_text(
        json.dumps(
            {
                "registry_fingerprint": fingerprint,
                "modes_requested": modes,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_city(city: pd.Series, kind: str, force: bool = False) -> dict:
    """Build one graph for one city and return the manifest row."""
    city_key = str(city["city_key"])
    feed_modes, unavailable = normalise_modes(city.get("gtfs_modes", ""))
    modes = ALL_OSM_MODES
    destination = osm_graph_path(city_key, kind, modes)
    row = {
        "city_key": city_key,
        "city_name": city.get("city_name", ""),
        "kind": kind,
        "osm_id": city.get("osm_id", ""),
        "modes_requested": "+".join(modes),
        # Kept so the analysis can intersect the two sides; see ALL_OSM_MODES.
        "modes_in_feed": "+".join(feed_modes),
        "modes_unavailable_in_osm": "+".join(unavailable),
        "path": str(destination.relative_to(PAPER_DIR)),
        "status": "skipped",
        "reason": "",
        "elapsed_s": 0.0,
    }

    # A pedestrian graph holds no transit constants, so nothing about the registry
    # can make it stale; only the transit and intermodal graphs are fingerprinted.
    fingerprint = registry_fingerprint(registry_for(modes)) if kind != "walk" else None
    if destination.exists() and not force:
        if fingerprint is None or _fingerprint_on_disk(destination) == fingerprint:
            row["reason"] = "already built"
            return row
        logger.info("%s: constants changed since this graph was built, rebuilding", city_key)

    # The column holds blanks, so pandas types it as float and ids arrive as
    # "19910275.0"; an isdigit check rejects every one of them.
    osm_id = parse_osm_id(city.get("osm_id", ""))
    if osm_id is None:
        row["status"] = "failed"
        row["reason"] = "no boundary osm_id"
        return row

    started = time.monotonic()
    try:
        if kind == "pt":
            graph = get_public_transport_graph(
                osm_id=osm_id,
                transport_types=modes,
                clip_by_territory=True,
                transport_registry=registry_for(modes),
            )
        elif kind == "walk":
            graph = get_walk_graph(osm_id=osm_id, clip_by_territory=True, keep_largest_subgraph=False)
        elif kind == "intermodal":
            graph = get_intermodal_graph(
                osm_id=osm_id,
                clip_by_territory=True,
                keep_largest_subgraph=True,
                walk_kwargs={"keep_largest_subgraph": False},
                pt_kwargs={"transport_types": modes, "transport_registry": registry_for(modes)},
            )
        else:
            raise ValueError(f"unknown kind {kind!r}")
    except Exception as exc:  # noqa: BLE001 - one city must not stop the sweep
        row["status"] = "failed"
        row["reason"] = f"{type(exc).__name__}: {exc}"[:300]
        row["elapsed_s"] = round(time.monotonic() - started, 1)
        return row

    row["elapsed_s"] = round(time.monotonic() - started, 1)
    if graph.nodes_gdf.empty:
        row["status"] = "empty"
        row["reason"] = "OSM returned no objects for these modes"
        return row

    OSM_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    write_urban_graph(graph, destination)
    if fingerprint is not None:
        _write_fingerprint(destination, fingerprint, modes)
    row.update(_summarise(graph))
    row["status"] = "ok"
    return row


def plan(cities: pd.DataFrame, kind: str) -> pd.DataFrame:
    """What a run would do, without touching the network."""
    rows = []
    fingerprint = registry_fingerprint(registry_for(ALL_OSM_MODES)) if kind != "walk" else None
    for _, city in cities.iterrows():
        feed_modes, unavailable = normalise_modes(city.get("gtfs_modes", ""))
        # The same path the builder writes to. Naming it from the feed's own modes
        # made the plan report nothing as cached while 137 graphs sat on disk.
        destination = osm_graph_path(str(city["city_key"]), kind, ALL_OSM_MODES)
        on_disk = destination.exists()
        stale = on_disk and fingerprint is not None and _fingerprint_on_disk(destination) != fingerprint
        rows.append(
            {
                "city_key": city["city_key"],
                "city_name": city.get("city_name", ""),
                "osm_id": city.get("osm_id", ""),
                "modes_requested": "+".join(ALL_OSM_MODES),
                "modes_in_feed": "+".join(feed_modes) or "-",
                "modes_unavailable_in_osm": "+".join(unavailable) or "-",
                "cached": on_disk and not stale,
                "stale": stale,
            }
        )
    return pd.DataFrame(rows)


def run(cities: pd.DataFrame, kind: str, force: bool = False) -> pd.DataFrame:
    use_benchmark_cache()
    rows = []
    for index, (_, city) in enumerate(cities.iterrows(), start=1):
        row = build_city(city, kind, force)
        if row["status"] != "skipped":
            append_row(OSM_GRAPHS_CSV, row)
        rows.append(row)
        logger.info(
            "[%d/%d] %-10s %-24s %-22s %s",
            index,
            len(cities),
            row["status"],
            str(row["city_key"])[:24],
            row["modes_requested"],
            row["reason"] or f"{row.get('n_nodes', 0)} nodes / {row.get('n_edges', 0)} edges",
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["pt", "walk", "intermodal"], default="pt")
    parser.add_argument("--cities", nargs="*", default=None, help="only these city keys")
    parser.add_argument("--run", action="store_true", help="actually download; without it only the plan is printed")
    parser.add_argument("--force", action="store_true", help="rebuild graphs that already exist")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if not CITIES_CSV.exists():
        raise SystemExit(f"{CITIES_CSV} does not exist yet — resolve feeds to cities first (wide_places.py)")
    cities = pd.read_csv(CITIES_CSV).fillna("")
    if arguments.cities:
        cities = cities[cities["city_key"].astype(str).isin(arguments.cities)]

    if not arguments.run:
        table = plan(cities, arguments.kind)
        print(table.to_string(index=False))
        stale = int(table["stale"].sum()) if "stale" in table else 0
        summary = f"{len(table)} cities, {int(table['cached'].sum())} already cached"
        if stale:
            summary += f", {stale} built with different constants and due a rebuild"
        print(chr(10) + summary + ".")
        print("Nothing was downloaded. Re-run with --run to fetch.")
        return

    result = run(cities, arguments.kind, arguments.force)
    print("\n" + str(result["status"].value_counts().to_dict()))


if __name__ == "__main__":
    main()
