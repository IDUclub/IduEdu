"""The OpenStreetMap side: Geofabrik extracts, one file per city, transit and walk graphs.

Every OSM graph comes from one snapshot (Geofabrik, replication 2026-10-15T20:21Z).
Each city is cut from the smallest Geofabrik region covering 98 % of its boundary
(``inputs/geofabrik_regions.csv``), and its graphs are built from that city file
with the boundary as the territory (``iduedu`` with ``osm_source="pbf"``); no
request goes to Overpass. The transit graph always holds every mode the builder
knows (``ALL_OSM_MODES``) and records the registry fingerprint it was built with;
a graph built under other constants is rebuilt.

    python pipeline/osm.py download            # regional extracts, checked against .md5
    python pipeline/osm.py split               # one .osm.pbf per city
    python pipeline/osm.py graphs --kind pt    # then --kind walk; never both at once
"""

import argparse
import hashlib
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import requests
from common import (
    ALL_OSM_MODES,
    CITY_PBF_DIR,
    OSM_GRAPHS_CSV,
    PBF_DIR,
    USER_AGENT,
    boundary,
    cities,
    city_pbf_path,
    osm_graph_path,
    read_graph_meta,
    region_pbf_path,
    regions,
    registry_fingerprint,
    restrict,
    setup,
    sweep,
    write_graph_meta,
)

from iduedu import DEFAULT_REGISTRY_W_TRAIN, config, get_public_transport_graph, get_walk_graph, write_urban_graph


def _md5(path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(logger) -> None:
    """Every region in the plan, resumable (curl ``-C -``), checked against the server's ``.md5``.

    The URLs point at Geofabrik's latest extracts: a later download is a later snapshot.
    """
    table = regions().drop_duplicates("region").sort_values("bytes")
    PBF_DIR.mkdir(parents=True, exist_ok=True)
    for index, region in enumerate(table.itertuples(), start=1):
        target = region_pbf_path(region.region)
        expected = requests.get(region.url + ".md5", headers={"User-Agent": USER_AGENT}, timeout=60).text.split()[0]
        if target.exists() and _md5(target) == expected:
            logger.info("[%d/%d] %s: present", index, len(table), region.region)
            continue
        logger.info("[%d/%d] %s", index, len(table), region.region)
        curl = ["curl", "-sS", "-L", "-C", "-", "--retry", "10", "--retry-delay", "30", "-A", USER_AGENT]
        subprocess.run(curl + ["-o", str(target), region.url], check=True)
        if _md5(target) != expected:
            target.unlink()
            raise SystemExit(f"{region.region}: md5 mismatch, file removed; rerun to fetch it again")


def _split_region(region: str, keys: list[str]) -> int:
    from iduedu import split_by_polygons

    # flex_mem keeps the node index in memory; a dense file array is sized by the
    # largest node id in OSM and filled ~195 GB of disk on Japan, Quebec and Ontario.
    written = split_by_polygons(
        region_pbf_path(region), {key: boundary(key) for key in keys}, CITY_PBF_DIR, location_storage="flex_mem"
    )
    return len(written)


def split(logger, workers: int) -> None:
    """One file per city, cut from its region in one pass per region."""
    CITY_PBF_DIR.mkdir(parents=True, exist_ok=True)
    plan = regions().reset_index()
    todo = plan[~plan["city_key"].map(lambda key: city_pbf_path(key).exists())]
    groups = todo.groupby("region")["city_key"].apply(list)
    logger.info("%d cities to cut from %d regions", len(todo), len(groups))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_split_region, region, keys): region for region, keys in groups.items()}
        for index, future in enumerate(as_completed(futures), start=1):
            logger.info("[%d/%d] %s: %d cities", index, len(groups), futures[future], future.result())


def _summary(graph) -> dict:
    nodes, edges = graph.nodes_gdf, graph.edges_gdf
    row = {"n_nodes": len(nodes), "n_edges": len(edges)}
    if not edges.empty and "type" in edges:
        row["edge_types"] = ";".join(f"{k}:{v}" for k, v in sorted(edges["type"].value_counts().items()))
    if not nodes.empty and "route" in nodes:
        row["n_routes"] = int(nodes["route"].dropna().map(str).nunique())
    return row


def build_graph(city_key: str, kind: str) -> dict:
    """One graph of one city from its .pbf."""
    from iduedu import snapshot_time

    destination = osm_graph_path(city_key, kind)
    region = regions().at[city_key, "region"]
    stamp = snapshot_time(region_pbf_path(region)) if region_pbf_path(region).exists() else None
    snapshot = stamp.strftime("%Y-%m-%dT%H:%M:%SZ") if stamp else regions().at[city_key, "snapshot_utc"]
    row = {"city_key": city_key, "kind": kind, "osm_base": snapshot, "source": f"pbf:{region}", "status": "failed"}
    if not city_pbf_path(city_key).exists():
        row["reason"] = f"no city extract {city_pbf_path(city_key).name} (run: osm.py split)"
        return row
    started = time.monotonic()
    with config.use_pbf(city_pbf_path(city_key)):
        if kind == "pt":
            graph = get_public_transport_graph(
                territory=boundary(city_key),
                transport_types=ALL_OSM_MODES,
                clip_by_territory=True,
                transport_registry=DEFAULT_REGISTRY_W_TRAIN,
            )
        else:
            graph = get_walk_graph(territory=boundary(city_key), clip_by_territory=True, keep_largest_subgraph=False)
    row["elapsed_s"] = round(time.monotonic() - started, 1)
    if graph.nodes_gdf.empty:
        row.update(status="empty", reason="OSM holds no objects for these modes here")
        return row
    write_urban_graph(graph, destination)
    fingerprint = registry_fingerprint(DEFAULT_REGISTRY_W_TRAIN) if kind == "pt" else None
    write_graph_meta(
        destination,
        registry_fingerprint=fingerprint,
        modes_requested=ALL_OSM_MODES,
        osm_id=int(cities().at[city_key, "osm_id"]),
        osm_base=snapshot,
        source=row["source"],
    )
    row.update(_summary(graph), status="ok")
    return row


def up_to_date(city_key: str, kind: str) -> bool:
    destination = osm_graph_path(city_key, kind)
    if not destination.exists():
        return False
    return kind == "walk" or read_graph_meta(destination).get("registry_fingerprint") == registry_fingerprint(
        DEFAULT_REGISTRY_W_TRAIN
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["download", "split", "graphs"])
    parser.add_argument("--kind", choices=["pt", "walk"], default="pt")
    parser.add_argument("--cities", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="split: regions cut in parallel")
    arguments = parser.parse_args()
    logger = setup("osm")
    if arguments.step == "download":
        download(logger)
    elif arguments.step == "split":
        split(logger, arguments.workers)
    else:
        keys = restrict(cities().index, arguments.cities)
        done = None if arguments.force else {key for key in keys if up_to_date(key, arguments.kind)}
        sweep(
            keys,
            lambda key: build_graph(key, arguments.kind),
            OSM_GRAPHS_CSV,
            force=arguments.force,
            done=done,
            row_keys=["city_key", "kind"],
            describe=lambda rows: f"{rows[0]['status']} {rows[0].get('n_nodes', 0)} nodes",
            logger=logger,
        )


if __name__ == "__main__":
    main()
