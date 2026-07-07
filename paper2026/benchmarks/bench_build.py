#!/usr/bin/env python3
"""
B1 — Graph construction benchmark: IduEdu vs OSMnx vs Pyrosm.

Measures wall time, node/edge counts, and resulting in-memory graph
representation size for walk and drive graphs over the same AOI (bbox from the
PBF header), with the simplify on/off ablation for IduEdu and OSMnx (pyrosm has
no comparable switch).

One CSV row per attempt; resume-safe. Overpass responses are cached by both
IduEdu and OSMnx after the first (warm-up) call, so measured times reflect
parsing/graph assembly rather than network I/O — stated explicitly in the paper.

Usage:
    python bench_build.py                 # all cities, all libraries available
    python bench_build.py --smoke         # tiny cached territory, quick harness check
    python bench_build.py --areas "Helsinki,London" --networks walk
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from bench_common import (
    AREAS,
    RESULTS_DIR,
    bbox_from_pbf,
    dump_environment,
    force_cleanup,
    make_key,
    measure,
    networkx_graph_memory_mb,
    resolve_area_pbf,
    urban_graph_memory_mb,
)

OUT_CSV = RESULTS_DIR / "build_benchmark.csv"
KEY_COLUMNS = ["library", "area", "network", "simplify", "attempt"]

ATTEMPTS = 3
SLEEP_SEC = 0.5
SIMPLIFY_SETTINGS = [True, False]
NETWORKS = ["walk", "drive"]

SMOKE_OSM_ID = 1114252  # small SPb district used by the test suite (cached)


def _row_key(row) -> tuple:
    return make_key(row[column] for column in KEY_COLUMNS)


def _needs_graph_memory(row) -> bool:
    """Rows written before representation_size_mb existed should be re-measured."""
    return row.get("representation_size_mb", "") == ""


def normalize_build_csv_schema(csv_path: Path) -> pd.DataFrame:
    """Load B1 CSV and migrate old memory columns to the current schema."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    changed = False
    if "representation_size_mb" not in df.columns:
        if "graph_memory_mb" in df.columns:
            df = df.rename(columns={"graph_memory_mb": "representation_size_mb"})
        else:
            df["representation_size_mb"] = ""
        changed = True
    if "graph_memory_mb" in df.columns:
        df = df.drop(columns=["graph_memory_mb"])
        changed = True
    columns = [*KEY_COLUMNS, "time_sec", "n_nodes", "n_edges", "representation_size_mb"]
    extra_columns = [column for column in df.columns if column not in columns]
    if extra_columns:
        df = df.drop(columns=extra_columns)
        changed = True
    if changed:
        df.to_csv(csv_path, index=False)
    return df


def load_existing_build_keys(csv_path: Path) -> set[tuple]:
    """Return complete B1 keys, treating rows without representation size as pending."""
    if not csv_path.exists():
        return set()
    df = normalize_build_csv_schema(csv_path)
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} exists but misses key columns {missing}; delete or fix it")
    return {_row_key(row) for _, row in df.iterrows() if not _needs_graph_memory(row)}


def append_or_replace_build_row(csv_path: Path, row: dict) -> None:
    """Append a B1 row, replacing a stale row with the same resume key when present."""
    if not csv_path.exists():
        pd.DataFrame([row]).to_csv(csv_path, index=False)
        return

    df = normalize_build_csv_schema(csv_path)
    for column in row:
        if column not in df.columns:
            df[column] = ""

    key = make_key(row[column] for column in KEY_COLUMNS)
    key_mask = pd.Series(True, index=df.index)
    for column, value in zip(KEY_COLUMNS, key):
        key_mask &= df[column].astype(str) == value

    aligned = {column: row.get(column, "") for column in df.columns}
    if key_mask.any():
        first = key_mask[key_mask].index[0]
        df.loc[first, :] = aligned
        df = df.loc[~(key_mask & (df.index != first))]
        df.to_csv(csv_path, index=False)
        return

    pd.DataFrame([aligned]).to_csv(csv_path, mode="a", header=False, index=False)


# ----------------------------
# Builders: each returns (n_nodes, n_edges, graph_or_none).
# The graph is returned (not cleaned up) only when a post-build metric needs it;
# run_one computes that metric OUTSIDE the timed region and cleans up.
# ----------------------------


def build_iduedu(polygon, network: str, simplify: bool):
    from iduedu import get_drive_graph, get_walk_graph

    builder = get_walk_graph if network == "walk" else get_drive_graph
    graph = builder(territory=polygon, simplify=simplify, keep_largest_subgraph=True)
    return len(graph.nodes_gdf), len(graph.edges_gdf), graph


def build_osmnx(polygon, network: str, simplify: bool):
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.log_console = False
    graph = ox.graph_from_polygon(
        polygon=polygon,
        network_type=network,
        simplify=simplify,
        retain_all=False,
        truncate_by_edge=False,
    )
    return graph.number_of_nodes(), graph.number_of_edges(), graph


def build_pyrosm(pbf_path: str, network: str):
    from pyrosm import OSM

    network_type = "walking" if network == "walk" else "driving"
    osm = OSM(pbf_path)
    nodes, edges = osm.get_network(nodes=True, network_type=network_type)
    graph = osm.to_graph(nodes, edges, graph_type="networkx", network_type=network_type)
    force_cleanup(osm, nodes, edges)
    return graph.number_of_nodes(), graph.number_of_edges(), graph


def pyrosm_available() -> bool:
    try:
        import pyrosm  # noqa: F401

        return True
    except ImportError:
        return False


# ----------------------------
# Main
# ----------------------------


def has_pending(existing: set, library: str, area: str, network: str, simplifies: list) -> bool:
    """True if any (simplify, attempt) measurement for this library is still missing."""
    return any(
        make_key([library, area, network, simplify, attempt]) not in existing
        for simplify in simplifies
        for attempt in range(1, ATTEMPTS + 1)
    )


def run_one(
    existing: set, library: str, area: str, network: str, simplify: bool | None, attempt: int, build_fn, *args
) -> None:
    key = make_key([library, area, network, simplify, attempt])
    if key in existing:
        print(f"  [skip] {key}")
        return
    print(f"  [run ] {library} {area} {network} simplify={simplify} attempt={attempt}")
    m = measure(build_fn, *args)
    n_nodes, n_edges, graph = m.result

    # Deterministic size of the resulting graph representation, computed outside
    # the timed region so it does not pollute time_sec.
    representation_size_mb = ""
    if graph is not None:
        if library == "iduedu":
            representation_size_mb = round(urban_graph_memory_mb(graph), 2)
        else:
            representation_size_mb = round(networkx_graph_memory_mb(graph), 2)
        force_cleanup(graph)

    append_or_replace_build_row(
        OUT_CSV,
        dict(
            library=library,
            area=area,
            network=network,
            simplify="NA" if simplify is None else simplify,
            attempt=attempt,
            time_sec=round(m.time_sec, 3),
            n_nodes=n_nodes,
            n_edges=n_edges,
            representation_size_mb=representation_size_mb,
        ),
    )
    existing.add(key)
    time.sleep(SLEEP_SEC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny cached territory, iduedu+osmnx only")
    parser.add_argument("--areas", default=None, help="comma-separated subset of areas")
    parser.add_argument("--networks", default=",".join(NETWORKS))
    args = parser.parse_args()

    dump_environment("build")
    existing = load_existing_build_keys(OUT_CSV)
    if existing:
        print(f"[resume] {len(existing)} measurements already in {OUT_CSV}")

    networks = [n.strip() for n in args.networks.split(",") if n.strip()]

    if args.smoke:
        from iduedu import get_4326_boundary

        polygon = get_4326_boundary(osm_id=SMOKE_OSM_ID)
        for network in networks:
            for simplify in SIMPLIFY_SETTINGS:
                run_one(existing, "iduedu", "smoke", network, simplify, 1, build_iduedu, polygon, network, simplify)
                run_one(existing, "osmnx", "smoke", network, simplify, 1, build_osmnx, polygon, network, simplify)
        print(f"[done] smoke results -> {OUT_CSV}")
        return

    areas = [a.strip() for a in args.areas.split(",")] if args.areas else AREAS
    has_pyrosm = pyrosm_available()
    if not has_pyrosm:
        print("[warn] pyrosm not importable in this environment; run its rows from a conda env later")

    for area in areas:
        # Work out what is still pending up front, so a fully-recorded area needs
        # neither a PBF download (hundreds of MB) nor a warm-up.
        idu_pending = {n: has_pending(existing, "iduedu", area, n, SIMPLIFY_SETTINGS) for n in networks}
        osm_pending = {n: has_pending(existing, "osmnx", area, n, SIMPLIFY_SETTINGS) for n in networks}
        pyr_pending = {n: has_pyrosm and has_pending(existing, "pyrosm", area, n, [None]) for n in networks}

        if not any({**idu_pending, **osm_pending, **pyr_pending}.values()):
            print(f"\n=== {area}: all measurements already recorded, skipping ===")
            continue

        pbf_path = resolve_area_pbf(area)
        bounds = bbox_from_pbf(pbf_path)
        print(f"\n=== {area} | bbox={bounds.bbox} ===")

        # Warm-up populates the Overpass cache for iduedu/osmnx so the first
        # recorded run excludes the download. Skip per library+network when
        # nothing is pending there.
        for network in networks:
            if idu_pending[network]:
                print(f"  [warm] iduedu {network}")
                force_cleanup(build_iduedu(bounds.polygon_4326, network, True)[2])
            if osm_pending[network]:
                print(f"  [warm] osmnx {network}")
                force_cleanup(build_osmnx(bounds.polygon_4326, network, True)[2])

        for network in networks:
            if has_pyrosm:
                for attempt in range(1, ATTEMPTS + 1):
                    run_one(existing, "pyrosm", area, network, None, attempt, build_pyrosm, pbf_path, network)
            for simplify in SIMPLIFY_SETTINGS:
                for attempt in range(1, ATTEMPTS + 1):
                    run_one(
                        existing,
                        "iduedu",
                        area,
                        network,
                        simplify,
                        attempt,
                        build_iduedu,
                        bounds.polygon_4326,
                        network,
                        simplify,
                    )
                for attempt in range(1, ATTEMPTS + 1):
                    run_one(
                        existing,
                        "osmnx",
                        area,
                        network,
                        simplify,
                        attempt,
                        build_osmnx,
                        bounds.polygon_4326,
                        network,
                        simplify,
                    )

    print(f"\n[done] results -> {OUT_CSV}")


if __name__ == "__main__":
    main()
