#!/usr/bin/env python3
"""
B2 — Multimodal (walk + PT + join) pipeline benchmark, IduEdu only.

No other open-source tool builds a static PT graph directly from OSM, so this
benchmark reports the absolute cost and stage decomposition of the IduEdu
pipeline across cities: walk build, PT build, join, and the resulting sizes.

One CSV row per attempt; resume-safe.

Usage:
    python bench_intermodal.py
    python bench_intermodal.py --smoke
    python bench_intermodal.py --areas "Helsinki,Saint Petersburg"
"""

from __future__ import annotations

import argparse
import time

from bench_common import (
    AREAS,
    RESULTS_DIR,
    append_row,
    bbox_from_pbf,
    dump_environment,
    force_cleanup,
    load_existing_keys,
    make_key,
    measure,
    resolve_area_pbf,
)

OUT_CSV = RESULTS_DIR / "intermodal_benchmark.csv"
KEY_COLUMNS = ["area", "attempt"]

ATTEMPTS = 3
SLEEP_SEC = 0.5
SMOKE_OSM_ID = 1114252


def has_pending(existing: set, area: str) -> bool:
    """True if any attempt for this area is still missing."""
    return any(make_key([area, attempt]) not in existing for attempt in range(1, ATTEMPTS + 1))


def run_area(existing: set, area: str, polygon) -> None:
    from iduedu import get_public_transport_graph, get_walk_graph, join_pt_walk_graph

    for attempt in range(1, ATTEMPTS + 1):
        key = make_key([area, attempt])
        if key in existing:
            print(f"  [skip] {key}")
            continue
        print(f"  [run ] {area} attempt={attempt}")

        m_walk = measure(get_walk_graph, territory=polygon, simplify=True, keep_largest_subgraph=False)
        walk_g = m_walk.result

        m_pt = measure(get_public_transport_graph, territory=polygon)
        pt_g = m_pt.result

        if pt_g.nodes_gdf.empty:
            print(f"  [warn] {area}: empty PT graph, join skipped")
            m_join_time = 0.0
            intermodal_nodes = len(walk_g.nodes_gdf)
            intermodal_edges = len(walk_g.edges_gdf)
        else:
            m_join = measure(join_pt_walk_graph, pt_g, walk_g, keep_largest_subgraph=True)
            intermodal = m_join.result
            m_join_time = m_join.time_sec
            intermodal_nodes = len(intermodal.nodes_gdf)
            intermodal_edges = len(intermodal.edges_gdf)
            force_cleanup(intermodal)

        append_row(
            OUT_CSV,
            dict(
                area=area,
                attempt=attempt,
                time_walk_sec=round(m_walk.time_sec, 3),
                time_pt_sec=round(m_pt.time_sec, 3),
                time_join_sec=round(m_join_time, 3),
                time_total_sec=round(m_walk.time_sec + m_pt.time_sec + m_join_time, 3),
                n_nodes_walk=len(walk_g.nodes_gdf),
                n_edges_walk=len(walk_g.edges_gdf),
                n_nodes_pt=len(pt_g.nodes_gdf),
                n_edges_pt=len(pt_g.edges_gdf),
                n_nodes_intermodal=intermodal_nodes,
                n_edges_intermodal=intermodal_edges,
            ),
        )
        existing.add(key)
        force_cleanup(walk_g, pt_g)
        time.sleep(SLEEP_SEC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--areas", default=None)
    args = parser.parse_args()

    dump_environment("intermodal")
    existing = load_existing_keys(OUT_CSV, KEY_COLUMNS)
    if existing:
        print(f"[resume] {len(existing)} measurements already in {OUT_CSV}")

    if args.smoke:
        from iduedu import get_4326_boundary

        polygon = get_4326_boundary(osm_id=SMOKE_OSM_ID)
        run_area(existing, "smoke", polygon)
        print(f"[done] smoke results -> {OUT_CSV}")
        return

    areas = [a.strip() for a in args.areas.split(",")] if args.areas else AREAS
    for area in areas:
        # Fully-recorded area needs neither a PBF download nor a warm-up.
        if not has_pending(existing, area):
            print(f"\n=== {area}: all attempts already recorded, skipping ===")
            continue

        pbf_path = resolve_area_pbf(area)
        bounds = bbox_from_pbf(pbf_path)
        print(f"\n=== {area} | bbox={bounds.bbox} ===")

        # Warm-up: populate Overpass caches (not recorded). Only reached when the
        # area still has pending attempts.
        from iduedu import get_public_transport_graph, get_walk_graph

        print("  [warm] walk + pt")
        force_cleanup(get_walk_graph(territory=bounds.polygon_4326, simplify=True, keep_largest_subgraph=False))
        force_cleanup(get_public_transport_graph(territory=bounds.polygon_4326))

        run_area(existing, area, bounds.polygon_4326)

    print(f"\n[done] results -> {OUT_CSV}")


if __name__ == "__main__":
    main()
