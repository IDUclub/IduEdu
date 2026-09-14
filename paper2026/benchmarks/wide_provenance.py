"""Which feeds were generated from OSM rather than surveyed?

This matters twice over. A feed drawn from OSM cannot be independent evidence
about OSM's completeness -- the two sources are one source wearing two hats --
and its run times, if they were modelled rather than observed, must not calibrate
the model that produced them.

Metadata is useless for this: barely any derived feed declares its origin.
Geometry is decisive. A generator copies OSM coordinates verbatim, so the
vertices of ``shapes.txt`` land on OSM vertices to the centimetre. A vehicle
trace never does: GPS noise puts every point a few metres off, and no amount of
smoothing recreates an exact match.

The measure separates cleanly in practice: known derivatives such as Trujillo
(0.88) and Ernakulam (0.69) sit far above surveyed feeds, three quarters of
which fall below 0.05.
"""

import argparse
import io
import logging
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from bench_common import sweep
from scipy.spatial import cKDTree
from wide_cohort import measurable, restrict
from wide_paths import ALL_OSM_MODES, PROVENANCE_CSV, city_feed_path, osm_graph_path

from iduedu import read_urban_graph

logger = logging.getLogger(__name__)

#: A copied coordinate is exact to the seventh decimal of a degree, roughly a
#: centimetre. Ten centimetres allows for the reprojection both sides go through
#: and still excludes anything a GPS produced.
COINCIDENCE_M = 0.10
LOOSE_M = 2.0

#: Enough to estimate a share; whole feeds run to millions of vertices.
MAX_VERTICES = 60000


def _feed_vertices(city_key: str, crs) -> np.ndarray:
    archive = city_feed_path(city_key)
    if not archive.exists():
        return np.empty((0, 2))
    with zipfile.ZipFile(archive) as handle:
        if "shapes.txt" not in handle.namelist():
            return np.empty((0, 2))
        shapes = pd.read_csv(io.BytesIO(handle.read("shapes.txt")), dtype=str)
    latitude = pd.to_numeric(shapes.get("shape_pt_lat"), errors="coerce")
    longitude = pd.to_numeric(shapes.get("shape_pt_lon"), errors="coerce")
    valid = latitude.notna() & longitude.notna()
    if not valid.any():
        return np.empty((0, 2))
    points = gpd.GeoSeries(gpd.points_from_xy(longitude[valid], latitude[valid]), crs="EPSG:4326").to_crs(crs)
    coordinates = np.column_stack([points.x.to_numpy(), points.y.to_numpy()])
    if len(coordinates) > MAX_VERTICES:
        generator = np.random.default_rng(0)
        coordinates = coordinates[generator.choice(len(coordinates), MAX_VERTICES, replace=False)]
    return coordinates


def _osm_vertices(city_key: str) -> tuple[np.ndarray, object]:
    path = osm_graph_path(city_key, "pt", ALL_OSM_MODES)
    if not path.exists():
        return np.empty((0, 2)), None
    graph = read_urban_graph(path)
    edges = graph.edges_gdf
    if edges.empty or edges.geometry.isna().all():
        return np.empty((0, 2)), graph.nodes_gdf.crs
    coordinates = np.concatenate(
        [np.asarray(geometry.coords) for geometry in edges.geometry.dropna() if not geometry.is_empty]
    )
    return coordinates[:, :2], edges.crs


def measure_city(city_key: str) -> dict:
    osm_points, crs = _osm_vertices(city_key)
    row = {"city_key": city_key, "n_osm_vertices": len(osm_points)}
    if crs is None or len(osm_points) == 0:
        row["status"] = "no_osm_geometry"
        return row

    feed_points = _feed_vertices(city_key, crs)
    row["n_feed_vertices"] = len(feed_points)
    if len(feed_points) == 0:
        row["status"] = "no_feed_shapes"
        return row

    distances, _ = cKDTree(osm_points).query(feed_points)
    row["coincident_share"] = round(float((distances <= COINCIDENCE_M).mean()), 4)
    row["near_share"] = round(float((distances <= LOOSE_M).mean()), 4)
    row["median_distance_m"] = round(float(np.median(distances)), 3)
    row["status"] = "ok"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="*")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sweep(
        restrict(measurable(), arguments.cities),
        measure_city,
        PROVENANCE_CSV,
        force=arguments.force,
        logger=logger,
        describe=lambda rows: (
            f"{rows[0].get('status')} coincident {rows[0].get('coincident_share')} "
            f"median {rows[0].get('median_distance_m')} m"
        ),
    )


if __name__ == "__main__":
    main()
