"""Download OpenStreetMap extracts from Geofabrik for a territory, as an alternative to Overpass.

:func:`download_osm_pbf` takes the same territory arguments as the graph builders (``osm_id`` or
``territory``), picks the smallest Geofabrik extract that covers it, downloads that regional file once (with
resume and an md5 check) and cuts a compact file for the territory out of it with
:func:`~iduedu.osm_pbf.split_by_polygons`. ``use_pbf`` calls it itself when given a territory instead of a path::

    with iduedu.use_pbf(osm_id=1114252):  # same as config.use_pbf(download_osm_pbf(osm_id=1114252))
        graph = get_intermodal_graph(osm_id=1114252)

Regional files and cut files are cached under ``config.overpass_cache_dir`` (``geofabrik/`` and
``geofabrik/territories/``); a cut file that already exists is returned without any download.
"""

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import geopandas as gpd
import requests
import shapely
from shapely import MultiPolygon, Polygon
from tqdm.auto import tqdm

from iduedu import config
from iduedu.osm_pbf.extract import split_by_polygons
from iduedu.osm_pbf.reader import PbfSource
from iduedu.overpass.downloaders import _boundary_from_elements, get_4326_boundary, get_boundary_by_osm_id

logger = config.logger

GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1.json"
INDEX_MAX_AGE_DAYS = 7
_CHUNK_BYTES = 1 << 20


def _cache_root(out_dir: str | os.PathLike | None) -> Path:
    root = Path(out_dir) if out_dir is not None else Path(config.overpass_cache_dir or ".iduedu_cache") / "geofabrik"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _get(url: str, *, stream: bool = False, headers: dict | None = None, timeout: float | None = None):
    response = requests.get(
        url,
        stream=stream,
        timeout=timeout or config.timeout,
        headers={"User-Agent": config.user_agent, **(headers or {})},
        proxies=config.proxies,
        verify=config.verify_ssl,
    )
    response.raise_for_status()
    return response


def geofabrik_regions(*, refresh: bool = False, out_dir: str | os.PathLike | None = None) -> gpd.GeoDataFrame:
    """Geofabrik extracts as a GeoDataFrame in EPSG:4326.

    The index (``index-v1.json``) is cached for :data:`INDEX_MAX_AGE_DAYS` days.

    Parameters:
        refresh: Download the index even if a fresh cached copy exists.
        out_dir: Cache directory; defaults to ``<config.overpass_cache_dir>/geofabrik``.

    Returns:
        GeoDataFrame indexed by region id (e.g. ``"northwestern-fed-district"``) with columns ``name``,
        ``parent`` and ``pbf_url`` and the region outline as geometry.
    """
    path = _cache_root(out_dir) / "index-v1.json"
    fresh = path.exists() and time.time() - path.stat().st_mtime < INDEX_MAX_AGE_DAYS * 86400
    if refresh or not fresh:
        try:
            logger.debug(f"Downloading the Geofabrik index from {GEOFABRIK_INDEX_URL} ...")
            data = _get(GEOFABRIK_INDEX_URL).content
            path.write_bytes(data)
        except requests.RequestException:
            if not path.exists():
                raise
            logger.warning("Could not refresh the Geofabrik index; using the cached copy")
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    features = [f for f in features if f.get("geometry") and f["properties"].get("urls", {}).get("pbf")]
    regions = gpd.GeoDataFrame(
        {
            "id": [f["properties"]["id"] for f in features],
            "name": [f["properties"].get("name") for f in features],
            "parent": [f["properties"].get("parent") for f in features],
            "pbf_url": [f["properties"]["urls"]["pbf"] for f in features],
        },
        geometry=[shapely.geometry.shape(f["geometry"]) for f in features],
        crs=4326,
    )
    return regions.set_index("id")


def find_geofabrik_region(
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | gpd.GeoSeries, *, out_dir: str | os.PathLike | None = None
) -> str:
    """Id of the smallest Geofabrik extract that fully covers ``territory`` (EPSG:4326 for shapely input).

    Raises:
        ValueError: If no extract covers the territory.
    """
    polygon = get_4326_boundary(territory=territory)
    regions = geofabrik_regions(out_dir=out_dir)
    covering = regions[regions.covers(polygon)]
    if covering.empty:
        raise ValueError("No Geofabrik extract covers the territory; pass region= or use Overpass")
    if len(covering) == 1:
        return str(covering.index[0])
    return str(covering.to_crs(covering.estimate_utm_crs()).area.idxmin())


@contextmanager
def _overpass_source():
    """Query Overpass even when the caller has switched IduEdu to a local file."""
    previous = (config.osm_source, config.pbf_path)
    config.set_osm_source("overpass")
    try:
        yield
    finally:
        config.set_osm_source(previous[0], path=previous[1])


def _relation_boundary(osm_id: int, region_pbf: Path | None) -> Polygon:
    """Boundary of relation ``osm_id`` normalized like the graph builders do, from the extract or Overpass."""
    if region_pbf is not None:
        boundary = _boundary_from_elements(PbfSource(region_pbf).boundary(osm_id))
    else:
        with _overpass_source():
            boundary = get_boundary_by_osm_id(osm_id)
    if boundary.is_empty:
        source = f"the extract {region_pbf.name}" if region_pbf is not None else "Overpass"
        raise ValueError(f"Relation {osm_id} was not found in {source}")
    return get_4326_boundary(territory=boundary)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for block in iter(lambda: file.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _remote_md5(url: str) -> str | None:
    try:
        return _get(f"{url}.md5").text.split()[0].strip().lower()
    except (requests.RequestException, IndexError):
        logger.warning(f"No md5 checksum for {url}; the download is not verified")
        return None


def _download(url: str, path: Path, expected_md5: str | None) -> None:
    """Stream ``url`` into ``path`` through ``<path>.part``, resuming a partial download."""
    part = path.with_name(path.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    try:
        response = _get(url, stream=True, headers={"Range": f"bytes={done}-"} if done else None)
    except requests.HTTPError as exc:
        if not (done and exc.response is not None and exc.response.status_code == 416):
            raise
        response = None  # the partial file is already complete
    if response is not None and done and response.status_code != 206:  # the server ignored the range
        done = 0
    total = (int(response.headers.get("Content-Length", 0)) + done or None) if response is not None else done
    with (
        part.open("ab" if done else "wb") as file,
        tqdm(
            total=total,
            initial=done,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {path.name}",
            disable=not config.enable_tqdm_bar,
        ) as progress,
    ):
        for block in response.iter_content(_CHUNK_BYTES) if response is not None else ():
            file.write(block)
            progress.update(len(block))
    if expected_md5 is not None and _md5(part) != expected_md5:
        part.unlink()
        raise OSError(f"Checksum mismatch for {url}; the partial file was removed, try again")
    part.replace(path)


def download_geofabrik_region(region: str, *, out_dir: str | os.PathLike | None = None, update: bool = False) -> Path:
    """Download the Geofabrik extract ``region`` (an id from :func:`geofabrik_regions`).

    Parameters:
        region: Geofabrik region id, e.g. ``"northwestern-fed-district"``.
        out_dir: Cache directory; defaults to ``<config.overpass_cache_dir>/geofabrik``.
        update: Re-download when Geofabrik has a newer file (compared by md5). Without it an existing file is
            reused as is, which keeps results reproducible.

    Returns:
        Path of the ``<region>-latest.osm.pbf`` file.
    """
    regions = geofabrik_regions(out_dir=out_dir)
    if region not in regions.index:
        raise ValueError(f"Unknown Geofabrik region {region!r}; see geofabrik_regions()")
    url = regions.at[region, "pbf_url"]
    path = _cache_root(out_dir) / Path(url).name
    checksum_file = path.with_name(path.name + ".md5")
    if path.exists() and not update:
        logger.debug(f"Using cached Geofabrik extract {path}")
        return path
    remote = _remote_md5(url)
    if path.exists() and remote is not None and checksum_file.exists():
        if checksum_file.read_text(encoding="utf-8").strip() == remote:
            logger.debug(f"Geofabrik extract {path.name} is up to date")
            return path
    logger.info(f"Downloading Geofabrik extract {region} from {url}")
    _download(url, path, remote)
    checksum_file.write_text(remote or _md5(path), encoding="utf-8")
    return path


def download_osm_pbf(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    region: str | None = None,
    name: str | None = None,
    out_dir: str | os.PathLike | None = None,
    update: bool = False,
    overwrite: bool = False,
) -> Path:
    """OSM file for a territory, cut out of a Geofabrik extract: a drop-in replacement for Overpass downloads.

    The territory is given as in the graph builders. The smallest Geofabrik extract covering it is chosen
    automatically (or set with ``region``), downloaded once and cached, and the territory is cut out of it
    with :func:`~iduedu.osm_pbf.split_by_polygons`: the file holds everything the graph builders read for
    that territory, including public-transport routes leaving it and, for ``osm_id``, the boundary relation,
    so ``get_*_graph(osm_id=...)`` works on it in pbf mode.

    Parameters:
        osm_id: OSM relation id of the territory. Without ``region`` its boundary is fetched once through
            Overpass (cached) to choose the extract; with ``region`` it is read from the extract, so no
            Overpass request is made.
        territory: Territory polygon in EPSG:4326, or a GeoDataFrame/GeoSeries in any CRS.
        region: Geofabrik region id to cut from; chosen automatically when omitted.
        name: Output file name without extension; defaults to ``osm_<osm_id>`` or a hash of the territory.
        out_dir: Cache directory; defaults to ``<config.overpass_cache_dir>/geofabrik``. Cut files go to its
            ``territories/`` subdirectory.
        update: Refresh a cached regional extract when Geofabrik has a newer one and cut again.
        overwrite: Cut again even if the territory file exists.

    Returns:
        Path of ``<name>.osm.pbf``.

    Raises:
        ValueError: If neither ``osm_id`` nor ``territory`` is given, or no extract covers the territory.
    """
    if osm_id is None and territory is None:
        raise ValueError("Either osm_id or territory must be specified")
    polygon = get_4326_boundary(territory=territory) if territory is not None else None
    if name is None:
        name = f"osm_{osm_id}" if polygon is None else "territory_" + hashlib.sha1(polygon.wkb).hexdigest()[:12]
    target_dir = _cache_root(out_dir) / "territories"
    target = target_dir / f"{name}.osm.pbf"
    if target.exists() and not (overwrite or update):
        logger.debug(f"Using cached territory file {target}")
        return target

    if region is None:
        if polygon is None:
            polygon = _relation_boundary(osm_id, None)
        region = find_geofabrik_region(polygon, out_dir=out_dir)
        logger.info(f"Smallest Geofabrik extract covering the territory: {region}")
    region_pbf = download_geofabrik_region(region, out_dir=out_dir, update=update)
    if polygon is None:
        polygon = _relation_boundary(osm_id, region_pbf)

    logger.info(f"Cutting {name} out of {region_pbf.name} ...")
    keep = {name: [int(osm_id)]} if osm_id is not None else None
    files = split_by_polygons(
        region_pbf,
        {name: polygon},
        target_dir,
        keep_relations=keep,
        overwrite=True,
    )
    return files[name]
