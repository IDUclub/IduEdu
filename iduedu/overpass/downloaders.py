import datetime as dt
import math
import re
import time
from collections import defaultdict
from typing import Literal

import geopandas as gpd
import pandas as pd
import requests
from shapely import LineString, MultiPolygon, Polygon, unary_union
from shapely.ops import polygonize

from iduedu import config
from iduedu.constants.transport_specs import osm_route_values
from iduedu.overpass.cache import cache_load, cache_save_async

logger = config.logger

import threading
from email.utils import parsedate_to_datetime
from time import monotonic


class RateLimiter:
    """Thread-safe fixed-interval rate limiter for Overpass requests."""

    def __init__(self, min_interval: float):
        """Initialize the exception with request context."""
        self.min_interval = float(min_interval)
        self._next_ts = 0.0
        self._cv = threading.Condition()

    def wait(self):
        """Block until the next request slot is available."""
        with self._cv:
            while True:
                now = monotonic()
                if now >= self._next_ts:
                    # Fixed slot schedule.
                    self._next_ts = max(self._next_ts, now) + self.min_interval
                    self._cv.notify_all()
                    return
                to_sleep = self._next_ts - now
                self._cv.wait(timeout=to_sleep)


OVERPASS_MIN_INTERVAL = config.overpass_min_interval
OVERPASS_RL = RateLimiter(OVERPASS_MIN_INTERVAL)


def using_pbf() -> bool:
    """True when ``config.osm_source`` is ``"pbf"``: queries are answered from a local OSM file."""
    return config.osm_source == "pbf"


def osm_source_label() -> str:
    """Human-readable name of the active OSM data source, for log messages."""
    if using_pbf():
        return f"PBF file {config.pbf_path}"
    return "Overpass"


def active_overpass_date() -> str | None:
    """The Overpass snapshot date in effect: ``config.overpass_date``, or ``None`` in pbf mode,
    where the file itself is the snapshot."""
    if using_pbf():
        return None
    return config.overpass_date


def _pbf_source():
    """The :class:`~iduedu.osm_pbf.PbfSource` for ``config.pbf_path``."""
    if not config.pbf_path:
        raise ValueError("config.osm_source is 'pbf' but config.pbf_path is not set; use config.set_osm_source")
    try:
        from iduedu.osm_pbf.reader import PbfSource  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:  # pragma: no cover - the reader soft-imports osmium itself
        if exc.name != "osmium":
            raise
        raise ImportError("osm_source='pbf' needs pyosmium. Install it with: pip install 'iduedu[pbf]'") from exc
    return PbfSource(config.pbf_path)


def _overpass_http(
    method: Literal["GET", "POST"], url: str, *, params=None, data=None, timeout=None
) -> requests.Response:
    OVERPASS_RL.wait()
    headers = {"User-Agent": config.user_agent}
    proxies = config.proxies
    verify = config.verify_ssl
    req_timeout = timeout or config.timeout

    if method == "GET":
        return requests.get(url, params=params, timeout=req_timeout, headers=headers, proxies=proxies, verify=verify)
    else:
        return requests.post(url, data=data, timeout=req_timeout, headers=headers, proxies=proxies, verify=verify)


class RequestError(RuntimeError):
    """
    Basic error for eny requests problems
    """

    def __init__(self, message, status_code=None, reason=None, response_text=None, response_content=None):
        """Initialize the exception with request context."""
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason
        self.response_text = response_text
        self.response_content = response_content

    def __str__(self):  # pragma: no cover
        if self.status_code == 400:
            return (
                f"{super().__str__()} (status: {self.status_code}, reason: {self.reason}). "
                f"Make sure provided polygon is in CRS 4326."
            )
        return f"{super().__str__()} (status: {self.status_code}, reason: {self.reason})."


def _get_overpass_pause(
    base_endpoint: str,
    recursion_pause: float = 5,
) -> float:
    url = base_endpoint[: -len("/interpreter")] + "/status"
    try:
        response = _overpass_http("GET", url, timeout=config.timeout)
        response_text = response.text
    except requests.RequestException as e:  # pragma: no cover
        raise RequestError(f"Unable to reach {url}, {e}") from e

    try:
        status = response_text.split("\n")[4]
        status_first_part = status.split(" ")[0]
    except (AttributeError, IndexError, ValueError):  # pragma: no cover
        raise RequestError(f"Unable to parse {url} response: {response_text}")

    try:
        _ = int(status_first_part)  # number of available slots
        pause: float = 0
    except ValueError:  # pragma: no cover
        if status_first_part == "Slot":
            utc_time_str = status.split(" ")[3]
            pattern = "%Y-%m-%dT%H:%M:%SZ,"
            utc_time = dt.datetime.strptime(utc_time_str, pattern).replace(tzinfo=dt.timezone.utc)
            utc_now = dt.datetime.now(tz=dt.timezone.utc)
            seconds = int(math.ceil((utc_time - utc_now).total_seconds()))
            pause = max(seconds, 1)
        elif status_first_part == "Currently":  # pragma: no cover
            time.sleep(recursion_pause)
            pause = _get_overpass_pause(base_endpoint, recursion_pause=recursion_pause)
        else:
            raise RequestError(f"Unrecognized server status: {status!r}")
    return pause


def _overpass_request(
    method: Literal["GET", "POST"],
    overpass_url: str,
    params: dict | None = None,
    data: dict | None = None,
    timeout: float | None = None,
    *,
    max_retries: int | None = None,
    backoff_base: float | None = None,
) -> requests.Response:

    if max_retries is None:
        max_retries = config.overpass_max_retries
    if backoff_base is None:
        backoff_base = config.overpass_backoff_base

    retryable_statuses = set(config.overpass_retry_statuses)

    last_err_text = None
    for attempt in range(max_retries + 1):
        try:
            resp = _overpass_http(method, overpass_url, params=params, data=data, timeout=timeout or config.timeout)
        except requests.RequestException as e:  # pragma: no cover
            last_err_text = str(e)
            if attempt < max_retries:
                sleep_s = min(60, backoff_base**attempt)
                logger.warning(f"Network exception {e}, retrying in {sleep_s} (attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_s)
                continue
            raise RequestError(f"Request error: {e}") from e

        if resp.status_code == 200:
            return resp

        if resp.status_code == 429:  # pragma: no cover
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    wait_s = int(retry_after)
                except ValueError:
                    try:
                        ra_dt = parsedate_to_datetime(retry_after)
                        wait_s = max(1, int((ra_dt - dt.datetime.now(tz=ra_dt.tzinfo)).total_seconds()))
                    except Exception:
                        wait_s = 5
                wait_s = max(wait_s, 1)
                logger.debug(f"HTTP 429: honoring Retry-After={wait_s}")
                time.sleep(wait_s)
            else:
                try:
                    pause = _get_overpass_pause(overpass_url)
                except RequestError:
                    pause = 0
                wait_s = max(1, pause or int(backoff_base**attempt))
                logger.debug(f"HTTP 429: waiting {wait_s} seconds before retry")
                time.sleep(wait_s)

            if attempt < max_retries:
                continue

        if resp.status_code in retryable_statuses:
            if attempt < max_retries:
                wait_s = min(60, backoff_base**attempt)
                logger.warning(f"HTTP {resp.status_code}: retrying in {wait_s}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_s)
                continue
        last_err_text = resp.text
        break

    raise RequestError(
        message=f"Request failed with status code {resp.status_code}, reason: {resp.reason}",
        status_code=resp.status_code,
        reason=resp.reason,
        response_text=last_err_text,
        response_content=resp.content,
    )


def get_boundary_by_osm_id(osm_id) -> MultiPolygon | Polygon:
    """Download an OSM relation boundary and convert it to a Shapely geometry.

    In pbf mode (``config.osm_source == "pbf"``) the relation is read from ``config.pbf_path``.
    """
    if using_pbf():
        logger.debug(f"Reading territory bounds with osm_id <{osm_id}> from {osm_source_label()} ...")
        return _boundary_from_elements(_pbf_source().boundary(osm_id))

    header = config.overpass_header
    overpass_query = f"""
                    {header}
                    (relation({osm_id}););
                    out geom;
    """

    cache_key_src = f"{config.overpass_url}\nGET\n{overpass_query}"
    json_result = cache_load("boundary", cache_key_src)
    if json_result is None:
        logger.debug(f"Downloading territory bounds with osm_id <{osm_id}> ...")
        resp = _overpass_request(
            method="GET",
            overpass_url=config.overpass_url,
            params={"data": overpass_query},
        )
        json_result = resp.json()
        cache_save_async("boundary", cache_key_src, json_result)
    else:
        logger.debug(f"Using cached territory bounds with osm_id <{osm_id}>")
    return _boundary_from_elements(json_result["elements"])


def _boundary_from_elements(elements: list) -> MultiPolygon | Polygon:
    geoms = []
    for element in elements:
        geometries_inners = []
        geometries_outers = []
        for member in element["members"]:
            if "geometry" in member:
                if member["role"] == "outer":
                    geometries_outers.append(
                        LineString([(coords["lon"], coords["lat"]) for coords in member["geometry"]])
                    )
                if member["role"] == "inner":
                    geometries_inners.append(
                        LineString([(coords["lon"], coords["lat"]) for coords in member["geometry"]])
                    )
        outer_poly = unary_union(list(polygonize(geometries_outers)))
        inner_poly = unary_union(list(polygonize(geometries_inners)))
        geoms.append(outer_poly.difference(inner_poly))
    return unary_union(geoms)


def get_4326_boundary(
    *, osm_id: int | None = None, territory: Polygon | MultiPolygon | gpd.GeoDataFrame | gpd.GeoSeries | None = None
) -> Polygon:
    """
    Normalize a territory boundary to a single EPSG:4326 Polygon.

    Accepts either an `osm_id` (relation id) to fetch the boundary from OSM, a direct
    `Polygon`/`MultiPolygon`, or a `GeoDataFrame`. Returns a single `Polygon` in lon/lat.
    For `MultiPolygon`, the function returns its **convex hull** as a Polygon.

    Parameters:
        osm_id (int | None): OSM relation id. If provided, boundary is fetched via Overpass (or read from
            the local OSM file in pbf mode, see ``config.set_osm_source``).
        territory (Polygon | MultiPolygon | gpd.GeoDataFrame | gpd.GeoSeries | None): Existing boundary geometry
            or a GeoDataFrame containing it. If GeoDataFrame is given, it is reprojected to 4326
            and unioned (`.union_all()`).

    Returns:
        (shapely.Polygon): Boundary polygon in EPSG:4326.

    Notes:
        - Input `Polygon` is returned as-is (assumed already in EPSG:4326 by caller).
        - For `MultiPolygon`, a **convex hull** is returned (may slightly expand the area and
          fill gaps between parts).
        - For `GeoDataFrame`, the geometry is first reprojected to 4326 and dissolved via
          `.union_all()`; the result is then normalized to a `Polygon` (convex hull if needed).

    Examples:
        >>> get_4326_boundary(osm_id=1114252)             # fetch by OSM id
        >>> get_4326_boundary(territory=poly4326)         # keep polygon
        >>> get_4326_boundary(territory=multi_poly_4326)  # convex hull of multipart
        >>> get_4326_boundary(territory=territory_gdf)    # GDF -> to_crs(4326) -> union_all -> Polygon

    See also:
        https://iduclub.github.io/IduEdu/examples/get_any_graph.html
    """
    if osm_id is not None:
        territory = get_boundary_by_osm_id(osm_id)

    if territory is None:
        raise ValueError("Either osm_id or territory must be specified")

    if isinstance(territory, (gpd.GeoDataFrame, gpd.GeoSeries)):
        obj = territory.to_crs(4326)
        territory = obj.union_all() if isinstance(obj, gpd.GeoSeries) else obj.geometry.union_all()

    # Shapely geometries
    if isinstance(territory, Polygon):
        return territory

    if isinstance(territory, MultiPolygon):
        return Polygon(territory.convex_hull)

    raise TypeError("territory must be one of: Polygon, MultiPolygon, GeoDataFrame, GeoSeries (or provide osm_id)")


def _poly_to_overpass(poly: Polygon) -> str:
    return " ".join(f"{y} {x}" for x, y in poly.exterior.coords[:-1])


#: Name of the marker element (``make section name=...``) that separates the
#: stop-area members from the rest of a route response.
STOP_AREA_MEMBERS_SECTION = "stop_area_members"


def split_stop_area_members(elements: list) -> tuple[list, dict | None]:
    """Split a route response into its route elements and its stop-area members.

    Returns ``(elements_before_the_marker, member_payloads)``. ``member_payloads``
    maps ``(type, id)`` to tags and coordinates, as :func:`fetch_member_tags`
    returns them, or is ``None`` when the response carries no member section -- a
    response cached before the section existed, or one without subway details.
    """
    for index, element in enumerate(elements):
        if (
            isinstance(element, dict)
            and element.get("type") == "section"
            and (element.get("tags") or {}).get("name") == STOP_AREA_MEMBERS_SECTION
        ):
            return list(elements[:index]), member_payloads(elements[index + 1 :])
    return list(elements), None


def member_payloads(elements: list) -> dict:
    """Tags and coordinates of OSM elements, keyed by ``(type, id)``."""
    result = {}
    for e in elements or []:
        if not isinstance(e, dict) or "id" not in e or e.get("type") not in {"node", "way", "relation"}:
            continue
        payload = {"tags": e.get("tags", {}) or {}}
        if e.get("type") == "node" and ("lon" in e and "lat" in e):
            payload["__latlon__"] = (e["lon"], e["lat"])
        if e.get("type") != "node" and "center" in e:
            payload["__center__"] = (e["center"]["lon"], e["center"]["lat"])
        if "geometry" in e and e["geometry"]:
            payload["__geometry__"] = e["geometry"]
        result[(e["type"], int(e["id"]))] = payload
    return result


def get_routes_by_poly(polygon: Polygon, public_transport_types: list[str]) -> list[dict]:
    """Download public-transport route elements intersecting a polygon.

    In pbf mode the same elements are read from ``config.pbf_path`` (see :class:`~iduedu.osm_pbf.PbfSource`).
    """
    public_transport_types = sorted(set(public_transport_types))

    if not public_transport_types:
        return pd.DataFrame()

    if using_pbf():
        logger.info(f"Reading routes with types {public_transport_types} from {osm_source_label()} ...")
        return _pbf_source().routes_by_poly(polygon, public_transport_types)

    has_subway = "subway" in public_transport_types
    non_subway_types = [t for t in public_transport_types if t != "subway"]

    has_date = config.overpass_date is not None

    if has_subway and has_date:  # pragma: no cover
        logger.warning(
            f"Overpass date is set ({config.overpass_date}); skipping subway stop area / station details "
            "and querying subway as regular route relations only."
        )

    polygon_coords = _poly_to_overpass(polygon)

    header = config.overpass_header
    query_parts = [header]

    # A request for "tram" has to fetch the light rail OSM files under its own tag,
    # or the mode is simply absent from the graph in every city that tags it that
    # way. See OSM_ROUTE_ALIASES.
    simple_route_types = sorted({value for t in non_subway_types for value in osm_route_values(t)})

    if simple_route_types:
        if len(simple_route_types) == 1:
            route_filter = f'["route"="{simple_route_types[0]}"]'
        else:
            pattern = "|".join(re.escape(t) for t in simple_route_types)
            route_filter = f'["route"~"^({pattern})$"]'

        query_parts.append(f'rel(poly:"{polygon_coords}"){route_filter}->.routes_basic;')
        query_parts.append(
            """
            way(r.routes_basic);
            out tags qt;
            """.strip()
        )

    enable_subway_details = has_subway and not has_date

    if enable_subway_details:
        query_parts.append(
            f"""
            rel(poly:"{polygon_coords}")["route"="subway"]->.routes;
            node(r.routes)->.route_nodes;
            rel(bn.route_nodes)->.stop_areas;
            rel(br.stop_areas)["public_transport"="stop_area_group"]["type"="public_transport"]->.groups;
            nwr(r.stop_areas)["public_transport"="station"]->.stations;
            nwr(r.stop_areas)->.stop_area_members;
            """.strip()
        )

    if simple_route_types:
        query_parts.append(".routes_basic out geom qt;")

    if enable_subway_details:
        query_parts.append(
            f"""
            .stop_areas out geom qt;
            .groups     out body qt;
            .stations   out tags qt;
            make section name="{STOP_AREA_MEMBERS_SECTION}";
            out;
            .stop_area_members out center qt;
            """.strip()
        )

    overpass_query = "\n".join(query_parts)

    cache_key_src = f"{config.overpass_url}\nPOST\n{overpass_query}"
    json_root = cache_load("routes", cache_key_src)

    if json_root is None:
        resp = _overpass_request(
            method="POST",
            overpass_url=config.overpass_url,
            data={"data": overpass_query},
        )
        json_root = resp.json()
        cache_save_async("routes", cache_key_src, json_root)
        logger.info(f"Downloading routes via Overpass with types {public_transport_types} done!")
    else:
        logger.info("Cache hit. Retrieving routes from local cache.")

    json_result = json_root.get("elements", [])
    return json_result


def get_network_by_filters(polygon: Polygon, way_filter: str) -> pd.DataFrame:
    """Download OSM way elements matching an Overpass filter inside a polygon.

    In pbf mode the ways are read from ``config.pbf_path`` with the same filter and ``poly:`` semantics.
    """
    if using_pbf():
        logger.debug(f"Reading network with filters <{way_filter}> from {osm_source_label()} ...")
        elements = _pbf_source().ways_by_filter(polygon, way_filter)
        logger.info("Network successfully read!")
        return pd.DataFrame(elements)

    polygon_coords = _poly_to_overpass(polygon)
    header = config.overpass_header
    overpass_query = f"""
                    {header}
                    (way{way_filter}(poly:"{polygon_coords}"););
                    out geom;
    """
    cache_key_src = f"{config.overpass_url}\nPOST\n{overpass_query}"
    json_root = cache_load("network", cache_key_src)

    if json_root is None:
        logger.debug(f"Downloading network from OSM with filters <{way_filter}> ...")
        resp = _overpass_request(
            method="POST",
            overpass_url=config.overpass_url,
            data={"data": overpass_query},
        )
        json_root = resp.json()
        cache_save_async("network", cache_key_src, json_root)
        logger.info("Network successfully downloaded!")
    else:
        logger.info("Cache hit. Retrieving network from local cache.")

    json_result = json_root.get("elements", [])
    return pd.DataFrame(json_result)


def fetch_member_tags(members_missing):
    """
    members_missing: iterable of dicts {'type': 'node|way|relation', 'ref': int}

    returns:
      {
        ("node", 123): {"tags": {...}, "__latlon__": (lon,lat)},
        ("way", 456):  {"tags": {...}, "__center__": (lon,lat), "__geometry__": [...]},
        ("relation",7):{"tags": {...}, "__center__": (lon,lat), "__geometry__": [...]},
      }
    """
    ids = defaultdict(list)
    for m in members_missing:
        ids[m["type"]].append(int(m["ref"]))

    for k in list(ids.keys()):
        ids[k] = sorted(set(ids[k]))

    if not any(ids.values()):
        return {}

    if using_pbf():
        return _pbf_source().member_tags([{"type": t, "ref": ref} for t, refs in ids.items() for ref in refs])

    def _build_query(all_ids: dict) -> str:
        parts = []
        if all_ids.get("node"):
            parts.append(f'node(id:{",".join(map(str, all_ids["node"]))});')
        if all_ids.get("way"):
            parts.append(f'way(id:{",".join(map(str, all_ids["way"]))});')
        if all_ids.get("relation"):
            parts.append(f'rel(id:{",".join(map(str, all_ids["relation"]))});')

        body = "\n".join(parts)
        header = config.overpass_header

        return f"{header}\n(\n{body}\n);\nout center qt;"

    q = _build_query(ids)
    cache_key_src = f"{config.overpass_url}\nPOST\n{q}"
    json_root = cache_load("members", cache_key_src)

    if json_root is None:
        resp = _overpass_request(method="POST", overpass_url=config.overpass_url, data={"data": q})
        json_root = resp.json()
        cache_save_async("members", cache_key_src, json_root)
    else:
        logger.debug("Using cached fetch_member_tags")

    return member_payloads(json_root.get("elements", []))
