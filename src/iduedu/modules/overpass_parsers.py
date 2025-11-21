import math
import warnings
from collections import defaultdict
from itertools import chain, combinations

import numpy as np
import pandas as pd
from pandas import DataFrame
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from shapely import LineString, Point
from shapely.ops import substring

from iduedu.modules.overpass_downloaders import fetch_member_tags

PLATFORM_ROLES = ["platform_entry_only", "platform", "platform_exit_only"]
STOPS_ROLES = ["stop", "stop_exit_only", "stop_entry_only"]
THRESHOLD_METERS = 100


def _link_unconnected(disconnected_ways) -> list:
    # Считаем связи между линиями
    connect_points = [point for coords in disconnected_ways for point in (coords[0], coords[-1])]
    distances = cdist(connect_points, connect_points)
    n = distances.shape[0]
    mask = (np.arange(n)[:, None] // 2) == (np.arange(n) // 2)
    distances[mask] = np.inf

    def relative_point(point: int):
        rel_point = point // 2 * 2
        return rel_point if rel_point != point else rel_point + 1

    connected_ways = []
    # pylint: disable=unbalanced-tuple-unpacking
    first_con_1, first_con_2 = np.unravel_index(np.argmin(distances), distances.shape)
    distances[first_con_1, :] = np.inf
    distances[first_con_2, :] = np.inf
    distances[:, first_con_2] = np.inf
    distances[:, first_con_1] = np.inf
    first_con_1_rel, first_con_2_rel = [relative_point(x) for x in (first_con_1, first_con_2)]
    distances[first_con_1_rel, first_con_2_rel] = np.inf
    distances[first_con_2_rel, first_con_1_rel] = np.inf
    distances[:, first_con_1_rel] = np.inf
    distances[:, first_con_2_rel] = np.inf

    line1, line2 = first_con_1 // 2, first_con_2 // 2

    if first_con_1 % 2 == 1:
        connected_ways += disconnected_ways[line1]
    else:
        connected_ways += disconnected_ways[line1][::-1]
    if first_con_2 % 2 == 0:
        connected_ways += disconnected_ways[line2]
    else:
        connected_ways += disconnected_ways[line2][::-1]

    extreme_points = [first_con_1_rel, first_con_2_rel]

    for _ in range(len(disconnected_ways) - 2):
        # pylint: disable=invalid-sequence-index
        position, ind = np.unravel_index(np.argmin(distances[extreme_points]), (2, n))
        next_con = (extreme_points[position], ind)
        rel_point = relative_point(next_con[1])

        line = ind // 2
        if position == 0:
            connected_ways = (
                disconnected_ways[line] + connected_ways
                if ind % 2 == 1
                else (disconnected_ways[line][::-1] + connected_ways)
            )
            extreme_points = [rel_point, extreme_points[1]]
        else:
            connected_ways = (
                connected_ways + disconnected_ways[line]
                if ind % 2 == 0
                else (connected_ways + disconnected_ways[line][::-1])
            )
            extreme_points = [extreme_points[0], rel_point]

        # Чтоб нельзя было зациклиться
        distances[:, rel_point] = np.inf
        distances[next_con[0], :] = np.inf
        distances[next_con[1], :] = np.inf
        distances[:, next_con[0]] = np.inf
        distances[:, next_con[1]] = np.inf
        rel_point_2 = relative_point(next_con[0])
        distances[rel_point, rel_point_2] = np.inf
        distances[rel_point_2, rel_point] = np.inf

    return connected_ways


def parse_overpass_route_response(loc: dict, crs: CRS, needed_tags: list[str], loc_id) -> pd.Series:
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # TODO надо писать версию с merge_line и отдельными запросами по максимальной скорости
    def transform_geometry(loc):
        if isinstance(loc["geometry"], float):
            return transformer.transform(loc["lon"], loc["lat"])
        p = LineString([transformer.transform(coords["lon"], coords["lat"]) for coords in loc["geometry"]]).centroid
        return p.x, p.y

    def process_roles(route, roles):
        filtered = route[route["role"].isin(roles)]
        return filtered.apply(transform_geometry, axis=1).tolist(), filtered["ref"].tolist()

    def extract_needed(loc_obj: dict, keys: list[str]) -> dict | None:
        out = {}
        tags = loc_obj.get("tags", {}) if isinstance(loc_obj.get("tags"), dict) else {}
        for k in keys:
            value_found = None
            if "." in k:
                # dotted path
                parts = k.split(".")
                cur = loc_obj
                ok = True
                for part in parts:
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        ok = False
                        break
                if ok:
                    value_found = cur
            else:
                if k in tags:
                    value_found = tags[k]
                elif k in loc_obj:
                    value_found = loc_obj[k]
            if value_found is not None:
                out[k] = value_found
        return out or None

    tags = loc.get("tags", {}) if isinstance(loc.get("tags"), dict) else {}

    transport_name = f"Unnamed_{loc_id}"
    if "ref" in tags:
        transport_name = tags["ref"]
    elif "name" in tags:
        transport_name = tags["name"]

    members = loc.get("members", [])
    route = pd.DataFrame(members) if isinstance(members, list) else pd.DataFrame()

    for col in ("geometry", "lat", "lon", "role", "type"):
        if col not in route.columns:
            route[col] = np.nan

    route = route.dropna(subset=["lat", "lon", "geometry"], how="all")

    platforms, platforms_refs = process_roles(route, PLATFORM_ROLES)
    stops, stops_refs = process_roles(route, STOPS_ROLES)

    ways_df = route[(route["type"] == "way") & (route["role"].fillna("").isin(["", "forward", "backward"]))]

    if not ways_df.empty:
        ways = (
            ways_df["geometry"]
            .reset_index(drop=True)
            .apply(lambda g: [transformer.transform(pt["lon"], pt["lat"]) for pt in g])
            .tolist()
        )
        connected_ways = [[]]
        cur_way = 0
        for coords in ways:
            if not coords:
                continue
            # Соединяем маршруты, если всё ок, идут без пропусков
            if not connected_ways[cur_way]:
                connected_ways[cur_way] += coords
                continue
            if coords[0] == coords[-1]:
                # Круговое движение зацикленное зачастую в осм, можно отработать, но сходу не придумал
                continue
            if connected_ways[cur_way][-1] == coords[0]:
                connected_ways[cur_way] += coords[1:]

            elif connected_ways[cur_way][-1] == coords[-1]:
                connected_ways[cur_way] += coords[::-1][1:]

            elif connected_ways[cur_way][0] == coords[0]:
                connected_ways[cur_way] = coords[1:][::-1] + connected_ways[cur_way]

            elif connected_ways[cur_way][0] == coords[-1]:
                connected_ways[cur_way] = coords + connected_ways[cur_way][1:]

            # Случай если нету соединяющей точки между линиями маршрута
            else:
                connected_ways += [coords]
                cur_way += 1
        # Соединяем линии по ближайшим точкам этих линий

        if len(connected_ways) > 1:
            # удаляем все круговые движения
            to_del = [i for i, data in enumerate(connected_ways) if data[0] == data[-1]]
            # Если кол-во удалений == кол-во путей, надо оставить хотя бы самый большой
            if len(to_del) == len(connected_ways):
                # Найдём индекс самого длинного пути среди замкнутых
                longest_index = max(to_del, key=lambda i: len(connected_ways[i]))
                # Удалим все, кроме самого длинного
                to_del.remove(longest_index)
            connected_ways = [i for j, i in enumerate(connected_ways) if j not in to_del]
        if len(connected_ways) > 1:
            path = _link_unconnected(connected_ways)
        else:
            if not connected_ways or not connected_ways[0]:
                raise Exception("No connected ways")
            path = connected_ways[0]

    else:
        path = None

    extra = extract_needed(loc, needed_tags)

    return pd.Series(
        {
            "path": path,
            "platforms": platforms,
            "platforms_refs": platforms_refs,
            "stops": stops,
            "stops_refs": stops_refs,
            "route": transport_name,
            "extra_data": extra,
        }
    )


def geometry_to_graph_edge_node_df(loc: pd.Series, transport_type, loc_id) -> DataFrame | None:

    name = loc.route

    platforms = list(loc.platforms or [])  # same size ->
    platforms_refs = list(loc.platforms_refs or [])

    stops = list(loc.stops or [])
    stops_refs = list(loc.stops_refs or [])

    path = loc.path
    extra_data = loc.extra_data

    if not path:
        return None

    def _side_left_or_right(point):
        # 1 if left 0 if right для определения с какой стороны от линии точки
        dist = path.project(point)

        d1 = dist - 1 if dist - 1 > 0 else 0
        d2 = dist + 1 if dist + 1 < path.length else path.length
        line = substring(path, d1, d2)
        x1, y1 = line.coords[0]
        x2, y2 = line.coords[-1]
        x, y = point.coords[0]

        cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross_product > 0:
            return 1
        return 0

    def _offset_point(point, direction, distance=7) -> tuple[float, float]:
        # для размещения платформы по одну сторону от пути на расстоянии
        dist = path.project(point)
        d1 = dist - 1 if dist - 1 > 0 else 0
        d2 = dist + 1.1 if dist + 1.1 < path.length else path.length
        nearest_pt_on_line = path.interpolate(dist)
        line = substring(path, d1, d2)

        x1, y1 = line.coords[0]
        x2, y2 = line.coords[-1]

        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        dx, dy = dx / length, dy / length

        if direction == 0:  # Вправо
            nx, ny = dy, -dx
        else:  # Влево
            nx, ny = -dy, dx

        # Смещенная точка
        offset_x = nearest_pt_on_line.x + nx * distance
        offset_y = nearest_pt_on_line.y + ny * distance
        return offset_x, offset_y

    path = LineString(path)

    items = []  # [(platform_coords, platform_ref, stop_coords, stop_ref), на выходе не может быть пары без платформы

    platform_len, stops_len = len(platforms), len(stops)
    matched_p, matched_s = set(), set()
    pairs = {}  # p_idx -> s_idx

    if platform_len and stops_len:

        stop_tree = cKDTree(stops)
        k = min(2, stops_len)
        dists, idxs = stop_tree.query(platforms, k=k)

        dists = np.asarray(dists)
        idxs = np.asarray(idxs)
        if dists.ndim == 1:  # это случается, когда k == 1
            dists = dists[:, None]
            idxs = idxs[:, None]

        candidates = []
        for p_i in range(platform_len):
            for rank in range(dists.shape[1]):
                dist = float(dists[p_i, rank])
                if not np.isfinite(dist) or dist > THRESHOLD_METERS:
                    continue
                s_i = int(idxs[p_i, rank])
                candidates.append((dist, p_i, s_i))

        candidates.sort(key=lambda t: t[0])
        for dist, p_i, s_i in candidates:
            if p_i not in matched_p and s_i not in matched_s:
                pairs[p_i] = s_i
                matched_p.add(p_i)
                matched_s.add(s_i)

    for p_i, s_i in pairs.items():
        items.append(
            {
                "p": platforms[p_i],
                "pref": platforms_refs[p_i],
                "s": stops[s_i],
                "sref": stops_refs[s_i],
            }
        )

    if stops_len:
        base_dir = _side_left_or_right(Point(platforms[platform_len // 2])) if platform_len else 1
        for s_i in range(stops_len):
            if s_i in matched_s:
                continue
            s_pt = Point(stops[s_i])
            items.append(
                {
                    "p": _offset_point(s_pt, base_dir),
                    "pref": f"from_{stops_refs[s_i]}",  # new platform
                    "s": s_pt.xy,
                    "sref": stops_refs[s_i],
                }
            )

    for p_i in range(platform_len):
        if p_i in matched_p:
            continue
        items.append(
            {
                "p": platforms[p_i],
                "pref": platforms_refs[p_i],
                "s": None,
                "sref": f"from_{platforms_refs[p_i]}",
            }
        )

    if not items:
        items.append({"p": _offset_point(path.interpolate(0), 1, 7), "pref": None, "s": None, "sref": None})
        items.append({"p": _offset_point(path.interpolate(path.length), 1, 7), "pref": None, "s": None, "sref": None})

    items.sort(key=lambda d: path.project(Point(d["p"])))

    graph_data: list[dict] = []
    node_id = 0
    last_dist = None
    last_projected_stop_id = None

    def add_node(x, y, node_type=None, ref_id=None):
        x = round(x, 5)
        y = round(y, 5)
        graph_data.append(
            {
                "node_id": (loc_id, node_id),
                "point": (x, y),
                "route": name,
                "type": (node_type if node_type else transport_type),
                "ref_id": ref_id,
            }
        )

    def add_edge(u, v, geometry=None, transport=None):

        payload = {
            "u": (loc_id, u),
            "v": (loc_id, v),
            "route": name,
            "type": (transport_type if transport else "boarding"),
            "extra_data": extra_data,
        }
        if geometry is not None:
            payload["geometry"] = LineString([(round(x, 5), round(y, 5)) for x, y in geometry.coords])
        graph_data.append(payload)

    for item in items:
        p_xy = item["p"]
        p_ref = item.get("pref")

        s_xy = item["s"]
        if s_xy is None:
            s_xy = p_xy
        s_ref = item.get("sref")

        platform = Point(p_xy)
        stop = Point(s_xy)

        stop_dist = path.project(stop)

        projected_stop = path.interpolate(stop_dist)

        # Платформы лежат нереалистично далеко от спроецированных остановок, ошибка в данных осм
        if projected_stop.distance(platform) > 100:
            continue

        add_node(projected_stop.x, projected_stop.y, ref_id=s_ref)

        if last_dist is not None:
            seg = substring(path, last_dist, stop_dist)
            if isinstance(seg, Point):
                seg = LineString((seg, seg))
            add_edge(last_projected_stop_id, node_id, geometry=seg, transport=True)

        last_projected_stop_id = node_id
        last_dist = stop_dist
        node_id += 1

        add_node(platform.x, platform.y, node_type="platform", ref_id=p_ref)
        boarding_geom = LineString(
            [(round(projected_stop.x, 5), round(projected_stop.y, 5)), (round(platform.x, 5), round(platform.y, 5))]
        )
        add_edge(node_id - 1, node_id, geometry=boarding_geom)
        add_edge(node_id, node_id - 1, geometry=boarding_geom)
        node_id += 1

    to_return = pd.DataFrame(graph_data)
    return to_return


def parse_overpass_to_edgenode(loc, crs, needed_tags) -> pd.DataFrame | None:
    loc_id = loc.name
    transport_type = loc.transport_type
    parsed_geometry = parse_overpass_route_response(loc, crs, needed_tags, loc_id)
    edgenode = geometry_to_graph_edge_node_df(parsed_geometry, transport_type, loc_id)
    return edgenode


def infer_role_from_tags(tags: dict) -> str:
    _TRUE = {"yes", "true", "1"}

    def _is_true(v):
        return str(v).lower() in _TRUE

    if not tags:
        return ""

    pt = (tags.get("public_transport", "")).lower()
    rail = (tags.get("railway", "")).lower()
    stat = (tags.get("station", "")).lower()
    entr = (tags.get("entrance", "")).lower()
    entry = (tags.get("entry", "")).lower()
    exit_ = (tags.get("exit", "")).lower()

    # station
    if pt == "station" or rail == "station" or stat == "subway":
        return "station"

    # platform
    if pt == "platform" or rail == "platform":
        return "platform"

    # stop / stop_position
    if pt in {"stop_position", "stop"} or rail in {"stop", "halt"}:
        return "stop"

    # entrances
    if rail == "subway_entrance" or entr in {"yes", "main", "service", "entrance", "entry", "exit"}:
        if entr == "entry" or (_is_true(entry) and not _is_true(exit_)):
            return "entry_only"
        if entr == "exit" or (_is_true(exit_) and not _is_true(entry)):
            return "exit_only"
        return "entrance"
    return ""


def patch_members_roles_inplace(stop_areas_df):

    missing = []
    for _, r in stop_areas_df.iterrows():
        for m in r.get("members") or []:
            role = (m.get("role") or "").strip()
            if role == "":
                missing.append({"type": m["type"], "ref": int(m["ref"])})

    if not missing:
        return

    tags_map = fetch_member_tags(missing)

    for _, r in stop_areas_df.iterrows():
        for m in r.get("members") or []:
            if (m.get("role") or "").strip():
                continue
            key = (m["type"], int(m["ref"]))
            tags = tags_map.get(key, {})
            role = infer_role_from_tags(tags)
            if role:
                m["role"] = role
            if "lon" not in m and "__center__" in tags:
                cx, cy = tags["__center__"]
                m["lon"], m["lat"] = cx, cy


def parse_overpass_subway_data(
    stop_areas, stop_areas_group, stations_data, to_crs
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    graph_nodes = []
    graph_edges = []

    transformer = Transformer.from_crs("EPSG:4326", to_crs, always_xy=True)

    station_parent_ref = {}
    osm_roles_to_iduedu = {
        "station": "subway_station",
        "stop": "subway",
        "platform": "subway_platform",
        "entrance": "subway_entry_exit",
        "subway_entrance": "subway_entry_exit",
        "entrance_yes": "subway_entry_exit",
        "entrance_main": "subway_entry_exit",
        "entry_only": "subway_entry",
        "entrance_entry_only": "subway_entry",
        "entry": "subway_entry",
        "exit_only": "subway_exit",
        "entrance_exit_only": "subway_exit",
        "exit": "subway_exit",
    }

    def _collect_by_roles(members, parent_id):
        roles = defaultdict(list)
        for m in members or []:
            role = (m.get("role") or "").lower()
            roles[role].append(m)

        stations = roles.get("station", [])
        for s in stations:
            station_parent_ref[parent_id] = s["ref"]
        stops = roles.get("stop", [])
        platforms = roles.get("platform", [])

        entrances = (
            roles.get("entrance", [])
            + roles.get("subway_entrance", [])
            + roles.get("entrance_yes", [])
            + roles.get("entrance_main", [])
        )
        entry_only = roles.get("entry_only", []) + roles.get("entrance_entry_only", []) + roles.get("entry", [])
        exit_only = roles.get("exit_only", []) + roles.get("entrance_exit_only", []) + roles.get("exit", [])

        return stations, stops, platforms, entrances, entry_only, exit_only

    def add_node(ref_id, x, y, node_type):

        if x is None or y is None:
            point = None
        else:
            point = transformer.transform(x, y)

        graph_nodes.append(
            {
                "ref_id": int(ref_id),
                "point": point,
                "type": osm_roles_to_iduedu.get(node_type),
            }
        )

    def add_edge(u, v, edge_type):
        graph_edges.append(
            {
                "u_ref": int(u),
                "v_ref": int(v),
                "type": edge_type,
            }
        )

    patch_members_roles_inplace(stop_areas)

    for _, stop_area in stop_areas.iterrows():

        members = stop_area["members"]
        all_nodes = _collect_by_roles(members, stop_area["id"])
        start_idx = len(graph_nodes)

        for node in chain.from_iterable(all_nodes):
            if "geometry" in node:
                new_lon, new_lat = LineString((xy["lon"], xy["lat"]) for xy in node["geometry"]).centroid.xy
                node["lon"], node["lat"] = new_lon[0], new_lat[0]
            if "lon" not in node:
                node["lon"], node["lat"] = None, None

            add_node(node["ref"], node["lon"], node["lat"], node["role"])

        ref2idx = {graph_nodes[i]["ref_id"]: i for i in range(start_idx, len(graph_nodes))}

        stations, stops, platforms, entrances, entry_only, exit_only = all_nodes

        for stop in stops:
            for platform in platforms:
                add_edge(stop["ref"], platform["ref"], "boarding")
                add_edge(platform["ref"], stop["ref"], "boarding")

        has_station = len(stations) > 0

        if has_station:
            for platform in platforms:
                for station in stations:
                    add_edge(station["ref"], platform["ref"], "subway_station")
                    add_edge(platform["ref"], station["ref"], "subway_station")

        targets = [s["ref"] for s in stations] if has_station else [p["ref"] for p in platforms]

        has_entrance = (len(entrances) + len(entry_only)) > 0
        has_exit = (len(entrances) + len(exit_only)) > 0

        for entrance in entrances:
            for t in targets:
                add_edge(entrance["ref"], t, "subway_entrance")
                add_edge(t, entrance["ref"], "subway_exit")

        for entrance in entry_only:
            for t in targets:
                add_edge(entrance["ref"], t, "subway_entrance")

        for entrance in exit_only:
            for t in targets:
                add_edge(t, entrance["ref"], "subway_exit")

        if not (has_entrance and has_exit):
            for t in targets:
                idx = ref2idx.get(int(t))
                if idx is not None:
                    graph_nodes[idx]["type"] = "platform"

    for _, stop_area_group in stop_areas_group.iterrows():
        members = stop_area_group["members"]

        connect_refs = [station_parent_ref[mem["ref"]] for mem in members if mem["ref"] in station_parent_ref]

        pairs = list(combinations(connect_refs, 2))
        for pair in pairs:
            if pair[0] != pair[1]:
                add_edge(pair[0], pair[1], "subway_transfer")
                add_edge(pair[1], pair[0], "subway_transfer")

    edges_gdf = pd.DataFrame(graph_edges)
    nodes_gdf = pd.DataFrame(graph_nodes)
    nodes_gdf["extra_data"] = {}

    for _, station_data in stations_data.iterrows():
        tags = station_data["tags"]
        ref_id = station_data["id"]
        station_ind = nodes_gdf[nodes_gdf["ref_id"] == int(ref_id)].index
        extra_data = {k: v for k, v in tags.items() if k in ["name", "depth"]}
        extra_data["depth"] = extra_data.get("depth", 0)
        nodes_gdf.loc[station_ind, "extra_data"] = [extra_data for _ in station_ind]

    edges_gdf = edges_gdf.drop_duplicates(subset=["u_ref", "v_ref"])
    edges_gdf["length_meter"] = 0.0
    edges_gdf["time_min"] = 0.0
    edges_gdf["geometry"] = None

    def _depth_from_extra(d):
        if isinstance(d, dict):
            v = d.get("depth", 0)
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    if len(nodes_gdf) == 0:
        return None
    nodes_info = nodes_gdf[["ref_id", "point", "type", "extra_data"]].copy()
    nodes_info["depth_m"] = nodes_info["extra_data"].apply(_depth_from_extra)

    nodes_lut = nodes_info.set_index("ref_id")[["point", "type", "depth_m"]]

    edges_nodes = (
        edges_gdf.join(nodes_lut.add_prefix("u_"), on="u_ref")
        .join(nodes_lut.add_prefix("v_"), on="v_ref")
        .drop_duplicates(subset=["u_ref", "v_ref"])
    )

    def _horiz_len(pu, pv):
        if pu is None or pv is None:
            return np.nan
        try:
            return LineString([pu, pv]).length
        except Exception:
            return np.nan

    edges_nodes["horiz_m"] = edges_nodes.apply(lambda r: _horiz_len(r["u_point"], r["v_point"]), axis=1)

    edges_nodes["station_depth_m"] = np.where(
        edges_nodes["u_type"].eq("subway_station"),
        edges_nodes["u_depth_m"].fillna(0.0),
        np.where(edges_nodes["v_type"].eq("subway_station"), edges_nodes["v_depth_m"].fillna(0.0), 0.0),
    ).astype(float)

    ESCALATOR_SPEED_MPMIN = 0.75 * 60
    WALK_SPEED_MPMIN = 5 * 1000 / 60
    TRANSFER_PATH_FACTOR = 1.5

    mask_escal = edges_nodes["type"].isin(["subway_entrance", "subway_exit"]) & edges_nodes["horiz_m"].notna()
    edges_nodes.loc[mask_escal, "length_meter"] = np.hypot(
        edges_nodes.loc[mask_escal, "horiz_m"], edges_nodes.loc[mask_escal, "station_depth_m"]
    )
    edges_nodes.loc[mask_escal, "time_min"] = edges_nodes.loc[mask_escal, "length_meter"] / ESCALATOR_SPEED_MPMIN

    def _straight_geom(pu, pv):
        if pu is None or pv is None:
            return None
        try:
            return LineString([pu, pv])
        except Exception:
            return None

    edges_nodes.loc[mask_escal, "geometry"] = edges_nodes.loc[mask_escal].apply(
        lambda r: _straight_geom(r["u_point"], r["v_point"]), axis=1
    )

    mask_transfer = edges_nodes["type"].eq("subway_transfer") & edges_nodes["horiz_m"].notna()
    edges_nodes.loc[mask_transfer, "length_meter"] = TRANSFER_PATH_FACTOR * edges_nodes.loc[
        mask_transfer, "horiz_m"
    ].round(1)
    edges_nodes.loc[mask_transfer, "time_min"] = (
        edges_nodes.loc[mask_transfer, "length_meter"] / WALK_SPEED_MPMIN
    ).round(1)
    edges_nodes.loc[mask_transfer, "geometry"] = edges_nodes.loc[mask_transfer].apply(
        lambda r: _straight_geom(r["u_point"], r["v_point"]), axis=1
    )

    edges_gdf[["length_meter", "time_min", "geometry"]] = edges_nodes[["length_meter", "time_min", "geometry"]]
    return edges_gdf, nodes_gdf
