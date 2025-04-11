import math
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from shapely import Point
from shapely.geometry import LineString
from shapely.ops import substring

PLATFORM_ROLES = ["platform_entry_only", "platform", "platform_exit_only"]
STOPS_ROLES = ["stop", "stop_exit_only", "stop_entry_only"]


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


def parse_overpass_route_response(loc: dict, crs: CRS) -> pd.Series:
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def transform_geometry(loc):
        if isinstance(loc["geometry"], float):
            return transformer.transform(loc["lon"], loc["lat"])
        p = LineString([transformer.transform(coords["lon"], coords["lat"]) for coords in loc["geometry"]]).centroid
        return p.x, p.y

    def process_roles(route, roles):
        filtered = route[route["role"].isin(roles)]
        if len(filtered) == 0:
            return None
        return filtered.apply(transform_geometry, axis=1).tolist()

    if "ref" in loc["tags"].keys():
        transport_name = loc["tags"]["ref"]
    elif "name" in loc["tags"].keys():
        transport_name = loc["tags"]["name"]
    else:
        transport_name = None
    route = pd.DataFrame(loc["members"])

    if "geometry" not in route.columns:
        route["geometry"] = np.nan
    if "lat" not in route.columns:
        route["lat"] = np.nan
    if "lon" not in route.columns:
        route["lon"] = np.nan

    route = route.dropna(subset=["lat", "lon", "geometry"], how="all")

    platforms = process_roles(route, PLATFORM_ROLES)
    stops = process_roles(route, STOPS_ROLES)

    ways = route[(route["type"] == "way") & (route["role"] == "")]

    if len(ways) > 0:
        ways = ways["geometry"].reset_index(drop=True)
        ways = ways.apply(lambda x: ([transformer.transform(coords["lon"], coords["lat"]) for coords in x])).tolist()
        connected_ways = [[]]
        cur_way = 0
        for coords in ways:
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
            to_del = [i for i, data in enumerate(connected_ways) if data[0] == data[-1]]
            connected_ways = [i for j, i in enumerate(connected_ways) if j not in to_del]
        if len(connected_ways) > 1:
            connected_ways = _link_unconnected(connected_ways)
        else:
            connected_ways = connected_ways[0]

    else:
        connected_ways = None

    return pd.Series({"path": connected_ways, "platforms": platforms, "stops": stops, "route": transport_name})


def geometry_to_graph_edge_node_df(loc: pd.Series, transport_type, loc_id) -> DataFrame | None:
    graph_data = []
    node_id = 0
    name = loc.route
    last_dist = None
    last_projected_stop_id = None
    platforms = loc.platforms
    stops = loc.stops
    path = loc.path

    def add_node(desc, x, y, transport=None):
        x = round(x, 5)
        y = round(y, 5)
        if not transport:
            graph_data.append({"node_id": (loc_id, node_id), "point": (x, y), "route": name, "type": desc})
        else:
            graph_data.append({"node_id": (loc_id, node_id), "point": (x, y), "route": name, "type": transport_type})

    def add_edge(u, v, geometry=None, transport=None):
        if not transport:
            graph_data.append(
                {
                    "u": (loc_id, u),
                    "v": (loc_id, v),
                    "route": name,
                    "type": "boarding",
                }
            )
        else:
            graph_data.append(
                {
                    "u": (loc_id, u),
                    "v": (loc_id, v),
                    "geometry": LineString([(round(x, 5), round(y, 5)) for x, y in geometry.coords]),
                    "route": name,
                    "type": transport_type,
                }
            )

    def offset_direction(point):
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

    def offset_point(point, direction, distance=5):
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

    def process_platform(platform):
        nonlocal node_id, last_dist, last_projected_stop_id

        dist = path.project(platform)
        projected_stop = path.interpolate(dist)
        add_node("stop", projected_stop.x, projected_stop.y, transport=True)
        if last_dist is not None:
            cur_path = substring(path, last_dist, dist)
            if isinstance(cur_path, Point):
                cur_path = LineString((cur_path, cur_path))
            add_edge(last_projected_stop_id, node_id, geometry=cur_path, transport=True)
        last_projected_stop_id = node_id

        node_id += 1
        add_node("platform", platform.x, platform.y)
        add_edge(node_id - 1, node_id)
        add_edge(node_id, node_id - 1)
        node_id += 1
        return dist, last_projected_stop_id, node_id

    if not path:
        warnings.warn("В одном из маршрутов нет пути, пока не работаю с этим", UserWarning)
        return None
    path = LineString(path)

    # Если нет платформ
    if not platforms:
        if not stops:  # Строим маршрут по пути начало - конец
            platforms = [offset_point(path.interpolate(0), 1, 7), offset_point(path.interpolate(path.length), 1, 7)]
        else:  # Если есть только остановки - превращаем их в платформы
            platforms = [offset_point(Point(stop), 1, 7) for stop in stops]
            stops = None

    stops = [] if not stops else stops

    # Если остановок больше чем платформ - найти остановки без платформ и добавить новые платформы
    if len(stops) > len(platforms):
        stop_tree = cKDTree(stops)
        _, indices = stop_tree.query(platforms)
        connection = [(platforms[platform], stop) for platform, stop in enumerate(indices)]
        connection += [(-1, stop) for stop in set(range(len(stops))) ^ set(indices)]
        connection.sort(key=lambda x: x[1])
        direction = offset_direction(Point(platforms[len(platforms) // 2]))
        stops_to_platforms = {
            stop: offset_point(Point(stops[stop]), direction, 7) for stop in set(range(len(stops))) ^ set(indices)
        }
        platforms = [coord if (coord != -1) else (stops_to_platforms.get(ind)) for coord, ind in connection]
        stops = []

    # Если получилось только одна платформа
    if len(platforms) == 1:
        platform = Point(platforms[0])
        dist = path.project(platform)
        if dist in (path.length, 0):  # Если платформа является конечной
            platforms = [offset_point(path.interpolate(0), 1, 7), offset_point(path.interpolate(path.length), 1, 7)]
        else:  # Если платформа не является конечной
            platforms = [
                offset_point(path.interpolate(0), 1, 7),
                platform,
                offset_point(path.interpolate(path.length), 1, 7),
            ]
    platforms = [Point(coords) for coords in platforms]
    if len(platforms) >= len(stops):
        for platform in platforms:
            if not last_dist:
                last_dist, last_projected_stop_id, node_id = process_platform(platform)
                if last_dist > path.length / 2:
                    path = path.reverse()
                    last_dist = path.project(platform)
            else:
                last_dist, last_projected_stop_id, node_id = process_platform(platform)

    to_return = pd.DataFrame(graph_data)
    return to_return


def parse_overpass_to_edgenode(loc, crs) -> pd.DataFrame | None:
    loc_id = loc.name
    transport_type = loc.transport_type
    parsed_geometry = parse_overpass_route_response(loc, crs)
    edgenode = geometry_to_graph_edge_node_df(parsed_geometry, transport_type, loc_id)
    return edgenode
