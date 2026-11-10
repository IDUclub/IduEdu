"""Figures of the documentation, published on the ``assets`` branch.

    python docs/figures/make_figures.py                 # every figure
    python docs/figures/make_figures.py routing_search  # only the named ones

The graphs are built for Vasilyevsky Island, Saint Petersburg (OSM relation 1114252), from the Geofabrik
extract of the Northwestern Federal District (downloaded once, about 650 MB) and cached in
``.iduedu_cache/figures``. The PNG files are written to ``docs/figures/out``; the pages link them from
``https://raw.githubusercontent.com/IDUclub/IduEdu/assets/<name>.png``.
"""

import heapq
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
import numpy as np  # noqa: E402  pylint: disable=wrong-import-position

import iduedu  # noqa: E402  pylint: disable=wrong-import-position
from iduedu.graph.routing import (  # noqa: E402  pylint: disable=wrong-import-position
    _landmarks,
    _node_coordinates,
    _prepare_graph,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
CACHE = ROOT / ".iduedu_cache" / "figures"
OSM_ID = 1114252
REGION = "northwestern-fed-district"

# Colours (the reference palette of the dataviz guide): three categorical slots, a blue ramp, ink and surface.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2 = "#0b0b0b", "#52514e"
SURFACE, STREETS = "#fcfcfb", "#d6d4ce"
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "legend.frameon": False,
    }
)


# --- data -------------------------------------------------------------------------------------------------------


def _cached(name, build):
    path = CACHE / f"{name}.urbangraph"
    if path.exists():
        return iduedu.read_urban_graph(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    graph = build()
    graph.write(path)
    return graph


def boundary():
    path = CACHE / "boundary.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    with iduedu.use_pbf(osm_id=OSM_ID, region=REGION):
        result = iduedu.get_4326_boundary(osm_id=OSM_ID)
    frame = gpd.GeoDataFrame(geometry=[result], crs=4326)
    frame.to_parquet(path)
    return frame


def _build(builder, **kwargs):
    with iduedu.use_pbf(osm_id=OSM_ID, region=REGION):
        return builder(territory=boundary().geometry.iloc[0], clip_by_territory=True, **kwargs)


def walk_graph():
    return _cached("walk", lambda: _build(iduedu.get_walk_graph))


def intermodal_graph():
    return _cached("intermodal", lambda: _build(iduedu.get_intermodal_graph))


def drive_graph():
    return _cached("drive", lambda: _build(iduedu.get_drive_graph))


def bike_graph():
    return _cached("bike", lambda: _build(iduedu.get_bike_graph))


def walk_segments():
    """The walk graph without merging: one edge per straight segment of every way."""
    return _cached("walk_segments", lambda: _build(iduedu.get_walk_graph, simplify=False))


# --- drawing ----------------------------------------------------------------------------------------------------


def frame(ax, graph, *, streets=True, edges=None):
    """Background: the walkable streets in light grey and the district boundary; no axes."""
    if streets:
        lines = graph.edges_gdf if edges is None else edges
        lines.plot(ax=ax, color=STREETS, linewidth=0.4, zorder=1)
    boundary().to_crs(graph.crs).boundary.plot(ax=ax, color=INK, linewidth=0.8, zorder=2)
    ax.set_axis_off()
    ax.set_aspect("equal")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=120, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def node_xy(graph):
    """Node coordinates in the order of the adjacency matrix (``graph.adjacency_nodelist``)."""
    _prepare_graph(graph, weight="time_min", cutoff=None, reverse=False)
    xs, ys = _node_coordinates(graph)
    return xs, ys


def thousands(value):
    return f"{value:,}".replace(",", " ")


def nearest(xs, ys, x, y):
    return int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))


# --- reference searches that record the settled nodes ------------------------------------------------------------


def _csr(graph, weight):
    _prepare_graph(graph, weight=weight, cutoff=None, reverse=False)
    m = graph.adjacency_matrix.tocsr()
    return m.indptr, m.indices, m.data.astype(float)


def _transpose(indptr, indices, data, n):
    import scipy.sparse as sp  # pylint: disable=import-outside-toplevel

    t = sp.csr_matrix((data, indices, indptr), shape=(n, n)).T.tocsr()
    return t.indptr, t.indices, t.data


def search_dijkstra(csr, s, t, estimate=None):
    """Dijkstra (A* with ``estimate``) from s to t: cost, route and the nodes settled in order."""
    indptr, indices, data = csr
    dist, parent, settled = {s: 0.0}, {s: -1}, []
    done = set()
    heap = [(estimate(s) if estimate else 0.0, s)]
    while heap:
        _, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        settled.append(u)
        if u == t:
            break
        for k in range(indptr[u], indptr[u + 1]):
            v, nd = indices[k], dist[u] + data[k]
            if nd < dist.get(v, np.inf):
                dist[v], parent[v] = nd, u
                heapq.heappush(heap, (nd + (estimate(v) if estimate else 0.0), v))
    route, node = [], t
    while node != -1:
        route.append(node)
        node = parent[node]
    return dist[t], route[::-1], settled


def search_bidirectional(csr, rcsr, s, t):
    """Bidirectional Dijkstra: cost, route and the nodes settled from each end."""
    sides = [
        {"csr": csr, "dist": {s: 0.0}, "parent": {s: -1}, "heap": [(0.0, s)], "done": set(), "settled": []},
        {"csr": rcsr, "dist": {t: 0.0}, "parent": {t: -1}, "heap": [(0.0, t)], "done": set(), "settled": []},
    ]
    best, meet = np.inf, None

    def top(side):
        while side["heap"] and side["heap"][0][1] in side["done"]:
            heapq.heappop(side["heap"])
        return side["heap"][0][0] if side["heap"] else np.inf

    while True:
        tf, tb = top(sides[0]), top(sides[1])
        if tf + tb >= best:
            break
        index = 0 if tf <= tb else 1
        this, other = sides[index], sides[1 - index]
        _, u = heapq.heappop(this["heap"])
        this["done"].add(u)
        this["settled"].append(u)
        indptr, indices, data = this["csr"]
        for k in range(indptr[u], indptr[u + 1]):
            v, nd = indices[k], this["dist"][u] + data[k]
            if v in other["dist"] and nd + other["dist"][v] < best:
                best = nd + other["dist"][v]
                meet = (u, v) if index == 0 else (v, u)
            if nd < this["dist"].get(v, np.inf):
                this["dist"][v], this["parent"][v] = nd, u
                heapq.heappush(this["heap"], (nd, v))
    route, node = [], meet[0]
    while node != -1:
        route.append(node)
        node = sides[0]["parent"][node]
    route = route[::-1]
    node = meet[1]
    while node != -1:
        route.append(node)
        node = sides[1]["parent"][node]
    return best, route, sides[0]["settled"], sides[1]["settled"]


def all_costs(csr, source):
    """Costs from source to every node (Dijkstra without a target)."""
    indptr, indices, data = csr
    n = len(indptr) - 1
    dist = np.full(n, np.inf)
    dist[source] = 0.0
    heap, done = [(0.0, source)], np.zeros(n, dtype=bool)
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for k in range(indptr[u], indptr[u + 1]):
            v = indices[k]
            if d + data[k] < dist[v]:
                dist[v] = d + data[k]
                heapq.heappush(heap, (dist[v], v))
    return dist


# --- routing figures --------------------------------------------------------------------------------------------


def routing_search():
    """Nodes settled by Dijkstra, the bidirectional search, A* and ALT for one walking route."""
    graph = walk_graph()
    weight = "length_meter"
    csr = _csr(graph, weight)
    n = len(csr[0]) - 1
    rcsr = _transpose(*csr, n)
    xs, ys = node_xy(graph)
    x0, y0 = np.median(xs), np.median(ys)
    s = nearest(xs, ys, x0 - 1600, y0 - 300)
    t = nearest(xs, ys, x0 + 1400, y0 + 500)

    cost_d, route_d, settled_d = search_dijkstra(csr, s, t)
    cost_b, route_b, settled_f, settled_r = search_bidirectional(csr, rcsr, s, t)
    # A*: the scale is the smallest cost per metre of straight distance over the edges.
    lengths = np.hypot(xs[csr[1]] - np.repeat(xs, np.diff(csr[0])), ys[csr[1]] - np.repeat(ys, np.diff(csr[0])))
    scale = np.min(csr[2][lengths > 0] / lengths[lengths > 0]) * (1 - 1e-6)
    cost_a, route_a, settled_a = search_dijkstra(
        csr, s, t, estimate=lambda v: scale * np.hypot(xs[v] - xs[t], ys[v] - ys[t])
    )
    # ALT with the landmarks IduEdu picks for this graph.
    landmarks = _landmarks(graph).nodes
    from_l = [all_costs(csr, int(lm)) for lm in landmarks]
    to_l = [all_costs(rcsr, int(lm)) for lm in landmarks]

    def alt_estimate(v):
        best = 0.0
        for fl, tl in zip(from_l, to_l):
            for bound in (fl[t] - fl[v], tl[v] - tl[t]):
                if np.isfinite(bound) and bound > best:
                    best = bound
        return best

    cost_l, route_l, settled_l = search_dijkstra(csr, s, t, estimate=alt_estimate)
    library, _ = iduedu.shortest_path(graph, graph.adjacency_nodelist[s], graph.adjacency_nodelist[t], weight=weight)
    for cost in (cost_d, cost_b, cost_a, cost_l):
        assert abs(cost - library) <= 1e-3 * library, (cost, library)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    panels = [
        ("Dijkstra", [(settled_d, BLUE)], route_d),
        ("Bidirectional Dijkstra", [(settled_f, BLUE), (settled_r, ORANGE)], route_b),
        ("A*, straight-line estimate", [(settled_a, BLUE)], route_a),
        ("ALT, 16 landmarks", [(settled_l, BLUE)], route_l),
    ]
    for ax, (title, groups, route) in zip(axes.flat, panels):
        frame(ax, graph)
        count = 0
        for nodes, colour in groups:
            nodes = np.asarray(nodes)
            ax.scatter(xs[nodes], ys[nodes], s=2.5, color=colour, alpha=0.55, linewidths=0, zorder=3)
            count += len(nodes)
        route = np.asarray(route)
        ax.plot(xs[route], ys[route], color=INK, linewidth=2.2, zorder=5, solid_capstyle="round")
        ax.scatter(xs[[s, t]], ys[[s, t]], s=60, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=6)
        if title.startswith("ALT"):
            ax.scatter(
                xs[landmarks], ys[landmarks], s=45, marker="^", color=AQUA, edgecolor=SURFACE, linewidth=1, zorder=6
            )
        ax.set_title(f"{title}: {thousands(count)} nodes settled")
    fig.suptitle(
        f"One walking route of {library / 1000:.1f} km on Vasilyevsky Island ({thousands(n)} nodes)", fontsize=14
    )
    fig.tight_layout()
    save(fig, "routing_search_space")


def routing_alternatives():
    """Yen's k shortest routes against the penalty method for one walking route."""
    graph = walk_graph()
    xs, ys = node_xy(graph)
    ids = graph.adjacency_nodelist
    x0, y0 = np.median(xs), np.median(ys)
    s = ids[nearest(xs, ys, x0 - 1500, y0 - 200)]
    t = ids[nearest(xs, ys, x0 + 1500, y0 + 400)]
    yen = iduedu.k_shortest_paths(graph, s, t, 3, weight="length_meter")
    alternatives = iduedu.alternative_paths(graph, s, t, 3, weight="length_meter")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    colours = [BLUE, ORANGE, AQUA]
    for ax, title, table in (
        (axes[0], "k_shortest_paths (Yen), k = 3", yen),
        (axes[1], "alternative_paths, penalty 1.5", alternatives),
    ):
        frame(ax, graph)
        for rank, row in table.iterrows():
            edges = graph.path_to_edges(row["path"], weight="length_meter")
            label = f"{rank}: {row['dist'] / 1000:.2f} km"
            if "overlap" in table:
                label += f", {row['overlap'] * 100:.0f} % shared"
            edges.plot(ax=ax, color=colours[rank], linewidth=4.2 - 1.1 * rank, label=label, zorder=4 + rank)
        ax.legend(loc="lower left", title="rank: length", fontsize=10)
        ax.set_title(title)
    fig.tight_layout()
    save(fig, "routing_alternatives")


def routing_single_source():
    """Travel time from one place on the intermodal graph, and the nearest of several sources."""
    graph = intermodal_graph()
    walk_edges = graph.edges_gdf[graph.edges_gdf["type"] == "walk"]
    walk_nodes = np.unique(np.concatenate([walk_edges["u"].to_numpy(), walk_edges["v"].to_numpy()]))
    xs, ys = node_xy(graph)
    ids = np.asarray(graph.adjacency_nodelist)
    position = {node: i for i, node in enumerate(ids)}
    x0, y0 = np.median(xs), np.median(ys)
    source = ids[nearest(xs, ys, x0 + 1200, y0 + 300)]
    times = iduedu.single_source_dijkstra(graph, source, weight="time_min", cutoff=30, output="distances")
    times = times.sparse.to_dense()
    times = times[times.index.isin(walk_nodes)]
    # Subway entrances (the walking side of the entrance edges) and the station each one leads to.
    entrances = graph.edges_gdf[graph.edges_gdf["type"] == "subway_entrance"]
    station_of = dict(zip(entrances["u"], entrances["v"]))
    nearest_entrance = iduedu.multi_source_dijkstra_nearest_source(
        graph, source_nodes=list(station_of), weight="time_min", cutoff=40
    )
    nearest_entrance = nearest_entrance[nearest_entrance.index.isin(walk_nodes)]
    station_names = {
        "Горный институт": "Gorny Institut",
        "Василеостровская": "Vasileostrovskaya",
        "Приморская": "Primorskaya",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    frame(axes[0], graph, edges=walk_edges)
    p = [position[node] for node in times.index]
    sc = axes[0].scatter(
        xs[p], ys[p], c=times.to_numpy(), cmap=BLUES.reversed(), s=3, vmin=0, vmax=30, linewidths=0, zorder=3
    )
    axes[0].scatter(xs[position[source]], ys[position[source]], s=80, color=INK, edgecolor=SURFACE, zorder=6)
    cbar = fig.colorbar(sc, ax=axes[0], shrink=0.6, pad=0.01)
    cbar.set_label("minutes, walking and public transport")
    axes[0].set_title("Travel time from one place, up to 30 min")

    frame(axes[1], graph, edges=walk_edges)
    stations = sorted(set(station_of.values()))
    colour_of = dict(zip(stations, (BLUE, ORANGE, AQUA)))
    owners = nearest_entrance["source_node"].map(station_of)
    p = [position[node] for node in nearest_entrance.index]
    axes[1].scatter(xs[p], ys[p], c=owners.map(colour_of).tolist(), s=3, linewidths=0, zorder=3)
    for station in stations:
        doors = [position[node] for node, st in station_of.items() if st == station]
        name = str(graph.nodes_gdf.loc[station, "name"])
        axes[1].scatter(
            xs[doors],
            ys[doors],
            s=80,
            color=colour_of[station],
            edgecolor=INK,
            linewidth=1.2,
            zorder=6,
            label=station_names.get(name, name),
        )
    axes[1].legend(loc="lower left", title="subway station (entrances)", fontsize=10, markerscale=0.8)
    axes[1].set_title("Nearest subway entrance, coloured by station")
    fig.tight_layout()
    save(fig, "routing_single_source")


# --- network analysis figures -----------------------------------------------------------------------------------


def buildings():
    """Buildings of the example data (Vasilyevsky Island) with ``resident_number``, in the graph CRS."""
    return gpd.read_parquet(ROOT / "docs" / "examples" / "data" / "spb_buildings.parquet")


def colour_bar(fig, ax, mappable, label):
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label(label)
    cbar.outline.set_visible(False)


def square_grid(graph, size):
    """Squares of ``size`` metres over the district, clipped to its boundary."""
    from shapely.geometry import box  # pylint: disable=import-outside-toplevel

    area = boundary().to_crs(graph.crs).geometry.iloc[0]
    minx, miny, maxx, maxy = area.bounds
    cells = [box(x, y, x + size, y + size) for x in np.arange(minx, maxx, size) for y in np.arange(miny, maxy, size)]
    grid = gpd.GeoDataFrame(geometry=cells, crs=graph.crs)
    grid = grid[grid.intersects(area)].copy()
    grid["geometry"] = grid.intersection(area)
    # Cells mostly outside the land give unstable densities.
    return grid[grid.area > 0.5 * size * size].reset_index(drop=True)


def analysis_statistics():
    """graph_stats by zones and local_stats around every node of the walk graph."""
    graph = walk_graph()
    grid = square_grid(graph, 500)
    zones = iduedu.graph_stats(graph, zones=grid)
    grid["intersection_density"] = zones["intersection_density"].to_numpy()
    local = iduedu.local_stats(graph, 400)
    xs, ys = node_xy(graph)
    position = {node: i for i, node in enumerate(graph.adjacency_nodelist)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    frame(axes[0], graph, streets=False)
    grid.plot(ax=axes[0], column="intersection_density", cmap=BLUES, edgecolor=SURFACE, linewidth=1.5, zorder=1)
    graph.edges_gdf.plot(ax=axes[0], color=INK, linewidth=0.15, alpha=0.35, zorder=2)
    sm = plt.cm.ScalarMappable(cmap=BLUES, norm=plt.Normalize(*grid["intersection_density"].agg(["min", "max"])))
    colour_bar(fig, axes[0], sm, "intersections per km²")
    axes[0].set_title("graph_stats by 500 m squares: intersection density")

    frame(axes[1], graph)
    p = [position[node] for node in local.index]
    values = local["street_length_400"].to_numpy() / 1000
    sc = axes[1].scatter(xs[p], ys[p], c=values, cmap=BLUES, s=2.5, linewidths=0, zorder=3)
    colour_bar(fig, axes[1], sc, "km of walkable street within 400 m")
    axes[1].set_title("local_stats: street length within 400 m of every node")
    fig.tight_layout()
    save(fig, "analysis_statistics")


def analysis_objects():
    """Residents within a 10-minute walk, and the walk to the nearest public-transport platform."""
    walk = walk_graph()
    homes = buildings().to_crs(walk.crs)
    homes = homes[homes["resident_number"].fillna(0) > 0].copy()
    homes["geometry"] = homes.geometry.representative_point()
    xs, ys = node_xy(walk)
    position = {node: i for i, node in enumerate(walk.adjacency_nodelist)}
    residents = iduedu.aggregate_objects(
        walk, 10, origins_nodes=walk.adjacency_nodelist, gdf_objects=homes, columns=["resident_number"]
    )

    graph = intermodal_graph()
    platforms = graph.nodes_gdf[graph.nodes_gdf["type"] == "platform"]
    walk_part = iduedu.subgraph_by_nodes(
        graph, np.unique(graph.edges_gdf.loc[graph.edges_gdf["type"] == "walk", ["u", "v"]].to_numpy())
    )
    stops = platforms[platforms.index.isin(walk_part.nodes_gdf.index)]
    origins = buildings().to_crs(walk.crs)
    origins["geometry"] = origins.geometry.representative_point()
    nearest_stop = iduedu.nearest_objects(
        walk_part, gdf_origins=origins, object_nodes=stops.index, k=1, weight="time_min"
    )
    minutes = nearest_stop.xs(1, level="rank")["dist"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    frame(axes[0], walk)
    p = [position[node] for node in residents.index]
    sc = axes[0].scatter(
        xs[p],
        ys[p],
        c=residents["resident_number_sum_10"].fillna(0).to_numpy(),
        cmap=BLUES,
        s=2.5,
        linewidths=0,
        zorder=3,
    )
    colour_bar(fig, axes[0], sc, "residents within a 10-minute walk")
    axes[0].set_title("aggregate_objects: residents within 10 min")

    frame(axes[1], walk)
    footprints = buildings().to_crs(walk.crs)
    footprints["minutes"] = minutes.reindex(footprints.index).to_numpy()
    footprints.plot(ax=axes[1], column="minutes", cmap=BLUES, vmin=0, vmax=8, zorder=3)
    stops.plot(ax=axes[1], color=ORANGE, markersize=10, zorder=4, label="platform")
    sm = plt.cm.ScalarMappable(cmap=BLUES, norm=plt.Normalize(0, 8))
    colour_bar(fig, axes[1], sm, "minutes to the nearest platform")
    axes[1].legend(loc="lower left")
    axes[1].set_title("nearest_objects: walk from every building to the nearest stop")
    fig.tight_layout()
    save(fig, "analysis_objects")


def analysis_centrality():
    """Harmonic closeness of the nodes and betweenness of the edges of the walk graph."""
    graph = walk_graph()
    closeness = iduedu.centrality(graph, ["harmonic"], cutoffs=[800])
    betweenness = iduedu.edge_betweenness(graph, cutoffs=[2000])
    xs, ys = node_xy(graph)
    position = {node: i for i, node in enumerate(graph.adjacency_nodelist)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    frame(axes[0], graph)
    p = [position[node] for node in closeness.index]
    sc = axes[0].scatter(
        xs[p], ys[p], c=closeness["harmonic_800"].to_numpy(), cmap=BLUES, s=2.5, linewidths=0, zorder=3
    )
    colour_bar(fig, axes[0], sc, "Σ 1 / d over the nodes within 800 m (1/m)")
    axes[0].set_title("centrality: harmonic closeness within 800 m")

    frame(axes[1], graph, streets=False)
    edges = graph.edges_gdf.assign(b=betweenness["betweenness_2000"].to_numpy() / 1000)
    edges = edges.sort_values("b")
    top = edges["b"].quantile(0.995)
    norm = plt.Normalize(0, top)
    edges.plot(
        ax=axes[1],
        column="b",
        cmap=BLUES,
        norm=norm,
        linewidth=0.3 + 2.7 * np.clip(edges["b"].to_numpy() / top, 0, 1),
        zorder=3,
    )
    colour_bar(
        fig, axes[1], plt.cm.ScalarMappable(cmap=BLUES, norm=norm), "thousand shortest paths within 2 km along the edge"
    )
    axes[1].set_title("edge_betweenness within 2 km")
    fig.tight_layout()
    save(fig, "analysis_centrality")


def analysis_elevation():
    """Profile of a walk across a bridge: raster heights against the graph, and the grade speed models."""
    import rasterio  # pylint: disable=import-outside-toplevel

    from iduedu.graph.elevation import cycling, tobler  # pylint: disable=import-outside-toplevel

    raster = ROOT / ".iduedu_cache" / "dem" / "spb_glo30.tif"
    graph = walk_graph()
    graph = iduedu.add_elevation(graph, raster)
    bridges = graph.edges_gdf[graph.edges_gdf["bridge"].fillna("no").astype(str) != "no"]
    xs, ys = node_xy(graph)
    ids = graph.adjacency_nodelist
    # The longest bridge that a walk of 250 m before and after it actually crosses.
    for _, bridge in bridges.sort_values("length_meter", ascending=False).iterrows():
        u, v = ids.index(bridge["u"]), ids.index(bridge["v"])
        unit = np.array([xs[v] - xs[u], ys[v] - ys[u]]) / bridge["length_meter"]
        s = ids[nearest(xs, ys, *(np.array([xs[u], ys[u]]) - 250 * unit))]
        t = ids[nearest(xs, ys, *(np.array([xs[v], ys[v]]) + 250 * unit))]
        _, route = iduedu.shortest_path(graph, s, t, weight="length_meter")
        edges = graph.path_to_edges(route, weight="length_meter")
        if edges["bridge"].fillna("no").astype(str).ne("no").any():
            break
    from shapely.geometry import LineString  # pylint: disable=import-outside-toplevel

    # Edge geometries turned to the direction of the walk and joined into one line.
    coords = []
    for node, geom in zip(route, edges.geometry):
        part = np.asarray(geom.coords)
        start = np.array([xs[ids.index(node)], ys[ids.index(node)]])
        if np.hypot(*(part[0] - start)) > np.hypot(*(part[-1] - start)):
            part = part[::-1]
        coords.append(part)
    line = LineString(np.concatenate(coords))
    distances = np.arange(0, line.length, 5.0)
    points = [line.interpolate(d) for d in distances]
    coords = gpd.GeoSeries(points, crs=graph.crs).to_crs(rasterio.open(raster).crs)
    with rasterio.open(raster) as src:
        surface = np.array([value[0] for value in src.sample([(p.x, p.y) for p in coords])], dtype=float)
    node_distance = np.concatenate([[0.0], np.cumsum(edges["length_meter"].to_numpy())])
    node_height = graph.nodes_gdf.loc[route, "elevation"].to_numpy()
    on_bridge = edges["bridge"].fillna("no").astype(str).ne("no").to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.plot(distances, surface, color=INK_2, linewidth=1.2, label="raster (Copernicus GLO-30 surface model)")
    ax.plot(node_distance, node_height, color=BLUE, linewidth=2.2, label="graph nodes after add_elevation")
    for i in np.flatnonzero(on_bridge):
        ax.axvspan(node_distance[i], node_distance[i + 1], color=ORANGE, alpha=0.12, linewidth=0)
    span = np.flatnonzero(on_bridge)
    ax.text(
        (node_distance[span[0]] + node_distance[span[-1] + 1]) / 2,
        0.45 * ax.get_ylim()[1],
        "bridge",
        color=ORANGE,
        ha="center",
        va="center",
    )
    ax.set_xlabel("distance along the walk, m")
    ax.set_ylabel("height, m")
    ax.legend(loc="upper right")
    ax.set_title(f"Walk across a bridge of {bridge['length_meter']:.0f} m")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = axes[1]
    grades = np.linspace(-0.15, 0.15, 301)
    ax.plot(grades * 100, tobler(grades), color=BLUE, linewidth=2, label="tobler (walking)")
    ax.plot(grades * 100, cycling(grades), color=ORANGE, linewidth=2, label="cycling")
    ax.axhline(1, color=INK_2, linewidth=0.8)
    ax.set_xlabel("grade, %")
    ax.set_ylabel("speed relative to flat ground")
    ax.legend(loc="upper right")
    ax.set_title("Speed models of apply_grade_speed")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, "analysis_elevation")


# --- building figures -------------------------------------------------------------------------------------------

YELLOW = "#eda100"
MODE_COLOURS = {"bus": BLUE, "trolleybus": ORANGE, "tram": AQUA, "subway": YELLOW}


def building_streets():
    """Drive graph by speed and bike graph by level of traffic stress."""
    drive, bike = drive_graph(), bike_graph()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    frame(axes[0], drive, streets=False)
    edges = drive.edges_gdf.assign(speed=drive.edges_gdf["length_meter"] / drive.edges_gdf["time_min"] * 0.06)
    edges = edges.sort_values("speed")
    norm = plt.Normalize(edges["speed"].min(), edges["speed"].max())
    edges.plot(ax=axes[0], column="speed", cmap=BLUES, norm=norm, linewidth=1.0, zorder=3)
    colour_bar(fig, axes[0], plt.cm.ScalarMappable(cmap=BLUES, norm=norm), "km/h, length over time_min")
    axes[0].set_title(f"get_drive_graph: {thousands(len(drive.edges_gdf))} directed edges")

    frame(axes[1], bike, streets=False)
    steps = ["#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
    for level, colour in zip(range(1, 5), steps):
        part = bike.edges_gdf[bike.edges_gdf["lts"] == level]
        share = part["length_meter"].sum() / bike.edges_gdf["length_meter"].sum()
        part.plot(
            ax=axes[1], color=colour, linewidth=0.5 + 0.35 * level, zorder=2 + level, label=f"{level}  ({share:.0%})"
        )
    axes[1].legend(loc="lower left", title="lts (share of length)")
    axes[1].set_title("get_bike_graph: level of traffic stress")
    fig.tight_layout()
    save(fig, "building_streets")


def building_simplify():
    """A block of the walk graph with simplify=False and simplify=True."""
    raw, merged = walk_segments(), walk_graph()
    centre = merged.nodes_gdf.geometry.union_all().centroid
    x0, y0 = centre.x + 900, centre.y - 300
    half = 230
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, graph, title in (
        (axes[0], raw, "simplify=False"),
        (axes[1], merged, "simplify=True"),
    ):
        nodes = graph.nodes_gdf.cx[x0 - half : x0 + half, y0 - half : y0 + half]
        edges = graph.edges_gdf.cx[x0 - half : x0 + half, y0 - half : y0 + half]
        on_structure = edges["bridge"].notna() | edges["tunnel"].notna()
        edges[~on_structure].plot(ax=ax, color=INK_2, linewidth=1.0, zorder=2)
        if on_structure.any():
            edges[on_structure].plot(
                ax=ax, color=ORANGE, linewidth=2.0, zorder=3, label="bridge or tunnel (building passages included)"
            )
        nodes.plot(ax=ax, color=BLUE, markersize=9, zorder=4, label="node")
        ax.set_xlim(x0 - half, x0 + half)
        ax.set_ylim(y0 - half, y0 + half)
        ax.set_axis_off()
        ax.set_aspect("equal")
        whole = f"{thousands(len(graph.nodes_gdf))} nodes, {thousands(len(graph.edges_gdf))} edges in the district"
        ax.set_title(f"{title}\n{whole}")
    from matplotlib.lines import Line2D  # pylint: disable=import-outside-toplevel

    handles = [
        Line2D([], [], color=INK_2, linewidth=1.0, label="edge"),
        Line2D([], [], color=ORANGE, linewidth=2.0, label="bridge or tunnel (building passages included)"),
        Line2D([], [], color=BLUE, marker="o", linestyle="", markersize=5, label="node"),
    ]
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, "building_simplify")


def building_public_transport():
    """Public-transport routes by mode, and the effective speed of the default registry."""
    graph = intermodal_graph()
    edges = graph.edges_gdf
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.25, 1]})
    walk = edges[edges["type"] == "walk"]
    frame(axes[0], graph, edges=walk)
    for mode, colour in MODE_COLOURS.items():
        part = edges[edges["type"] == mode]
        if len(part):
            routes = part["route"].nunique()
            part.plot(
                ax=axes[0],
                color=colour,
                linewidth=2.2 if mode == "subway" else 1.4,
                zorder=4,
                label=f"{mode} ({routes} route{'s' if routes > 1 else ''})",
            )
    platforms = graph.nodes_gdf[graph.nodes_gdf["type"] == "platform"]
    platforms.plot(ax=axes[0], color=INK, markersize=5, zorder=5, label="platform")
    axes[0].legend(loc="lower left")
    axes[0].set_title("get_public_transport_graph: routes by mode")

    ax = axes[1]
    lengths = np.linspace(50, 3000, 300)
    for mode, colour in MODE_COLOURS.items():
        spec = iduedu.DEFAULT_REGISTRY.get(mode)
        minutes = np.array([spec.travel_time_min(length) for length in lengths])
        ax.plot(lengths, lengths / minutes * 0.06, color=colour, linewidth=2, label=mode)
    ax.set_xlabel("distance between stops, m")
    ax.set_ylabel("km/h, distance over time_min (dwell included)")
    ax.legend(loc="upper left")
    ax.set_title("Speed between stops in DEFAULT_REGISTRY")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, "building_public_transport")


def building_intermodal():
    """The join around a subway station, and the minutes public transport saves against walking."""
    graph = intermodal_graph()
    nodes, edges = graph.nodes_gdf, graph.edges_gdf
    station = nodes[(nodes["type"] == "subway_station") & (nodes["name"] == "Василеостровская")]
    names = {"Горный институт": "Gorny Institut", "Василеостровская": "Vasileostrovskaya", "Приморская": "Primorskaya"}
    x0, y0 = station.geometry.iloc[0].x, station.geometry.iloc[0].y
    half = 260
    window = edges.cx[x0 - half : x0 + half, y0 - half : y0 + half]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    window[window["type"] == "walk"].plot(ax=ax, color=STREETS, linewidth=1.0, zorder=1)
    styles = [
        ("boarding", INK, "-", 1.6, "boarding / alighting"),
        ("subway_entrance", AQUA, "--", 1.6, "subway_entrance / subway_exit"),
    ]
    for kind, colour, style, width, label in styles:
        part = window[window["type"] == kind]
        if len(part):
            part.plot(ax=ax, color=colour, linestyle=style, linewidth=width, zorder=3, label=label)
    for mode, colour in MODE_COLOURS.items():
        part = window[window["type"] == mode]
        if len(part):
            part.plot(ax=ax, color=colour, linewidth=2.2, zorder=2, label=mode)
    near = nodes.cx[x0 - half : x0 + half, y0 - half : y0 + half]
    near[near["type"] == "platform"].plot(ax=ax, color=ORANGE, edgecolor=INK, markersize=40, zorder=5, label="platform")
    entry = near[near["type"] == "subway_entry_exit"]
    if len(entry):
        entry.plot(ax=ax, color=AQUA, edgecolor=INK, marker="^", markersize=70, zorder=5, label="subway entrance")
    station.plot(ax=ax, color=YELLOW, edgecolor=INK, marker="s", markersize=80, zorder=5, label="subway_station")
    ax.set_xlim(x0 - half, x0 + half)
    ax.set_ylim(y0 - half, y0 + half)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=9)
    name = str(station["name"].iloc[0])
    ax.set_title(f"Around {names.get(name, name)} station")

    # Minutes saved: walking only against walking with public transport, from one place.
    walk_edges = edges[edges["type"] == "walk"]
    walk_nodes = np.unique(np.concatenate([walk_edges["u"].to_numpy(), walk_edges["v"].to_numpy()]))
    walk_only = iduedu.subgraph_by_nodes(graph, walk_nodes)
    walk_only = iduedu.UrbanGraph(
        nodes_gdf=walk_only.nodes_gdf,
        edges_gdf=walk_only.edges_gdf[walk_only.edges_gdf["type"] == "walk"],
        is_multigraph=True,
        is_directed=True,
        edge_direction_column="oneway",
        crs=graph.crs,
    )
    xs, ys = node_xy(graph)
    ids = np.asarray(graph.adjacency_nodelist)
    position = {node: i for i, node in enumerate(ids)}
    x_med, y_med = np.median(xs[np.isin(ids, walk_nodes)]), np.median(ys[np.isin(ids, walk_nodes)])
    candidates = [node for node in walk_nodes]
    cx, cy = xs[[position[n] for n in candidates]], ys[[position[n] for n in candidates]]
    source = candidates[nearest(cx, cy, x_med - 1500, y_med + 700)]
    both = iduedu.single_source_dijkstra(graph, source, weight="time_min", output="distances").sparse.to_dense()
    walking = iduedu.single_source_dijkstra(walk_only, source, weight="time_min", output="distances").sparse.to_dense()
    saved = (walking - both.reindex(walking.index)).clip(lower=0)
    saved = saved[np.isfinite(saved)]
    ax = axes[1]
    frame(ax, graph, edges=walk_edges)
    order = saved.sort_values().index
    p = [position[node] for node in order]
    sc = ax.scatter(xs[p], ys[p], c=saved.loc[order].to_numpy(), cmap=BLUES, s=3, vmin=0, linewidths=0, zorder=3)
    ax.scatter(xs[position[source]], ys[position[source]], s=80, color=ORANGE, edgecolor=INK, zorder=6)
    colour_bar(fig, ax, sc, "minutes saved against walking only")
    ax.set_title("What public transport adds, from one place")
    fig.tight_layout()
    save(fig, "building_intermodal")


FIGURES = {
    "routing_search": routing_search,
    "routing_alternatives": routing_alternatives,
    "routing_single_source": routing_single_source,
    "analysis_statistics": analysis_statistics,
    "analysis_objects": analysis_objects,
    "analysis_centrality": analysis_centrality,
    "analysis_elevation": analysis_elevation,
    "building_streets": building_streets,
    "building_simplify": building_simplify,
    "building_public_transport": building_public_transport,
    "building_intermodal": building_intermodal,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
