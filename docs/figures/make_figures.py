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


FIGURES = {
    "routing_search": routing_search,
    "routing_alternatives": routing_alternatives,
    "routing_single_source": routing_single_source,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
