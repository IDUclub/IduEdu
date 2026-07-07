# pylint: disable=protected-access

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import CRS as PyprojCRS
from shapely import LineString
from shapely.geometry import Polygon

from iduedu import DEFAULT_REGISTRY_W_TRAIN
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.graph_builders import drive_walk_builders, public_transport_builders
from iduedu.graph_builders.drive_walk_builders import (
    _assign_edge_keys,
    _build_nodes_and_uv,
    _get_highway_properties,
    get_drive_graph,
    get_walk_graph,
)
from iduedu.graph_builders.intermodal_builders import join_pt_walk_graph
from iduedu.graph_builders.public_transport_builders import get_public_transport_graph
from tests.factories import CRS, tiny_public_transport_graph_with_sequence_attrs, tiny_walk_graph

pytestmark = pytest.mark.unit

# Small territory around a few synthetic streets, far away from any real OSM data since
# get_network_by_filters is monkeypatched in these tests.
TERRITORY = Polygon([(30.30, 59.90), (30.32, 59.90), (30.32, 59.92), (30.30, 59.92)])

WAY_A = {
    "type": "way",
    "id": 10,
    "tags": {"highway": "residential", "oneway": "yes", "maxspeed": "30", "name": "Street A"},
    "geometry": [{"lat": 59.900, "lon": 30.300}, {"lat": 59.901, "lon": 30.301}],
}
WAY_B = {
    "type": "way",
    "id": 11,
    "tags": {"highway": "primary", "oneway": "-1"},
    "geometry": [{"lat": 59.905, "lon": 30.305}, {"lat": 59.906, "lon": 30.306}],
}
WAY_C = {
    "type": "way",
    "id": 12,
    "tags": {"highway": "unclassified"},
    "geometry": [{"lat": 59.910, "lon": 30.310}, {"lat": 59.911, "lon": 30.311}],
}


def _fake_network_by_filters(ways):
    def _fake(polygon, way_filter):  # pylint: disable=unused-argument
        return pd.DataFrame(ways)

    return _fake


# ---------------------------------------------------------------------------
# _get_highway_properties
# ---------------------------------------------------------------------------


def test_get_highway_properties_single_known_type():
    category, speed = _get_highway_properties("motorway")
    assert category == "federal"
    assert speed == pytest.approx(110 * 1000 / 60)


def test_get_highway_properties_list_picks_lowest_reg_and_min_speed():
    category, speed = _get_highway_properties(["motorway", "residential"])
    assert category == "local"
    assert speed == pytest.approx(40 * 1000 / 60)


@pytest.mark.parametrize("highway", [None, [], "not_a_real_highway_type"])
def test_get_highway_properties_defaults_for_missing_or_unknown(highway):
    category, speed = _get_highway_properties(highway)
    assert category == "local"
    assert speed == pytest.approx(40 * 1000 / 60)


# ---------------------------------------------------------------------------
# _assign_edge_keys
# ---------------------------------------------------------------------------


def test_assign_edge_keys_increments_per_uv_group():
    edges = pd.DataFrame({"u": [0, 0, 1, 0], "v": [1, 1, 2, 1]})
    result = _assign_edge_keys(edges)
    assert result["k"].tolist() == [0, 1, 0, 2]


# ---------------------------------------------------------------------------
# _build_nodes_and_uv
# ---------------------------------------------------------------------------


def test_build_nodes_and_uv_dedupes_shared_endpoints():
    edges = gpd.GeoDataFrame(
        {"way_idx": [0, 1]},
        geometry=[LineString([(0.0, 0.0), (10.0, 0.0)]), LineString([(10.0, 0.0), (20.0, 0.0)])],
        crs=CRS,
    )
    nodes, edges_out = _build_nodes_and_uv(edges, PyprojCRS.from_user_input(CRS))

    assert len(nodes) == 3
    assert edges_out.loc[0, "v"] == edges_out.loc[1, "u"]
    assert edges_out.loc[0, "u"] != edges_out.loc[1, "v"]


# ---------------------------------------------------------------------------
# Parameter validation (raises before any network access)
# ---------------------------------------------------------------------------


def test_get_drive_graph_rejects_unknown_network_type():
    with pytest.raises(ValueError, match="Unknown road_type"):
        get_drive_graph(territory=TERRITORY, network_type="bogus")


def test_get_drive_graph_custom_requires_filter():
    with pytest.raises(ValueError, match="custom_filter"):
        get_drive_graph(territory=TERRITORY, network_type="custom")


def test_get_walk_graph_rejects_unknown_network_type():
    with pytest.raises(ValueError, match="Unknown road_type"):
        get_walk_graph(territory=TERRITORY, network_type="bogus")


def test_get_walk_graph_custom_requires_filter():
    with pytest.raises(ValueError, match="custom_filter"):
        get_walk_graph(territory=TERRITORY, network_type="custom")


def test_get_public_transport_graph_rejects_unknown_transport_type():
    with pytest.raises(ValueError, match="Unknown transport type"):
        get_public_transport_graph(transport_types=["bus", "not_a_mode"])


def test_get_public_transport_graph_rejects_negative_boarding_time():
    with pytest.raises(ValueError, match="avg_boarding_time_min"):
        get_public_transport_graph(avg_boarding_time_min=-1.0)


# ---------------------------------------------------------------------------
# get_drive_graph pipeline (Overpass response mocked)
# ---------------------------------------------------------------------------


def test_get_drive_graph_builds_expected_edges(monkeypatch):
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", _fake_network_by_filters([WAY_A, WAY_B, WAY_C]))

    graph = get_drive_graph(territory=TERRITORY, simplify=False, keep_largest_subgraph=False)

    assert len(graph.nodes_gdf) == 6
    assert len(graph.edges_gdf) == 3
    assert graph.is_directed is True
    assert graph.is_multigraph is True
    assert graph.edge_direction_column == "oneway"
    assert graph.edges_gdf["oneway"].dtype == bool

    categories = dict(zip(graph.edges_gdf["highway"], graph.edges_gdf["category"]))
    assert categories.get("residential") == "local"
    assert categories.get("primary") == "regional"
    assert categories.get("unclassified") == "local"

    # way A has an explicit maxspeed (30 km/h) overriding the residential default (40 km/h)
    way_a_edge = graph.edges_gdf[graph.edges_gdf["highway"] == "residential"].iloc[0]
    expected_speed_mpm = 30 * 1000 / 60
    expected_time_min = round(way_a_edge["length_meter"] / expected_speed_mpm, 3)
    assert way_a_edge["time_min"] == pytest.approx(expected_time_min, abs=1e-3)

    assert graph.edges_gdf["length_meter"].gt(0).all()
    assert graph.edges_gdf["time_min"].gt(0).all()


def test_get_drive_graph_reverses_geometry_for_oneway_minus_one(monkeypatch):
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", _fake_network_by_filters([WAY_A, WAY_B]))

    graph = get_drive_graph(territory=TERRITORY, simplify=False, keep_largest_subgraph=False)

    edge_a = graph.edges_gdf[graph.edges_gdf["highway"] == "residential"].iloc[0]
    edge_b = graph.edges_gdf[graph.edges_gdf["highway"] == "primary"].iloc[0]

    coords_a = list(edge_a.geometry.coords)
    coords_b = list(edge_b.geometry.coords)

    # way A (oneway="yes") keeps its original west-to-east coordinate order
    assert coords_a[0][0] < coords_a[-1][0]
    # way B (oneway="-1") is reversed to east-to-west
    assert coords_b[0][0] > coords_b[-1][0]
    assert bool(edge_a["oneway"]) is True
    assert bool(edge_b["oneway"]) is True


def test_get_drive_graph_omits_category_when_disabled(monkeypatch):
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", _fake_network_by_filters([WAY_A]))

    graph = get_drive_graph(territory=TERRITORY, simplify=False, add_road_category=False, keep_largest_subgraph=False)

    assert "category" not in graph.edges_gdf.columns


def test_get_drive_graph_returns_empty_graph_when_no_data(monkeypatch):
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", _fake_network_by_filters([]))

    graph = get_drive_graph(territory=TERRITORY)

    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty
    assert graph.type == "drive"


def test_get_drive_graph_returns_empty_graph_when_clip_removes_everything(monkeypatch):
    # Ways sit far outside TERRITORY, so clipping should drop them entirely.
    far_way = {
        "type": "way",
        "id": 99,
        "tags": {"highway": "residential"},
        "geometry": [{"lat": 10.0, "lon": 10.0}, {"lat": 10.001, "lon": 10.001}],
    }
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", _fake_network_by_filters([far_way]))

    graph = get_drive_graph(territory=TERRITORY, clip_by_territory=True)

    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty


# ---------------------------------------------------------------------------
# get_walk_graph pipeline (Overpass response mocked)
# ---------------------------------------------------------------------------


def test_get_walk_graph_builds_undirected_multigraph(monkeypatch):
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", _fake_network_by_filters([WAY_A, WAY_C]))

    graph = get_walk_graph(territory=TERRITORY, simplify=False, keep_largest_subgraph=False, walk_speed=100.0)

    assert len(graph.edges_gdf) == 2
    assert graph.is_directed is False
    assert graph.edge_direction_column is None
    assert "oneway" not in graph.edges_gdf.columns
    assert "category" not in graph.edges_gdf.columns

    expected_time_min = (graph.edges_gdf["length_meter"] / 100.0).round(3)
    assert graph.edges_gdf["time_min"].to_numpy() == pytest.approx(expected_time_min.to_numpy(), abs=1e-3)


def test_get_walk_graph_returns_empty_graph_when_no_data(monkeypatch):
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", _fake_network_by_filters([]))

    graph = get_walk_graph(territory=TERRITORY)

    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty
    assert graph.type == "walk"


def test_get_drive_graph_empty_when_response_has_no_ways(monkeypatch):
    # only a node element, no ways -> _build_edges_from_overpass returns empty
    node_only = [{"type": "node", "id": 1, "tags": {}}]
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", _fake_network_by_filters(node_only))

    graph = get_drive_graph(territory=TERRITORY)

    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty


def test_get_drive_graph_empty_when_ways_have_single_point(monkeypatch):
    degenerate = {
        "type": "way",
        "id": 20,
        "tags": {"highway": "residential"},
        "geometry": [{"lat": 59.90, "lon": 30.30}],  # < 2 coordinates
    }
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", _fake_network_by_filters([degenerate]))

    graph = get_drive_graph(territory=TERRITORY)

    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty


def test_get_drive_graph_json_normalize_path_for_many_tags(monkeypatch):
    # > 50 requested tags switches _build_edges_from_overpass to the json_normalize branch
    many_tags = [f"tag_{i}" for i in range(60)]
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", _fake_network_by_filters([WAY_A]))

    graph = get_drive_graph(territory=TERRITORY, simplify=False, keep_largest_subgraph=False, osm_edge_tags=many_tags)

    # the json_normalize branch still yields a valid single-edge graph
    assert len(graph.edges_gdf) == 1
    assert graph.edges_gdf.iloc[0]["category"] == "local"
    assert graph.edges_gdf.iloc[0]["time_min"] > 0


def test_get_drive_graph_simplify_path_merges_segments(monkeypatch):
    # Exercises the simplify=True branch (line_merge + nearest-midpoint attribute transfer).
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", _fake_network_by_filters([WAY_A, WAY_C]))

    graph = get_drive_graph(territory=TERRITORY, simplify=True, keep_largest_subgraph=False, osm_edge_tags=["name"])

    assert len(graph.edges_gdf) >= 1
    assert graph.edges_gdf["oneway"].dtype == bool
    assert (graph.edges_gdf["time_min"] > 0).all()


def test_get_walk_graph_returns_empty_graph_when_clip_removes_everything(monkeypatch):
    far_way = {
        "type": "way",
        "id": 30,
        "tags": {"highway": "footway"},
        "geometry": [{"lat": 10.0, "lon": 10.0}, {"lat": 10.001, "lon": 10.001}],
    }
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", _fake_network_by_filters([far_way]))

    graph = get_walk_graph(territory=TERRITORY, clip_by_territory=True)

    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty


# ---------------------------------------------------------------------------
# get_public_transport_graph pipeline (Overpass response mocked)
# ---------------------------------------------------------------------------


def test_get_public_transport_graph_returns_empty_graph_when_no_routes(monkeypatch):
    monkeypatch.setattr(public_transport_builders, "get_routes_by_poly", lambda polygon, types: [])

    graph = get_public_transport_graph(transport_types="bus", territory=TERRITORY)

    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty
    assert graph.is_directed is True
    assert graph.edge_direction_column == "oneway"
    assert graph.type == "public_transport"


def test_get_public_transport_graph_accepts_train_with_custom_registry(monkeypatch):
    monkeypatch.setattr(public_transport_builders, "get_routes_by_poly", lambda polygon, types: [])

    graph = get_public_transport_graph(
        transport_types="train", territory=TERRITORY, transport_registry=DEFAULT_REGISTRY_W_TRAIN
    )

    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty


def test_get_public_transport_graph_drops_unrequested_leaked_route_type(monkeypatch):
    # The subway station-context query can pull in unrelated route relations (e.g.
    # route=train sharing nodes with a metro station). Such leaked types must not
    # reach the parser / travel-time registry when they were not requested.
    parsed_types = []

    def fake_routes_to_df(response, enable_subway_details):
        return pd.DataFrame(
            {
                "transport_type": ["bus", "train"],
                "is_way_data": [False, False],
                "members": [[], []],
            }
        )

    def fake_ground_parser(row, local_crs, ref2speed, needed_tags):
        parsed_types.append(row.transport_type)
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()

    monkeypatch.setattr(public_transport_builders, "get_routes_by_poly", lambda polygon, types: [object()])
    monkeypatch.setattr(public_transport_builders, "overpass_routes_to_df", fake_routes_to_df)
    monkeypatch.setattr(public_transport_builders, "overpass_ground_transport2edgenode", fake_ground_parser)

    # 'train' is present in the Overpass response but was not requested and is not
    # in the default registry: it must be filtered out (no KeyError, no train edges).
    graph = get_public_transport_graph(transport_types="bus", territory=TERRITORY)

    assert "bus" in parsed_types
    assert "train" not in parsed_types
    assert graph.edges_gdf.empty


# ---------------------------------------------------------------------------
# join_pt_walk_graph edge cases (no network needed)
# ---------------------------------------------------------------------------


def test_join_pt_walk_graph_raises_on_crs_mismatch():
    walk_graph = tiny_walk_graph()
    pt_graph = tiny_public_transport_graph_with_sequence_attrs()
    mismatched_walk = walk_graph.copy()
    mismatched_walk.crs = "EPSG:4326"
    mismatched_walk.nodes_gdf = mismatched_walk.nodes_gdf.set_crs("EPSG:4326", allow_override=True)
    mismatched_walk.edges_gdf = mismatched_walk.edges_gdf.set_crs("EPSG:4326", allow_override=True)

    with pytest.raises(ValueError, match="CRS mismatch"):
        join_pt_walk_graph(pt_graph, mismatched_walk)


def test_join_pt_walk_graph_returns_walk_graph_when_pt_graph_empty():
    walk_graph = tiny_walk_graph()
    empty_pt = UrbanGraph.empty(
        crs=walk_graph.crs, is_multigraph=True, is_directed=True, edge_direction_column="oneway"
    )

    result = join_pt_walk_graph(empty_pt, walk_graph)

    assert result is walk_graph


def test_join_pt_walk_graph_returns_pt_graph_when_walk_graph_empty():
    pt_graph = tiny_public_transport_graph_with_sequence_attrs()
    empty_walk = UrbanGraph.empty(crs=pt_graph.crs, is_multigraph=True, is_directed=False)

    result = join_pt_walk_graph(pt_graph, empty_walk)

    assert result is pt_graph
