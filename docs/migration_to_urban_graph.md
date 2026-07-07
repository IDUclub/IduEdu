# Migrating to UrbanGraph

IduEdu now uses `UrbanGraph` as the primary graph representation. The old
NetworkX-first workflow is still available through optional compatibility
helpers, but new builders, matrix functions and graph utilities operate on
`UrbanGraph`.

## Main changes

- Graph builders such as `get_walk_graph`, `get_drive_graph`,
  `get_public_transport_graph` and `get_intermodal_graph` return `UrbanGraph`.
- Node and edge data live in `graph.nodes_gdf` and `graph.edges_gdf`.
- OD matrices and shortest-path helpers accept `UrbanGraph` directly.
- NetworkX conversion is optional: use `urban_graph2nx_graph()` and
  `nx_graph2urban_graph()` when integration with NetworkX code is required.
- GML helpers and legacy NetworkX utilities require optional NetworkX support.

## Node inputs

Functions that accept origins, destinations or source nodes can work with either
explicit node ids or GeoDataFrames:

```python
from iduedu import nearest_nodes, od_matrix

sources["graph_node_id"] = nearest_nodes(graph, sources)

matrix = od_matrix(
    graph,
    gdf_sources=sources,
    gdf_targets=targets,
    weight="time_min",
)
```

If a GeoDataFrame already contains `graph_node_id`, IduEdu uses it directly. If
the column is absent, geometries are matched to the nearest graph nodes.

## Validation

Use `graph.validate()` or `validate_graph(graph)` to check node ids, edge
endpoints, required columns, geometries and CRS consistency after custom edits:

```python
from iduedu import validate_graph

graph.edges_gdf["time_min"] = graph.edges_gdf["length_meter"] / 80
validate_graph(graph)
```

## Storage

Use `.urbangraph` archives to persist `UrbanGraph` objects without converting
them to NetworkX or GML. The archive stores `metadata.json`, `nodes.parquet` and
`edges.parquet`; adjacency matrices can be included as an optional cache.

```python
from iduedu import UrbanGraph, read_urban_graph, write_urban_graph

write_urban_graph(graph, "walk.urbangraph")
graph = read_urban_graph("walk.urbangraph")

# Equivalent object-oriented API:
graph.write("walk.urbangraph")
graph = UrbanGraph.read("walk.urbangraph")
```

Install `iduedu[io]` when parquet support is not already available.

## Common replacements

| Old workflow | New workflow |
| --- | --- |
| Read nodes and edges from a NetworkX graph | Use `graph.nodes_gdf` and `graph.edges_gdf` |
| Convert GeoDataFrames to NetworkX before OD calculations | Pass `UrbanGraph` directly to `od_matrix()` |
| Manually snap objects to node ids before every matrix call | Use `nearest_nodes()` or provide `graph_node_id` |
| Keep NetworkX as the canonical in-memory graph | Keep `UrbanGraph`; convert to NetworkX only at boundaries |
