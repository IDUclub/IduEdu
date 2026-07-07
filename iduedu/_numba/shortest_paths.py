from heapq import heappop, heappush

import numba as nb
import numpy as np
from numba.typed import List

from iduedu._numba import F32CSRMatrix, f32csr_type

# Edge weights and accumulated distances are float32 in their original units
# (minutes or meters). Heap items and result rows carry the distance as float32,
# so they use distinct tuple layouts:
#   heap item   = (distance, node)          -> (float32, int32)
#   result pair = (node, distance)          -> (int32, float32)
heap_pair_type = nb.types.Tuple((nb.float32, nb.int32))
result_pair_type = nb.types.Tuple((nb.int32, nb.float32))
heap_triplet_type = nb.types.Tuple((nb.float32, nb.int32, nb.int32))  # (distance, owner, node)
result_triplet_type = nb.types.Tuple((nb.int32, nb.int32, nb.float32))  # (node, source, distance)

result_row_type = nb.types.ListType(result_pair_type)
result_triplet_row_type = nb.types.ListType(result_triplet_type)
result_rows_type = nb.types.ListType(result_row_type)

_UNREACHED = np.float32(-1.0)


def _njit(signature=None, **kwargs):
    if signature is None:
        return nb.njit(**kwargs)
    return nb.njit(signature, **kwargs)


_dijkstra_from_fringe_signature = None
_dijkstra_distances_signature = None
_single_source_signature = None
_multi_source_signature = None
_nearest_source_signature = None
_od_parallel_signature = None
_path_parallel_signature = None
if f32csr_type is not None:
    _dijkstra_from_fringe_signature = nb.types.Array(nb.float32, 1, "C")(
        f32csr_type,
        nb.types.Array(nb.float32, 1, "C"),
        nb.types.ListType(heap_pair_type),
        nb.float32,
    )
    _dijkstra_distances_signature = nb.types.Array(nb.float32, 1, "C")(
        f32csr_type,
        nb.types.Array(nb.int32, 1, "C"),
        nb.float32,
    )
    _single_source_signature = result_row_type(
        f32csr_type,
        nb.int32,
        nb.float32,
    )
    _multi_source_signature = result_row_type(
        f32csr_type,
        nb.types.Array(nb.int32, 1, "C"),
        nb.float32,
    )
    _nearest_source_signature = result_triplet_row_type(
        f32csr_type,
        nb.types.Array(nb.int32, 1, "C"),
        nb.float32,
    )
    _od_parallel_signature = result_rows_type(
        f32csr_type,
        nb.types.Array(nb.int32, 1, "C"),
        nb.types.Array(nb.int32, 1, "C"),
        nb.float32,
    )
    _path_parallel_signature = result_rows_type(
        f32csr_type,
        nb.types.Array(nb.int32, 1, "C"),
        nb.float32,
    )


@_njit(_dijkstra_from_fringe_signature, cache=True)
def _dijkstra_numba_distances_from_fringe(
    numba_adj_matrix: F32CSRMatrix, seen: np.ndarray, fringe: list, cutoff: np.float32
):
    distances = np.full(numba_adj_matrix.tot_rows, _UNREACHED, dtype=np.float32)

    while len(fringe) > 0:
        d, v = heappop(fringe)
        if distances[v] != _UNREACHED:
            continue
        distances[v] = d
        cols = numba_adj_matrix.get_cols(v)
        vals = numba_adj_matrix.get_vals(v)
        for edge_i in range(len(cols)):
            u = cols[edge_i]
            weight = vals[edge_i]
            uv_dist = np.float32(d + weight)
            if uv_dist > cutoff:
                continue
            if seen[u] == _UNREACHED or uv_dist < seen[u]:
                seen[u] = uv_dist
                heappush(fringe, (uv_dist, np.int32(u)))

    return distances


@_njit(_dijkstra_distances_signature, cache=True)
def _dijkstra_numba_distances(numba_adj_matrix: F32CSRMatrix, origins: np.ndarray, cutoff: np.float32):
    seen = np.full(numba_adj_matrix.tot_rows, _UNREACHED, dtype=np.float32)
    fringe = List.empty_list(heap_pair_type)

    for origin in origins:
        if seen[origin] == _UNREACHED:
            seen[origin] = np.float32(0.0)
            heappush(fringe, (np.float32(0.0), np.int32(origin)))

    return _dijkstra_numba_distances_from_fringe(numba_adj_matrix, seen, fringe, cutoff)


@_njit(_single_source_signature, cache=True)
def single_source_dijkstra_numba_path_length(numba_adj_matrix: F32CSRMatrix, origin: np.int32, cutoff: np.float32):
    """Run single-source Dijkstra and return reachable node distances."""
    seen = np.full(numba_adj_matrix.tot_rows, _UNREACHED, dtype=np.float32)
    fringe = List.empty_list(heap_pair_type)
    seen[origin] = np.float32(0.0)
    fringe.append((np.float32(0.0), np.int32(origin)))
    distances = _dijkstra_numba_distances_from_fringe(numba_adj_matrix, seen, fringe, cutoff)

    row = List.empty_list(result_pair_type)
    for node in range(len(distances)):
        distance = distances[node]
        if distance >= 0.0:
            row.append((np.int32(node), np.float32(distance)))
    return row


@_njit(_multi_source_signature, cache=True)
def multi_source_dijkstra_numba_path_length(numba_adj_matrix: F32CSRMatrix, sources: np.ndarray, cutoff: np.float32):
    """Run multi-source Dijkstra and return nearest-source distances."""
    distances = _dijkstra_numba_distances(numba_adj_matrix, sources, cutoff)

    row = List.empty_list(result_pair_type)
    for node in range(len(distances)):
        distance = distances[node]
        if distance >= 0.0:
            row.append((np.int32(node), np.float32(distance)))
    return row


@_njit(_nearest_source_signature, cache=True)
def multi_source_dijkstra_numba_nearest_source(numba_adj_matrix: F32CSRMatrix, origins: np.ndarray, cutoff: np.float32):
    """Run multi-source Dijkstra and return nearest source ids with distances."""
    distances = np.full(numba_adj_matrix.tot_rows, _UNREACHED, dtype=np.float32)
    seen = np.full(numba_adj_matrix.tot_rows, _UNREACHED, dtype=np.float32)
    seen_source = np.full(numba_adj_matrix.tot_rows, -1, dtype=np.int32)
    fringe = List.empty_list(heap_triplet_type)

    for origin in origins:
        if seen[origin] == _UNREACHED or origin < seen_source[origin]:
            seen[origin] = np.float32(0.0)
            seen_source[origin] = origin
            heappush(fringe, (np.float32(0.0), np.int32(origin), np.int32(origin)))

    while len(fringe) > 0:
        d, owner, v = heappop(fringe)
        if distances[v] != _UNREACHED:
            continue
        distances[v] = d
        cols = numba_adj_matrix.get_cols(v)
        vals = numba_adj_matrix.get_vals(v)
        for edge_i in range(len(cols)):
            u = cols[edge_i]
            weight = vals[edge_i]
            uv_dist = np.float32(d + weight)
            if uv_dist > cutoff:
                continue
            if seen[u] == _UNREACHED or uv_dist < seen[u] or (uv_dist == seen[u] and owner < seen_source[u]):
                seen[u] = uv_dist
                seen_source[u] = owner
                heappush(fringe, (uv_dist, owner, np.int32(u)))

    row = List.empty_list(result_triplet_type)
    for node in range(len(distances)):
        if distances[node] >= 0.0:
            row.append((np.int32(node), seen_source[node], np.float32(distances[node])))
    return row


@_njit(_od_parallel_signature, cache=True, parallel=True)
def dijkstra_numba_od_parallel(
    numba_adj_matrix: F32CSRMatrix, origins: np.ndarray, destinations: np.ndarray, cutoff: np.float32
):
    """Compute sparse OD rows for many origins in parallel."""
    result = List.empty_list(result_row_type)
    for _ in range(len(origins)):
        result.append(List.empty_list(result_pair_type))
    for i in nb.prange(len(origins)):
        seen = np.full(numba_adj_matrix.tot_rows, _UNREACHED, dtype=np.float32)
        fringe = List.empty_list(heap_pair_type)
        source = origins[i]
        seen[source] = np.float32(0.0)
        fringe.append((np.float32(0.0), np.int32(source)))
        distances = _dijkstra_numba_distances_from_fringe(numba_adj_matrix, seen, fringe, cutoff)
        row = List.empty_list(result_pair_type)
        for j in range(len(destinations)):
            distance = distances[destinations[j]]
            if distance >= 0.0:
                row.append((np.int32(j), np.float32(distance)))
        result[i] = row
    return result


@_njit(_path_parallel_signature, cache=True, parallel=True)
def dijkstra_numba_path_length_parallel(numba_adj_matrix: F32CSRMatrix, origins: np.ndarray, cutoff: np.float32):
    """Compute sparse path-length rows for many origins in parallel."""
    result = List.empty_list(result_row_type)
    for _ in range(len(origins)):
        result.append(List.empty_list(result_pair_type))
    for i in nb.prange(len(origins)):
        seen = np.full(numba_adj_matrix.tot_rows, _UNREACHED, dtype=np.float32)
        fringe = List.empty_list(heap_pair_type)
        source = origins[i]
        seen[source] = np.float32(0.0)
        fringe.append((np.float32(0.0), np.int32(source)))
        distances = _dijkstra_numba_distances_from_fringe(numba_adj_matrix, seen, fringe, cutoff)
        row = List.empty_list(result_pair_type)
        for node in range(len(distances)):
            distance = distances[node]
            if distance >= 0.0:
                row.append((np.int32(node), np.float32(distance)))
        result[i] = row
    return result
