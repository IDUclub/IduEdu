from heapq import heappop, heappush

import numba as nb
import numpy as np
from numba.typed import List

# Edge weights and accumulated distances are float32 in their original units
# (minutes or meters). Heap items carry (distance, owner, node); a compact search
# returns settled node positions, their costs, parents and source owners.
heap_triplet_type = nb.types.Tuple((nb.float32, nb.int32, nb.int32))
compact_search_type = nb.types.Tuple(
    (nb.types.int32[::1], nb.types.float32[::1], nb.types.int32[::1], nb.types.int32[::1])
)

_UNREACHED = np.float32(-1.0)


@nb.njit(cache=True)
def dijkstra_numba_search(numba_adj_matrix, sources, targets, cutoff, store_predecessors):
    """Return compact settled nodes, costs, optional parents and source owners.

    An empty target array means all reachable nodes. Otherwise stop as soon
    as every distinct target settles. Only settled distances are returned:
    tentative distances left in the queue at early termination are not results.
    """
    size = numba_adj_matrix.tot_rows
    seen = np.full(size, _UNREACHED, dtype=np.float32)
    owners = np.full(size, -1, dtype=np.int32)
    settled = np.zeros(size, dtype=np.bool_)
    parents = np.full(size if store_predecessors else 0, -1, dtype=np.int32)
    target_mask = np.zeros(size if len(targets) else 0, dtype=np.bool_)
    targets_left = 0
    for target in targets:
        if not target_mask[target]:
            target_mask[target] = True
            targets_left += 1
    fringe = List.empty_list(heap_triplet_type)
    for source in sources:
        if seen[source] == _UNREACHED:
            seen[source] = np.float32(0.0)
            owners[source] = source
            heappush(fringe, (np.float32(0.0), np.int32(source), np.int32(source)))
    while len(fringe):
        distance, owner, node = heappop(fringe)
        if settled[node]:
            continue
        settled[node] = True
        if len(targets) and target_mask[node]:
            targets_left -= 1
            if targets_left == 0:
                break
        columns = numba_adj_matrix.get_cols(node)
        weights = numba_adj_matrix.get_vals(node)
        for edge_i in range(len(columns)):
            target = columns[edge_i]
            if settled[target]:
                continue
            candidate = np.float32(distance + weights[edge_i])
            if not np.isfinite(candidate) or candidate > cutoff:
                continue
            if (
                seen[target] == _UNREACHED
                or candidate < seen[target]
                or (candidate == seen[target] and owner < owners[target])
            ):
                seen[target] = candidate
                owners[target] = owner
                if store_predecessors:
                    parents[target] = node
                heappush(fringe, (candidate, owner, np.int32(target)))
    positions = np.where(settled if store_predecessors or not len(targets) else settled & target_mask)[0].astype(
        np.int32
    )
    compact_parents = parents[positions] if store_predecessors else np.empty(0, dtype=np.int32)
    return positions, seen[positions], compact_parents, owners[positions]


@nb.njit(cache=True, parallel=True)
def dijkstra_numba_search_parallel(
    numba_adj_matrix, sources, targets, target_starts, target_ends, cutoff, store_predecessors
):
    """Run one search per unique source; preserve compact trees for lazy routes.

    Source i stops once targets[target_starts[i]:target_ends[i]] settle; an
    empty slice means all reachable nodes.
    """
    result = List.empty_list(compact_search_type)
    for _ in range(len(sources)):
        result.append(
            (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )
        )
    for i in nb.prange(len(sources)):
        row = np.int64(i)
        result[row] = dijkstra_numba_search(
            numba_adj_matrix,
            sources[row : row + 1],
            targets[target_starts[row] : target_ends[row]],
            cutoff,
            store_predecessors,
        )
    return result


@nb.njit(cache=True)
def reconstruct_numba_compact_path(nodes, parents, target, reverse):
    """Reconstruct one compact-tree route without accessing a live graph."""
    route = List.empty_list(nb.int32)
    node = target
    while node >= 0:
        offset = np.searchsorted(nodes, node)
        if offset == len(nodes) or nodes[offset] != node:
            return np.empty(0, dtype=np.int32)
        route.append(np.int32(node))
        node = parents[offset]
    result = np.empty(len(route), dtype=np.int32)
    for i in range(len(route)):
        result[i] = route[i if reverse else len(route) - i - 1]
    return result
