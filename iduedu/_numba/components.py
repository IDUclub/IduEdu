import numba as nb
import numpy as np

from iduedu._numba.csr import BCSRMatrix, bcsr_type


def _njit(signature=None, **kwargs):
    if signature is None:
        return nb.njit(**kwargs)
    return nb.njit(signature, **kwargs)


_connected_components_signature = None
_strongly_connected_components_signature = None
if bcsr_type is not None:
    _connected_components_signature = nb.types.Array(nb.int32, 1, "C")(bcsr_type)
    _strongly_connected_components_signature = nb.types.Array(nb.int32, 1, "C")(
        bcsr_type,
        bcsr_type,
    )


@_njit(_connected_components_signature, cache=True)
def connected_components_numba(adj_matrix: BCSRMatrix):
    """Label connected components in a boolean CSR adjacency matrix."""
    labels = np.full(adj_matrix.tot_rows, -1, dtype=np.int32)
    stack = np.empty(adj_matrix.tot_rows, dtype=np.int32)
    component_id = np.int32(0)

    for start in range(adj_matrix.tot_rows):
        if labels[start] != -1:
            continue

        stack_size = 1
        stack[0] = np.int32(start)
        labels[start] = component_id

        while stack_size > 0:
            stack_size -= 1
            node = stack[stack_size]
            cols = adj_matrix.get_cols(node)

            for i in range(len(cols)):
                neighbor = np.int32(cols[i])
                if labels[neighbor] == -1:
                    labels[neighbor] = component_id
                    stack[stack_size] = neighbor
                    stack_size += 1

        component_id += np.int32(1)

    return labels


@_njit(_strongly_connected_components_signature, cache=True)
def strongly_connected_components_numba(adj_matrix: BCSRMatrix, reverse_adj_matrix: BCSRMatrix):
    """Label strongly connected components using iterative Kosaraju traversal."""
    visited = np.zeros(adj_matrix.tot_rows, dtype=np.bool_)
    order = np.empty(adj_matrix.tot_rows, dtype=np.int32)
    order_size = 0
    stack_nodes = np.empty(adj_matrix.tot_rows, dtype=np.int32)
    stack_next_neighbor = np.empty(adj_matrix.tot_rows, dtype=np.int32)

    for start in range(adj_matrix.tot_rows):
        if visited[start]:
            continue

        visited[start] = True
        stack_size = 1
        stack_nodes[0] = np.int32(start)
        stack_next_neighbor[0] = np.int32(0)

        while stack_size > 0:
            top = stack_size - 1
            node = stack_nodes[top]
            next_neighbor = stack_next_neighbor[top]
            cols = adj_matrix.get_cols(node)

            if next_neighbor < len(cols):
                neighbor = np.int32(cols[next_neighbor])
                stack_next_neighbor[top] = next_neighbor + np.int32(1)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack_nodes[stack_size] = neighbor
                    stack_next_neighbor[stack_size] = np.int32(0)
                    stack_size += 1
            else:
                order[order_size] = node
                order_size += 1
                stack_size -= 1

    labels = np.full(adj_matrix.tot_rows, -1, dtype=np.int32)
    stack = np.empty(adj_matrix.tot_rows, dtype=np.int32)
    component_id = np.int32(0)

    for order_pos in range(order_size - 1, -1, -1):
        start = order[order_pos]
        if labels[start] != -1:
            continue

        stack_size = 1
        stack[0] = start
        labels[start] = component_id

        while stack_size > 0:
            stack_size -= 1
            node = stack[stack_size]
            cols = reverse_adj_matrix.get_cols(node)

            for i in range(len(cols)):
                neighbor = np.int32(cols[i])
                if labels[neighbor] == -1:
                    labels[neighbor] = component_id
                    stack[stack_size] = neighbor
                    stack_size += 1

        component_id += np.int32(1)

    return labels
