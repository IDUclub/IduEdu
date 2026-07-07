import warnings
from typing import Any, Literal

import numpy as np

from iduedu._numba.components import connected_components_numba, strongly_connected_components_numba
from iduedu._numba.csr import sparse_row2numba_bool_matrix
from iduedu.graph.adjacency import build_boolean_adjacency_matrix
from iduedu.graph.urban_graph import UrbanGraph

ComponentMode = Literal["auto", "connected", "weak", "strong"]


def _validate_graph(graph: UrbanGraph) -> None:
    if not isinstance(graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(graph).__name__}")


def _components_from_labels(nodelist: list[Any], labels: np.ndarray) -> list[set[Any]]:
    def sort_components(components: list[set[Any]]) -> list[set[Any]]:
        """Sort components by descending size and stable textual node order."""
        return sorted(
            components, key=lambda component: (-len(component), min(map(repr, component)) if component else "")
        )

    components_by_label: dict[int, set[Any]] = {}
    for pos, label in enumerate(labels):
        components_by_label.setdefault(int(label), set()).add(nodelist[pos])
    return sort_components(list(components_by_label.values()))


def connected_components(graph: UrbanGraph) -> list[set[Any]]:
    """
    Return connected components of an undirected ``UrbanGraph``.

    For directed graphs use :func:`weakly_connected_components` or
    :func:`strongly_connected_components` explicitly.

    See also:
        https://iduclub.github.io/IduEdu/examples/connectivity.html
    """

    _validate_graph(graph)
    if graph.is_directed:
        raise ValueError("connected_components is not defined for directed UrbanGraph; use weak or strong components")

    nodelist = graph.nodes_gdf.index.to_list()
    adjacency = build_boolean_adjacency_matrix(graph, nodelist=nodelist, weak=True)
    return _components_from_labels(nodelist, connected_components_numba(sparse_row2numba_bool_matrix(adjacency)))


def weakly_connected_components(graph: UrbanGraph) -> list[set[Any]]:
    """
    Return weakly connected components.

    Edge directions are ignored. For undirected graphs this is equivalent to
    :func:`connected_components`.

    See also:
        https://iduclub.github.io/IduEdu/examples/connectivity.html
    """

    _validate_graph(graph)
    nodelist = graph.nodes_gdf.index.to_list()
    adjacency = build_boolean_adjacency_matrix(graph, nodelist=nodelist, weak=True)
    return _components_from_labels(nodelist, connected_components_numba(sparse_row2numba_bool_matrix(adjacency)))


def strongly_connected_components(graph: UrbanGraph) -> list[set[Any]]:
    """
    Return strongly connected components.

    For undirected graphs this is equivalent to connected components and emits
    a warning.

    See also:
        https://iduclub.github.io/IduEdu/examples/connectivity.html
    """

    _validate_graph(graph)
    if not graph.is_directed:
        warnings.warn(
            "strongly_connected_components called for an undirected UrbanGraph; "
            "returning connected components instead.",
            UserWarning,
            stacklevel=2,
        )
        return connected_components(graph)

    nodelist = graph.nodes_gdf.index.to_list()
    adjacency = build_boolean_adjacency_matrix(graph, nodelist=nodelist, weak=False)
    labels = strongly_connected_components_numba(
        sparse_row2numba_bool_matrix(adjacency),
        sparse_row2numba_bool_matrix(adjacency.T.tocsr()),
    )
    return _components_from_labels(nodelist, labels)


def number_connected_components(graph: UrbanGraph) -> int:
    """Return the number of connected components in an undirected graph.

    See also:
        https://iduclub.github.io/IduEdu/examples/connectivity.html
    """
    return len(connected_components(graph))


def number_weakly_connected_components(graph: UrbanGraph) -> int:
    """Return the number of weakly connected components.

    See also:
        https://iduclub.github.io/IduEdu/examples/connectivity.html
    """
    return len(weakly_connected_components(graph))


def number_strongly_connected_components(graph: UrbanGraph) -> int:
    """Return the number of strongly connected components.

    See also:
        https://iduclub.github.io/IduEdu/examples/connectivity.html
    """
    return len(strongly_connected_components(graph))


def largest_connected_component(graph: UrbanGraph) -> set[Any]:
    """Return the largest connected component of an undirected graph.

    See also:
        https://iduclub.github.io/IduEdu/examples/connectivity.html
    """
    components = connected_components(graph)
    return components[0] if components else set()


def largest_weakly_connected_component(graph: UrbanGraph) -> set[Any]:
    """Return the largest weakly connected component.

    See also:
        https://iduclub.github.io/IduEdu/examples/connectivity.html
    """
    components = weakly_connected_components(graph)
    return components[0] if components else set()


def largest_strongly_connected_component(graph: UrbanGraph) -> set[Any]:
    """Return the largest strongly connected component.

    See also:
        https://iduclub.github.io/IduEdu/examples/connectivity.html
    """
    components = strongly_connected_components(graph)
    return components[0] if components else set()


def largest_component(graph: UrbanGraph, mode: ComponentMode = "auto") -> set[Any]:
    """Return the largest component according to the selected connectivity mode.

    Args:
        graph: Graph to inspect.
        mode: Connectivity mode. ``"auto"`` selects ``"strong"`` for directed
            graphs and ``"connected"`` for undirected graphs.

    Returns:
        Set of node ids in the largest component. Returns an empty set for an
        empty graph.

    Raises:
        ValueError: If ``mode`` is not supported.

    See also:
        https://iduclub.github.io/IduEdu/examples/connectivity.html
    """
    _validate_graph(graph)
    if mode == "auto":
        mode = "strong" if graph.is_directed else "connected"
    if mode == "connected":
        return largest_connected_component(graph)
    if mode == "weak":
        return largest_weakly_connected_component(graph)
    if mode == "strong":
        return largest_strongly_connected_component(graph)
    raise ValueError(f"mode must be 'auto', 'connected', 'weak' or 'strong', got {mode!r}")
