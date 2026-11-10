"""Sums of node values within a travel cost of each source.

The shared core of local network statistics, object aggregations and reach-type centralities: one cut-off
Dijkstra search per source in the Rust kernel, which adds up the values of the nodes it settles.
"""

from typing import Iterable

import numpy as np
from scipy import sparse

from iduedu.graph.routing import _prepare_graph
from iduedu.graph.urban_graph import UrbanGraph

DECAYS = ("flat", "linear", "exp", "inverse")


def cost_bands(cutoffs: float | Iterable[float] | None, name: str = "cutoffs") -> np.ndarray:
    """Ascending unique float32 cutoffs; ``None`` is one unlimited band."""
    if cutoffs is None:
        return np.array([np.inf], dtype=np.float32)
    values = np.atleast_1d(np.asarray(cutoffs, dtype=float))
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a number or a non-empty list of numbers")
    if np.isnan(values).any() or (values < 0).any():
        raise ValueError(f"{name} must be non-negative and must not be NaN")
    return np.unique(values).astype(np.float32)


def band_label(value: float) -> str:
    """Column suffix of a cutoff: ``400``, ``7.5``, ``inf``."""
    if np.isinf(value):
        return "inf"
    return f"{float(value):g}"


def value_rows(n_rows: int, rows, cols, data, n_cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse rows ``(indptr, cols, data)`` from ``(row, col, value)`` triples; duplicates are summed."""
    matrix = sparse.coo_matrix(
        (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))), shape=(n_rows, n_cols)
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return (
        np.ascontiguousarray(matrix.indptr, dtype=np.int64),
        np.ascontiguousarray(matrix.indices, dtype=np.int32),
        np.ascontiguousarray(matrix.data, dtype=np.float32),
    )


def band_betas(bands: np.ndarray, decay: str, beta: float | Iterable[float] | None) -> np.ndarray:
    """One ``beta`` per band for exponential decay: given, or ``4 / cutoff`` (weight 0.018 at the cutoff)."""
    if decay not in DECAYS:
        raise ValueError(f"decay must be one of {DECAYS}, got {decay!r}")
    if decay != "exp":
        return np.zeros(len(bands), dtype=np.float32)
    if beta is None:
        if np.isinf(bands).any():
            raise ValueError("exponential decay without a finite cutoff needs beta")
        if (bands == 0).any():
            raise ValueError("exponential decay needs positive cutoffs or beta")
        return (4.0 / bands).astype(np.float32)
    betas = np.broadcast_to(np.asarray(beta, dtype=float), bands.shape)
    if np.isnan(betas).any() or (betas < 0).any():
        raise ValueError("beta must be non-negative")
    return np.ascontiguousarray(betas, dtype=np.float32)


def sum_within(
    urban_graph: UrbanGraph,
    sources: np.ndarray,
    values: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_cols: int,
    bands: np.ndarray,
    *,
    weight: str,
    decay: str = "flat",
    betas: np.ndarray | None = None,
    reverse: bool = False,
    skip_source: bool = False,
    max_workers: int | None = None,
) -> np.ndarray:
    """``out[i, band, col]``: decayed sum of the node values of column ``col`` within ``bands[band]`` of source ``i``.

    ``sources`` are node positions in the adjacency matrix, ``values`` sparse rows per node from
    :func:`value_rows`. With ``reverse`` costs are measured towards the source (directed graphs).
    """
    csr = _prepare_graph(urban_graph, weight=weight, cutoff=None, reverse=reverse)
    if betas is None:
        betas = band_betas(bands, decay, None)
    return csr.aggregate(
        np.ascontiguousarray(sources, dtype=np.int32),
        *values,
        n_cols,
        np.ascontiguousarray(bands, dtype=np.float32),
        decay,
        betas,
        skip_source,
        max_workers,
    )


def spread_within(
    urban_graph: UrbanGraph,
    sources: np.ndarray,
    values: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_cols: int,
    slots: np.ndarray,
    n_slots: int,
    bands: np.ndarray,
    *,
    weight: str,
    decay: str = "flat",
    betas: np.ndarray | None = None,
    reverse: bool = False,
    max_workers: int | None = None,
) -> np.ndarray:
    """The sums of :func:`sum_within` computed from the side of the values.

    Source ``i`` carries row ``i`` of ``values`` and adds it to every node around it that has a slot
    (``slots[node] >= 0``); the result is ``(n_slots, bands, n_cols)``. Cheaper than :func:`sum_within` when
    the values sit on fewer nodes than there are places to sum at.
    """
    csr = _prepare_graph(urban_graph, weight=weight, cutoff=None, reverse=reverse)
    if betas is None:
        betas = band_betas(bands, decay, None)
    return csr.aggregate_scatter(
        np.ascontiguousarray(sources, dtype=np.int32),
        *values,
        n_cols,
        np.ascontiguousarray(slots, dtype=np.int32),
        int(n_slots),
        np.ascontiguousarray(bands, dtype=np.float32),
        decay,
        betas,
        max_workers,
    )
