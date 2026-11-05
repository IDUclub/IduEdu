"""Routing results and compact, graph-independent shortest-path storage."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas._libs.sparse import IntIndex  # pylint: disable=no-name-in-module

from iduedu._numba.shortest_paths import reconstruct_numba_compact_path


def _sparse_array(values, positions, length, dtype, fill_value):
    """Wrap stored values at sorted unique positions without a dense round trip."""
    # pandas >= 3.1 offers from_indices and deprecates the sparse_index argument.
    if hasattr(pd.arrays.SparseArray, "from_indices"):
        return pd.arrays.SparseArray.from_indices(values, positions, length, fill_value=fill_value, dtype=dtype)
    return pd.arrays.SparseArray(
        values,
        sparse_index=IntIndex(length, np.asarray(positions, dtype=np.int32)),
        dtype=pd.SparseDtype(dtype, fill_value),
    )


def _sparse_frame(rows, cols, values, index, columns, dtype, fill_value):
    """Build a sparse table column by column from coordinate entries."""
    order = np.lexsort((rows, cols))
    rows, cols, values = rows[order], cols[order], values[order]
    bounds = np.searchsorted(cols, np.arange(len(columns) + 1))
    arrays = {
        j: _sparse_array(values[start:end], rows[start:end], len(index), dtype, fill_value)
        for j, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]))
    }
    result = pd.DataFrame(arrays, index=index) if arrays else pd.DataFrame(index=index)
    result.columns = columns
    return result


def _groups(labels, count):
    """Return positions ordered by label and offsets of each label's run."""
    order = np.argsort(labels, kind="stable")
    offsets = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.bincount(labels, minlength=count), out=offsets[1:])
    return order, offsets


def _object_array(values):
    # Fill cells one by one: equally long routes must remain individual cells
    # rather than becoming a two-dimensional numpy array.
    objects = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        objects[i] = value
    return objects


def _path_series(values, index, name):
    objects = _object_array(values)
    return pd.Series(pd.arrays.SparseArray(objects, fill_value=None), index=index, name=name)


@dataclass
class _PathTree:
    """Only settled node positions and their parents, sorted by node position."""

    nodes: np.ndarray
    parents: np.ndarray

    def reconstruct(self, target: int, node_ids: np.ndarray, reverse: bool) -> list[Any] | None:
        positions = reconstruct_numba_compact_path(self.nodes, self.parents, np.int32(target), reverse)
        return node_ids[positions].tolist() if len(positions) else None


@dataclass
class _PathStore:
    """Snapshot of search trees and the mapping to public result cells."""

    # The store explicitly records both search orientation and public layout.
    # pylint: disable=too-many-instance-attributes

    trees: list[_PathTree]
    node_ids: np.ndarray
    tree_indices: np.ndarray
    origins: np.ndarray
    targets: np.ndarray
    index: pd.Index
    columns: pd.Index | None
    layout: Literal["series", "matrix", "pairwise"] = "matrix"
    transposed: bool = False
    reverse: bool = False
    attrs: dict | None = None

    def path_at(self, row: int, column: int = 0) -> list[Any] | None:
        if self.layout == "series":
            tree_index, target = 0, self.targets[row]
        elif self.layout == "pairwise":
            tree_index, target = self.tree_indices[row], self.targets[row]
        elif self.transposed:
            tree_index, target = self.tree_indices[column], self.origins[row]
        else:
            tree_index, target = self.tree_indices[row], self.targets[column]
        return self.trees[tree_index].reconstruct(int(target), self.node_ids, self.reverse)

    def _matrix_paths(self) -> pd.DataFrame:
        # Reconstruct only cells whose target settled in the tree; rows that
        # share a tree (repeated sources) reuse the route as separate lists.
        owners, targets = self.tree_indices, self.origins if self.transposed else self.targets
        order, offsets = _groups(owners, len(self.trees))
        rows, cols, values = [], [], []
        for tree_index, tree in enumerate(self.trees):
            members = order[offsets[tree_index] : offsets[tree_index + 1]]
            if members.size == 0 or tree.nodes.size == 0:
                continue
            found = np.searchsorted(tree.nodes, targets).clip(max=len(tree.nodes) - 1)
            found = np.flatnonzero(tree.nodes[found] == targets)
            routes = [tree.reconstruct(int(targets[k]), self.node_ids, self.reverse) for k in found]
            for owner in members:
                owner_axis = np.full(len(found), owner)
                rows.append(found if self.transposed else owner_axis)
                cols.append(owner_axis if self.transposed else found)
                values.extend(list(route) for route in routes)
        stacked = (np.concatenate(axis) if axis else np.empty(0, dtype=np.int64) for axis in (rows, cols))
        return _sparse_frame(*stacked, _object_array(values), self.index, self.columns, object, None)

    def materialize(self) -> pd.Series | pd.DataFrame:
        if self.layout == "series":
            result = _path_series([self.path_at(i) for i in range(len(self.index))], self.index, "path")
        elif self.layout == "pairwise":
            result = _path_series([self.path_at(i) for i in range(len(self.index))], self.index, "path").to_frame()
        else:
            result = self._matrix_paths()
        result.attrs.update(self.attrs or {})
        return result


class ShortestPathResult:
    """Distances and paths from the same Dijkstra searches.

    ``distances`` is a sparse Series or DataFrame. Accessing ``paths`` creates
    and caches the matching path table. With ``path_storage="lazy"``, use
    ``get_path`` to reconstruct individual routes without materializing that
    table or rerunning Dijkstra. Results retain a snapshot of node ids and
    predecessors, so later graph edits do not change the saved routes.

    For single-source and nearest-source results, call ``get_path(target)``.
    For matrices and pairwise results, call ``get_path(origin, destination)``
    using the result labels (object labels when object tables were supplied).
    Duplicate labels require positional access through ``get_path_at``.

    Attributes:
        distances: Shortest costs in the selected weight units.
        nearest_sources: Winning graph node ids for a nearest-source search;
            otherwise None. Unreachable requested targets have a None owner.
    """

    def __init__(self, distances, path_store: _PathStore, *, path_storage="eager", nearest_sources=None):
        self.distances = distances
        self.nearest_sources = nearest_sources
        self._store = path_store
        self._index = path_store.index
        self._columns = path_store.columns
        self._layout = path_store.layout
        self._paths = None
        if path_storage == "eager":
            self._materialize()

    def _materialize(self):
        if self._paths is None:
            self._paths = self._store.materialize()
            self._store = None
        return self._paths

    @property
    def paths(self) -> pd.Series | pd.DataFrame:
        """Materialize and cache the sparse path table, without another search."""
        return self._materialize()

    @property
    def paths_materialized(self) -> bool:
        """Whether the complete path table has been created."""
        return self._paths is not None

    @staticmethod
    def _position(index, label):
        position = index.get_loc(label)
        if not isinstance(position, (int, np.integer)):
            raise ValueError("Path labels are ambiguous; use get_path_at() for duplicate labels")
        return int(position)

    def get_path(self, origin: Any, destination: Any = None) -> list[Any] | None:
        """Return one route by labels, or None if it is unreachable or beyond cutoff."""
        if self._layout == "series":
            if destination is not None:
                raise TypeError("Series results use get_path(target)")
            return self.get_path_at(self._position(self._index, origin))
        if self._layout == "pairwise":
            return self.get_path_at(self._position(self._index, (origin, destination)))
        return self.get_path_at(self._position(self._index, origin), self._position(self._columns, destination))

    def get_path_at(self, row: int, column: int = 0) -> list[Any] | None:
        """Return a route by zero-based table positions (also supports duplicate labels)."""
        if not isinstance(row, (int, np.integer)) or not isinstance(column, (int, np.integer)):
            raise TypeError("row and column must be integers")
        width = len(self._columns) if self._layout == "matrix" else 1
        if not 0 <= row < len(self._index) or not 0 <= column < width:
            raise IndexError("Path position is outside the result")
        if self._store is not None:
            return self._store.path_at(row, column)
        path = self._paths.iloc[row] if self._layout == "series" else self._paths.iloc[row, column]
        return list(path) if isinstance(path, list) else None
