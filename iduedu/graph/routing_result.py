"""Routing results and compact, graph-independent shortest-path storage."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from iduedu._numba.shortest_paths import reconstruct_numba_compact_path


def _path_series(values, index, name):
    # Construct an object array explicitly: equally long routes must remain
    # individual cells rather than becoming a two-dimensional numpy array.
    objects = np.empty(len(values), dtype=object)
    objects[:] = values
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

    def materialize(self) -> pd.Series | pd.DataFrame:
        if self.layout == "series":
            result = _path_series([self.path_at(i) for i in range(len(self.index))], self.index, "path")
        elif self.layout == "pairwise":
            result = _path_series([self.path_at(i) for i in range(len(self.index))], self.index, "path").to_frame()
        else:
            columns = [
                _path_series([self.path_at(i, j) for i in range(len(self.index))], self.index, label)
                for j, label in enumerate(self.columns)
            ]
            result = pd.concat(columns, axis=1) if columns else pd.DataFrame(index=self.index)
            result.columns = self.columns
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
