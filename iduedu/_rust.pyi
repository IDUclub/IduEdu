"""Type stubs of the compiled extension built from ``rust/`` (see ``rust/README.md``)."""

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt

__version__: str

Compact = tuple[npt.NDArray[np.int32], npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.int32]]

class CsrGraph:
    """Weighted CSR graph owned by Rust, with pooled search buffers."""

    def __init__(
        self, indptr: npt.NDArray[np.int32], indices: npt.NDArray[np.int32], data: npt.NDArray[np.float32]
    ) -> None: ...
    @property
    def n_nodes(self) -> int: ...
    @property
    def n_arcs(self) -> int: ...
    def transpose(self) -> "CsrGraph": ...
    def search(
        self,
        sources: npt.NDArray[np.int32],
        targets: npt.NDArray[np.int32],
        cutoff: float,
        store_predecessors: bool,
    ) -> Compact: ...
    def search_many(
        self,
        sources: npt.NDArray[np.int32],
        targets: npt.NDArray[np.int32],
        target_starts: npt.NDArray[np.int64],
        target_ends: npt.NDArray[np.int64],
        cutoff: float,
        store_predecessors: bool,
        n_threads: int | None = None,
    ) -> list[Compact]: ...
    def distance_matrix(
        self,
        sources: npt.NDArray[np.int32],
        targets: npt.NDArray[np.int32],
        cutoff: float,
        n_threads: int | None = None,
    ) -> npt.NDArray[np.float32]: ...

def reconstruct_paths(
    nodes: npt.NDArray[np.int32], parents: npt.NDArray[np.int32], targets: npt.NDArray[np.int32], reverse: bool
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
def build_csr(
    n: int,
    rows: npt.NDArray[np.int64],
    cols: npt.NDArray[np.int64],
    data: npt.NDArray[np.float32],
    reduce: str,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float32]]: ...
def connected_components(indptr: npt.NDArray[np.int32], indices: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]: ...
def strongly_connected_components(
    indptr: npt.NDArray[np.int32],
    indices: npt.NDArray[np.int32],
    rindptr: npt.NDArray[np.int32],
    rindices: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]: ...

Ids = npt.NDArray[np.int64]
Tags = list[dict[str, str]]
Ways = tuple[Ids, Ids, Ids, Ids, npt.NDArray[np.float64], npt.NDArray[np.float64], Tags]
Relations = tuple[Ids, Ids, npt.NDArray[np.uint8], Ids, list[str], Tags]

class PbfFile:
    """An OpenStreetMap ``.pbf`` file; queries reuse the block index of the first pass."""

    def __init__(self, path: str | os.PathLike) -> None: ...
    @property
    def path(self) -> Path: ...
    def header(self) -> dict[str, object]: ...
    def ways(
        self,
        clauses: list[tuple[str, str, str | None, bool]] = ...,
        *,
        any_keys: list[str] | None = None,
        ids: Ids | None = None,
        bboxes: list[tuple[float, float, float, float]] | None = None,
        locations: bool = True,
        tags: bool = True,
    ) -> Ways: ...
    def nodes(self, ids: Ids) -> tuple[Ids, npt.NDArray[np.float64], npt.NDArray[np.float64], Tags]: ...
    def relations(
        self,
        *,
        ids: Ids | None = None,
        tags: list[tuple[str, str]] | None = None,
        contains_nodes: Ids | None = None,
    ) -> Relations: ...
    def extract(
        self,
        outputs: list[tuple[str | os.PathLike, Ids, Ids, Ids, tuple[float, float, float, float] | None]],
        writing_program: str,
    ) -> None: ...

def write_pbf(
    path: str | os.PathLike,
    nodes: list[tuple[int, float, float, list[tuple[str, str]]]],
    ways: list[tuple[int, list[int], list[tuple[str, str]]]],
    relations: list[tuple[int, list[tuple[str, int, str]], list[tuple[str, str]]]],
    *,
    replication_timestamp: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    sorted: bool = True,
    writing_program: str | None = None,
) -> None: ...
