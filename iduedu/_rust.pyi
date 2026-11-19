"""Type stubs of the compiled extension built from ``rust/`` (see ``rust/README.md``)."""

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt

__version__: str

Compact = tuple[npt.NDArray[np.int32], npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.int32]]

class LandmarkTable:
    """Landmarks of a graph with the costs from and to every node, for ALT searches."""

    @property
    def nodes(self) -> npt.NDArray[np.int32]: ...
    @property
    def nbytes(self) -> int: ...

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
    def shortest_pairs(
        self,
        reverse: "CsrGraph",
        sources: npt.NDArray[np.int32],
        targets: npt.NDArray[np.int32],
        cutoff: float,
        routes: bool,
        n_threads: int | None = None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
    def astar_pairs(
        self,
        xs: npt.NDArray[np.float64],
        ys: npt.NDArray[np.float64],
        sources: npt.NDArray[np.int32],
        targets: npt.NDArray[np.int32],
        cutoff: float,
        routes: bool,
        n_threads: int | None = None,
    ) -> tuple[tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.int64]], tuple[float, float]]: ...
    def landmarks(self, reverse: "CsrGraph", count: int, n_threads: int | None = None) -> "LandmarkTable": ...
    def alt_pairs(
        self,
        landmarks: "LandmarkTable",
        sources: npt.NDArray[np.int32],
        targets: npt.NDArray[np.int32],
        cutoff: float,
        routes: bool,
        n_threads: int | None = None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
    def alternatives(
        self,
        reverse: "CsrGraph",
        source: int,
        target: int,
        k: int,
        penalty: float,
        max_overlap: float,
        max_cost_ratio: float,
        cutoff: float,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
    def k_shortest(
        self,
        reverse: "CsrGraph",
        source: int,
        target: int,
        k: int,
        cutoff: float,
        n_threads: int | None = None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.int64]]: ...
    def distance_matrix(
        self,
        sources: npt.NDArray[np.int32],
        targets: npt.NDArray[np.int32],
        cutoff: float,
        n_threads: int | None = None,
    ) -> npt.NDArray[np.float32]: ...
    def aggregate(
        self,
        sources: npt.NDArray[np.int32],
        val_indptr: npt.NDArray[np.int64],
        val_cols: npt.NDArray[np.int32],
        val_data: npt.NDArray[np.float32],
        n_cols: int,
        cutoffs: npt.NDArray[np.float32],
        decay: list[str],
        betas: npt.NDArray[np.float32],
        skip_source: bool,
        n_threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def aggregate_scatter(
        self,
        sources: npt.NDArray[np.int32],
        val_indptr: npt.NDArray[np.int64],
        val_cols: npt.NDArray[np.int32],
        val_data: npt.NDArray[np.float32],
        n_cols: int,
        slots: npt.NDArray[np.int32],
        n_slots: int,
        cutoffs: npt.NDArray[np.float32],
        decay: list[str],
        betas: npt.NDArray[np.float32],
        n_threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def nearest(
        self,
        sources: npt.NDArray[np.int32],
        counts: npt.NDArray[np.int32],
        k: int,
        cutoff: float,
        n_threads: int | None = None,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
    def betweenness(
        self,
        sources: npt.NDArray[np.int32],
        cutoffs: npt.NDArray[np.float32],
        weights: npt.NDArray[np.float32] | None = None,
        edges: bool = False,
        n_threads: int | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]: ...

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
def merge_oriented_chains(
    starts: npt.NDArray[np.uint32],
    ends: npt.NDArray[np.uint32],
    keys: npt.NDArray[np.uint32],
    two_way: npt.NDArray[np.bool_],
    keep: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.bool_], npt.NDArray[np.int64]]: ...
def split_edges(
    xy: npt.NDArray[np.float64],
    coord_offsets: npt.NDArray[np.int64],
    distances: npt.NDArray[np.float64],
    cut_offsets: npt.NDArray[np.int64],
) -> tuple[
    npt.NDArray[np.float64],  # flat XY coordinates
    npt.NDArray[np.int64],  # piece coordinate offsets
    npt.NDArray[np.int64],  # source edge positions
    npt.NDArray[np.int64],  # start cut positions (-1 for u)
    npt.NDArray[np.int64],  # end cut positions (-2 for v)
]: ...

PtRoutes = tuple[
    npt.NDArray[np.int64],  # node route
    npt.NDArray[np.int64],  # node id: 2 * reference, + 1 for a made-up node
    npt.NDArray[np.float64],  # node xy, (n, 2)
    npt.NDArray[np.bool_],  # node is a platform
    npt.NDArray[np.int64],  # edge route
    npt.NDArray[np.int64],  # edge u
    npt.NDArray[np.int64],  # edge v
    npt.NDArray[np.bool_],  # edge is a boarding edge
    npt.NDArray[np.int64],  # edge point offsets, n_edges + 1
    npt.NDArray[np.float64],  # edge points, (m, 2)
    npt.NDArray[np.float64],  # edge speed, NaN for boarding edges
    list[tuple[int, str]],  # routes left out, with why
]

def parse_pt_routes(
    xy: npt.NDArray[np.float64],
    way_offsets: npt.NDArray[np.int64],
    way_points: npt.NDArray[np.int64],
    way_speeds: npt.NDArray[np.float64],
    member_offsets: npt.NDArray[np.int64],
    member_points: npt.NDArray[np.int64],
    member_refs: npt.NDArray[np.int64],
    member_ranks: npt.NDArray[np.int64],
    member_platform: npt.NDArray[np.bool_],
    member_stand_ins: npt.NDArray[np.int64],
    member_stand_in_xy: npt.NDArray[np.float64],
    kinds_apart: bool = False,
    n_threads: int | None = None,
) -> PtRoutes: ...
