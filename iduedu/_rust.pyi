"""Type stubs of the compiled extension built from ``rust/`` (see ``rust/README.md``)."""

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
