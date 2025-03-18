from collections import OrderedDict

import numba as nb
import numpy as np
from numba.core.types import float32, uint8, uint32

numba_csr_val_types = [float32[:], uint32[:], uint8[:]]


def __build_csr_cls(nb_type):

    spec_csr_matrix = [
        ("values", nb_type),
        ("col_index", nb.types.uint32[:]),
        ("row_index", nb.types.uint32[:]),
        ("nnz_rows", nb.types.uint32[:]),
        ("tot_rows", uint32),
    ]
    spec_csr_matrix = OrderedDict(spec_csr_matrix)

    @nb.experimental.jitclass(spec_csr_matrix)
    class CSRMatrix:  # pragma: no cover
        """
        a minimal CSR matrix implementation
        get_nnz and get_row should only be used on rows for which a value is present
        otherwise IndexErrors will be raised

        """

        def __init__(self, values, col_index, row_index):
            self.values = values
            self.col_index = col_index
            self.row_index = row_index
            self.nnz_rows = self.__set_nnz_rows()
            self.tot_rows = len(row_index) - 1

        def get_nnz(self, row):
            row_start = self.row_index[row]
            row_end = self.row_index[row + 1]
            return self.col_index[row_start:row_end]

        def get_row(self, row):
            row_start = self.row_index[row]
            row_end = self.row_index[row + 1]
            return self.values[row_start:row_end]

        def __set_nnz_rows(self):
            rows = []
            for row in np.arange(len(self.row_index[:-1]), dtype=np.uint32):
                if len(self.get_nnz(row)) > 0:
                    rows.append(row)
            return np.array(rows, dtype=np.uint32)

        def get_nnz_rows(self):
            # get rows that have non-zero values
            return self.nnz_rows

    return CSRMatrix


UI32CSRMatrix = __build_csr_cls(nb.uint32[:])

try:
    ui32csr_type = UI32CSRMatrix.class_type.instance_type
except Exception:
    ui32csr_type = None
