//! CSR graphs: a borrowed view used by the kernels and the builder that turns edge lists into it.

/// Borrowed CSR graph: row `u` holds the arcs `u -> indices[k]` with weights `data[k]`,
/// `k in indptr[u]..indptr[u + 1]`.
#[derive(Clone, Copy)]
pub struct Csr<'a> {
    pub indptr: &'a [i32],
    pub indices: &'a [i32],
    pub data: &'a [f32],
}

impl<'a> Csr<'a> {
    /// Number of nodes.
    #[inline]
    pub fn n(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    /// Neighbours of `u` and the arc weights.
    #[inline]
    pub fn row(&self, u: usize) -> (&'a [i32], &'a [f32]) {
        let s = self.indptr[u] as usize;
        let e = self.indptr[u + 1] as usize;
        (&self.indices[s..e], &self.data[s..e])
    }

    /// Check the structure, and that weights are non-negative and not NaN (infinity is allowed and
    /// means an impassable arc). Dijkstra and the packed heap keys rely on both.
    pub fn validate(&self, check_weights: bool) -> Result<(), String> {
        let (indptr, indices, data) = (self.indptr, self.indices, self.data);
        if indptr.is_empty() || indptr[0] != 0 {
            return Err("indptr must start with 0".into());
        }
        if indices.len() != data.len() {
            return Err("indices and data must have the same length".into());
        }
        if *indptr.last().unwrap() as usize != indices.len() {
            return Err("indptr does not match the number of entries".into());
        }
        if indptr.windows(2).any(|w| w[1] < w[0]) {
            return Err("indptr must be non-decreasing".into());
        }
        let n = self.n();
        if indices.iter().any(|&v| v < 0 || v as usize >= n) {
            return Err("CSR column index out of range".into());
        }
        if check_weights && data.iter().any(|&w| w.is_nan() || w < 0.0) {
            return Err("edge weights must be non-negative and not NaN".into());
        }
        Ok(())
    }
}

/// How parallel arcs between the same pair of nodes are combined.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reduce {
    Min,
    Max,
}

impl Reduce {
    #[inline]
    fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            Reduce::Min => a.min(b),
            Reduce::Max => a.max(b),
        }
    }
}

/// Build a CSR matrix of `n` nodes from arcs `rows[k] -> cols[k]` with weights `data[k]`.
///
/// Columns are sorted within each row and arcs between the same pair are merged with `reduce`
/// (unlike SciPy's `coo.tocsr()`, which sums them). Callers pass both directions of a two-way edge
/// explicitly, so parallel edges stored in opposite orientations are merged too.
pub fn build(
    n: usize,
    rows: &[i64],
    cols: &[i64],
    data: &[f32],
    reduce: Reduce,
) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let nnz = rows.len();
    let mut starts = vec![0usize; n + 1];
    for &r in rows {
        starts[r as usize + 1] += 1;
    }
    for i in 0..n {
        starts[i + 1] += starts[i];
    }
    let mut next = starts.clone();
    let mut entries = vec![(0i32, 0f32); nnz];
    for k in 0..nnz {
        let r = rows[k] as usize;
        entries[next[r]] = (cols[k] as i32, data[k]);
        next[r] += 1;
    }
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    indptr.push(0i32);
    for r in 0..n {
        let row = &mut entries[starts[r]..starts[r + 1]];
        row.sort_unstable_by_key(|e| e.0);
        let row_start = indices.len();
        for &(c, w) in row.iter() {
            if indices.len() > row_start && *indices.last().unwrap() == c {
                let last = values.last_mut().unwrap();
                *last = reduce.apply(*last, w);
            } else {
                indices.push(c);
                values.push(w);
            }
        }
        indptr.push(indices.len() as i32);
    }
    (indptr, indices, values)
}

/// Transpose of a CSR matrix (the graph with every arc reversed), columns sorted within rows.
pub fn transpose(g: Csr) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let n = g.n();
    let nnz = g.indices.len();
    let mut indptr = vec![0i32; n + 1];
    for &v in g.indices {
        indptr[v as usize + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }
    let mut next: Vec<usize> = indptr.iter().map(|&p| p as usize).collect();
    let mut indices = vec![0i32; nnz];
    let mut values = vec![0f32; nnz];
    // Rows are visited in increasing order, so every transposed row comes out sorted.
    for u in 0..n {
        let (cols, weights) = g.row(u);
        for (&v, &w) in cols.iter().zip(weights) {
            let p = next[v as usize];
            indices[p] = u as i32;
            values[p] = w;
            next[v as usize] += 1;
        }
    }
    (indptr, indices, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_reduces_parallel_arcs() {
        // 0 -> 1 twice (3.0 and 1.0), 1 -> 0 once, columns out of order in row 0.
        let rows = [0i64, 0, 1, 0];
        let cols = [1i64, 1, 0, 0];
        let data = [3.0f32, 1.0, 2.0, 5.0];
        let (ip, ix, d) = build(2, &rows, &cols, &data, Reduce::Min);
        assert_eq!(ip, vec![0, 2, 3]);
        assert_eq!(ix, vec![0, 1, 0]);
        assert_eq!(d, vec![5.0, 1.0, 2.0]);
        let (_, _, d) = build(2, &rows, &cols, &data, Reduce::Max);
        assert_eq!(d, vec![5.0, 3.0, 2.0]);
    }

    #[test]
    fn transpose_reverses_arcs() {
        let (ip, ix, d) = build(3, &[0, 0, 1], &[1, 2, 2], &[1.0, 2.0, 3.0], Reduce::Min);
        let g = Csr {
            indptr: &ip,
            indices: &ix,
            data: &d,
        };
        let (tp, tx, td) = transpose(g);
        assert_eq!(tp, vec![0, 0, 1, 3]);
        assert_eq!(tx, vec![0, 0, 1]);
        assert_eq!(td, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn validate_rejects_negative_weights() {
        let (ip, ix) = (vec![0, 1, 1], vec![1]);
        let bad = vec![-1.0f32];
        let g = Csr {
            indptr: &ip,
            indices: &ix,
            data: &bad,
        };
        assert!(g.validate(true).is_err());
        assert!(g.validate(false).is_ok());
    }
}
