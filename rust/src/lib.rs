//! Python bindings: compiled into the `iduedu._rust` extension module.
//!
//! Node positions are int32, weights float32 (as in the SciPy CSR matrices of `UrbanGraph`). Every
//! kernel releases the GIL; parallel kernels run on a rayon pool of `n_threads` workers (`None`: one
//! worker per CPU).

pub mod analysis;
pub mod components;
pub mod csr;
pub mod dijkstra;
pub mod pbf;
pub mod pbf_write;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use csr::{Csr, Reduce};
use dijkstra::{Compact, Scratch};

/// Settled nodes, costs, parents (empty without predecessors) and owning sources of one search.
type CompactPy<'py> = (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
);

/// Routes as a flat node array and offsets.
type RoutesPy<'py> = (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i64>>);

/// CSR arrays: indptr, indices, data.
type CsrPy<'py> = (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
);

fn pools() -> &'static Mutex<HashMap<usize, Arc<rayon::ThreadPool>>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Run `f` on a cached pool of `n_threads` workers (`None` or `0`: the global pool).
fn in_pool<T: Send>(n_threads: Option<usize>, f: impl FnOnce() -> T + Send) -> PyResult<T> {
    match n_threads {
        Some(n) if n > 0 => {
            let pool = {
                let mut map = pools().lock().unwrap();
                match map.get(&n) {
                    Some(pool) => pool.clone(),
                    None => {
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(n)
                            .build()
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                        let pool = Arc::new(pool);
                        map.insert(n, pool.clone());
                        pool
                    }
                }
            };
            Ok(pool.install(f))
        }
        _ => Ok(f()),
    }
}

fn check_nodes(nodes: &[i32], n: usize, what: &str) -> PyResult<()> {
    if nodes.iter().any(|&v| v < 0 || v as usize >= n) {
        return Err(PyValueError::new_err(format!("{what} out of range")));
    }
    Ok(())
}

fn compact_py(py: Python<'_>, c: Compact) -> CompactPy<'_> {
    (
        c.nodes.into_pyarray(py),
        c.dist.into_pyarray(py),
        c.parents.into_pyarray(py),
        c.owners.into_pyarray(py),
    )
}

fn csr_py(py: Python<'_>, (indptr, indices, data): (Vec<i32>, Vec<i32>, Vec<f32>)) -> CsrPy<'_> {
    (
        indptr.into_pyarray(py),
        indices.into_pyarray(py),
        data.into_pyarray(py),
    )
}

/// Nearest objects of every source: flat nodes and costs, and offsets.
type NearestPy<'py> = (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<i64>>,
);

/// Betweenness sums of nodes and, when asked for, of arcs.
type BetweennessPy<'py> = (Bound<'py, PyArray2<f64>>, Option<Bound<'py, PyArray2<f64>>>);

fn check_cutoffs(cutoffs: &[f32]) -> PyResult<()> {
    if cutoffs.is_empty()
        || cutoffs.iter().any(|c| c.is_nan() || *c < 0.0)
        || cutoffs.windows(2).any(|w| w[1] < w[0])
    {
        return Err(PyValueError::new_err(
            "cutoffs must be a non-empty ascending array of non-negative values",
        ));
    }
    Ok(())
}

/// One decay per value column, from their names.
fn decays_of(names: &[String], n_cols: usize) -> PyResult<Vec<analysis::Decay>> {
    if names.len() != n_cols {
        return Err(PyValueError::new_err(
            "decay must name one decay per value column",
        ));
    }
    names
        .iter()
        .map(|name| analysis::Decay::parse(name).map_err(PyValueError::new_err))
        .collect()
}

fn bands_of<'a>(
    cutoffs: &'a [f32],
    betas: &'a [f32],
    decays: &'a [analysis::Decay],
) -> PyResult<analysis::Bands<'a>> {
    check_cutoffs(cutoffs)?;
    if betas.len() != cutoffs.len() || betas.iter().any(|b| !b.is_finite() || *b < 0.0) {
        return Err(PyValueError::new_err(
            "betas must hold one finite non-negative value per cutoff",
        ));
    }
    Ok(analysis::Bands {
        cutoffs,
        betas,
        decays,
    })
}

fn add_into(mut a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
    a
}

/// A weighted graph owned by Rust: the CSR arrays (validated once) and a pool of search buffers, so
/// repeated searches neither re-check the graph nor allocate arrays of the graph size.
#[pyclass(frozen, module = "iduedu._rust")]
struct CsrGraph {
    indptr: Vec<i32>,
    indices: Vec<i32>,
    data: Vec<f32>,
    scratch: Mutex<Vec<Scratch>>,
}

impl CsrGraph {
    fn view(&self) -> Csr<'_> {
        Csr {
            indptr: &self.indptr,
            indices: &self.indices,
            data: &self.data,
        }
    }

    /// Run `f` with a pooled scratch buffer (at most one per concurrent search).
    fn with_scratch<T>(&self, f: impl FnOnce(&mut Option<Scratch>) -> T) -> T {
        let mut sc = self.scratch.lock().unwrap().pop();
        let out = f(&mut sc);
        if let Some(sc) = sc {
            self.scratch.lock().unwrap().push(sc);
        }
        out
    }

    /// Run `step` for every index in `0..n` on the workers of the current pool. Each worker sums into
    /// its own accumulator from `init`; the accumulators are merged at the end (`None` for `n == 0`).
    fn fold_sources<A: Send>(
        &self,
        n: usize,
        init: impl Fn() -> A + Sync,
        step: impl Fn(&mut A, usize, &mut Scratch) + Sync,
        merge: impl Fn(A, A) -> A + Sync + Send,
    ) -> Option<A> {
        let next = AtomicUsize::new(0);
        let workers = rayon::current_num_threads().min(n);
        (0..workers)
            .into_par_iter()
            .map(|_| {
                let mut acc = init();
                self.with_scratch(|sc| {
                    let sc = sc.get_or_insert_with(|| Scratch::new(self.view().n()));
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        step(&mut acc, i, sc);
                    }
                });
                acc
            })
            .reduce_with(merge)
    }
}

#[pymethods]
impl CsrGraph {
    #[new]
    fn new(
        indptr: PyReadonlyArray1<'_, i32>,
        indices: PyReadonlyArray1<'_, i32>,
        data: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Self> {
        let g = Csr {
            indptr: indptr.as_slice()?,
            indices: indices.as_slice()?,
            data: data.as_slice()?,
        };
        g.validate(true).map_err(PyValueError::new_err)?;
        Ok(CsrGraph {
            indptr: g.indptr.to_vec(),
            indices: g.indices.to_vec(),
            data: g.data.to_vec(),
            scratch: Mutex::new(Vec::new()),
        })
    }

    /// Number of nodes.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.view().n()
    }

    /// Number of arcs.
    #[getter]
    fn n_arcs(&self) -> usize {
        self.indices.len()
    }

    /// The graph with every arc reversed.
    fn transpose(&self, py: Python<'_>) -> CsrGraph {
        let (indptr, indices, data) = py.detach(|| csr::transpose(self.view()));
        CsrGraph {
            indptr,
            indices,
            data,
            scratch: Mutex::new(Vec::new()),
        }
    }

    /// One search from all ``sources`` at once (each node is owned by its nearest source).
    ///
    /// ``targets`` empty: every reachable node; otherwise stop once all targets are settled.
    #[pyo3(signature = (sources, targets, cutoff, store_predecessors))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        sources: PyReadonlyArray1<'py, i32>,
        targets: PyReadonlyArray1<'py, i32>,
        cutoff: f32,
        store_predecessors: bool,
    ) -> PyResult<CompactPy<'py>> {
        let g = self.view();
        let (s, t) = (sources.as_slice()?, targets.as_slice()?);
        check_nodes(s, g.n(), "sources")?;
        check_nodes(t, g.n(), "targets")?;
        let c = py.detach(|| {
            self.with_scratch(|sc| dijkstra::search(g, s, t, cutoff, store_predecessors, sc))
        });
        Ok(compact_py(py, c))
    }

    /// One search per source, in parallel. Source ``i`` looks for
    /// ``targets[target_starts[i]:target_ends[i]]`` (an empty slice: every reachable node).
    #[pyo3(signature = (sources, targets, target_starts, target_ends, cutoff, store_predecessors, n_threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn search_many<'py>(
        &self,
        py: Python<'py>,
        sources: PyReadonlyArray1<'py, i32>,
        targets: PyReadonlyArray1<'py, i32>,
        target_starts: PyReadonlyArray1<'py, i64>,
        target_ends: PyReadonlyArray1<'py, i64>,
        cutoff: f32,
        store_predecessors: bool,
        n_threads: Option<usize>,
    ) -> PyResult<Vec<CompactPy<'py>>> {
        let g = self.view();
        let (s, t) = (sources.as_slice()?, targets.as_slice()?);
        let (starts, ends) = (target_starts.as_slice()?, target_ends.as_slice()?);
        check_nodes(s, g.n(), "sources")?;
        check_nodes(t, g.n(), "targets")?;
        if starts.len() != s.len() || ends.len() != s.len() {
            return Err(PyValueError::new_err(
                "target_starts and target_ends must match sources",
            ));
        }
        if starts
            .iter()
            .zip(ends)
            .any(|(&a, &b)| a < 0 || b < a || b as usize > t.len())
        {
            return Err(PyValueError::new_err("target slice out of range"));
        }
        let results = py.detach(|| {
            in_pool(n_threads, || {
                (0..s.len())
                    .into_par_iter()
                    .map(|i| {
                        let targets = &t[starts[i] as usize..ends[i] as usize];
                        self.with_scratch(|sc| {
                            dijkstra::search(
                                g,
                                &s[i..i + 1],
                                targets,
                                cutoff,
                                store_predecessors,
                                sc,
                            )
                        })
                    })
                    .collect::<Vec<_>>()
            })
        })?;
        Ok(results.into_iter().map(|c| compact_py(py, c)).collect())
    }

    /// Dense ``(len(sources), len(targets))`` float32 cost matrix, infinity where a target is not
    /// reached within ``cutoff``; one search per source, in parallel.
    #[pyo3(signature = (sources, targets, cutoff, n_threads=None))]
    fn distance_matrix<'py>(
        &self,
        py: Python<'py>,
        sources: PyReadonlyArray1<'py, i32>,
        targets: PyReadonlyArray1<'py, i32>,
        cutoff: f32,
        n_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let g = self.view();
        let (s, t) = (sources.as_slice()?, targets.as_slice()?);
        check_nodes(s, g.n(), "sources")?;
        check_nodes(t, g.n(), "targets")?;
        let m = t.len();
        let mut out = vec![f32::INFINITY; s.len() * m];
        if m > 0 {
            py.detach(|| {
                in_pool(n_threads, || {
                    out.par_chunks_mut(m)
                        .zip(s.par_iter())
                        .for_each(|(row, &src)| {
                            self.with_scratch(|sc| {
                                dijkstra::search_into_row(g, src, t, cutoff, sc, row)
                            })
                        })
                })
            })?;
        }
        let arr = Array2::from_shape_vec((s.len(), m), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// Sums of node values around every source: ``out[i, band, col]`` adds ``weight(d) * value`` over the
    /// nodes within ``cutoffs[band]`` of ``sources[i]``. Values are sparse rows per node
    /// (``val_indptr``, ``val_cols``, ``val_data``); ``decay`` is ``"flat"``, ``"linear"``, ``"exp"``
    /// (with ``betas`` per band) or ``"inverse"``. One search per source, in parallel.
    #[pyo3(signature = (sources, val_indptr, val_cols, val_data, n_cols, cutoffs, decay, betas, skip_source, n_threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn aggregate<'py>(
        &self,
        py: Python<'py>,
        sources: PyReadonlyArray1<'py, i32>,
        val_indptr: PyReadonlyArray1<'py, i64>,
        val_cols: PyReadonlyArray1<'py, i32>,
        val_data: PyReadonlyArray1<'py, f32>,
        n_cols: usize,
        cutoffs: PyReadonlyArray1<'py, f32>,
        decay: Vec<String>,
        betas: PyReadonlyArray1<'py, f32>,
        skip_source: bool,
        n_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let g = self.view();
        let s = sources.as_slice()?;
        check_nodes(s, g.n(), "sources")?;
        let values = analysis::Values {
            indptr: val_indptr.as_slice()?,
            cols: val_cols.as_slice()?,
            data: val_data.as_slice()?,
        };
        values
            .validate(g.n(), n_cols)
            .map_err(PyValueError::new_err)?;
        let (cutoffs, betas) = (cutoffs.as_slice()?, betas.as_slice()?);
        let decays = decays_of(&decay, n_cols)?;
        let bands = bands_of(cutoffs, betas, &decays)?;
        let width = cutoffs.len() * n_cols;
        let mut out = vec![0.0f64; s.len() * width];
        if width > 0 {
            py.detach(|| {
                in_pool(n_threads, || {
                    out.par_chunks_mut(width)
                        .zip(s.par_iter())
                        .for_each(|(row, &src)| {
                            self.with_scratch(|sc| {
                                let sc = sc.get_or_insert_with(|| Scratch::new(g.n()));
                                analysis::gather(
                                    g,
                                    sc,
                                    src,
                                    values,
                                    n_cols,
                                    bands,
                                    skip_source,
                                    row,
                                )
                            })
                        })
                })
            })?;
        }
        let arr = Array3::from_shape_vec((s.len(), cutoffs.len(), n_cols), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// The same sums as :meth:`aggregate`, computed from the side of the values: source ``i`` carries the
    /// sparse row ``i`` of values and spreads it to the nodes around it that have a slot
    /// (``slots[node] >= 0``). Returns ``(n_slots, bands, n_cols)``.
    #[pyo3(signature = (sources, val_indptr, val_cols, val_data, n_cols, slots, n_slots, cutoffs, decay, betas, n_threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn aggregate_scatter<'py>(
        &self,
        py: Python<'py>,
        sources: PyReadonlyArray1<'py, i32>,
        val_indptr: PyReadonlyArray1<'py, i64>,
        val_cols: PyReadonlyArray1<'py, i32>,
        val_data: PyReadonlyArray1<'py, f32>,
        n_cols: usize,
        slots: PyReadonlyArray1<'py, i32>,
        n_slots: usize,
        cutoffs: PyReadonlyArray1<'py, f32>,
        decay: Vec<String>,
        betas: PyReadonlyArray1<'py, f32>,
        n_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let g = self.view();
        let s = sources.as_slice()?;
        check_nodes(s, g.n(), "sources")?;
        let values = analysis::Values {
            indptr: val_indptr.as_slice()?,
            cols: val_cols.as_slice()?,
            data: val_data.as_slice()?,
        };
        values
            .validate(s.len(), n_cols)
            .map_err(PyValueError::new_err)?;
        let slots = slots.as_slice()?;
        if slots.len() != g.n() || slots.iter().any(|&x| x >= n_slots as i32) {
            return Err(PyValueError::new_err(
                "slots must give a slot below n_slots or -1 for every node",
            ));
        }
        let (cutoffs, betas) = (cutoffs.as_slice()?, betas.as_slice()?);
        let decays = decays_of(&decay, n_cols)?;
        let bands = bands_of(cutoffs, betas, &decays)?;
        let size = n_slots * cutoffs.len() * n_cols;
        let out = py.detach(|| {
            in_pool(n_threads, || {
                self.fold_sources(
                    s.len(),
                    || vec![0.0f64; size],
                    |acc, i, sc| {
                        let (a, b) = (values.indptr[i] as usize, values.indptr[i + 1] as usize);
                        analysis::scatter(
                            g,
                            sc,
                            s[i],
                            &values.cols[a..b],
                            &values.data[a..b],
                            n_cols,
                            slots,
                            bands,
                            acc,
                        )
                    },
                    add_into,
                )
                .unwrap_or_else(|| vec![0.0f64; size])
            })
        })?;
        let arr = Array3::from_shape_vec((n_slots, cutoffs.len(), n_cols), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// For every source the nearest nodes that hold objects (``counts[node] > 0``), nearest first, until
    /// ``k`` objects are found or ``cutoff`` is passed: flat node and cost arrays and
    /// ``len(sources) + 1`` offsets.
    #[pyo3(signature = (sources, counts, k, cutoff, n_threads=None))]
    fn nearest<'py>(
        &self,
        py: Python<'py>,
        sources: PyReadonlyArray1<'py, i32>,
        counts: PyReadonlyArray1<'py, i32>,
        k: i64,
        cutoff: f32,
        n_threads: Option<usize>,
    ) -> PyResult<NearestPy<'py>> {
        let g = self.view();
        let s = sources.as_slice()?;
        check_nodes(s, g.n(), "sources")?;
        let counts = counts.as_slice()?;
        if counts.len() != g.n() {
            return Err(PyValueError::new_err("counts must have one entry per node"));
        }
        let found: Vec<(Vec<i32>, Vec<f32>)> = py.detach(|| {
            in_pool(n_threads, || {
                s.par_iter()
                    .map(|&src| {
                        self.with_scratch(|sc| {
                            let sc = sc.get_or_insert_with(|| Scratch::new(g.n()));
                            let (mut nodes, mut costs) = (Vec::new(), Vec::new());
                            analysis::nearest(
                                g, sc, src, counts, k, cutoff, &mut nodes, &mut costs,
                            );
                            (nodes, costs)
                        })
                    })
                    .collect()
            })
        })?;
        let mut offsets = Vec::with_capacity(s.len() + 1);
        offsets.push(0i64);
        let (mut nodes, mut costs) = (Vec::new(), Vec::new());
        for (n, c) in found {
            nodes.extend(n);
            costs.extend(c);
            offsets.push(nodes.len() as i64);
        }
        Ok((
            nodes.into_pyarray(py),
            costs.into_pyarray(py),
            offsets.into_pyarray(py),
        ))
    }

    /// Betweenness sums over the shortest paths that start at ``sources``, once per cutoff band:
    /// ``(n_nodes, bands)`` for nodes and, with ``edges``, ``(n_arcs, bands)`` for the arcs of this graph
    /// in CSR order. Paths between ``s`` and ``t`` weigh ``weights[s] * weights[t]``.
    #[pyo3(signature = (sources, cutoffs, weights=None, edges=false, n_threads=None))]
    fn betweenness<'py>(
        &self,
        py: Python<'py>,
        sources: PyReadonlyArray1<'py, i32>,
        cutoffs: PyReadonlyArray1<'py, f32>,
        weights: Option<PyReadonlyArray1<'py, f32>>,
        edges: bool,
        n_threads: Option<usize>,
    ) -> PyResult<BetweennessPy<'py>> {
        let g = self.view();
        let s = sources.as_slice()?;
        check_nodes(s, g.n(), "sources")?;
        let cutoffs = cutoffs.as_slice()?;
        check_cutoffs(cutoffs)?;
        let weights = match &weights {
            Some(w) => {
                let w = w.as_slice()?;
                if w.len() != g.n() || w.iter().any(|x| !x.is_finite() || *x < 0.0) {
                    return Err(PyValueError::new_err(
                        "weights must hold one finite non-negative value per node",
                    ));
                }
                Some(w)
            }
            None => None,
        };
        let (n, m, bands) = (g.n(), g.indices.len(), cutoffs.len());
        let arcs = if edges { m * bands } else { 0 };
        let (nodes_out, arcs_out) = py.detach(|| {
            in_pool(n_threads, || {
                let rev = analysis::ReverseArcs::new(g);
                self.fold_sources(
                    s.len(),
                    || {
                        (
                            vec![0.0f64; n * bands],
                            vec![0.0f64; arcs],
                            analysis::Brandes::new(n),
                        )
                    },
                    |acc, i, sc| {
                        let edge_acc = if edges { Some(&mut acc.1[..]) } else { None };
                        analysis::betweenness(
                            g, &rev, sc, &mut acc.2, s[i], cutoffs, weights, &mut acc.0, edge_acc,
                        )
                    },
                    |mut a, b| {
                        a.0 = add_into(std::mem::take(&mut a.0), b.0);
                        a.1 = add_into(std::mem::take(&mut a.1), b.1);
                        a
                    },
                )
                .map(|acc| (acc.0, acc.1))
                .unwrap_or_else(|| (vec![0.0f64; n * bands], vec![0.0f64; arcs]))
            })
        })?;
        let nodes_arr = Array2::from_shape_vec((n, bands), nodes_out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let arcs_arr = if edges {
            Some(
                Array2::from_shape_vec((m, bands), arcs_out)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py),
            )
        } else {
            None
        };
        Ok((nodes_arr.into_pyarray(py), arcs_arr))
    }
}

/// Routes to ``targets`` from one compact search tree: a flat node array and ``len(targets) + 1``
/// offsets (an empty slice for a target that is not in the tree). ``reverse`` lists target first.
#[pyfunction]
fn reconstruct_paths<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray1<'py, i32>,
    parents: PyReadonlyArray1<'py, i32>,
    targets: PyReadonlyArray1<'py, i32>,
    reverse: bool,
) -> PyResult<RoutesPy<'py>> {
    let (n, p, t) = (nodes.as_slice()?, parents.as_slice()?, targets.as_slice()?);
    if n.len() != p.len() {
        return Err(PyValueError::new_err(
            "nodes and parents must have the same length",
        ));
    }
    let (flat, offsets) = py.detach(|| {
        let mut flat = Vec::new();
        let mut offsets = Vec::with_capacity(t.len() + 1);
        offsets.push(0i64);
        for &target in t {
            dijkstra::reconstruct(n, p, target, reverse, &mut flat);
            offsets.push(flat.len() as i64);
        }
        (flat, offsets)
    });
    Ok((flat.into_pyarray(py), offsets.into_pyarray(py)))
}

/// CSR matrix of ``n`` nodes from arcs ``rows[k] -> cols[k]`` with ``data[k]``; parallel arcs merged
/// with ``reduce`` (``"min"`` or ``"max"``), columns sorted within rows.
#[pyfunction]
fn build_csr<'py>(
    py: Python<'py>,
    n: usize,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f32>,
    reduce: &str,
) -> PyResult<CsrPy<'py>> {
    let (r, c, d) = (rows.as_slice()?, cols.as_slice()?, data.as_slice()?);
    let reduce = match reduce {
        "min" => Reduce::Min,
        "max" => Reduce::Max,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported reduce {other:?}"
            )))
        }
    };
    if r.len() != c.len() || r.len() != d.len() {
        return Err(PyValueError::new_err(
            "rows, cols and data must have the same length",
        ));
    }
    if r.iter().chain(c).any(|&v| v < 0 || v as usize >= n) {
        return Err(PyValueError::new_err("node position out of range"));
    }
    if r.len() > i32::MAX as usize {
        return Err(PyValueError::new_err("too many arcs for int32 CSR indices"));
    }
    Ok(csr_py(py, py.detach(|| csr::build(n, r, c, d, reduce))))
}

fn pattern<'a>(indptr: &'a [i32], indices: &'a [i32]) -> PyResult<()> {
    let data = vec![0f32; indices.len()];
    let g = Csr {
        indptr,
        indices,
        data: &data,
    };
    g.validate(false).map_err(PyValueError::new_err)
}

/// Connected component label of every node of a symmetric CSR pattern.
#[pyfunction]
fn connected_components<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i32>,
    indices: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let (ip, ix) = (indptr.as_slice()?, indices.as_slice()?);
    pattern(ip, ix)?;
    Ok(py.detach(|| components::connected(ip, ix)).into_pyarray(py))
}

/// Strongly connected component label of every node, given the CSR pattern and its transpose.
#[pyfunction]
fn strongly_connected_components<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i32>,
    indices: PyReadonlyArray1<'py, i32>,
    rindptr: PyReadonlyArray1<'py, i32>,
    rindices: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let (ip, ix) = (indptr.as_slice()?, indices.as_slice()?);
    let (rip, rix) = (rindptr.as_slice()?, rindices.as_slice()?);
    pattern(ip, ix)?;
    pattern(rip, rix)?;
    if ip.len() != rip.len() || ix.len() != rix.len() {
        return Err(PyValueError::new_err(
            "the transpose must have the same shape",
        ));
    }
    Ok(py
        .detach(|| components::strong(ip, ix, rip, rix))
        .into_pyarray(py))
}

fn tags_py(py: Python<'_>, tags: Vec<pbf::Tags>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    tags.into_iter()
        .map(|pairs| {
            let dict = PyDict::new(py);
            for (key, value) in pairs {
                dict.set_item(key, value)?;
            }
            Ok(dict)
        })
        .collect()
}

fn degrees(values: Vec<i32>) -> Vec<f64> {
    values.into_iter().map(|v| f64::from(v) / 1e7).collect()
}

/// Sorted unique ids from an optional int64 array.
fn sorted_ids(ids: Option<PyReadonlyArray1<'_, i64>>) -> Option<Vec<i64>> {
    ids.map(|ids| {
        let mut ids: Vec<i64> = ids.as_array().iter().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    })
}

fn read_error(e: String) -> PyErr {
    pyo3::exceptions::PyOSError::new_err(e)
}

/// Ways: ids, node-ref offsets, node refs, coordinate offsets, lon, lat, tags.
type WaysPy<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Vec<Bound<'py, PyDict>>,
);

/// Nodes: ids, lon, lat, tags.
type NodesPy<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Vec<Bound<'py, PyDict>>,
);

/// Relations: ids, member offsets, member types (0 node, 1 way, 2 relation), member refs, roles, tags.
type RelationsPy<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<i64>>,
    Vec<String>,
    Vec<Bound<'py, PyDict>>,
);

/// Ids of one output of `PbfFile.extract`: path, node ids, way ids, relation ids, optional bbox.
type OutputPy<'py> = (
    std::path::PathBuf,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    Option<(f64, f64, f64, f64)>,
);

/// An OpenStreetMap ``.pbf`` file. Queries decode blocks in parallel and return elements in file order;
/// the first one records a block index that lets later ones skip blocks that cannot match.
#[pyclass(frozen, module = "iduedu._rust")]
struct PbfFile {
    inner: pbf::PbfFile,
}

#[pymethods]
impl PbfFile {
    #[new]
    fn new(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Self> {
        let inner = py
            .detach(|| pbf::PbfFile::open(&path))
            .map_err(PyValueError::new_err)?;
        Ok(PbfFile { inner })
    }

    #[getter]
    fn path(&self) -> std::path::PathBuf {
        self.inner.path().to_path_buf()
    }

    /// The header block as a dict.
    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let header = py.detach(|| self.inner.header()).map_err(read_error)?;
        let dict = PyDict::new(py);
        dict.set_item("bbox", header.bbox.map(|b| (b[0], b[1], b[2], b[3])))?;
        dict.set_item("required_features", header.required_features)?;
        dict.set_item("optional_features", header.optional_features)?;
        dict.set_item("writing_program", header.writing_program)?;
        dict.set_item("source", header.source)?;
        dict.set_item(
            "osmosis_replication_timestamp",
            header.replication_timestamp,
        )?;
        dict.set_item(
            "osmosis_replication_sequence_number",
            header.replication_sequence_number,
        )?;
        dict.set_item("osmosis_replication_base_url", header.replication_base_url)?;
        Ok(dict)
    }

    /// Ways whose tags pass every clause ``(key, op, value, ignore_case)`` of an Overpass filter, carry one
    /// of ``any_keys`` and have an id in ``ids`` (each when given). Way ``i`` has refs
    /// ``refs[ref_offsets[i]:ref_offsets[i + 1]]`` and, with ``locations``, coordinates
    /// ``lon/lat[coord_offsets[i]:coord_offsets[i + 1]]`` of its nodes found in the file. ``bboxes``
    /// (``(minx, miny, maxx, maxy)`` in degrees, needs ``locations``) keeps the ways whose coordinate box
    /// meets one of them. Without ``tags`` the tag list is empty.
    #[pyo3(signature = (clauses=Vec::new(), *, any_keys=None, ids=None, bboxes=None, locations=true, tags=true))]
    #[allow(clippy::too_many_arguments)]
    fn ways<'py>(
        &self,
        py: Python<'py>,
        clauses: Vec<(String, String, Option<String>, bool)>,
        any_keys: Option<Vec<String>>,
        ids: Option<PyReadonlyArray1<'py, i64>>,
        bboxes: Option<Vec<(f64, f64, f64, f64)>>,
        locations: bool,
        tags: bool,
    ) -> PyResult<WaysPy<'py>> {
        let clauses = clauses
            .into_iter()
            .map(|(key, op, value, ignore_case)| pbf::Clause::new(key, &op, value, ignore_case))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        let ids = sorted_ids(ids);
        let bboxes: Option<Vec<[f64; 4]>> =
            bboxes.map(|b| b.into_iter().map(|(a, b, c, d)| [a, b, c, d]).collect());
        if bboxes.is_some() && !locations {
            return Err(PyValueError::new_err("bboxes needs locations=True"));
        }
        let query = pbf::WayQuery {
            clauses: &clauses,
            any_keys: any_keys.as_deref(),
            ids: ids.as_deref(),
            tags,
        };
        let file = &self.inner;
        let (ways, coord_offsets, lon, lat) = py
            .detach(|| {
                if locations {
                    pbf::ways_with_coords(file, &query, bboxes.as_deref())
                        .map(|w| (w.ways, w.coord_offsets, w.lon, w.lat))
                } else {
                    file.ways(&query).map(|w| {
                        let offsets = vec![0i64; w.ids.len() + 1];
                        (w, offsets, Vec::new(), Vec::new())
                    })
                }
            })
            .map_err(read_error)?;
        Ok((
            ways.ids.into_pyarray(py),
            ways.ref_offsets.into_pyarray(py),
            ways.refs.into_pyarray(py),
            coord_offsets.into_pyarray(py),
            degrees(lon).into_pyarray(py),
            degrees(lat).into_pyarray(py),
            tags_py(py, ways.tags)?,
        ))
    }

    /// The nodes with one of ``ids`` found in the file, with locations (degrees) and tags.
    fn nodes<'py>(
        &self,
        py: Python<'py>,
        ids: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<NodesPy<'py>> {
        let ids = sorted_ids(Some(ids)).unwrap_or_default();
        let nodes = py.detach(|| self.inner.nodes(&ids)).map_err(read_error)?;
        Ok((
            nodes.ids.into_pyarray(py),
            degrees(nodes.lon).into_pyarray(py),
            degrees(nodes.lat).into_pyarray(py),
            tags_py(py, nodes.tags)?,
        ))
    }

    /// Relations with an id in ``ids``, carrying one of the ``tags`` pairs and with a node member in
    /// ``contains_nodes``, each condition when given. Relation ``i`` has the members
    /// ``member_*[member_offsets[i]:member_offsets[i + 1]]``.
    #[pyo3(signature = (*, ids=None, tags=None, contains_nodes=None))]
    fn relations<'py>(
        &self,
        py: Python<'py>,
        ids: Option<PyReadonlyArray1<'py, i64>>,
        tags: Option<Vec<(String, String)>>,
        contains_nodes: Option<PyReadonlyArray1<'py, i64>>,
    ) -> PyResult<RelationsPy<'py>> {
        let ids = sorted_ids(ids);
        let contains_nodes = sorted_ids(contains_nodes);
        let query = pbf::RelationQuery {
            ids: ids.as_deref(),
            tags: tags.as_deref(),
            contains_nodes: contains_nodes.as_deref(),
        };
        let relations = py
            .detach(|| self.inner.relations(&query))
            .map_err(read_error)?;
        Ok((
            relations.ids.into_pyarray(py),
            relations.member_offsets.into_pyarray(py),
            relations.member_types.into_pyarray(py),
            relations.member_refs.into_pyarray(py),
            relations.member_roles,
            tags_py(py, relations.tags)?,
        ))
    }

    /// Write one new file per output ``(path, node_ids, way_ids, relation_ids, bbox)`` with exactly those
    /// elements, in one pass over this file. Headers keep the replication fields of this file.
    fn extract(
        &self,
        py: Python<'_>,
        outputs: Vec<OutputPy<'_>>,
        writing_program: String,
    ) -> PyResult<()> {
        let outputs: Vec<pbf_write::Output> = outputs
            .into_iter()
            .map(|(path, nodes, ways, relations, bbox)| pbf_write::Output {
                path,
                ids: [
                    sorted_ids(Some(nodes)).unwrap_or_default(),
                    sorted_ids(Some(ways)).unwrap_or_default(),
                    sorted_ids(Some(relations)).unwrap_or_default(),
                ],
                bbox: bbox.map(|(a, b, c, d)| [a, b, c, d]),
            })
            .collect();
        py.detach(|| pbf_write::extract(&self.inner, &outputs, &writing_program))
            .map_err(read_error)
    }
}

/// Node input of `write_pbf`: id, lon, lat (degrees) and tag pairs.
type NodeIn = (i64, f64, f64, Vec<(String, String)>);
/// Way input of `write_pbf`: id, node refs and tag pairs.
type WayIn = (i64, Vec<i64>, Vec<(String, String)>);
/// Relation input of `write_pbf`: id, members ``(type, ref, role)`` and tag pairs.
type RelationIn = (i64, Vec<(String, i64, String)>, Vec<(String, String)>);

/// Write a small ``.pbf`` file from element lists, nodes, then ways, then relations, each in the given
/// order. Member types are ``"n"``/``"node"``, ``"w"``/``"way"`` or ``"r"``/``"relation"``.
#[pyfunction]
#[pyo3(signature = (path, nodes, ways, relations, *, replication_timestamp=None, bbox=None, sorted=true, writing_program=None))]
#[allow(clippy::too_many_arguments)]
fn write_pbf(
    py: Python<'_>,
    path: std::path::PathBuf,
    nodes: Vec<NodeIn>,
    ways: Vec<WayIn>,
    relations: Vec<RelationIn>,
    replication_timestamp: Option<i64>,
    bbox: Option<(f64, f64, f64, f64)>,
    sorted: bool,
    writing_program: Option<String>,
) -> PyResult<()> {
    let to_int = |deg: f64| -> PyResult<i32> {
        let value = (deg * 1e7).round();
        if value.is_finite() && value.abs() <= f64::from(i32::MAX) {
            Ok(value as i32)
        } else {
            Err(PyValueError::new_err(format!("invalid coordinate {deg}")))
        }
    };
    let mut elements = Vec::with_capacity(nodes.len() + ways.len() + relations.len());
    for (id, lon, lat, tags) in nodes {
        elements.push(pbf_write::Element::Node {
            id,
            lon: to_int(lon)?,
            lat: to_int(lat)?,
            tags,
        });
    }
    for (id, refs, tags) in ways {
        elements.push(pbf_write::Element::Way { id, refs, tags });
    }
    for (id, members, tags) in relations {
        let members = members
            .into_iter()
            .map(|(kind, member, role)| {
                let code = match kind.as_str() {
                    "n" | "node" => pbf::Kind::Node,
                    "w" | "way" => pbf::Kind::Way,
                    "r" | "relation" => pbf::Kind::Relation,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown member type {other:?}"
                        )))
                    }
                };
                Ok((code as u8, member, role))
            })
            .collect::<PyResult<Vec<_>>>()?;
        elements.push(pbf_write::Element::Relation { id, members, tags });
    }
    let header = pbf::Header {
        bbox: bbox.map(|(a, b, c, d)| [a, b, c, d]),
        required_features: vec!["OsmSchema-V0.6".into(), "DenseNodes".into()],
        optional_features: if sorted {
            vec!["Sort.Type_then_ID".into()]
        } else {
            Vec::new()
        },
        writing_program: Some(writing_program.unwrap_or_else(|| "iduedu".into())),
        source: None,
        replication_timestamp,
        replication_sequence_number: None,
        replication_base_url: None,
    };
    py.detach(|| pbf_write::write(&path, &header, elements))
        .map_err(read_error)
}

#[pymodule(name = "_rust")]
fn iduedu_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CsrGraph>()?;
    m.add_function(wrap_pyfunction!(reconstruct_paths, m)?)?;
    m.add_function(wrap_pyfunction!(build_csr, m)?)?;
    m.add_function(wrap_pyfunction!(connected_components, m)?)?;
    m.add_function(wrap_pyfunction!(strongly_connected_components, m)?)?;
    m.add_class::<PbfFile>()?;
    m.add_function(wrap_pyfunction!(write_pbf, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
