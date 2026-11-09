//! Python bindings: compiled into the `iduedu._rust` extension module.
//!
//! Node positions are int32, weights float32 (as in the SciPy CSR matrices of `UrbanGraph`). Every
//! kernel releases the GIL; parallel kernels run on a rayon pool of `n_threads` workers (`None`: one
//! worker per CPU).

pub mod components;
pub mod csr;
pub mod dijkstra;
pub mod pbf;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
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

/// Ways read from a ``.pbf`` file: ids, node-ref offsets, node refs, coordinate offsets, lon, lat, tags.
type WaysPy<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Vec<Bound<'py, PyDict>>,
);

/// Ways of an OSM ``.pbf`` file whose tags pass every clause ``(key, op, value, ignore_case)`` of an
/// Overpass filter, in file order, with the coordinates of their nodes (degrees, as libosmium gives them).
/// Way ``i`` has refs ``refs[ref_offsets[i]:ref_offsets[i + 1]]`` and coordinates
/// ``lon/lat[coord_offsets[i]:coord_offsets[i + 1]]``; nodes missing from the file add no coordinate.
#[pyfunction]
fn read_pbf_ways<'py>(
    py: Python<'py>,
    path: std::path::PathBuf,
    clauses: Vec<(String, String, Option<String>, bool)>,
) -> PyResult<WaysPy<'py>> {
    let clauses = clauses
        .into_iter()
        .map(|(key, op, value, ignore_case)| pbf::Clause::new(key, &op, value, ignore_case))
        .collect::<Result<Vec<_>, _>>()
        .map_err(PyValueError::new_err)?;
    let ways = py
        .detach(|| pbf::read_ways(&path, &clauses))
        .map_err(pyo3::exceptions::PyOSError::new_err)?;
    let degrees = |values: Vec<i32>| {
        values
            .into_iter()
            .map(|v| f64::from(v) / 1e7)
            .collect::<Vec<f64>>()
    };
    let tags = ways
        .tags
        .into_iter()
        .map(|pairs| {
            let dict = PyDict::new(py);
            for (key, value) in pairs {
                dict.set_item(key, value)?;
            }
            Ok(dict)
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok((
        ways.ids.into_pyarray(py),
        ways.ref_offsets.into_pyarray(py),
        ways.refs.into_pyarray(py),
        ways.coord_offsets.into_pyarray(py),
        degrees(ways.lon).into_pyarray(py),
        degrees(ways.lat).into_pyarray(py),
        tags,
    ))
}

#[pymodule(name = "_rust")]
fn iduedu_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CsrGraph>()?;
    m.add_function(wrap_pyfunction!(reconstruct_paths, m)?)?;
    m.add_function(wrap_pyfunction!(build_csr, m)?)?;
    m.add_function(wrap_pyfunction!(connected_components, m)?)?;
    m.add_function(wrap_pyfunction!(strongly_connected_components, m)?)?;
    m.add_function(wrap_pyfunction!(read_pbf_ways, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
