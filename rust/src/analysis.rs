//! Network analysis on top of the Dijkstra searches: sums of values within a travel cost, nearest
//! objects and betweenness.
//!
//! Every kernel works on one source at a time, so the bindings can run sources in parallel. Travel
//! costs are float32 like the graph weights; sums are float64.

use crate::csr::Csr;
use crate::dijkstra::Scratch;

/// How a value counts at travel cost `d` from the source, within a band of cutoff `c`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decay {
    /// Weight 1.
    Flat,
    /// `1 - d / c` (weight 1 for an infinite cutoff).
    Linear,
    /// `exp(-beta * d)` with the beta of the band.
    Exp,
    /// `1 / d`; values at cost 0 do not count.
    Inverse,
}

impl Decay {
    pub fn parse(name: &str) -> Result<Decay, String> {
        match name {
            "flat" => Ok(Decay::Flat),
            "linear" => Ok(Decay::Linear),
            "exp" => Ok(Decay::Exp),
            "inverse" => Ok(Decay::Inverse),
            other => Err(format!("unknown decay {other:?}")),
        }
    }

    #[inline]
    fn weight(self, d: f32, cutoff: f32, beta: f32) -> f64 {
        match self {
            Decay::Flat => 1.0,
            Decay::Linear => {
                if cutoff.is_finite() && cutoff > 0.0 {
                    1.0 - f64::from(d) / f64::from(cutoff)
                } else {
                    1.0
                }
            }
            Decay::Exp => (-f64::from(beta) * f64::from(d)).exp(),
            Decay::Inverse => {
                if d > 0.0 {
                    1.0 / f64::from(d)
                } else {
                    0.0
                }
            }
        }
    }
}

/// Cost bands of one aggregation: ascending inclusive `cutoffs`, each with its own `beta`.
#[derive(Clone, Copy)]
pub struct Bands<'a> {
    pub cutoffs: &'a [f32],
    pub betas: &'a [f32],
    pub decay: Decay,
}

impl Bands<'_> {
    fn max(&self) -> f32 {
        self.cutoffs.last().copied().unwrap_or(f32::INFINITY)
    }

    /// Index of the first band that holds cost `d`.
    #[inline]
    fn first(&self, d: f32) -> usize {
        self.cutoffs.partition_point(|&c| c < d)
    }
}

/// Sparse rows of values: row `i` holds `(cols[k], data[k])` for `k` in `indptr[i]..indptr[i + 1]`.
#[derive(Clone, Copy)]
pub struct Values<'a> {
    pub indptr: &'a [i64],
    pub cols: &'a [i32],
    pub data: &'a [f32],
}

impl<'a> Values<'a> {
    #[inline]
    fn row(&self, i: usize) -> (&'a [i32], &'a [f32]) {
        let (a, b) = (self.indptr[i] as usize, self.indptr[i + 1] as usize);
        (&self.cols[a..b], &self.data[a..b])
    }

    /// Check the arrays against `rows` rows and `n_cols` columns.
    pub fn validate(&self, rows: usize, n_cols: usize) -> Result<(), String> {
        if self.indptr.len() != rows + 1 || self.indptr.first() != Some(&0) {
            return Err("values indptr must have one entry per row plus one and start at 0".into());
        }
        if self.indptr.windows(2).any(|w| w[1] < w[0])
            || *self.indptr.last().unwrap() as usize != self.cols.len()
            || self.cols.len() != self.data.len()
        {
            return Err("values arrays are inconsistent".into());
        }
        if self.cols.iter().any(|&c| c < 0 || c as usize >= n_cols) {
            return Err("values column out of range".into());
        }
        Ok(())
    }
}

/// Sum the values of the nodes around `source` into `out` (`bands × n_cols`, zeroed by the caller):
/// `out[band][col] = Σ weight(d) · value` over nodes within the band's cutoff. `values` has one row per
/// node. With `skip_source` the values at the source node itself do not count.
#[allow(clippy::too_many_arguments)]
pub fn gather(
    g: Csr,
    sc: &mut Scratch,
    source: i32,
    values: Values,
    n_cols: usize,
    bands: Bands,
    skip_source: bool,
    out: &mut [f64],
) {
    let n_bands = bands.cutoffs.len();
    sc.walk(g, source, bands.max(), |node, d| {
        if skip_source && node == source as usize {
            return true;
        }
        let (cols, data) = values.row(node);
        if cols.is_empty() {
            return true;
        }
        for band in bands.first(d)..n_bands {
            let w = bands
                .decay
                .weight(d, bands.cutoffs[band], bands.betas[band]);
            let row = &mut out[band * n_cols..(band + 1) * n_cols];
            for (&c, &v) in cols.iter().zip(data) {
                row[c as usize] += w * f64::from(v);
            }
        }
        true
    });
}

/// Spread the values of one source (`cols`, `data`) to the nodes around it:
/// `acc[slot][band][col] += weight(d) · value` for every node with a slot (`slots[node] >= 0`) within
/// the band's cutoff. `acc` is `n_slots × bands × n_cols`.
#[allow(clippy::too_many_arguments)]
pub fn scatter(
    g: Csr,
    sc: &mut Scratch,
    source: i32,
    cols: &[i32],
    data: &[f32],
    n_cols: usize,
    slots: &[i32],
    bands: Bands,
    acc: &mut [f64],
) {
    if cols.is_empty() {
        return;
    }
    let n_bands = bands.cutoffs.len();
    sc.walk(g, source, bands.max(), |node, d| {
        let slot = slots[node];
        if slot < 0 {
            return true;
        }
        let base = slot as usize * n_bands;
        for band in bands.first(d)..n_bands {
            let w = bands
                .decay
                .weight(d, bands.cutoffs[band], bands.betas[band]);
            let row = &mut acc[(base + band) * n_cols..(base + band + 1) * n_cols];
            for (&c, &v) in cols.iter().zip(data) {
                row[c as usize] += w * f64::from(v);
            }
        }
        true
    });
}

/// The nearest nodes holding objects: walks from `source` and appends `(node, cost)` of every node with
/// `counts[node] > 0` until at least `k` objects are found or `cutoff` is passed.
#[allow(clippy::too_many_arguments)]
pub fn nearest(
    g: Csr,
    sc: &mut Scratch,
    source: i32,
    counts: &[i32],
    k: i64,
    cutoff: f32,
    nodes: &mut Vec<i32>,
    costs: &mut Vec<f32>,
) {
    let mut found = 0i64;
    sc.walk(g, source, cutoff, |node, d| {
        let c = counts[node];
        if c > 0 {
            nodes.push(node as i32);
            costs.push(d);
            found += i64::from(c);
        }
        found < k
    });
}

/// Incoming arcs of every node: `(indptr, tail, arc)` where `arc` is the position of the arc in `g`.
pub struct ReverseArcs {
    indptr: Vec<usize>,
    tail: Vec<u32>,
    arc: Vec<u32>,
}

impl ReverseArcs {
    pub fn new(g: Csr) -> ReverseArcs {
        let n = g.n();
        let mut indptr = vec![0usize; n + 1];
        for &v in g.indices {
            indptr[v as usize + 1] += 1;
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }
        let mut next = indptr.clone();
        let mut tail = vec![0u32; g.indices.len()];
        let mut arc = vec![0u32; g.indices.len()];
        for u in 0..n {
            for a in g.indptr[u] as usize..g.indptr[u + 1] as usize {
                let v = g.indices[a] as usize;
                tail[next[v]] = u as u32;
                arc[next[v]] = a as u32;
                next[v] += 1;
            }
        }
        ReverseArcs { indptr, tail, arc }
    }
}

/// Per-worker buffers of the betweenness kernel.
pub struct Brandes {
    sigma: Vec<f64>,
    delta: Vec<f64>,
    rank: Vec<u32>,
}

const UNRANKED: u32 = u32::MAX;

impl Brandes {
    pub fn new(n: usize) -> Brandes {
        Brandes {
            sigma: vec![0.0; n],
            delta: vec![0.0; n],
            rank: vec![UNRANKED; n],
        }
    }
}

/// Add the shortest paths that start at `source` to the betweenness sums (Brandes' dependency
/// accumulation), once per cutoff band: only paths no longer than the cutoff count.
///
/// `node_acc` is `n × bands`, `edge_acc` (optional) `arcs × bands`, indexed by the arc positions of `g`.
/// Paths between `s` and `t` weigh `weights[s] · weights[t]` (1 without weights). End points are not on
/// their own paths. Equal-cost paths share their weight.
#[allow(clippy::too_many_arguments)]
pub fn betweenness(
    g: Csr,
    rev: &ReverseArcs,
    sc: &mut Scratch,
    work: &mut Brandes,
    source: i32,
    cutoffs: &[f32],
    weights: Option<&[f32]>,
    node_acc: &mut [f64],
    mut edge_acc: Option<&mut [f64]>,
) {
    let n_bands = cutoffs.len();
    let source_weight = weights.map_or(1.0, |w| f64::from(w[source as usize]));
    if source_weight == 0.0 {
        return;
    }
    sc.run_single(g, source, cutoffs.last().copied().unwrap_or(f32::INFINITY));
    let order = sc.settled_order();
    for (i, &v) in order.iter().enumerate() {
        work.rank[v as usize] = i as u32;
    }
    // Number of shortest paths from the source, in settling order.
    work.sigma[source as usize] = 1.0;
    for &w in &order[1..] {
        let w = w as usize;
        let mut sigma = 0.0;
        for k in rev.indptr[w]..rev.indptr[w + 1] {
            let v = rev.tail[k] as usize;
            if work.rank[v] < work.rank[w] && sc.cost(v) + g.data[rev.arc[k] as usize] == sc.cost(w)
            {
                sigma += work.sigma[v];
            }
        }
        work.sigma[w] = sigma;
    }

    for (band, &cutoff) in cutoffs.iter().enumerate() {
        let reach = order.partition_point(|&v| sc.cost(v as usize) <= cutoff);
        for &w in &order[..reach] {
            work.delta[w as usize] = 0.0;
        }
        for &w in order[1..reach].iter().rev() {
            let w = w as usize;
            let target_weight = weights.map_or(1.0, |x| f64::from(x[w]));
            let delta_w = work.delta[w];
            let share = (target_weight + delta_w) / work.sigma[w];
            for k in rev.indptr[w]..rev.indptr[w + 1] {
                let v = rev.tail[k] as usize;
                let arc = rev.arc[k] as usize;
                if work.rank[v] < work.rank[w] && sc.cost(v) + g.data[arc] == sc.cost(w) {
                    let through = work.sigma[v] * share;
                    work.delta[v] += through;
                    if let Some(edges) = edge_acc.as_deref_mut() {
                        edges[arc * n_bands + band] += source_weight * through;
                    }
                }
            }
            node_acc[w * n_bands + band] += source_weight * delta_w;
        }
    }

    for &v in order {
        let v = v as usize;
        work.rank[v] = UNRANKED;
        work.sigma[v] = 0.0;
        work.delta[v] = 0.0;
    }
    sc.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::{build, Reduce};

    fn graph(
        n: usize,
        edges: &[(i64, i64, f32)],
        undirected: bool,
    ) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for &(u, v, w) in edges {
            rows.push(u);
            cols.push(v);
            data.push(w);
            if undirected {
                rows.push(v);
                cols.push(u);
                data.push(w);
            }
        }
        build(n, &rows, &cols, &data, Reduce::Min)
    }

    #[test]
    fn gather_and_scatter_agree() {
        // Path 0 - 1 - 2 - 3 with unit weights; a value of 10 at node 0 and 1 at node 3.
        let (indptr, indices, data) = graph(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)], true);
        let g = Csr {
            indptr: &indptr,
            indices: &indices,
            data: &data,
        };
        let mut sc = Scratch::new(4);
        let values = Values {
            indptr: &[0, 1, 1, 1, 2],
            cols: &[0, 0],
            data: &[10.0, 1.0],
        };
        values.validate(4, 1).unwrap();
        let bands = Bands {
            cutoffs: &[1.0, 3.0],
            betas: &[0.0, 0.0],
            decay: Decay::Flat,
        };
        let mut out = vec![0.0; 2];
        gather(g, &mut sc, 1, values, 1, bands, false, &mut out);
        assert_eq!(out, vec![10.0, 11.0]);

        let linear = Bands {
            decay: Decay::Linear,
            ..bands
        };
        let mut out = vec![0.0; 2];
        gather(g, &mut sc, 1, values, 1, linear, false, &mut out);
        assert!(
            (out[0] - 0.0).abs() < 1e-12
                && (out[1] - (10.0 * (1.0 - 1.0 / 3.0) + 1.0 / 3.0)).abs() < 1e-12
        );

        // The same sums by spreading each value from its own node.
        let slots = [-1, 0, -1, -1];
        let mut acc = vec![0.0; 2];
        scatter(g, &mut sc, 0, &[0], &[10.0], 1, &slots, bands, &mut acc);
        scatter(g, &mut sc, 3, &[0], &[1.0], 1, &slots, bands, &mut acc);
        assert_eq!(acc, vec![10.0, 11.0]);

        let mut skipped = vec![0.0; 2];
        gather(g, &mut sc, 0, values, 1, bands, true, &mut skipped);
        assert_eq!(skipped, vec![0.0, 1.0]);
    }

    #[test]
    fn nearest_stops_after_k_objects() {
        let (indptr, indices, data) = graph(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)], true);
        let g = Csr {
            indptr: &indptr,
            indices: &indices,
            data: &data,
        };
        let mut sc = Scratch::new(4);
        let (mut nodes, mut costs) = (Vec::new(), Vec::new());
        nearest(
            g,
            &mut sc,
            0,
            &[0, 2, 1, 5],
            3,
            f32::INFINITY,
            &mut nodes,
            &mut costs,
        );
        assert_eq!((nodes.clone(), costs.clone()), (vec![1, 2], vec![1.0, 2.0]));
        nodes.clear();
        costs.clear();
        nearest(g, &mut sc, 0, &[0, 2, 1, 5], 9, 2.5, &mut nodes, &mut costs);
        assert_eq!(nodes, vec![1, 2]);
    }

    fn all_betweenness(
        n: usize,
        edges: &[(i64, i64, f32)],
        undirected: bool,
        cutoffs: &[f32],
    ) -> (Vec<f64>, Vec<f64>) {
        let (indptr, indices, data) = graph(n, edges, undirected);
        let g = Csr {
            indptr: &indptr,
            indices: &indices,
            data: &data,
        };
        let rev = ReverseArcs::new(g);
        let mut sc = Scratch::new(n);
        let mut work = Brandes::new(n);
        let mut nodes = vec![0.0; n * cutoffs.len()];
        let mut arcs = vec![0.0; indices.len() * cutoffs.len()];
        for s in 0..n {
            betweenness(
                g,
                &rev,
                &mut sc,
                &mut work,
                s as i32,
                cutoffs,
                None,
                &mut nodes,
                Some(&mut arcs),
            );
        }
        (nodes, arcs)
    }

    #[test]
    fn betweenness_of_a_path_and_a_square() {
        // Path 0 - 1 - 2 - 3: ordered pairs through node 1 are (0,2), (0,3), (2,0), (3,0).
        let path = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
        let (nodes, arcs) = all_betweenness(4, &path, true, &[f32::INFINITY]);
        assert_eq!(nodes, vec![0.0, 4.0, 4.0, 0.0]);
        // Arc 0 -> 1 carries the paths from 0 to 1, 2 and 3.
        assert_eq!(arcs[0], 3.0);
        // With a cutoff of 2 only the pairs two steps apart remain.
        let (nodes, _) = all_betweenness(4, &path, true, &[1.0, 2.0]);
        assert_eq!(nodes, vec![0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0]);

        // Square 0 - 1 - 2 - 3 - 0: opposite corners are joined by two equal paths, half a path each way.
        let square = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)];
        let (nodes, _) = all_betweenness(4, &square, true, &[f32::INFINITY]);
        assert_eq!(nodes, vec![1.0; 4]);

        // A one-way path has paths in one direction only.
        let (nodes, _) = all_betweenness(3, &[(0, 1, 1.0), (1, 2, 1.0)], false, &[f32::INFINITY]);
        assert_eq!(nodes, vec![0.0, 1.0, 0.0]);
    }
}
