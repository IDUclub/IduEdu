//! Flow assignment along shortest-path forests (gravity-type distribution of trips).
//!
//! Every cluster of sources runs one multi-source Dijkstra to the inclusive `cutoff`. Each reached node
//! other than a source is a destination weighted by `node_weights[node] * decay(cost)`; the summed flow of the
//! cluster's sources is split between the destinations in proportion to these weights and carried back
//! along the search tree towards the sources. An arc keeps the flow of a cluster only when that flow
//! exceeds `min_flow`, or when the arc leaves a source and the flow is positive; the kept flows of all
//! clusters are summed per arc.
//!
//! Search ties follow [`crate::dijkstra`]: at an equal cost a node joins the source with the lower
//! position. Costs are summed in float32, flows in float64.

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::csr::Csr;
use crate::dijkstra::Scratch;

/// Weight of a destination by its cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlowDecay {
    /// Weight 1.
    Flat,
    /// `exp(-beta * t)`.
    Exp { beta: f64 },
    /// `exp(-0.5 * ((t - mean) / std)^2)`.
    Gaussian { mean: f64, std: f64 },
}

impl FlowDecay {
    #[inline]
    fn weight(&self, t: f64) -> f64 {
        match *self {
            FlowDecay::Flat => 1.0,
            FlowDecay::Exp { beta } => (-beta * t).exp(),
            FlowDecay::Gaussian { mean, std } => {
                let z = (t - mean) / std;
                (-0.5 * z * z).exp()
            }
        }
    }
}

/// Position of the cheapest arc `u -> v` (the first of equally cheap ones, the arc the search relaxed).
fn tree_arc(g: Csr, u: usize, v: usize) -> usize {
    let (cols, weights) = g.row(u);
    let mut best = (usize::MAX, f32::INFINITY);
    for (i, (&c, &w)) in cols.iter().zip(weights).enumerate() {
        if c as usize == v && (best.0 == usize::MAX || w < best.1) {
            best = (i, w);
        }
    }
    g.indptr[u] as usize + best.0
}

/// Per-thread buffers: the search scratch, tree positions and branch flows.
struct FlowScratch {
    search: Scratch,
    tree_pos: Vec<u32>,
    branch: Vec<f64>,
}

impl FlowScratch {
    fn new(n: usize) -> Self {
        FlowScratch {
            search: Scratch::new(n),
            tree_pos: vec![0; n],
            branch: Vec::new(),
        }
    }

    /// Flows of one cluster added to `out` (one value per arc in CSR order).
    #[allow(clippy::too_many_arguments)]
    fn cluster(
        &mut self,
        g: Csr,
        sources: &[i32],
        flows: &[f64],
        node_weights: &[f64],
        decay: FlowDecay,
        cutoff: f32,
        min_flow: f64,
        out: &mut [f64],
    ) {
        let FlowScratch {
            search,
            tree_pos,
            branch,
        } = self;
        search.run_tree(g, sources, cutoff);
        let order = search.settled_order();
        let cluster_flow: f64 = flows.iter().sum();
        let mut score_sum = 0.0f64;
        branch.clear();
        branch.resize(order.len(), 0.0);
        for (k, &node) in order.iter().enumerate() {
            let node = node as usize;
            tree_pos[node] = k as u32;
            if search.parent(node) < 0 {
                continue;
            }
            let score = node_weights[node] * decay.weight(search.cost(node) as f64);
            if score > 0.0 {
                branch[k] = score;
                score_sum += score;
            }
        }
        if score_sum > 0.0 && cluster_flow > 0.0 {
            for b in branch.iter_mut() {
                *b = cluster_flow * *b / score_sum;
            }
            for k in (0..order.len()).rev() {
                let node = order[k] as usize;
                let parent = search.parent(node);
                if parent < 0 {
                    continue;
                }
                let parent = parent as usize;
                let flow = branch[k];
                branch[tree_pos[parent] as usize] += flow;
                let from_source = search.parent(parent) < 0;
                if flow > min_flow || (from_source && flow > 0.0) {
                    out[tree_arc(g, parent, node)] += flow;
                }
            }
        }
        search.clear();
    }
}

/// Summed flows per arc of all clusters; `cluster_offsets` slice `sources` and `source_flows`.
#[allow(clippy::too_many_arguments)]
pub fn assign_flows(
    g: Csr,
    sources: &[i32],
    source_flows: &[f64],
    cluster_offsets: &[i64],
    node_weights: &[f64],
    decay: FlowDecay,
    cutoff: f32,
    min_flow: f64,
) -> Vec<f64> {
    let (n, m) = (g.n(), g.indices.len());
    let clusters = cluster_offsets.len().saturating_sub(1);
    let next = AtomicUsize::new(0);
    let workers = rayon::current_num_threads().min(clusters).max(1);
    (0..workers)
        .into_par_iter()
        .map(|_| {
            let mut out = vec![0.0f64; m];
            let mut sc = FlowScratch::new(n);
            loop {
                let c = next.fetch_add(1, Ordering::Relaxed);
                if c >= clusters {
                    break;
                }
                let (a, b) = (cluster_offsets[c] as usize, cluster_offsets[c + 1] as usize);
                if a == b {
                    continue;
                }
                sc.cluster(
                    g,
                    &sources[a..b],
                    &source_flows[a..b],
                    node_weights,
                    decay,
                    cutoff,
                    min_flow,
                    &mut out,
                );
            }
            out
        })
        .reduce_with(|mut a, b| {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
            a
        })
        .unwrap_or_else(|| vec![0.0f64; m])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr<'a>(indptr: &'a [i32], indices: &'a [i32], data: &'a [f32]) -> Csr<'a> {
        Csr {
            indptr,
            indices,
            data,
        }
    }

    #[test]
    fn flow_of_a_path_is_carried_back_to_the_source() {
        // 0 -> 1 -> 2, weights only at 2: the whole flow runs along both arcs.
        let (indptr, indices, data) = ([0, 1, 2, 2], [1, 2], [1.0f32, 1.0]);
        let g = csr(&indptr, &indices, &data);
        let out = assign_flows(
            g,
            &[0],
            &[10.0],
            &[0, 1],
            &[0.0, 0.0, 1.0],
            FlowDecay::Flat,
            f32::INFINITY,
            0.0,
        );
        assert_eq!(out, vec![10.0, 10.0]);
    }

    #[test]
    fn equal_costs_go_to_the_lower_source_position() {
        // Sources 2 and 0 reach node 1 at cost 1 each: node 1 joins source 0.
        let (indptr, indices, data) = ([0, 1, 1, 3], [1, 1, 0], [1.0f32, 1.0, 5.0]);
        let g = csr(&indptr, &indices, &data);
        let out = assign_flows(
            g,
            &[2, 0],
            &[1.0, 1.0],
            &[0, 2],
            &[0.0, 1.0, 0.0],
            FlowDecay::Flat,
            3.0,
            0.0,
        );
        // Arc 0->1 (position 0) carries the cluster flow of 2.
        assert_eq!(out, vec![2.0, 0.0, 0.0]);
    }

    #[test]
    fn min_flow_drops_small_inner_arcs_but_keeps_arcs_from_sources() {
        // 0 -> 1 -> 2; weight 1 at node 1 and 1e-3 at node 2.
        let (indptr, indices, data) = ([0, 1, 2, 2], [1, 2], [1.0f32, 1.0]);
        let g = csr(&indptr, &indices, &data);
        let w = [0.0, 1.0, 1e-3];
        let out = assign_flows(
            g,
            &[0],
            &[1.0],
            &[0, 1],
            &w,
            FlowDecay::Flat,
            f32::INFINITY,
            0.01,
        );
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn empty_clusters_and_zero_weights_give_nothing() {
        let (indptr, indices, data) = ([0, 1, 1], [1], [1.0f32]);
        let g = csr(&indptr, &indices, &data);
        let out = assign_flows(
            g,
            &[0],
            &[5.0],
            &[0, 0, 1],
            &[0.0, 0.0],
            FlowDecay::Flat,
            10.0,
            0.0,
        );
        assert_eq!(out, vec![0.0]);
    }

    #[test]
    fn gaussian_decay_matches_its_formula() {
        let d = FlowDecay::Gaussian {
            mean: 10.0,
            std: 5.0,
        };
        assert!((d.weight(15.0) - (-0.5f64).exp()).abs() < 1e-15);
    }
}
