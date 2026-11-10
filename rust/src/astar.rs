//! A* search for point-to-point queries on graphs whose nodes have planar coordinates.
//!
//! The estimate of the remaining cost from `v` is `scale * |v - target|`, with `scale` the smallest
//! cost per unit of straight distance over the arcs of non-zero cost ([`Heuristic::of`]). No such arc
//! is cheaper than that rate times its straight length, so by the triangle inequality the estimate
//! exceeds the cost of a route only through arcs of zero cost between distinct points (in OSM graphs:
//! a few edges of a centimetre or two whose rounded time is zero), and by at most `slack`, the sum of
//! `scale * length` over those arcs. The search goes on until no queued node can improve the best
//! route by more than that slack, and reopens nodes reached more cheaply later, so it returns shortest
//! routes whatever the edge weights are (length, time or anything else). The scale is shrunk by a
//! millionth so that float32 rounding cannot make the estimate overestimate.
//!
//! The estimate pays off when costs follow straight distance (walking, driving by length). On a graph
//! with fast arcs (metro, motorways) the scale is set by the fastest of them and the estimate prunes
//! little.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::bidirectional::route_cost;
use crate::csr::Csr;

const UNREACHED: f32 = -1.0;

/// Scale of the straight-distance estimate and the most it can overestimate a route by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Heuristic {
    pub scale: f64,
    pub slack: f32,
}

impl Heuristic {
    /// The estimate for `g` with node coordinates `xs`, `ys` (arcs of infinite cost are ignored).
    pub fn of(g: Csr, xs: &[f64], ys: &[f64]) -> Self {
        let length = |u: usize, v: usize| (xs[u] - xs[v]).hypot(ys[u] - ys[v]);
        let mut scale = f64::INFINITY;
        for u in 0..g.n() {
            let (cols, weights) = g.row(u);
            for (&v, &w) in cols.iter().zip(weights) {
                let d = length(u, v as usize);
                if w > 0.0 && w.is_finite() && d > 0.0 {
                    scale = scale.min(w as f64 / d);
                }
            }
        }
        if !scale.is_finite() {
            return Heuristic {
                scale: 0.0,
                slack: 0.0,
            };
        }
        let scale = scale * (1.0 - 1e-6);
        let mut slack = 0f64;
        for u in 0..g.n() {
            let (cols, weights) = g.row(u);
            for (&v, &w) in cols.iter().zip(weights) {
                if w == 0.0 {
                    slack += scale * length(u, v as usize);
                }
            }
        }
        Heuristic {
            scale,
            // Rounded up, so that the float32 slack is not below the exact one.
            slack: (slack * (1.0 + 1e-6)) as f32,
        }
    }
}

/// Buffers of one search, kept between queries and reset through the list of touched nodes.
pub struct AStarScratch {
    dist: Vec<f32>,
    estimate: Vec<f32>,
    parent: Vec<i32>,
    touched: Vec<u32>,
    heap: BinaryHeap<Reverse<u64>>,
}

impl AStarScratch {
    pub fn new(n: usize) -> Self {
        AStarScratch {
            dist: vec![UNREACHED; n],
            estimate: vec![0.0; n],
            parent: vec![-1; n],
            touched: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    fn reset(&mut self) {
        for &v in &self.touched {
            let v = v as usize;
            self.dist[v] = UNREACHED;
            self.parent[v] = -1;
        }
        self.touched.clear();
        self.heap.clear();
    }
}

#[inline]
fn key(priority: f32, node: usize) -> u64 {
    ((priority.to_bits() as u64) << 32) | node as u64
}

/// Shortest route from `source` to `target` with the estimate of `h`; like
/// [`crate::bidirectional::shortest_pair`] it returns the cost (infinity beyond the inclusive
/// `cutoff`) and appends the route, source first, to `route`.
#[allow(clippy::too_many_arguments)]
pub fn shortest_pair(
    g: Csr,
    xs: &[f64],
    ys: &[f64],
    h: Heuristic,
    source: i32,
    target: i32,
    cutoff: f32,
    sc: &mut AStarScratch,
    route: Option<&mut Vec<i32>>,
) -> f32 {
    let (s, t) = (source as usize, target as usize);
    let (tx, ty) = (xs[t], ys[t]);
    let estimate = |v: usize| (h.scale * (xs[v] - tx).hypot(ys[v] - ty)) as f32;
    // A node is not worth queueing when even its estimate less the slack exceeds the cutoff.
    let limit = cutoff + h.slack;
    sc.dist[s] = 0.0;
    sc.estimate[s] = estimate(s);
    sc.touched.push(s as u32);
    sc.heap.push(Reverse(key(sc.estimate[s], s)));
    let mut best = f32::INFINITY;
    while let Some(Reverse(k)) = sc.heap.pop() {
        let priority = f32::from_bits((k >> 32) as u32);
        if priority >= best + h.slack {
            break;
        }
        let u = (k as u32) as usize;
        let du = sc.dist[u];
        // Stale entry: the node was reached more cheaply after it was pushed.
        if priority != du + sc.estimate[u] {
            continue;
        }
        if u == t {
            best = du;
            continue;
        }
        let (cols, weights) = g.row(u);
        for (&v, &w) in cols.iter().zip(weights) {
            let vi = v as usize;
            let cand = du + w;
            if !cand.is_finite() || cand > cutoff || cand >= best {
                continue;
            }
            let dv = sc.dist[vi];
            if dv == UNREACHED {
                sc.touched.push(vi as u32);
                sc.estimate[vi] = estimate(vi);
            } else if cand >= dv {
                continue;
            }
            let priority = cand + sc.estimate[vi];
            if priority > limit {
                // No route through `v` stays within the cutoff. `v` stays touched (its estimate is
                // kept) but unreached.
                if dv == UNREACHED {
                    sc.dist[vi] = f32::INFINITY;
                }
                continue;
            }
            sc.dist[vi] = cand;
            sc.parent[vi] = u as i32;
            sc.heap.push(Reverse(key(priority, vi)));
        }
    }
    let mut cost = f32::INFINITY;
    if best.is_finite() {
        let mut nodes = Vec::new();
        let mut node = target;
        while node >= 0 {
            nodes.push(node);
            node = sc.parent[node as usize];
        }
        nodes.reverse();
        let sum = route_cost(g, &nodes);
        if sum <= cutoff {
            cost = sum;
            if let Some(route) = route {
                route.extend_from_slice(&nodes);
            }
        }
    }
    sc.reset();
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr;
    use crate::dijkstra::{search, Scratch};

    /// Grid of `w * w` points one unit apart; arcs cost their length times a factor of 1 to 3.
    #[allow(clippy::type_complexity)]
    fn grid(w: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>, Vec<f64>, Vec<f64>) {
        let n = w * w;
        let (mut rows, mut cols, mut data) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let (r, c) = (i / w, i % w);
            for (j, ok) in [
                (i + 1, c + 1 < w),
                (i + w, r + 1 < w),
                (i + w + 1, c + 1 < w && r + 1 < w),
            ] {
                if ok {
                    let len = if j == i + w + 1 { 2f32.sqrt() } else { 1.0 };
                    for (a, b) in [(i, j), (j, i)] {
                        rows.push(a as i64);
                        cols.push(b as i64);
                        data.push(len * (1 + (a * 7 + b * 3) % 3) as f32);
                    }
                }
            }
        }
        let (ip, ix, d) = csr::build(n, &rows, &cols, &data, csr::Reduce::Min);
        let xs = (0..n).map(|i| (i % w) as f64).collect();
        let ys = (0..n).map(|i| (i / w) as f64).collect();
        (ip, ix, d, xs, ys)
    }

    #[test]
    fn equals_dijkstra() {
        let w = 25;
        let (ip, ix, d, xs, ys) = grid(w);
        let g = Csr {
            indptr: &ip,
            indices: &ix,
            data: &d,
        };
        let h = Heuristic::of(g, &xs, &ys);
        assert!(h.scale > 0.99 && h.scale <= 1.0 && h.slack == 0.0);
        let mut sc = AStarScratch::new(w * w);
        let mut one: Option<Scratch> = None;
        for cutoff in [f32::INFINITY, 9.0] {
            for s in (0..(w * w) as i32).step_by(53) {
                let tree = search(g, &[s], &[], cutoff, false, &mut one);
                for t in (0..(w * w) as i32).step_by(17) {
                    let mut route = Vec::new();
                    let cost =
                        shortest_pair(g, &xs, &ys, h, s, t, cutoff, &mut sc, Some(&mut route));
                    match tree.nodes.binary_search(&t) {
                        Ok(k) => {
                            assert!((cost - tree.dist[k]).abs() <= 1e-4 * cost.max(1.0));
                            assert_eq!(route.first(), Some(&s));
                            assert_eq!(route.last(), Some(&t));
                        }
                        Err(_) => assert_eq!(cost, f32::INFINITY),
                    }
                    assert!(sc.touched.is_empty());
                }
            }
        }
        // Scale zero: plain Dijkstra, same answers.
        let mut route = Vec::new();
        let cost = shortest_pair(
            g,
            &xs,
            &ys,
            Heuristic {
                scale: 0.0,
                slack: 0.0,
            },
            0,
            624,
            f32::INFINITY,
            &mut sc,
            Some(&mut route),
        );
        let tree = search(g, &[0], &[624], f32::INFINITY, false, &mut one);
        assert_eq!(cost, tree.dist[0]);
    }

    #[test]
    fn zero_cost_arcs_give_slack_and_stay_exact() {
        // Grid with some arcs of zero cost: the estimate overestimates through them, the slack keeps
        // the answers exact.
        let w = 20;
        let (ip, ix, mut d, xs, ys) = grid(w);
        for k in (0..d.len()).step_by(29) {
            d[k] = 0.0;
        }
        let g = Csr {
            indptr: &ip,
            indices: &ix,
            data: &d,
        };
        let h = Heuristic::of(g, &xs, &ys);
        assert!(h.scale > 0.99 && h.slack > 1.0);
        let mut sc = AStarScratch::new(w * w);
        let mut one: Option<Scratch> = None;
        for s in (0..(w * w) as i32).step_by(31) {
            let tree = search(g, &[s], &[], f32::INFINITY, false, &mut one);
            for t in (0..(w * w) as i32).step_by(7) {
                let cost = shortest_pair(g, &xs, &ys, h, s, t, f32::INFINITY, &mut sc, None);
                let k = tree.nodes.binary_search(&t).unwrap();
                assert!(
                    (cost - tree.dist[k]).abs() <= 1e-4 * cost.max(1.0),
                    "{s}->{t}"
                );
            }
        }
        // Only arcs of zero cost: no estimate.
        let (ip, ix, d) = csr::build(2, &[0], &[1], &[0.0], csr::Reduce::Min);
        let g = Csr {
            indptr: &ip,
            indices: &ix,
            data: &d,
        };
        assert_eq!(
            Heuristic::of(g, &[0.0, 5.0], &[0.0, 0.0]),
            Heuristic {
                scale: 0.0,
                slack: 0.0
            }
        );
    }
}
