//! ALT: A* with landmarks and the triangle inequality.
//!
//! A few nodes are chosen as landmarks and the costs from every landmark to every node, and from every
//! node back to every landmark, are computed once. For a landmark `L` the triangle inequality gives
//! two lower bounds of the cost from `v` to the target `t`: `d(L, t) - d(L, v)` and `d(v, L) - d(t, L)`.
//! The estimate is the largest of them over the landmarks. It follows the network, not the straight
//! line, so it stays sharp on graphs with fast edges (metro) where the straight-distance estimate of
//! [`crate::astar`] prunes nothing, and it needs no node coordinates.
//!
//! Landmarks are picked one by one as the node farthest from those already picked (the first one is
//! the node farthest from node 0), which spreads them along the edges of the graph. An infinite
//! difference means that the target cannot be reached from `v` at all, and the search skips `v`.
//!
//! The tables hold float32 sums, so a bound can exceed the real cost by rounding; the search is given
//! a slack of a hundred-thousandth of the largest table cost and stays exact up to float32 rounding.

use rayon::prelude::*;

use crate::csr::Csr;
use crate::dijkstra::Scratch;

/// Costs between the landmarks and every node, node-major: `from[v * count + i]` is the cost from
/// landmark `i` to `v` and `to[v * count + i]` the cost from `v` to landmark `i` (infinity when not
/// reached). `to` is empty for an undirected graph, where it equals `from`.
pub struct Landmarks {
    pub nodes: Vec<i32>,
    count: usize,
    from: Vec<f32>,
    to: Vec<f32>,
    pub slack: f32,
}

/// Costs from `source` to every node (infinity when not reached).
fn costs_from(g: Csr, source: i32, sc: &mut Scratch) -> Vec<f32> {
    let mut out = vec![f32::INFINITY; g.n()];
    sc.walk(g, source, f32::INFINITY, |v, d| {
        out[v] = d;
        true
    });
    out
}

/// Index of the largest finite value, ignoring `taken` entries.
fn farthest(values: &[f32], taken: &[i32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (v, &d) in values.iter().enumerate() {
        if d.is_finite() && best.is_none_or(|(_, b)| d > b) && !taken.contains(&(v as i32)) {
            best = Some((v, d));
        }
    }
    best.map(|(v, _)| v)
}

impl Landmarks {
    /// Choose up to `count` landmarks of `g` and compute their tables. `rev` is `g` with every arc
    /// reversed, `None` for an undirected graph. Runs the searches of the backward tables in parallel.
    pub fn build(g: Csr, rev: Option<Csr>, count: usize) -> Self {
        let n = g.n();
        let mut nodes: Vec<i32> = Vec::new();
        let mut rows: Vec<Vec<f32>> = Vec::new();
        if n > 0 && count > 0 {
            let mut sc = Scratch::new(n);
            let start = costs_from(g, 0, &mut sc);
            let mut next = farthest(&start, &[]).unwrap_or(0);
            let mut nearest = vec![f32::INFINITY; n];
            while nodes.len() < count {
                nodes.push(next as i32);
                let row = costs_from(g, next as i32, &mut sc);
                for (m, &d) in nearest.iter_mut().zip(&row) {
                    *m = m.min(d);
                }
                rows.push(row);
                match farthest(&nearest, &nodes) {
                    // A node at cost zero from a landmark adds nothing.
                    Some(v) if nearest[v] > 0.0 => next = v,
                    _ => break,
                }
            }
        }
        let count = nodes.len();
        let interleave = |rows: &[Vec<f32>]| {
            let mut out = vec![f32::INFINITY; n * count];
            for (i, row) in rows.iter().enumerate() {
                for (v, &d) in row.iter().enumerate() {
                    out[v * count + i] = d;
                }
            }
            out
        };
        let from = interleave(&rows);
        let to = match rev {
            Some(r) => {
                let back: Vec<Vec<f32>> = nodes
                    .par_iter()
                    .map(|&l| costs_from(r, l, &mut Scratch::new(n)))
                    .collect();
                interleave(&back)
            }
            None => Vec::new(),
        };
        let largest = from
            .iter()
            .chain(&to)
            .filter(|d| d.is_finite())
            .fold(0f32, |a, &b| a.max(b));
        Landmarks {
            nodes,
            count,
            from,
            to,
            slack: largest * 1e-5,
        }
    }

    /// Number of nodes the tables were built for.
    pub fn n(&self) -> usize {
        self.from.len().checked_div(self.count).unwrap_or(0)
    }

    /// Bytes held by the tables.
    pub fn nbytes(&self) -> usize {
        (self.from.len() + self.to.len()) * std::mem::size_of::<f32>()
    }

    /// Lower bound of the cost from `v` to `t` (infinity: `t` cannot be reached from `v`).
    #[inline]
    pub fn estimate(&self, v: usize, t: usize) -> f32 {
        let c = self.count;
        let from_v = &self.from[v * c..(v + 1) * c];
        let from_t = &self.from[t * c..(t + 1) * c];
        let (to_v, to_t) = if self.to.is_empty() {
            (from_v, from_t)
        } else {
            (&self.to[v * c..(v + 1) * c], &self.to[t * c..(t + 1) * c])
        };
        let mut h = 0f32;
        for i in 0..c {
            // `max` drops the NaN of infinity minus infinity.
            h = h.max(from_t[i] - from_v[i]).max(to_v[i] - to_t[i]);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astar::{search, AStarScratch};
    use crate::bidirectional::tests::random_graph;
    use crate::csr;
    use crate::dijkstra::search as dijkstra;

    #[test]
    fn estimates_are_lower_bounds_and_search_is_exact() {
        for (seed, two_way) in [(11, false), (12, true)] {
            let n = 400;
            let (ip, ix, d) = random_graph(n, 1200, seed, two_way);
            let g = Csr {
                indptr: &ip,
                indices: &ix,
                data: &d,
            };
            let (rip, rix, rd) = csr::transpose(g);
            let r = Csr {
                indptr: &rip,
                indices: &rix,
                data: &rd,
            };
            let lm = Landmarks::build(g, if two_way { None } else { Some(r) }, 6);
            assert_eq!(lm.nodes.len(), 6);
            assert_eq!(lm.n(), n);
            let mut one = None;
            let mut sc = AStarScratch::new(n);
            for s in (0..n as i32).step_by(23) {
                let tree = dijkstra(g, &[s], &[], f32::INFINITY, false, &mut one);
                let mut exact = vec![f32::INFINITY; n];
                for (&v, &c) in tree.nodes.iter().zip(&tree.dist) {
                    exact[v as usize] = c;
                }
                for t in (0..n as i32).step_by(7) {
                    // The estimate from the source is a lower bound of the real cost.
                    let h = lm.estimate(s as usize, t as usize);
                    assert!(
                        h <= exact[t as usize] + lm.slack,
                        "{s}->{t}: {h} > {}",
                        exact[t as usize]
                    );
                    let est = |v: usize| lm.estimate(v, t as usize);
                    for cutoff in [f32::INFINITY, 4.0] {
                        let cost = search(g, est, lm.slack, s, t, cutoff, &mut sc, None);
                        let expected = if exact[t as usize] <= cutoff {
                            exact[t as usize]
                        } else {
                            f32::INFINITY
                        };
                        if expected.is_finite() {
                            assert!((cost - expected).abs() <= 1e-4 * expected.max(1.0));
                        } else {
                            assert_eq!(cost, f32::INFINITY, "{s}->{t}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn small_and_disconnected_graphs() {
        // Two components: 0 - 1 and 2 - 3.
        let (ip, ix, d) = csr::build(4, &[0, 1, 2, 3], &[1, 0, 3, 2], &[1.0; 4], csr::Reduce::Min);
        let g = Csr {
            indptr: &ip,
            indices: &ix,
            data: &d,
        };
        let lm = Landmarks::build(g, None, 8);
        assert!(!lm.nodes.is_empty() && lm.nodes.len() <= 2);
        // A landmark in the component of 0 proves that 2 cannot be reached from 0.
        assert_eq!(lm.estimate(0, 2), f32::INFINITY);
        let mut sc = AStarScratch::new(4);
        let est = |v: usize| lm.estimate(v, 2);
        assert_eq!(
            search(g, est, lm.slack, 0, 2, f32::INFINITY, &mut sc, None),
            f32::INFINITY
        );
        let empty = Landmarks::build(g, None, 0);
        assert!(empty.nodes.is_empty());
        assert_eq!(empty.estimate(0, 1), 0.0);
    }
}
