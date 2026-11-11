//! Bidirectional Dijkstra for point-to-point queries.
//!
//! One search grows from the source over the forward graph and one from the target over the reversed
//! graph; each step settles a node on the side whose next cost is smaller. Every arc scanned from a
//! settled node that reaches a node already seen by the other side gives a candidate route, and the
//! search stops once the two next costs add up to at least the best candidate (Goldberg and Harrelson's
//! rule). Each side settles only the nodes within about half of the route cost, so a query settles far
//! fewer nodes than a one-sided search that stops at the target.
//!
//! The cost of the returned route is summed again from the source in the order of the route, so it
//! equals the float32 sum a one-sided search computes along the same route, and `cutoff` is applied to
//! that sum.
//!
//! [`shortest_pair_avoiding`] skips some nodes and some arcs out of the source: the spur searches of
//! Yen's k shortest paths ([`crate::yen`]).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::csr::Csr;

const UNREACHED: f32 = -1.0;

/// State of one side of the search.
struct Side {
    dist: Vec<f32>,
    settled: Vec<bool>,
    parent: Vec<i32>,
    touched: Vec<u32>,
    heap: BinaryHeap<Reverse<u64>>,
}

impl Side {
    fn new(n: usize) -> Self {
        Side {
            dist: vec![UNREACHED; n],
            settled: vec![false; n],
            parent: vec![-1; n],
            touched: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    fn start(&mut self, node: i32) {
        let u = node as usize;
        self.dist[u] = 0.0;
        self.touched.push(u as u32);
        self.heap.push(Reverse(key(0.0, node)));
    }

    /// Cost of the next node to settle, dropping heap entries of nodes already settled.
    fn top(&mut self) -> Option<f32> {
        while let Some(&Reverse(k)) = self.heap.peek() {
            if self.settled[(k as u32) as usize] {
                self.heap.pop();
            } else {
                return Some(f32::from_bits((k >> 32) as u32));
            }
        }
        None
    }

    fn reset(&mut self) {
        for &v in &self.touched {
            let v = v as usize;
            self.dist[v] = UNREACHED;
            self.settled[v] = false;
            self.parent[v] = -1;
        }
        self.touched.clear();
        self.heap.clear();
    }
}

#[inline]
fn key(dist: f32, node: i32) -> u64 {
    ((dist.to_bits() as u64) << 32) | node as u32 as u64
}

/// Buffers of both sides, kept between queries and reset through the lists of touched nodes.
pub struct BiScratch {
    forward: Side,
    backward: Side,
    /// Avoided nodes, allocated on the first search that avoids any.
    mask: Vec<bool>,
}

impl BiScratch {
    pub fn new(n: usize) -> Self {
        BiScratch {
            forward: Side::new(n),
            backward: Side::new(n),
            mask: Vec::new(),
        }
    }
}

/// What a search must not use: the nodes flagged in `nodes`, and the arcs from the source to `succ`.
struct Avoid<'a> {
    nodes: &'a [bool],
    succ: &'a [i32],
}

/// Whether the arc `a -> b` (in the forward orientation) may be used by a search from `source`.
#[inline(always)]
fn allowed(avoid: &Avoid, source: i32, a: i32, b: i32) -> bool {
    !avoid.nodes[a as usize]
        && !avoid.nodes[b as usize]
        && !(a == source && avoid.succ.contains(&b))
}

/// Settle the next node of `this` and offer the routes through its arcs. `best` holds the cost of the
/// best route found so far and the arc `(from, to)` where its two halves meet, oriented like the
/// forward graph.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step<const AVOID: bool>(
    g: Csr,
    this: &mut Side,
    other: &Side,
    cutoff: f32,
    forward: bool,
    best: &mut (f32, i32, i32),
    avoid: &Avoid,
    source: i32,
) {
    let Some(Reverse(k)) = this.heap.pop() else {
        return;
    };
    let u = (k as u32) as usize;
    this.settled[u] = true;
    let du = this.dist[u];
    let (cols, weights) = g.row(u);
    for (&v, &w) in cols.iter().zip(weights) {
        let (from, to) = if forward {
            (u as i32, v)
        } else {
            (v, u as i32)
        };
        if AVOID && !allowed(avoid, source, from, to) {
            continue;
        }
        let vi = v as usize;
        let cand = du + w;
        if !cand.is_finite() || cand > cutoff {
            continue;
        }
        let dv_other = other.dist[vi];
        if dv_other != UNREACHED {
            let total = cand + dv_other;
            if total < best.0 {
                *best = (total, from, to);
            }
        }
        if this.settled[vi] {
            continue;
        }
        let dv = this.dist[vi];
        if dv == UNREACHED || cand < dv {
            if dv == UNREACHED {
                this.touched.push(vi as u32);
            }
            this.dist[vi] = cand;
            this.parent[vi] = u as i32;
            this.heap.push(Reverse(key(cand, v)));
        }
    }
}

/// Cheapest arc `a -> b` of `g`, or infinity when there is none.
pub fn arc_weight(g: Csr, a: usize, b: i32) -> f32 {
    let (cols, weights) = g.row(a);
    cols.iter()
        .zip(weights)
        .filter(|(&c, _)| c == b)
        .map(|(_, &w)| w)
        .fold(f32::INFINITY, f32::min)
}

/// Cost of a route summed from its first node, as a one-sided search sums it.
pub fn route_cost(g: Csr, route: &[i32]) -> f32 {
    let mut sum = 0f32;
    for pair in route.windows(2) {
        sum += arc_weight(g, pair[0] as usize, pair[1]);
    }
    sum
}

/// Shortest route from `source` to `target` over `g` (`rev` is `g` with every arc reversed; the same
/// graph for an undirected one). Returns its cost, or infinity when the target is not reached within
/// the inclusive `cutoff`. With `route`, the route's node positions are appended to it, source first
/// (nothing is appended when the target is not reached).
pub fn shortest_pair(
    g: Csr,
    rev: Csr,
    source: i32,
    target: i32,
    cutoff: f32,
    sc: &mut BiScratch,
    route: Option<&mut Vec<i32>>,
) -> f32 {
    let none = Avoid {
        nodes: &[],
        succ: &[],
    };
    let (fw, bw) = (&mut sc.forward, &mut sc.backward);
    run::<false>(g, rev, source, target, cutoff, fw, bw, route, &none)
}

/// [`shortest_pair`] that uses neither the nodes `avoid_nodes` nor the arcs from `source` to
/// `avoid_succ` (the source and the target must not be among the avoided nodes).
#[allow(clippy::too_many_arguments)]
pub fn shortest_pair_avoiding(
    g: Csr,
    rev: Csr,
    source: i32,
    target: i32,
    cutoff: f32,
    sc: &mut BiScratch,
    route: Option<&mut Vec<i32>>,
    avoid_nodes: &[i32],
    avoid_succ: &[i32],
) -> f32 {
    let BiScratch {
        forward,
        backward,
        mask,
    } = sc;
    if mask.is_empty() {
        mask.resize(g.n(), false);
    }
    for &v in avoid_nodes {
        mask[v as usize] = true;
    }
    let avoid = Avoid {
        nodes: mask,
        succ: avoid_succ,
    };
    let cost = run::<true>(
        g, rev, source, target, cutoff, forward, backward, route, &avoid,
    );
    for &v in avoid_nodes {
        mask[v as usize] = false;
    }
    cost
}

#[allow(clippy::too_many_arguments)]
fn run<const AVOID: bool>(
    g: Csr,
    rev: Csr,
    source: i32,
    target: i32,
    cutoff: f32,
    fw: &mut Side,
    bw: &mut Side,
    route: Option<&mut Vec<i32>>,
    avoid: &Avoid,
) -> f32 {
    if source == target {
        if let Some(route) = route {
            route.push(source);
        }
        return 0.0;
    }
    fw.start(source);
    bw.start(target);
    let mut best = (f32::INFINITY, -1i32, -1i32);
    while let (Some(tf), Some(tb)) = (fw.top(), bw.top()) {
        if tf + tb >= best.0 || tf + tb > cutoff {
            break;
        }
        if tf <= tb {
            step::<AVOID>(g, fw, bw, cutoff, true, &mut best, avoid, source);
        } else {
            step::<AVOID>(rev, bw, fw, cutoff, false, &mut best, avoid, source);
        }
    }
    let mut cost = f32::INFINITY;
    if best.1 >= 0 {
        // Source to the meeting arc through the forward parents, then on to the target through the
        // backward ones.
        let mut nodes = Vec::new();
        let mut node = best.1;
        while node >= 0 {
            nodes.push(node);
            node = fw.parent[node as usize];
        }
        nodes.reverse();
        let mut node = best.2;
        while node >= 0 {
            nodes.push(node);
            node = bw.parent[node as usize];
        }
        let sum = route_cost(g, &nodes);
        if sum <= cutoff {
            cost = sum;
            if let Some(route) = route {
                route.extend_from_slice(&nodes);
            }
        }
    }
    fw.reset();
    bw.reset();
    cost
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::csr;
    use crate::dijkstra::{search, Scratch};

    /// Ring of `n` nodes with `m` random chords, some of zero cost and some impassable.
    pub(crate) fn random_graph(
        n: usize,
        m: usize,
        seed: u64,
        two_way: bool,
    ) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let (mut rows, mut cols, mut data) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            rows.push(i as i64);
            cols.push(((i + 1) % n) as i64);
            data.push((next() % 9) as f32 * 0.37 + 0.1);
        }
        for _ in 0..m {
            let (a, b) = (next() % n, next() % n);
            rows.push(a as i64);
            cols.push(b as i64);
            data.push(match next() % 20 {
                0 => 0.0,
                1 => f32::INFINITY,
                r => r as f32 * 0.53,
            });
        }
        if two_way {
            let (r, c) = (rows.clone(), cols.clone());
            rows.extend(c);
            cols.extend(r);
            let d = data.clone();
            data.extend(d);
        }
        csr::build(n, &rows, &cols, &data, csr::Reduce::Min)
    }

    #[test]
    fn equals_one_sided_search() {
        for (seed, two_way) in [(1, false), (2, false), (3, true), (4, true)] {
            let n = 300;
            let (ip, ix, d) = random_graph(n, 900, seed, two_way);
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
            let mut bi = BiScratch::new(n);
            let mut one: Option<Scratch> = None;
            for cutoff in [f32::INFINITY, 3.0, 0.0] {
                for s in (0..n as i32).step_by(37) {
                    let tree = search(g, &[s], &[], cutoff, true, &mut one);
                    for t in (0..n as i32).step_by(11) {
                        let mut route = Vec::new();
                        let cost = shortest_pair(g, r, s, t, cutoff, &mut bi, Some(&mut route));
                        match tree.nodes.binary_search(&t) {
                            Ok(k) => {
                                let expected = tree.dist[k];
                                assert!(
                                    (cost - expected).abs() <= 1e-4 * expected.max(1.0),
                                    "{s}->{t}: {cost} vs {expected}"
                                );
                                assert_eq!(route.first(), Some(&s));
                                assert_eq!(route.last(), Some(&t));
                                assert_eq!(route_cost(g, &route), cost);
                            }
                            Err(_) => {
                                assert_eq!(cost, f32::INFINITY, "{s}->{t} should be unreachable");
                                assert!(route.is_empty());
                            }
                        }
                        assert!(bi.forward.touched.is_empty() && bi.backward.touched.is_empty());
                    }
                }
            }
        }
    }

    #[test]
    fn same_node_and_no_route() {
        let (ip, ix, d) = csr::build(3, &[0], &[1], &[2.0], csr::Reduce::Min);
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
        let mut sc = BiScratch::new(3);
        let mut route = Vec::new();
        assert_eq!(
            shortest_pair(g, r, 2, 2, 0.0, &mut sc, Some(&mut route)),
            0.0
        );
        assert_eq!(route, vec![2]);
        assert_eq!(
            shortest_pair(g, r, 1, 0, f32::INFINITY, &mut sc, None),
            f32::INFINITY
        );
        assert_eq!(shortest_pair(g, r, 0, 1, 1.5, &mut sc, None), f32::INFINITY);
        assert_eq!(shortest_pair(g, r, 0, 1, 2.0, &mut sc, None), 2.0);
    }

    #[test]
    fn avoids_nodes_and_arcs() {
        // 0 -> 1 -> 3 (cost 2), 0 -> 2 -> 3 (cost 4), 0 -> 3 (cost 10).
        let (ip, ix, d) = csr::build(
            4,
            &[0, 1, 0, 2, 0],
            &[1, 3, 2, 3, 3],
            &[1.0, 1.0, 2.0, 2.0, 10.0],
            csr::Reduce::Min,
        );
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
        let mut sc = BiScratch::new(4);
        let inf = f32::INFINITY;
        let mut route = Vec::new();
        let cost = shortest_pair_avoiding(g, r, 0, 3, inf, &mut sc, Some(&mut route), &[], &[]);
        assert_eq!((cost, route.clone()), (2.0, vec![0, 1, 3]));
        route.clear();
        let cost = shortest_pair_avoiding(g, r, 0, 3, inf, &mut sc, Some(&mut route), &[1], &[]);
        assert_eq!((cost, route.clone()), (4.0, vec![0, 2, 3]));
        route.clear();
        let cost = shortest_pair_avoiding(g, r, 0, 3, inf, &mut sc, Some(&mut route), &[1], &[2]);
        assert_eq!((cost, route.clone()), (10.0, vec![0, 3]));
        // The mask is clear again.
        assert!(sc.mask.iter().all(|&m| !m));
        assert_eq!(shortest_pair(g, r, 0, 3, inf, &mut sc, None), 2.0);
    }
}
