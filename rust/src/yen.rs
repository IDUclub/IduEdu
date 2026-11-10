//! Yen's k shortest loopless routes between two nodes.
//!
//! The first route is the shortest one. Every next route leaves an earlier one at some node (the spur
//! node): it follows the earlier route up to there (the root), then takes the shortest way to the
//! target that touches no node of the root and does not continue the way any route found so far
//! continues from the same root. Those candidates are kept in a heap, and the cheapest one not yet
//! taken is the next route. Lawler's rule spurs only from the node where a route left its parent
//! onwards, and the spur searches of one route run in parallel, each a bidirectional Dijkstra.
//!
//! Costs are summed from the source along each route ([`route_cost`]); a route is kept only within
//! the inclusive `cutoff`.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Mutex;

use rayon::prelude::*;

use crate::bidirectional::{
    arc_weight, route_cost, shortest_pair, shortest_pair_avoiding, BiScratch,
};
use crate::csr::Csr;

/// Run `f` with a buffer from `pool`, created for `n` nodes when the pool is empty.
fn with_pooled<T>(
    pool: &Mutex<Vec<BiScratch>>,
    n: usize,
    f: impl FnOnce(&mut BiScratch) -> T,
) -> T {
    let popped = pool.lock().unwrap().pop();
    let mut sc = popped.unwrap_or_else(|| BiScratch::new(n));
    let out = f(&mut sc);
    pool.lock().unwrap().push(sc);
    out
}

/// Up to `k` shortest loopless routes from `source` to `target`, cheapest first, with their costs.
/// Equal costs are ordered by the node sequence, so the result does not depend on thread timing.
pub fn k_shortest(
    g: Csr,
    rev: Csr,
    source: i32,
    target: i32,
    k: usize,
    cutoff: f32,
    pool: &Mutex<Vec<BiScratch>>,
) -> Vec<(f32, Vec<i32>)> {
    let n = g.n();
    let mut found: Vec<(f32, Vec<i32>, usize)> = Vec::new();
    if k == 0 {
        return Vec::new();
    }
    let mut first = Vec::new();
    let cost = with_pooled(pool, n, |sc| {
        shortest_pair(g, rev, source, target, cutoff, sc, Some(&mut first))
    });
    if !cost.is_finite() {
        return Vec::new();
    }
    found.push((cost, first, 0));
    let mut candidates: BinaryHeap<Reverse<(u32, Vec<i32>, usize)>> = BinaryHeap::new();
    let mut seen: HashSet<Vec<i32>> = HashSet::new();
    seen.insert(found[0].1.clone());
    while found.len() < k {
        let (_, last, deviation) = found.last().unwrap().clone();
        // Cost of the root up to every spur node.
        let mut prefix = vec![0f32; last.len()];
        for j in 1..last.len() {
            prefix[j] = prefix[j - 1] + arc_weight(g, last[j - 1] as usize, last[j]);
        }
        let spurs: Vec<Option<Vec<i32>>> = (deviation..last.len().saturating_sub(1))
            .into_par_iter()
            .map(|j| {
                let root = &last[..=j];
                let mut succ: Vec<i32> = found
                    .iter()
                    .filter(|(_, route, _)| route.len() > j + 1 && &route[..=j] == root)
                    .map(|(_, route, _)| route[j + 1])
                    .collect();
                succ.sort_unstable();
                succ.dedup();
                let budget = cutoff - prefix[j];
                let mut spur = Vec::new();
                let cost = with_pooled(pool, n, |sc| {
                    shortest_pair_avoiding(
                        g,
                        rev,
                        last[j],
                        target,
                        budget,
                        sc,
                        Some(&mut spur),
                        &last[..j],
                        &succ,
                    )
                });
                cost.is_finite().then(|| {
                    let mut route = last[..j].to_vec();
                    route.extend_from_slice(&spur);
                    route
                })
            })
            .collect();
        for (offset, route) in spurs.into_iter().enumerate() {
            let Some(route) = route else { continue };
            let cost = route_cost(g, &route);
            if cost <= cutoff && seen.insert(route.clone()) {
                candidates.push(Reverse((cost.to_bits(), route, deviation + offset)));
            }
        }
        let Some(Reverse((bits, route, deviation))) = candidates.pop() else {
            break;
        };
        found.push((f32::from_bits(bits), route, deviation));
    }
    found
        .into_iter()
        .map(|(cost, route, _)| (cost, route))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidirectional::tests::random_graph;
    use crate::csr;

    /// Every loopless route from `s` to `t` by depth-first search, with its cost.
    fn all_routes(g: Csr, s: i32, t: i32) -> Vec<(f32, Vec<i32>)> {
        fn walk(
            g: Csr,
            t: i32,
            route: &mut Vec<i32>,
            on: &mut Vec<bool>,
            out: &mut Vec<(f32, Vec<i32>)>,
        ) {
            let u = *route.last().unwrap();
            if u == t {
                let cost = route_cost(g, route);
                if cost.is_finite() {
                    out.push((cost, route.clone()));
                }
                return;
            }
            let (cols, _) = g.row(u as usize);
            for &v in cols {
                if !on[v as usize] {
                    on[v as usize] = true;
                    route.push(v);
                    walk(g, t, route, on, out);
                    route.pop();
                    on[v as usize] = false;
                }
            }
        }
        let mut on = vec![false; g.n()];
        on[s as usize] = true;
        let mut out = Vec::new();
        walk(g, t, &mut vec![s], &mut on, &mut out);
        out.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        out
    }

    #[test]
    fn equals_brute_force() {
        for (seed, two_way) in [(5, false), (6, true), (7, false)] {
            let n = 12;
            let (ip, ix, d) = random_graph(n, 14, seed, two_way);
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
            let pool = Mutex::new(Vec::new());
            for (s, t) in [(0, 7), (3, 11), (9, 2)] {
                let expected = all_routes(g, s, t);
                for cutoff in [f32::INFINITY, 6.0] {
                    let got = k_shortest(g, r, s, t, 8, cutoff, &pool);
                    let within: Vec<_> = expected.iter().filter(|e| e.0 <= cutoff).collect();
                    assert_eq!(got.len(), within.len().min(8), "{s}->{t}");
                    for (a, b) in got.iter().zip(&within) {
                        // Costs match in order; routes of equal cost may come in another order.
                        assert!((a.0 - b.0).abs() <= 1e-4 * b.0.max(1.0));
                        assert_eq!(route_cost(g, &a.1), a.0);
                        assert_eq!(a.1.first(), Some(&s));
                        assert_eq!(a.1.last(), Some(&t));
                        let mut nodes = a.1.clone();
                        nodes.sort_unstable();
                        nodes.dedup();
                        assert_eq!(nodes.len(), a.1.len(), "route with a loop");
                    }
                    let distinct: HashSet<_> = got.iter().map(|r| r.1.clone()).collect();
                    assert_eq!(distinct.len(), got.len());
                }
            }
        }
    }

    #[test]
    fn same_node_and_unreachable() {
        let (ip, ix, d) = csr::build(3, &[0], &[1], &[1.0], csr::Reduce::Min);
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
        let pool = Mutex::new(Vec::new());
        assert_eq!(
            k_shortest(g, r, 2, 2, 3, f32::INFINITY, &pool),
            vec![(0.0, vec![2])]
        );
        assert!(k_shortest(g, r, 1, 0, 3, f32::INFINITY, &pool).is_empty());
        assert!(k_shortest(g, r, 0, 1, 0, f32::INFINITY, &pool).is_empty());
    }
}
