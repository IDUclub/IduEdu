//! Alternative routes by the penalty method.
//!
//! The shortest route is found first. Then the costs of the arcs it uses are multiplied by `penalty`
//! (both directions of each street) and the search runs again on the penalised costs, so it leaves the
//! streets already taken wherever a detour exists and keeps them where it does not (a bridge, the
//! only way out of a cul-de-sac). Every route found is penalised in turn. A route is kept when its real
//! cost stays within `max_cost_ratio` times the shortest one and at most `max_overlap` of its real cost
//! runs on arcs of a route already kept; the search stops after `k` routes or `4 * k` attempts. The
//! routes are returned cheapest first, each with the largest share of its cost that runs on the
//! streets of another returned route.
//!
//! Unlike Yen's k shortest routes, which on a street graph differ by a sidewalk or a corner, the
//! routes differ by whole streets.

use std::collections::HashSet;

use crate::bidirectional::{arc_weight, route_cost, shortest_pair, BiScratch};
use crate::csr::Csr;

/// Position of the cheapest arc `a -> b` in the arrays of `g`.
fn arc_index(g: Csr, a: usize, b: i32) -> Option<usize> {
    let start = g.indptr[a] as usize;
    let (cols, weights) = g.row(a);
    cols.iter()
        .zip(weights)
        .enumerate()
        .filter(|(_, (&c, _))| c == b)
        .min_by(|x, y| x.1 .1.total_cmp(y.1 .1))
        .map(|(k, _)| start + k)
}

/// The street of an arc, the same for both directions.
#[inline]
fn street(a: i32, b: i32) -> (i32, i32) {
    (a.min(b), a.max(b))
}

/// One alternative: real cost, the largest share of it on the streets of another returned route, and
/// the route.
pub type Alternative = (f32, f32, Vec<i32>);

/// Up to `k` alternative routes from `source` to `target`, cheapest first. `rev` is `g` with every
/// arc reversed, `None` for an undirected graph. Routes beyond the inclusive `cutoff` are not kept.
#[allow(clippy::too_many_arguments)]
pub fn alternatives(
    g: Csr,
    rev: Option<Csr>,
    source: i32,
    target: i32,
    k: usize,
    penalty: f32,
    max_overlap: f32,
    max_cost_ratio: f32,
    cutoff: f32,
    sc: &mut BiScratch,
) -> Vec<Alternative> {
    let mut kept: Vec<Alternative> = Vec::new();
    if k == 0 {
        return kept;
    }
    let mut data = g.data.to_vec();
    let mut rev_data = rev.map(|r| r.data.to_vec());
    let mut kept_streets: Vec<HashSet<(i32, i32)>> = Vec::new();
    let mut seen: HashSet<Vec<i32>> = HashSet::new();
    let mut shortest = f32::INFINITY;
    for _ in 0..4 * k {
        let pg = Csr {
            indptr: g.indptr,
            indices: g.indices,
            data: &data,
        };
        let pr = match (&rev, &rev_data) {
            (Some(r), Some(d)) => Csr {
                indptr: r.indptr,
                indices: r.indices,
                data: d,
            },
            _ => pg,
        };
        let mut route = Vec::new();
        let found = shortest_pair(pg, pr, source, target, f32::INFINITY, sc, Some(&mut route));
        if !found.is_finite() {
            break;
        }
        let cost = route_cost(g, &route);
        if kept.is_empty() {
            if cost > cutoff {
                break;
            }
            shortest = cost;
        }
        if seen.insert(route.clone()) && cost <= cutoff && cost <= shortest * max_cost_ratio {
            let overlap = kept_streets
                .iter()
                .map(|streets| {
                    let shared: f32 = route
                        .windows(2)
                        .filter(|p| streets.contains(&street(p[0], p[1])))
                        .map(|p| arc_weight(g, p[0] as usize, p[1]))
                        .sum();
                    if cost > 0.0 {
                        shared / cost
                    } else {
                        1.0
                    }
                })
                .fold(0f32, f32::max);
            if kept.is_empty() || overlap <= max_overlap {
                kept_streets.push(route.windows(2).map(|p| street(p[0], p[1])).collect());
                kept.push((cost, overlap, route.clone()));
                if kept.len() == k {
                    break;
                }
            }
        }
        if route.len() < 2 {
            break; // source == target: nothing to penalise
        }
        for p in route.windows(2) {
            for (a, b) in [(p[0], p[1]), (p[1], p[0])] {
                if let Some(i) = arc_index(g, a as usize, b) {
                    data[i] *= penalty;
                }
                if let (Some(r), Some(d)) = (&rev, &mut rev_data) {
                    // The arc `a -> b` sits in row `b` of the reversed graph.
                    if let Some(i) = arc_index(*r, b as usize, a) {
                        d[i] *= penalty;
                    }
                }
            }
        }
    }
    // Cheapest first (the shortest route stays first), and overlaps against all other routes.
    kept.sort_by(|a, b| a.0.total_cmp(&b.0));
    let streets: Vec<HashSet<(i32, i32)>> = kept
        .iter()
        .map(|(_, _, route)| route.windows(2).map(|p| street(p[0], p[1])).collect())
        .collect();
    for (i, (cost, overlap, route)) in kept.iter_mut().enumerate() {
        *overlap = 0.0;
        for (j, other) in streets.iter().enumerate() {
            if i == j {
                continue;
            }
            let shared: f32 = route
                .windows(2)
                .filter(|p| other.contains(&street(p[0], p[1])))
                .map(|p| arc_weight(g, p[0] as usize, p[1]))
                .sum();
            let share = if *cost > 0.0 { shared / *cost } else { 1.0 };
            *overlap = overlap.max(share);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr;

    /// Ladder of two parallel streets 0-1-2-3 (cheap) and 4-5-6-7, joined at the ends and in the
    /// middle by rungs 1-5 and 2-6.
    fn ladder() -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let streets = [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (4, 5, 1.2),
            (5, 6, 1.2),
            (6, 7, 1.2),
            (0, 4, 0.5),
            (3, 7, 0.5),
            (1, 5, 0.5),
            (2, 6, 0.5),
        ];
        let (mut rows, mut cols, mut data) = (Vec::new(), Vec::new(), Vec::new());
        for (a, b, w) in streets {
            for (x, y) in [(a, b), (b, a)] {
                rows.push(x as i64);
                cols.push(y as i64);
                data.push(w);
            }
        }
        csr::build(8, &rows, &cols, &data, csr::Reduce::Min)
    }

    #[test]
    fn routes_take_other_streets() {
        let (ip, ix, d) = ladder();
        let g = Csr {
            indptr: &ip,
            indices: &ix,
            data: &d,
        };
        let mut sc = BiScratch::new(8);
        let alts = alternatives(g, None, 0, 3, 3, 2.0, 0.5, 2.0, f32::INFINITY, &mut sc);
        assert_eq!((alts[0].0, alts[0].2.clone()), (3.0, vec![0, 1, 2, 3]));
        // The other street, 0-4-5-6-7-3, is among the alternatives and shares nothing with the first.
        let other = alts.iter().find(|a| a.2 == vec![0, 4, 5, 6, 7, 3]).unwrap();
        assert!((other.0 - 4.6).abs() < 1e-5);
        assert!(alts.windows(2).all(|w| w[0].0 <= w[1].0));
        for (cost, overlap, route) in &alts {
            assert_eq!(*cost, route_cost(g, route));
            assert!((0.0..=1.0).contains(overlap));
        }
        if alts.len() == 2 {
            assert_eq!((alts[0].1, alts[1].1), (0.0, 0.0));
        }
        // The cost ratio stops longer detours, the cutoff even the shortest route.
        let tight = alternatives(g, None, 0, 3, 3, 2.0, 0.5, 1.2, f32::INFINITY, &mut sc);
        assert_eq!(tight.len(), 1);
        assert!(alternatives(g, None, 0, 3, 3, 2.0, 0.5, 2.0, 2.9, &mut sc).is_empty());
        assert_eq!(
            alternatives(g, None, 2, 2, 3, 2.0, 0.5, 2.0, f32::INFINITY, &mut sc),
            vec![(0.0, 0.0, vec![2])]
        );
    }

    #[test]
    fn directed_graph_penalises_the_reversed_arcs_too() {
        let (ip, ix, d) = ladder();
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
        let mut sc = BiScratch::new(8);
        let undirected = alternatives(g, None, 0, 3, 3, 2.0, 0.5, 2.0, f32::INFINITY, &mut sc);
        let directed = alternatives(g, Some(r), 0, 3, 3, 2.0, 0.5, 2.0, f32::INFINITY, &mut sc);
        assert_eq!(undirected, directed);
    }
}
