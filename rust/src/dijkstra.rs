//! Dijkstra searches over a [`Csr`] graph.
//!
//! A search starts from one or more sources, optionally stops once every target is settled, never
//! settles a node whose cost exceeds the inclusive `cutoff`, and can record predecessors. The result is
//! [`Compact`]: settled node positions in ascending order with their float32 costs, parents (only with
//! predecessors) and the source each node was reached from. Equal costs are resolved towards the lower
//! source position, so multi-source searches assign every node to its nearest source deterministically.
//!
//! Without predecessors and with targets, only the settled targets are returned. With predecessors the
//! whole settled tree is returned, so routes to the targets can be rebuilt from it.
//!
//! Implementation: a binary heap of packed integer keys (`dist_bits << 32 | node` for one source, with
//! the owner in between for several), and per-thread scratch buffers that are reset in time
//! proportional to the nodes a search touched, not to the graph size.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::csr::Csr;

const UNREACHED: f32 = -1.0;

/// Compact search result.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct Compact {
    pub nodes: Vec<i32>,
    pub dist: Vec<f32>,
    pub parents: Vec<i32>,
    pub owners: Vec<i32>,
}

/// Heap key ordered like the `(dist, owner, node)` triple. Non-negative floats order like their bits.
trait Key: Ord + Copy {
    fn make(dist: f32, owner: i32, node: i32) -> Self;
    fn owner(self, single: i32) -> i32;
    fn node(self) -> i32;
}

/// One source: the owner is the same for every entry and is not stored.
impl Key for u64 {
    #[inline]
    fn make(dist: f32, _owner: i32, node: i32) -> Self {
        ((dist.to_bits() as u64) << 32) | node as u32 as u64
    }
    #[inline]
    fn owner(self, single: i32) -> i32 {
        single
    }
    #[inline]
    fn node(self) -> i32 {
        self as u32 as i32
    }
}

/// Several sources: `dist_bits << 64 | owner << 32 | node`.
impl Key for u128 {
    #[inline]
    fn make(dist: f32, owner: i32, node: i32) -> Self {
        ((dist.to_bits() as u128) << 64) | ((owner as u32 as u128) << 32) | node as u32 as u128
    }
    #[inline]
    fn owner(self, _: i32) -> i32 {
        (self >> 32) as u32 as i32
    }
    #[inline]
    fn node(self) -> i32 {
        self as u32 as i32
    }
}

/// Buffers of one search, kept between searches and reset through the list of touched nodes.
pub struct Scratch {
    seen: Vec<f32>,
    owners: Vec<i32>,
    settled: Vec<bool>,
    parents: Vec<i32>,
    target_mask: Vec<bool>,
    touched: Vec<u32>,
    settled_list: Vec<u32>,
    heap_single: BinaryHeap<Reverse<u64>>,
    heap_multi: BinaryHeap<Reverse<u128>>,
}

impl Scratch {
    pub fn new(n: usize) -> Self {
        Scratch {
            seen: vec![UNREACHED; n],
            owners: vec![-1; n],
            settled: vec![false; n],
            parents: vec![-1; n],
            target_mask: vec![false; n],
            touched: Vec::new(),
            settled_list: Vec::new(),
            heap_single: BinaryHeap::new(),
            heap_multi: BinaryHeap::new(),
        }
    }

    /// Undo the last search.
    fn reset(&mut self, targets: &[i32]) {
        for &v in &self.touched {
            let v = v as usize;
            self.seen[v] = UNREACHED;
            self.owners[v] = -1;
            self.settled[v] = false;
            self.parents[v] = -1;
        }
        for &t in targets {
            self.target_mask[t as usize] = false;
        }
        self.touched.clear();
        self.settled_list.clear();
    }

    /// Run one search; its state stays in the buffers until `reset`.
    fn run(&mut self, g: Csr, sources: &[i32], targets: &[i32], cutoff: f32, store_pred: bool) {
        if sources.len() == 1 {
            let mut heap = std::mem::take(&mut self.heap_single);
            self.run_with(&mut heap, g, sources, targets, cutoff, store_pred);
            self.heap_single = heap;
        } else {
            let mut heap = std::mem::take(&mut self.heap_multi);
            self.run_with(&mut heap, g, sources, targets, cutoff, store_pred);
            self.heap_multi = heap;
        }
    }

    #[inline(always)]
    fn run_with<K: Key>(
        &mut self,
        heap: &mut BinaryHeap<Reverse<K>>,
        g: Csr,
        sources: &[i32],
        targets: &[i32],
        cutoff: f32,
        store_pred: bool,
    ) {
        heap.clear();
        let has_targets = !targets.is_empty();
        let mut targets_left = 0usize;
        for &t in targets {
            let t = t as usize;
            if !self.target_mask[t] {
                self.target_mask[t] = true;
                targets_left += 1;
            }
        }
        let single = if sources.len() == 1 { sources[0] } else { -1 };
        for &s in sources {
            let su = s as usize;
            if self.seen[su] == UNREACHED {
                self.seen[su] = 0.0;
                self.owners[su] = s;
                self.touched.push(su as u32);
                heap.push(Reverse(K::make(0.0, s, s)));
            }
        }
        let seen = &mut self.seen[..];
        let owners = &mut self.owners[..];
        let settled = &mut self.settled[..];
        let parents = &mut self.parents[..];
        let target_mask = &self.target_mask[..];
        while let Some(Reverse(key)) = heap.pop() {
            let node = key.node();
            let u = node as usize;
            if settled[u] {
                continue;
            }
            settled[u] = true;
            self.settled_list.push(u as u32);
            if has_targets && target_mask[u] {
                targets_left -= 1;
                if targets_left == 0 {
                    break;
                }
            }
            // The first pop of a node carries its minimum key, the value kept in `seen`.
            let distance = seen[u];
            let owner = key.owner(single);
            let (cols, weights) = g.row(u);
            for (&v, &w) in cols.iter().zip(weights) {
                let v = v as usize;
                if settled[v] {
                    continue;
                }
                let cand = distance + w;
                if !cand.is_finite() || cand > cutoff {
                    continue;
                }
                let sv = seen[v];
                if sv == UNREACHED || cand < sv || (cand == sv && owner < owners[v]) {
                    if sv == UNREACHED {
                        self.touched.push(v as u32);
                    }
                    seen[v] = cand;
                    owners[v] = owner;
                    if store_pred {
                        parents[v] = node;
                    }
                    heap.push(Reverse(K::make(cand, owner, v as i32)));
                }
            }
        }
    }

    /// Single-source search to the inclusive `cutoff` that hands every settled node and its cost to
    /// `visit`, in order of cost, and stops when `visit` returns `false`. Leaves the buffers clean.
    pub fn walk(
        &mut self,
        g: Csr,
        source: i32,
        cutoff: f32,
        mut visit: impl FnMut(usize, f32) -> bool,
    ) {
        let mut heap = std::mem::take(&mut self.heap_single);
        heap.clear();
        let s = source as usize;
        self.seen[s] = 0.0;
        self.touched.push(s as u32);
        heap.push(Reverse(<u64 as Key>::make(0.0, source, source)));
        while let Some(Reverse(key)) = heap.pop() {
            let u = key.node() as usize;
            if self.settled[u] {
                continue;
            }
            self.settled[u] = true;
            let distance = self.seen[u];
            if !visit(u, distance) {
                break;
            }
            let (cols, weights) = g.row(u);
            for (&v, &w) in cols.iter().zip(weights) {
                let v = v as usize;
                if self.settled[v] {
                    continue;
                }
                let cand = distance + w;
                if !cand.is_finite() || cand > cutoff {
                    continue;
                }
                let sv = self.seen[v];
                if sv == UNREACHED || cand < sv {
                    if sv == UNREACHED {
                        self.touched.push(v as u32);
                    }
                    self.seen[v] = cand;
                    heap.push(Reverse(<u64 as Key>::make(cand, source, v as i32)));
                }
            }
        }
        self.heap_single = heap;
        self.reset(&[]);
    }

    /// Single-source search to `cutoff` whose state stays in the buffers: [`Scratch::settled_order`]
    /// and [`Scratch::cost`] read it, [`Scratch::clear`] must follow.
    pub(crate) fn run_single(&mut self, g: Csr, source: i32, cutoff: f32) {
        self.run(g, &[source], &[], cutoff, false);
    }

    /// Nodes settled by the last search, in order of cost.
    pub(crate) fn settled_order(&self) -> &[u32] {
        &self.settled_list
    }

    /// Cost of a node settled by the last search.
    pub(crate) fn cost(&self, node: usize) -> f32 {
        self.seen[node]
    }

    pub(crate) fn clear(&mut self) {
        self.reset(&[]);
    }

    /// Compact arrays of the last search.
    fn compact(&self, targets: &[i32], store_pred: bool) -> Compact {
        let keep_all = store_pred || targets.is_empty();
        let n = self.settled.len();
        let nodes: Vec<i32> = if self.settled_list.len() * 8 > n {
            // Large result: scanning the flags is cheaper than sorting the settled list.
            (0..n)
                .filter(|&i| self.settled[i] && (keep_all || self.target_mask[i]))
                .map(|i| i as i32)
                .collect()
        } else {
            let mut nodes: Vec<i32> = self
                .settled_list
                .iter()
                .filter(|&&v| keep_all || self.target_mask[v as usize])
                .map(|&v| v as i32)
                .collect();
            nodes.sort_unstable();
            nodes
        };
        let dist = nodes.iter().map(|&i| self.seen[i as usize]).collect();
        let owners = nodes.iter().map(|&i| self.owners[i as usize]).collect();
        let parents = if store_pred {
            nodes.iter().map(|&i| self.parents[i as usize]).collect()
        } else {
            Vec::new()
        };
        Compact {
            nodes,
            dist,
            parents,
            owners,
        }
    }
}

/// One search, returning the compact arrays. `scratch` is created on first use and reused.
pub fn search(
    g: Csr,
    sources: &[i32],
    targets: &[i32],
    cutoff: f32,
    store_pred: bool,
    scratch: &mut Option<Scratch>,
) -> Compact {
    let sc = scratch.get_or_insert_with(|| Scratch::new(g.n()));
    sc.run(g, sources, targets, cutoff, store_pred);
    let out = sc.compact(targets, store_pred);
    sc.reset(targets);
    out
}

/// Costs of one single-source search at `targets` (infinity when not reached), written into `row`.
pub fn search_into_row(
    g: Csr,
    source: i32,
    targets: &[i32],
    cutoff: f32,
    scratch: &mut Option<Scratch>,
    row: &mut [f32],
) {
    let sc = scratch.get_or_insert_with(|| Scratch::new(g.n()));
    sc.run(g, &[source], targets, cutoff, false);
    for (slot, &t) in row.iter_mut().zip(targets) {
        let t = t as usize;
        *slot = if sc.settled[t] {
            sc.seen[t]
        } else {
            f32::INFINITY
        };
    }
    sc.reset(targets);
}

/// Append the route to `target` from a compact tree (`nodes` ascending, `parents` aligned with them) to
/// `out`, source first (target first with `reverse`). Returns `false` and appends nothing when the
/// target is not in the tree.
pub fn reconstruct(
    nodes: &[i32],
    parents: &[i32],
    target: i32,
    reverse: bool,
    out: &mut Vec<i32>,
) -> bool {
    let start = out.len();
    let mut node = target;
    while node >= 0 {
        match nodes.binary_search(&node) {
            Ok(k) => {
                out.push(node);
                node = parents[k];
            }
            Err(_) => {
                out.truncate(start);
                return false;
            }
        }
        if out.len() - start > nodes.len() {
            // A parent cycle cannot come from a search; guard against malformed input anyway.
            out.truncate(start);
            return false;
        }
    }
    if !reverse {
        out[start..].reverse();
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straight reference implementation: fresh arrays, triple heap, full scan.
    fn reference(
        g: Csr,
        sources: &[i32],
        targets: &[i32],
        cutoff: f32,
        store_pred: bool,
    ) -> Compact {
        let n = g.n();
        let mut seen = vec![UNREACHED; n];
        let mut owners = vec![-1i32; n];
        let mut settled = vec![false; n];
        let mut parents = vec![-1i32; n];
        let mut mask = vec![false; n];
        let mut left = 0;
        for &t in targets {
            if !mask[t as usize] {
                mask[t as usize] = true;
                left += 1;
            }
        }
        let mut heap = BinaryHeap::new();
        for &s in sources {
            if seen[s as usize] == UNREACHED {
                seen[s as usize] = 0.0;
                owners[s as usize] = s;
                heap.push(Reverse((0f32.to_bits(), s, s)));
            }
        }
        while let Some(Reverse((dbits, owner, node))) = heap.pop() {
            let u = node as usize;
            if settled[u] {
                continue;
            }
            settled[u] = true;
            if !targets.is_empty() && mask[u] {
                left -= 1;
                if left == 0 {
                    break;
                }
            }
            let d = f32::from_bits(dbits);
            let (cols, ws) = g.row(u);
            for (&v, &w) in cols.iter().zip(ws) {
                let v = v as usize;
                if settled[v] {
                    continue;
                }
                let c = d + w;
                if !c.is_finite() || c > cutoff {
                    continue;
                }
                if seen[v] == UNREACHED || c < seen[v] || (c == seen[v] && owner < owners[v]) {
                    seen[v] = c;
                    owners[v] = owner;
                    parents[v] = node;
                    heap.push(Reverse((c.to_bits(), owner, v as i32)));
                }
            }
        }
        let keep_all = store_pred || targets.is_empty();
        let nodes: Vec<i32> = (0..n)
            .filter(|&i| settled[i] && (keep_all || mask[i]))
            .map(|i| i as i32)
            .collect();
        Compact {
            dist: nodes.iter().map(|&i| seen[i as usize]).collect(),
            owners: nodes.iter().map(|&i| owners[i as usize]).collect(),
            parents: if store_pred {
                nodes.iter().map(|&i| parents[i as usize]).collect()
            } else {
                Vec::new()
            },
            nodes,
        }
    }

    fn grid(w: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let n = w * w;
        let mut indptr = vec![0i32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            let (r, c) = (i / w, i % w);
            let mut nb = Vec::new();
            if r > 0 {
                nb.push(i - w);
            }
            if c > 0 {
                nb.push(i - 1);
            }
            if c + 1 < w {
                nb.push(i + 1);
            }
            if r + 1 < w {
                nb.push(i + w);
            }
            for j in nb {
                indices.push(j as i32);
                data.push(((i * 7 + j * 13) % 5) as f32 * 0.5);
            }
            indptr.push(indices.len() as i32);
        }
        (indptr, indices, data)
    }

    #[test]
    fn matches_reference_and_reuses_scratch() {
        let (ip, ix, d) = grid(30);
        let g = Csr {
            indptr: &ip,
            indices: &ix,
            data: &d,
        };
        let mut sc = None;
        for (sources, targets) in [
            (vec![0], vec![]),
            (vec![5, 400], vec![]),
            (vec![17], vec![899, 3, 3, 450]),
            (vec![17, 17, 600], vec![1]),
        ] {
            for cutoff in [f32::INFINITY, 4.0, 0.0] {
                for pred in [false, true] {
                    let expected = reference(g, &sources, &targets, cutoff, pred);
                    assert_eq!(
                        search(g, &sources, &targets, cutoff, pred, &mut sc),
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn row_and_routes() {
        let (ip, ix, d) = grid(10);
        let g = Csr {
            indptr: &ip,
            indices: &ix,
            data: &d,
        };
        let targets = [99, 0, 55];
        let mut row = vec![0f32; 3];
        let mut sc = None;
        search_into_row(g, 0, &targets, f32::INFINITY, &mut sc, &mut row);
        let full = search(g, &[0], &[], f32::INFINITY, true, &mut sc);
        for (k, &t) in targets.iter().enumerate() {
            let pos = full.nodes.binary_search(&t).unwrap();
            assert_eq!(row[k], full.dist[pos]);
        }
        let mut route = Vec::new();
        assert!(reconstruct(
            &full.nodes,
            &full.parents,
            99,
            false,
            &mut route
        ));
        assert_eq!(route.first(), Some(&0));
        assert_eq!(route.last(), Some(&99));
        assert!(!reconstruct(
            &full.nodes,
            &full.parents,
            1000,
            false,
            &mut route
        ));
    }
}
