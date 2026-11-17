//! Merging street segments into edges: maximal chains of segments that follow one another.
//!
//! A segment runs from node `starts[i]` to node `ends[i]` and carries a key (its attributes). Two
//! segments are joined at a node when they are the only segments there, one ends and the other starts at
//! it, and their keys are equal. So a chain never runs through a junction, never against the direction of
//! a segment, and never across a change of attributes (a bridge stays apart from the road it continues).

const NONE: u32 = u32::MAX;

/// Chains of segments: `order[offsets[c]..offsets[c + 1]]` are the segments of chain `c`, in travel order.
/// Chains come in the order of their first segment; a closed ring starts at its lowest segment.
#[allow(clippy::needless_range_loop)]
pub fn merge(starts: &[u32], ends: &[u32], keys: &[u32], n_nodes: usize) -> (Vec<u32>, Vec<i64>) {
    let m = starts.len();
    let mut degree = vec![0u32; n_nodes];
    let mut incoming = vec![NONE; n_nodes];
    let mut outgoing = vec![NONE; n_nodes];
    let mut n_in = vec![0u32; n_nodes];
    let mut n_out = vec![0u32; n_nodes];
    for s in 0..m {
        let (a, b) = (starts[s] as usize, ends[s] as usize);
        degree[a] += 1;
        degree[b] += 1;
        outgoing[a] = s as u32;
        n_out[a] += 1;
        incoming[b] = s as u32;
        n_in[b] += 1;
    }
    let joint = |node: usize| {
        degree[node] == 2
            && n_in[node] == 1
            && n_out[node] == 1
            && incoming[node] != outgoing[node]
            && keys[incoming[node] as usize] == keys[outgoing[node] as usize]
    };
    let next = |s: usize| {
        let b = ends[s] as usize;
        if joint(b) {
            outgoing[b]
        } else {
            NONE
        }
    };

    let mut order = Vec::with_capacity(m);
    let mut offsets = Vec::with_capacity(m / 2 + 1);
    offsets.push(0i64);
    let mut done = vec![false; m];
    let follow = |first: usize, order: &mut Vec<u32>, done: &mut Vec<bool>| {
        let mut s = first;
        loop {
            done[s] = true;
            order.push(s as u32);
            let n = next(s);
            if n == NONE || done[n as usize] {
                break;
            }
            s = n as usize;
        }
    };
    // Open chains start at a segment that nothing leads into.
    for s in 0..m {
        if !joint(starts[s] as usize) {
            follow(s, &mut order, &mut done);
            offsets.push(order.len() as i64);
        }
    }
    // What is left are closed rings without a junction.
    for s in 0..m {
        if !done[s] {
            follow(s, &mut order, &mut done);
            offsets.push(order.len() as i64);
        }
    }
    (order, offsets)
}

/// Chains of edges of a finished graph, which may be travelled against their drawing: two edges are joined
/// at a node that is not in `keep`, is an end of exactly these two edges, and where their keys are equal and
/// their directions agree. Two two-way edges always agree (one of them is turned when they meet head to head
/// or tail to tail); two one-way edges agree when one ends and the other starts at the node; a two-way and a
/// one-way edge never do. A chain is travelled in the direction of its one-way edges.
///
/// Returns `(order, reversed, offsets)`: `order[offsets[c]..offsets[c + 1]]` are the edges of chain `c` in
/// travel order and `reversed[i]` tells whether `order[i]` is travelled from `ends` to `starts`. Chains come in
/// the order of their first edge; a closed ring starts at its lowest edge, travelled as drawn.
#[allow(clippy::needless_range_loop, clippy::type_complexity)]
pub fn merge_oriented(
    starts: &[u32],
    ends: &[u32],
    keys: &[u32],
    two_way: &[bool],
    keep: &[bool],
    n_nodes: usize,
) -> (Vec<u32>, Vec<bool>, Vec<i64>) {
    let m = starts.len();
    let mut degree = vec![0u32; n_nodes];
    let mut first = vec![NONE; n_nodes];
    let mut second = vec![NONE; n_nodes];
    for e in 0..m {
        for node in [starts[e] as usize, ends[e] as usize] {
            degree[node] += 1;
            if first[node] == NONE {
                first[node] = e as u32;
            } else if second[node] == NONE {
                second[node] = e as u32;
            }
        }
    }
    let joint = |node: usize| {
        if keep[node] || degree[node] != 2 || first[node] == second[node] {
            return false;
        }
        let (a, b) = (first[node] as usize, second[node] as usize);
        if keys[a] != keys[b] {
            return false;
        }
        match (two_way[a], two_way[b]) {
            (true, true) => true,
            (false, false) => (ends[a] as usize == node) != (ends[b] as usize == node),
            _ => false,
        }
    };
    let other = |node: usize, e: u32| {
        if first[node] == e {
            second[node]
        } else {
            first[node]
        }
    };

    let mut order = Vec::with_capacity(m);
    let mut reversed = Vec::with_capacity(m);
    let mut offsets = Vec::with_capacity(m / 2 + 1);
    offsets.push(0i64);
    let mut done = vec![false; m];
    let follow = |mut e: usize,
                  mut rev: bool,
                  order: &mut Vec<u32>,
                  reversed: &mut Vec<bool>,
                  done: &mut Vec<bool>| loop {
        done[e] = true;
        order.push(e as u32);
        reversed.push(rev);
        let head = if rev { starts[e] } else { ends[e] } as usize;
        if !joint(head) {
            break;
        }
        let n = other(head, e as u32) as usize;
        if done[n] {
            break;
        }
        rev = starts[n] as usize != head;
        e = n;
    };
    // Open chains start at an edge with a free end: its tail, or for a two-way edge either end.
    for e in 0..m {
        if done[e] {
            continue;
        }
        let start = if !joint(starts[e] as usize) {
            Some(false)
        } else if two_way[e] && !joint(ends[e] as usize) {
            Some(true)
        } else {
            None
        };
        if let Some(rev) = start {
            follow(e, rev, &mut order, &mut reversed, &mut done);
            offsets.push(order.len() as i64);
        }
    }
    // What is left are closed rings without a junction.
    for e in 0..m {
        if !done[e] {
            follow(e, false, &mut order, &mut reversed, &mut done);
            offsets.push(order.len() as i64);
        }
    }
    (order, reversed, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chains_stop_at_junctions_key_changes_and_opposed_segments() {
        // 0 -> 1 -> 2 -> 3 with a branch 2 -> 4; then 3 -> 5 with another key; 6 -> 5 runs against it.
        let starts = [0, 1, 2, 2, 3, 6];
        let ends = [1, 2, 3, 4, 5, 5];
        let keys = [0, 0, 0, 0, 1, 1];
        let (order, offsets) = merge(&starts, &ends, &keys, 7);
        let chains: Vec<&[u32]> = offsets
            .windows(2)
            .map(|w| &order[w[0] as usize..w[1] as usize])
            .collect();
        assert_eq!(chains, vec![&[0, 1][..], &[2], &[3], &[4], &[5]]);
    }

    #[test]
    fn rings_and_single_segments() {
        // A ring 0 -> 1 -> 2 -> 0 and a loose segment 3 -> 4.
        let (order, offsets) = merge(&[0, 1, 2, 3], &[1, 2, 0, 4], &[0, 0, 0, 0], 5);
        assert_eq!((order, offsets), (vec![3, 0, 1, 2], vec![0, 1, 4]));
        assert_eq!(merge(&[], &[], &[], 0), (vec![], vec![0]));
    }
    fn chains(order: &[u32], offsets: &[i64]) -> Vec<Vec<u32>> {
        offsets
            .windows(2)
            .map(|w| order[w[0] as usize..w[1] as usize].to_vec())
            .collect()
    }

    #[test]
    fn two_way_edges_drawn_head_to_head_are_joined_and_turned() {
        // 0 -> 1 <- 2 -> 3, all two-way with one key: one chain, the middle edge travelled backwards.
        let (order, rev, offsets) = merge_oriented(
            &[0, 2, 2],
            &[1, 1, 3],
            &[0, 0, 0],
            &[true; 3],
            &[false; 4],
            4,
        );
        assert_eq!(chains(&order, &offsets), vec![vec![0, 1, 2]]);
        assert_eq!(rev, vec![false, true, false]);
    }

    #[test]
    fn one_way_edges_join_only_head_to_tail() {
        // 0 -> 1 -> 2 one-way joins; 2 <- 3 one-way meets head to head and stays apart; 3 - 4 two-way stays apart.
        let (order, _, offsets) = merge_oriented(
            &[0, 1, 3, 3],
            &[1, 2, 2, 4],
            &[0, 0, 0, 0],
            &[false, false, false, true],
            &[false; 5],
            5,
        );
        assert_eq!(chains(&order, &offsets), vec![vec![0, 1], vec![2], vec![3]]);
    }

    #[test]
    fn a_one_way_chain_is_found_from_its_tail_whatever_the_edge_order() {
        let (order, rev, offsets) =
            merge_oriented(&[1, 0], &[2, 1], &[0, 0], &[false, false], &[false; 3], 3);
        assert_eq!(chains(&order, &offsets), vec![vec![1, 0]]);
        assert_eq!(rev, vec![false, false]);
    }

    #[test]
    fn keys_kept_nodes_rings_and_loops() {
        // A two-way ring 0-1-2-0 drawn with mixed directions; node 4 kept splits 3-4-5; a loop at 6.
        let starts = [0, 2, 2, 3, 4, 6];
        let ends = [1, 1, 0, 4, 5, 6];
        let keep = [false, false, false, false, true, false, false];
        let (order, rev, offsets) = merge_oriented(&starts, &ends, &[0; 6], &[true; 6], &keep, 7);
        assert_eq!(
            chains(&order, &offsets),
            vec![vec![3], vec![4], vec![5], vec![0, 1, 2]]
        );
        assert_eq!(&rev[3..], &[false, true, false]);
        // A change of key splits the ring into chains that start at the change.
        let (order, _, offsets) = merge_oriented(
            &[0, 1, 2],
            &[1, 2, 0],
            &[0, 0, 1],
            &[true; 3],
            &[false; 3],
            3,
        );
        assert_eq!(chains(&order, &offsets), vec![vec![0, 1], vec![2]]);
    }
}
