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
}
