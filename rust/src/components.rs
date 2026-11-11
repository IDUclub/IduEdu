//! Connected and strongly connected components of a CSR pattern (weights are ignored).
//!
//! Labels are numbered in discovery order, starting from the lowest unlabelled node position.

/// Connected components of a symmetric CSR pattern (iterative depth-first search).
pub fn connected(indptr: &[i32], indices: &[i32]) -> Vec<i32> {
    let n = indptr.len().saturating_sub(1);
    let mut labels = vec![-1i32; n];
    let mut stack: Vec<i32> = Vec::with_capacity(n);
    let mut comp = 0i32;
    for start in 0..n {
        if labels[start] != -1 {
            continue;
        }
        stack.push(start as i32);
        labels[start] = comp;
        while let Some(node) = stack.pop() {
            let u = node as usize;
            for &v in &indices[indptr[u] as usize..indptr[u + 1] as usize] {
                if labels[v as usize] == -1 {
                    labels[v as usize] = comp;
                    stack.push(v);
                }
            }
        }
        comp += 1;
    }
    labels
}

/// Strongly connected components (iterative Kosaraju), given the CSR pattern and its transpose.
pub fn strong(indptr: &[i32], indices: &[i32], rindptr: &[i32], rindices: &[i32]) -> Vec<i32> {
    let n = indptr.len().saturating_sub(1);
    let mut visited = vec![false; n];
    let mut order: Vec<i32> = Vec::with_capacity(n);
    let mut stack: Vec<(i32, i32)> = Vec::with_capacity(n);
    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        stack.push((start as i32, 0));
        while let Some(top) = stack.last_mut() {
            let u = top.0 as usize;
            let s = indptr[u] as usize;
            let deg = indptr[u + 1] as usize - s;
            if (top.1 as usize) < deg {
                let v = indices[s + top.1 as usize];
                top.1 += 1;
                if !visited[v as usize] {
                    visited[v as usize] = true;
                    stack.push((v, 0));
                }
            } else {
                order.push(top.0);
                stack.pop();
            }
        }
    }
    let mut labels = vec![-1i32; n];
    let mut st: Vec<i32> = Vec::with_capacity(n);
    let mut comp = 0i32;
    for &start in order.iter().rev() {
        if labels[start as usize] != -1 {
            continue;
        }
        st.push(start);
        labels[start as usize] = comp;
        while let Some(node) = st.pop() {
            let u = node as usize;
            for &v in &rindices[rindptr[u] as usize..rindptr[u + 1] as usize] {
                if labels[v as usize] == -1 {
                    labels[v as usize] = comp;
                    st.push(v);
                }
            }
        }
        comp += 1;
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_of_small_graphs() {
        // 0 - 1   2 - 3 - 4   5
        let ip = [0, 1, 2, 3, 5, 6, 6];
        let ix = [1, 0, 3, 2, 4, 3];
        assert_eq!(connected(&ip, &ix), vec![0, 0, 1, 1, 1, 2]);
        // 0 -> 1 -> 2 -> 0, 2 -> 3
        let ip = [0, 1, 2, 4, 4];
        let ix = [1, 2, 0, 3];
        let rip = [0, 1, 2, 3, 4];
        let rix = [2, 0, 1, 2];
        let labels = strong(&ip, &ix, &rip, &rix);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_ne!(labels[2], labels[3]);
    }
}
