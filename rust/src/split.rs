//! Split projected edges at sorted distances, returning the original cut positions for node lookup.

use std::hint::black_box;

pub(crate) const START: i64 = -1;
pub(crate) const END: i64 = -2;

pub(crate) struct Pieces {
    pub coords: Vec<f64>,
    pub offsets: Vec<i64>,
    pub edges: Vec<i64>,
    pub starts: Vec<i64>,
    pub ends: Vec<i64>,
}

struct Line<'a> {
    coords: &'a [f64],
    segment_lengths: Vec<f64>,
    geos_distances: Vec<f64>,
    vertex_distances: Vec<f64>,
}

impl<'a> Line<'a> {
    fn new(coords: &'a [f64]) -> Self {
        let count = coords.len() / 2;
        let mut segment_lengths = Vec::with_capacity(count - 1);
        let mut geos_distances = Vec::with_capacity(count);
        let mut vertex_distances = Vec::with_capacity(count);
        geos_distances.push(0.0);
        vertex_distances.push(0.0);
        let (mut geos, mut vertex) = (0.0, 0.0);
        // Shapely's substring uses Python's float ** for the interior vertex distances.
        let (two, half) = (black_box(2.0), black_box(0.5));
        for i in (2..coords.len()).step_by(2) {
            let dx = coords[i] - coords[i - 2];
            let dy = coords[i + 1] - coords[i - 1];
            let segment = (dx * dx + dy * dy).sqrt();
            geos += segment;
            vertex += (dx.powf(two) + dy.powf(two)).powf(half);
            segment_lengths.push(segment);
            geos_distances.push(geos);
            vertex_distances.push(vertex);
        }
        Self {
            coords,
            segment_lengths,
            geos_distances,
            vertex_distances,
        }
    }

    fn length(&self) -> f64 {
        *self.geos_distances.last().unwrap()
    }

    fn interpolate(&self, distance: f64) -> [f64; 2] {
        // GEOS walks until the end of the segment lies strictly beyond the requested index.
        let i = self.geos_distances[1..].partition_point(|&end| end <= distance);
        let x = self.coords[2 * i];
        let y = self.coords[2 * i + 1];
        if distance <= 0.0 {
            return [self.coords[0], self.coords[1]];
        }
        if i == self.segment_lengths.len() {
            return [x, y];
        }
        let frac = (distance - self.geos_distances[i]) / self.segment_lengths[i];
        let next_x = self.coords[2 * i + 2];
        let next_y = self.coords[2 * i + 3];
        if frac <= 0.0 {
            return [x, y];
        }
        if frac >= 1.0 {
            return [next_x, next_y];
        }
        [(next_x - x) * frac + x, (next_y - y) * frac + y]
    }

    fn piece(&self, start: f64, end: f64, coords: &mut Vec<f64>) {
        coords.extend(self.interpolate(start));
        let interior = &self.vertex_distances[..self.segment_lengths.len()];
        let first = interior.partition_point(|&distance| distance <= start);
        let last = interior.partition_point(|&distance| distance < end);
        coords.extend_from_slice(&self.coords[2 * first..2 * last]);
        coords.extend(self.interpolate(end));
    }
}

impl Pieces {
    fn push(&mut self, line: &Line<'_>, edge: usize, start: (f64, i64), end: (f64, i64)) {
        if start.0 >= end.0 || start.1 == end.1 {
            return;
        }
        let first = self.coords.len();
        line.piece(start.0, end.0, &mut self.coords);
        let piece = &self.coords[first..];
        let nonzero = (2..piece.len()).step_by(2).any(|i| {
            let dx = piece[i] - piece[i - 2];
            let dy = piece[i + 1] - piece[i - 1];
            dx * dx + dy * dy > 0.0
        });
        if !nonzero {
            self.coords.truncate(first);
            return;
        }
        self.offsets.push((self.coords.len() / 2) as i64);
        self.edges.push(edge as i64);
        self.starts.push(start.1);
        self.ends.push(end.1);
    }
}

/// Offsets index coordinate pairs and cuts separately. A piece endpoint is START, END, or a
/// position in the flat cuts array. The caller resolves those positions to the graph's node ids.
pub(crate) fn split_edges(
    coords: &[f64],
    coord_offsets: &[i64],
    cuts: &[f64],
    cut_offsets: &[i64],
) -> Pieces {
    let mut out = Pieces {
        coords: Vec::with_capacity(coords.len() + 4 * cuts.len()),
        offsets: vec![0],
        edges: Vec::new(),
        starts: Vec::new(),
        ends: Vec::new(),
    };
    for (edge, (vertices, range)) in coord_offsets
        .windows(2)
        .zip(cut_offsets.windows(2))
        .enumerate()
    {
        let line = Line::new(&coords[2 * vertices[0] as usize..2 * vertices[1] as usize]);
        let length = line.length();
        let eps = 1e-6 * length;
        let mut previous = (0.0, START);
        let mut group_start = f64::NEG_INFINITY;
        let edge_cuts = &cuts[range[0] as usize..range[1] as usize];
        for (offset, &distance) in edge_cuts.iter().enumerate() {
            if distance - group_start <= eps {
                continue;
            }
            group_start = distance;
            if distance.abs() <= eps {
                continue;
            }
            if (distance - length).abs() <= eps {
                break;
            }
            let next = (distance, range[0] + offset as i64);
            out.push(&line, edge, previous, next);
            previous = next;
        }
        out.push(&line, edge, previous, (length, END));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_at_vertex_and_between_vertices() {
        let result = split_edges(
            &[0., 0., 3., 0., 3., 4., 6., 4.],
            &[0, 4],
            &[3., 8.],
            &[0, 2],
        );
        assert_eq!(result.edges, [0, 0, 0]);
        assert_eq!(result.starts, [START, 0, 1]);
        assert_eq!(result.ends, [0, 1, END]);
        assert_eq!(result.offsets, [0, 2, 5, 7]);
        assert_eq!(
            result.coords,
            [0., 0., 3., 0., 3., 0., 3., 4., 4., 4., 4., 4., 6., 4.]
        );
    }

    #[test]
    fn snap_ends_and_group_relative_to_the_first_cut() {
        let cuts = [0., 0.00005, 25., 25.00009, 25.00018, 99.99995, 100.];
        let result = split_edges(&[0., 0., 100., 0.], &[0, 2], &cuts, &[0, 7]);
        assert_eq!(result.starts, [START, 2, 4]);
        assert_eq!(result.ends, [2, 4, END]);
        assert_eq!(result.offsets, [0, 2, 4, 6]);
    }

    #[test]
    fn skip_zero_length_edges_and_keep_flat_cut_indices() {
        let result = split_edges(
            &[7., 7., 7., 7., 0., 0., 2., 0.],
            &[0, 2, 4],
            &[0., 1.],
            &[0, 1, 2],
        );
        assert_eq!(result.edges, [1, 1]);
        assert_eq!(result.starts, [START, 1]);
        assert_eq!(result.ends, [1, END]);
    }
}
