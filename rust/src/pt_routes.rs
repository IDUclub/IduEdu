//! Public-transport routes (bus, trolleybus, tram, train, subway and the like) from their OSM relations, all
//! routes in one call: the ways of a route stitched in member order into its path, its stops paired with its
//! platforms, and its stop nodes, platform nodes, travel edges between consecutive stops and boarding edges.
//!
//! The geometry repeats, operation for operation, what GEOS does when shapely asks it for the distance along
//! a line to a point (`project`), the point at a distance along it (`interpolate`), the distance of a point
//! from a line and the centroid of a line, so the results equal those of the Python parser this replaced.
//! The vertex distances that decide which vertices a cut keeps are summed from `pow(.., 0.5)` as Python's
//! `** 0.5` takes them; the exponent goes through `black_box` so LLVM cannot turn the call into `sqrt`.

use std::collections::{BTreeMap, HashSet};
use std::hint::black_box;

use rayon::prelude::*;

pub type Xy = [f64; 2];

/// Furthest a platform may lie from its stop on the path, and a stop from the path when it votes on the
/// direction of the route.
const THRESHOLD_METERS: f64 = 100.0;
/// Longest straight join drawn between two pieces of one route. Pieces further apart are a hole in the
/// relation rather than a short way OSM missed, and the route is cut there instead of bridged: in Delhi the
/// joins cluster under 500 m, thin out towards 1 km and grow in number again beyond it, where a straight
/// line would stand in for kilometres of road nobody mapped.
const MAX_JOIN_METERS: f64 = 1000.0;
/// Distance from the path of a platform made up for a stop that has none.
const PLATFORM_OFFSET_METERS: f64 = 7.0;
const MAX_IMPROVE_ROUNDS: usize = 50;
const TOLERANCE: f64 = 1e-6;

/// A stop or a platform of a route: its point, the index of its OSM reference and its rank, the position of
/// its first stop or platform member with that reference in the relation (negative when unknown).
///
/// A stop may know a platform from elsewhere (a subway stop area): a stop left without a platform of the
/// route takes that one (`stand_in`, negative when none, at `stand_in_xy` unless it is NaN) instead of one
/// made up beside the path.
#[derive(Clone, Copy, Debug)]
pub struct Member {
    pub xy: Xy,
    pub reference: i64,
    pub rank: i64,
    pub stand_in: i64,
    pub stand_in_xy: Xy,
}

impl Member {
    /// A member without a stand-in platform.
    #[cfg(test)]
    fn new(xy: Xy, reference: i64, rank: i64) -> Self {
        Member {
            xy,
            reference,
            rank,
            stand_in: -1,
            stand_in_xy: [f64::NAN, f64::NAN],
        }
    }
}

/// A way of a route in member order: its points (none when the way has no geometry) and its speed.
#[derive(Clone, Copy, Debug)]
pub struct Way<'a> {
    pub coords: &'a [Xy],
    pub speed: f64,
}

pub struct Route<'a> {
    pub ways: Vec<Way<'a>>,
    pub stops: Vec<Member>,
    pub platforms: Vec<Member>,
    /// Keep a stop node and a platform node with one id apart (the subway parser does) instead of keeping
    /// whichever came first.
    pub kinds_apart: bool,
}

/// Node id: `2 * reference` for a member, `2 * reference + 1` for the node made up from it (a platform for
/// a stop that has none, a stop for a platform that has none), which Python names `from_<reference>`.
fn member_id(reference: i64) -> i64 {
    2 * reference
}

fn made_up_id(reference: i64) -> i64 {
    2 * reference + 1
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub id: i64,
    pub xy: Xy,
    pub platform: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub u: i64,
    pub v: i64,
    pub boarding: bool,
    pub coords: Vec<Xy>,
    /// Length-weighted speed of the ways under a travel edge; NaN for a boarding edge.
    pub speed: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Parsed {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

/// A stretch of a route: its vertices and one speed per vertex, for the segment leaving it.
#[derive(Clone, Debug, Default, PartialEq)]
struct Piece {
    pub coords: Vec<Xy>,
    pub speeds: Vec<f64>,
}

// ---------------------------------------------------------------------------------------------------------
// Geometry, as GEOS computes it
// ---------------------------------------------------------------------------------------------------------

/// `CoordinateXY::distance`.
#[inline]
fn distance(a: Xy, b: Xy) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// `Distance::pointToSegment`.
fn point_to_segment(p: Xy, a: Xy, b: Xy) -> f64 {
    if a == b {
        return distance(p, a);
    }
    let len2 = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]);
    let r = ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / len2;
    if r <= 0.0 {
        return distance(p, a);
    }
    if r >= 1.0 {
        return distance(p, b);
    }
    let s = ((a[1] - p[1]) * (b[0] - a[0]) - (a[0] - p[0]) * (b[1] - a[1])) / len2;
    s.abs() * len2.sqrt()
}

/// `Length::ofLine`.
fn line_length(line: &[Xy]) -> f64 {
    let mut length = 0.0;
    for w in line.windows(2) {
        let dx = w[1][0] - w[0][0];
        let dy = w[1][1] - w[0][1];
        length += (dx * dx + dy * dy).sqrt();
    }
    length
}

/// Distance of a point from a line (`DistanceOp` between a line and a point).
fn line_distance(line: &[Xy], p: Xy) -> f64 {
    if line.len() == 1 {
        return distance(line[0], p);
    }
    let mut min = f64::INFINITY;
    for w in line.windows(2) {
        let d = point_to_segment(p, w[0], w[1]);
        if d < min {
            min = d;
        }
    }
    min
}

/// `LineSegment::projectionFactor`.
fn projection_factor(a: Xy, b: Xy, p: Xy) -> f64 {
    if p == a {
        return 0.0;
    }
    if p == b {
        return 1.0;
    }
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len2 = dx * dx + dy * dy;
    ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2
}

/// Distance along a line to the point of it nearest `p` (`LengthIndexOfPoint::indexOf`, shapely's `project`).
fn project(line: &[Xy], p: Xy) -> f64 {
    let min_index = -1.0;
    let mut min_distance = f64::INFINITY;
    let mut measure = min_index;
    let mut start = 0.0;
    for w in line.windows(2) {
        let (a, b) = (w[0], w[1]);
        let seg_distance = point_to_segment(p, a, b);
        let factor = projection_factor(a, b, p);
        let seg_measure = if factor <= 0.0 {
            start
        } else if factor <= 1.0 {
            start + factor * distance(a, b)
        } else {
            start + distance(a, b)
        };
        if seg_distance < min_distance && seg_measure > min_index {
            measure = seg_measure;
            min_distance = seg_distance;
        }
        start += distance(a, b);
    }
    measure
}

/// Point at a distance along a line (`LengthIndexedLine::extractPoint`, shapely's `interpolate`).
fn interpolate(line: &[Xy], length: f64) -> Xy {
    let length = if length < 0.0 {
        line_length(line) + length
    } else {
        length
    };
    if length <= 0.0 {
        return line[0];
    }
    let mut total = 0.0;
    for i in 0..line.len() - 1 {
        let (p0, p1) = (line[i], line[i + 1]);
        let seg = distance(p1, p0);
        if total + seg > length {
            let frac = (length - total) / seg;
            if frac <= 0.0 {
                return p0;
            }
            if frac >= 1.0 {
                return p1;
            }
            return [
                (p1[0] - p0[0]) * frac + p0[0],
                (p1[1] - p0[1]) * frac + p0[1],
            ];
        }
        total += seg;
    }
    line[line.len() - 1]
}

/// Centroid of a member's points (`Centroid` of a line; a lone point is itself).
pub fn centroid(points: &[Xy]) -> Option<Xy> {
    match points.len() {
        0 => None,
        1 => Some(points[0]),
        _ => {
            let (mut length, mut sx, mut sy) = (0.0, 0.0, 0.0);
            for w in points.windows(2) {
                let segment = distance(w[0], w[1]);
                if segment == 0.0 {
                    continue;
                }
                length += segment;
                sx += segment * ((w[0][0] + w[1][0]) / 2.0);
                sy += segment * ((w[0][1] + w[1][1]) / 2.0);
            }
            if length > 0.0 {
                Some([sx / length, sy / length])
            } else {
                Some(points[0])
            }
        }
    }
}

/// Distance of every vertex from the start of the line, from `pow(.., 0.5)` per segment as Python sums it.
fn vertex_distances(line: &[Xy]) -> Vec<f64> {
    let (two, half) = (black_box(2.0), black_box(0.5));
    let mut out = Vec::with_capacity(line.len());
    let mut total = 0.0;
    out.push(total);
    for w in line.windows(2) {
        let dx: f64 = w[1][0] - w[0][0];
        let dy: f64 = w[1][1] - w[0][1];
        total += (dx.powf(two) + dy.powf(two)).powf(half);
        out.push(total);
    }
    out
}

/// Cumulative `hypot` of the segments, as `numpy.cumsum(numpy.hypot(..))` gives it.
fn hypot_distances(line: &[Xy]) -> Vec<f64> {
    let mut out = Vec::with_capacity(line.len());
    let mut total = 0.0;
    out.push(total);
    for w in line.windows(2) {
        total += (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]);
        out.push(total);
    }
    out
}

/// The line between distances `start` and `end` along it (backwards when `start > end`): the vertices strictly
/// between them and `start_point`, `end_point` at its ends, as `shapely.ops.substring`; a two-point line when
/// the distances are equal.
fn cut_line(
    line: &[Xy],
    vertex_dist: &[f64],
    start: f64,
    end: f64,
    start_point: Xy,
    end_point: Xy,
) -> Vec<Xy> {
    if start == end {
        return vec![start_point, start_point];
    }
    let (low, high) = if start < end {
        (start, end)
    } else {
        (end, start)
    };
    let first = vertex_dist.partition_point(|&d| d <= low);
    let last = vertex_dist
        .partition_point(|&d| d < high)
        .min(line.len() - 1);
    let mut out = Vec::with_capacity(last.saturating_sub(first) + 2);
    out.push(start_point);
    if first < last {
        if start > end {
            out.extend(line[first..last].iter().rev());
        } else {
            out.extend_from_slice(&line[first..last]);
        }
    }
    out.push(end_point);
    out
}

/// Length-weighted mean of the segment speeds between two distances along a line; NaN for an empty interval.
fn speed_on_interval(a: f64, b: f64, cumdist: &[f64], seg_speeds: &[f64]) -> f64 {
    let (a, b) = if b < a { (b, a) } else { (a, b) };
    if b <= a {
        return f64::NAN;
    }
    let i0 = (cumdist.partition_point(|&d| d <= a) as isize - 1).max(0) as usize;
    let i1 = cumdist.partition_point(|&d| d < b).min(seg_speeds.len());
    let (mut total, mut acc) = (0.0, 0.0);
    for i in i0..i1 {
        let low = if cumdist[i] > a { cumdist[i] } else { a };
        let high = if cumdist[i + 1] < b {
            cumdist[i + 1]
        } else {
            b
        };
        let overlap = high - low;
        if overlap > 0.0 {
            acc += overlap * seg_speeds[i];
            total += overlap;
        }
    }
    if total > 0.0 {
        acc / total
    } else {
        f64::NAN
    }
}

/// `numpy.sign`.
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        f64::NAN
    }
}

/// Index of the path a point lies nearest; the first on a tie.
fn nearest_part(p: Xy, paths: &[Vec<Xy>]) -> usize {
    if paths.len() == 1 {
        return 0;
    }
    let mut best = (f64::INFINITY, 0);
    for (k, path) in paths.iter().enumerate() {
        let d = line_distance(path, p);
        if d < best.0 || k == 0 {
            best = (d, k);
        }
    }
    best.1
}

/// Distance along a path to `p` and the ends of a short stretch of the path around it.
fn stretch_around(path: &[Xy], p: Xy, after: f64) -> (f64, Xy, Xy) {
    let dist = project(path, p);
    let length = line_length(path);
    let d1 = if dist - 1.0 > 0.0 { dist - 1.0 } else { 0.0 };
    let d2 = if dist + after < length {
        dist + after
    } else {
        length
    };
    (dist, interpolate(path, d1), interpolate(path, d2))
}

/// 1 when a point lies left of a path, 0 when right of it or on it.
fn side_of(p: Xy, path: &[Xy]) -> u8 {
    let (_, a, b) = stretch_around(path, p, 1.0);
    let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    u8::from(cross > 0.0)
}

/// The point of a path nearest `p`, moved a fixed distance to one side of the path (1 left, 0 right).
fn offset_point(p: Xy, path: &[Xy], side: u8) -> Result<Xy, String> {
    let (dist, a, b) = stretch_around(path, p, 1.1);
    let nearest = interpolate(path, dist);
    let (dx, dy): (f64, f64) = (b[0] - a[0], b[1] - a[1]);
    // `math.sqrt(dx**2 + dy**2)`: `**` is the C `pow`.
    let two = black_box(2.0);
    let length = (dx.powf(two) + dy.powf(two)).sqrt();
    if length == 0.0 {
        return Err("a stop without a platform lies on a path of zero length".into());
    }
    let (dx, dy) = (dx / length, dy / length);
    let (nx, ny) = if side == 0 { (dy, -dx) } else { (-dy, dx) };
    Ok([
        nearest[0] + nx * PLATFORM_OFFSET_METERS,
        nearest[1] + ny * PLATFORM_OFFSET_METERS,
    ])
}

// ---------------------------------------------------------------------------------------------------------
// The path of a route
// ---------------------------------------------------------------------------------------------------------

/// Ways joined end to end in member order; a way touching nothing before it starts a new piece. A closed way
/// after the first is left out; the first piece is turned round when the second way meets its start.
fn stitch(ways: &[Way]) -> Vec<Piece> {
    let mut pieces = vec![Piece::default()];
    let mut started = false;
    for way in ways {
        let coords = way.coords;
        if coords.is_empty() {
            continue;
        }
        let n = coords.len();
        let speed = way.speed;
        let current = pieces.len() - 1;
        let dst = &mut pieces[current];
        if dst.coords.is_empty() {
            dst.coords = coords.to_vec();
            dst.speeds = vec![speed; n];
            started = true;
            continue;
        }
        let (head, tail) = (coords[0], coords[n - 1]);
        if head == tail {
            continue;
        }
        let touches = |p: Xy| p == head || p == tail;
        if started && touches(dst.coords[0]) && !touches(dst.coords[dst.coords.len() - 1]) {
            dst.coords.reverse();
            dst.speeds.reverse();
        }
        let first = dst.coords[0];
        let last = dst.coords[dst.coords.len() - 1];
        if last == head {
            dst.coords.extend_from_slice(&coords[1..]);
            dst.speeds.extend(std::iter::repeat_n(speed, n - 1));
        } else if last == tail {
            dst.coords.extend(coords[..n - 1].iter().rev());
            dst.speeds.extend(std::iter::repeat_n(speed, n - 1));
        } else if first == head || first == tail {
            let mut joined: Vec<Xy> = Vec::with_capacity(n + dst.coords.len() - 1);
            if first == head {
                joined.extend(coords.iter().rev());
            } else {
                joined.extend_from_slice(coords);
            }
            joined.extend_from_slice(&dst.coords[1..]);
            let mut speeds = vec![speed; n - 1];
            speeds.extend_from_slice(&dst.speeds);
            dst.coords = joined;
            dst.speeds = speeds;
        } else {
            pieces.push(Piece {
                coords: coords.to_vec(),
                speeds: vec![speed; n],
            });
        }
    }
    pieces
}

/// Pieces without the closed loops among them; when every piece is one, the longest stays.
fn drop_closed_loops(pieces: Vec<Piece>) -> Vec<Piece> {
    if pieces.len() <= 1 {
        return pieces;
    }
    let closed = |p: &Piece| p.coords.len() >= 2 && p.coords[0] == p.coords[p.coords.len() - 1];
    let mut drop: Vec<bool> = pieces.iter().map(closed).collect();
    if drop.iter().all(|&d| d) {
        let mut longest = 0;
        for (i, piece) in pieces.iter().enumerate() {
            if piece.coords.len() > pieces[longest].coords.len() {
                longest = i;
            }
        }
        drop[longest] = false;
    }
    pieces
        .into_iter()
        .zip(drop)
        .filter_map(|(piece, d)| (!d).then_some(piece))
        .collect()
}

/// A piece in a chain: its index and whether it is travelled against its vertex order.
type Placed = (usize, bool);

/// Endpoint indices a placed piece is entered and left by: endpoint `i` is the first vertex of piece `i`,
/// endpoint `n + i` its last.
fn piece_ends(piece: Placed, n: usize) -> (usize, usize) {
    let (i, reversed) = piece;
    if reversed {
        (n + i, i)
    } else {
        (i, n + i)
    }
}

struct Joins {
    n: usize,
    distances: Vec<f64>,
}

impl Joins {
    fn at(&self, a: usize, b: usize) -> f64 {
        self.distances[a * 2 * self.n + b]
    }

    /// Join length added by placing a piece entered at `head` and left at `tail` before `sequence[pos]`.
    fn insertion_cost(&self, sequence: &[Placed], pos: usize, head: usize, tail: usize) -> f64 {
        let prev_tail = (pos > 0).then(|| piece_ends(sequence[pos - 1], self.n).1);
        let next_head = (pos < sequence.len()).then(|| piece_ends(sequence[pos], self.n).0);
        let mut added = 0.0;
        if let Some(prev) = prev_tail {
            added += self.at(prev, head);
        }
        if let Some(next) = next_head {
            added += self.at(tail, next);
        }
        if let (Some(prev), Some(next)) = (prev_tail, next_head) {
            added -= self.at(prev, next);
        }
        added
    }

    /// A chain grown from the longest piece, every other piece placed wherever it adds the least.
    fn insert_cheapest(&self, lengths: &[f64]) -> Vec<Placed> {
        let n = self.n;
        let mut seed = 0;
        for (i, &length) in lengths.iter().enumerate() {
            if length > lengths[seed] {
                seed = i;
            }
        }
        let mut sequence = vec![(seed, false)];
        let mut remaining: Vec<usize> = (0..n).filter(|&i| i != seed).collect();
        while !remaining.is_empty() {
            let mut best: (f64, usize, Option<Placed>) = (f64::INFINITY, 0, None);
            for &i in &remaining {
                for reversed in [false, true] {
                    let (head, tail) = piece_ends((i, reversed), n);
                    for pos in 0..=sequence.len() {
                        let added = self.insertion_cost(&sequence, pos, head, tail);
                        if added < best.0 {
                            best = (added, pos, Some((i, reversed)));
                        }
                    }
                }
            }
            // Only NaN joins leave no best place; the piece then goes last, where nothing compares.
            let piece = best.2.unwrap_or((remaining[0], false));
            let pos = if best.2.is_some() {
                best.1
            } else {
                sequence.len()
            };
            sequence.insert(pos, piece);
            remaining.retain(|&i| i != piece.0);
        }
        sequence
    }

    /// The chain with shorter joins: runs of pieces reversed and single pieces moved elsewhere.
    fn improve(&self, mut sequence: Vec<Placed>) -> Vec<Placed> {
        let n = self.n;
        for _ in 0..MAX_IMPROVE_ROUNDS {
            let mut improved = false;
            let size = sequence.len();

            // Reversing a run flips every piece in it, so only the two joins around the run change.
            for i in 0..size {
                for j in i..size {
                    let head_i = piece_ends(sequence[i], n).0;
                    let tail_j = piece_ends(sequence[j], n).1;
                    let (mut before, mut after) = (0.0, 0.0);
                    if i > 0 {
                        let prev_tail = piece_ends(sequence[i - 1], n).1;
                        before += self.at(prev_tail, head_i);
                        after += self.at(prev_tail, tail_j);
                    }
                    if j < size - 1 {
                        let next_head = piece_ends(sequence[j + 1], n).0;
                        before += self.at(tail_j, next_head);
                        after += self.at(head_i, next_head);
                    }
                    if after < before - TOLERANCE {
                        sequence[i..=j].reverse();
                        for piece in &mut sequence[i..=j] {
                            piece.1 = !piece.1;
                        }
                        improved = true;
                    }
                }
            }

            for idx in 0..size {
                let piece = sequence[idx];
                let mut rest = sequence.clone();
                rest.remove(idx);
                let (head, tail) = piece_ends(piece, n);
                let mut best: (f64, Option<(usize, Placed)>) = (
                    self.insertion_cost(&rest, idx, head, tail) - TOLERANCE,
                    None,
                );
                for reversed in [false, true] {
                    let (head, tail) = piece_ends((piece.0, reversed), n);
                    for pos in 0..=rest.len() {
                        let added = self.insertion_cost(&rest, pos, head, tail);
                        if added < best.0 {
                            best = (added, Some((pos, (piece.0, reversed))));
                        }
                    }
                }
                if let Some((pos, placed)) = best.1 {
                    rest.insert(pos, placed);
                    sequence = rest;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }
        sequence
    }
}

/// Pieces of a route that share no vertex, ordered and turned so the straight joins between them are as short
/// as possible in total, then joined into the parts of the route in travel order. A join longer than
/// `max_join` is never drawn: the route is cut there into separate parts. The joins are the same either way
/// round, so the member order decides which way the route runs. A straight join runs at the speed of the
/// road it leaves.
fn link_pieces(pieces: Vec<Piece>, max_join: f64) -> Vec<Piece> {
    let pieces: Vec<Piece> = pieces
        .into_iter()
        .filter(|p| !p.coords.is_empty())
        .collect();
    if pieces.len() <= 1 {
        return pieces;
    }
    let n = pieces.len();
    let endpoints: Vec<Xy> = pieces
        .iter()
        .map(|p| p.coords[0])
        .chain(pieces.iter().map(|p| p.coords[p.coords.len() - 1]))
        .collect();
    let mut distances = vec![0.0; 4 * n * n];
    for (a, &pa) in endpoints.iter().enumerate() {
        for (b, &pb) in endpoints.iter().enumerate() {
            distances[a * 2 * n + b] = distance(pa, pb);
        }
    }
    let joins = Joins { n, distances };
    let lengths: Vec<f64> = pieces.iter().map(|p| line_length(&p.coords)).collect();

    let mut sequence = joins.improve(joins.insert_cheapest(&lengths));

    let order: Vec<i64> = sequence.iter().map(|&(i, _)| i as i64).collect();
    let mut concordance = 0i64;
    for k in 0..order.len() {
        for &b in &order[k + 1..] {
            concordance += (b - order[k]).signum();
        }
    }
    let turned = sequence.iter().filter(|p| p.1).count();
    if concordance < 0 || (concordance == 0 && 2 * turned > sequence.len()) {
        sequence = sequence.iter().rev().map(|&(i, r)| (i, !r)).collect();
    }

    let mut parts: Vec<(Vec<Xy>, Vec<f64>)> = Vec::new();
    let (mut coords_acc, mut speeds_acc): (Vec<Xy>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut previous: Option<Placed> = None;
    for &piece in &sequence {
        let (i, reversed) = piece;
        let source = &pieces[i];
        let segment_speeds = &source.speeds[..source.speeds.len().saturating_sub(1)];
        let (coords, speeds): (Vec<Xy>, Vec<f64>) = if reversed {
            (
                source.coords.iter().rev().copied().collect(),
                segment_speeds.iter().rev().copied().collect(),
            )
        } else {
            (source.coords.clone(), segment_speeds.to_vec())
        };

        if let Some(prev) = previous {
            if joins.at(piece_ends(prev, n).1, piece_ends(piece, n).0) > max_join {
                parts.push((
                    std::mem::take(&mut coords_acc),
                    std::mem::take(&mut speeds_acc),
                ));
            }
        }
        let mut skip = 0;
        if !coords_acc.is_empty() && coords_acc[coords_acc.len() - 1] == coords[0] {
            skip = 1;
        } else if !coords_acc.is_empty() {
            let join_speed = match speeds_acc.last() {
                Some(&s) => s,
                None => speeds.first().copied().unwrap_or(f64::NAN),
            };
            speeds_acc.push(join_speed);
        }
        coords_acc.extend_from_slice(&coords[skip..]);
        speeds_acc.extend_from_slice(&speeds);
        previous = Some(piece);
    }
    parts.push((coords_acc, speeds_acc));

    // One speed per vertex: the last vertex repeats the segment before it.
    parts
        .into_iter()
        .map(|(coords, mut speeds)| {
            speeds.push(speeds.last().copied().unwrap_or(f64::NAN));
            Piece { coords, speeds }
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------
// Stops and platforms
// ---------------------------------------------------------------------------------------------------------

/// A stop and its platform on a route. A stop without a platform gets one made up beside the path, a platform
/// without a stop is its own stop (`s` is None).
#[derive(Clone, Copy, Debug)]
struct Pair {
    p: Xy,
    pref: i64,
    s: Option<Xy>,
    sref: i64,
    rank: i64,
}

impl Pair {
    fn anchor(&self) -> Xy {
        self.s.unwrap_or(self.p)
    }
}

fn min_rank(a: i64, b: i64) -> i64 {
    match (a >= 0, b >= 0) {
        (true, true) => a.min(b),
        (true, false) => a,
        (false, true) => b,
        (false, false) => -1,
    }
}

/// Platforms paired with stops: each platform looks at its two nearest stops within `THRESHOLD_METERS`, and
/// the closest pairs are taken first. Returns the pairs in the order they were taken and the matched flags.
fn pair_stops(
    platforms: &[Member],
    stops: &[Member],
) -> (Vec<(usize, usize)>, Vec<bool>, Vec<bool>) {
    let mut matched_p = vec![false; platforms.len()];
    let mut matched_s = vec![false; stops.len()];
    let mut pairs = Vec::new();
    if platforms.is_empty() || stops.is_empty() {
        return (pairs, matched_p, matched_s);
    }
    let k = stops.len().min(2);
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    let mut nearest: Vec<(f64, usize)> = Vec::with_capacity(stops.len());
    for (p_i, platform) in platforms.iter().enumerate() {
        nearest.clear();
        nearest.extend(
            stops
                .iter()
                .enumerate()
                .map(|(s_i, stop)| (distance(platform.xy, stop.xy), s_i)),
        );
        nearest.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for &(dist, s_i) in nearest.iter().take(k) {
            if dist.is_finite() && dist <= THRESHOLD_METERS {
                candidates.push((dist, p_i, s_i));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, p_i, s_i) in candidates {
        if !matched_p[p_i] && !matched_s[s_i] {
            pairs.push((p_i, s_i));
            matched_p[p_i] = true;
            matched_s[s_i] = true;
        }
    }
    (pairs, matched_p, matched_s)
}

/// Pairs grouped by the path their stop lies nearest, each group in travel order along its path (sorted by
/// part index). A path is reversed when its stops clearly run against the relation's stop order: at least
/// three stops within `THRESHOLD_METERS` of it vote, and a majority of their pairs disagree with it.
fn pairs_by_part(pairs: Vec<Pair>, paths: &mut [Vec<Xy>]) -> (Vec<(usize, Vec<Pair>)>, Vec<bool>) {
    let mut groups: BTreeMap<usize, Vec<Pair>> = BTreeMap::new();
    for pair in pairs {
        groups
            .entry(nearest_part(pair.anchor(), paths))
            .or_default()
            .push(pair);
    }
    let mut reversed = vec![false; paths.len()];
    let mut out = Vec::with_capacity(groups.len());
    for (k, mut items) in groups {
        let known: Vec<(f64, i64)> = items
            .iter()
            .filter(|item| {
                item.rank >= 0 && line_distance(&paths[k], item.anchor()) <= THRESHOLD_METERS
            })
            .map(|item| (project(&paths[k], item.anchor()), item.rank))
            .collect();
        let (mut votes, mut decided) = (0.0, 0usize);
        for (i, &(a1, r1)) in known.iter().enumerate() {
            for &(a2, r2) in &known[i + 1..] {
                let vote = sign(a2 - a1) * ((r2 - r1).signum() as f64);
                if vote != 0.0 {
                    decided += 1;
                }
                votes += vote;
            }
        }
        if known.len() >= 3 && decided > 0 && votes <= -0.5 * decided as f64 {
            paths[k].reverse();
            reversed[k] = true;
        }
        let path = &paths[k];
        let mut keyed: Vec<(f64, Pair)> = items
            .drain(..)
            .map(|item| (project(path, item.p), item))
            .collect();
        keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        out.push((k, keyed.into_iter().map(|(_, item)| item).collect()));
    }
    (out, reversed)
}

// ---------------------------------------------------------------------------------------------------------
// One route, all routes
// ---------------------------------------------------------------------------------------------------------

/// Nodes and edges of one route; nothing for a route without ways or without stops and platforms, and
/// nothing for a route left with a single stop (a fragment of a route mostly outside the data): boarding there
/// leads nowhere.
fn parse_route(route: &Route) -> Result<Parsed, String> {
    let mut out = Parsed::default();
    if route.ways.is_empty() || (route.stops.is_empty() && route.platforms.is_empty()) {
        return Ok(out);
    }
    let pieces = drop_closed_loops(stitch(&route.ways));
    let parts = if pieces.len() > 1 {
        link_pieces(pieces, MAX_JOIN_METERS)
    } else {
        pieces
    };
    if parts.is_empty() || parts.iter().any(|part| part.coords.len() < 2) {
        return Err("no connected ways".into());
    }
    let mut paths: Vec<Vec<Xy>> = parts.iter().map(|part| part.coords.clone()).collect();
    let (stops, platforms) = (&route.stops, &route.platforms);

    let (matched, matched_p, matched_s) = pair_stops(platforms, stops);
    let mut pairs: Vec<Pair> = matched
        .iter()
        .map(|&(p_i, s_i)| Pair {
            p: platforms[p_i].xy,
            pref: member_id(platforms[p_i].reference),
            s: Some(stops[s_i].xy),
            sref: member_id(stops[s_i].reference),
            rank: min_rank(stops[s_i].rank, platforms[p_i].rank),
        })
        .collect();

    if !stops.is_empty() {
        let mut side = 1;
        if !platforms.is_empty() {
            let middle = platforms[platforms.len() / 2].xy;
            side = side_of(middle, &paths[nearest_part(middle, &paths)]);
        }
        for (s_i, stop) in stops.iter().enumerate() {
            if matched_s[s_i] {
                continue;
            }
            let mut p = offset_point(stop.xy, &paths[nearest_part(stop.xy, &paths)], side)?;
            let mut pref = made_up_id(stop.reference);
            if stop.stand_in >= 0 {
                pref = member_id(stop.stand_in);
                if !stop.stand_in_xy[0].is_nan() {
                    p = stop.stand_in_xy;
                }
            }
            pairs.push(Pair {
                p,
                pref,
                s: Some(stop.xy),
                sref: member_id(stop.reference),
                rank: stop.rank,
            });
        }
    }
    for (p_i, platform) in platforms.iter().enumerate() {
        if matched_p[p_i] {
            continue;
        }
        pairs.push(Pair {
            p: platform.xy,
            pref: member_id(platform.reference),
            s: None,
            sref: made_up_id(platform.reference),
            rank: platform.rank,
        });
    }

    let mut seen: HashSet<(i64, bool)> = HashSet::new();
    let kinds_apart = route.kinds_apart;
    let mut add_node = |nodes: &mut Vec<Node>, id: i64, xy: Xy, platform: bool| {
        if seen.insert((id, kinds_apart && platform)) {
            nodes.push(Node { id, xy, platform });
        }
    };

    let (groups, reversed) = pairs_by_part(pairs, &mut paths);
    for (k, items) in groups {
        let path = &paths[k];
        let cumdist = hypot_distances(path);
        let vertex_dist = vertex_distances(path);
        let speeds = &parts[k].speeds;
        let mut seg_speeds: Vec<f64> = speeds[..speeds.len() - 1].to_vec();
        if reversed[k] {
            seg_speeds.reverse();
        }
        let mut last: Option<(f64, i64, Xy)> = None;
        for item in items {
            let platform = item.p;
            let stop = item.anchor();
            let stop_dist = project(path, stop);
            let projected = interpolate(path, stop_dist);
            // A platform this far from its stop on the path is an error in the data.
            if distance(projected, platform) > THRESHOLD_METERS {
                continue;
            }
            add_node(&mut out.nodes, item.sref, projected, false);
            if let Some((last_dist, last_id, last_point)) = last {
                out.edges.push(Edge {
                    u: last_id,
                    v: item.sref,
                    boarding: false,
                    coords: cut_line(
                        path,
                        &vertex_dist,
                        last_dist,
                        stop_dist,
                        last_point,
                        projected,
                    ),
                    speed: speed_on_interval(last_dist, stop_dist, &cumdist, &seg_speeds),
                });
            }
            last = Some((stop_dist, item.sref, projected));
            add_node(&mut out.nodes, item.pref, platform, true);
            out.edges.push(Edge {
                u: item.sref,
                v: item.pref,
                boarding: true,
                coords: vec![projected, platform],
                speed: f64::NAN,
            });
        }
    }
    if out.edges.iter().all(|edge| edge.boarding) {
        return Ok(Parsed::default());
    }
    Ok(out)
}

/// Every route, in parallel; the results in route order.
pub fn parse_routes(routes: &[Route]) -> Vec<Result<Parsed, String>> {
    routes.par_iter().map(parse_route).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(coords: &[Xy], speeds: Option<&[f64]>) -> Piece {
        Piece {
            coords: coords.to_vec(),
            speeds: speeds.map_or_else(|| vec![f64::NAN; coords.len()], <[f64]>::to_vec),
        }
    }

    fn longest_step(coords: &[Xy]) -> f64 {
        coords
            .windows(2)
            .map(|w| distance(w[0], w[1]))
            .fold(0.0, f64::max)
    }

    #[test]
    fn a_detour_goes_between_the_pieces_around_it() {
        let before = piece(&[[-5000.0, 0.0], [0.0, 0.0]], Some(&[1.0, 1.0]));
        let detour = piece(&[[100.0, 300.0], [50.0, 60.0]], Some(&[2.0, 2.0]));
        let after = piece(&[[0.0, 0.0], [5000.0, 0.0]], Some(&[3.0, 3.0]));
        let parts = link_pieces(vec![before, detour, after], MAX_JOIN_METERS);
        assert_eq!(parts.len(), 1);
        let coords = &parts[0].coords;
        assert_eq!(coords[..2], [[-5000.0, 0.0], [0.0, 0.0]]);
        let mut middle = coords[2..4].to_vec();
        middle.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(middle, [[50.0, 60.0], [100.0, 300.0]]);
        assert_eq!(coords[4..], [[0.0, 0.0], [5000.0, 0.0]]);
        assert!(longest_step(&coords[1..5]) < 400.0);
        // One speed per vertex, each for the segment leaving it; a straight join keeps the road it leaves.
        assert_eq!(parts[0].speeds, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn pieces_run_in_member_order() {
        let first = piece(&[[0.0, 0.0], [1000.0, 0.0]], None);
        let second = piece(&[[1050.0, 0.0], [2000.0, 0.0]], None);
        let parts = link_pieces(vec![second, first], MAX_JOIN_METERS);
        assert_eq!(
            parts[0].coords,
            [[2000.0, 0.0], [1050.0, 0.0], [1000.0, 0.0], [0.0, 0.0]]
        );
    }

    #[test]
    fn a_hole_cuts_the_route() {
        let near = piece(&[[0.0, 0.0], [1000.0, 0.0]], None);
        let close = piece(&[[1900.0, 0.0], [3000.0, 0.0]], None);
        let far = piece(&[[4100.0, 0.0], [5000.0, 0.0]], None);
        let parts = link_pieces(vec![near, close, far], MAX_JOIN_METERS);
        let coords: Vec<Vec<Xy>> = parts.into_iter().map(|p| p.coords).collect();
        // 900 m is still a way OSM missed; 1100 m is not.
        assert_eq!(
            coords,
            [
                vec![[0.0, 0.0], [1000.0, 0.0], [1900.0, 0.0], [3000.0, 0.0]],
                vec![[4100.0, 0.0], [5000.0, 0.0]],
            ]
        );
    }

    #[test]
    fn stitching_follows_the_members_when_the_first_way_is_drawn_backwards() {
        let a = [[10.0, 0.0], [0.0, 0.0]];
        let b = [[10.0, 0.0], [20.0, 0.0]];
        let ways = [
            Way {
                coords: &a,
                speed: 1.0,
            },
            Way {
                coords: &b,
                speed: 2.0,
            },
        ];
        let pieces = stitch(&ways);
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].coords, [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]);
        assert_eq!(pieces[0].speeds, [1.0, 1.0, 2.0]);
    }

    #[test]
    fn project_and_interpolate_walk_the_line() {
        let line = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]];
        assert_eq!(project(&line, [5.0, 3.0]), 5.0);
        assert_eq!(project(&line, [12.0, 4.0]), 14.0);
        assert_eq!(project(&line, [-3.0, -3.0]), 0.0);
        assert_eq!(project(&line, [20.0, 20.0]), 20.0);
        assert_eq!(interpolate(&line, 5.0), [5.0, 0.0]);
        assert_eq!(interpolate(&line, 14.0), [10.0, 4.0]);
        assert_eq!(interpolate(&line, 99.0), [10.0, 10.0]);
        assert_eq!(interpolate(&line, 0.0), [0.0, 0.0]);
        assert_eq!(line_distance(&line, [5.0, 3.0]), 3.0);
    }

    #[test]
    fn cut_line_keeps_the_vertices_between_the_cuts() {
        let line = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [20.0, 10.0]];
        let vd = vertex_distances(&line);
        let forward = cut_line(&line, &vd, 5.0, 25.0, [5.0, 0.0], [15.0, 10.0]);
        assert_eq!(
            forward,
            [[5.0, 0.0], [10.0, 0.0], [10.0, 10.0], [15.0, 10.0]]
        );
        let backward = cut_line(&line, &vd, 25.0, 5.0, [15.0, 10.0], [5.0, 0.0]);
        assert_eq!(
            backward,
            [[15.0, 10.0], [10.0, 10.0], [10.0, 0.0], [5.0, 0.0]]
        );
        // A cut exactly at a vertex does not repeat it.
        let at_vertex = cut_line(&line, &vd, 10.0, 20.0, [10.0, 0.0], [10.0, 10.0]);
        assert_eq!(at_vertex, [[10.0, 0.0], [10.0, 10.0]]);
        assert_eq!(
            cut_line(&line, &vd, 3.0, 3.0, [3.0, 0.0], [3.0, 0.0]),
            [[3.0, 0.0], [3.0, 0.0]]
        );
    }

    #[test]
    fn speeds_are_weighted_by_length() {
        let cumdist = [0.0, 10.0, 30.0];
        let speeds = [1.0, 4.0];
        assert_eq!(
            speed_on_interval(5.0, 20.0, &cumdist, &speeds),
            (5.0 + 40.0) / 15.0
        );
        assert_eq!(
            speed_on_interval(20.0, 5.0, &cumdist, &speeds),
            (5.0 + 40.0) / 15.0
        );
        assert!(speed_on_interval(5.0, 5.0, &cumdist, &speeds).is_nan());
    }

    #[test]
    fn a_route_pairs_stops_with_platforms_and_makes_up_the_missing_ones() {
        let way = [[0.0, 0.0], [1000.0, 0.0]];
        let route = Route {
            ways: vec![Way {
                coords: &way,
                speed: 10.0,
            }],
            stops: vec![
                Member::new([100.0, 0.0], 0, 0),
                Member::new([900.0, 0.0], 1, 2),
            ],
            platforms: vec![Member::new([100.0, 10.0], 2, 1)],
            kinds_apart: false,
        };
        let parsed = parse_route(&route).unwrap();
        let ids: Vec<(i64, bool)> = parsed.nodes.iter().map(|n| (n.id, n.platform)).collect();
        assert_eq!(
            ids,
            [
                (member_id(0), false),
                (member_id(2), true),
                (member_id(1), false),
                (made_up_id(1), true)
            ]
        );
        let travel: Vec<&Edge> = parsed.edges.iter().filter(|e| !e.boarding).collect();
        assert_eq!(travel.len(), 1);
        assert_eq!((travel[0].u, travel[0].v), (member_id(0), member_id(1)));
        assert_eq!(travel[0].coords, [[100.0, 0.0], [900.0, 0.0]]);
        assert_eq!(travel[0].speed, 10.0);
        // The made-up platform lies on the side of the real one, 7 m off the path.
        let made_up = parsed.nodes.iter().find(|n| n.id == made_up_id(1)).unwrap();
        assert_eq!(made_up.xy, [900.0, 7.0]);
    }

    #[test]
    fn a_stop_takes_the_platform_of_its_stop_area_instead_of_a_made_up_one() {
        let way = [[0.0, 0.0], [1000.0, 0.0]];
        let mut lone = Member::new([500.0, 0.0], 0, 0);
        lone.stand_in = 5;
        lone.stand_in_xy = [500.0, -20.0];
        let mut placed_by_path = Member::new([900.0, 0.0], 1, 1);
        placed_by_path.stand_in = 6;
        let route = Route {
            ways: vec![Way {
                coords: &way,
                speed: 1.0,
            }],
            stops: vec![lone, placed_by_path],
            platforms: vec![],
            kinds_apart: false,
        };
        let parsed = parse_route(&route).unwrap();
        let platforms: Vec<(i64, Xy)> = parsed
            .nodes
            .iter()
            .filter(|n| n.platform)
            .map(|n| (n.id, n.xy))
            .collect();
        // A stand-in without a point keeps the point made up beside the path.
        assert_eq!(
            platforms,
            [(member_id(5), [500.0, -20.0]), (member_id(6), [900.0, 7.0])]
        );
    }

    #[test]
    fn kinds_apart_keeps_a_stop_and_a_platform_with_one_id() {
        let way = [[0.0, 0.0], [1000.0, 0.0]];
        let mut stop = Member::new([500.0, 0.0], 0, 0);
        stop.stand_in = 0; // the stop node is also its own stop area's platform
        let mut next = Member::new([900.0, 0.0], 1, 1);
        next.stand_in = 1;
        let parse = |kinds_apart| {
            let route = Route {
                ways: vec![Way {
                    coords: &way,
                    speed: 1.0,
                }],
                stops: vec![stop, next],
                platforms: vec![],
                kinds_apart,
            };
            parse_route(&route).unwrap().nodes.len()
        };
        assert_eq!(parse(false), 2);
        assert_eq!(parse(true), 4);
    }

    #[test]
    fn a_route_with_a_single_stop_gives_nothing() {
        let way = [[0.0, 0.0], [1000.0, 0.0]];
        let route = Route {
            ways: vec![Way {
                coords: &way,
                speed: 1.0,
            }],
            stops: vec![Member::new([500.0, 0.0], 0, 0)],
            platforms: vec![Member::new([500.0, 8.0], 1, 1)],
            kinds_apart: false,
        };
        assert_eq!(parse_route(&route).unwrap(), Parsed::default());
    }

    #[test]
    fn a_route_without_stops_or_ways_gives_nothing() {
        let way = [[0.0, 0.0], [1000.0, 0.0]];
        let without_stops = Route {
            ways: vec![Way {
                coords: &way,
                speed: 1.0,
            }],
            stops: vec![],
            platforms: vec![],
            kinds_apart: false,
        };
        assert_eq!(parse_route(&without_stops).unwrap(), Parsed::default());
        let without_ways = Route {
            ways: vec![],
            stops: vec![Member::new([0.0, 0.0], 0, 0)],
            platforms: vec![],
            kinds_apart: false,
        };
        assert_eq!(parse_route(&without_ways).unwrap(), Parsed::default());
    }

    #[test]
    fn centroid_weights_segments_by_length() {
        assert_eq!(
            centroid(&[[0.0, 0.0], [10.0, 0.0], [10.0, 0.0]]),
            Some([5.0, 0.0])
        );
        assert_eq!(centroid(&[[3.0, 4.0]]), Some([3.0, 4.0]));
        assert_eq!(centroid(&[[3.0, 4.0], [3.0, 4.0]]), Some([3.0, 4.0]));
        assert_eq!(centroid(&[]), None);
    }
}
