//! Reading ways from an OpenStreetMap `.pbf` file with an Overpass tag filter.
//!
//! Two passes over the file, each decoding blocks in parallel: the first keeps the ways whose tags pass the
//! filter (ids, node refs, tags), the second looks up the locations of exactly the nodes those ways use.
//! Locations are integers in 10⁻⁷ degrees as libosmium stores them, so `lon = x / 1e7` gives the same
//! doubles pyosmium returns. Ways keep the order of the file; a node without a location in the file keeps
//! its ref but adds no coordinate, as in libosmium.

use std::path::Path;

use osmpbf::{BlobDecode, BlobReader, PrimitiveBlock};
use rayon::prelude::*;
use regex::Regex;

/// One bracketed clause of an Overpass tag filter (see `iduedu/osm_pbf/filters.py`).
pub enum Clause {
    Exists(String),
    Absent(String),
    Equals(String, String),
    NotEquals(String, String),
    Matches(String, Regex),
    NotMatches(String, Regex),
}

impl Clause {
    /// Build a clause from the `(key, op, value, ignore_case)` of `TagClause`.
    pub fn new(
        key: String,
        op: &str,
        value: Option<String>,
        ignore_case: bool,
    ) -> Result<Clause, String> {
        let value_of =
            |value: Option<String>| value.ok_or_else(|| format!("operator {op:?} needs a value"));
        Ok(match op {
            "exists" => Clause::Exists(key),
            "absent" => Clause::Absent(key),
            "=" => Clause::Equals(key, value_of(value)?),
            "!=" => Clause::NotEquals(key, value_of(value)?),
            "~" | "!~" => {
                let pattern = value_of(value)?;
                let pattern = if ignore_case {
                    format!("(?i){pattern}")
                } else {
                    pattern
                };
                let regex =
                    Regex::new(&pattern).map_err(|e| format!("invalid regular expression: {e}"))?;
                if op == "~" {
                    Clause::Matches(key, regex)
                } else {
                    Clause::NotMatches(key, regex)
                }
            }
            other => return Err(format!("unknown operator {other:?}")),
        })
    }

    fn matches(&self, tags: &[(&str, &str)]) -> bool {
        let get = |key: &str| tags.iter().find(|(k, _)| *k == key).map(|(_, v)| *v);
        match self {
            Clause::Exists(k) => get(k).is_some(),
            Clause::Absent(k) => get(k).is_none(),
            Clause::Equals(k, v) => get(k) == Some(v.as_str()),
            Clause::NotEquals(k, v) => get(k) != Some(v.as_str()),
            Clause::Matches(k, re) => get(k).is_some_and(|value| re.is_match(value)),
            Clause::NotMatches(k, re) => get(k).is_none_or(|value| !re.is_match(value)),
        }
    }
}

/// Ways read from a file, flattened. Way `i` has node refs `refs[ref_offsets[i]..ref_offsets[i + 1]]`,
/// coordinates `lon/lat[coord_offsets[i]..coord_offsets[i + 1]]` (10⁻⁷ degrees) and tags `tags[i]`.
#[derive(Default)]
pub struct Ways {
    pub ids: Vec<i64>,
    pub ref_offsets: Vec<i64>,
    pub refs: Vec<i64>,
    pub coord_offsets: Vec<i64>,
    pub lon: Vec<i32>,
    pub lat: Vec<i32>,
    pub tags: Vec<Vec<(String, String)>>,
}

struct BlockWays {
    ids: Vec<i64>,
    refs: Vec<Vec<i64>>,
    tags: Vec<Vec<(String, String)>>,
}

fn open(path: &Path) -> Result<BlobReader<std::io::BufReader<std::fs::File>>, String> {
    BlobReader::from_path(path).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

/// Decode the data blocks of the file in parallel; `f(index, block)` results come back in file order.
fn map_blocks<T: Send>(
    path: &Path,
    f: impl Fn(&PrimitiveBlock) -> T + Sync + Send,
) -> Result<Vec<T>, String> {
    let reader = open(path)?;
    let mut results: Vec<(usize, Result<Option<T>, String>)> = reader
        .enumerate()
        .par_bridge()
        .map(|(index, blob)| {
            let decoded = blob
                .map_err(|e| e.to_string())
                .and_then(|blob| match blob.decode() {
                    Ok(BlobDecode::OsmData(block)) => Ok(Some(f(&block))),
                    Ok(_) => Ok(None),
                    Err(e) => Err(e.to_string()),
                });
            (index, decoded)
        })
        .collect();
    results.sort_unstable_by_key(|(index, _)| *index);
    let mut out = Vec::with_capacity(results.len());
    for (_, result) in results {
        if let Some(value) = result? {
            out.push(value);
        }
    }
    Ok(out)
}

fn filtered_ways(block: &PrimitiveBlock, clauses: &[Clause]) -> BlockWays {
    let mut found = BlockWays {
        ids: Vec::new(),
        refs: Vec::new(),
        tags: Vec::new(),
    };
    let mut tags: Vec<(&str, &str)> = Vec::new();
    for group in block.groups() {
        for way in group.ways() {
            tags.clear();
            tags.extend(way.tags());
            if clauses.iter().all(|clause| clause.matches(&tags)) {
                found.ids.push(way.id());
                found.refs.push(way.refs().collect());
                found.tags.push(
                    tags.iter()
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .collect(),
                );
            }
        }
    }
    found
}

/// Locations of the nodes in `wanted` (sorted, unique) found in one block, as `(position in wanted, lon, lat)`.
fn wanted_locations(block: &PrimitiveBlock, wanted: &[i64]) -> Vec<(usize, i32, i32)> {
    let mut found = Vec::new();
    for group in block.groups() {
        for node in group.dense_nodes() {
            if let Ok(position) = wanted.binary_search(&node.id()) {
                found.push((position, node.decimicro_lon(), node.decimicro_lat()));
            }
        }
        for node in group.nodes() {
            if let Ok(position) = wanted.binary_search(&node.id()) {
                found.push((position, node.decimicro_lon(), node.decimicro_lat()));
            }
        }
    }
    found
}

/// Ways passing every clause, with the coordinates of their nodes, in file order.
pub fn read_ways(path: &Path, clauses: &[Clause]) -> Result<Ways, String> {
    let blocks = map_blocks(path, |block| filtered_ways(block, clauses))?;

    let mut wanted: Vec<i64> = blocks
        .iter()
        .flat_map(|b| b.refs.iter().flatten().copied())
        .collect();
    wanted.sort_unstable();
    wanted.dedup();

    const MISSING: i32 = i32::MIN;
    let mut lon = vec![MISSING; wanted.len()];
    let mut lat = vec![MISSING; wanted.len()];
    if !wanted.is_empty() {
        for found in map_blocks(path, |block| wanted_locations(block, &wanted))? {
            for (position, x, y) in found {
                lon[position] = x;
                lat[position] = y;
            }
        }
    }

    let mut ways = Ways::default();
    ways.ref_offsets.push(0);
    ways.coord_offsets.push(0);
    for block in blocks {
        for ((id, refs), tags) in block.ids.into_iter().zip(block.refs).zip(block.tags) {
            for node in &refs {
                let position = wanted.binary_search(node).expect("every ref is wanted");
                if lon[position] != MISSING {
                    ways.lon.push(lon[position]);
                    ways.lat.push(lat[position]);
                }
            }
            ways.ids.push(id);
            ways.refs.extend_from_slice(&refs);
            ways.ref_offsets.push(ways.refs.len() as i64);
            ways.coord_offsets.push(ways.lon.len() as i64);
            ways.tags.push(tags);
        }
    }
    Ok(ways)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clauses_follow_overpass_semantics() {
        let tags = [("highway", "footway"), ("access", "Private")];
        let clause = |key: &str, op: &str, value: Option<&str>, i: bool| {
            Clause::new(key.into(), op, value.map(String::from), i).unwrap()
        };
        assert!(clause("highway", "exists", None, false).matches(&tags));
        assert!(clause("area", "absent", None, false).matches(&tags));
        assert!(clause("highway", "=", Some("footway"), false).matches(&tags));
        assert!(clause("area", "!=", Some("yes"), false).matches(&tags));
        assert!(clause("highway", "~", Some("foot|path"), false).matches(&tags));
        assert!(clause("area", "!~", Some("yes"), false).matches(&tags));
        assert!(clause("access", "!~", Some("private"), false).matches(&tags));
        assert!(!clause("access", "!~", Some("private"), true).matches(&tags));
        assert!(!clause("highway", "!~", Some("foot"), false).matches(&tags));
        assert!(Clause::new("k".into(), "~", Some("(".into()), false).is_err());
    }
}
