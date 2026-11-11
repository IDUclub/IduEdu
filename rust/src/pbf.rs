//! Reading OpenStreetMap `.pbf` files: the queries of `iduedu/osm_pbf`, without libosmium.
//!
//! A [`PbfFile`] decodes the data blocks of a file in parallel. Its first pass also records, for every
//! block, the byte offset and the id ranges of the nodes, ways and relations in it; later queries on the
//! same `PbfFile` read only the blocks that can hold what they look for, so a relation query skips the
//! node and way blocks, and a lookup of node ids skips the blocks whose id range holds none of them.
//!
//! Locations are integers in 10⁻⁷ degrees, as the file stores them and libosmium keeps them, so
//! `lon = x / 1e7` gives the same doubles pyosmium returns. Results keep the order of the file.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use osmpbf::{BlobDecode, BlobReader, ByteOffset, PrimitiveBlock, RelMemberType};
use rayon::prelude::*;
use regex::Regex;

/// Coordinate of a node that is not in the file.
pub const MISSING: i32 = i32::MIN;

/// Blocks decoded at once; bounds the memory of a pass over a large file.
const CHUNK: usize = 256;

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

/// Element type, also the member type code of relations (`0` node, `1` way, `2` relation).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Node = 0,
    Way = 1,
    Relation = 2,
}

pub type Tags = Vec<(String, String)>;

fn owned(tags: &[(&str, &str)]) -> Tags {
    tags.iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// `true` if the sorted `ids` hold a value in `lo..=hi`.
fn holds_any(ids: &[i64], lo: i64, hi: i64) -> bool {
    let i = ids.partition_point(|&x| x < lo);
    i < ids.len() && ids[i] <= hi
}

/// Walks increasing ids against the sorted `wanted`, yielding the position of each id found in it.
struct Lookup<'a> {
    wanted: &'a [i64],
    cursor: usize,
    last: i64,
}

impl<'a> Lookup<'a> {
    fn new(wanted: &'a [i64]) -> Self {
        Lookup {
            wanted,
            cursor: 0,
            last: i64::MAX,
        }
    }

    fn find(&mut self, id: i64) -> Option<usize> {
        if id < self.last || self.cursor == 0 {
            // The first id, or ids out of order: start again from a binary search.
            self.cursor = self.wanted.partition_point(|&x| x < id);
        } else {
            // Usually a step or two; a binary search over the rest when the gap is long.
            let rest = &self.wanted[self.cursor..];
            self.cursor += if rest.len() > 16 && rest[16] < id {
                rest.partition_point(|&x| x < id)
            } else {
                rest.iter().take_while(|&&x| x < id).count()
            };
        }
        self.last = id;
        (self.cursor < self.wanted.len() && self.wanted[self.cursor] == id).then_some(self.cursor)
    }
}

/// Byte offset of a data block and the id range of each element type in it.
#[derive(Clone, Debug)]
struct BlockInfo {
    offset: u64,
    ranges: [Option<(i64, i64)>; 3],
}

impl BlockInfo {
    fn of(offset: u64, block: &PrimitiveBlock) -> BlockInfo {
        let mut ranges: [Option<(i64, i64)>; 3] = [None; 3];
        let mut add = |kind: Kind, id: i64| {
            let range = &mut ranges[kind as usize];
            *range = Some(match *range {
                Some((lo, hi)) => (lo.min(id), hi.max(id)),
                None => (id, id),
            });
        };
        for group in block.groups() {
            for node in group.dense_nodes() {
                add(Kind::Node, node.id());
            }
            for node in group.nodes() {
                add(Kind::Node, node.id());
            }
            for way in group.ways() {
                add(Kind::Way, way.id());
            }
            for relation in group.relations() {
                add(Kind::Relation, relation.id());
            }
        }
        BlockInfo { offset, ranges }
    }

    /// Can the block hold an element of `kind`, with one of the sorted `ids` if given?
    fn holds(&self, kind: Kind, ids: Option<&[i64]>) -> bool {
        match (self.ranges[kind as usize], ids) {
            (None, _) => false,
            (Some(_), None) => true,
            (Some((lo, hi)), Some(ids)) => holds_any(ids, lo, hi),
        }
    }
}

/// The header block of a file.
#[derive(Clone, Debug, Default)]
pub struct Header {
    /// `(left, bottom, right, top)` in degrees.
    pub bbox: Option<[f64; 4]>,
    pub required_features: Vec<String>,
    pub optional_features: Vec<String>,
    pub writing_program: Option<String>,
    pub source: Option<String>,
    /// Seconds since the epoch.
    pub replication_timestamp: Option<i64>,
    pub replication_sequence_number: Option<i64>,
    pub replication_base_url: Option<String>,
}

/// Ways read from a file, flattened: way `i` has node refs `refs[ref_offsets[i]..ref_offsets[i + 1]]`.
#[derive(Default)]
pub struct Ways {
    pub ids: Vec<i64>,
    pub ref_offsets: Vec<i64>,
    pub refs: Vec<i64>,
    /// Empty when tags were not asked for.
    pub tags: Vec<Tags>,
}

impl Ways {
    fn extend(&mut self, other: Ways) {
        let base = self.refs.len() as i64;
        self.ids.extend(other.ids);
        self.ref_offsets
            .extend(other.ref_offsets.into_iter().skip(1).map(|o| o + base));
        self.refs.extend(other.refs);
        self.tags.extend(other.tags);
    }

    fn new() -> Ways {
        Ways {
            ref_offsets: vec![0],
            ..Ways::default()
        }
    }

    pub fn refs_of(&self, i: usize) -> &[i64] {
        &self.refs[self.ref_offsets[i] as usize..self.ref_offsets[i + 1] as usize]
    }
}

/// Nodes read from a file.
#[derive(Default)]
pub struct Nodes {
    pub ids: Vec<i64>,
    pub lon: Vec<i32>,
    pub lat: Vec<i32>,
    pub tags: Vec<Tags>,
}

/// Relations read from a file, flattened: relation `i` has the members
/// `member_*[member_offsets[i]..member_offsets[i + 1]]`; types are [`Kind`] codes.
#[derive(Default)]
pub struct Relations {
    pub ids: Vec<i64>,
    pub member_offsets: Vec<i64>,
    pub member_types: Vec<u8>,
    pub member_refs: Vec<i64>,
    pub member_roles: Vec<String>,
    pub tags: Vec<Tags>,
}

impl Relations {
    fn new() -> Relations {
        Relations {
            member_offsets: vec![0],
            ..Relations::default()
        }
    }

    fn extend(&mut self, other: Relations) {
        let base = self.member_refs.len() as i64;
        self.ids.extend(other.ids);
        self.member_offsets
            .extend(other.member_offsets.into_iter().skip(1).map(|o| o + base));
        self.member_types.extend(other.member_types);
        self.member_refs.extend(other.member_refs);
        self.member_roles.extend(other.member_roles);
        self.tags.extend(other.tags);
    }
}

/// Which ways [`PbfFile::ways`] returns: those passing every clause, carrying one of `any_keys` (when
/// given) and with an id in the sorted `ids` (when given).
pub struct WayQuery<'a> {
    pub clauses: &'a [Clause],
    pub any_keys: Option<&'a [String]>,
    pub ids: Option<&'a [i64]>,
    pub tags: bool,
}

/// Which relations [`PbfFile::relations`] returns: those with an id in the sorted `ids`, carrying one of
/// the `tags` pairs and having a node member in the sorted `contains_nodes`, each condition when given.
#[derive(Default)]
pub struct RelationQuery<'a> {
    pub ids: Option<&'a [i64]>,
    pub tags: Option<&'a [(String, String)]>,
    pub contains_nodes: Option<&'a [i64]>,
}

/// A `.pbf` file with the block index of its first pass.
pub struct PbfFile {
    path: PathBuf,
    index: Mutex<Option<Arc<Vec<BlockInfo>>>>,
}

fn io_error(path: &Path, e: impl std::fmt::Display) -> String {
    format!("cannot read {}: {e}", path.display())
}

impl PbfFile {
    /// Open a file, reading its header to check that it is an OSM `.pbf` file.
    pub fn open(path: &Path) -> Result<PbfFile, String> {
        let file = PbfFile {
            path: path.to_path_buf(),
            index: Mutex::new(None),
        };
        file.header()?;
        Ok(file)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The header block, the first block of the file.
    pub fn header(&self) -> Result<Header, String> {
        let not_pbf = || format!("{} is not an OSM .pbf file", self.path.display());
        let mut reader = BlobReader::from_path(&self.path).map_err(|e| io_error(&self.path, e))?;
        let blob = reader.next().ok_or_else(not_pbf)?.map_err(|_| not_pbf())?;
        let header = match blob.decode().map_err(|_| not_pbf())? {
            BlobDecode::OsmHeader(header) => header,
            _ => return Err(not_pbf()),
        };
        Ok(Header {
            bbox: header
                .bbox()
                .map(|b| [b.left, b.bottom.min(b.top), b.right, b.top.max(b.bottom)]),
            required_features: header.required_features().to_vec(),
            optional_features: header.optional_features().to_vec(),
            writing_program: header.writing_program().map(String::from),
            source: header.source().map(String::from),
            replication_timestamp: header.osmosis_replication_timestamp(),
            replication_sequence_number: header.osmosis_replication_sequence_number(),
            replication_base_url: header.osmosis_replication_base_url().map(String::from),
        })
    }

    /// Decode, in parallel and in chunks, the blocks that `select` accepts and give `f`'s results to
    /// `sink` in file order. Without an index the whole file is read once and the index is recorded.
    fn scan<T: Send>(
        &self,
        select: impl Fn(&BlockInfo) -> bool + Sync,
        f: impl Fn(&PrimitiveBlock) -> T + Sync + Send,
        mut sink: impl FnMut(T) -> Result<(), String>,
    ) -> Result<(), String> {
        let index = self.index.lock().expect("index lock").clone();
        let building = index.is_none();
        let path = &self.path;
        type Decoded<T> = Result<Option<(Option<BlockInfo>, Option<T>)>, String>;
        let decode = |blob: osmpbf::Blob| -> Decoded<T> {
            let offset = blob.offset().map_or(0, |o| o.0);
            match blob.decode().map_err(|e| io_error(path, e))? {
                BlobDecode::OsmData(block) if building => {
                    let info = BlockInfo::of(offset, &block);
                    let value = select(&info).then(|| f(&block));
                    Ok(Some((Some(info), value)))
                }
                BlobDecode::OsmData(block) => Ok(Some((None, Some(f(&block))))),
                _ => Ok(None),
            }
        };
        let mut process = |chunk: Vec<osmpbf::Blob>, infos: Option<&mut Vec<BlockInfo>>| {
            let decoded: Vec<_> = chunk.into_par_iter().map(&decode).collect();
            let mut infos = infos;
            for item in decoded {
                if let Some((info, value)) = item? {
                    if let (Some(infos), Some(info)) = (infos.as_deref_mut(), info) {
                        infos.push(info);
                    }
                    if let Some(value) = value {
                        sink(value)?;
                    }
                }
            }
            Ok::<(), String>(())
        };

        if let Some(index) = index {
            let mut reader = BlobReader::seekable_from_path(path).map_err(|e| io_error(path, e))?;
            let mut chunk = Vec::with_capacity(CHUNK);
            for info in index.iter().filter(|info| select(info)) {
                chunk.push(
                    reader
                        .blob_from_offset(ByteOffset(info.offset))
                        .map_err(|e| io_error(path, e))?,
                );
                if chunk.len() == CHUNK {
                    process(std::mem::take(&mut chunk), None)?;
                }
            }
            process(chunk, None)?;
        } else {
            let reader = BlobReader::from_path(path).map_err(|e| io_error(path, e))?;
            let mut infos = Vec::new();
            let mut chunk = Vec::with_capacity(CHUNK);
            for blob in reader {
                chunk.push(blob.map_err(|e| io_error(path, e))?);
                if chunk.len() == CHUNK {
                    process(std::mem::take(&mut chunk), Some(&mut infos))?;
                }
            }
            process(chunk, Some(&mut infos))?;
            *self.index.lock().expect("index lock") = Some(Arc::new(infos));
        }
        Ok(())
    }

    /// Results of `f` on every block that can hold elements of `kind` (with one of `ids`), in file order.
    fn collect<T: Send>(
        &self,
        kind: Kind,
        ids: Option<&[i64]>,
        f: impl Fn(&PrimitiveBlock) -> T + Sync + Send,
    ) -> Result<Vec<T>, String> {
        let mut out = Vec::new();
        self.scan(
            |info| info.holds(kind, ids),
            f,
            |value| {
                out.push(value);
                Ok(())
            },
        )?;
        Ok(out)
    }

    /// Ways matching `query`, in file order.
    pub fn ways(&self, query: &WayQuery) -> Result<Ways, String> {
        let blocks = self.collect(Kind::Way, query.ids, |block| {
            let mut found = Ways::new();
            let mut tags: Vec<(&str, &str)> = Vec::new();
            for group in block.groups() {
                for way in group.ways() {
                    if let Some(ids) = query.ids {
                        if ids.binary_search(&way.id()).is_err() {
                            continue;
                        }
                    }
                    tags.clear();
                    tags.extend(way.tags());
                    if let Some(keys) = query.any_keys {
                        if !tags.iter().any(|(k, _)| keys.iter().any(|key| key == k)) {
                            continue;
                        }
                    }
                    if !query.clauses.iter().all(|clause| clause.matches(&tags)) {
                        continue;
                    }
                    found.ids.push(way.id());
                    found.refs.extend(way.refs());
                    found.ref_offsets.push(found.refs.len() as i64);
                    if query.tags {
                        found.tags.push(owned(&tags));
                    }
                }
            }
            found
        })?;
        let mut ways = Ways::new();
        for block in blocks {
            ways.extend(block);
        }
        Ok(ways)
    }

    /// Locations of the sorted, unique node ids `wanted`, aligned with it; [`MISSING`] for absent nodes.
    pub fn locations(&self, wanted: &[i64]) -> Result<(Vec<i32>, Vec<i32>), String> {
        let mut lon = vec![MISSING; wanted.len()];
        let mut lat = vec![MISSING; wanted.len()];
        if wanted.is_empty() {
            return Ok((lon, lat));
        }
        let blocks = self.collect(Kind::Node, Some(wanted), |block| {
            let mut found: Vec<(usize, i32, i32)> = Vec::new();
            for group in block.groups() {
                let mut lookup = Lookup::new(wanted);
                for node in group.dense_nodes() {
                    if let Some(position) = lookup.find(node.id()) {
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
        })?;
        for (position, x, y) in blocks.into_iter().flatten() {
            lon[position] = x;
            lat[position] = y;
        }
        Ok((lon, lat))
    }

    /// The nodes with one of the sorted, unique `ids`, with their tags, in file order.
    pub fn nodes(&self, ids: &[i64]) -> Result<Nodes, String> {
        if ids.is_empty() {
            return Ok(Nodes::default());
        }
        let blocks = self.collect(Kind::Node, Some(ids), |block| {
            let mut found = Nodes::default();
            for group in block.groups() {
                let mut lookup = Lookup::new(ids);
                for node in group.dense_nodes() {
                    if lookup.find(node.id()).is_some() {
                        found.ids.push(node.id());
                        found.lon.push(node.decimicro_lon());
                        found.lat.push(node.decimicro_lat());
                        found.tags.push(
                            node.tags()
                                .map(|(k, v)| (k.to_string(), v.to_string()))
                                .collect(),
                        );
                    }
                }
                for node in group.nodes() {
                    if ids.binary_search(&node.id()).is_ok() {
                        found.ids.push(node.id());
                        found.lon.push(node.decimicro_lon());
                        found.lat.push(node.decimicro_lat());
                        found.tags.push(
                            node.tags()
                                .map(|(k, v)| (k.to_string(), v.to_string()))
                                .collect(),
                        );
                    }
                }
            }
            found
        })?;
        let mut nodes = Nodes::default();
        for block in blocks {
            nodes.ids.extend(block.ids);
            nodes.lon.extend(block.lon);
            nodes.lat.extend(block.lat);
            nodes.tags.extend(block.tags);
        }
        Ok(nodes)
    }

    /// Relations matching `query`, in file order.
    pub fn relations(&self, query: &RelationQuery) -> Result<Relations, String> {
        let blocks = self.collect(Kind::Relation, query.ids, |block| {
            let mut found = Relations::new();
            let mut tags: Vec<(&str, &str)> = Vec::new();
            for group in block.groups() {
                for relation in group.relations() {
                    if let Some(ids) = query.ids {
                        if ids.binary_search(&relation.id()).is_err() {
                            continue;
                        }
                    }
                    tags.clear();
                    tags.extend(relation.tags());
                    if let Some(pairs) = query.tags {
                        if !tags
                            .iter()
                            .any(|(k, v)| pairs.iter().any(|(pk, pv)| pk == k && pv == v))
                        {
                            continue;
                        }
                    }
                    if let Some(nodes) = query.contains_nodes {
                        let hit = relation.members().any(|m| {
                            m.member_type == RelMemberType::Node
                                && nodes.binary_search(&m.member_id).is_ok()
                        });
                        if !hit {
                            continue;
                        }
                    }
                    found.ids.push(relation.id());
                    for member in relation.members() {
                        found.member_types.push(match member.member_type {
                            RelMemberType::Node => Kind::Node as u8,
                            RelMemberType::Way => Kind::Way as u8,
                            RelMemberType::Relation => Kind::Relation as u8,
                        });
                        found.member_refs.push(member.member_id);
                        found
                            .member_roles
                            .push(member.role().unwrap_or_default().to_string());
                    }
                    found.member_offsets.push(found.member_refs.len() as i64);
                    found.tags.push(owned(&tags));
                }
            }
            found
        })?;
        let mut relations = Relations::new();
        for block in blocks {
            relations.extend(block);
        }
        Ok(relations)
    }

    /// Copy the elements of every block that `select` accepts through `f`, giving results to `sink` in
    /// file order (used by the extract writer).
    pub(crate) fn stream<T: Send>(
        &self,
        select: impl Fn(&[Option<(i64, i64)>; 3]) -> bool + Sync,
        f: impl Fn(&PrimitiveBlock) -> T + Sync + Send,
        sink: impl FnMut(T) -> Result<(), String>,
    ) -> Result<(), String> {
        self.scan(|info| select(&info.ranges), f, sink)
    }
}

/// Ways with the coordinates of their nodes: the ways of `query` and, per way, the locations of the nodes
/// found in the file (`coord_offsets` like `ref_offsets`). With `bboxes` (`[minx, miny, maxx, maxy]`
/// in degrees) only ways with a coordinate box meeting one of them are kept.
pub struct WaysWithCoords {
    pub ways: Ways,
    pub coord_offsets: Vec<i64>,
    pub lon: Vec<i32>,
    pub lat: Vec<i32>,
}

pub fn ways_with_coords(
    file: &PbfFile,
    query: &WayQuery,
    bboxes: Option<&[[f64; 4]]>,
) -> Result<WaysWithCoords, String> {
    let ways = file.ways(query)?;
    let mut wanted = ways.refs.clone();
    wanted.sort_unstable();
    wanted.dedup();
    let (lon, lat) = file.locations(&wanted)?;

    let mut out = WaysWithCoords {
        ways: Ways::new(),
        coord_offsets: vec![0],
        lon: Vec::new(),
        lat: Vec::new(),
    };
    let mut tags = ways.tags.into_iter();
    for i in 0..ways.ids.len() {
        let refs = &ways.refs[ways.ref_offsets[i] as usize..ways.ref_offsets[i + 1] as usize];
        let start = out.lon.len();
        for node in refs {
            let position = wanted.binary_search(node).expect("every ref is wanted");
            if lon[position] != MISSING {
                out.lon.push(lon[position]);
                out.lat.push(lat[position]);
            }
        }
        let way_tags = if query.tags { tags.next() } else { None };
        if let Some(bboxes) = bboxes {
            let xs = &out.lon[start..];
            let ys = &out.lat[start..];
            let keep = !xs.is_empty() && {
                let deg = |v: i32| f64::from(v) / 1e7;
                let (minx, maxx) = (
                    deg(*xs.iter().min().unwrap()),
                    deg(*xs.iter().max().unwrap()),
                );
                let (miny, maxy) = (
                    deg(*ys.iter().min().unwrap()),
                    deg(*ys.iter().max().unwrap()),
                );
                bboxes
                    .iter()
                    .any(|b| minx <= b[2] && maxx >= b[0] && miny <= b[3] && maxy >= b[1])
            };
            if !keep {
                out.lon.truncate(start);
                out.lat.truncate(start);
                continue;
            }
        }
        out.ways.ids.push(ways.ids[i]);
        out.ways.refs.extend_from_slice(refs);
        out.ways.ref_offsets.push(out.ways.refs.len() as i64);
        out.coord_offsets.push(out.lon.len() as i64);
        if let Some(t) = way_tags {
            out.ways.tags.push(t);
        }
    }
    Ok(out)
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

    #[test]
    fn lookup_finds_ids_in_and_out_of_order() {
        let wanted = [2, 5, 9, 12];
        let mut lookup = Lookup::new(&wanted);
        let found: Vec<_> = [1, 2, 3, 5, 9, 4, 12, 13]
            .iter()
            .map(|&id| lookup.find(id))
            .collect();
        assert_eq!(
            found,
            vec![None, Some(0), None, Some(1), Some(2), None, Some(3), None]
        );
        assert!(holds_any(&wanted, 3, 5) && !holds_any(&wanted, 6, 8) && !holds_any(&[], 0, 9));
    }
}
