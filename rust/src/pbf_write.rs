//! Writing OpenStreetMap `.pbf` files: per-territory extracts of a larger file and small files from
//! element lists.
//!
//! Blocks hold one element type each, at most 8000 elements, zlib-compressed, nodes as `DenseNodes`
//! with the default granularity (10⁻⁷ degrees). Element metadata (version, timestamp, user) is not
//! written; IduEdu reads none of it.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use flate2::write::ZlibEncoder;
use flate2::Compression;
use osmpbf::{PrimitiveBlock, RelMemberType};

use crate::pbf::{Header, Kind, PbfFile, Tags};

const BLOCK_ELEMENTS: usize = 8000;
/// Flush a block before its uncompressed size could approach the 32 MB limit of the format.
const BLOCK_BYTES: usize = 8 << 20;

/// An element to write. Locations are in 10⁻⁷ degrees; member types are [`Kind`] codes.
pub enum Element {
    Node {
        id: i64,
        lon: i32,
        lat: i32,
        tags: Tags,
    },
    Way {
        id: i64,
        refs: Vec<i64>,
        tags: Tags,
    },
    Relation {
        id: i64,
        members: Vec<(u8, i64, String)>,
        tags: Tags,
    },
}

impl Element {
    fn kind(&self) -> Kind {
        match self {
            Element::Node { .. } => Kind::Node,
            Element::Way { .. } => Kind::Way,
            Element::Relation { .. } => Kind::Relation,
        }
    }

    fn approx_bytes(&self) -> usize {
        let tags = |tags: &Tags| {
            tags.iter()
                .map(|(k, v)| k.len() + v.len() + 4)
                .sum::<usize>()
        };
        match self {
            Element::Node { tags: t, .. } => 16 + tags(t),
            Element::Way { refs, tags: t, .. } => 16 + 10 * refs.len() + tags(t),
            Element::Relation {
                members, tags: t, ..
            } => {
                16 + members
                    .iter()
                    .map(|(_, _, role)| 16 + role.len())
                    .sum::<usize>()
                    + tags(t)
            }
        }
    }
}

// ----------------------------------------------------------------------------- protobuf encoding

fn varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn key(buf: &mut Vec<u8>, field: u32, wire: u32) {
    varint(buf, u64::from((field << 3) | wire));
}

fn varint_field(buf: &mut Vec<u8>, field: u32, value: u64) {
    key(buf, field, 0);
    varint(buf, value);
}

fn bytes_field(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
    key(buf, field, 2);
    varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

/// A packed repeated varint field; nothing is written for no values.
fn packed(
    buf: &mut Vec<u8>,
    field: u32,
    values: impl IntoIterator<Item = u64>,
    scratch: &mut Vec<u8>,
) {
    scratch.clear();
    for value in values {
        varint(scratch, value);
    }
    if !scratch.is_empty() {
        bytes_field(buf, field, scratch);
    }
}

/// Zig-zag deltas of a sequence, as the format stores ids, coordinates and refs.
fn deltas(values: impl IntoIterator<Item = i64>) -> impl Iterator<Item = u64> {
    let mut last = 0i64;
    values.into_iter().map(move |v| {
        let d = v.wrapping_sub(last);
        last = v;
        zigzag(d)
    })
}

#[derive(Default)]
struct StringTable {
    strings: Vec<Vec<u8>>,
    index: HashMap<String, u64>,
}

impl StringTable {
    fn new() -> Self {
        StringTable {
            strings: vec![Vec::new()], // entry 0 is the empty string, never referenced
            index: HashMap::new(),
        }
    }

    fn id(&mut self, s: &str) -> u64 {
        if let Some(&id) = self.index.get(s) {
            return id;
        }
        let id = self.strings.len() as u64;
        self.strings.push(s.as_bytes().to_vec());
        self.index.insert(s.to_string(), id);
        id
    }
}

fn encode_block(elements: &[Element]) -> Vec<u8> {
    let mut table = StringTable::new();
    let mut group = Vec::new();
    let mut scratch = Vec::new();
    match elements.first().map(Element::kind) {
        Some(Kind::Node) => {
            let (mut ids, mut lats, mut lons, mut keys_vals) = (vec![], vec![], vec![], vec![]);
            let mut any_tags = false;
            for element in elements {
                if let Element::Node { id, lon, lat, tags } = element {
                    ids.push(*id);
                    lats.push(i64::from(*lat));
                    lons.push(i64::from(*lon));
                    for (k, v) in tags {
                        keys_vals.push(table.id(k));
                        keys_vals.push(table.id(v));
                        any_tags = true;
                    }
                    keys_vals.push(0);
                }
            }
            let mut dense = Vec::new();
            packed(&mut dense, 1, deltas(ids), &mut scratch);
            packed(&mut dense, 8, deltas(lats), &mut scratch);
            packed(&mut dense, 9, deltas(lons), &mut scratch);
            if any_tags {
                packed(&mut dense, 10, keys_vals, &mut scratch);
            }
            bytes_field(&mut group, 2, &dense);
        }
        Some(Kind::Way) => {
            let mut message = Vec::new();
            for element in elements {
                if let Element::Way { id, refs, tags } = element {
                    message.clear();
                    varint_field(&mut message, 1, *id as u64);
                    let keys: Vec<u64> = tags.iter().map(|(k, _)| table.id(k)).collect();
                    let vals: Vec<u64> = tags.iter().map(|(_, v)| table.id(v)).collect();
                    packed(&mut message, 2, keys, &mut scratch);
                    packed(&mut message, 3, vals, &mut scratch);
                    packed(&mut message, 8, deltas(refs.iter().copied()), &mut scratch);
                    bytes_field(&mut group, 3, &message);
                }
            }
        }
        Some(Kind::Relation) => {
            let mut message = Vec::new();
            for element in elements {
                if let Element::Relation { id, members, tags } = element {
                    message.clear();
                    varint_field(&mut message, 1, *id as u64);
                    let keys: Vec<u64> = tags.iter().map(|(k, _)| table.id(k)).collect();
                    let vals: Vec<u64> = tags.iter().map(|(_, v)| table.id(v)).collect();
                    packed(&mut message, 2, keys, &mut scratch);
                    packed(&mut message, 3, vals, &mut scratch);
                    let roles: Vec<u64> = members.iter().map(|(_, _, r)| table.id(r)).collect();
                    packed(&mut message, 8, roles, &mut scratch);
                    packed(
                        &mut message,
                        9,
                        deltas(members.iter().map(|(_, r, _)| *r)),
                        &mut scratch,
                    );
                    packed(
                        &mut message,
                        10,
                        members.iter().map(|(t, _, _)| u64::from(*t)),
                        &mut scratch,
                    );
                    bytes_field(&mut group, 4, &message);
                }
            }
        }
        None => {}
    }
    let mut strings = Vec::new();
    for s in &table.strings {
        bytes_field(&mut strings, 1, s);
    }
    let mut block = Vec::new();
    bytes_field(&mut block, 1, &strings);
    bytes_field(&mut block, 2, &group);
    block
}

fn encode_header(header: &Header) -> Vec<u8> {
    let mut out = Vec::new();
    if let Some([left, bottom, right, top]) = header.bbox {
        let nano = |deg: f64| zigzag((deg * 1e9).round() as i64);
        let mut bbox = Vec::new();
        varint_field(&mut bbox, 1, nano(left));
        varint_field(&mut bbox, 2, nano(right));
        varint_field(&mut bbox, 3, nano(top));
        varint_field(&mut bbox, 4, nano(bottom));
        // sint64 fields: wire type 0 with zig-zag values, as written above
        bytes_field(&mut out, 1, &bbox);
    }
    for feature in &header.required_features {
        bytes_field(&mut out, 4, feature.as_bytes());
    }
    for feature in &header.optional_features {
        bytes_field(&mut out, 5, feature.as_bytes());
    }
    if let Some(program) = &header.writing_program {
        bytes_field(&mut out, 16, program.as_bytes());
    }
    if let Some(source) = &header.source {
        bytes_field(&mut out, 17, source.as_bytes());
    }
    if let Some(stamp) = header.replication_timestamp {
        varint_field(&mut out, 32, stamp as u64);
    }
    if let Some(sequence) = header.replication_sequence_number {
        varint_field(&mut out, 33, sequence as u64);
    }
    if let Some(url) = &header.replication_base_url {
        bytes_field(&mut out, 34, url.as_bytes());
    }
    out
}

// ----------------------------------------------------------------------------- writer

/// A `.pbf` file being written. Elements must come grouped by type (nodes, ways, relations).
pub struct Writer {
    out: BufWriter<File>,
    path: PathBuf,
    pending: Vec<Element>,
    pending_bytes: usize,
}

impl Writer {
    pub fn create(path: &Path, header: &Header) -> Result<Writer, String> {
        let file =
            File::create(path).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        let mut writer = Writer {
            out: BufWriter::new(file),
            path: path.to_path_buf(),
            pending: Vec::new(),
            pending_bytes: 0,
        };
        writer.write_blob("OSMHeader", &encode_header(header))?;
        Ok(writer)
    }

    fn write_blob(&mut self, kind: &str, data: &[u8]) -> Result<(), String> {
        let error = |e: std::io::Error| format!("cannot write {}: {e}", self.path.display());
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data).map_err(error)?;
        let compressed = encoder.finish().map_err(error)?;
        let mut blob = Vec::with_capacity(compressed.len() + 16);
        varint_field(&mut blob, 2, data.len() as u64);
        bytes_field(&mut blob, 3, &compressed);
        let mut blob_header = Vec::new();
        bytes_field(&mut blob_header, 1, kind.as_bytes());
        varint_field(&mut blob_header, 3, blob.len() as u64);
        self.out
            .write_all(&(blob_header.len() as u32).to_be_bytes())
            .and_then(|()| self.out.write_all(&blob_header))
            .and_then(|()| self.out.write_all(&blob))
            .map_err(error)
    }

    fn flush_block(&mut self) -> Result<(), String> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let block = encode_block(&self.pending);
        self.pending.clear();
        self.pending_bytes = 0;
        self.write_blob("OSMData", &block)
    }

    pub fn add(&mut self, element: Element) -> Result<(), String> {
        let new_kind = self
            .pending
            .first()
            .is_some_and(|first| first.kind() != element.kind());
        if new_kind || self.pending.len() >= BLOCK_ELEMENTS || self.pending_bytes >= BLOCK_BYTES {
            self.flush_block()?;
        }
        self.pending_bytes += element.approx_bytes();
        self.pending.push(element);
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), String> {
        self.flush_block()?;
        let path = self.path.clone();
        self.out
            .flush()
            .map_err(|e| format!("cannot write {}: {e}", path.display()))
    }
}

// ----------------------------------------------------------------------------- extracts

/// One output of [`extract`]: the sorted ids of the nodes, ways and relations to copy, and its header box.
pub struct Output {
    pub path: PathBuf,
    pub ids: [Vec<i64>; 3],
    pub bbox: Option<[f64; 4]>,
}

fn tags_of<'a>(tags: impl Iterator<Item = (&'a str, &'a str)>) -> Tags {
    tags.map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

/// The elements of one block wanted by each output, in block order.
fn wanted_elements(block: &PrimitiveBlock, outputs: &[Output]) -> Vec<Vec<Element>> {
    let mut found: Vec<Vec<Element>> = outputs.iter().map(|_| Vec::new()).collect();
    let wants = |kind: Kind, id: i64| {
        outputs
            .iter()
            .enumerate()
            .filter(move |(_, o)| o.ids[kind as usize].binary_search(&id).is_ok())
            .map(|(i, _)| i)
    };
    for group in block.groups() {
        for node in group.dense_nodes() {
            for i in wants(Kind::Node, node.id()) {
                found[i].push(Element::Node {
                    id: node.id(),
                    lon: node.decimicro_lon(),
                    lat: node.decimicro_lat(),
                    tags: tags_of(node.tags()),
                });
            }
        }
        for node in group.nodes() {
            for i in wants(Kind::Node, node.id()) {
                found[i].push(Element::Node {
                    id: node.id(),
                    lon: node.decimicro_lon(),
                    lat: node.decimicro_lat(),
                    tags: tags_of(node.tags()),
                });
            }
        }
        for way in group.ways() {
            for i in wants(Kind::Way, way.id()) {
                found[i].push(Element::Way {
                    id: way.id(),
                    refs: way.refs().collect(),
                    tags: tags_of(way.tags()),
                });
            }
        }
        for relation in group.relations() {
            for i in wants(Kind::Relation, relation.id()) {
                found[i].push(Element::Relation {
                    id: relation.id(),
                    members: relation
                        .members()
                        .map(|m| {
                            let kind = match m.member_type {
                                RelMemberType::Node => Kind::Node,
                                RelMemberType::Way => Kind::Way,
                                RelMemberType::Relation => Kind::Relation,
                            };
                            (
                                kind as u8,
                                m.member_id,
                                m.role().unwrap_or_default().to_string(),
                            )
                        })
                        .collect(),
                    tags: tags_of(relation.tags()),
                });
            }
        }
    }
    found
}

/// Copy the listed elements of `file` into one new file per output, in one pass over `file`. The new
/// headers keep the replication fields and the sort flag of the source.
pub fn extract(file: &PbfFile, outputs: &[Output], writing_program: &str) -> Result<(), String> {
    let source = file.header()?;
    let mut writers = Vec::with_capacity(outputs.len());
    for output in outputs {
        let header = Header {
            bbox: output.bbox,
            required_features: vec!["OsmSchema-V0.6".into(), "DenseNodes".into()],
            optional_features: source
                .optional_features
                .iter()
                .filter(|f| f.as_str() == "Sort.Type_then_ID")
                .cloned()
                .collect(),
            writing_program: Some(writing_program.to_string()),
            source: None,
            replication_timestamp: source.replication_timestamp,
            replication_sequence_number: source.replication_sequence_number,
            replication_base_url: source.replication_base_url.clone(),
        };
        writers.push(Writer::create(&output.path, &header)?);
    }
    let wanted_block = |ranges: &[Option<(i64, i64)>; 3]| {
        outputs.iter().any(|o| {
            ranges.iter().zip(&o.ids).any(|(range, ids)| {
                range.is_some_and(|(lo, hi)| {
                    let i = ids.partition_point(|&x| x < lo);
                    i < ids.len() && ids[i] <= hi
                })
            })
        })
    };
    file.stream(
        wanted_block,
        |block| wanted_elements(block, outputs),
        |found| {
            for (writer, elements) in writers.iter_mut().zip(found) {
                for element in elements {
                    writer.add(element)?;
                }
            }
            Ok(())
        },
    )?;
    for writer in writers {
        writer.finish()?;
    }
    Ok(())
}

/// Write `elements` (grouped by type, nodes first) into a new file.
pub fn write(path: &Path, header: &Header, elements: Vec<Element>) -> Result<(), String> {
    let mut writer = Writer::create(path, header)?;
    for element in elements {
        writer.add(element)?;
    }
    writer.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbf::{RelationQuery, WayQuery};

    fn tags(pairs: &[(&str, &str)]) -> Tags {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn varints_and_zigzag() {
        let mut buf = Vec::new();
        varint(&mut buf, 300);
        assert_eq!(buf, vec![0xac, 0x02]);
        assert_eq!((zigzag(0), zigzag(-1), zigzag(1), zigzag(-2)), (0, 1, 2, 3));
    }

    #[test]
    fn written_files_read_back() {
        let dir = std::env::temp_dir().join(format!("iduedu-pbf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("small.osm.pbf");
        let header = Header {
            required_features: vec!["OsmSchema-V0.6".into(), "DenseNodes".into()],
            replication_timestamp: Some(1_790_000_000),
            ..Header::default()
        };
        let mut elements = Vec::new();
        for id in 1..=20_000i64 {
            let t = if id % 3 == 0 {
                tags(&[("highway", "bus_stop")])
            } else {
                Vec::new()
            };
            elements.push(Element::Node {
                id,
                lon: 300_000_000 + id as i32,
                lat: -600_000_000 - id as i32,
                tags: t,
            });
        }
        elements.push(Element::Way {
            id: 7,
            refs: vec![3, 1, 20_000],
            tags: tags(&[("highway", "residential"), ("name", "Ñ")]),
        });
        elements.push(Element::Relation {
            id: 9,
            members: vec![
                (0, 3, "stop".into()),
                (1, 7, String::new()),
                (2, 9, "x".into()),
            ],
            tags: tags(&[("type", "route"), ("route", "bus")]),
        });
        write(&path, &header, elements).unwrap();

        let file = PbfFile::open(&path).unwrap();
        assert_eq!(
            file.header().unwrap().replication_timestamp,
            Some(1_790_000_000)
        );
        let query = WayQuery {
            clauses: &[],
            any_keys: None,
            ids: None,
            tags: true,
        };
        for _ in 0..2 {
            // the second time through the block index
            let ways = file.ways(&query).unwrap();
            assert_eq!(
                (ways.ids.clone(), ways.refs_of(0).to_vec()),
                (vec![7], vec![3, 1, 20_000])
            );
            assert_eq!(ways.tags[0][1], ("name".to_string(), "Ñ".to_string()));
            let (lon, lat) = file.locations(&[1, 3, 19_999, 30_000]).unwrap();
            assert_eq!(
                lon,
                vec![300_000_001, 300_000_003, 300_019_999, crate::pbf::MISSING]
            );
            assert_eq!(lat[2], -600_019_999);
            let nodes = file.nodes(&[2, 3, 18_000]).unwrap();
            assert_eq!(nodes.ids, vec![2, 3, 18_000]);
            assert_eq!(nodes.tags[1], tags(&[("highway", "bus_stop")]));
            let relations = file
                .relations(&RelationQuery {
                    contains_nodes: Some(&[3]),
                    ..RelationQuery::default()
                })
                .unwrap();
            assert_eq!(relations.ids, vec![9]);
            assert_eq!(relations.member_types, vec![0, 1, 2]);
            assert_eq!(relations.member_roles, vec!["stop", "", "x"]);
        }

        let out = dir.join("part.osm.pbf");
        let outputs = [Output {
            path: out.clone(),
            ids: [vec![1, 3, 20_000], vec![7], vec![]],
            bbox: Some([30.0, -61.0, 31.0, -60.0]),
        }];
        extract(&file, &outputs, "test").unwrap();
        let part = PbfFile::open(&out).unwrap();
        let header = part.header().unwrap();
        assert_eq!(header.writing_program.as_deref(), Some("test"));
        assert_eq!(header.replication_timestamp, Some(1_790_000_000));
        let bbox = header.bbox.unwrap();
        assert!((bbox[0] - 30.0).abs() < 1e-9 && (bbox[1] + 61.0).abs() < 1e-9);
        assert_eq!(
            part.nodes(&[1, 2, 3, 20_000]).unwrap().ids,
            vec![1, 3, 20_000]
        );
        assert_eq!(part.ways(&query).unwrap().ids, vec![7]);
        assert!(part
            .relations(&RelationQuery::default())
            .unwrap()
            .ids
            .is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
