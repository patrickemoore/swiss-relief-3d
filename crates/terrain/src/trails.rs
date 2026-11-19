//! Hiking trails as a layer over the terrain: polylines in local metres,
//! classified the way Swiss signposts are, cut into the same tiles as the
//! height pyramid and simplified per level.
//!
//! A dataset with trails has a `trails/` directory next to its heights:
//!
//! ```text
//! trails/index.bin              which tiles exist, per level
//! trails/<level>/<x>_<y>.bin    the polylines crossing that tile, deflated
//! ```
//!
//! A tile holds every polyline that crosses it, clipped to the tile with a
//! margin of one cell, so that lines run on to the neighbouring tile without
//! a gap. Coordinates are stored in eighths of a cell from the tile's
//! south-west corner, as zigzag varint deltas. Heights are not stored: the
//! renderer drapes the lines on whatever terrain it is drawing.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::dataset::Meta;
use crate::pack::TileIndex;
use crate::terrain::{TILED_LEVELS, TILE_SIZE};

pub const TRAILS_DIR: &str = "trails";
pub const INDEX_FILE: &str = "index.bin";
/// How far past its edge a tile keeps lines, in cells.
const MARGIN_CELLS: f32 = 1.0;
/// Coordinate resolution inside a tile, in steps per cell.
const UNITS_PER_CELL: f32 = 8.0;
/// Simplification tolerance per level, in cells.
const TOLERANCE_CELLS: f32 = 0.25;
/// Longest segment kept per level, in cells: a line is draped on the terrain
/// at its vertices, so a long segment would cut through the ground between
/// them.
const MAX_SEGMENT_CELLS: f32 = 2.0;
const DEFLATE_LEVEL: u8 = 7;

/// The three classes of Swiss hiking trail, in the order of their signposts:
/// yellow, white-red-white and white-blue-white.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum TrailClass {
    Hiking = 1,
    Mountain = 2,
    Alpine = 3,
}

impl TrailClass {
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(Self::Hiking),
            2 => Some(Self::Mountain),
            3 => Some(Self::Alpine),
            _ => None,
        }
    }

    /// The class named by swissTLM3D's `WANDERWEGE` attribute.
    pub fn from_swisstlm3d(value: &str) -> Option<Self> {
        match value.trim() {
            "Wanderweg" => Some(Self::Hiking),
            "Bergwanderweg" => Some(Self::Mountain),
            "Alpinwanderweg" => Some(Self::Alpine),
            _ => None,
        }
    }
}

/// A trail, or a piece of one, in local metres.
#[derive(Clone, Debug, PartialEq)]
pub struct Polyline {
    pub class: TrailClass,
    pub points: Vec<[f32; 2]>,
}

impl Polyline {
    pub fn length(&self) -> f32 {
        self.points.windows(2).map(|pair| distance(pair[0], pair[1])).sum()
    }
}

/// What a dataset records about its trails.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrailsInfo {
    pub source: String,
    /// Trails in the dataset, and their length in kilometres.
    pub polylines: usize,
    pub kilometres: f32,
}

/// Where the tiles of trails come from.
pub trait TrailSource: Send + Sync {
    /// The polylines of the tile of `level` whose south-west cell is
    /// `(x, y) * TILE_SIZE`, or `None` where there are none.
    fn tile(&self, level: usize, x: usize, y: usize) -> Result<Option<Vec<Polyline>>>;
}

/// The trails of a dataset: which tiles exist, and where to read them.
pub struct Trails {
    pub index: TileIndex,
    pub source: Arc<dyn TrailSource>,
}

/// Opens the trails of the dataset in `dir`, if it has any.
pub fn open(dir: &Path, meta: &Meta) -> Result<Option<Trails>> {
    let path = dir.join(TRAILS_DIR).join(INDEX_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let index = TileIndex::from_bytes(&bytes).with_context(|| format!("reading {}", path.display()))?;
    let source = DiskTrails { dir: dir.join(TRAILS_DIR), cell: meta.cell_size };
    Ok(Some(Trails { index, source: Arc::new(source) }))
}

pub fn tile_path(trails_dir: &Path, level: usize, x: usize, y: usize) -> PathBuf {
    trails_dir.join(level.to_string()).join(format!("{x}_{y}.bin"))
}

struct DiskTrails {
    dir: PathBuf,
    cell: f32,
}

impl TrailSource for DiskTrails {
    fn tile(&self, level: usize, x: usize, y: usize) -> Result<Option<Vec<Polyline>>> {
        let path = tile_path(&self.dir, level, x, y);
        match fs::read(&path) {
            Ok(bytes) => decode_tile(&bytes, level, [x, y], self.cell)
                .with_context(|| format!("reading {}", path.display()))
                .map(Some),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry.

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Distance from `p` to the segment `a`–`b`.
fn distance_to_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let length2 = dx * dx + dy * dy;
    if length2 <= 0.0 {
        return distance(p, a);
    }
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2).clamp(0.0, 1.0);
    distance(p, [a[0] + t * dx, a[1] + t * dy])
}

/// Douglas–Peucker: the fewest of `points` that stay within `tolerance` of
/// the original line.
pub fn simplify(points: &[[f32; 2]], tolerance: f32) -> Vec<[f32; 2]> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut stack = vec![(0, points.len() - 1)];
    while let Some((first, last)) = stack.pop() {
        if last <= first + 1 {
            continue;
        }
        let (mut farthest, mut index) = (0.0, first);
        for (i, &point) in points.iter().enumerate().take(last).skip(first + 1) {
            let d = distance_to_segment(point, points[first], points[last]);
            if d > farthest {
                (farthest, index) = (d, i);
            }
        }
        if farthest > tolerance {
            keep[index] = true;
            stack.push((first, index));
            stack.push((index, last));
        }
    }
    points.iter().zip(keep).filter_map(|(point, kept)| kept.then_some(*point)).collect()
}

/// Adds vertices so that no segment is longer than `max_length`.
pub fn densify(points: &[[f32; 2]], max_length: f32) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity(points.len());
    for (i, &point) in points.iter().enumerate() {
        if i > 0 {
            let previous = points[i - 1];
            let pieces = (distance(previous, point) / max_length).ceil().max(1.0) as usize;
            for k in 1..pieces {
                let t = k as f32 / pieces as f32;
                out.push([previous[0] + t * (point[0] - previous[0]), previous[1] + t * (point[1] - previous[1])]);
            }
        }
        out.push(point);
    }
    out
}

/// Liang–Barsky: the part of the segment `a`–`b` inside the rectangle, if any.
fn clip_segment(a: [f32; 2], b: [f32; 2], min: [f32; 2], max: [f32; 2]) -> Option<([f32; 2], [f32; 2])> {
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    let d = [b[0] - a[0], b[1] - a[1]];
    for axis in 0..2 {
        for (p, q) in [(-d[axis], a[axis] - min[axis]), (d[axis], max[axis] - a[axis])] {
            if p == 0.0 {
                if q < 0.0 {
                    return None;
                }
            } else {
                let t = q / p;
                if p < 0.0 {
                    t0 = t0.max(t);
                } else {
                    t1 = t1.min(t);
                }
                if t0 > t1 {
                    return None;
                }
            }
        }
    }
    let at = |t: f32| [a[0] + t * d[0], a[1] + t * d[1]];
    Some((at(t0), at(t1)))
}

/// The pieces of a polyline inside a rectangle.
fn clip_polyline(line: &Polyline, min: [f32; 2], max: [f32; 2]) -> Vec<Polyline> {
    let mut pieces = Vec::new();
    let mut current: Vec<[f32; 2]> = Vec::new();
    for pair in line.points.windows(2) {
        match clip_segment(pair[0], pair[1], min, max) {
            Some((start, end)) => {
                if current.last() != Some(&start) {
                    if current.len() >= 2 {
                        pieces.push(Polyline { class: line.class, points: std::mem::take(&mut current) });
                    } else {
                        current.clear();
                    }
                    current.push(start);
                }
                current.push(end);
            }
            None => {
                if current.len() >= 2 {
                    pieces.push(Polyline { class: line.class, points: std::mem::take(&mut current) });
                } else {
                    current.clear();
                }
            }
        }
    }
    if current.len() >= 2 {
        pieces.push(Polyline { class: line.class, points: current });
    }
    pieces
}

/// Cuts polylines into the tiles of `level`, each clipped to its tile plus
/// the margin. `cell0` is the cell size of level 0, `size` the terrain's
/// side in cells of level 0.
pub fn cut(polylines: &[Polyline], level: usize, cell0: f32, size: usize) -> BTreeMap<[usize; 2], Vec<Polyline>> {
    let cell = cell0 * (1 << level) as f32;
    let tile_metres = TILE_SIZE as f32 * cell;
    let margin = MARGIN_CELLS * cell;
    let tiles_per_side = (size >> level).div_ceil(TILE_SIZE);
    let mut tiles: BTreeMap<[usize; 2], Vec<Polyline>> = BTreeMap::new();
    for line in polylines {
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for p in &line.points {
            for axis in 0..2 {
                lo[axis] = lo[axis].min(p[axis]);
                hi[axis] = hi[axis].max(p[axis]);
            }
        }
        let first =
            |v: f32| (((v - margin) / tile_metres).floor().max(0.0) as usize).min(tiles_per_side.saturating_sub(1));
        let last =
            |v: f32| (((v + margin) / tile_metres).floor().max(0.0) as usize).min(tiles_per_side.saturating_sub(1));
        for y in first(lo[1])..=last(hi[1]) {
            for x in first(lo[0])..=last(hi[0]) {
                let min = [x as f32 * tile_metres - margin, y as f32 * tile_metres - margin];
                let max = [(x + 1) as f32 * tile_metres + margin, (y + 1) as f32 * tile_metres + margin];
                let pieces = clip_polyline(line, min, max);
                if !pieces.is_empty() {
                    tiles.entry([x, y]).or_default().extend(pieces);
                }
            }
        }
    }
    tiles
}

// ---------------------------------------------------------------------------
// Encoding.

fn put_varint(bytes: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        bytes.push((value as u8) | 0x80);
        value >>= 7;
    }
    bytes.push(value as u8);
}

fn get_varint(bytes: &[u8], at: &mut usize) -> Result<u32> {
    let (mut value, mut shift) = (0u32, 0);
    loop {
        ensure!(*at < bytes.len(), "a trail tile is cut short");
        let byte = bytes[*at];
        *at += 1;
        value |= ((byte & 0x7f) as u32) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        ensure!(shift < 35, "a trail tile is malformed");
    }
}

fn zigzag(value: i32) -> u32 {
    ((value << 1) ^ (value >> 31)) as u32
}

fn unzigzag(value: u32) -> i32 {
    ((value >> 1) as i32) ^ -((value & 1) as i32)
}

/// The south-west corner of a tile and its coordinate unit, in metres.
fn tile_frame(level: usize, tile: [usize; 2], cell0: f32) -> ([f32; 2], f32) {
    let cell = cell0 * (1 << level) as f32;
    let tile_metres = TILE_SIZE as f32 * cell;
    ([tile[0] as f32 * tile_metres, tile[1] as f32 * tile_metres], cell / UNITS_PER_CELL)
}

pub fn encode_tile(polylines: &[Polyline], level: usize, tile: [usize; 2], cell0: f32) -> Vec<u8> {
    let (origin, unit) = tile_frame(level, tile, cell0);
    let mut bytes = Vec::new();
    put_varint(&mut bytes, polylines.len() as u32);
    for line in polylines {
        bytes.push(line.class as u8);
        put_varint(&mut bytes, line.points.len() as u32);
        let mut last = [0i32; 2];
        for point in &line.points {
            for axis in 0..2 {
                let value = ((point[axis] - origin[axis]) / unit).round() as i32;
                put_varint(&mut bytes, zigzag(value - last[axis]));
                last[axis] = value;
            }
        }
    }
    miniz_oxide::deflate::compress_to_vec(&bytes, DEFLATE_LEVEL)
}

pub fn decode_tile(bytes: &[u8], level: usize, tile: [usize; 2], cell0: f32) -> Result<Vec<Polyline>> {
    let (origin, unit) = tile_frame(level, tile, cell0);
    let bytes =
        miniz_oxide::inflate::decompress_to_vec(bytes).map_err(|error| anyhow::anyhow!("inflating: {error:?}"))?;
    let mut at = 0;
    let count = get_varint(&bytes, &mut at)? as usize;
    let mut polylines = Vec::with_capacity(count);
    for _ in 0..count {
        ensure!(at < bytes.len(), "a trail tile is cut short");
        let class = TrailClass::from_byte(bytes[at]).context("an unknown trail class")?;
        at += 1;
        let points = get_varint(&bytes, &mut at)? as usize;
        let mut line = Polyline { class, points: Vec::with_capacity(points) };
        let mut last = [0i32; 2];
        for _ in 0..points {
            let mut point = [0.0; 2];
            for axis in 0..2 {
                last[axis] += unzigzag(get_varint(&bytes, &mut at)?);
                point[axis] = origin[axis] + last[axis] as f32 * unit;
            }
            line.points.push(point);
        }
        polylines.push(line);
    }
    Ok(polylines)
}

// ---------------------------------------------------------------------------
// Export.

#[derive(Clone, Debug, Default)]
pub struct ExportReport {
    pub polylines: usize,
    pub kilometres: f32,
    pub tiles: [usize; TILED_LEVELS],
    pub bytes: [u64; TILED_LEVELS],
}

/// Writes the trails of a dataset into `dir/trails`: one set of tiles per
/// streamed level, simplified to that level's cells, and the index.
pub fn export(polylines: &[Polyline], meta: &Meta, dir: &Path) -> Result<(TileIndex, ExportReport)> {
    ensure!(!polylines.is_empty(), "there are no trails to write");
    let trails_dir = dir.join(TRAILS_DIR);
    if trails_dir.exists() {
        fs::remove_dir_all(&trails_dir).with_context(|| format!("clearing {}", trails_dir.display()))?;
    }
    let mut report = ExportReport {
        polylines: polylines.len(),
        kilometres: polylines.iter().map(Polyline::length).sum::<f32>() / 1000.0,
        ..Default::default()
    };
    let mut levels = Vec::with_capacity(TILED_LEVELS);
    for level in 0..TILED_LEVELS {
        let cell = meta.cell_size * (1 << level) as f32;
        let simplified: Vec<Polyline> = polylines
            .par_iter()
            .map(|line| Polyline {
                class: line.class,
                points: densify(&simplify(&line.points, TOLERANCE_CELLS * cell), MAX_SEGMENT_CELLS * cell),
            })
            .filter(|line| line.points.len() >= 2)
            .collect();
        let tiles = cut(&simplified, level, meta.cell_size, meta.size);
        let level_dir = trails_dir.join(level.to_string());
        fs::create_dir_all(&level_dir).with_context(|| format!("creating {}", level_dir.display()))?;
        let written: Vec<([usize; 2], u64)> = tiles
            .par_iter()
            .map(|(&tile, lines)| {
                let bytes = encode_tile(lines, level, tile, meta.cell_size);
                let path = tile_path(&trails_dir, level, tile[0], tile[1]);
                fs::write(&path, &bytes).with_context(|| format!("writing {}", path.display()))?;
                Ok((tile, bytes.len() as u64))
            })
            .collect::<Result<_>>()?;
        let per_side = (meta.size >> level).div_ceil(TILE_SIZE);
        let mut bits = vec![0u8; (per_side * per_side).div_ceil(8)];
        for (tile, bytes) in &written {
            let index = tile[1] * per_side + tile[0];
            bits[index / 8] |= 1 << (index % 8);
            report.bytes[level] += bytes;
        }
        report.tiles[level] = written.len();
        levels.push((per_side, per_side, bits));
    }
    let index = TileIndex::new(levels);
    let path = trails_dir.join(INDEX_FILE);
    fs::write(&path, index.to_bytes()).with_context(|| format!("writing {}", path.display()))?;
    Ok((index, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(class: TrailClass, points: &[[f32; 2]]) -> Polyline {
        Polyline { class, points: points.to_vec() }
    }

    #[test]
    fn simplification_keeps_corners_and_drops_the_rest() {
        let points: Vec<[f32; 2]> = (0..=10).map(|i| [i as f32, if i == 5 { 3.0 } else { 0.0 }]).collect();
        let kept = simplify(&points, 0.5);
        // The spike's neighbours lie far from the line to its tip, so they stay.
        assert_eq!(kept, vec![[0.0, 0.0], [4.0, 0.0], [5.0, 3.0], [6.0, 0.0], [10.0, 0.0]]);
        let flat: Vec<[f32; 2]> = (0..=10).map(|i| [i as f32, 0.1 * (i % 2) as f32]).collect();
        assert_eq!(simplify(&flat, 0.5), vec![[0.0, 0.0], [10.0, 0.0]]);
    }

    #[test]
    fn clipping_splits_a_line_that_leaves_and_returns() {
        let excursion = [[-5.0, 5.0], [5.0, 5.0], [5.0, 15.0], [8.0, 15.0], [8.0, 5.0], [15.0, 5.0]];
        let pieces = clip_polyline(&line(TrailClass::Hiking, &excursion), [0.0, 0.0], [10.0, 10.0]);
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0].points, vec![[0.0, 5.0], [5.0, 5.0], [5.0, 10.0]]);
        assert_eq!(pieces[1].points, vec![[8.0, 10.0], [8.0, 5.0], [10.0, 5.0]]);
        assert!(clip_polyline(&line(TrailClass::Alpine, &[[20.0, 20.0], [30.0, 30.0]]), [0.0; 2], [10.0; 2]).is_empty());
    }

    #[test]
    fn tiles_round_trip_through_the_encoding() {
        let cell = 10.0;
        let trail = line(TrailClass::Mountain, &[[2600.0, 2605.0], [2700.0, 2800.0], [3000.0, 2000.0]]);
        let tiles = cut(std::slice::from_ref(&trail), 0, cell, 4096);
        // Tiles are 2,560 m: the line starts just inside the second one.
        assert_eq!(tiles.keys().copied().collect::<Vec<_>>(), vec![[1, 0], [1, 1]]);
        for (&tile, lines) in &tiles {
            let bytes = encode_tile(lines, 0, tile, cell);
            let back = decode_tile(&bytes, 0, tile, cell).unwrap();
            assert_eq!(back.len(), lines.len());
            for (a, b) in back.iter().zip(lines) {
                assert_eq!(a.class, b.class);
                for (p, q) in a.points.iter().zip(&b.points) {
                    assert!(distance(*p, *q) <= cell / UNITS_PER_CELL, "{p:?} vs {q:?}");
                }
            }
        }
    }

    #[test]
    fn densifying_bounds_the_segment_length() {
        let points = densify(&[[0.0, 0.0], [10.0, 0.0], [10.0, 3.0]], 4.0);
        assert_eq!(points.len(), 5, "the long segment is cut in three, the short one kept");
        assert_eq!((points[0], points[3], points[4]), ([0.0, 0.0], [10.0, 0.0], [10.0, 3.0]));
        assert!(points.windows(2).all(|pair| distance(pair[0], pair[1]) <= 4.0 + 1e-5));
    }

    #[test]
    fn varints_and_zigzag_round_trip() {
        for value in [0i32, 1, -1, 63, -64, 1_000_000, -1_000_000] {
            let mut bytes = Vec::new();
            put_varint(&mut bytes, zigzag(value));
            let mut at = 0;
            assert_eq!(unzigzag(get_varint(&bytes, &mut at).unwrap()), value);
            assert_eq!(at, bytes.len());
        }
    }
}
