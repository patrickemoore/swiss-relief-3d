//! A terrain packed for the web.
//!
//! The tiled dataset is made for a local disk: raw floats, 600 KB a tile and
//! gigabytes for a country. Served over HTTP the same terrain has to be much
//! smaller, and a browser has to be able to start from a single small file.
//!
//! ```text
//! meta.json                 as for any dataset, with a `packed` entry
//! base.bin                  everything that is loaded up front
//! tiles/<level>/<x>_<y>.bin  one tile: both pyramids and the lake mask
//! ```
//!
//! Heights are quantized to steps of a few centimetres, predicted from their
//! neighbours and deflated; what is left of a tile is about a fifth of its
//! floats. `base.bin` holds the first whole level, from which the coarser
//! ones are rebuilt, the fields that cover the terrain, and a list of the
//! tiles that exist, so a viewer never asks for one that does not.
//!
//! Only the part of the terrain square that holds data is stored. The rest
//! is filled in on loading exactly as it was when the dataset was built.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use miniz_oxide::deflate::compress_to_vec;
use miniz_oxide::inflate::decompress_to_vec;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::dataset::{write_meta, Meta};
use crate::fill::fill_holes;
use crate::grid::Grid;
use crate::lut::Lut;
use crate::pyramid::build_heights;
use crate::terrain::{
    Terrain, Tile, TileSource, EDGE_DISTANCE_CAP_M, LEAF_CELLS, NO_DATA, TILED_LEVELS, TILE_SIZE, TILE_SPAN,
};

/// The file that holds everything loaded up front.
pub const BASE_FILE: &str = "base.bin";
/// Heights are stored in steps of this many metres wherever their range
/// fits 16 bits at that step, which it does in all but the coarsest tiles of
/// the highest mountains. It is well below the noise of the source data.
const HEIGHT_STEP_M: f32 = 0.05;
/// Leaf height ranges only bound the terrain, so they are stored coarsely.
const LEAF_STEP_M: f32 = 0.5;
const LEAF_OFFSET_M: f32 = -1000.0;
/// Stored in place of the lowest height of a leaf without data.
const LEAF_NO_DATA: u16 = u16::MAX;
const DEFLATE_LEVEL: u8 = 8;

/// What a packed terrain adds to a dataset's description.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackInfo {
    /// The finest pyramid level that tiles are provided for. A viewer shows
    /// nothing finer, which trades detail for a smaller dataset.
    pub finest_level: usize,
    /// The part of the square that is stored, from its south-west corner:
    /// how many of the coarsest tiles east and north.
    pub area: [usize; 2],
}

impl PackInfo {
    /// The stored part of a square grid that is `side` texels wide over the
    /// whole terrain, in texels.
    fn rect(&self, meta: &Meta, side: usize) -> [usize; 2] {
        let coarsest_tile = TILE_SIZE << (TILED_LEVELS - 1);
        self.area.map(|tiles| (side * tiles * coarsest_tile / meta.size).min(side))
    }

    /// Columns and rows of tiles of `level` in the stored area.
    fn tiles(&self, level: usize) -> [usize; 2] {
        self.area.map(|tiles| tiles << (TILED_LEVELS - 1 - level))
    }
}

/// Which tiles of a packed terrain exist.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileIndex {
    /// For each tiled level: columns, rows and one bit per tile, row by row.
    levels: Vec<(usize, usize, Vec<u8>)>,
}

impl TileIndex {
    pub fn has(&self, level: usize, x: usize, y: usize) -> bool {
        let Some((columns, rows, bits)) = self.levels.get(level) else { return false };
        let index = y * columns + x;
        x < *columns && y < *rows && bits[index / 8] >> (index % 8) & 1 == 1
    }

    /// How many tiles of `level` exist.
    pub fn count(&self, level: usize) -> usize {
        self.levels[level].2.iter().map(|byte| byte.count_ones() as usize).sum()
    }
}

/// A linear mapping between heights and 16-bit samples.
#[derive(Clone, Copy, Debug)]
struct Scale {
    offset: f32,
    step: f32,
}

impl Scale {
    /// The finest scale, down to [`HEIGHT_STEP_M`], that holds all `heights`.
    fn covering<'a>(heights: impl Iterator<Item = &'a f32>) -> Self {
        let (low, high) = heights.fold((f32::MAX, f32::MIN), |(low, high), &h| (low.min(h), high.max(h)));
        Self { offset: low, step: HEIGHT_STEP_M.max((high - low) / 65_000.0) }
    }

    fn encode(&self, height: f32) -> u16 {
        ((height - self.offset) / self.step).round().clamp(0.0, 65_535.0) as u16
    }

    fn decode(&self, sample: u16) -> f32 {
        self.offset + sample as f32 * self.step
    }

    fn write(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.offset.to_le_bytes());
        bytes.extend_from_slice(&self.step.to_le_bytes());
    }

    fn read(bytes: &[u8]) -> Result<(Self, &[u8])> {
        ensure!(bytes.len() >= 8, "a height scale is cut short");
        let float = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
        Ok((Self { offset: float(0), step: float(4) }, &bytes[8..]))
    }
}

/// Signed values as unsigned ones with small magnitudes first.
fn zigzag(value: i16) -> u16 {
    ((value << 1) ^ (value >> 15)) as u16
}

fn unzigzag(value: u16) -> i16 {
    (value >> 1) as i16 ^ -((value & 1) as i16)
}

/// Appends 16-bit values as all their low bytes, then all their high bytes.
/// The high bytes of small values are runs of zeros, which deflate well.
fn push_planes(bytes: &mut Vec<u8>, values: impl Iterator<Item = u16> + Clone) {
    bytes.extend(values.clone().map(|value| value as u8));
    bytes.extend(values.map(|value| (value >> 8) as u8));
}

fn planes(bytes: &[u8], count: usize) -> impl Iterator<Item = u16> + '_ {
    (0..count).map(move |i| bytes[i] as u16 | (bytes[count + i] as u16) << 8)
}

/// Appends a grid of samples as what is left of each after predicting it
/// from its western, southern and south-western neighbours. Terrain is
/// locally close to a plane, so the remainders are small.
fn push_predicted(bytes: &mut Vec<u8>, samples: &[u16], width: usize) {
    let at = |i: usize, back: usize, inside: bool| if inside { samples[i - back] } else { 0 };
    let residual = move |i: usize| {
        let (west, south) = (!i.is_multiple_of(width), i >= width);
        let predicted = at(i, 1, west).wrapping_add(at(i, width, south)).wrapping_sub(at(i, width + 1, west && south));
        zigzag(samples[i].wrapping_sub(predicted) as i16)
    };
    push_planes(bytes, (0..samples.len()).map(residual));
}

fn predicted(bytes: &[u8], count: usize, width: usize) -> Vec<u16> {
    let mut samples = vec![0u16; count];
    for (i, residual) in planes(bytes, count).enumerate() {
        let (west, south) = (!i.is_multiple_of(width), i >= width);
        let at = |back: usize, inside: bool| if inside { samples[i - back] } else { 0 };
        let predicted = at(1, west).wrapping_add(at(width, south)).wrapping_sub(at(width + 1, west && south));
        samples[i] = predicted.wrapping_add(unzigzag(residual) as u16);
    }
    samples
}

/// Appends two height grids of the same shape: the first predicted, the
/// second as its difference from the first, which is zero almost everywhere.
fn push_heights(bytes: &mut Vec<u8>, plain: &[f32], accentuated: Option<&[f32]>, width: usize) {
    let scale = Scale::covering(plain.iter().chain(accentuated.into_iter().flatten()));
    scale.write(bytes);
    let samples: Vec<u16> = plain.iter().map(|&height| scale.encode(height)).collect();
    push_predicted(bytes, &samples, width);
    if let Some(accentuated) = accentuated {
        let difference = |(&height, &sample): (&f32, &u16)| zigzag(scale.encode(height).wrapping_sub(sample) as i16);
        push_planes(bytes, accentuated.iter().zip(&samples).map(difference));
    }
}

/// Reads what [`push_heights`] wrote and returns the rest of the bytes.
fn heights(bytes: &[u8], count: usize, width: usize, both: bool) -> Result<(Vec<f32>, Vec<f32>, &[u8])> {
    let (scale, bytes) = Scale::read(bytes)?;
    let stored = if both { 4 * count } else { 2 * count };
    ensure!(bytes.len() >= stored, "packed heights are cut short");
    let samples = predicted(bytes, count, width);
    let plain: Vec<f32> = samples.iter().map(|&sample| scale.decode(sample)).collect();
    let accentuated = if both {
        let difference = planes(&bytes[2 * count..], count);
        samples.iter().zip(difference).map(|(&s, d)| scale.decode(s.wrapping_add(unzigzag(d) as u16))).collect()
    } else {
        plain.clone()
    };
    Ok((plain, accentuated, &bytes[stored..]))
}

fn inflate(bytes: &[u8]) -> Result<Vec<u8>> {
    decompress_to_vec(bytes).map_err(|error| anyhow::anyhow!("corrupt packed data: {error}"))
}

/// Packs one tile. At level 0 the two pyramids are the same and one is kept.
pub fn encode_tile(tile: &Tile, level: usize) -> Vec<u8> {
    let mut raw = Vec::with_capacity(4 * TILE_SPAN * TILE_SPAN + TILE_SIZE * TILE_SIZE + 8);
    push_heights(&mut raw, &tile.plain, (level > 0).then_some(&tile.accentuated[..]), TILE_SPAN);
    raw.extend_from_slice(&tile.water);
    compress_to_vec(&raw, DEFLATE_LEVEL)
}

pub fn decode_tile(bytes: &[u8], level: usize) -> Result<Tile> {
    let raw = inflate(bytes)?;
    let (plain, accentuated, water) = heights(&raw, TILE_SPAN * TILE_SPAN, TILE_SPAN, level > 0)?;
    ensure!(water.len() == TILE_SIZE * TILE_SIZE, "a packed tile has the wrong size");
    Ok(Tile { plain, accentuated, water: water.to_vec() })
}

/// The lower-left `width` x `height` part of a grid.
fn crop<T: Copy + Send + Sync>(grid: &Grid<T>, [width, height]: [usize; 2]) -> Vec<T> {
    (0..height).flat_map(|y| grid.row(y)[..width].iter().copied()).collect()
}

/// A `side` x `side` grid with `part` in its lower-left corner and `rest`
/// everywhere else.
fn embed<T: Copy + Send + Sync>(part: &[T], [width, height]: [usize; 2], side: usize, rest: T) -> Grid<T> {
    Grid::from_fn(side, side, |x, y| if x < width && y < height { part[y * width + x] } else { rest })
}

/// Writes length-prefixed, deflated sections one after another.
#[derive(Default)]
struct Sections(Vec<u8>);

impl Sections {
    fn push(&mut self, raw: &[u8]) {
        let packed = compress_to_vec(raw, DEFLATE_LEVEL);
        self.0.extend_from_slice(&(packed.len() as u32).to_le_bytes());
        self.0.extend_from_slice(&packed);
    }
}

struct SectionReader<'a>(&'a [u8]);

impl SectionReader<'_> {
    fn next(&mut self) -> Result<Vec<u8>> {
        ensure!(self.0.len() >= 4, "the packed terrain is cut short");
        let length = u32::from_le_bytes(self.0[..4].try_into().expect("four bytes")) as usize;
        ensure!(self.0.len() >= 4 + length, "the packed terrain is cut short");
        let section = inflate(&self.0[4..4 + length])?;
        self.0 = &self.0[4 + length..];
        Ok(section)
    }

    /// A section that must hold exactly `length` bytes.
    fn sized(&mut self, length: usize) -> Result<Vec<u8>> {
        let section = self.next()?;
        ensure!(section.len() == length, "a section of the packed terrain has the wrong size");
        Ok(section)
    }
}

/// Everything a viewer loads up front, as the bytes of [`BASE_FILE`].
fn pack_base(terrain: &Terrain, info: &PackInfo, index: &TileIndex) -> Vec<u8> {
    let meta = &terrain.meta;
    let coarse = info.rect(meta, meta.size >> TILED_LEVELS);
    let mut sections = Sections::default();

    let mut raw = vec![];
    let (plain, accentuated) = (crop(&terrain.coarse[0][0], coarse), crop(&terrain.coarse[1][0], coarse));
    push_heights(&mut raw, &plain, Some(&accentuated), coarse[0]);
    sections.push(&raw);
    sections.push(&crop(&terrain.coarse_water, coarse));

    // Both components of the orientation lie in -1..1.
    let orientation = crop(&terrain.orientation, info.rect(meta, meta.orientation_size));
    let signed = |channel: usize| orientation.iter().map(move |pair| (pair[channel] * 127.0).round() as i8 as u8);
    sections.push(&signed(0).chain(signed(1)).collect::<Vec<u8>>());

    let edge = crop(&terrain.edge_distance, coarse);
    sections.push(&edge.iter().map(|metres| (metres / EDGE_DISTANCE_CAP_M * 255.0).round() as u8).collect::<Vec<u8>>());

    // Ranges are rounded outward, so they still bound the terrain.
    let leaves = crop(&terrain.leaf_ranges, info.rect(meta, meta.size / LEAF_CELLS));
    let step = |height: f32| (height - LEAF_OFFSET_M) / LEAF_STEP_M;
    let lowest =
        leaves.iter().map(|&range| if range == NO_DATA { LEAF_NO_DATA } else { step(range[0]).floor() as u16 });
    let highest = leaves.iter().map(|&range| if range == NO_DATA { 0 } else { step(range[1]).ceil() as u16 });
    let mut raw = vec![];
    push_planes(&mut raw, lowest);
    push_planes(&mut raw, highest);
    sections.push(&raw);

    sections.push(&index.levels.iter().flat_map(|(_, _, bits)| bits.iter().copied()).collect::<Vec<u8>>());

    let lut = &terrain.lut;
    let mut raw = vec![];
    raw.extend_from_slice(&(lut.width as u32).to_le_bytes());
    raw.extend_from_slice(&(lut.height as u32).to_le_bytes());
    raw.extend_from_slice(&lut.rgba);
    sections.push(&raw);
    sections.0
}

/// Rebuilds a terrain from its description and the bytes of [`BASE_FILE`].
/// `tiles` provides its tiles; a viewer that streams them some other way can
/// pass [`NoTiles`].
pub fn unpack(meta: Meta, base: &[u8], tiles: Arc<dyn TileSource>) -> Result<Terrain> {
    let info = meta.packed.context("the terrain is not packed")?;
    let side = meta.size >> TILED_LEVELS;
    let coarse @ [width, height] = info.rect(&meta, side);
    let count = width * height;
    let mut sections = SectionReader(base);

    let raw = sections.next()?;
    let (plain, accentuated, rest) = heights(&raw, count, width, true)?;
    ensure!(rest.is_empty(), "a section of the packed terrain has the wrong size");
    // Beyond the stored part the terrain is continued smoothly, as it was
    // when the dataset was built, and the coarser levels follow from that.
    let cell = meta.cell_size * (1u64 << TILED_LEVELS) as f32;
    let whole = |part: &[f32]| {
        let mut grid = embed(part, coarse, side, f32::NAN);
        fill_holes(&mut grid);
        grid
    };
    let coarse_levels =
        [build_heights(whole(&plain), cell, None), build_heights(whole(&accentuated), cell, Some(&meta.accentuation))];
    let coarse_water = embed(&sections.sized(count)?, coarse, side, 0);

    let oriented @ [width, height] = info.rect(&meta, meta.orientation_size);
    let raw = sections.sized(2 * width * height)?;
    let component = |i: usize, channel: usize| raw[channel * width * height + i] as i8 as f32 / 127.0;
    let pairs: Vec<[f32; 2]> = (0..width * height).map(|i| [component(i, 0), component(i, 1)]).collect();
    let orientation = embed(&pairs, oriented, meta.orientation_size, [0.0, 0.0]);

    let metres: Vec<f32> =
        sections.sized(count)?.iter().map(|&edge| edge as f32 / 255.0 * EDGE_DISTANCE_CAP_M).collect();
    let edge_distance = embed(&metres, coarse, side, 0.0);

    let leaf_side = meta.size / LEAF_CELLS;
    let leafy @ [width, height] = info.rect(&meta, leaf_side);
    let count = width * height;
    let raw = sections.sized(4 * count)?;
    let height_of = |sample: u16| LEAF_OFFSET_M + sample as f32 * LEAF_STEP_M;
    let ranges: Vec<[f32; 2]> = planes(&raw, count)
        .zip(planes(&raw[2 * count..], count))
        .map(|(low, high)| if low == LEAF_NO_DATA { NO_DATA } else { [height_of(low), height_of(high)] })
        .collect();
    let leaf_ranges = embed(&ranges, leafy, leaf_side, NO_DATA);

    let mut bits = &sections.next()?[..];
    let mut levels = vec![];
    for level in 0..TILED_LEVELS {
        let [columns, rows] = info.tiles(level);
        let length = (columns * rows).div_ceil(8);
        ensure!(bits.len() >= length, "the tile index is cut short");
        levels.push((columns, rows, bits[..length].to_vec()));
        bits = &bits[length..];
    }

    let raw = sections.next()?;
    ensure!(raw.len() >= 8, "the colour table is cut short");
    let number = |at: usize| u32::from_le_bytes(raw[at..at + 4].try_into().expect("four bytes")) as usize;
    let lut = Lut { width: number(0), height: number(4), rgba: raw[8..].to_vec() };
    ensure!(lut.rgba.len() == 4 * lut.width * lut.height, "the colour table has the wrong size");

    Ok(Terrain {
        meta,
        coarse: coarse_levels,
        coarse_water,
        orientation,
        edge_distance,
        leaf_ranges,
        lut,
        tiles,
        index: Some(TileIndex { levels }),
    })
}

/// How large a packed terrain came out.
#[derive(Clone, Copy, Debug, Default)]
pub struct PackReport {
    pub base_bytes: u64,
    /// Tiles written at each level, and their total size.
    pub tiles: [usize; TILED_LEVELS],
    pub tile_bytes: [u64; TILED_LEVELS],
}

fn tile_path(dir: &Path, level: usize, x: usize, y: usize) -> PathBuf {
    dir.join("tiles").join(level.to_string()).join(format!("{x}_{y}.bin"))
}

/// Packs a terrain into `dir`, with tiles from `finest_level` up.
pub fn export(terrain: &Terrain, dir: &Path, finest_level: usize) -> Result<PackReport> {
    ensure!(finest_level < TILED_LEVELS, "the finest level must be one of the {TILED_LEVELS} tiled levels");
    let meta = &terrain.meta;
    // The data is stored up to the coarsest tile that still holds some.
    let coarsest_tile = meta.cell_size * (TILE_SIZE << (TILED_LEVELS - 1)) as f32;
    let [_, _, east, north] = meta.data_region();
    let info = PackInfo { finest_level, area: [east, north].map(|metres| (metres / coarsest_tile).ceil() as usize) };

    let mut report = PackReport::default();
    let mut levels = vec![];
    for level in 0..TILED_LEVELS {
        let [columns, rows] = info.tiles(level);
        let mut bits = vec![0u8; (columns * rows).div_ceil(8)];
        if level >= finest_level {
            fs::create_dir_all(dir.join("tiles").join(level.to_string()))
                .with_context(|| format!("creating {}", dir.display()))?;
            let written: Vec<Result<Option<u64>>> = (0..columns * rows)
                .into_par_iter()
                .map(|index| {
                    let (x, y) = (index % columns, index / columns);
                    let Some(tile) = terrain.tiles.tile(level, x, y)? else { return Ok(None) };
                    let bytes = encode_tile(&tile, level);
                    let path = tile_path(dir, level, x, y);
                    fs::write(&path, &bytes).with_context(|| format!("writing {}", path.display()))?;
                    Ok(Some(bytes.len() as u64))
                })
                .collect();
            for (index, size) in written.into_iter().enumerate() {
                if let Some(size) = size? {
                    bits[index / 8] |= 1 << (index % 8);
                    report.tiles[level] += 1;
                    report.tile_bytes[level] += size;
                }
            }
        }
        levels.push((columns, rows, bits));
    }

    let base = pack_base(terrain, &info, &TileIndex { levels });
    report.base_bytes = base.len() as u64;
    let path = dir.join(BASE_FILE);
    fs::write(&path, base).with_context(|| format!("writing {}", path.display()))?;
    // Written last, so a directory with a meta file is a complete dataset.
    write_meta(dir, &Meta { tiled: true, packed: Some(info), ..meta.clone() })?;
    Ok(report)
}

/// Packed tiles read from a directory.
struct DiskTiles {
    dir: PathBuf,
}

impl TileSource for DiskTiles {
    fn tile(&self, level: usize, x: usize, y: usize) -> Result<Option<Tile>> {
        let path = tile_path(&self.dir, level, x, y);
        match fs::read(&path) {
            Ok(bytes) => decode_tile(&bytes, level).with_context(|| format!("reading {}", path.display())).map(Some),
            // Tiles without data are simply not stored.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
        }
    }
}

/// A source without tiles, for a viewer that streams them itself.
pub struct NoTiles;

impl TileSource for NoTiles {
    fn tile(&self, _level: usize, _x: usize, _y: usize) -> Result<Option<Tile>> {
        Ok(None)
    }
}

/// Opens a packed terrain from a directory.
pub(crate) fn open(dir: &Path, meta: Meta) -> Result<Terrain> {
    let path = dir.join(BASE_FILE);
    let base = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    if meta.packed.is_none() {
        bail!("{} does not hold a packed terrain", dir.display());
    }
    unpack(meta, &base, Arc::new(DiskTiles { dir: dir.to_owned() }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;

    #[test]
    fn zigzag_round_trips_and_keeps_small_values_small() {
        for value in [0, 1, -1, 2, -2, 1000, -1000, i16::MAX, i16::MIN] {
            assert_eq!(unzigzag(zigzag(value)), value);
        }
        assert_eq!([zigzag(0), zigzag(-1), zigzag(1), zigzag(-2)], [0, 1, 2, 3]);
    }

    #[test]
    fn prediction_round_trips_and_flattens_a_plane() {
        // A tilted plane with one bump.
        let width = 7;
        let mut samples: Vec<u16> = (0..35).map(|i| (300 + 20 * (i % width) + 50 * (i / width)) as u16).collect();
        samples[17] += 9;
        let mut bytes = vec![];
        push_predicted(&mut bytes, &samples, width);
        assert_eq!(predicted(&bytes, samples.len(), width), samples);
        // Away from the first row and column, only the bump leaves anything.
        let residuals: Vec<u16> = planes(&bytes, samples.len()).collect();
        let interior = |i: &usize| !i.is_multiple_of(width) && *i >= width;
        let left: Vec<usize> = (0..35).filter(interior).filter(|&i| residuals[i] != 0).collect();
        assert_eq!(left, [17, 18, 24, 25]);
    }

    #[test]
    fn a_tile_survives_packing_to_within_half_a_step() {
        let terrain = Terrain::from(synthetic::cone_dataset(1024).unwrap());
        for level in [0, 3] {
            let tile = terrain.tiles.tile(level, 0, 0).unwrap().unwrap();
            let bytes = encode_tile(&tile, level);
            let unpacked = decode_tile(&bytes, level).unwrap();
            let worst = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(worst(&tile.plain, &unpacked.plain) <= 0.5 * HEIGHT_STEP_M + 1e-3);
            assert!(worst(&tile.accentuated, &unpacked.accentuated) <= 0.5 * HEIGHT_STEP_M + 1e-3);
            assert_eq!(tile.water, unpacked.water);
            // Raw, the tile takes four bytes a height.
            assert!(bytes.len() < TILE_SPAN * TILE_SPAN, "a level {level} tile packs to {} bytes", bytes.len());
        }
        assert!(decode_tile(&[1, 2, 3], 0).is_err());
    }

    #[test]
    fn a_packed_terrain_opens_like_the_original() {
        let dir = std::env::temp_dir().join("relief_terrain_pack_round_trip");
        fs::remove_dir_all(&dir).ok();
        let original = Terrain::from(synthetic::cone_dataset(1024).unwrap());
        let report = export(&original, &dir, 1).unwrap();
        // 1024 cells are four tiles a side at level 0, which is left out.
        assert_eq!(report.tiles, [0, 4, 1, 1, 1]);

        let packed = Terrain::open(&dir).unwrap();
        assert_eq!(packed.meta.packed, Some(PackInfo { finest_level: 1, area: [1, 1] }));
        let index = packed.index.as_ref().unwrap();
        assert!(index.has(1, 1, 1) && !index.has(0, 0, 0) && !index.has(1, 2, 0));
        assert_eq!(index.count(1), 4);

        for (level, (unpacked, whole)) in packed.coarse[1].iter().zip(&original.coarse[1]).enumerate() {
            let worst = unpacked.data().iter().zip(whole.data()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(worst < 0.1, "accentuated level {} is {worst} m off", TILED_LEVELS + level);
        }
        assert_eq!(packed.coarse_water, original.coarse_water);
        assert_eq!(packed.edge_distance, original.edge_distance);
        assert_eq!(packed.lut, original.lut);
        let [low, high] = packed.leaf_ranges.get(8, 8);
        let [true_low, true_high] = original.leaf_ranges.get(8, 8);
        assert!(low <= true_low && true_low - low < 1.0 && high >= true_high && high - true_high < 1.0);

        let tile = packed.tiles.tile(2, 0, 0).unwrap().unwrap();
        let expected = original.tiles.tile(2, 0, 0).unwrap().unwrap();
        assert!((tile.plain[5000] - expected.plain[5000]).abs() < 0.05);
        assert!(packed.tiles.tile(0, 0, 0).unwrap().is_none());
        fs::remove_dir_all(&dir).ok();
    }
}
