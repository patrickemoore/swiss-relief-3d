//! Reading swissALTI3D GeoTIFF tiles.
//!
//! The tiles are Cloud Optimized GeoTIFFs: after the header comes a 4 m
//! overview of the tile, then the 2 m data. The overview can therefore be
//! read from the first part of a file alone.

use std::fs::{self, File};
use std::io::{BufReader, Cursor, Read, Seek};
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::tags::Tag;

/// A decoded elevation tile. Rows run from north to south, as in the file.
pub struct Tile {
    /// LV95 easting of the western edge.
    pub west: f64,
    /// LV95 northing of the northern edge.
    pub north: f64,
    /// Pixel size in metres.
    pub pixel_size: f64,
    /// Where each sample's footprint starts relative to its pixel, in pixels.
    /// Zero for the 2 m data. An overview sample is the first of the 2 m
    /// samples under its pixel, so it sits before the pixel's centre and its
    /// footprint starts early.
    pub sample_offset: f64,
    pub width: usize,
    pub height: usize,
    pub heights: Vec<f32>,
    pub nodata: Option<f32>,
}

impl Tile {
    /// Whether a sample holds a real elevation.
    pub fn is_valid(&self, value: f32) -> bool {
        value.is_finite() && Some(value) != self.nodata && value > -1000.0
    }
}

/// Which image of a tile to read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// The 2 m data.
    Full,
    /// The 4 m overview: every other sample of the 2 m data.
    Overview,
}

/// Reads a tile in full resolution.
pub fn read_tile(path: &Path) -> Result<Tile> {
    read(path, Resolution::Full)
}

/// Reads a tile's overview from a file that holds just the overview prefix.
///
/// A rare tile carries an overview that cannot be decoded. The download
/// keeps such a tile whole, and its 2 m data is read instead.
pub fn read_overview(path: &Path) -> Result<Tile> {
    let bytes = fs::read(path).with_context(|| format!("opening {}", path.display()))?;
    let read = || {
        let whole = bytes.len() > overview_prefix_len(&bytes)?;
        decode_guarded(Cursor::new(&bytes), if whole { Resolution::Full } else { Resolution::Overview })
    };
    read().with_context(|| format!("reading {}", path.display()))
}

/// Whether the overview in the first part of a tile can be decoded.
pub fn overview_decodes(prefix: &[u8]) -> bool {
    decode_guarded(Cursor::new(prefix), Resolution::Overview).is_ok()
}

fn read(path: &Path, resolution: Resolution) -> Result<Tile> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    decode_guarded(BufReader::new(file), resolution).with_context(|| format!("reading {}", path.display()))
}

/// Decodes a tile. The decoder asserts on some malformed streams instead of
/// returning an error; among tens of thousands of tiles, that must not take
/// the whole run down without naming the tile.
fn decode_guarded(reader: impl Read + Seek, resolution: Resolution) -> Result<Tile> {
    panic::catch_unwind(AssertUnwindSafe(|| decode(reader, resolution)))
        .unwrap_or_else(|_| bail!("the TIFF decoder gave up"))
}

fn decode(reader: impl Read + Seek, resolution: Resolution) -> Result<Tile> {
    let mut decoder = Decoder::new(reader)?.with_limits(Limits::unlimited());

    // The georeference lives in the first directory, with the 2 m data.
    let (full_width, _) = decoder.dimensions()?;
    // ModelTiepoint maps raster (i, j, k) to model (x, y, z); swisstopo ties
    // the upper-left corner of the upper-left pixel.
    let tiepoint = decoder.get_tag_f64_vec(Tag::ModelTiepointTag)?;
    let scale = decoder.get_tag_f64_vec(Tag::ModelPixelScaleTag)?;
    ensure!(tiepoint.len() >= 6 && scale.len() >= 2, "no georeference");
    ensure!(tiepoint[0] == 0.0 && tiepoint[1] == 0.0, "not tied at the upper-left corner");
    ensure!((scale[0] - scale[1]).abs() < 1e-9, "non-square pixels");
    let nodata = decoder
        .get_tag_ascii_string(Tag::GdalNodata)
        .ok()
        .and_then(|text| text.trim_matches(|c: char| c.is_whitespace() || c == '\0').parse().ok());

    if resolution == Resolution::Overview {
        ensure!(decoder.more_images(), "the tile has no overview");
        decoder.next_image()?;
    }
    let (width, height) = decoder.dimensions()?;
    let heights = match decoder.read_image()? {
        DecodingResult::F32(heights) => heights,
        _ => bail!("not 32-bit float elevations"),
    };
    ensure!(heights.len() == (width * height) as usize, "not single-band");

    let decimation = full_width as f64 / width as f64;
    Ok(Tile {
        west: tiepoint[3],
        north: tiepoint[4],
        pixel_size: scale[0] * decimation,
        sample_offset: 0.5 / decimation - 0.5,
        width: width as usize,
        height: height as usize,
        heights,
        nodata,
    })
}

/// How many bytes from the start of a tile are needed to read its overview:
/// the end of the overview's pixel data. `prefix` must hold the header and
/// both image directories, which the first few kilobytes do.
pub fn overview_prefix_len(prefix: &[u8]) -> Result<usize> {
    let mut decoder = Decoder::new(Cursor::new(prefix))?.with_limits(Limits::unlimited());
    ensure!(decoder.more_images(), "the tile has no overview");
    decoder.next_image()?;
    let offsets = decoder.get_tag_u64_vec(Tag::TileOffsets)?;
    let byte_counts = decoder.get_tag_u64_vec(Tag::TileByteCounts)?;
    let end = offsets.iter().zip(&byte_counts).map(|(offset, count)| offset + count).max();
    Ok(end.context("the overview has no tiles")? as usize)
}
