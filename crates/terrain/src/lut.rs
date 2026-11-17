//! The colour lookup table (plan section 7.2).
//!
//! A small 2D table indexed by shaded brightness (x) and normalized elevation
//! (y), after Jenny & Hurni (2006). Elevation moves from greyish green
//! lowlands through warm beige to near-white summits; brightness shifts lit
//! slopes toward warm yellow and shaded slopes toward cool blue-grey, as in
//! Imhof's printing technique.
//!
//! The table round-trips through a PNG so it can be painted by hand. In the
//! PNG, brightness grows to the right and elevation grows **upward**.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use anyhow::{ensure, Context, Result};

use crate::oklab::{self, Lab};

pub const LUT_WIDTH: usize = 256;
pub const LUT_HEIGHT: usize = 64;

/// One elevation anchor: colours for full shade, flat ground and full light.
struct Anchor {
    elevation: f32,
    shade: u32,
    flat: u32,
    lit: u32,
}

/// The default palette as 8-bit sRGB. Flat ground is deliberately light:
/// Swiss reliefs keep plains near paper white and spend ink on slopes.
const ANCHORS: [Anchor; 5] = [
    Anchor { elevation: 0.00, shade: 0x4d6164, flat: 0xa8bc9f, lit: 0xe4edc6 },
    Anchor { elevation: 0.30, shade: 0x4d6068, flat: 0xc3cba3, lit: 0xf4f2c6 },
    Anchor { elevation: 0.55, shade: 0x4c5b74, flat: 0xd9cba4, lit: 0xfff4c8 },
    Anchor { elevation: 0.80, shade: 0x48577a, flat: 0xddd5c6, lit: 0xfffbe8 },
    Anchor { elevation: 1.00, shade: 0x44527c, flat: 0xe2e0de, lit: 0xffffff },
];

/// RGBA8 colour table, sRGB-encoded. Row 0 is the lowest elevation.
#[derive(Clone, Debug, PartialEq)]
pub struct Lut {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl Lut {
    /// Builds the default table. `flat_brightness` is the brightness value at
    /// which the flat-ground colour sits (the shader's `B_flat`).
    pub fn swiss_default(flat_brightness: f32) -> Self {
        let mut rgba = Vec::with_capacity(LUT_WIDTH * LUT_HEIGHT * 4);
        for row in 0..LUT_HEIGHT {
            let elevation = row as f32 / (LUT_HEIGHT - 1) as f32;
            let (shade, flat, lit) = palette_at(elevation);
            for column in 0..LUT_WIDTH {
                let brightness = column as f32 / (LUT_WIDTH - 1) as f32;
                let colour = if brightness < flat_brightness {
                    oklab::mix(shade, flat, brightness / flat_brightness)
                } else {
                    oklab::mix(flat, lit, (brightness - flat_brightness) / (1.0 - flat_brightness))
                };
                rgba.extend_from_slice(&oklab::oklab_to_srgb8(colour));
                rgba.push(255);
            }
        }
        Self { width: LUT_WIDTH, height: LUT_HEIGHT, rgba }
    }

    pub fn save_png(&self, path: &Path) -> Result<()> {
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), self.width as u32, self.height as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&self.flipped_rows())?;
        Ok(())
    }

    pub fn load_png(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut reader = png::Decoder::new(BufReader::new(file)).read_info()?;
        let mut buffer = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buffer)?;
        ensure!(info.bit_depth == png::BitDepth::Eight, "colour table must be an 8-bit PNG");
        let pixels = &buffer[..info.buffer_size()];
        let rgba: Vec<u8> = match info.color_type {
            png::ColorType::Rgba => pixels.to_vec(),
            png::ColorType::Rgb => pixels.as_chunks::<3>().0.iter().flat_map(|&[r, g, b]| [r, g, b, 255]).collect(),
            other => anyhow::bail!("colour table must be RGB or RGBA, found {other:?}"),
        };
        let table = Self { width: info.width as usize, height: info.height as usize, rgba };
        Ok(Self { rgba: table.flipped_rows(), ..table })
    }

    /// The table with its row order reversed (PNG rows run top to bottom).
    fn flipped_rows(&self) -> Vec<u8> {
        self.rgba.chunks_exact(self.width * 4).rev().flatten().copied().collect()
    }
}

/// Shade, flat and lit colours at a normalized elevation.
fn palette_at(elevation: f32) -> (Lab, Lab, Lab) {
    let upper = ANCHORS.iter().position(|a| a.elevation >= elevation).unwrap_or(ANCHORS.len() - 1).max(1);
    let (lo, hi) = (&ANCHORS[upper - 1], &ANCHORS[upper]);
    let t = ((elevation - lo.elevation) / (hi.elevation - lo.elevation)).clamp(0.0, 1.0);
    let blend = |a: u32, b: u32| oklab::mix(oklab::oklab_from_hex(a), oklab::oklab_from_hex(b), t);
    (blend(lo.shade, hi.shade), blend(lo.flat, hi.flat), blend(lo.lit, hi.lit))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn luma(lut: &Lut, column: usize, row: usize) -> f32 {
        let i = (row * lut.width + column) * 4;
        0.2126 * lut.rgba[i] as f32 + 0.7152 * lut.rgba[i + 1] as f32 + 0.0722 * lut.rgba[i + 2] as f32
    }

    #[test]
    fn brightness_is_monotonic_along_every_row() {
        let lut = Lut::swiss_default(0.75);
        for row in 0..lut.height {
            for column in 1..lut.width {
                assert!(luma(&lut, column, row) >= luma(&lut, column - 1, row) - 0.75);
            }
        }
    }

    #[test]
    fn summits_are_lighter_than_lowlands_on_flat_ground() {
        let lut = Lut::swiss_default(0.75);
        assert!(luma(&lut, 191, lut.height - 1) > luma(&lut, 191, 0));
    }

    #[test]
    fn png_round_trip_restores_row_order() {
        let lut = Lut::swiss_default(0.75);
        let path = std::env::temp_dir().join("relief_terrain_lut_round_trip.png");
        lut.save_png(&path).unwrap();
        assert_eq!(Lut::load_png(&path).unwrap(), lut);
        std::fs::remove_file(path).ok();
    }
}
