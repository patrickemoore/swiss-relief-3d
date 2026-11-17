//! Mosaic, resample and crop (plan section 4.2).
//!
//! Tiles are resampled to the output cell size by area averaging, never by
//! nearest neighbour, which would alias the heights. The mosaic is stored
//! south-up, so GeoTIFF rows are flipped on the way in.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Result};
use rayon::prelude::*;
use relief_terrain::Grid;

use crate::geotiff::Tile;
use crate::region::Region;

/// Tiles decoded at once. Bounds the memory held in resampled blocks when a
/// mosaic covers a whole country.
const TILES_PER_BATCH: usize = 2048;

/// The raster a mosaic is assembled on.
#[derive(Clone, Debug, PartialEq)]
pub struct Raster {
    /// LV95 easting and northing of the south-west corner.
    pub origin: [f64; 2],
    /// Cell size in metres.
    pub cell: f64,
    pub width: usize,
    pub height: usize,
}

impl From<&Region> for Raster {
    fn from(region: &Region) -> Self {
        Self { origin: region.origin, cell: region.cell, width: region.size, height: region.size }
    }
}

/// A tile resampled to output cells, positioned in the mosaic.
struct Block {
    /// Column and row of the block's south-west cell in the mosaic.
    column: isize,
    row: isize,
    cells: Grid<f32>,
}

/// Assembles tiles on a raster. Cells that no tile covers are NaN.
pub fn mosaic(tile_paths: &[PathBuf], raster: &Raster, read: fn(&Path) -> Result<Tile>) -> Result<Grid<f32>> {
    let (width, height) = (raster.width as isize, raster.height as isize);
    let mut dem = Grid::new(raster.width, raster.height, f32::NAN);
    for batch in tile_paths.chunks(TILES_PER_BATCH) {
        let blocks: Vec<Block> = batch.par_iter().map(|path| resample(&read(path)?, raster)).collect::<Result<_>>()?;
        for block in &blocks {
            for y in 0..block.cells.height() {
                let row = block.row + y as isize;
                if !(0..height).contains(&row) {
                    continue;
                }
                for x in 0..block.cells.width() {
                    let column = block.column + x as isize;
                    if (0..width).contains(&column) {
                        dem.set(column as usize, row as usize, block.cells.get(x, y));
                    }
                }
            }
        }
    }
    Ok(dem)
}

/// Builds a square region's DEM from its tiles. Cells without data are
/// filled from their surroundings.
pub fn build_dem(tile_paths: &[PathBuf], region: &Region, read: fn(&Path) -> Result<Tile>) -> Result<Grid<f32>> {
    let mut dem = mosaic(tile_paths, &Raster::from(region), read)?;
    let missing = fill_missing(&mut dem)?;
    if missing > 0 {
        println!("filled {missing} cells without data ({:.2} %)", 100.0 * missing as f64 / dem.data().len() as f64);
    }
    Ok(dem)
}

/// For every output cell along one axis, the input pixels whose footprints it
/// overlaps and by how much. Pixel `p` stands for the span
/// `p + offset .. p + offset + 1`; with no offset and an integer ratio every
/// weight is 1.
fn overlaps(outputs: usize, pixels: usize, ratio: f64, offset: f64) -> Vec<Vec<(usize, f32)>> {
    (0..outputs)
        .map(|cell| {
            let (start, end) = (cell as f64 * ratio - offset, (cell + 1) as f64 * ratio - offset);
            let first = (start + 1e-9).floor().max(0.0) as usize;
            let last = ((end - 1e-9).ceil() as usize).min(pixels);
            (first..last).map(|pixel| (pixel, (end.min(pixel as f64 + 1.0) - start.max(pixel as f64)) as f32)).collect()
        })
        .collect()
}

/// Area-averages a tile down to the raster's cell size.
fn resample(tile: &Tile, raster: &Raster) -> Result<Block> {
    let ratio = raster.cell / tile.pixel_size;
    let (width, height) = ((tile.width as f64 / ratio).round() as usize, (tile.height as f64 / ratio).round() as usize);
    ensure!(
        ratio >= 1.0 && (width as f64 * ratio - tile.width as f64).abs() < 1e-6,
        "a tile of {} pixels of {} m does not divide into cells of {} m",
        tile.width,
        tile.pixel_size,
        raster.cell
    );

    let columns = overlaps(width, tile.width, ratio, tile.sample_offset);
    let rows = overlaps(height, tile.height, ratio, tile.sample_offset);
    let cell_area = (ratio * ratio) as f32;
    let cells = Grid::from_fn(width, height, |x, y| {
        // Output row y counts from the south, source rows from the north.
        let (mut sum, mut area) = (0.0f64, 0.0f32);
        for &(row, row_weight) in &rows[height - 1 - y] {
            for &(column, column_weight) in &columns[x] {
                let value = tile.heights[row * tile.width + column];
                if tile.is_valid(value) {
                    sum += (value * row_weight * column_weight) as f64;
                    area += row_weight * column_weight;
                }
            }
        }
        // Require a majority of valid samples so border cells are not biased.
        if 2.0 * area >= cell_area {
            (sum / area as f64) as f32
        } else {
            f32::NAN
        }
    });

    let south = tile.north - tile.height as f64 * tile.pixel_size;
    Ok(Block {
        column: ((tile.west - raster.origin[0]) / raster.cell).round() as isize,
        row: ((south - raster.origin[1]) / raster.cell).round() as isize,
        cells,
    })
}

const NEIGHBOURS: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// Fills NaN cells layer by layer with the mean of their known neighbours and
/// returns how many cells were filled.
fn fill_missing(dem: &mut Grid<f32>) -> Result<usize> {
    let (width, height) = (dem.width() as isize, dem.height() as isize);
    let missing = dem.data().iter().filter(|value| value.is_nan()).count();
    ensure!(missing < dem.data().len(), "no tile overlaps the region");

    let inside = |x: isize, y: isize| (0..width).contains(&x) && (0..height).contains(&y);
    let mut queued = Grid::new(dem.width(), dem.height(), false);
    let mut frontier = VecDeque::new();
    for y in 0..height {
        for x in 0..width {
            let borders_data = NEIGHBOURS
                .iter()
                .any(|(dx, dy)| inside(x + dx, y + dy) && !dem.get((x + dx) as usize, (y + dy) as usize).is_nan());
            if dem.get(x as usize, y as usize).is_nan() && borders_data {
                queued.set(x as usize, y as usize, true);
                frontier.push_back((x, y));
            }
        }
    }

    while !frontier.is_empty() {
        // Compute the whole layer before writing it, so the result does not
        // depend on the visiting order.
        let layer: Vec<(isize, isize, f32)> = frontier
            .drain(..)
            .map(|(x, y)| {
                let known: Vec<f32> = NEIGHBOURS
                    .iter()
                    .filter(|(dx, dy)| inside(x + dx, y + dy))
                    .map(|(dx, dy)| dem.get((x + dx) as usize, (y + dy) as usize))
                    .filter(|value| !value.is_nan())
                    .collect();
                (x, y, known.iter().sum::<f32>() / known.len() as f32)
            })
            .collect();
        for &(x, y, value) in &layer {
            dem.set(x as usize, y as usize, value);
        }
        for &(x, y, _) in &layer {
            for (dx, dy) in NEIGHBOURS {
                let (nx, ny) = (x + dx, y + dy);
                if inside(nx, ny) && dem.get(nx as usize, ny as usize).is_nan() && !queued.get(nx as usize, ny as usize)
                {
                    queued.set(nx as usize, ny as usize, true);
                    frontier.push_back((nx, ny));
                }
            }
        }
    }
    Ok(missing)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raster(cell: f64) -> Raster {
        Raster { origin: [2_600_000.0, 1_200_000.0], cell, width: 4, height: 4 }
    }

    fn tile(pixel_size: f64, size: usize, heights: Vec<f32>) -> Tile {
        let north = 1_200_020.0 + size as f64 * pixel_size;
        Tile {
            west: 2_600_020.0,
            north,
            pixel_size,
            sample_offset: 0.0,
            width: size,
            height: size,
            heights,
            nodata: Some(-9999.0),
        }
    }

    #[test]
    fn resample_averages_blocks_and_flips_rows() {
        // A 4 x 4 tile of 5 m pixels whose northern half is 100 m higher.
        let heights = (0..16).map(|i| if i < 8 { 200.0 } else { 100.0 }).collect();
        let block = resample(&tile(5.0, 4, heights), &raster(10.0)).unwrap();
        assert_eq!((block.column, block.row), (2, 2));
        assert_eq!(block.cells.data(), &[100.0, 100.0, 200.0, 200.0]);
    }

    #[test]
    fn fractional_ratios_weight_pixels_by_overlap() {
        // 4 m pixels to 10 m cells: each cell spans two and a half pixels.
        assert_eq!(
            overlaps(2, 5, 2.5, 0.0),
            vec![vec![(0, 1.0), (1, 1.0), (2, 0.5)], vec![(2, 0.5), (3, 1.0), (4, 1.0)]]
        );

        // Overview samples sit a quarter pixel before their centre, so each
        // footprint starts a quarter pixel early. The weights then centre on
        // the cell: samples at 1, 5 and 9 m for the cell from 0 to 10 m.
        let shifted = overlaps(2, 5, 2.5, -0.25);
        assert_eq!(shifted[0], vec![(0, 0.75), (1, 1.0), (2, 0.75)]);
        assert_eq!(shifted[1], vec![(2, 0.25), (3, 1.0), (4, 1.0)]);

        // Pixel columns hold 2, 6, 10, 14 and 18 m. The middle one is shared.
        let heights = (0..25).map(|i| 4.0 * (i % 5) as f32 + 2.0).collect();
        let block = resample(&tile(4.0, 5, heights), &raster(10.0)).unwrap();
        assert_eq!(block.cells.width(), 2);
        assert!((block.cells.get(0, 0) - (2.0 + 6.0 + 5.0) / 2.5).abs() < 1e-4);
        assert!((block.cells.get(1, 1) - (5.0 + 14.0 + 18.0) / 2.5).abs() < 1e-4);
    }

    #[test]
    fn cells_without_a_majority_of_data_stay_empty() {
        let mut heights = vec![300.0; 16];
        heights[..12].fill(-9999.0);
        let block = resample(&tile(5.0, 4, heights), &raster(10.0)).unwrap();
        // Only the southern row of pixels is valid: half of the southern cells.
        assert_eq!(block.cells.get(0, 0), 300.0);
        assert!(block.cells.get(0, 1).is_nan());
    }

    #[test]
    fn fill_missing_spreads_inward() {
        let mut dem = Grid::new(4, 1, f32::NAN);
        dem.set(0, 0, 10.0);
        assert_eq!(fill_missing(&mut dem).unwrap(), 3);
        assert_eq!(dem.data(), &[10.0, 10.0, 10.0, 10.0]);
    }
}
