//! A terrain too large to hold whole, stored as tiles.
//!
//! The terrain is a square of level-0 cells, of which the data may fill only
//! a part: a country inside its bounding square. Levels 0 to 4 are cut into
//! tiles of 256 texels and only tiles that hold data are stored; level 5 and
//! coarser cover the whole square and are small.
//!
//! ```text
//! meta.json
//! tiles/<level>/<x>_<y>.bin    plain heights, accentuated heights (not at
//!                              level 0, where they are the same), lake mask
//! coarse_height_05.f32 ...     the whole levels, plain
//! coarse_height_accent_05.f32  and accentuated
//! coarse_water.u8              lake coverage at level 5
//! orientation.f32              doubled-angle landform orientation pairs
//! edge_distance.f32            metres to the nearest place without data
//! leaves.f32                   height range of every quadtree leaf
//! lut.png                      colour table
//! ```
//!
//! Cells without data are filled with a smooth continuation of the terrain
//! before anything is derived, so that generalizing across the border works.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use rayon::prelude::*;

use crate::dataset::{read_raw, write_meta, write_raw, BuildOptions, Meta, DEFAULT_FLAT_BRIGHTNESS};
use crate::fill::fill_holes;
use crate::filter::gaussian_blur;
use crate::grid::Grid;
use crate::lut::Lut;
use crate::orientation::orientation_of_level;
use crate::pyramid::{build_heights, next_level};
use crate::stats::ElevationStats;
use crate::terrain::{
    leaf_ranges, Terrain, Tile, TileSource, EDGE_DISTANCE_CAP_M, LEAF_CELLS, TILED_LEVELS, TILE_SIZE, TILE_SPAN,
};
use crate::water::{halve_mask, lake_mask};

/// How far the distance to the edge of the data is smoothed, in cells of the
/// first whole level. Source data comes in square sheets, so the edge is a
/// staircase; smoothed over about one step, the terrain ends in a soft line.
const EDGE_SMOOTHING_CELLS: f32 = 3.0;

/// One cell in this many feeds the elevation statistics.
const STATS_STRIDE: usize = 61;

/// What a tiled terrain is built from.
pub struct TiledInput {
    pub name: String,
    pub source: String,
    /// Heights of the part of the square that can hold data, from its
    /// south-west corner, with NaN where there is none. Both sides must be
    /// multiples of the coarsest tile.
    pub dem: Grid<f32>,
    pub cell_size: f32,
    pub origin_lv95: [f64; 2],
    /// Side length of the terrain square in cells, a power of two.
    pub square: usize,
    /// Where built-in viewpoints and flights are laid out, in local metres.
    pub view_region: Option<[f32; 4]>,
}

fn tile_path(dir: &Path, level: usize, x: usize, y: usize) -> PathBuf {
    dir.join("tiles").join(level.to_string()).join(format!("{x}_{y}.bin"))
}

/// Analyzes a large terrain and writes it to `dir`.
pub fn build(input: TiledInput, options: &BuildOptions, dir: &Path) -> Result<Meta> {
    let TiledInput { name, source, mut dem, cell_size, origin_lv95, square, view_region } = input;
    let (width, height) = (dem.width(), dem.height());
    let coarsest_tile = TILE_SIZE << (TILED_LEVELS - 1);
    ensure!(
        width.is_multiple_of(coarsest_tile) && height.is_multiple_of(coarsest_tile),
        "the data area must be a multiple of {coarsest_tile} cells each way"
    );
    ensure!(
        square.is_power_of_two() && square >= width.max(height),
        "the square must be a power of two that holds the data"
    );
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;

    // What has to be known before the holes are filled. A grid of this size
    // takes gigabytes, so nothing here keeps a second one for long.
    let samples: Vec<f32> = dem.data().par_iter().step_by(STATS_STRIDE).copied().filter(|h| !h.is_nan()).collect();
    let elevation = ElevationStats::from_samples(samples).context("the terrain has no data")?;
    let leaf_has_data = Grid::from_fn(width / LEAF_CELLS, height / LEAF_CELLS, |x, y| {
        (y * LEAF_CELLS..(y + 1) * LEAF_CELLS)
            .any(|row| dem.row(row)[x * LEAF_CELLS..(x + 1) * LEAF_CELLS].iter().any(|h| !h.is_nan()))
    });
    let data_region = data_region(&leaf_has_data, LEAF_CELLS as f32 * cell_size).context("the terrain has no data")?;
    let coarse_side = square >> TILED_LEVELS;
    let edge_distance = edge_distance(&dem, coarse_side, cell_size);
    let holes = dem.data().par_iter().filter(|h| h.is_nan()).count() as f64 / dem.data().len() as f64;

    let mut water = lake_mask(&dem, cell_size, &options.lakes);
    let lake_km2 = water.data().par_iter().filter(|&&cell| cell > 0).count() as f32 * cell_size * cell_size / 1e6;
    println!("  lakes cover {lake_km2:.0} km2");

    println!("  filling {:.0} % of the area that has no data", 100.0 * holes);
    fill_holes(&mut dem);
    let leaves = leaf_ranges(&dem, square / LEAF_CELLS, |x, y| leaf_has_data.get(x, y));

    // The tiled levels. At level 0 the two pyramids are the same.
    let mut plain = dem;
    let mut accentuated: Option<Grid<f32>> = None;
    let mut orientation = None;
    for level in 0..TILED_LEVELS {
        let tiles = write_tiles(dir, level, &plain, accentuated.as_ref(), &water, &leaf_has_data)?;
        println!("  level {level}: {tiles} tiles");

        water = halve_mask(&water);
        let next_cell = cell_size * (2u64 << level) as f32;
        let fine = accentuated.as_ref().unwrap_or(&plain);
        let next_accentuated = next_level(fine, next_cell, Some(&options.accentuation));
        plain = next_level(&plain, next_cell, None);
        accentuated = Some(next_accentuated);
        if level + 1 == options.orientation.source_level {
            let texel = cell_size * (1u64 << TILED_LEVELS) as f32;
            orientation = Some(orientation_of_level(&plain, next_cell, &options.orientation, coarse_side, texel));
        }
    }
    let orientation = orientation.context("the orientation source level must be one of the tiled levels")?;
    let accentuated = accentuated.expect("the loop ran at least once");

    // The whole levels cover the square, most of which may be empty.
    let coarse_cell = cell_size * (1u64 << TILED_LEVELS) as f32;
    let embed = |grid: &Grid<f32>| {
        let mut whole = Grid::from_fn(coarse_side, coarse_side, |x, y| {
            if x < grid.width() && y < grid.height() {
                grid.get(x, y)
            } else {
                f32::NAN
            }
        });
        fill_holes(&mut whole);
        whole
    };
    let coarse_plain = build_heights(embed(&plain), coarse_cell, None);
    let coarse_accentuated = build_heights(embed(&accentuated), coarse_cell, Some(&options.accentuation));
    let coarse_water = Grid::from_fn(coarse_side, coarse_side, |x, y| {
        if x < water.width() && y < water.height() {
            water.get(x, y)
        } else {
            0
        }
    });

    for (prefix, pyramid) in [("", &coarse_plain), ("accent_", &coarse_accentuated)] {
        for (index, grid) in pyramid.iter().enumerate() {
            write_raw(&dir.join(format!("coarse_height_{prefix}{:02}.f32", TILED_LEVELS + index)), grid.data())?;
        }
    }
    write_raw(&dir.join("coarse_water.u8"), coarse_water.data())?;
    write_raw(&dir.join("orientation.f32"), orientation.data())?;
    write_raw(&dir.join("edge_distance.f32"), edge_distance.data())?;
    write_raw(&dir.join("leaves.f32"), leaves.data())?;
    Lut::swiss_default(DEFAULT_FLAT_BRIGHTNESS).save_png(&dir.join("lut.png"))?;

    let meta = Meta {
        name,
        source,
        size: square,
        cell_size,
        origin_lv95,
        levels: square.trailing_zeros() as usize + 1,
        orientation_size: coarse_side,
        elevation,
        accentuation: options.accentuation,
        tiled: true,
        data_region: Some(data_region),
        view_region,
        packed: None,
        trails: None,
    };
    // Written last, so a directory with a meta file is a complete dataset.
    write_meta(dir, &meta)?;
    Ok(meta)
}

/// The bounding rectangle of the leaves that hold data, in metres.
fn data_region(leaf_has_data: &Grid<bool>, leaf_size: f32) -> Option<[f32; 4]> {
    let occupied = |x: usize, y: usize| leaf_has_data.get(x, y);
    let columns: Vec<usize> =
        (0..leaf_has_data.width()).filter(|&x| (0..leaf_has_data.height()).any(|y| occupied(x, y))).collect();
    let rows: Vec<usize> =
        (0..leaf_has_data.height()).filter(|&y| (0..leaf_has_data.width()).any(|x| occupied(x, y))).collect();
    let (west, east) = (*columns.first()?, *columns.last()? + 1);
    let (south, north) = (*rows.first()?, *rows.last()? + 1);
    Some([west, south, east, north].map(|leaf| leaf as f32 * leaf_size))
}

/// Metres from every cell of the first whole level to the nearest such cell
/// that is mostly without data, by a two-pass chamfer distance transform,
/// smoothed so that the corners of the data's outline are rounded off.
/// `heights` are those of level 0, with NaN where there is no data.
fn edge_distance(heights: &Grid<f32>, side: usize, cell_size: f32) -> Grid<f32> {
    let block = 1 << TILED_LEVELS;
    let cell = cell_size * block as f32;
    let cap = EDGE_DISTANCE_CAP_M / cell;
    let has_data = |x: usize, y: usize| {
        if (x + 1) * block > heights.width() || (y + 1) * block > heights.height() {
            return false;
        }
        let covered: usize = (y * block..(y + 1) * block)
            .map(|row| heights.row(row)[x * block..(x + 1) * block].iter().filter(|h| !h.is_nan()).count())
            .sum();
        2 * covered >= block * block
    };
    let mut distance = Grid::from_fn(side, side, |x, y| if has_data(x, y) { cap } else { 0.0 });

    const DIAGONAL: f32 = std::f32::consts::SQRT_2;
    let side = side as isize;
    let mut relax = |x: isize, y: isize, neighbours: [(isize, isize, f32); 4]| {
        let nearest = neighbours
            .iter()
            .filter(|(dx, dy, _)| (0..side).contains(&(x + dx)) && (0..side).contains(&(y + dy)))
            .map(|(dx, dy, step)| distance.get((x + dx) as usize, (y + dy) as usize) + step)
            .fold(distance.get(x as usize, y as usize), f32::min);
        distance.set(x as usize, y as usize, nearest);
    };
    for y in 0..side {
        for x in 0..side {
            relax(x, y, [(-1, 0, 1.0), (0, -1, 1.0), (-1, -1, DIAGONAL), (1, -1, DIAGONAL)]);
        }
    }
    for y in (0..side).rev() {
        for x in (0..side).rev() {
            relax(x, y, [(1, 0, 1.0), (0, 1, 1.0), (1, 1, DIAGONAL), (-1, 1, DIAGONAL)]);
        }
    }

    // Smoothing a distance that stops at zero lifts it along a straight
    // edge by sigma / sqrt(2 pi). Taking that off again puts the zero back
    // on the edge; the rest is stretched so that the cap stays the cap.
    let lift = EDGE_SMOOTHING_CELLS / std::f32::consts::TAU.sqrt();
    let mut distance = gaussian_blur(&distance, EDGE_SMOOTHING_CELLS);
    let to_metres = cell * cap / (cap - lift);
    distance.data_mut().iter_mut().for_each(|value| *value = ((*value - lift).max(0.0) * to_metres).min(cap * cell));
    distance
}

/// Writes every tile of a level that holds data and returns how many.
fn write_tiles(
    dir: &Path,
    level: usize,
    plain: &Grid<f32>,
    accentuated: Option<&Grid<f32>>,
    water: &Grid<u8>,
    leaf_has_data: &Grid<bool>,
) -> Result<usize> {
    fs::create_dir_all(dir.join("tiles").join(level.to_string()))?;
    // Leaves along one side of a tile of this level.
    let leaves = (TILE_SIZE << level) / LEAF_CELLS;
    let holds_data = |x: usize, y: usize| {
        (y * leaves..(y + 1) * leaves).any(|row| leaf_has_data.row(row)[x * leaves..(x + 1) * leaves].contains(&true))
    };
    let cut = |grid: &Grid<f32>, x: usize, y: usize| -> Vec<f32> {
        let (west, south) = ((x * TILE_SIZE) as isize - 1, (y * TILE_SIZE) as isize - 1);
        (0..TILE_SPAN * TILE_SPAN)
            .map(|i| grid.get_clamped(west + (i % TILE_SPAN) as isize, south + (i / TILE_SPAN) as isize))
            .collect()
    };

    let (columns, rows) = (plain.width() / TILE_SIZE, plain.height() / TILE_SIZE);
    let written: Vec<Result<bool>> = (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            let (x, y) = (index % columns, index / columns);
            if !holds_data(x, y) {
                return Ok(false);
            }
            let mut bytes: Vec<u8> = bytemuck::cast_slice(&cut(plain, x, y)).to_vec();
            if let Some(accentuated) = accentuated {
                bytes.extend_from_slice(bytemuck::cast_slice(&cut(accentuated, x, y)));
            }
            for row in y * TILE_SIZE..(y + 1) * TILE_SIZE {
                bytes.extend_from_slice(&water.row(row)[x * TILE_SIZE..(x + 1) * TILE_SIZE]);
            }
            let path = tile_path(dir, level, x, y);
            fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
            Ok(true)
        })
        .collect();
    let mut count = 0;
    for tile in written {
        count += tile? as usize;
    }
    Ok(count)
}

/// Tiles read from a dataset directory.
struct DiskTiles {
    dir: PathBuf,
}

impl TileSource for DiskTiles {
    fn tile(&self, level: usize, x: usize, y: usize) -> Result<Option<Tile>> {
        let path = tile_path(&self.dir, level, x, y);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            // Tiles without data are simply not stored.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
        };
        let heights = TILE_SPAN * TILE_SPAN * 4;
        let pyramids = if level == 0 { 1 } else { 2 };
        ensure!(
            bytes.len() == pyramids * heights + TILE_SIZE * TILE_SIZE,
            "{} is not a level {level} tile",
            path.display()
        );

        let plain: Vec<f32> = bytemuck::pod_collect_to_vec(&bytes[..heights]);
        let accentuated =
            if level == 0 { plain.clone() } else { bytemuck::pod_collect_to_vec(&bytes[heights..2 * heights]) };
        Ok(Some(Tile { plain, accentuated, water: bytes[pyramids * heights..].to_vec() }))
    }
}

/// Opens a tiled dataset.
pub(crate) fn open(dir: &Path, meta: Meta) -> Result<Terrain> {
    let coarse_side = meta.size >> TILED_LEVELS;
    let leaves = meta.size / LEAF_CELLS;
    let pyramid = |prefix: &str| -> Result<Vec<Grid<f32>>> {
        (TILED_LEVELS..meta.levels)
            .map(|level| {
                let side = (meta.size >> level).max(1);
                read_raw(&dir.join(format!("coarse_height_{prefix}{level:02}.f32")), side, side)
            })
            .collect()
    };
    Ok(Terrain {
        coarse: [pyramid("")?, pyramid("accent_")?],
        coarse_water: read_raw(&dir.join("coarse_water.u8"), coarse_side, coarse_side)?,
        orientation: read_raw(&dir.join("orientation.f32"), meta.orientation_size, meta.orientation_size)?,
        edge_distance: read_raw(&dir.join("edge_distance.f32"), coarse_side, coarse_side)?,
        leaf_ranges: read_raw(&dir.join("leaves.f32"), leaves, leaves)?,
        lut: Lut::load_png(&dir.join("lut.png"))?,
        tiles: Arc::new(DiskTiles { dir: dir.to_owned() }),
        index: None,
        trails: None,
        meta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;
    use crate::terrain::NO_DATA;

    /// A cone that fills the west half of an 8192-cell data area inside a
    /// 16384-cell square.
    fn build_half_cone(dir: &Path) -> Meta {
        let cone = synthetic::cone(4096, 10.0);
        let dem = Grid::from_fn(8192, 4096, |x, y| if x < 4096 { cone.get(x, y) } else { f32::NAN });
        let input = TiledInput {
            name: "half".into(),
            source: "synthetic".into(),
            dem,
            cell_size: 10.0,
            origin_lv95: [0.0, 0.0],
            square: 16_384,
            view_region: None,
        };
        build(input, &BuildOptions::default(), dir).unwrap()
    }

    #[test]
    fn build_and_open_a_partly_empty_terrain() {
        let dir = std::env::temp_dir().join("relief_terrain_tiled_round_trip");
        fs::remove_dir_all(&dir).ok();
        let meta = build_half_cone(&dir);
        assert_eq!((meta.levels, meta.tiled), (15, true));
        assert_eq!(meta.data_region, Some([0.0, 0.0, 40_960.0, 40_960.0]));

        let terrain = Terrain::open(&dir).unwrap();

        // Tiles exist where the cone is and nowhere else.
        let tile = terrain.tiles.tile(0, 8, 8).unwrap().expect("the cone's summit has a tile");
        let summit = tile.plain[TILE_SPAN + 1];
        assert!((summit - 2500.0).abs() < 20.0, "summit at {summit} m");
        assert!(terrain.tiles.tile(0, 16, 8).unwrap().is_none());
        assert!(terrain.tiles.tile(4, 0, 0).unwrap().is_some());
        assert!(terrain.tiles.tile(4, 1, 0).unwrap().is_none());

        // Leaves and edge distances know where the data ends.
        assert_ne!(terrain.leaf_ranges.get(10, 10), NO_DATA);
        assert_eq!(terrain.leaf_ranges.get(70, 10), NO_DATA);
        assert!((terrain.edge_distance.get(64, 64) - EDGE_DISTANCE_CAP_M).abs() < 0.01);
        assert_eq!(terrain.edge_distance.get(200, 64), 0.0);
        // Two cells in from a straight edge; the smoothing pulls it down.
        let near_edge = terrain.edge_distance.get(126, 64);
        assert!((300.0..640.0).contains(&near_edge), "{near_edge} m from the edge");
        let on_edge = terrain.edge_distance.get(128, 64);
        assert!(on_edge < 100.0, "{on_edge} m at the edge");

        // Far from the data the fill settles on a height typical of it
        // instead of dropping to zero.
        let far = terrain.coarse[0][0].get(300, 300);
        assert!((450.0..1300.0).contains(&far), "far fill at {far} m");

        // Tiles are read on demand, so the directory has to outlive the checks.
        fs::remove_dir_all(&dir).ok();
    }
}
