//! A terrain as the renderer consumes it, whatever its size.
//!
//! The finest pyramid levels are cut into tiles that the renderer loads
//! around the camera; the coarser levels are small and held whole. A region
//! in memory and a country on disk both present themselves this way, so the
//! renderer has a single path.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;

use crate::dataset::{read_meta, Dataset, Meta};
use crate::grid::Grid;
use crate::lut::Lut;
use crate::pack::{self, TileIndex};
use crate::tiled;
use crate::water::halve_mask;

/// Texels along one side of a tile.
pub const TILE_SIZE: usize = 256;
/// Texels along one side of a tile's heights, which carry a one-texel apron
/// so that slopes can be taken at the rim.
pub const TILE_SPAN: usize = TILE_SIZE + 2;
/// Pyramid levels stored as tiles. Level 5 and coarser are held whole.
pub const TILED_LEVELS: usize = 5;
/// Cells along one side of a leaf of the mesh quadtree.
pub const LEAF_CELLS: usize = 64;
/// Distances to the edge of the data are stored up to this many metres.
pub const EDGE_DISTANCE_CAP_M: f32 = 8000.0;

/// Height range marking a quadtree leaf without data.
pub const NO_DATA: [f32; 2] = [1.0, 0.0];

/// One tile of one pyramid level.
#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    /// `TILE_SPAN` x `TILE_SPAN` plain heights, apron included.
    pub plain: Vec<f32>,
    /// The same from the ridge-accentuated pyramid.
    pub accentuated: Vec<f32>,
    /// `TILE_SIZE` x `TILE_SIZE` lake coverage.
    pub water: Vec<u8>,
}

/// Where the tiles of a terrain come from.
pub trait TileSource: Send + Sync {
    /// The tile of `level` whose south-west texel is `(x, y) * TILE_SIZE`, or
    /// `None` where the terrain has no data.
    fn tile(&self, level: usize, x: usize, y: usize) -> Result<Option<Tile>>;
}

pub struct Terrain {
    pub meta: Meta,
    /// Heights of every level from `TILED_LEVELS` up, finest first: the plain
    /// pyramid, then the accentuated one.
    pub coarse: [Vec<Grid<f32>>; 2],
    /// Lake coverage at the resolution of level `TILED_LEVELS`.
    pub coarse_water: Grid<u8>,
    /// Doubled-angle landform orientation over the whole square.
    pub orientation: Grid<[f32; 2]>,
    /// Metres from each cell of level `TILED_LEVELS` to the nearest place
    /// without data, capped at [`EDGE_DISTANCE_CAP_M`].
    pub edge_distance: Grid<f32>,
    /// Lowest and highest height in every quadtree leaf, or [`NO_DATA`].
    pub leaf_ranges: Grid<[f32; 2]>,
    pub lut: Lut,
    pub tiles: Arc<dyn TileSource>,
    /// Which tiles exist, if the terrain knows without asking for them.
    pub index: Option<TileIndex>,
}

impl Terrain {
    /// Opens a dataset directory of either kind.
    pub fn open(dir: &Path) -> Result<Self> {
        let meta = read_meta(dir)?;
        if meta.packed.is_some() {
            pack::open(dir, meta)
        } else if meta.tiled {
            tiled::open(dir, meta)
        } else {
            Ok(Dataset::load_with(dir, meta)?.into())
        }
    }

    /// Cell size of a pyramid level in metres.
    pub fn cell(&self, level: usize) -> f32 {
        self.meta.cell_size * (1u64 << level) as f32
    }

    /// The finest level the terrain has tiles for.
    pub fn finest_level(&self) -> usize {
        self.meta.packed.map_or(0, |packed| packed.finest_level)
    }

    /// The plain heights of `level` over a region given as west, south, east
    /// and north in metres.
    ///
    /// The cells are the level's size and start at the region's corner.
    /// Where that lies between the level's texels they are interpolated, so
    /// the grid sits exactly where the region is.
    pub fn heights_in(&self, level: usize, region: [f32; 4]) -> Result<Grid<f32>> {
        let cell = self.cell(level);
        let [west, south, east, north] = region.map(|metres| (metres / cell).max(0.0));
        let cells = |span: f32| (span.round() as usize).max(1);
        let (width, height) = (cells(east - west), cells(north - south));

        // The texels around the cells, and how far past them the cells lie.
        let texels = self.texels(level, [west.floor() as usize, south.floor() as usize], [width + 1, height + 1])?;
        let (fx, fy) = (west.fract(), south.fract());
        Ok(Grid::from_fn(width, height, |x, y| {
            let along = |y: usize| texels.get(x, y) * (1.0 - fx) + texels.get(x + 1, y) * fx;
            along(y) * (1.0 - fy) + along(y + 1) * fy
        }))
    }

    /// A block of the plain texels of `level` that starts at texel `first`,
    /// assembled from tiles where the level is tiled. Texels without data
    /// take the heights of the finest whole level.
    fn texels(&self, level: usize, first: [usize; 2], size: [usize; 2]) -> Result<Grid<f32>> {
        let ([west, south], [width, height]) = (first, size);
        let (east, north) = (west + width, south + height);

        if level >= TILED_LEVELS {
            let source = &self.coarse[0][level - TILED_LEVELS];
            return Ok(Grid::from_fn(width, height, |x, y| {
                source.get_clamped((west + x) as isize, (south + y) as isize)
            }));
        }

        let fallback = &self.coarse[0][0];
        let side = (self.meta.size >> level) as f32;
        let mut heights = Grid::from_fn(width, height, |x, y| {
            fallback.sample(((west + x) as f32 + 0.5) / side, ((south + y) as f32 + 0.5) / side)
        });
        for tile_y in south / TILE_SIZE..=(north - 1) / TILE_SIZE {
            for tile_x in west / TILE_SIZE..=(east - 1) / TILE_SIZE {
                let Some(tile) = self.tiles.tile(level, tile_x, tile_y)? else { continue };
                for row in 0..TILE_SIZE {
                    let Some(y) = (tile_y * TILE_SIZE + row).checked_sub(south).filter(|&y| y < height) else {
                        continue;
                    };
                    for column in 0..TILE_SIZE {
                        if let Some(x) = (tile_x * TILE_SIZE + column).checked_sub(west).filter(|&x| x < width) {
                            heights.set(x, y, tile.plain[(row + 1) * TILE_SPAN + column + 1]);
                        }
                    }
                }
            }
        }
        Ok(heights)
    }
}

/// The height range of every quadtree leaf of a terrain `leaves` leaves wide.
///
/// `heights` may cover only part of that square, from its south-west corner;
/// leaves beyond it, and leaves in which `has_data` finds nothing, are marked
/// [`NO_DATA`].
pub fn leaf_ranges(
    heights: &Grid<f32>,
    leaves: usize,
    has_data: impl Fn(usize, usize) -> bool + Sync,
) -> Grid<[f32; 2]> {
    let (width, height) = (heights.width(), heights.height());
    Grid::from_fn(leaves, leaves, |x, y| {
        if x * LEAF_CELLS >= width || y * LEAF_CELLS >= height || !has_data(x, y) {
            return NO_DATA;
        }
        // One extra cell all round covers bilinear filtering at the rim.
        let cells =
            |leaf: usize, limit: usize| (leaf * LEAF_CELLS).saturating_sub(1)..((leaf + 1) * LEAF_CELLS + 1).min(limit);
        cells(y, height)
            .flat_map(|row| heights.row(row)[cells(x, width)].iter())
            .fold([f32::MAX, f32::MIN], |[low, high], &h| [low.min(h), high.max(h)])
    })
}

/// Reads a cell, continuing the slope one cell beyond the edge so that
/// slopes taken across a tile's apron match those inside the terrain.
fn extended(grid: &Grid<f32>, x: isize, y: isize) -> f32 {
    let (width, height) = (grid.width() as isize, grid.height() as isize);
    let (cx, cy) = (x.clamp(0, width - 1), y.clamp(0, height - 1));
    let inward = |outside: isize, limit: isize| match outside {
        -1 => 1,
        o if o == limit => -1,
        _ => 0,
    };
    let (step_x, step_y) = (inward(x, width), inward(y, height));
    let centre = grid.get(cx as usize, cy as usize);
    let slope = |dx: isize, dy: isize| centre - grid.get_clamped(cx + dx, cy + dy);
    centre + if step_x != 0 { slope(step_x, 0) } else { 0.0 } + if step_y != 0 { slope(0, step_y) } else { 0.0 }
}

/// Tiles cut on demand from pyramids held in memory.
struct MemoryTiles {
    /// The tiled levels of the plain and the accentuated pyramid.
    heights: [Vec<Grid<f32>>; 2],
    /// Lake coverage at every tiled level.
    water: Vec<Grid<u8>>,
}

impl TileSource for MemoryTiles {
    fn tile(&self, level: usize, x: usize, y: usize) -> Result<Option<Tile>> {
        let (west, south) = ((x * TILE_SIZE) as isize, (y * TILE_SIZE) as isize);
        let plain = &self.heights[0][level];
        if west >= plain.width() as isize || south >= plain.height() as isize {
            return Ok(None);
        }
        let cut = |grid: &Grid<f32>| {
            (0..TILE_SPAN * TILE_SPAN)
                .map(|i| extended(grid, west + (i % TILE_SPAN) as isize - 1, south + (i / TILE_SPAN) as isize - 1))
                .collect()
        };
        let water = &self.water[level];
        Ok(Some(Tile {
            plain: cut(plain),
            accentuated: cut(&self.heights[1][level]),
            water: (0..TILE_SIZE * TILE_SIZE)
                .map(|i| water.get_clamped(west + (i % TILE_SIZE) as isize, south + (i / TILE_SIZE) as isize))
                .collect(),
        }))
    }
}

impl From<Dataset> for Terrain {
    fn from(dataset: Dataset) -> Self {
        let Dataset { meta, mut plain, mut accentuated, orientation, water, lut } = dataset;
        assert!(meta.levels > TILED_LEVELS, "a terrain needs at least {} cells a side", 1 << TILED_LEVELS);

        let leaves = meta.size / LEAF_CELLS;
        let leaf_ranges = leaf_ranges(&plain[0], leaves, |_, _| true);

        let mut water_levels = vec![water];
        for _ in 0..TILED_LEVELS {
            water_levels.push(halve_mask(water_levels.last().expect("the mask chain has a base level")));
        }
        let coarse_water = water_levels.pop().expect("the mask chain has a coarse level");

        let coarse = [plain.split_off(TILED_LEVELS), accentuated.split_off(TILED_LEVELS)];
        // All of the square holds data; its own edge is found analytically.
        let coarse_side = meta.size >> TILED_LEVELS;
        Self {
            edge_distance: Grid::new(coarse_side, coarse_side, EDGE_DISTANCE_CAP_M),
            leaf_ranges,
            coarse,
            coarse_water,
            orientation,
            lut,
            tiles: Arc::new(MemoryTiles { heights: [plain, accentuated], water: water_levels }),
            index: None,
            meta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyramid::gradient;
    use crate::synthetic;

    #[test]
    fn extended_continues_the_slope_past_the_edge() {
        let ramp = Grid::from_fn(4, 4, |x, y| 10.0 * x as f32 + y as f32);
        assert_eq!(extended(&ramp, 2, 1), 21.0);
        assert_eq!(extended(&ramp, -1, 1), -9.0);
        assert_eq!(extended(&ramp, 4, 4), 44.0);
    }

    #[test]
    fn memory_tiles_carry_an_apron_that_reproduces_the_gradients() {
        let dataset = synthetic::cone_dataset(512).unwrap();
        let cell = dataset.meta.cell_size;
        let expected = gradient(&dataset.plain[0], cell);
        let terrain = Terrain::from(dataset);

        // The second tile along x: its apron reaches into the first tile on
        // the west and past the terrain's edge on the south and east.
        let tile = terrain.tiles.tile(0, 1, 0).unwrap().unwrap();
        // Tile texel (x, y) sits at apron index (x + 1, y + 1).
        let apron = |x: usize, y: usize| tile.plain[y * TILE_SPAN + x];
        for (x, y) in [(0, 0), (100, 37), (255, 255)] {
            let gx = (apron(x + 2, y + 1) - apron(x, y + 1)) / (2.0 * cell);
            let gy = (apron(x + 1, y + 2) - apron(x + 1, y)) / (2.0 * cell);
            let [ex, ey] = expected.get(TILE_SIZE + x, y);
            assert!((gx - ex).abs() < 1e-4 && (gy - ey).abs() < 1e-4, "({gx}, {gy}) vs ({ex}, {ey}) at ({x}, {y})");
        }
        assert!(terrain.tiles.tile(0, 2, 0).unwrap().is_none());
    }

    #[test]
    fn heights_in_matches_the_pyramid() {
        let dataset = synthetic::cone_dataset(512).unwrap();
        let level2 = dataset.plain[2].clone();
        let terrain = Terrain::from(dataset);
        let extent = terrain.meta.extent();
        assert_eq!(terrain.heights_in(2, [0.0, 0.0, extent, extent]).unwrap(), level2);

        let crop = terrain.heights_in(0, [2000.0, 1000.0, 3000.0, 1500.0]).unwrap();
        assert_eq!((crop.width(), crop.height()), (100, 50));
    }

    #[test]
    fn heights_in_sits_exactly_on_its_region() {
        // Every height is its own easting, at every level.
        let ramp = Grid::from_fn(512, 512, |x, _| 10.0 * (x as f32 + 0.5));
        let dataset = Dataset::build("ramp", "test", ramp, 10.0, [0.0; 2], &crate::BuildOptions::default());
        let terrain = Terrain::from(dataset.unwrap());

        // A region that starts half a cell off the grid of a tiled level and
        // a quarter cell off the grid of a whole one.
        for (level, west) in [(0, 1005.0), (2, 1220.0), (5, 1680.0)] {
            let cell = terrain.cell(level);
            let heights = terrain.heights_in(level, [west, 1600.0, west + 4.0 * cell, 1600.0 + 2.0 * cell]).unwrap();
            assert_eq!((heights.width(), heights.height()), (4, 2));
            for x in 0..4 {
                let easting = west + (x as f32 + 0.5) * cell;
                assert!(
                    (heights.get(x, 1) - easting).abs() < 0.01,
                    "level {level}: {} at {easting}",
                    heights.get(x, 1)
                );
            }
        }
    }

    #[test]
    fn leaves_without_data_are_marked() {
        let heights = Grid::from_fn(128, 64, |x, y| (x + y) as f32);
        let ranges = leaf_ranges(&heights, 4, |x, _| x == 0);
        // Columns 0 to 64 (one cell of margin) and rows 0 to 63.
        assert_eq!(ranges.get(0, 0), [0.0, 64.0 + 63.0]);
        assert_eq!(ranges.get(1, 0), NO_DATA);
        assert_eq!(ranges.get(0, 1), NO_DATA);
    }
}
