//! All of Switzerland as one dataset.
//!
//! The country is far too large to analyze from 2 m tiles the way a 41 km
//! region is: that would be a 49 GB download. Every tile carries a 4 m
//! overview at its start, so only that part of each file is fetched, and the
//! overviews are area-averaged to the same 10 m cells.

use std::fs;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use relief_terrain::tiled::{self, TiledInput};
use relief_terrain::{BuildOptions, Meta};

use crate::download;
use crate::geotiff;
use crate::mosaic::{self, Raster};
use crate::region::TILE_SIZE_M;
use crate::stac::{self, TileRef, COUNTRY_LISTING_STRIPS};

/// LV95 easting and northing of the south-west corner of the terrain square.
pub const ORIGIN: [f64; 2] = [2_480_000.0, 1_070_000.0];
/// Cell size of the finest level, in metres.
pub const CELL_M: f64 = 10.0;
/// The part of the square, from its south-west corner, that Switzerland lies
/// in, as cells east and north. Both are multiples of the coarsest tile.
pub const DATA_CELLS: [usize; 2] = [36_864, 24_576];
/// Side length of the terrain square in cells: the power of two that holds
/// the data area.
const SQUARE_CELLS: usize = 65_536;
/// Where the built-in viewpoints and flights are laid out, in metres from the
/// origin: the Jungfrau region, the same square the regional dataset covers.
const VIEW_REGION: [f32; 4] = [140_000.0, 66_000.0, 180_960.0, 106_960.0];

/// File in the tile cache that remembers the tile listing.
const INDEX_FILE: &str = "index.json";

/// Kilometre indices of the source tiles along one axis of the data area.
fn tile_range(axis: usize) -> RangeInclusive<u32> {
    let first = (ORIGIN[axis] / TILE_SIZE_M) as u32;
    let last = ((ORIGIN[axis] + DATA_CELLS[axis] as f64 * CELL_M) / TILE_SIZE_M).ceil() as u32 - 1;
    first..=last
}

/// Lists every tile of the country, or reads the listing cached by an
/// earlier run. The listing takes a few hundred requests.
fn list(agent: &ureq::Agent, cache: &Path, relist: bool) -> Result<Vec<TileRef>> {
    let index = cache.join(INDEX_FILE);
    if !relist && index.exists() {
        let text = fs::read_to_string(&index).with_context(|| format!("reading {}", index.display()))?;
        return serde_json::from_str(&text).with_context(|| format!("parsing {}", index.display()));
    }
    println!("listing the swissALTI3D tiles of Switzerland");
    let tiles = stac::list_tile_ranges(agent, tile_range(0), tile_range(1), COUNTRY_LISTING_STRIPS)?;
    fs::create_dir_all(cache).with_context(|| format!("creating {}", cache.display()))?;
    fs::write(&index, serde_json::to_string(&tiles)?).with_context(|| format!("writing {}", index.display()))?;
    Ok(tiles)
}

/// Downloads the overview of every tile into `cache` and returns the paths.
pub fn fetch(cache: &Path, jobs: usize, relist: bool) -> Result<Vec<PathBuf>> {
    let agent = download::http_agent()?;
    let tiles = list(&agent, cache, relist)?;
    println!("{} tiles cover Switzerland", tiles.len());
    download::download_overviews(&agent, &tiles, cache, jobs)
}

/// Mosaics the overviews and writes the country as a tiled dataset.
pub fn build(tile_paths: &[PathBuf], out: &Path) -> Result<Meta> {
    let raster = Raster { origin: ORIGIN, cell: CELL_M, width: DATA_CELLS[0], height: DATA_CELLS[1] };
    println!("mosaicking {} tiles at {CELL_M} m", tile_paths.len());
    let dem = mosaic::mosaic(tile_paths, &raster, geotiff::read_overview)?;

    println!("analyzing terrain");
    let input = TiledInput {
        name: "switzerland".to_owned(),
        source: "swisstopo swissALTI3D, 4 m overviews, area-averaged".to_owned(),
        dem,
        cell_size: CELL_M as f32,
        origin_lv95: ORIGIN,
        square: SQUARE_CELLS,
        view_region: Some(VIEW_REGION),
    };
    tiled::build(input, &BuildOptions::default(), out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_data_area_holds_all_of_switzerland() {
        // The country spans about E 2 485 000 to 2 834 000 and
        // N 1 075 000 to 1 296 000.
        assert_eq!(tile_range(0), 2480..=2848);
        assert_eq!(tile_range(1), 1070..=1315);
    }
}
