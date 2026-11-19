//! A small terrain analyzed and held in memory as a whole.
//!
//! This is the form of a region of a few thousand cells a side, and of the
//! synthetic cone. Every pyramid level is a file of raw little-endian floats
//! next to a small `meta.json`, which is easier to inspect and debug than an
//! image container:
//!
//! ```text
//! meta.json               sizes, cell size, LV95 origin, elevation statistics
//! height_00.f32 ...       plain height pyramid, one file per level
//! height_accent_00.f32    ridge-accentuated height pyramid
//! orientation.f32         doubled-angle landform orientation pairs
//! water.u8                lake mask
//! lut.png                 colour table, hand-editable
//! ```
//!
//! Terrains too large for that are stored as tiles; see [`crate::tiled`].

use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::grid::Grid;
use crate::lut::Lut;
use crate::orientation::{orientation_field, OrientationParams};
use crate::pack::PackInfo;
use crate::pyramid::{build_heights, Accentuation};
use crate::stats::ElevationStats;
use crate::water::{lake_mask, LakeParams};

#[cfg(not(target_endian = "little"))]
compile_error!("dataset files are little-endian and are read without byte swapping");

/// Brightness of flat ground the default colour table is built around.
pub const DEFAULT_FLAT_BRIGHTNESS: f32 = 0.75;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    pub name: String,
    /// Where the elevation data came from.
    pub source: String,
    /// Side length of the terrain square at level 0, in cells.
    pub size: usize,
    /// Cell size of level 0 in metres.
    pub cell_size: f32,
    /// LV95 easting and northing of the south-west corner, kept in double
    /// precision: the renderer works in local metres from this corner.
    pub origin_lv95: [f64; 2],
    pub levels: usize,
    pub orientation_size: usize,
    pub elevation: ElevationStats,
    pub accentuation: Accentuation,
    /// Whether the finest levels are stored as tiles and streamed.
    #[serde(default)]
    pub tiled: bool,
    /// The part of the square that holds data, as west, south, east and north
    /// in local metres. `None` if all of it does.
    #[serde(default)]
    pub data_region: Option<[f32; 4]>,
    /// The part of the terrain the built-in viewpoints and flights are laid
    /// out in, in the same form. `None` uses all of it.
    #[serde(default)]
    pub view_region: Option<[f32; 4]>,
    /// Set if the terrain is packed for the web; see [`crate::pack`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed: Option<PackInfo>,
    /// Hiking trails, if `relief-prep trails` has added them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trails: Option<crate::trails::TrailsInfo>,
}

impl Meta {
    /// Side length of the terrain square in metres.
    pub fn extent(&self) -> f32 {
        self.size as f32 * self.cell_size
    }

    /// The region that holds data, as west, south, east and north in metres.
    pub fn data_region(&self) -> [f32; 4] {
        self.data_region.unwrap_or([0.0, 0.0, self.extent(), self.extent()])
    }

    /// The region built-in viewpoints and flights are laid out in.
    pub fn view_region(&self) -> [f32; 4] {
        self.view_region.unwrap_or_else(|| self.data_region())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BuildOptions {
    pub accentuation: Accentuation,
    pub orientation: OrientationParams,
    pub lakes: LakeParams,
}

/// A whole terrain in memory.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub meta: Meta,
    /// The generalized height pyramid, finest level first.
    pub plain: Vec<Grid<f32>>,
    /// The same with major ridges and valleys kept sharp.
    pub accentuated: Vec<Grid<f32>>,
    pub orientation: Grid<[f32; 2]>,
    pub water: Grid<u8>,
    pub lut: Lut,
}

impl Dataset {
    /// Runs the whole terrain analysis on a square, power-of-two DEM.
    pub fn build(
        name: &str,
        source: &str,
        dem: Grid<f32>,
        cell_size: f32,
        origin_lv95: [f64; 2],
        options: &BuildOptions,
    ) -> Result<Self> {
        let size = dem.width();
        ensure!(dem.height() == size && size.is_power_of_two(), "the DEM must be square with a power-of-two size");

        let elevation = ElevationStats::compute(&dem);
        let water = lake_mask(&dem, cell_size, &options.lakes);
        let plain = build_heights(dem.clone(), cell_size, None);
        let accentuated = build_heights(dem, cell_size, Some(&options.accentuation));
        let orientation_params =
            OrientationParams { output_size: options.orientation.output_size.min(size), ..options.orientation };
        let orientation = orientation_field(&plain, cell_size, &orientation_params);

        let meta = Meta {
            name: name.to_owned(),
            source: source.to_owned(),
            size,
            cell_size,
            origin_lv95,
            levels: plain.len(),
            orientation_size: orientation.width(),
            elevation,
            accentuation: options.accentuation,
            tiled: false,
            data_region: None,
            view_region: None,
            packed: None,
            trails: None,
        };
        Ok(Self { meta, plain, accentuated, orientation, water, lut: Lut::swiss_default(DEFAULT_FLAT_BRIGHTNESS) })
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        for (prefix, pyramid) in [("", &self.plain), ("accent_", &self.accentuated)] {
            for (level, grid) in pyramid.iter().enumerate() {
                write_raw(&dir.join(format!("height_{prefix}{level:02}.f32")), grid.data())?;
            }
        }
        write_raw(&dir.join("orientation.f32"), self.orientation.data())?;
        write_raw(&dir.join("water.u8"), self.water.data())?;
        self.lut.save_png(&dir.join("lut.png"))?;
        // Written last, so a directory with a meta file is a complete dataset.
        write_meta(dir, &self.meta)
    }

    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with(dir, read_meta(dir)?)
    }

    pub(crate) fn load_with(dir: &Path, meta: Meta) -> Result<Self> {
        let load_pyramid = |prefix: &str| -> Result<Vec<Grid<f32>>> {
            (0..meta.levels)
                .map(|level| {
                    let side = (meta.size >> level).max(1);
                    read_raw(&dir.join(format!("height_{prefix}{level:02}.f32")), side, side)
                })
                .collect()
        };
        Ok(Self {
            plain: load_pyramid("")?,
            accentuated: load_pyramid("accent_")?,
            orientation: read_raw(&dir.join("orientation.f32"), meta.orientation_size, meta.orientation_size)?,
            water: read_raw(&dir.join("water.u8"), meta.size, meta.size)?,
            lut: Lut::load_png(&dir.join("lut.png"))?,
            meta,
        })
    }
}

pub fn write_meta(dir: &Path, meta: &Meta) -> Result<()> {
    let path = dir.join("meta.json");
    fs::write(&path, serde_json::to_string_pretty(meta)?).with_context(|| format!("writing {}", path.display()))
}

pub fn read_meta(dir: &Path) -> Result<Meta> {
    let path = dir.join("meta.json");
    let text =
        fs::read_to_string(&path).with_context(|| format!("reading {} (run relief-prep first)", path.display()))?;
    serde_json::from_str(&text).context("parsing meta.json")
}

pub(crate) fn write_raw<T: bytemuck::Pod>(path: &Path, data: &[T]) -> Result<()> {
    fs::write(path, bytemuck::cast_slice(data)).with_context(|| format!("writing {}", path.display()))
}

pub(crate) fn read_raw<T: bytemuck::Pod + Send + Sync>(path: &Path, width: usize, height: usize) -> Result<Grid<T>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        bytes.len() == width * height * std::mem::size_of::<T>(),
        "{} does not hold a {width} x {height} raster",
        path.display()
    );
    Ok(Grid::from_vec(width, height, bytemuck::pod_collect_to_vec(&bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyramid::gradient;
    use crate::synthetic;

    #[test]
    fn build_save_load_round_trip() {
        let dataset = synthetic::cone_dataset(64).unwrap();
        assert_eq!(dataset.meta.levels, 7);

        let dir = std::env::temp_dir().join("relief_terrain_dataset_round_trip");
        dataset.save(&dir).unwrap();
        let loaded = Dataset::load(&dir).unwrap();
        fs::remove_dir_all(&dir).ok();

        assert_eq!(loaded.meta, dataset.meta);
        assert_eq!(loaded.plain, dataset.plain);
        assert_eq!(loaded.accentuated, dataset.accentuated);
        assert_eq!(loaded.orientation, dataset.orientation);
        assert_eq!(loaded.lut, dataset.lut);
    }

    #[test]
    fn older_meta_files_still_parse() {
        // Datasets written before tiling have none of the later fields.
        let dataset = synthetic::cone_dataset(64).unwrap();
        let mut json = serde_json::to_value(&dataset.meta).unwrap();
        for field in ["tiled", "data_region", "view_region"] {
            json.as_object_mut().unwrap().remove(field);
        }
        let meta: Meta = serde_json::from_value(json).unwrap();
        assert_eq!(meta, dataset.meta);
        assert_eq!(meta.view_region(), [0.0, 0.0, 640.0, 640.0]);
    }

    #[test]
    fn cone_north_west_flank_faces_north_west() {
        // On the north-west flank the terrain rises toward the south-east,
        // so dh/dx is positive and dh/dy (toward north) is negative.
        let dataset = synthetic::cone_dataset(256).unwrap();
        let [gx, gy] = gradient(&dataset.plain[0], dataset.meta.cell_size).get(96, 160);
        assert!(gx > 0.1 && gy < -0.1, "got ({gx}, {gy})");
    }
}
