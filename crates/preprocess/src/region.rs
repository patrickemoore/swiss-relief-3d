//! The square region a dataset covers, in LV95 coordinates.

use std::ops::RangeInclusive;

use anyhow::{bail, ensure, Result};

/// swissALTI3D is delivered in tiles of 1 km x 1 km.
pub const TILE_SIZE_M: f64 = 1000.0;

/// Named regions as `(name, easting, northing)` of the south-west corner.
///
/// Each is a 40.96 km square that lies entirely inside Switzerland, so the
/// mosaic needs no data from across the border.
const PRESETS: [(&str, f64, f64); 1] = [
    // Lake Thun and Lake Brienz up to the Jungfrau, the Aletsch glacier and
    // the Finsteraarhorn: 560 m to 4274 m in one frame.
    ("jungfrau", 2_620_000.0, 1_136_000.0),
];

#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub name: String,
    /// LV95 easting and northing of the south-west corner.
    pub origin: [f64; 2],
    /// Side length in output cells.
    pub size: usize,
    /// Output cell size in metres.
    pub cell: f64,
}

impl Region {
    /// Resolves a preset name, or a custom region when an origin is given.
    pub fn resolve(name: &str, origin: Option<[f64; 2]>, size: usize, cell: f64) -> Result<Self> {
        ensure!(size.is_power_of_two(), "the region size must be a power of two, got {size}");
        let origin = match origin {
            Some(origin) => origin,
            None => match PRESETS.iter().find(|(preset, ..)| *preset == name) {
                Some(&(_, east, north)) => [east, north],
                None => bail!("unknown region '{name}': pass --origin-e and --origin-n to define it"),
            },
        };
        Ok(Self { name: name.to_owned(), origin, size, cell })
    }

    pub fn extent(&self) -> f64 {
        self.size as f64 * self.cell
    }

    /// Kilometre indices of the tiles that overlap the region, east then north.
    pub fn tile_ranges(&self) -> (RangeInclusive<u32>, RangeInclusive<u32>) {
        let range = |origin: f64| {
            let first = (origin / TILE_SIZE_M).floor() as u32;
            let last = ((origin + self.extent()) / TILE_SIZE_M).ceil() as u32 - 1;
            first..=last
        };
        (range(self.origin[0]), range(self.origin[1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jungfrau_needs_41_by_41_tiles() {
        let region = Region::resolve("jungfrau", None, 4096, 10.0).unwrap();
        let (east, north) = region.tile_ranges();
        assert_eq!((east, north), (2620..=2660, 1136..=1176));
    }

    #[test]
    fn unknown_region_needs_an_origin() {
        assert!(Region::resolve("nowhere", None, 4096, 10.0).is_err());
        assert!(Region::resolve("nowhere", Some([2_700_000.0, 1_200_000.0]), 1024, 10.0).is_ok());
    }
}
