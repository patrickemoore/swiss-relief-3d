//! Tile discovery through swisstopo's STAC API.

use std::collections::BTreeMap;
use std::ops::RangeInclusive;
use std::thread;

use anyhow::{Context, Result};
use relief_terrain::lv95;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::region::{Region, TILE_SIZE_M};

const ITEMS_URL: &str = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items";
/// Ground sample distance of the product to download, in metres. The 0.5 m
/// product is far more detail than a flyaround can show.
const SOURCE_GSD: f64 = 2.0;
/// Latitude strips a country-sized listing is split into, each paged on its
/// own connection.
pub const COUNTRY_LISTING_STRIPS: usize = 24;

/// One downloadable swissALTI3D tile.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TileRef {
    /// Kilometre index of the western edge, e.g. 2639.
    pub east_km: u32,
    /// Kilometre index of the southern edge, e.g. 1153.
    pub north_km: u32,
    pub year: u32,
    pub url: String,
}

impl TileRef {
    /// File name of the cached tile; `suffix` tells full tiles (`2m`) from
    /// their overviews (`4m`).
    pub fn file_name(&self, suffix: &str) -> String {
        format!("swissalti3d_{}_{}-{}_{suffix}.tif", self.year, self.east_km, self.north_km)
    }
}

/// The newest tile known for each kilometre square.
type Newest = BTreeMap<(u32, u32), TileRef>;

/// Lists the newest 2 m tile for every kilometre square of a region.
pub fn list_tiles(agent: &ureq::Agent, region: &Region) -> Result<Vec<TileRef>> {
    let (east_km, north_km) = region.tile_ranges();
    list_tile_ranges(agent, east_km, north_km, 1)
}

/// Lists the newest 2 m tile for every kilometre square in the given ranges
/// of kilometre indices, querying `strips` latitude bands in parallel.
pub fn list_tile_ranges(
    agent: &ureq::Agent,
    east_km: RangeInclusive<u32>,
    north_km: RangeInclusive<u32>,
    strips: usize,
) -> Result<Vec<TileRef>> {
    let corner = |east: u32, north: u32| lv95::to_wgs84(east as f64 * TILE_SIZE_M, north as f64 * TILE_SIZE_M);
    // The LV95 grid is slightly rotated against the graticule, so take the
    // extremes of all four corners.
    let corners = [
        corner(*east_km.start(), *north_km.start()),
        corner(*east_km.end() + 1, *north_km.start()),
        corner(*east_km.start(), *north_km.end() + 1),
        corner(*east_km.end() + 1, *north_km.end() + 1),
    ];
    let extreme = |pick: fn(&(f64, f64)) -> f64, fold: fn(f64, f64) -> f64, start: f64| {
        corners.iter().map(pick).fold(start, fold)
    };
    let (west, east) = (extreme(|c| c.0, f64::min, f64::MAX), extreme(|c| c.0, f64::max, f64::MIN));
    let (south, north) = (extreme(|c| c.1, f64::min, f64::MAX), extreme(|c| c.1, f64::max, f64::MIN));

    let strips = strips.max(1);
    let strip_height = (north - south) / strips as f64;
    let listings: Vec<Result<Newest>> = thread::scope(|scope| {
        let workers: Vec<_> = (0..strips)
            .map(|strip| {
                let (east_km, north_km) = (east_km.clone(), north_km.clone());
                scope.spawn(move || {
                    let strip_south = south + strip as f64 * strip_height;
                    let bbox = format!("{west:.5},{strip_south:.5},{east:.5},{:.5}", strip_south + strip_height);
                    list_box(agent, &bbox, |tile| east_km.contains(&tile.east_km) && north_km.contains(&tile.north_km))
                })
            })
            .collect();
        workers.into_iter().map(|worker| worker.join().expect("listing worker panicked")).collect()
    });

    let mut newest = Newest::new();
    for listing in listings {
        for tile in listing?.into_values() {
            keep_newest(&mut newest, tile);
        }
    }
    Ok(newest.into_values().collect())
}

/// Pages through every item that intersects a `west,south,east,north` box.
fn list_box(agent: &ureq::Agent, bbox: &str, wanted: impl Fn(&TileRef) -> bool) -> Result<Newest> {
    let mut newest = Newest::new();
    let mut next_url = Some(format!("{ITEMS_URL}?bbox={bbox}&limit=100"));
    while let Some(url) = next_url {
        let body = agent.get(&url).call().with_context(|| format!("requesting {url}"))?.into_string()?;
        let page: Value = serde_json::from_str(&body).context("parsing the STAC response")?;

        for tile in page["features"].as_array().into_iter().flatten().filter_map(parse_item) {
            if wanted(&tile) {
                keep_newest(&mut newest, tile);
            }
        }
        next_url = page["links"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|link| link["rel"] == "next")
            .and_then(|link| link["href"].as_str())
            .map(str::to_owned);
    }
    Ok(newest)
}

fn keep_newest(newest: &mut Newest, tile: TileRef) {
    let key = (tile.east_km, tile.north_km);
    if newest.get(&key).is_none_or(|known| known.year < tile.year) {
        newest.insert(key, tile);
    }
}

/// Parses an item such as `swissalti3d_2019_2639-1153` and picks its 2 m
/// GeoTIFF asset.
fn parse_item(item: &Value) -> Option<TileRef> {
    let mut parts = item["id"].as_str()?.split('_').skip(1);
    let year = parts.next()?.parse().ok()?;
    let (east_km, north_km) = parts.next()?.split_once('-')?;
    let url = item["assets"]
        .as_object()?
        .iter()
        .find(|(name, asset)| name.ends_with(".tif") && asset["eo:gsd"].as_f64() == Some(SOURCE_GSD))
        .and_then(|(_, asset)| asset["href"].as_str())?;
    Some(TileRef { east_km: east_km.parse().ok()?, north_km: north_km.parse().ok()?, year, url: url.to_owned() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(year: u32) -> TileRef {
        TileRef { east_km: 2639, north_km: 1153, year, url: format!("https://example/{year}.tif") }
    }

    #[test]
    fn parses_a_stac_item() {
        let item: Value = serde_json::json!({
            "id": "swissalti3d_2019_2639-1153",
            "assets": {
                "swissalti3d_2019_2639-1153_0.5_2056_5728.tif": { "href": "https://example/half.tif", "eo:gsd": 0.5 },
                "swissalti3d_2019_2639-1153_2_2056_5728.tif": { "href": "https://example/two.tif", "eo:gsd": 2.0 },
                "swissalti3d_2019_2639-1153_2_2056_5728.xyz.zip": { "href": "https://example/two.zip", "eo:gsd": 2.0 }
            }
        });
        let tile = parse_item(&item).unwrap();
        assert_eq!((tile.east_km, tile.north_km, tile.year), (2639, 1153, 2019));
        assert_eq!(tile.url, "https://example/two.tif");
    }

    #[test]
    fn newer_surveys_replace_older_ones() {
        let mut newest = Newest::new();
        for year in [2019, 2024, 2021] {
            keep_newest(&mut newest, tile(year));
        }
        assert_eq!(newest.len(), 1);
        assert_eq!(newest[&(2639, 1153)].year, 2024);
    }

    #[test]
    fn cached_names_tell_tiles_from_overviews() {
        assert_eq!(tile(2024).file_name("2m"), "swissalti3d_2024_2639-1153_2m.tif");
        assert_eq!(tile(2024).file_name("4m"), "swissalti3d_2024_2639-1153_4m.tif");
    }
}
