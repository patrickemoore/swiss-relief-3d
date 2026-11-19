//! Hiking trails from swissTLM3D into a dataset.
//!
//! swissTLM3D's road layer carries every path in the country with its
//! hiking classification. The three files of that layer are fetched out of
//! swisstopo's zip (see `zipped`), read as a Shapefile, filtered to the
//! classified trails, moved into the dataset's local metres and written as
//! tiles per level by `relief_terrain::trails`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use relief_terrain::dataset::{read_meta, write_meta};
use relief_terrain::terrain::TILED_LEVELS;
use relief_terrain::trails::{self, Polyline, TrailClass, TrailsInfo};
use shapefile::dbase::FieldValue;
use shapefile::Shape;

use crate::zipped;

/// The release of swissTLM3D to take the trails from.
pub const DEFAULT_RELEASE: &str = "swisstlm3d_2026-02";
/// The layer with the paths, and the attribute with the classification.
const LAYER: &str = "TLM_STRASSEN/swissTLM3D_TLM_STRASSE";
const CLASS_FIELD: &str = "WANDERWEGE";

fn archive_url(release: &str) -> String {
    format!("https://data.geo.admin.ch/ch.swisstopo.swisstlm3d/{release}/{release}_2056_5728.shp.zip")
}

/// Makes sure the road layer is in `cache`, fetching it if not.
pub fn fetch_layer(release: &str, cache: &Path) -> Result<PathBuf> {
    let url = archive_url(release);
    let names = ["shp", "shx", "dbf", "cpg"].map(|extension| format!("{LAYER}.{extension}"));
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let agent = crate::download::http_agent()?;
    let paths = zipped::ensure_files(&agent, &url, cache, &names)?;
    Ok(paths[0].clone())
}

/// Classified paths in LV95 metres.
type RawTrails = Vec<(TrailClass, Vec<[f64; 2]>)>;

/// Reads the classified trails of the layer, in LV95 metres, and counts
/// every value the classification takes.
pub fn read_trails(shp: &Path) -> Result<(RawTrails, BTreeMap<String, usize>)> {
    let mut reader = shapefile::Reader::from_path(shp).with_context(|| format!("opening {}", shp.display()))?;
    let mut trails = Vec::new();
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    let mut field_missing = true;
    for item in reader.iter_shapes_and_records() {
        let (shape, record) = item.with_context(|| format!("reading {}", shp.display()))?;
        let value = match record.get(CLASS_FIELD) {
            Some(FieldValue::Character(Some(value))) => value.trim().to_owned(),
            Some(FieldValue::Character(None)) => String::new(),
            Some(other) => format!("{other:?}"),
            None => {
                if field_missing {
                    let fields: Vec<String> = record.into_iter().map(|(name, _)| name).collect();
                    bail!("{} has no {CLASS_FIELD} field; its fields are {}", shp.display(), fields.join(", "));
                }
                continue;
            }
        };
        field_missing = false;
        *seen.entry(value.clone()).or_default() += 1;
        let Some(class) = TrailClass::from_swisstlm3d(&value) else { continue };
        let parts: Vec<Vec<[f64; 2]>> = match shape {
            Shape::PolylineZ(line) => {
                line.parts().iter().map(|part| part.iter().map(|p| [p.x, p.y]).collect()).collect()
            }
            Shape::Polyline(line) => {
                line.parts().iter().map(|part| part.iter().map(|p| [p.x, p.y]).collect()).collect()
            }
            Shape::PolylineM(line) => {
                line.parts().iter().map(|part| part.iter().map(|p| [p.x, p.y]).collect()).collect()
            }
            _ => continue,
        };
        for part in parts {
            if part.len() >= 2 {
                trails.push((class, part));
            }
        }
    }
    Ok((trails, seen))
}

/// Builds the trails of the dataset in `data` from the layer in `cache`.
pub fn build(data: &Path, cache: &Path, release: &str) -> Result<()> {
    let meta = read_meta(data).with_context(|| format!("loading the dataset in {}", data.display()))?;
    let shp = fetch_layer(release, cache)?;

    let started = Instant::now();
    let (raw, seen) = read_trails(&shp)?;
    println!("read {} classified paths in {:.1} s", raw.len(), started.elapsed().as_secs_f32());
    for (value, count) in &seen {
        let shown = if value.is_empty() { "(none)" } else { value.as_str() };
        println!("  {shown:16} {count:8}");
    }

    // Into local metres, keeping what touches the terrain square.
    let extent = meta.extent() as f64;
    let [east, north] = meta.origin_lv95;
    let polylines: Vec<Polyline> = raw
        .into_iter()
        .filter_map(|(class, points)| {
            let local: Vec<[f32; 2]> = points.iter().map(|[x, y]| [(x - east) as f32, (y - north) as f32]).collect();
            let inside = points
                .iter()
                .any(|[x, y]| (0.0..=extent).contains(&(x - east)) && (0.0..=extent).contains(&(y - north)));
            inside.then_some(Polyline { class, points: local })
        })
        .collect();
    if polylines.is_empty() {
        bail!("no trails fall inside the dataset");
    }

    let started = Instant::now();
    let (_, report) = trails::export(&polylines, &meta, data)?;
    println!(
        "wrote {} trails, {:.0} km, in {:.1} s",
        report.polylines,
        report.kilometres,
        started.elapsed().as_secs_f32()
    );
    for level in 0..TILED_LEVELS {
        println!(
            "  level {level} ({:4.0} m)  {:6.1} MB in {} tiles",
            meta.cell_size * (1 << level) as f32,
            report.bytes[level] as f64 / 1_048_576.0,
            report.tiles[level]
        );
    }
    let info = TrailsInfo {
        source: format!("swissTLM3D {release}"),
        polylines: report.polylines,
        kilometres: report.kilometres,
    };
    write_meta(data, &relief_terrain::Meta { trails: Some(info), ..meta })?;
    Ok(())
}
