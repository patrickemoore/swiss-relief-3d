//! `relief-prep`: the offline stage of the renderer.
//!
//! Analyzes a terrain once and writes the dataset the viewer loads: height
//! and gradient pyramids, the landform orientation field, the lake mask and
//! the colour table. Nothing here depends on the camera.

mod compare;
mod country;
mod download;
mod geotiff;
mod mosaic;
mod region;
mod stac;
mod trails;
mod zipped;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::{Args, Parser, Subcommand};
use relief_terrain::dataset::DEFAULT_FLAT_BRIGHTNESS;
use relief_terrain::{pack, synthetic, BuildOptions, Dataset, Grid, Lut, Terrain};

use crate::region::Region;

#[derive(Parser)]
#[command(name = "relief-prep", version, about = "Builds terrain datasets for the Swiss relief viewer")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download swissALTI3D tiles for a region and build its dataset
    Build(BuildArgs),
    /// Download and build all of Switzerland as one streamed dataset
    Country(CountryArgs),
    /// Pack a dataset into the compact form the web viewer streams
    Web(WebArgs),
    /// Add the hiking trails of swissTLM3D to a dataset
    Trails(TrailArgs),
    /// Build a synthetic cone, the test target for gradient signs
    Synth {
        #[arg(long, default_value = "data/cone")]
        out: PathBuf,
        /// Side length in cells (a power of two)
        #[arg(long, default_value_t = 1024)]
        size: usize,
    },
    /// Rewrite a dataset's colour table with the default palette
    Lut {
        /// Dataset directory
        dataset: PathBuf,
    },
    /// Compare two PNG images of equal size with SSIM and correlation
    Compare { a: PathBuf, b: PathBuf },
}

#[derive(Args)]
struct TrailArgs {
    /// Dataset directory to add the trails to
    #[arg(long, default_value = "data/switzerland")]
    data: PathBuf,
    /// Where the swissTLM3D layer is kept once fetched
    #[arg(long, default_value = "data/swisstlm3d")]
    cache: PathBuf,
    /// The swissTLM3D release to fetch
    #[arg(long, default_value = trails::DEFAULT_RELEASE)]
    release: String,
}

#[derive(Args)]
struct BuildArgs {
    /// Named region, or a label for a custom one
    #[arg(long, default_value = "jungfrau")]
    region: String,
    /// LV95 easting of the south-west corner of a custom region
    #[arg(long, requires = "origin_n")]
    origin_e: Option<f64>,
    /// LV95 northing of the south-west corner of a custom region
    #[arg(long, requires = "origin_e")]
    origin_n: Option<f64>,
    /// Side length in cells (a power of two)
    #[arg(long, default_value_t = 4096)]
    size: usize,
    /// Cell size in metres (a multiple of the 2 m source pixels)
    #[arg(long, default_value_t = 10.0)]
    cell: f64,
    /// Output directory [default: data/<region>]
    #[arg(long)]
    out: Option<PathBuf>,
    /// Where downloaded tiles are cached
    #[arg(long, default_value = "data/tiles")]
    tiles: PathBuf,
    /// Parallel downloads
    #[arg(long, default_value_t = 12)]
    jobs: usize,
    /// Skip download and mosaic, and rerun the analysis on the dataset's
    /// existing level-0 heights
    #[arg(long)]
    reuse_dem: bool,
    /// Download only each tile's 4 m overview instead of its 2 m data, as the
    /// country build does: a fifth of the download for slightly noisier cells
    #[arg(long)]
    overviews: bool,
}

#[derive(Args)]
struct CountryArgs {
    /// Output directory
    #[arg(long, default_value = "data/switzerland")]
    out: PathBuf,
    /// Where the tile overviews are cached (about 10 GB)
    #[arg(long, default_value = "data/tiles_4m")]
    tiles: PathBuf,
    /// Parallel downloads
    #[arg(long, default_value_t = 16)]
    jobs: usize,
    /// Ask swisstopo for the tile list again instead of using the cached one
    #[arg(long)]
    relist: bool,
    /// Download the tiles and stop
    #[arg(long)]
    fetch_only: bool,
}

#[derive(Args)]
struct WebArgs {
    /// Dataset directory to pack
    #[arg(long, default_value = "data/switzerland")]
    data: PathBuf,
    /// Output directory, to be served as static files
    #[arg(long, default_value = "web/demo/public/terrain")]
    out: PathBuf,
    /// Finest pyramid level to include: 0 keeps the 10 m cells, 1 stops at
    /// 20 m, 2 at 40 m. Each level left out makes the result about three
    /// times smaller
    #[arg(long, default_value_t = 0)]
    finest: usize,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Build(args) => build(&args),
        Command::Country(args) => build_country(&args),
        Command::Web(args) => pack_for_web(&args),
        Command::Trails(args) => trails::build(&args.data, &args.cache, &args.release),
        Command::Synth { out, size } => save(&synthetic::cone_dataset(size)?, &out),
        Command::Lut { dataset } => {
            let path = dataset.join("lut.png");
            Lut::swiss_default(DEFAULT_FLAT_BRIGHTNESS).save_png(&path)?;
            println!("wrote {}", path.display());
            Ok(())
        }
        Command::Compare { a, b } => {
            let similarity = compare::compare(&a, &b)?;
            println!("SSIM        {:.4}", similarity.ssim);
            println!("correlation {:.4}", similarity.correlation);
            Ok(())
        }
    }
}

fn pack_for_web(args: &WebArgs) -> Result<()> {
    let terrain =
        Terrain::open(&args.data).with_context(|| format!("loading the dataset in {}", args.data.display()))?;
    let started = Instant::now();
    let report = pack::export(&terrain, &args.out, args.finest)?;
    // Trails come along as they are: their tiles are already compact.
    if terrain.trails.is_some() {
        let (from, to) =
            (args.data.join(relief_terrain::trails::TRAILS_DIR), args.out.join(relief_terrain::trails::TRAILS_DIR));
        copy_dir(&from, &to).with_context(|| format!("copying {} to {}", from.display(), to.display()))?;
    }

    const MEGABYTE: f64 = 1_048_576.0;
    println!("wrote {} in {:.1} s", args.out.display(), started.elapsed().as_secs_f32());
    println!("  loaded up front  {:8.1} MB", report.base_bytes as f64 / MEGABYTE);
    for (level, (&tiles, &bytes)) in
        report.tiles.iter().zip(&report.tile_bytes).enumerate().filter(|level| *level.1 .0 > 0)
    {
        let cell = terrain.cell(level);
        println!("  level {level} ({cell:3.0} m)  {:8.1} MB in {tiles} tiles", bytes as f64 / MEGABYTE);
    }
    let total = report.base_bytes + report.tile_bytes.iter().sum::<u64>();
    println!("  in all           {:8.1} MB", total as f64 / MEGABYTE);
    Ok(())
}

fn build_country(args: &CountryArgs) -> Result<()> {
    let paths = country::fetch(&args.tiles, args.jobs, args.relist)?;
    if args.fetch_only {
        println!("{} tile overviews are in {}", paths.len(), args.tiles.display());
        return Ok(());
    }
    let started = Instant::now();
    let meta = country::build(&paths, &args.out)?;
    println!(
        "wrote {}: {:.0} km square at {} m, elevation {:.0} to {:.0} m",
        args.out.display(),
        meta.extent() / 1000.0,
        meta.cell_size,
        meta.elevation.min,
        meta.elevation.max
    );
    println!("done in {:.1} s", started.elapsed().as_secs_f32());
    Ok(())
}

fn build(args: &BuildArgs) -> Result<()> {
    let origin = args.origin_e.zip(args.origin_n).map(|(east, north)| [east, north]);
    let region = Region::resolve(&args.region, origin, args.size, args.cell)?;
    let out = args.out.clone().unwrap_or_else(|| Path::new("data").join(&region.name));

    let started = Instant::now();
    let dem = if args.reuse_dem {
        load_existing_dem(&out, &region)?
    } else {
        let agent = download::http_agent()?;
        println!("listing swissALTI3D tiles for '{}'", region.name);
        let tiles = stac::list_tiles(&agent, &region)?;
        ensure!(!tiles.is_empty(), "swissALTI3D has no tiles in this region");
        let (paths, read): (_, fn(&Path) -> Result<geotiff::Tile>) = if args.overviews {
            (download::download_overviews(&agent, &tiles, &args.tiles, args.jobs)?, geotiff::read_overview)
        } else {
            (download::download_tiles(&agent, &tiles, &args.tiles, args.jobs)?, geotiff::read_tile)
        };
        println!("mosaicking {} tiles at {} m", paths.len(), region.cell);
        mosaic::build_dem(&paths, &region, read)?
    };

    println!("analyzing terrain");
    let dataset = Dataset::build(
        &region.name,
        "swisstopo swissALTI3D, 2 m, area-averaged",
        dem,
        region.cell as f32,
        region.origin,
        &BuildOptions::default(),
    )?;
    save(&dataset, &out)?;
    println!("done in {:.1} s", started.elapsed().as_secs_f32());
    Ok(())
}

/// Reads level 0 of an existing dataset as the DEM.
fn load_existing_dem(dir: &Path, region: &Region) -> Result<Grid<f32>> {
    let path = dir.join("height_00.f32");
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        bytes.len() == region.size * region.size * 4,
        "{} does not match a {} x {} region",
        path.display(),
        region.size,
        region.size
    );
    let heights = bytes.as_chunks::<4>().0.iter().map(|&bytes| f32::from_le_bytes(bytes)).collect();
    Ok(Grid::from_vec(region.size, region.size, heights))
}

fn save(dataset: &Dataset, dir: &Path) -> Result<()> {
    dataset.save(dir)?;
    let meta = &dataset.meta;
    let lake_cells = dataset.water.data().iter().filter(|&&cell| cell > 0).count();
    println!(
        "wrote {}: {} x {} cells at {} m, {} levels",
        dir.display(),
        meta.size,
        meta.size,
        meta.cell_size,
        meta.levels
    );
    println!(
        "  elevation {:.0} to {:.0} m (5th to 95th percentile {:.0} to {:.0} m)",
        meta.elevation.min, meta.elevation.max, meta.elevation.p5, meta.elevation.p95
    );
    println!("  lakes cover {:.1} km2", lake_cells as f32 * meta.cell_size * meta.cell_size / 1e6);
    Ok(())
}

/// Copies a directory tree, replacing what is at `to`.
fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    if to.exists() {
        std::fs::remove_dir_all(to)?;
    }
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}
