//! `relief-viewer`: real-time 3D terrain with camera-relative Swiss-style
//! relief shading.
//!
//! Without a subcommand it opens the interactive viewer. `snapshot` renders
//! a single frame to a PNG and `evaluate` runs the evaluation of the plan's
//! section 10; neither opens a window.

mod app;
mod capture;
mod evaluate;
mod headless;
mod session;
mod ui;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use glam::{Vec2, Vec3};
use relief_terrain::{synthetic, Terrain};

use headless::{Headless, DEFAULT_FOV_Y};
use relief_render::camera::{OrbitView, Pose};
use relief_render::light_rig::LightRig;
use relief_render::params::{Ablation, Condition, DebugView, ShadingParams};
use relief_render::path::{self, Scene, Viewpoint};

/// Side length of the synthetic cone in cells.
const CONE_SIZE: usize = 1024;
/// Where relief-prep writes the country and the default region.
const COUNTRY_DATA: &str = "data/switzerland";
const REGION_DATA: &str = "data/jungfrau";

#[derive(Parser)]
#[command(name = "relief-viewer", version, about = "Real-time Swiss-style relief shading for 3D terrain")]
struct Cli {
    /// Dataset directory written by relief-prep [default: data/switzerland
    /// if it exists, else data/jungfrau]
    #[arg(long, global = true)]
    data: Option<PathBuf>,
    /// Render a synthetic cone instead of a dataset; its north-west flank
    /// must come out brightest
    #[arg(long, global = true)]
    cone: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Render one frame to a PNG without opening a window
    Snapshot(SnapshotArgs),
    /// Render every condition from the fixed viewpoints and flights, and
    /// write stills and metrics
    Evaluate(EvaluateArgs),
}

#[derive(Args)]
struct SnapshotArgs {
    /// Output PNG
    #[arg(long, default_value = "captures/snapshot.png")]
    out: PathBuf,
    /// Built-in viewpoint or named place; `list` prints the names
    #[arg(long, default_value = "oblique_north")]
    view: String,
    /// Explicit camera instead of a viewpoint: `x,y,z,heading,pitch` in local
    /// metres and degrees (heading clockwise from north, pitch below the horizon)
    #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
    pose: Option<Vec<f32>>,
    /// A view of any point instead of a viewpoint: `east,north,heading,distance,pitch`
    /// with the point in LV95, the compass heading to look in, the distance in
    /// metres and the pitch in degrees
    #[arg(long, value_delimiter = ',', allow_hyphen_values = true, conflicts_with = "pose")]
    orbit: Option<Vec<f64>>,
    /// Evaluation condition A to E
    #[arg(long, default_value_t = 'D')]
    condition: char,
    /// Preset file to use instead of a condition
    #[arg(long)]
    preset: Option<PathBuf>,
    /// Switch one component of the method off, e.g. `no_fog`
    #[arg(long)]
    ablation: Option<String>,
    /// Condition A to E shown on the right half for comparison
    #[arg(long)]
    compare: Option<char>,
    /// Debug view number (1 to 13)
    #[arg(long)]
    debug: Option<u32>,
    #[arg(long, default_value_t = 1920)]
    width: u32,
    #[arg(long, default_value_t = 1080)]
    height: u32,
}

#[derive(Args)]
struct EvaluateArgs {
    /// Output directory
    #[arg(long, default_value = "evaluation")]
    out: PathBuf,
    #[arg(long, default_value_t = 1920)]
    width: u32,
    #[arg(long, default_value_t = 1080)]
    height: u32,
    /// Simulated frames per second on the flights
    #[arg(long, default_value_t = 60.0)]
    fps: f32,
    /// Shorten the flights and skip the ablations, for a fast check
    #[arg(long)]
    quick: bool,
    /// Skip the flights, for tuning parameters on the fixed viewpoints
    #[arg(long)]
    stills_only: bool,
    /// Preset file to evaluate next to the conditions; may be repeated
    #[arg(long)]
    preset: Vec<PathBuf>,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,relief_viewer=info")).init();
    let cli = Cli::parse();
    let terrain = load_terrain(cli.data.as_deref(), cli.cone)?;
    match cli.command {
        Some(Command::Snapshot(args)) => snapshot(&terrain, &args),
        Some(Command::Evaluate(args)) => evaluate(&terrain, &args),
        None => app::run(terrain),
    }
}

fn load_terrain(dir: Option<&Path>, cone: bool) -> Result<Terrain> {
    if cone {
        return Ok(synthetic::cone_dataset(CONE_SIZE)?.into());
    }
    // Prefer the whole country when it has been built.
    let dir = dir.unwrap_or_else(|| {
        let country = Path::new(COUNTRY_DATA);
        if country.join("meta.json").exists() {
            country
        } else {
            Path::new(REGION_DATA)
        }
    });
    Terrain::open(dir).with_context(|| format!("loading the dataset in {}", dir.display()))
}

fn condition_params(letter: char) -> Result<ShadingParams> {
    let condition =
        Condition::from_letter(letter).with_context(|| format!("unknown condition '{letter}', expected A to E"))?;
    Ok(condition.params())
}

fn evaluate(terrain: &Terrain, args: &EvaluateArgs) -> Result<()> {
    let presets = args
        .preset
        .iter()
        .map(|file| {
            let name = file.file_stem().context("a preset needs a file name")?.to_string_lossy().into_owned();
            Ok((name, ShadingParams::load(file)?))
        })
        .collect::<Result<_>>()?;
    let options = evaluate::Options {
        out: &args.out,
        size: [args.width, args.height],
        fps: args.fps,
        quick: args.quick,
        stills_only: args.stills_only,
        presets,
    };
    evaluate::run(terrain, &options)
}

fn snapshot(terrain: &Terrain, args: &SnapshotArgs) -> Result<()> {
    let mut params = match (&args.preset, &args.ablation) {
        (Some(path), _) => ShadingParams::load(path)?,
        (None, Some(name)) => Ablation::ALL
            .into_iter()
            .find(|ablation| ablation.name() == name)
            .with_context(|| format!("unknown ablation '{name}'"))?
            .params(),
        (None, None) => condition_params(args.condition)?,
    };
    if let Some(number) = args.debug {
        params.debug_view = *DebugView::ALL.get(number as usize).context("debug view out of range")?;
    }
    let compare = args.compare.map(condition_params).transpose()?;

    let mut headless = Headless::new(terrain, [args.width, args.height])?;
    // The evaluation viewpoints, then the named places.
    let z_scale = params.geometry_exaggeration;
    let mut viewpoints = Scene::new(terrain, &headless.ground, z_scale, DEFAULT_FOV_Y)?.viewpoints();
    let places = path::places(terrain);
    viewpoints
        .extend(places.iter().map(|place| Viewpoint { name: place.id, pose: place.pose(&headless.ground, z_scale) }));
    if args.view == "list" {
        viewpoints.iter().for_each(|view| println!("{}", view.name));
        return Ok(());
    }
    let pose = match (args.pose.as_deref(), args.orbit.as_deref()) {
        (Some(&[x, y, z, heading, pitch]), _) => {
            Pose { position: Vec3::new(x, y, z), heading: heading.to_radians(), pitch: pitch.to_radians() }
        }
        (Some(_), _) => bail!("--pose takes five numbers: x,y,z,heading,pitch"),
        (None, Some(&[east, north, heading, distance, pitch])) => {
            let [origin_east, origin_north] = terrain.meta.origin_lv95;
            let view = OrbitView {
                pivot: Vec2::new((east - origin_east) as f32, (north - origin_north) as f32),
                distance: distance as f32,
                heading: (heading as f32).to_radians(),
                pitch: (pitch as f32).to_radians(),
            };
            view.pose(&headless.ground, z_scale)
        }
        (None, Some(_)) => bail!("--orbit takes five numbers: east,north,heading,distance,pitch"),
        (None, None) => {
            viewpoints
                .iter()
                .find(|view| view.name == args.view)
                .with_context(|| format!("unknown viewpoint '{}', try --view list", args.view))?
                .pose
        }
    };

    let (mut rig, mut compare_rig) = (LightRig::default(), LightRig::default());
    // The second frame lets the light field see its own previous state.
    for _ in 0..2 {
        let compared = compare.as_ref().map(|compared| ((compared, &mut compare_rig), 0.5));
        headless.render(pose, (&params, &mut rig), compared, 1.0 / 60.0)?;
    }
    headless.finish();
    capture::save_png(&args.out, headless.size, &headless.read_output()?)?;
    println!("wrote {}", args.out.display());
    Ok(())
}
