//! The evaluation of plan section 10, run without a window.
//!
//! Every condition and ablation is rendered from the same fixed viewpoints
//! and the same camera paths at a fixed time step, so the results are
//! reproducible. Three files of numbers come out, next to the stills:
//!
//! * `stills.csv`: lost relief and vanished ridges per viewpoint,
//! * `flights.csv`: the same over each flight, plus flicker and relighting,
//! * `performance.csv`: frame rate and GPU time per pass.
//!
//! Top-down fidelity against a reference relief (H1) is measured separately
//! with `relief-prep compare` on the top-down stills written here.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use glam::{Mat4, Vec2, Vec3};
use relief_terrain::{Grid, Terrain};

use crate::camera::{Ground, Pose};
use crate::capture::{self, Readback};
use crate::headless::{Headless, DEFAULT_FOV_Y};
use crate::light_rig::LightRig;
use crate::params::{Ablation, Condition, ShadingParams};
use crate::path::{CameraPath, Scene};
use crate::renderer::timers::Pass;

/// Pyramid level at which major ridgelines are detected: 320 m cells.
const RIDGE_LEVEL: usize = 5;
/// Slope change across a crest, per cell, that makes it a major ridge.
const RIDGE_CONVEXITY: f32 = 0.25;
/// How far either side of a crest the flanks are compared, in cells.
const FLANK_OFFSET_CELLS: f32 = 1.5;
/// Grey-value difference below which two tones count as the same.
const SAME_TONE: f32 = 0.04;
/// Terrain points per side of the grid tracked for flicker.
const TRACKED_GRID: usize = 64;
/// Length of the low-pass that separates relighting from flicker.
const LOW_PASS_SECONDS: f32 = 0.5;
/// A point is visible if the depth buffer agrees with its distance this well.
const DEPTH_TOLERANCE: f32 = 0.03;
/// Frames rendered before a still is read, so the light field settles.
const SETTLE_FRAMES: usize = 3;
/// Every n-th flight frame is measured for lost relief and vanished ridges.
const SHADING_METRIC_INTERVAL: usize = 10;
/// Frames at the start of a timing run that are not measured.
const WARM_UP_FRAMES: usize = 10;
/// The passes that run without a window.
const TIMED_PASSES: [Pass; 4] = [Pass::LocalLight, Pass::Terrain, Pass::DepthEdges, Pass::Composite];

pub struct Options<'a> {
    pub out: &'a Path,
    pub size: [u32; 2],
    /// Simulated frames per second on the flights.
    pub fps: f32,
    /// Shorten the flights and skip the ablations, for a fast check.
    pub quick: bool,
    /// Skip the flights, for tuning parameters on the fixed viewpoints.
    pub stills_only: bool,
    /// Named parameter sets to evaluate next to the conditions.
    pub presets: Vec<(String, ShadingParams)>,
}

/// One rendered frame read back from the GPU.
struct Frame {
    size: [u32; 2],
    camera: Vec3,
    view_projection: Mat4,
    /// Display-encoded output, RGBA8.
    color: Vec<u8>,
    /// Natural log of the distance to the camera.
    log_depth: Vec<f32>,
    /// Grey value, steep-slope flag, flat-ground grey and terrain mask; empty
    /// if the frame was read without its shading.
    shading: Vec<u8>,
}

/// Staging buffers for reading a frame's targets back in one submission.
struct FrameReader {
    color: Readback,
    log_depth: Readback,
    shading: Readback,
}

impl FrameReader {
    fn new(headless: &Headless) -> Result<Self> {
        let targets = headless.renderer.targets();
        Ok(Self {
            color: Readback::new(&headless.gpu, headless.output_texture())?,
            log_depth: Readback::new(&headless.gpu, &targets.log_depth_texture)?,
            shading: Readback::new(&headless.gpu, &targets.metrics_texture)?,
        })
    }

    /// Reads the frame rendered last from `pose`. The shading target is only
    /// fetched `with_shading`; the stability metric does not need it.
    fn read(&self, headless: &Headless, pose: &Pose, with_shading: bool) -> Result<Frame> {
        let targets = headless.renderer.targets();
        let mut jobs = vec![(&self.color, headless.output_texture()), (&self.log_depth, &targets.log_depth_texture)];
        if with_shading {
            jobs.push((&self.shading, &targets.metrics_texture));
        }
        let mut buffers = Readback::read_all(&headless.gpu, &jobs)?.into_iter();
        let mut next = || buffers.next().expect("one buffer per readback job");

        let [width, height] = headless.size;
        Ok(Frame {
            size: headless.size,
            camera: pose.position,
            view_projection: pose.view_projection(DEFAULT_FOV_Y, width as f32 / height as f32),
            color: next(),
            log_depth: capture::floats(&next()),
            shading: if with_shading { next() } else { vec![] },
        })
    }
}

impl Frame {
    fn index(&self, pixel: Vec2) -> usize {
        let x = (pixel.x.max(0.0) as usize).min(self.size[0] as usize - 1);
        let y = (pixel.y.max(0.0) as usize).min(self.size[1] as usize - 1);
        y * self.size[0] as usize + x
    }

    /// Where a terrain point appears, if it is on screen and not hidden.
    fn project(&self, point: Vec3) -> Option<Vec2> {
        let clip = self.view_projection * point.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        if ndc.x.abs() >= 1.0 || ndc.y.abs() >= 1.0 {
            return None;
        }
        let pixel =
            Vec2::new(0.5 + 0.5 * ndc.x, 0.5 - 0.5 * ndc.y) * Vec2::new(self.size[0] as f32, self.size[1] as f32);
        let distance = point.distance(self.camera);
        let drawn = self.log_depth[self.index(pixel)].exp();
        ((drawn - distance).abs() < DEPTH_TOLERANCE * distance).then_some(pixel)
    }

    /// Display luminance in `0..=1`, interpolated between pixel centres.
    fn luminance(&self, pixel: Vec2) -> f32 {
        let luma = |x: f32, y: f32| {
            let i = 4 * self.index(Vec2::new(x, y));
            (0.2126 * self.color[i] as f32 + 0.7152 * self.color[i + 1] as f32 + 0.0722 * self.color[i + 2] as f32)
                / 255.0
        };
        let corner = (pixel - 0.5).floor();
        let Vec2 { x: fx, y: fy } = pixel - 0.5 - corner;
        let top = luma(corner.x, corner.y) * (1.0 - fx) + luma(corner.x + 1.0, corner.y) * fx;
        let bottom = luma(corner.x, corner.y + 1.0) * (1.0 - fx) + luma(corner.x + 1.0, corner.y + 1.0) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    /// The shading model's grey value in `0..=1`, before colour and haze.
    fn grey(&self, pixel: Vec2) -> f32 {
        self.shading[4 * self.index(pixel)] as f32 / 255.0
    }

    /// Fraction of pixels on slopes steeper than 15 degrees whose brightness
    /// is within a small band of flat-ground brightness. Lower is better.
    fn lost_relief(&self) -> Option<f32> {
        let (mut steep, mut lost) = (0u32, 0u32);
        for &[grey, steep_flag, flat_grey, terrain] in self.shading.as_chunks::<4>().0 {
            if terrain > 127 && steep_flag > 127 {
                steep += 1;
                lost += (grey.abs_diff(flat_grey) as f32 / 255.0 < SAME_TONE) as u32;
            }
        }
        (steep > 0).then(|| lost as f32 / steep as f32)
    }

    /// Fraction of visible major ridges whose two flanks have the same tone,
    /// with the number of ridges that could be measured.
    fn vanished_ridges(&self, flanks: &[[Vec3; 2]]) -> Option<(f32, usize)> {
        let differences: Vec<f32> = flanks
            .iter()
            .filter_map(|pair| {
                let [a, b] = pair.map(|flank| self.project(flank));
                Some((self.grey(a?) - self.grey(b?)).abs())
            })
            .collect();
        let vanished = differences.iter().filter(|&&difference| difference < SAME_TONE).count();
        (!differences.is_empty()).then(|| (vanished as f32 / differences.len() as f32, differences.len()))
    }
}

/// A major ridge crest, as the two ground positions either side of it.
struct Ridge {
    flanks: [Vec2; 2],
}

/// Finds major ridgelines from the curvature of a generalized height level.
///
/// A cell is on a ridge if it is higher than both neighbours along some
/// axis and the slope changes strongly across it; that axis is the
/// direction across the ridge.
///
/// `heights` is a generalized level with cells of `cell` metres whose
/// south-west corner lies at `origin`.
fn find_ridges(heights: &Grid<f32>, origin: Vec2, cell: f32) -> Vec<Ridge> {
    let axes = [(1isize, 0isize), (0, 1), (1, 1), (1, -1)];
    let mut ridges = vec![];
    for y in 2..heights.height() as isize - 2 {
        for x in 2..heights.width() as isize - 2 {
            let here = heights.get(x as usize, y as usize);
            let across = axes
                .iter()
                .filter_map(|&(dx, dy)| {
                    let direction = Vec2::new(dx as f32, dy as f32);
                    let (ahead, behind) = (heights.get_clamped(x + dx, y + dy), heights.get_clamped(x - dx, y - dy));
                    let convexity = (2.0 * here - ahead - behind) / (direction.length() * cell);
                    let is_crest = here > ahead && here > behind && convexity > RIDGE_CONVEXITY;
                    is_crest.then_some((convexity, direction.normalize()))
                })
                .max_by(|a, b| a.0.total_cmp(&b.0));
            if let Some((_, direction)) = across {
                let crest = origin + Vec2::new(x as f32 + 0.5, y as f32 + 0.5) * cell;
                let offset = direction * FLANK_OFFSET_CELLS * cell;
                ridges.push(Ridge { flanks: [crest - offset, crest + offset] });
            }
        }
    }
    ridges
}

/// The flanks of every ridge as points on the terrain drawn at `z_scale`.
fn flank_points(ridges: &[Ridge], ground: &Ground, z_scale: f32) -> Vec<[Vec3; 2]> {
    ridges.iter().map(|ridge| ridge.flanks.map(|xy| xy.extend(ground.height(xy) * z_scale))).collect()
}

/// Luminance of fixed terrain points over a flight, for the flicker metric.
///
/// Following terrain points is the reprojection of the plan made explicit:
/// each point's luminance over time is split by a half-second low-pass into
/// a smooth part, the intended relighting as the camera moves, and a
/// high-frequency residual, the flicker.
struct Tracker {
    points: Vec<Vec3>,
    /// Per point, its luminance in every frame, or `None` while hidden.
    series: Vec<Vec<Option<f32>>>,
}

struct Stability {
    /// Mean absolute high-frequency residual, in 8-bit grey levels.
    flicker: f32,
    /// Mean absolute change of the smooth part, in grey levels per second.
    relighting: f32,
}

impl Tracker {
    /// Spreads the tracked points over a region given as west, south, east
    /// and north in metres.
    fn new(ground: &Ground, region: [f32; 4], z_scale: f32) -> Self {
        let (origin, size) = (Vec2::new(region[0], region[1]), Vec2::new(region[2] - region[0], region[3] - region[1]));
        // Stay clear of the border, where the terrain may fade into haze.
        let margin = 0.08;
        let points: Vec<Vec3> = (0..TRACKED_GRID * TRACKED_GRID)
            .map(|i| {
                let cell = Vec2::new((i % TRACKED_GRID) as f32, (i / TRACKED_GRID) as f32) / (TRACKED_GRID - 1) as f32;
                let xy = origin + (margin + (1.0 - 2.0 * margin) * cell) * size;
                xy.extend(ground.height(xy) * z_scale)
            })
            .collect();
        Self { series: vec![vec![]; points.len()], points }
    }

    fn observe(&mut self, frame: &Frame) {
        for (point, series) in self.points.iter().zip(&mut self.series) {
            series.push(frame.project(*point).map(|pixel| frame.luminance(pixel)));
        }
    }

    fn stability(&self, fps: f32) -> Option<Stability> {
        // Two box filters in a row. An even length cancels frame-to-frame
        // jitter exactly, and two passes are centred on a whole frame again.
        let window = (((0.5 * LOW_PASS_SECONDS * fps) as usize) & !1).max(2);
        let box_filter = |values: &[f32]| -> Vec<f32> {
            values.windows(window).map(|window| window.iter().sum::<f32>() / window.len() as f32).collect()
        };

        let (mut residual, mut residual_count) = (0.0f64, 0usize);
        let (mut drift, mut drift_count) = (0.0f64, 0usize);
        for series in &self.series {
            // Only unbroken runs of visibility can be filtered.
            for run in series.split(Option::is_none).filter(|run| run.len() >= 4 * window) {
                let values: Vec<f32> = run.iter().flatten().copied().collect();
                let smooth = box_filter(&box_filter(&values));
                // Each smooth sample is centred `window - 1` frames into the run.
                for (i, mean) in smooth.iter().enumerate() {
                    residual += (values[i + window - 1] - mean).abs() as f64;
                }
                residual_count += smooth.len();
                drift += smooth.windows(2).map(|pair| (pair[1] - pair[0]).abs() as f64).sum::<f64>();
                drift_count += smooth.len() - 1;
            }
        }
        (residual_count > 0 && drift_count > 0).then(|| Stability {
            flicker: 255.0 * (residual / residual_count as f64) as f32,
            relighting: 255.0 * fps * (drift / drift_count as f64) as f32,
        })
    }
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

/// Formats an optional measurement for a CSV cell.
fn cell(value: Option<f32>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.4}"))
}

/// Cuts a path to its first `seconds`.
fn truncate(path: &mut CameraPath, seconds: f32) {
    path.keyframes.retain(|key| key.time <= seconds);
}

pub fn run(terrain: &Terrain, options: &Options) -> Result<()> {
    let mut headless = Headless::new(terrain, options.size)?;
    // Depths are read back as 32-bit floats.
    ensure!(headless.gpu.float_filtering, "the evaluation needs a graphics adapter that filters 32-bit float textures");
    let z_scale = ShadingParams::default().geometry_exaggeration;
    // Everything is measured in the region the viewpoints are laid out in.
    let region = terrain.meta.view_region();

    let (viewpoints, mut flights) = {
        let scene = Scene::new(terrain, &headless.ground, z_scale, DEFAULT_FOV_Y)?;
        let mut flights = scene.flights();
        flights.push(scene.rotation_in_place());
        (scene.viewpoints(), flights)
    };
    if options.quick {
        flights.iter_mut().for_each(|flight| truncate(flight, 4.0));
    }
    let ridge_level = RIDGE_LEVEL.min(terrain.meta.levels - 1);
    let ridges = find_ridges(
        &terrain.heights_in(ridge_level, region)?,
        Vec2::new(region[0], region[1]),
        terrain.cell(ridge_level),
    );
    println!("evaluating on {} at {} x {}", headless.gpu.adapter_name(), options.size[0], options.size[1]);
    println!("{} viewpoints, {} flights, {} major ridge samples", viewpoints.len(), flights.len(), ridges.len());

    let mut conditions: Vec<(String, ShadingParams)> =
        Condition::ALL.into_iter().map(|condition| (condition.letter().to_string(), condition.params())).collect();
    conditions.extend(options.presets.iter().cloned());
    let mut variants = conditions.clone();
    if !options.quick {
        variants
            .extend(Ablation::ALL.into_iter().map(|ablation| (format!("D_{}", ablation.name()), ablation.params())));
    }
    fs::create_dir_all(options.out).with_context(|| format!("creating {}", options.out.display()))?;
    let reader = FrameReader::new(&headless)?;
    let dt = 1.0 / options.fps;

    // Stills: every variant from every fixed viewpoint.
    let mut stills = "variant,viewpoint,lost_relief,vanished_ridges,ridges_measured\n".to_owned();
    for (name, params) in &variants {
        let flanks = flank_points(&ridges, &headless.ground, params.geometry_exaggeration);
        for view in &viewpoints {
            let mut rig = LightRig::default();
            headless.renderer.forget_history();
            for _ in 0..SETTLE_FRAMES {
                headless.render(view.pose, (params, &mut rig), None, dt)?;
            }
            headless.finish();
            let frame = reader.read(&headless, &view.pose, true)?;
            let file = options.out.join("stills").join(name).join(format!("{}.png", view.name));
            capture::save_png(&file, options.size, &frame.color)?;

            let vanished = frame.vanished_ridges(&flanks);
            writeln!(
                stills,
                "{name},{},{},{},{}",
                view.name,
                cell(frame.lost_relief()),
                cell(vanished.map(|(fraction, _)| fraction)),
                vanished.map_or(0, |(_, count)| count)
            )?;
        }
        println!("stills {name}");
    }
    fs::write(options.out.join("stills.csv"), stills)?;
    if options.stills_only {
        println!("wrote {}", options.out.display());
        return Ok(());
    }

    // Flights: the conditions over each camera path.
    let mut flight_rows =
        "condition,flight,frames,lost_relief,vanished_ridges,flicker_grey_levels,relighting_grey_levels_per_s\n"
            .to_owned();
    let pass_columns: Vec<String> = TIMED_PASSES.iter().map(|pass| format!("{}_ms", pass.key())).collect();
    let mut performance_rows = format!("condition,flight,mean_fps,min_fps,{},gpu_total_ms\n", pass_columns.join(","));
    for (name, params) in &conditions {
        let flanks = flank_points(&ridges, &headless.ground, params.geometry_exaggeration);
        for flight in &flights {
            let frames = (flight.duration() * options.fps) as usize;

            // Measured pass: every frame is read back.
            let mut tracker = Tracker::new(&headless.ground, region, params.geometry_exaggeration);
            let mut rig = LightRig::default();
            headless.renderer.forget_history();
            let (mut lost, mut vanished) = (vec![], vec![]);
            for index in 0..frames {
                let pose = flight.sample(index as f32 * dt);
                headless.render(pose, (params, &mut rig), None, dt)?;
                let measure_shading = index % SHADING_METRIC_INTERVAL == 0;
                let frame = reader.read(&headless, &pose, measure_shading)?;
                tracker.observe(&frame);
                if measure_shading {
                    lost.extend(frame.lost_relief());
                    vanished.extend(frame.vanished_ridges(&flanks).map(|(fraction, _)| fraction));
                }
            }
            let stability = tracker.stability(options.fps);
            writeln!(
                flight_rows,
                "{name},{},{frames},{:.4},{:.4},{},{}",
                flight.name,
                mean(&lost),
                mean(&vanished),
                cell(stability.as_ref().map(|stability| stability.flicker)),
                cell(stability.as_ref().map(|stability| stability.relighting)),
            )?;

            // Timing pass: nothing is read back, each frame runs to completion.
            let mut rig = LightRig::default();
            headless.renderer.forget_history();
            let (mut frame_seconds, mut pass_ms) = (vec![], vec![vec![]; TIMED_PASSES.len()]);
            for index in 0..frames {
                let pose = flight.sample(index as f32 * dt);
                let started = Instant::now();
                headless.render(pose, (params, &mut rig), None, dt)?;
                headless.finish();
                if index >= WARM_UP_FRAMES {
                    frame_seconds.push(started.elapsed().as_secs_f32());
                    for (samples, pass) in pass_ms.iter_mut().zip(TIMED_PASSES) {
                        samples.push(headless.renderer.timers.latest_ms(pass));
                    }
                }
            }
            let pass_means: Vec<f32> = pass_ms.iter().map(|samples| mean(samples)).collect();
            writeln!(
                performance_rows,
                "{name},{},{:.1},{:.1},{},{:.3}",
                flight.name,
                1.0 / mean(&frame_seconds).max(1e-6),
                1.0 / frame_seconds.iter().copied().fold(1e-6, f32::max),
                pass_means.iter().map(|ms| format!("{ms:.3}")).collect::<Vec<_>>().join(","),
                pass_means.iter().sum::<f32>()
            )?;
            println!("flight {name} {}", flight.name);
        }
    }
    fs::write(options.out.join("flights.csv"), flight_rows)?;
    fs::write(options.out.join("performance.csv"), performance_rows)?;
    println!("wrote {}", options.out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_detection_finds_a_crest_and_its_flanks() {
        // A ridge running north-south through the middle of a 64-cell grid.
        let heights = Grid::from_fn(64, 64, |x, _| 2000.0 - 200.0 * (x as f32 - 32.0).abs());
        let ridges = find_ridges(&heights, Vec2::ZERO, 320.0);
        assert_eq!(ridges.len(), 60);
        for ridge in &ridges {
            let [west, east] = ridge.flanks;
            assert!(west.x < 32.5 * 320.0 && east.x > 32.5 * 320.0);
            assert_eq!(west.y, east.y);
        }
        assert!(find_ridges(&Grid::new(64, 64, 500.0), Vec2::ZERO, 320.0).is_empty());
    }

    #[test]
    fn tracker_separates_flicker_from_relighting() {
        let fps = 60.0;
        let frames = 240;
        let tracker = |luminance: &dyn Fn(usize) -> Option<f32>| Tracker {
            points: vec![Vec3::ZERO],
            series: vec![(0..frames).map(luminance).collect()],
        };

        // A slow ramp is relighting without flicker.
        let ramp = tracker(&|i| Some(0.2 + 0.002 * i as f32)).stability(fps).unwrap();
        assert!(ramp.flicker < 0.01, "flicker {}", ramp.flicker);
        assert!((ramp.relighting - 255.0 * 0.002 * fps).abs() < 0.5, "relighting {}", ramp.relighting);

        // Alternating frames are flicker without relighting.
        let jitter = tracker(&|i| Some(0.5 + 0.02 * (i % 2) as f32)).stability(fps).unwrap();
        assert!(jitter.flicker > 2.0, "flicker {}", jitter.flicker);
        assert!(jitter.relighting < 1.0, "relighting {}", jitter.relighting);

        // Points that are never seen for long give no measurement.
        assert!(tracker(&|i| (i % 8 != 0).then_some(0.5)).stability(fps).is_none());
    }
}
