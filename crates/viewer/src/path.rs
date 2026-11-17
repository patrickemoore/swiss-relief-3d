//! Camera paths: recording, playback and the built-in evaluation set.
//!
//! Every evaluation figure comes from a fixed viewpoint or a recorded path,
//! so results are reproducible (plan sections 9 and 10).

use std::f32::consts::{FRAC_PI_2, TAU};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use glam::{Vec2, Vec3};
use relief_terrain::{Grid, Terrain};
use serde::{Deserialize, Serialize};

use crate::camera::{wrap_angle, Ground, OrbitView, Pose};

/// Directory recorded paths are saved to.
pub const PATH_DIR: &str = "paths";
/// Interval between recorded keyframes, in seconds.
const RECORD_INTERVAL_S: f32 = 1.0 / 30.0;
/// Pyramid level on which the summit is found and the valley run traced.
const SCENE_LEVEL: usize = 4;

/// Places worth a visit in Switzerland: a short name, the name to show, LV95
/// easting and northing of what to look at, compass heading to look in,
/// distance in metres and pitch in degrees.
///
/// The terrain ends at the border, so the views are chosen to look into the
/// country: a view of the Matterhorn from the north would otherwise be half
/// empty sky where Italy is.
const PLACES: [(&str, &str, f64, f64, f32, f32, f32); 13] = [
    ("switzerland", "Switzerland", 2_660_000.0, 1_185_000.0, 0.0, 300_000.0, 90.0),
    ("lake_lucerne", "Lake Lucerne", 2_676_000.0, 1_207_000.0, 170.0, 30_000.0, 32.0),
    ("jungfrau", "Jungfrau", 2_641_000.0, 1_155_300.0, 150.0, 22_000.0, 30.0),
    ("lauterbrunnen", "Lauterbrunnen", 2_636_200.0, 1_157_500.0, 185.0, 11_000.0, 26.0),
    ("aletsch", "Aletsch Glacier", 2_647_000.0, 1_147_500.0, 330.0, 16_000.0, 35.0),
    ("matterhorn", "Matterhorn", 2_618_500.0, 1_093_500.0, 215.0, 15_000.0, 40.0),
    ("gotthard", "Gotthard", 2_686_500.0, 1_156_500.0, 0.0, 25_000.0, 35.0),
    ("locarno", "Locarno", 2_705_000.0, 1_117_000.0, 350.0, 24_000.0, 36.0),
    ("bernina", "Piz Bernina", 2_790_000.0, 1_140_500.0, 195.0, 17_000.0, 33.0),
    ("chur", "Chur", 2_760_000.0, 1_191_000.0, 200.0, 30_000.0, 30.0),
    ("saentis", "S\u{e4}ntis", 2_744_180.0, 1_234_920.0, 200.0, 16_000.0, 25.0),
    ("lake_geneva", "Lake Geneva", 2_557_000.0, 1_143_000.0, 80.0, 32_000.0, 34.0),
    ("creux_du_van", "Creux du Van", 2_545_800.0, 1_198_300.0, 200.0, 5_500.0, 32.0),
];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Keyframe {
    pub time: f32,
    pub pose: Pose,
}

/// A timed sequence of camera poses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraPath {
    pub name: String,
    pub keyframes: Vec<Keyframe>,
}

impl CameraPath {
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map_or(0.0, |key| key.time)
    }

    /// The pose at `time`, clamped to the ends of the path. Angles
    /// interpolate along the shorter way round.
    pub fn sample(&self, time: f32) -> Pose {
        let next = self.keyframes.partition_point(|key| key.time <= time);
        let (Some(a), Some(b)) = (self.keyframes.get(next.wrapping_sub(1)), self.keyframes.get(next)) else {
            let end = if next == 0 { self.keyframes.first() } else { self.keyframes.last() };
            return end.expect("a camera path has at least one keyframe").pose;
        };
        let t = (time - a.time) / (b.time - a.time).max(1e-6);
        Pose {
            position: a.pose.position.lerp(b.pose.position, t),
            heading: wrap_angle(a.pose.heading + wrap_angle(b.pose.heading - a.pose.heading) * t),
            pitch: a.pose.pitch + (b.pose.pitch - a.pose.pitch) * t,
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, serde_json::to_string(self)?).with_context(|| format!("writing {}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}

/// Saved paths in [`PATH_DIR`], sorted by name.
pub fn list_saved_paths() -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = fs::read_dir(PATH_DIR)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "json"))
        .collect();
    paths.sort();
    paths
}

/// Samples the live camera into a path at a fixed rate.
pub struct Recorder {
    path: CameraPath,
    elapsed: f32,
}

impl Recorder {
    pub fn new(name: &str, start: Pose) -> Self {
        let path = CameraPath { name: name.to_owned(), keyframes: vec![Keyframe { time: 0.0, pose: start }] };
        Self { path, elapsed: 0.0 }
    }

    pub fn elapsed(&self) -> f32 {
        self.elapsed
    }

    pub fn advance(&mut self, pose: Pose, dt: f32) {
        self.elapsed += dt;
        if self.elapsed - self.path.duration() >= RECORD_INTERVAL_S {
            self.path.keyframes.push(Keyframe { time: self.elapsed, pose });
        }
    }

    pub fn finish(self) -> CameraPath {
        self.path
    }
}

/// A named still camera position.
#[derive(Clone, Debug)]
pub struct Viewpoint {
    pub name: &'static str,
    pub pose: Pose,
}

/// The id of the place that is the whole country seen from above.
pub const WHOLE_COUNTRY: &str = "switzerland";

/// Somewhere worth a visit: what to look at, and a good view of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    /// A short name for command lines and code.
    pub id: &'static str,
    /// The name to show.
    pub name: &'static str,
    pub view: OrbitView,
}

impl Place {
    /// The camera pose of the place's view.
    pub fn pose(&self, ground: &Ground, z_scale: f32) -> Pose {
        self.view.pose(ground, z_scale)
    }
}

/// The named places that lie on a terrain.
pub fn places(terrain: &Terrain) -> Vec<Place> {
    let [origin_east, origin_north] = terrain.meta.origin_lv95;
    let [west, south, east, north] = terrain.meta.data_region();
    PLACES
        .iter()
        .filter_map(|&(id, name, lv95_east, lv95_north, heading, distance, pitch)| {
            let pivot = Vec2::new((lv95_east - origin_east) as f32, (lv95_north - origin_north) as f32);
            let on_terrain = (west..east).contains(&pivot.x) && (south..north).contains(&pivot.y);
            let view = OrbitView { pivot, distance, heading: heading.to_radians(), pitch: pitch.to_radians() };
            on_terrain.then_some(Place { id, name, view })
        })
        .collect()
}

/// What the built-in viewpoints and flights need to know about a terrain.
///
/// They are laid out in the terrain's view region: all of a small terrain,
/// one mountain region of a country.
pub struct Scene<'a> {
    ground: &'a Ground,
    /// South-west corner of the view region in metres.
    origin: Vec2,
    /// Side length of the view region in metres.
    extent: f32,
    /// A generalized height level of the view region, on which the summit is
    /// found and the valley run traced.
    coarse_heights: Grid<f32>,
    /// Vertical scale the terrain is drawn with.
    z_scale: f32,
    fov_y: f32,
}

impl<'a> Scene<'a> {
    pub fn new(terrain: &Terrain, ground: &'a Ground, z_scale: f32, fov_y: f32) -> Result<Self> {
        let region = terrain.meta.view_region();
        let [west, south, east, north] = region;
        Ok(Self {
            ground,
            origin: Vec2::new(west, south),
            extent: (east - west).min(north - south),
            coarse_heights: terrain.heights_in(SCENE_LEVEL.min(terrain.meta.levels - 1), region)?,
            z_scale,
            fov_y,
        })
    }

    /// A position in the view region, given as fractions of its side.
    fn at(&self, x: f32, y: f32) -> Vec2 {
        self.origin + Vec2::new(x, y) * self.extent
    }

    fn surface(&self, xy: Vec2) -> Vec3 {
        xy.extend(self.ground.height(xy) * self.z_scale)
    }

    /// Position of the highest summit.
    fn summit(&self) -> Vec3 {
        let heights = &self.coarse_heights;
        let (index, _) = heights
            .data()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("the terrain has at least one cell");
        let cell = self.extent / heights.width() as f32;
        let texel = Vec2::new((index % heights.width()) as f32, (index / heights.width()) as f32);
        self.surface(self.origin + (texel + 0.5) * cell)
    }

    /// The fixed evaluation viewpoints: two top-down, four oblique at 45
    /// degrees, four low at 15 degrees and two toward the horizon.
    pub fn viewpoints(&self) -> Vec<Viewpoint> {
        let extent = self.extent;
        let centre = self.surface(self.at(0.5, 0.5));
        let summit = self.summit();
        // Distance from which the whole region fills the image height.
        let overview = 0.5 * extent / (0.5 * self.fov_y).tan();

        let mut views = vec![
            Viewpoint { name: "top_down_overview", pose: Pose::orbiting(centre, overview, 0.0, FRAC_PI_2) },
            Viewpoint { name: "top_down_summit", pose: Pose::orbiting(summit, 0.3 * extent, 0.0, FRAC_PI_2) },
        ];
        // Each set faces north, east, south and west in turn.
        for (pitch_deg, target, distance, names) in [
            (45.0f32, centre, 0.55 * extent, ["oblique_north", "oblique_east", "oblique_south", "oblique_west"]),
            (15.0, summit, 0.4 * extent, ["low_north", "low_east", "low_south", "low_west"]),
        ] {
            for (quarter_turns, name) in names.into_iter().enumerate() {
                let heading = quarter_turns as f32 * FRAC_PI_2;
                views.push(Viewpoint { name, pose: Pose::orbiting(target, distance, heading, pitch_deg.to_radians()) });
            }
        }

        // Toward the horizon: from above the northern and southern edges,
        // looking across the whole region.
        let above = |xy: Vec2| xy.extend(self.ground.height(xy) * self.z_scale + 1500.0);
        views.push(Viewpoint {
            name: "horizon_south",
            pose: Pose { position: above(self.at(0.5, 0.9)), heading: 2.0 * FRAC_PI_2, pitch: 0.06 },
        });
        views.push(Viewpoint {
            name: "horizon_north",
            pose: Pose { position: above(self.at(0.45, 0.12)), heading: 0.0, pitch: 0.06 },
        });
        views
    }

    /// The three evaluation flights: an orbit around the highest summit, a
    /// high flyover and a low valley run. A region too small to hold a
    /// valley has no valley run.
    pub fn flights(&self) -> Vec<CameraPath> {
        let mut flights = vec![self.summit_orbit(), self.high_flyover(), self.valley_run()];
        flights.retain(|flight| !flight.keyframes.is_empty());
        flights
    }

    /// A full turn on the spot. The terrain does not move relative to the
    /// camera, so any change of shading at a terrain point is light swim.
    pub fn rotation_in_place(&self) -> CameraPath {
        const DURATION_S: f32 = 20.0;
        let xy = self.at(0.5, 0.2);
        let position = xy.extend(self.ground.height(xy) * self.z_scale + 0.12 * self.extent);
        let keyframes = (0..=200)
            .map(|i| {
                let t = i as f32 / 200.0;
                Keyframe { time: t * DURATION_S, pose: Pose { position, heading: t * TAU, pitch: 25f32.to_radians() } }
            })
            .collect();
        CameraPath { name: "rotation_in_place".into(), keyframes }
    }

    fn summit_orbit(&self) -> CameraPath {
        const DURATION_S: f32 = 30.0;
        let summit = self.summit();
        let (distance, pitch) = (0.2 * self.extent, 20f32.to_radians());
        let keyframes = (0..=300)
            .map(|i| {
                let t = i as f32 / 300.0;
                Keyframe { time: t * DURATION_S, pose: Pose::orbiting(summit, distance, t * TAU, pitch) }
            })
            .collect();
        CameraPath { name: "summit_orbit".into(), keyframes }
    }

    fn high_flyover(&self) -> CameraPath {
        const DURATION_S: f32 = 30.0;
        let altitude = self.summit().z + 3000.0;
        let (start, end) = (self.at(0.1, 0.1), self.at(0.75, 0.75));
        let heading = (end - start).x.atan2((end - start).y);
        let keyframes = (0..=120)
            .map(|i| {
                let t = i as f32 / 120.0;
                let position = start.lerp(end, t).extend(altitude);
                Keyframe { time: t * DURATION_S, pose: Pose { position, heading, pitch: 35f32.to_radians() } }
            })
            .collect();
        CameraPath { name: "high_flyover".into(), keyframes }
    }

    /// Follows the valley that drains the highest summit, flying up it so
    /// the camera faces the mountains.
    fn valley_run(&self) -> CameraPath {
        const SPEED_M_S: f32 = 300.0;
        const CLEARANCE_M: f32 = 450.0;
        const SPACING_M: f32 = 150.0;

        let mut line = self.trace_descent();
        line.reverse();
        let line = resample(&smooth(&line, 12), SPACING_M);
        // Smooth the altitude separately so the camera does not bob over
        // every bump of the valley floor.
        let altitudes: Vec<f32> = line.iter().map(|&xy| self.ground.height(xy) * self.z_scale + CLEARANCE_M).collect();
        let altitudes = smooth_scalars(&altitudes, 10);

        let look_ahead = 8;
        let keyframes = (0..line.len().saturating_sub(look_ahead))
            .map(|i| {
                let direction = line[i + look_ahead] - line[i];
                Keyframe {
                    time: i as f32 * SPACING_M / SPEED_M_S,
                    pose: Pose {
                        position: line[i].extend(altitudes[i]),
                        heading: direction.x.atan2(direction.y),
                        pitch: 10f32.to_radians(),
                    },
                }
            })
            .collect();
        CameraPath { name: "valley_run".into(), keyframes }
    }

    /// Walks downhill from the summit on the generalized terrain. Small pits
    /// are crossed by searching a growing neighbourhood for lower ground.
    fn trace_descent(&self) -> Vec<Vec2> {
        const MAX_SEARCH_RADIUS: isize = 6;
        let heights = &self.coarse_heights;
        let (width, height) = (heights.width() as isize, heights.height() as isize);
        let cell = self.extent / width as f32;
        let summit = (self.summit().truncate() - self.origin) / cell;
        let (mut x, mut y) = (summit.x as isize, summit.y as isize);

        let mut line = vec![];
        'walk: for _ in 0..(width + height) {
            line.push(self.origin + Vec2::new(x as f32 + 0.5, y as f32 + 0.5) * cell);
            let here = heights.get(x as usize, y as usize);
            for radius in 1..=MAX_SEARCH_RADIUS {
                let lowest = (-radius..=radius)
                    .flat_map(|dy| (-radius..=radius).map(move |dx| (x + dx, y + dy)))
                    .filter(|&(nx, ny)| (0..width).contains(&nx) && (0..height).contains(&ny))
                    .map(|(nx, ny)| (nx, ny, heights.get(nx as usize, ny as usize)))
                    .min_by(|a, b| a.2.total_cmp(&b.2));
                if let Some((nx, ny, _)) = lowest.filter(|&(_, _, lower)| lower < here - 0.01) {
                    (x, y) = (nx, ny);
                    continue 'walk;
                }
            }
            break;
        }
        line
    }
}

/// Moving average over `radius` neighbours on each side.
fn smooth(points: &[Vec2], radius: usize) -> Vec<Vec2> {
    (0..points.len())
        .map(|i| {
            let window = &points[i.saturating_sub(radius)..(i + radius + 1).min(points.len())];
            window.iter().sum::<Vec2>() / window.len() as f32
        })
        .collect()
}

fn smooth_scalars(values: &[f32], radius: usize) -> Vec<f32> {
    (0..values.len())
        .map(|i| {
            let window = &values[i.saturating_sub(radius)..(i + radius + 1).min(values.len())];
            window.iter().sum::<f32>() / window.len() as f32
        })
        .collect()
}

/// Re-spaces a polyline to points `spacing` apart along its length.
fn resample(points: &[Vec2], spacing: f32) -> Vec<Vec2> {
    let mut out = vec![];
    let mut carried = 0.0;
    for pair in points.windows(2) {
        let length = pair[0].distance(pair[1]);
        let mut along = carried;
        while along < length {
            out.push(pair[0].lerp(pair[1], along / length));
            along += spacing;
        }
        carried = along - length;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(x: f32, heading: f32) -> Pose {
        Pose { position: Vec3::new(x, 0.0, 0.0), heading, pitch: 0.5 }
    }

    #[test]
    fn sample_interpolates_and_clamps() {
        let path = CameraPath {
            name: "test".into(),
            keyframes: vec![
                Keyframe { time: 0.0, pose: pose(0.0, 3.0) },
                Keyframe { time: 2.0, pose: pose(10.0, -3.0) },
            ],
        };
        assert_eq!(path.sample(-1.0), pose(0.0, 3.0));
        assert_eq!(path.sample(5.0), pose(10.0, -3.0));
        let middle = path.sample(1.0);
        assert!((middle.position.x - 5.0).abs() < 1e-5);
        // 3.0 -> -3.0 radians passes through pi, not through zero.
        assert!(middle.heading.abs() > 3.0, "heading {}", middle.heading);
    }

    #[test]
    fn resample_spaces_points_evenly() {
        let line = [Vec2::ZERO, Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0)];
        let points = resample(&line, 2.5);
        assert_eq!(points.len(), 8);
        assert!(points.windows(2).all(|pair| (pair[0].distance(pair[1]) - 2.5).abs() < 1e-4));
    }

    #[test]
    fn built_in_sets_match_the_plan() {
        // A cone: the summit is in the middle and every descent reaches the plain.
        let terrain = Terrain::from(relief_terrain::synthetic::cone_dataset(2048).unwrap());
        let ground = Ground::new(&terrain);
        let scene = Scene::new(&terrain, &ground, 1.0, 0.8).unwrap();
        let summit = scene.summit();
        assert!((summit.truncate() - Vec2::splat(10240.0)).length() < 200.0 && summit.z > 2300.0, "summit at {summit}");

        assert_eq!(scene.viewpoints().len(), 12);
        let flights = scene.flights();
        assert_eq!(flights.len(), 3);
        for flight in &flights {
            let (keyframes, duration) = (flight.keyframes.len(), flight.duration());
            assert!(keyframes > 10 && duration > 5.0, "{}: {keyframes} keyframes, {duration} s", flight.name);
        }

        // The cone is nowhere near Switzerland.
        assert!(places(&terrain).is_empty());
    }

    #[test]
    fn places_are_found_by_their_lv95_coordinates() {
        // The same cone, placed with its centre on the Jungfrau.
        let mut dataset = relief_terrain::synthetic::cone_dataset(1024).unwrap();
        dataset.meta.origin_lv95 = [2_641_000.0 - 5120.0, 1_155_300.0 - 5120.0];
        let terrain = Terrain::from(dataset);
        let ground = Ground::new(&terrain);

        let found = places(&terrain);
        // Lauterbrunnen lies 5 km from the summit, on the cone too.
        let names: Vec<_> = found.iter().map(|place| (place.id, place.name)).collect();
        assert_eq!(names, [("jungfrau", "Jungfrau"), ("lauterbrunnen", "Lauterbrunnen")]);
        // 22 km away, looking down at the summit from the north-north-west.
        let pose = found[0].pose(&ground, 1.0);
        assert!(pose.position.y > 5120.0 && pose.position.x < 5120.0 && pose.position.z > 2500.0);
    }
}
