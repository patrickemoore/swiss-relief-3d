//! The roll-free camera and its orbit and fly controllers.
//!
//! The camera is described by a heading and a pitch only. Keeping it free of
//! roll is what lets the light rig depend on just those two angles.
//!
//! The orbit controller is the map-like one. Dragging moves the ground with
//! the pointer and keeps coasting when let go, the wheel zooms toward the
//! pointer, and a jump to another place is a flight that pulls back, travels
//! and descends in one movement.

use std::cell::RefCell;
use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, PI, TAU};
use std::sync::Arc;

use glam::{Mat4, Vec2, Vec3};
use relief_terrain::terrain::{TILE_SIZE, TILE_SPAN};
use relief_terrain::{Grid, Terrain, TileSource};
use serde::{Deserialize, Serialize};

/// Vertical field of view the viewer starts with.
pub const DEFAULT_FOV_Y: f32 = std::f32::consts::FRAC_PI_4;
const NEAR_PLANE_M: f32 = 5.0;
/// Closest the camera may come to the ground.
const MIN_CLEARANCE_M: f32 = 30.0;
/// Lowest pitch of an orbit: 12 degrees below the horizon. Flatter than
/// that most of the view is far terrain seen edge-on, which the shading
/// generalizes to very little; the flying camera may still look there.
const MIN_ORBIT_PITCH: f32 = 0.21;
const MIN_ORBIT_DISTANCE_M: f32 = 300.0;
/// How far an orbit may back away from any terrain. A terrain too large to
/// be seen whole from there allows more.
const ORBIT_REACH_M: f32 = 150_000.0;
/// Farthest the orbit pivot may leave the data, in metres.
const MAX_PIVOT_MARGIN_M: f32 = 10_000.0;
/// Rate at which the flying camera approaches its goal, per second.
const SMOOTHING_RATE: f32 = 12.0;
/// Pyramid level the camera collides with: 20 m cells on a 10 m terrain.
pub const GROUND_LEVEL: usize = 1;
/// Tiles of ground heights kept in memory.
const GROUND_CACHE_TILES: usize = 160;
/// Longest ray cast against the terrain, in metres.
const MAX_RAY_M: f32 = 1_500_000.0;

/// Radians of rotation per pixel of drag.
const ROTATE_PER_PIXEL: f32 = 0.005;
/// Radians per second the arrow keys turn and tilt by.
const KEY_TURN_RATE: f32 = 1.2;
/// Orbit distances per second the movement keys pan by, and with shift.
const KEY_PAN_RATE: f32 = 0.7;
const FAST_KEY_PAN_RATE: f32 = 2.0;
/// How quickly key-driven motion picks up and dies down, per second.
const KEY_RESPONSE: f32 = 9.0;
/// Change of the natural log of the orbit distance per wheel step.
const ZOOM_PER_STEP: f32 = 0.18;
/// Rate at which the orbit distance approaches where the wheel put it.
const ZOOM_RATE: f32 = 9.0;
/// A zoom this close to its goal, as a log ratio, has arrived.
const ZOOM_SETTLED: f32 = 1e-3;
/// Time over which a drag's speed is averaged for the coast that follows.
const VELOCITY_WINDOW_S: f32 = 0.05;
/// Rates at which a coasting pan and a coasting turn die down, per second.
const PAN_FRICTION: f32 = 4.5;
const TURN_FRICTION: f32 = 8.0;
/// Fastest a turn coasts, in radians per second. A flick should settle the
/// view a little further round, not spin it.
const MAX_TURN_COAST: f32 = 1.5;
/// Rate at which the height of the orbit's centre follows the ground below
/// it. Slow enough that panning across mountains does not shake the camera.
const HEIGHT_RATE: f32 = 4.0;
/// Dragging toward the horizon moves the ground this many times faster than
/// across, at most: the foreshortening of a view 20 degrees above level.
const MAX_FORESHORTENING: f32 = 2.9;

/// The trade between zooming and travelling on a flight (van Wijk and Nuij,
/// "Smooth and efficient zooming and panning", 2003). At this value a long
/// flight pulls back just far enough to keep the ground from streaking.
const FLIGHT_RHO: f64 = 1.42;
/// Flight speed in the paper's units, and the limits on a flight's duration.
const FLIGHT_SPEED: f32 = 0.85;
const FLIGHT_SECONDS: std::ops::RangeInclusive<f32> = 1.0..=6.0;
/// Width of ground in view per metre of orbit distance, roughly.
const VIEW_WIDTH_PER_DISTANCE: f64 = 1.5;

/// The plain heights of one tile, apron included.
pub type TileHeights = Arc<Vec<f32>>;

/// Wraps an angle to `[-pi, pi)`.
pub fn wrap_angle(angle: f32) -> f32 {
    (angle + PI).rem_euclid(TAU) - PI
}

/// The share of the way an exponential approach at `rate` covers in `dt`.
fn approach(rate: f32, dt: f32) -> f32 {
    1.0 - (-rate * dt).exp()
}

/// Camera position and orientation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    /// Local metres: x east, y north, z up.
    pub position: Vec3,
    /// Viewing direction as a compass angle in radians, clockwise from north.
    pub heading: f32,
    /// Radians below the horizon: 0 looks at the horizon, pi/2 straight down.
    pub pitch: f32,
}

/// The camera's right, up and forward vectors.
#[derive(Clone, Copy, Debug)]
pub struct Basis {
    pub right: Vec3,
    pub up: Vec3,
    pub forward: Vec3,
}

impl Basis {
    /// Built analytically, so it stays well defined looking straight down,
    /// where heading 0 puts north at the top of the screen.
    pub fn new(heading: f32, pitch: f32) -> Self {
        let (sin_h, cos_h) = heading.sin_cos();
        let (sin_p, cos_p) = pitch.sin_cos();
        Self {
            right: Vec3::new(cos_h, -sin_h, 0.0),
            up: Vec3::new(sin_p * sin_h, sin_p * cos_h, cos_p),
            forward: Vec3::new(cos_p * sin_h, cos_p * cos_h, -sin_p),
        }
    }
}

impl Pose {
    pub fn basis(&self) -> Basis {
        Basis::new(self.heading, self.pitch)
    }

    /// A pose looking at `target` from `distance` away.
    pub fn orbiting(target: Vec3, distance: f32, heading: f32, pitch: f32) -> Self {
        let position = target - Basis::new(heading, pitch).forward * distance;
        Self { position, heading, pitch }
    }

    /// Reverse-Z projection with an infinite far plane.
    pub fn view_projection(&self, fov_y: f32, aspect: f32) -> Mat4 {
        let basis = self.basis();
        let view = Mat4::look_to_rh(self.position, basis.forward, basis.up);
        Mat4::perspective_infinite_reverse_rh(fov_y, aspect, NEAR_PLANE_M) * view
    }

    /// The direction of the ray through a pixel, counted from the top-left
    /// of a `viewport` given in pixels.
    pub fn ray_through(&self, pixel: Vec2, viewport: Vec2, fov_y: f32) -> Vec3 {
        let basis = self.basis();
        let tan_half = (0.5 * fov_y).tan();
        let ndc = Vec2::new(2.0 * pixel.x / viewport.x - 1.0, 1.0 - 2.0 * pixel.y / viewport.y);
        (basis.forward + basis.right * (ndc.x * tan_half * viewport.x / viewport.y) + basis.up * (ndc.y * tan_half))
            .normalize()
    }
}

/// Where the camera's ground heights come from, a tile of the ground's
/// level at a time.
pub trait GroundTiles {
    /// The plain heights of a tile, or `None` if it has no data or is not
    /// at hand; the ground is then taken from the finest whole level.
    fn heights(&self, tile: [usize; 2]) -> Option<TileHeights>;
}

/// Ground tiles read from a terrain's own source when first needed and kept
/// in a small cache, so a terrain of any size costs the same.
struct ReadOnDemand {
    tiles: Arc<dyn TileSource>,
    cache: RefCell<HashMap<[usize; 2], Option<TileHeights>>>,
}

impl GroundTiles for ReadOnDemand {
    fn heights(&self, tile: [usize; 2]) -> Option<TileHeights> {
        let mut cache = self.cache.borrow_mut();
        if let Some(known) = cache.get(&tile) {
            return known.clone();
        }
        if cache.len() >= GROUND_CACHE_TILES {
            cache.clear();
        }
        let heights = match self.tiles.tile(GROUND_LEVEL, tile[0], tile[1]) {
            Ok(found) => found.map(|found| Arc::new(found.plain)),
            Err(error) => {
                // The whole level stands in; the camera just collides with
                // smoother ground there.
                log::warn!("{error:#}");
                None
            }
        };
        cache.insert(tile, heights.clone());
        heights
    }
}

/// Terrain heights on the CPU, for camera collision and picking.
pub struct Ground {
    tiles: Box<dyn GroundTiles>,
    /// The finest whole level, for places that have no tiles.
    whole: Grid<f32>,
    /// Side length of the terrain square in metres.
    extent: f32,
    /// Cell size of the level the tiles are of, in metres.
    cell: f32,
    /// Where the terrain holds data: west, south, east and north in metres.
    bounds: [f32; 4],
}

impl Ground {
    /// The ground of a terrain whose tiles can be read when asked for.
    pub fn new(terrain: &Terrain) -> Self {
        let tiles = ReadOnDemand { tiles: terrain.tiles.clone(), cache: RefCell::new(HashMap::new()) };
        Self::with_tiles(terrain, GROUND_LEVEL, Box::new(tiles))
    }

    /// The ground of a terrain whose tiles, of pyramid level `level`, arrive
    /// some other way, as they do in a browser.
    pub fn with_tiles(terrain: &Terrain, level: usize, tiles: Box<dyn GroundTiles>) -> Self {
        Self {
            tiles,
            whole: terrain.coarse[0][0].clone(),
            extent: terrain.meta.extent(),
            cell: terrain.cell(level),
            bounds: terrain.meta.data_region(),
        }
    }

    /// Where the terrain holds data: west, south, east and north in metres.
    pub fn bounds(&self) -> [f32; 4] {
        self.bounds
    }

    /// The farthest an orbit may back away: far enough to see all the data,
    /// even in a view much narrower than the data is wide.
    pub fn orbit_reach(&self) -> f32 {
        let [west, south, east, north] = self.bounds;
        ORBIT_REACH_M.max(3.0 * (east - west).max(north - south))
    }

    /// True terrain height in metres; positions outside the terrain clamp.
    pub fn height(&self, xy: Vec2) -> f32 {
        let xy = xy.clamp(Vec2::ZERO, Vec2::splat(self.extent));
        // In texels of the ground level, counted from the centre of texel 0.
        let texel = (xy / self.cell - 0.5).max(Vec2::ZERO);
        let tile = [texel.x as usize / TILE_SIZE, texel.y as usize / TILE_SIZE];
        let Some(heights) = self.tiles.heights(tile) else {
            return self.whole.sample(xy.x / self.extent, xy.y / self.extent);
        };

        // A tile's texels start one in from the edge of its heights, whose
        // apron always provides the next texel for interpolation.
        let local = texel - Vec2::new(tile[0] as f32, tile[1] as f32) * TILE_SIZE as f32 + 1.0;
        let (column, row) = (local.x as usize, local.y as usize);
        let (fx, fy) = (local.x - column as f32, local.y - row as f32);
        let at = |column: usize, row: usize| heights[row * TILE_SPAN + column];
        let south = at(column, row) * (1.0 - fx) + at(column + 1, row) * fx;
        let north = at(column, row + 1) * (1.0 - fx) + at(column + 1, row + 1) * fx;
        south * (1.0 - fy) + north * fy
    }

    /// First intersection of a ray with the terrain drawn at vertical scale
    /// `z_scale`: marches with steps proportional to the clearance, then
    /// bisects the step that went below ground.
    pub fn raycast(&self, origin: Vec3, direction: Vec3, z_scale: f32) -> Option<Vec3> {
        let clearance = |t: f32| {
            let point = origin + direction * t;
            point.z - self.height(point.truncate()) * z_scale
        };
        let (mut above, mut t) = (0.0, 0.0);
        while t < MAX_RAY_M {
            let height = clearance(t);
            if height <= 0.0 {
                let mut below = t;
                for _ in 0..16 {
                    let middle = 0.5 * (above + below);
                    if clearance(middle) > 0.0 {
                        above = middle;
                    } else {
                        below = middle;
                    }
                }
                return Some(origin + direction * below);
            }
            above = t;
            // The minimum step grows with distance, which bounds the march.
            t += (0.4 * height).max(2.0 + 0.002 * t);
        }
        None
    }
}

/// What an orbiting camera looks at, and from where.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrbitView {
    /// The ground point at the centre of the view, in local metres.
    pub pivot: Vec2,
    /// Distance from the camera to that point, in metres.
    pub distance: f32,
    pub heading: f32,
    pub pitch: f32,
}

impl OrbitView {
    /// The camera pose of the view over the ground as it truly is.
    pub fn pose(&self, ground: &Ground, z_scale: f32) -> Pose {
        let target = self.pivot.extend(ground.height(self.pivot) * z_scale);
        Pose::orbiting(target, self.distance, self.heading, self.pitch)
    }
}

/// The zoom and pan of a flight between two views of the ground: the path
/// through (position, view width) space that an observer finds smoothest.
/// It pulls back while it travels and comes down again at the far end.
#[derive(Clone, Copy, Debug)]
struct ZoomPath {
    /// Ground distance to cover, and the widths of view at either end.
    travel: f64,
    width: [f64; 2],
    /// The path's parameter at its start.
    start: f64,
    /// Length of the path in the units its speed is measured in.
    length: f64,
}

impl ZoomPath {
    fn new(travel: f64, width: [f64; 2]) -> Self {
        let [w0, w1] = width;
        // Hardly any travel: the path is a plain zoom.
        if travel < 1e-4 * w0.min(w1) {
            return Self { travel: 0.0, width, start: 0.0, length: (w1 / w0).ln().abs() / FLIGHT_RHO };
        }
        let rho2 = FLIGHT_RHO * FLIGHT_RHO;
        let end = |w: f64, sign: f64| {
            let b = (w1 * w1 - w0 * w0 + sign * rho2 * rho2 * travel * travel) / (2.0 * w * rho2 * travel);
            (-b + (b * b + 1.0).sqrt()).ln()
        };
        let (start, finish) = (end(w0, 1.0), end(w1, -1.0));
        Self { travel, width, start, length: (finish - start) / FLIGHT_RHO }
    }

    /// The share of the travel covered and the width of view after `s` of
    /// the path's length.
    fn at(&self, s: f64) -> (f64, f64) {
        let [w0, w1] = self.width;
        if self.travel == 0.0 {
            let share = if self.length > 0.0 { s / self.length } else { 1.0 };
            return (share, w0 * (w1 / w0).powf(share));
        }
        let along = FLIGHT_RHO * s + self.start;
        let covered = w0 / (FLIGHT_RHO * FLIGHT_RHO) * (self.start.cosh() * along.tanh() - self.start.sinh());
        (covered / self.travel, w0 * self.start.cosh() / along.cosh())
    }
}

/// A flight from one orbit view to another.
#[derive(Clone, Copy, Debug)]
struct Flight {
    from: OrbitView,
    to: OrbitView,
    path: ZoomPath,
    duration: f32,
    time: f32,
}

impl Flight {
    fn new(from: OrbitView, to: OrbitView) -> Self {
        let width = [from.distance, to.distance].map(|distance| distance as f64 * VIEW_WIDTH_PER_DISTANCE);
        let path = ZoomPath::new(from.pivot.distance(to.pivot) as f64, width);
        let duration = (path.length as f32 / FLIGHT_SPEED).clamp(*FLIGHT_SECONDS.start(), *FLIGHT_SECONDS.end());
        Self { from, to, path, duration, time: 0.0 }
    }

    fn finished(&self) -> bool {
        self.time >= self.duration
    }

    fn view(&self) -> OrbitView {
        if self.finished() {
            return self.to;
        }
        // Starts and ends at rest.
        let t = (self.time / self.duration).clamp(0.0, 1.0);
        let eased = t * t * t * (t * (6.0 * t - 15.0) + 10.0);
        let (share, width) = self.path.at(eased as f64 * self.path.length);
        OrbitView {
            pivot: self.from.pivot.lerp(self.to.pivot, share as f32),
            distance: (width / VIEW_WIDTH_PER_DISTANCE) as f32,
            heading: wrap_angle(self.from.heading + wrap_angle(self.to.heading - self.from.heading) * eased),
            pitch: self.from.pitch + (self.to.pitch - self.from.pitch) * eased,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraMode {
    /// Rotate around and pan a pivot on the ground.
    Orbit,
    /// Move freely.
    Fly,
}

/// Keys, pointer motion and gestures gathered since the last update.
#[derive(Clone, Copy, Debug, Default)]
pub struct CameraInput {
    /// Right, forward and up intent, each in `[-1, 1]`.
    pub movement: Vec3,
    /// Turn (clockwise) and tilt (down) intent from the keyboard.
    pub turn: Vec2,
    /// Pointer drag in pixels that moves the ground.
    pub primary_drag: Vec2,
    /// Pointer drag in pixels that turns and tilts the view.
    pub secondary_drag: Vec2,
    /// Scroll wheel steps; positive zooms in.
    pub scroll: f32,
    /// Zoom from a pinch, as the natural log of its factor; positive zooms
    /// in. Unlike the wheel it is followed at once.
    pub pinch: f32,
    /// Where the pointer is, in pixels from the top-left of the view.
    /// Zooming holds the ground there in place.
    pub cursor: Option<Vec2>,
    /// Whether a button or finger is down. When it lifts, the view coasts.
    pub dragging: bool,
    pub fast: bool,
    /// Size of the 3D view in pixels.
    pub viewport: Vec2,
}

impl CameraInput {
    /// Whether the input moves the camera this frame.
    pub fn steers(&self) -> bool {
        self.movement != Vec3::ZERO
            || self.turn != Vec2::ZERO
            || self.primary_drag != Vec2::ZERO
            || self.secondary_drag != Vec2::ZERO
            || self.scroll != 0.0
            || self.pinch != 0.0
    }
}

/// Turns input into a collision-free camera pose.
pub struct CameraController {
    pub mode: CameraMode,
    pub fov_y: f32,
    /// Fly speed in metres per second.
    pub fly_speed: f32,
    /// Radians per second the orbit turns by itself while nothing steers it.
    pub idle_spin: f32,
    /// The orbit; its heading and pitch also aim the flying camera.
    view: OrbitView,
    /// Height of the ground at the centre of the orbit, before vertical
    /// exaggeration. It follows the true height smoothly.
    pivot_height: f32,
    /// The orbit distance the wheel has asked for.
    zoom_goal: f32,
    /// The ground point a zoom holds in place.
    zoom_anchor: Option<Vec2>,
    /// What is left of a drag: metres and radians per second.
    pan_velocity: Vec2,
    turn_velocity: Vec2,
    /// Key-driven motion, eased in and out.
    key_pan: Vec2,
    key_turn: Vec2,
    flight: Option<Flight>,
    // Fly goal.
    position: Vec3,
    pose: Pose,
}

impl CameraController {
    pub fn new(pose: Pose, ground: &Ground, z_scale: f32) -> Self {
        let mut controller = Self {
            mode: CameraMode::Orbit,
            fov_y: DEFAULT_FOV_Y,
            fly_speed: 600.0,
            idle_spin: 0.0,
            view: OrbitView { pivot: Vec2::ZERO, distance: 1.0, heading: 0.0, pitch: FRAC_PI_2 },
            pivot_height: 0.0,
            zoom_goal: 1.0,
            zoom_anchor: None,
            pan_velocity: Vec2::ZERO,
            turn_velocity: Vec2::ZERO,
            key_pan: Vec2::ZERO,
            key_turn: Vec2::ZERO,
            flight: None,
            position: Vec3::ZERO,
            pose,
        };
        controller.set_pose(pose, ground, z_scale);
        controller
    }

    /// The pose to render with.
    pub fn pose(&self) -> Pose {
        self.pose
    }

    /// The orbit the camera is on. Meaningful in orbit mode.
    pub fn view(&self) -> OrbitView {
        self.view
    }

    /// Ground metres per pixel at the centre of a view `viewport_height`
    /// pixels high.
    pub fn metres_per_pixel(&self, viewport_height: f32) -> f32 {
        self.view.distance * 2.0 * (0.5 * self.fov_y).tan() / viewport_height.max(1.0)
    }

    /// Whether the camera will keep moving without further input.
    pub fn in_motion(&self, ground: &Ground) -> bool {
        let slow = 1e-4;
        self.flight.is_some()
            || self.idle_spin != 0.0
            || self.zoom_goal != self.view.distance
            || self.pan_velocity.length() > slow * self.view.distance
            || self.turn_velocity.length() > slow
            || self.key_pan.length() > slow
            || self.key_turn.length() > slow
            || match self.mode {
                CameraMode::Orbit => (ground.height(self.view.pivot) - self.pivot_height).abs() > 0.5,
                CameraMode::Fly => self.position.distance(self.pose.position) > 0.01,
            }
    }

    /// Whether the camera is on its way to a view set by
    /// [`fly_to`](Self::fly_to).
    pub fn in_flight(&self) -> bool {
        self.flight.is_some()
    }

    /// The view that terrain detail should be loaded for during a flight:
    /// where it left from until half-way, when the camera is high and
    /// fast, and from then on where it is going, so that the detail is
    /// there on arrival. Loading for the camera itself would fetch what
    /// only streaks past. `None` when not in flight.
    pub fn detail_view(&self) -> Option<OrbitView> {
        let flight = self.flight.as_ref()?;
        Some(if flight.time < 0.5 * flight.duration { flight.from } else { flight.to })
    }

    /// Jumps to a pose at once, for the evaluation viewpoints and path
    /// playback.
    ///
    /// An orbit can only reproduce a pose that looks at the ground, so a pose
    /// looking past the terrain switches the controller to fly mode.
    pub fn set_pose(&mut self, pose: Pose, ground: &Ground, z_scale: f32) {
        self.halt();
        self.pose = pose;
        self.position = pose.position;
        self.view.heading = pose.heading;
        self.view.pitch = pose.pitch;
        if !self.adopt_orbit_pivot(ground, z_scale) {
            self.mode = CameraMode::Fly;
        }
    }

    /// Jumps to an orbit view at once.
    pub fn set_view(&mut self, view: OrbitView, ground: &Ground, z_scale: f32) {
        self.halt();
        self.mode = CameraMode::Orbit;
        self.view = Self::within_limits(view, ground);
        self.zoom_goal = self.view.distance;
        self.pivot_height = ground.height(self.view.pivot);
        self.pose = self.orbit_pose(z_scale);
        self.position = self.pose.position;
    }

    /// Starts a flight to an orbit view. Any steering ends it where it is.
    pub fn fly_to(&mut self, view: OrbitView, ground: &Ground, z_scale: f32) {
        self.set_mode(CameraMode::Orbit, ground, z_scale);
        self.halt();
        self.flight = Some(Flight::new(self.view, Self::within_limits(view, ground)));
    }

    /// Zooms by a factor, eased like the wheel; above 1 zooms in.
    pub fn zoom_by(&mut self, factor: f32, ground: &Ground) {
        self.flight = None;
        self.zoom_anchor = None;
        self.zoom_goal = (self.zoom_goal / factor).clamp(MIN_ORBIT_DISTANCE_M, ground.orbit_reach());
    }

    pub fn set_mode(&mut self, mode: CameraMode, ground: &Ground, z_scale: f32) {
        if mode == self.mode {
            return;
        }
        self.halt();
        self.mode = mode;
        self.position = self.pose.position;
        self.view.heading = self.pose.heading;
        self.view.pitch = self.pose.pitch;
        if mode == CameraMode::Orbit && !self.adopt_orbit_pivot(ground, z_scale) {
            // Looking at the sky: orbit the ground some way ahead instead.
            let ahead = self.position.truncate() + Vec2::new(self.view.heading.sin(), self.view.heading.cos()) * 3000.0;
            self.pivot_height = ground.height(ahead);
            let offset = self.position - ahead.extend(self.pivot_height * z_scale);
            let distance = offset.length().clamp(MIN_ORBIT_DISTANCE_M, ground.orbit_reach());
            self.view = OrbitView {
                pivot: ahead,
                distance,
                heading: self.view.heading,
                pitch: (offset.z / distance).clamp(-1.0, 1.0).asin().clamp(MIN_ORBIT_PITCH, FRAC_PI_2),
            };
            self.zoom_goal = distance;
        }
    }

    /// Stops everything that would move the camera by itself.
    fn halt(&mut self) {
        self.flight = None;
        self.zoom_anchor = None;
        self.zoom_goal = self.view.distance;
        self.pan_velocity = Vec2::ZERO;
        self.turn_velocity = Vec2::ZERO;
        self.key_pan = Vec2::ZERO;
        self.key_turn = Vec2::ZERO;
    }

    fn within_limits(view: OrbitView, ground: &Ground) -> OrbitView {
        OrbitView {
            pivot: Self::within_bounds(view.pivot, ground),
            distance: view.distance.clamp(MIN_ORBIT_DISTANCE_M, ground.orbit_reach()),
            heading: wrap_angle(view.heading),
            pitch: view.pitch.clamp(MIN_ORBIT_PITCH, FRAC_PI_2),
        }
    }

    /// The pivot may leave the data by a quarter of its width, but never so
    /// far that a close view shows nothing at all.
    fn within_bounds(pivot: Vec2, ground: &Ground) -> Vec2 {
        let [west, south, east, north] = ground.bounds();
        let margin = (0.25 * (east - west)).min(MAX_PIVOT_MARGIN_M);
        pivot.clamp(Vec2::new(west, south) - margin, Vec2::new(east, north) + margin)
    }

    /// Takes the ground point the view looks at as the orbit pivot. Returns
    /// false if the view does not meet the ground.
    fn adopt_orbit_pivot(&mut self, ground: &Ground, z_scale: f32) -> bool {
        let forward = Basis::new(self.view.heading, self.view.pitch).forward;
        let Some(hit) = ground.raycast(self.position, forward, z_scale) else {
            return false;
        };
        let distance = (self.position - hit).length();
        if self.view.pitch < MIN_ORBIT_PITCH || !(MIN_ORBIT_DISTANCE_M..=ground.orbit_reach()).contains(&distance) {
            return false;
        }
        self.view.pivot = hit.truncate();
        self.view.distance = distance;
        self.zoom_goal = distance;
        // Exactly the height that was hit, so the orbit reproduces the pose.
        self.pivot_height = hit.z / z_scale;
        true
    }

    fn orbit_pose(&self, z_scale: f32) -> Pose {
        let target = self.view.pivot.extend(self.pivot_height * z_scale);
        Pose::orbiting(target, self.view.distance, self.view.heading, self.view.pitch)
    }

    pub fn update(&mut self, input: &CameraInput, dt: f32, ground: &Ground, z_scale: f32) {
        let dt = dt.max(1e-4);
        match self.mode {
            CameraMode::Orbit => {
                self.update_orbit(input, dt, ground, z_scale);
                self.pose = self.orbit_pose(z_scale);
                self.position = self.pose.position;
            }
            CameraMode::Fly => {
                self.update_fly(input, dt);
                let blend = approach(SMOOTHING_RATE, dt);
                self.pose.position += (self.position - self.pose.position) * blend;
                self.pose.heading =
                    wrap_angle(self.pose.heading + wrap_angle(self.view.heading - self.pose.heading) * blend);
                self.pose.pitch += (self.view.pitch - self.pose.pitch) * blend;
            }
        }
        let floor = ground.height(self.pose.position.truncate()) * z_scale + MIN_CLEARANCE_M;
        self.pose.position.z = self.pose.position.z.max(floor);
    }

    fn update_orbit(&mut self, input: &CameraInput, dt: f32, ground: &Ground, z_scale: f32) {
        if input.steers() {
            self.flight = None;
        }
        if let Some(flight) = &mut self.flight {
            flight.time += dt;
            self.view = flight.view();
            self.zoom_goal = self.view.distance;
            if flight.finished() {
                self.flight = None;
            }
        } else {
            self.turn(input, dt);
            self.zoom(input, dt, ground, z_scale);
            self.pan(input, dt, ground);
        }
        self.pivot_height += (ground.height(self.view.pivot) - self.pivot_height) * approach(HEIGHT_RATE, dt);
    }

    fn turn(&mut self, input: &CameraInput, dt: f32) {
        let dragged = input.secondary_drag * ROTATE_PER_PIXEL;
        self.key_turn += (input.turn * KEY_TURN_RATE - self.key_turn) * approach(KEY_RESPONSE, dt);
        let mut turned = dragged + self.key_turn * dt;
        if input.dragging {
            self.turn_velocity += (dragged / dt - self.turn_velocity) * approach(1.0 / VELOCITY_WINDOW_S, dt);
        } else {
            self.turn_velocity = self.turn_velocity.clamp_length_max(MAX_TURN_COAST);
            turned += self.turn_velocity * dt;
            self.turn_velocity *= (-TURN_FRICTION * dt).exp();
            if !input.steers() {
                turned.x += self.idle_spin * dt;
            }
        }
        self.view.heading = wrap_angle(self.view.heading + turned.x);
        self.view.pitch = (self.view.pitch + turned.y).clamp(MIN_ORBIT_PITCH, FRAC_PI_2);
    }

    fn zoom(&mut self, input: &CameraInput, dt: f32, ground: &Ground, z_scale: f32) {
        let asked = ZOOM_PER_STEP * input.scroll + input.pinch;
        if asked != 0.0 {
            self.zoom_goal = (self.zoom_goal * (-asked).exp()).clamp(MIN_ORBIT_DISTANCE_M, ground.orbit_reach());
            self.zoom_anchor = input.cursor.and_then(|cursor| {
                let ray = self.pose.ray_through(cursor, input.viewport, self.fov_y);
                ground.raycast(self.pose.position, ray, z_scale).map(Vec3::truncate)
            });
        }
        let before = self.view.distance;
        if before == self.zoom_goal {
            return;
        }
        // A pinch is followed at once; the wheel is eased, in log distance.
        let remaining = if input.pinch != 0.0 { 0.0 } else { (-ZOOM_RATE * dt).exp() };
        let mut after = self.zoom_goal * (before / self.zoom_goal).powf(remaining);
        if (after / self.zoom_goal).ln().abs() < ZOOM_SETTLED {
            after = self.zoom_goal;
        }
        if let Some(anchor) = self.zoom_anchor {
            self.view.pivot = Self::within_bounds(anchor + (self.view.pivot - anchor) * (after / before), ground);
        }
        self.view.distance = after;
        if after == self.zoom_goal {
            self.zoom_anchor = None;
        }
    }

    fn pan(&mut self, input: &CameraInput, dt: f32, ground: &Ground) {
        let (sin, cos) = self.view.heading.sin_cos();
        let (screen_right, screen_up) = (Vec2::new(cos, -sin), Vec2::new(sin, cos));
        // One pixel of drag covers one pixel's worth of ground at the pivot,
        // which lies farther apart toward the horizon in a tilted view.
        let metres_per_pixel = self.metres_per_pixel(input.viewport.y);
        let foreshortening = (1.0 / self.view.pitch.sin().max(1e-3)).min(MAX_FORESHORTENING);
        let dragged = (screen_right * -input.primary_drag.x + screen_up * input.primary_drag.y * foreshortening)
            * metres_per_pixel;

        let rate = if input.fast { FAST_KEY_PAN_RATE } else { KEY_PAN_RATE };
        self.key_pan += (input.movement.truncate() * rate - self.key_pan) * approach(KEY_RESPONSE, dt);
        let keyed = (screen_right * self.key_pan.x + screen_up * self.key_pan.y) * self.view.distance * dt;

        let mut moved = dragged + keyed;
        if input.dragging {
            self.pan_velocity += (dragged / dt - self.pan_velocity) * approach(1.0 / VELOCITY_WINDOW_S, dt);
        } else {
            moved += self.pan_velocity * dt;
            self.pan_velocity *= (-PAN_FRICTION * dt).exp();
        }
        let wanted = self.view.pivot + moved;
        self.view.pivot = Self::within_bounds(wanted, ground);
        if self.view.pivot != wanted {
            // Coasting into the edge of the terrain stops there.
            self.pan_velocity = Vec2::ZERO;
        }
    }

    fn update_fly(&mut self, input: &CameraInput, dt: f32) {
        const LOOK_PER_PIXEL: f32 = 0.003;
        const FLY_TURN_RATE: f32 = 1.0;
        const MAX_LOOK_UP: f32 = -1.2;

        let drag = input.primary_drag + input.secondary_drag;
        self.view.heading = wrap_angle(self.view.heading + drag.x * LOOK_PER_PIXEL + input.turn.x * FLY_TURN_RATE * dt);
        self.view.pitch = (self.view.pitch + drag.y * LOOK_PER_PIXEL + input.turn.y * FLY_TURN_RATE * dt)
            .clamp(MAX_LOOK_UP, FRAC_PI_2);
        self.fly_speed = (self.fly_speed * (0.2 * input.scroll).exp()).clamp(20.0, 20_000.0);

        let basis = Basis::new(self.view.heading, self.view.pitch);
        let speed = self.fly_speed * if input.fast { 4.0 } else { 1.0 };
        let velocity = basis.right * input.movement.x + basis.forward * input.movement.y + Vec3::Z * input.movement.z;
        self.position += velocity * speed * dt;
    }
}

#[cfg(test)]
mod tests {
    use relief_terrain::{BuildOptions, Dataset};

    use super::*;

    const VIEWPORT: Vec2 = Vec2::new(1600.0, 900.0);
    const DT: f32 = 1.0 / 60.0;

    /// A plain at 100 m, 8 km on a side.
    fn flat_ground() -> Ground {
        let dataset =
            Dataset::build("flat", "test", Grid::new(64, 64, 100.0), 125.0, [0.0; 2], &BuildOptions::default());
        Ground::new(&Terrain::from(dataset.unwrap()))
    }

    /// A camera 2 km above the middle of the plain, tilted 50 degrees.
    fn controller(ground: &Ground) -> CameraController {
        let view = OrbitView { pivot: Vec2::splat(4000.0), distance: 2000.0, heading: 0.4, pitch: 50f32.to_radians() };
        let mut controller = CameraController::new(Pose::orbiting(Vec3::ZERO, 1.0, 0.0, 1.0), ground, 1.0);
        controller.set_view(view, ground, 1.0);
        controller
    }

    fn input() -> CameraInput {
        CameraInput { viewport: VIEWPORT, ..Default::default() }
    }

    /// Runs the controller without input until it comes to rest.
    fn settle(controller: &mut CameraController, ground: &Ground) -> usize {
        let mut frames = 0;
        while controller.in_motion(ground) {
            controller.update(&input(), DT, ground, 1.0);
            frames += 1;
            assert!(frames < 2000, "the camera never comes to rest");
        }
        frames
    }

    #[test]
    fn ground_heights_follow_the_terrain() {
        let terrain = Terrain::from(relief_terrain::synthetic::cone_dataset(1024).unwrap());
        let ground = Ground::new(&terrain);
        // The cone rises from a plain at 500 m to 2500 m at the centre.
        assert!((ground.height(Vec2::new(300.0, 300.0)) - 500.0).abs() < 1.0);
        let summit = ground.height(Vec2::splat(5120.0));
        assert!(summit > 2400.0 && summit <= 2500.0, "summit at {summit} m");
        // Half-way up the slope, and unchanged by clamping outside.
        assert!((ground.height(Vec2::new(5120.0 + 1792.0, 5120.0)) - 1500.0).abs() < 15.0);
        assert_eq!(ground.height(Vec2::new(-500.0, 300.0)), ground.height(Vec2::new(0.0, 300.0)));
    }

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-5
    }

    #[test]
    fn looking_straight_down_puts_north_up() {
        let basis = Basis::new(0.0, FRAC_PI_2);
        assert!(close(basis.right, Vec3::X) && close(basis.up, Vec3::Y) && close(basis.forward, -Vec3::Z));
    }

    #[test]
    fn looking_east_at_the_horizon() {
        let basis = Basis::new(FRAC_PI_2, 0.0);
        assert!(close(basis.forward, Vec3::X) && close(basis.up, Vec3::Z) && close(basis.right, -Vec3::Y));
    }

    #[test]
    fn basis_is_orthonormal_and_right_handed() {
        for (heading, pitch) in [(0.3, 0.2), (2.0, 1.1), (-1.0, FRAC_PI_2), (4.0, -0.4)] {
            let basis = Basis::new(heading, pitch);
            assert!(close(basis.right.cross(basis.up), -basis.forward));
            assert!((basis.forward.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn wrap_angle_stays_in_range() {
        assert!((wrap_angle(3.0 * PI) - -PI).abs() < 1e-5);
        assert!((wrap_angle(-0.5) - -0.5).abs() < 1e-6);
        assert!((wrap_angle(TAU + 0.25) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn raycast_hits_a_plane() {
        let ground = flat_ground();
        let hit = ground.raycast(Vec3::new(4000.0, 4000.0, 1100.0), -Vec3::Z, 1.0).unwrap();
        assert!((hit.z - 100.0).abs() < 2.0);
        assert!(ground.raycast(Vec3::new(4000.0, 4000.0, 1100.0), Vec3::Z, 1.0).is_none());
    }

    #[test]
    fn the_ray_through_the_middle_of_the_view_is_forward() {
        let pose = Pose { position: Vec3::ZERO, heading: 1.0, pitch: 0.6 };
        assert!(close(pose.ray_through(0.5 * VIEWPORT, VIEWPORT, 0.8), pose.basis().forward));
        // The top of the view looks farther up than its middle.
        let top = pose.ray_through(Vec2::new(800.0, 0.0), VIEWPORT, 0.8);
        assert!(top.z > pose.basis().forward.z);
    }

    #[test]
    fn an_orbit_reproduces_the_pose_it_was_set_to() {
        let ground = flat_ground();
        let pose = Pose::orbiting(Vec3::new(4000.0, 4000.0, 100.0), 1500.0, 2.0, 0.7);
        let mut controller = CameraController::new(pose, &ground, 1.0);
        assert_eq!(controller.mode, CameraMode::Orbit);
        controller.update(&input(), DT, &ground, 1.0);
        assert!((controller.pose().position - pose.position).length() < 1.0);
        assert!(!controller.in_motion(&ground));
    }

    #[test]
    fn a_drag_moves_the_ground_with_the_pointer_and_then_coasts() {
        let ground = flat_ground();
        let mut controller = controller(&ground);
        let start = controller.view().pivot;
        let metres_per_pixel = controller.metres_per_pixel(VIEWPORT.y);

        // Dragging right moves the ground right, so the view moves left.
        let drag = CameraInput { primary_drag: Vec2::new(10.0, 0.0), dragging: true, ..input() };
        for _ in 0..10 {
            controller.update(&drag, DT, &ground, 1.0);
        }
        let dragged = controller.view().pivot - start;
        let (sin, cos) = controller.view().heading.sin_cos();
        let screen_right = Vec2::new(cos, -sin);
        assert!((dragged.dot(screen_right) + 100.0 * metres_per_pixel).abs() < 1e-2);
        assert!(dragged.perp_dot(screen_right).abs() < 1e-2);

        // Let go, and it keeps going the same way for a while, then stops.
        let frames = settle(&mut controller, &ground);
        let coasted = controller.view().pivot - start - dragged;
        assert!(frames > 20 && coasted.dot(dragged) > 0.0);
        assert!(coasted.length() > 0.5 * dragged.length() && coasted.length() < 3.0 * dragged.length());
    }

    #[test]
    fn the_wheel_zooms_toward_the_pointer() {
        let ground = flat_ground();
        let mut controller = controller(&ground);
        let cursor = Vec2::new(1200.0, 300.0);
        let under = |controller: &CameraController| {
            let pose = controller.pose();
            ground.raycast(pose.position, pose.ray_through(cursor, VIEWPORT, controller.fov_y), 1.0).unwrap()
        };
        let before = under(&controller);

        controller.update(&CameraInput { scroll: 3.0, cursor: Some(cursor), ..input() }, DT, &ground, 1.0);
        // It eases in rather than jumping.
        assert!(controller.view().distance > 2000.0 * (-ZOOM_PER_STEP * 3.0f32).exp() * 1.2);
        settle(&mut controller, &ground);
        assert!((controller.view().distance - 2000.0 * (-ZOOM_PER_STEP * 3.0f32).exp()).abs() < 1.0);
        // The ground under the pointer is still the same ground.
        assert!((under(&controller) - before).length() < 5.0, "moved {} m", (under(&controller) - before).length());
    }

    #[test]
    fn a_flight_arrives_at_rest_and_steering_ends_it() {
        let ground = flat_ground();
        let mut controller = controller(&ground);
        let far = OrbitView { pivot: Vec2::new(1500.0, 6500.0), distance: 900.0, heading: -2.5, pitch: 0.5 };
        controller.fly_to(far, &ground, 1.0);
        assert!(controller.in_flight());

        // Half-way there the camera has pulled back to see more ground.
        let mut widest = 0.0f32;
        while controller.in_flight() {
            controller.update(&input(), DT, &ground, 1.0);
            widest = widest.max(controller.view().distance);
        }
        assert!(widest > 2500.0, "the flight never pulled back: {widest} m");
        let arrived = controller.view();
        assert!(arrived.pivot.distance(far.pivot) < 0.1 && (arrived.distance - far.distance).abs() < 0.1);
        assert!(wrap_angle(arrived.heading - far.heading).abs() < 1e-4 && (arrived.pitch - far.pitch).abs() < 1e-4);

        controller.fly_to(OrbitView { pivot: Vec2::splat(4000.0), ..far }, &ground, 1.0);
        controller.update(&input(), DT, &ground, 1.0);
        controller.update(&CameraInput { scroll: 1.0, ..input() }, DT, &ground, 1.0);
        assert!(!controller.in_flight());
    }

    #[test]
    fn a_zoom_path_starts_and_ends_where_it_should() {
        for (travel, width) in
            [(120_000.0, [450_000.0, 21_000.0]), (5000.0, [3000.0, 3000.0]), (0.0, [900.0, 60_000.0])]
        {
            let path = ZoomPath::new(travel, width);
            let (start, end) = (path.at(0.0), path.at(path.length));
            assert!(start.0.abs() < 1e-9 && (start.1 / width[0] - 1.0).abs() < 1e-9);
            assert!((end.0 - 1.0).abs() < 1e-6 && (end.1 / width[1] - 1.0).abs() < 1e-6, "{end:?}");
            assert!(path.length > 0.0);
        }
        // Nothing to do takes no time and divides by nothing.
        let still = ZoomPath::new(0.0, [1000.0, 1000.0]);
        assert_eq!((still.length, still.at(0.0)), (0.0, (1.0, 1000.0)));
    }
}
