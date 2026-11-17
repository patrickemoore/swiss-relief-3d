//! The viewer as a visitor uses it: a map to move around in, places to fly
//! to, and two looks of the same terrain to compare.
//!
//! Nothing here touches the GPU or a window. The browser build drives it
//! from page events; it also knows when a frame would only repeat the last
//! one, so an idle map costs nothing.

use std::f32::consts::FRAC_PI_2;

use glam::{Vec2, Vec3};
use relief_terrain::{Meta, Terrain};

use crate::camera::{CameraController, CameraInput, Ground, OrbitView, Pose, DEFAULT_FOV_Y};
use crate::light_rig::LightRig;
use crate::params::{Condition, ShadingParams};
use crate::path::{self, Place, WHOLE_COUNTRY};

/// Seconds without steering after which an idle view starts to turn.
const IDLE_AFTER_S: f32 = 5.0;
/// How fast an idle view turns, in radians per second, and how quickly it
/// gets there and stops again, in radians per second squared.
const IDLE_SPIN: f32 = 0.03;
const IDLE_SPIN_RAMP: f32 = 0.03;
/// A view steeper than this is a map; it keeps north where it is.
const MAP_PITCH: f32 = 1.35;
/// Seconds a tour stays at each place.
const TOUR_DWELL_S: f32 = 8.0;
/// Lights that move less than this from one frame to the next have arrived.
const LIGHT_SETTLED: f32 = 1e-4;
/// A terrain this many times wider than its view region is a country, not a
/// region.
const COUNTRY_SIDES: f32 = 2.0;

/// The view a terrain opens on, for a view `aspect` times as wide as high.
pub fn opening_view(meta: &Meta, aspect: f32) -> OrbitView {
    opening_view_within(meta, Vec2::new(aspect, 1.0), [0.0; 4])
}

/// The view a terrain opens on in a `viewport` whose edges are partly
/// covered: `insets` are the widths covered at the top, right, bottom and
/// left, in the units of the viewport.
pub fn opening_view_within(meta: &Meta, viewport: Vec2, insets: [f32; 4]) -> OrbitView {
    let [west, south, east, north] = meta.view_region();
    let side = (east - west).min(north - south);
    let data = meta.data_region();
    let data_size = Vec2::new(data[2] - data[0], data[3] - data[1]);
    if data_size.max_element() > COUNTRY_SIDES * side {
        // A country opens the way its printed map looks: all of it, from
        // straight above, north up, in the part of the view that is free.
        let [top, right, bottom, left] = insets;
        let free = (viewport - Vec2::new(left + right, top + bottom)).max(0.2 * viewport);
        let metres_per_unit = 1.04 * (data_size / free).max_element();
        let distance = metres_per_unit * viewport.y / (2.0 * (0.5 * DEFAULT_FOV_Y).tan());
        // The middle of the free part lies off the middle of the view.
        let off_centre = Vec2::new(0.5 * (left - right), 0.5 * (top - bottom));
        let centre = Vec2::new(data[0], data[1]) + 0.5 * data_size;
        let pivot = centre + Vec2::new(-off_centre.x, off_centre.y) * metres_per_unit;
        OrbitView { pivot, distance, heading: 0.0, pitch: FRAC_PI_2 }
    } else {
        // An oblique view from the north-north-west across the region.
        // Around the Jungfrau that puts Lake Thun and Lake Brienz in the
        // foreground and the Bernese Alps behind them.
        OrbitView {
            pivot: Vec2::new(west, south) + Vec2::new(0.42, 0.72) * side,
            distance: 0.5 * side,
            heading: 150f32.to_radians(),
            pitch: 36f32.to_radians(),
        }
    }
}

/// One way of rendering the terrain, with the light that follows the camera
/// for it.
struct Look {
    params: ShadingParams,
    rig: LightRig,
}

/// A round of the places, one after another.
struct Tour {
    /// The place the tour is at or on its way to, once it has set off.
    current: Option<usize>,
    /// Seconds left at the current place.
    dwell: f32,
}

/// What to draw this frame.
pub struct Frame {
    pub pose: Pose,
    /// The light of each look.
    pub lights: [Vec3; 2],
    /// Terrain height below the camera.
    pub ground_height: f32,
    /// The pose to load terrain detail for, with the terrain height below
    /// it, when that is not the camera's: see
    /// [`CameraController::detail_view`].
    pub detail_for: Option<(Pose, f32)>,
    /// Whether nothing has changed since the last frame, so that drawing it
    /// again would repeat it.
    pub settled: bool,
}

pub struct Explorer {
    pub ground: Ground,
    pub controller: CameraController,
    looks: [Look; 2],
    /// Where the second look begins, as a share of the view's width, while
    /// the two are compared.
    compare: Option<f32>,
    places: Vec<Place>,
    /// Whether a view left alone starts to turn slowly.
    pub attract: bool,
    /// Seconds since anything steered the camera.
    idle: f32,
    tour: Option<Tour>,
    lights: [Vec3; 2],
    /// Set by anything that changes the picture without moving the camera.
    changed: bool,
}

impl Explorer {
    /// Opens on the whole terrain. The first look is the method at its
    /// best, the second ordinary lighting to compare it with.
    pub fn new(terrain: &Terrain, ground: Ground, aspect: f32) -> Self {
        let params = ShadingParams::showcase();
        let z_scale = params.geometry_exaggeration;
        let start = Pose { position: Vec3::ZERO, heading: 0.0, pitch: FRAC_PI_2 };
        let mut controller = CameraController::new(start, &ground, z_scale);
        controller.set_view(opening_view(&terrain.meta, aspect), &ground, z_scale);
        Self {
            ground,
            controller,
            looks: [
                Look { params, rig: LightRig::default() },
                Look { params: Condition::Realistic.params(), rig: LightRig::default() },
            ],
            compare: None,
            // The explorer has its own view of the whole terrain, which
            // fits the window it is shown in.
            places: path::places(terrain).into_iter().filter(|place| place.id != WHOLE_COUNTRY).collect(),
            attract: false,
            idle: 0.0,
            tour: None,
            lights: [Vec3::ZERO; 2],
            changed: true,
        }
    }

    pub fn places(&self) -> &[Place] {
        &self.places
    }

    /// Vertical scale the camera collides with: the first look's geometry.
    /// Vertical exaggeration of the drawn terrain.
    pub fn z_scale(&self) -> f32 {
        self.looks[0].params.geometry_exaggeration
    }

    pub fn params(&self, look: usize) -> &ShadingParams {
        &self.looks[look].params
    }

    /// The parameters of a look, to change them.
    pub fn params_mut(&mut self, look: usize) -> &mut ShadingParams {
        self.changed = true;
        &mut self.looks[look].params
    }

    pub fn compare(&self) -> Option<f32> {
        self.compare
    }

    pub fn set_compare(&mut self, split: Option<f32>) {
        self.changed |= split != self.compare;
        self.compare = split;
    }

    /// Marks the picture as changed, for instance after a resize.
    pub fn touch(&mut self) {
        self.changed = true;
    }

    /// Flies to a place by its id. False if there is no such place.
    pub fn fly_to_place(&mut self, id: &str) -> bool {
        let Some(place) = self.places.iter().find(|place| place.id == id) else { return false };
        self.fly_to(place.view);
        true
    }

    /// Flies to a view. This ends a tour, as steering does.
    pub fn fly_to(&mut self, view: OrbitView) {
        self.tour = None;
        self.idle = 0.0;
        let z_scale = self.z_scale();
        self.controller.fly_to(view, &self.ground, z_scale);
    }

    /// Jumps to a view at once.
    pub fn set_view(&mut self, view: OrbitView) {
        self.tour = None;
        self.changed = true;
        let z_scale = self.z_scale();
        self.controller.set_view(view, &self.ground, z_scale);
    }

    /// Starts or stops a round of the places. Steering stops it too.
    pub fn set_tour(&mut self, on: bool) {
        self.tour = (on && !self.places.is_empty()).then_some(Tour { current: None, dwell: 0.0 });
    }

    pub fn touring(&self) -> bool {
        self.tour.is_some()
    }

    /// The place a tour is at or on its way to.
    pub fn tour_place(&self) -> Option<&Place> {
        self.tour.as_ref()?.current.map(|index| &self.places[index])
    }

    /// Zooms by a factor, eased; above 1 zooms in.
    pub fn zoom_by(&mut self, factor: f32) {
        self.tour = None;
        self.idle = 0.0;
        self.controller.zoom_by(factor, &self.ground);
    }

    /// Advances the camera and the lights by `dt` seconds.
    pub fn frame(&mut self, input: &CameraInput, dt: f32) -> Frame {
        let z_scale = self.z_scale();
        if input.steers() || input.dragging {
            self.idle = 0.0;
            self.tour = None;
        } else {
            self.idle += dt;
        }

        if let Some(tour) = self.tour.as_mut().filter(|_| !self.controller.in_flight()) {
            tour.dwell -= dt;
            if tour.dwell <= 0.0 {
                let next = tour.current.map_or(0, |current| (current + 1) % self.places.len());
                *tour = Tour { current: Some(next), dwell: TOUR_DWELL_S };
                self.controller.fly_to(self.places[next].view, &self.ground, z_scale);
            }
        }

        // A tilted view left alone turns slowly; a map stays north-up.
        let turns = self.attract
            && self.idle > IDLE_AFTER_S
            && !self.controller.in_flight()
            && self.controller.view().pitch < MAP_PITCH;
        let goal = if turns { IDLE_SPIN } else { 0.0 };
        let step = IDLE_SPIN_RAMP * dt;
        self.controller.idle_spin += (goal - self.controller.idle_spin).clamp(-step, step);

        self.controller.update(input, dt, &self.ground, z_scale);
        let pose = self.controller.pose();
        let looks = &mut self.looks;
        let lights = [0, 1].map(|look| {
            let Look { params, rig } = &mut looks[look];
            rig.update(&pose, params, dt)
        });
        let lit = lights.iter().zip(&self.lights).all(|(now, before)| now.distance(*before) < LIGHT_SETTLED);
        self.lights = lights;

        let settled = !std::mem::take(&mut self.changed)
            && lit
            && self.tour.is_none()
            && !input.steers()
            && !self.controller.in_motion(&self.ground);
        let ground = &self.ground;
        let detail_for = self.controller.detail_view().map(|view| {
            let pose = view.pose(ground, z_scale);
            (pose, ground.height(pose.position.truncate()))
        });
        Frame { pose, lights, ground_height: ground.height(pose.position.truncate()), detail_for, settled }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    /// A cone with its summit on the Jungfrau, so that one place lies on it.
    fn explorer() -> Explorer {
        let mut dataset = relief_terrain::synthetic::cone_dataset(1024).unwrap();
        dataset.meta.origin_lv95 = [2_641_000.0 - 5120.0, 1_155_300.0 - 5120.0];
        let terrain = Terrain::from(dataset);
        Explorer::new(&terrain, Ground::new(&terrain), 1.6)
    }

    fn input() -> CameraInput {
        CameraInput { viewport: Vec2::new(1600.0, 1000.0), ..Default::default() }
    }

    /// Runs frames without input until one settles; returns how many.
    fn settle(explorer: &mut Explorer) -> usize {
        (1..4000).find(|_| explorer.frame(&input(), DT).settled).expect("the view never settles")
    }

    #[test]
    fn a_view_left_alone_settles_and_a_change_wakes_it() {
        let mut explorer = explorer();
        // The light takes a moment to arrive; then nothing moves.
        assert!(settle(&mut explorer) < 1200);
        assert!(explorer.frame(&input(), DT).settled);

        explorer.params_mut(0).color = false;
        assert!(!explorer.frame(&input(), DT).settled);
        assert!(explorer.frame(&input(), DT).settled);

        explorer.set_compare(Some(0.5));
        assert!(!explorer.frame(&input(), DT).settled);
        explorer.set_compare(Some(0.5));
        assert!(explorer.frame(&input(), DT).settled);
    }

    #[test]
    fn flying_to_a_place_arrives_at_its_view() {
        let mut explorer = explorer();
        assert!(!explorer.fly_to_place("matterhorn"));
        assert!(!explorer.fly_to_place("nowhere"));
        assert!(explorer.fly_to_place("jungfrau"));
        assert!(!explorer.frame(&input(), DT).settled);
        settle(&mut explorer);
        let (view, place) = (explorer.controller.view(), explorer.places()[0]);
        assert_eq!(place.name, "Jungfrau");
        assert!(view.pivot.distance(place.view.pivot) < 1.0 && (view.pitch - place.view.pitch).abs() < 1e-3);
    }

    #[test]
    fn an_idle_tilted_view_turns_until_it_is_steered() {
        let mut explorer = explorer();
        explorer.attract = true;
        let heading = explorer.controller.view().heading;
        for _ in 0..(60.0 * (IDLE_AFTER_S + 3.0)) as usize {
            assert!(!explorer.frame(&input(), DT).settled || explorer.idle <= IDLE_AFTER_S + DT);
        }
        assert!(explorer.controller.view().heading != heading);

        // A drag takes the view back; it stops turning by itself.
        let drag = CameraInput { dragging: true, ..input() };
        for _ in 0..120 {
            explorer.frame(&drag, DT);
        }
        assert_eq!(explorer.controller.idle_spin, 0.0);
    }

    #[test]
    fn a_tour_visits_the_places_until_steering_ends_it() {
        let mut explorer = explorer();
        explorer.set_tour(true);
        explorer.frame(&input(), DT);
        assert!(explorer.touring() && explorer.controller.in_flight());
        assert_eq!(explorer.tour_place().map(|place| place.id), Some("jungfrau"));
        explorer.frame(&CameraInput { scroll: 1.0, ..input() }, DT);
        assert!(!explorer.touring());
    }
}
