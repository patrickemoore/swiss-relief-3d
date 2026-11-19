//! The viewer's state between frames: what each side of the screen shows,
//! where the camera is, and what it is doing.
//!
//! Nothing here touches the GPU or the window, so the interface can edit it
//! freely and the frame loop only reads the result.

use std::f32::consts::FRAC_PI_2;
use std::path::{Path, PathBuf};

use anyhow::Result;
use glam::{Vec2, Vec3};
use relief_terrain::{Meta, Terrain};

use relief_render::camera::{CameraController, CameraInput, CameraMode, Ground, OrbitView, Pose, DEFAULT_FOV_Y};
use relief_render::explore;
use relief_render::labels::Labels;
use relief_render::light_rig::LightRig;
use relief_render::params::{self, Condition, ShadingParams, PRESET_DIR};
use relief_render::path::{self, CameraPath, Place, Recorder, Scene, Viewpoint, PATH_DIR};
use relief_render::renderer::DEFAULT_PIXELS_PER_QUAD;

/// Width over height of the window before its size is known.
const START_ASPECT: f32 = 16.0 / 9.0;
/// Height of the bar at the bottom of the view, in pixels of a window 1000
/// pixels high. The view of the whole terrain keeps clear of it.
const BAR_SHARE: f32 = 0.09;

/// One side of the split screen: its parameters and its own lagged light.
pub struct SideState {
    pub params: ShadingParams,
    pub rig: LightRig,
}

/// A camera path being played back.
pub struct Playback {
    pub path: CameraPath,
    pub time: f32,
}

pub struct Session {
    /// Side 0 fills the screen, or its left part when the split is enabled.
    pub sides: [SideState; 2],
    pub split_enabled: bool,
    /// Where side 1 begins, as a fraction of the window width.
    pub split_position: f32,
    /// The side the parameter panel edits.
    pub edited_side: usize,

    pub controller: CameraController,
    pub ground: Ground,
    /// Target size of a mesh quad on screen, in pixels.
    pub pixels_per_quad: f32,

    /// The fixed evaluation viewpoints.
    pub viewpoints: Vec<Viewpoint>,
    /// Named places on the terrain, if it lies in Switzerland.
    pub places: Vec<Place>,
    /// The place the camera was last sent to, until it is steered away, and
    /// whether it was sent to the view of the whole terrain instead.
    pub place: Option<&'static str>,
    pub at_home: bool,
    /// The view of the whole terrain that the viewer opens on.
    home: OrbitView,
    meta: Meta,
    pub flights: Vec<CameraPath>,
    pub playback: Option<Playback>,
    pub recorder: Option<Recorder>,
    /// The most recent recording, until it is saved or replaced.
    pub recording: Option<CameraPath>,
    pub saved_paths: Vec<PathBuf>,
    pub path_name: String,

    pub presets: Vec<PathBuf>,
    pub preset_name: String,

    /// Outcome of the last action, shown at the bottom of the panel.
    pub status: String,
    pub capture_requested: bool,
    /// Set when the camera jumped to an unrelated view; the frame loop then
    /// lets the renderer forget what it remembers of the old one.
    pub view_jumped: bool,
    pub vsync: bool,
    pub show_panel: bool,
    /// Names of peaks, lakes and towns, laid out over the view each frame.
    pub labels: Labels,
    pub show_labels: bool,
    /// Hiking trails, where the dataset has them.
    pub show_trails: bool,
}

impl Session {
    pub fn new(terrain: &Terrain) -> Result<Self> {
        let ground = Ground::new(terrain);
        // The window opens on the method at its best; the conditions of the
        // evaluation are a key away.
        let params = ShadingParams::showcase();
        let z_scale = params.geometry_exaggeration;
        let scene = Scene::new(terrain, &ground, z_scale, DEFAULT_FOV_Y)?;
        let (viewpoints, flights) = (scene.viewpoints(), scene.flights());
        let places = path::places(terrain);

        let start = Pose { position: Vec3::ZERO, heading: 0.0, pitch: FRAC_PI_2 };
        let mut controller = CameraController::new(start, &ground, z_scale);
        let home = explore::opening_view(&terrain.meta, START_ASPECT);
        controller.set_view(home, &ground, z_scale);

        Ok(Self {
            sides: [
                SideState { params, rig: LightRig::default() },
                SideState { params: Condition::WorldFixed.params(), rig: LightRig::default() },
            ],
            split_enabled: false,
            split_position: 0.5,
            edited_side: 0,
            controller,
            ground,
            pixels_per_quad: DEFAULT_PIXELS_PER_QUAD,
            viewpoints,
            places,
            place: None,
            at_home: true,
            home,
            meta: terrain.meta.clone(),
            flights,
            playback: None,
            recorder: None,
            recording: None,
            saved_paths: path::list_saved_paths(),
            path_name: "my_flight".to_owned(),
            presets: params::list_presets(),
            preset_name: "my_preset".to_owned(),
            status: String::new(),
            capture_requested: false,
            view_jumped: false,
            vsync: true,
            show_panel: false,
            labels: Labels::builtin(terrain.meta.origin_lv95, terrain.meta.data_region()),
            show_labels: true,
            show_trails: true,
        })
    }

    /// Vertical scale the camera collides with: the left side's geometry.
    pub fn z_scale(&self) -> f32 {
        self.sides[0].params.geometry_exaggeration
    }

    /// The parameters the panel is editing.
    pub fn edited_params(&mut self) -> &mut ShadingParams {
        &mut self.sides[self.edited_side].params
    }

    /// Split position for the renderer, or `None` when side 0 fills the screen.
    pub fn split(&self) -> Option<f32> {
        self.split_enabled.then_some(self.split_position)
    }

    pub fn apply_condition(&mut self, condition: Condition) {
        *self.edited_params() = condition.params();
        self.status = format!("Condition {}", condition.label());
    }

    /// Moves the camera to a pose at once, without the light trailing in
    /// from the old view.
    pub fn jump_to(&mut self, pose: Pose) {
        self.playback = None;
        (self.place, self.at_home) = (None, false);
        let z_scale = self.z_scale();
        self.controller.set_pose(pose, &self.ground, z_scale);
        self.sides.iter_mut().for_each(|side| side.rig.reset());
        self.view_jumped = true;
    }

    /// Flies the camera to a view of the ground; the light follows it there.
    pub fn fly_to(&mut self, view: OrbitView) {
        self.playback = None;
        (self.place, self.at_home) = (None, false);
        let z_scale = self.z_scale();
        self.controller.fly_to(view, &self.ground, z_scale);
    }

    /// Tells the session the size of the 3D view. The view of the whole
    /// terrain is fitted to it, and followed if the camera is still there.
    pub fn set_viewport(&mut self, viewport: Vec2) {
        self.home = explore::opening_view_within(&self.meta, viewport, [0.0, 0.0, BAR_SHARE * viewport.y, 0.0]);
        if self.at_home && self.playback.is_none() {
            let z_scale = self.z_scale();
            self.controller.set_view(self.home, &self.ground, z_scale);
        }
    }

    /// Flies to one of the named places.
    pub fn fly_to_place(&mut self, index: usize) {
        let place = self.places[index];
        self.fly_to(place.view);
        self.place = Some(place.id);
    }

    /// Flies back to the view of the whole terrain.
    pub fn fly_home(&mut self) {
        self.fly_to(self.home);
        self.at_home = true;
    }

    /// The pose to stream terrain detail for while a flight is under way,
    /// with the terrain height below it.
    pub fn detail_for(&self) -> Option<(Pose, f32)> {
        let pose = self.controller.detail_view()?.pose(&self.ground, self.z_scale());
        Some((pose, self.ground.height(pose.position.truncate())))
    }

    pub fn set_camera_mode(&mut self, mode: CameraMode) {
        let z_scale = self.z_scale();
        self.controller.set_mode(mode, &self.ground, z_scale);
    }

    pub fn play(&mut self, path: CameraPath) {
        if path.keyframes.is_empty() {
            self.status = format!("Path '{}' is empty", path.name);
            return;
        }
        self.jump_to(path.sample(0.0));
        self.status = format!("Playing '{}' ({:.0} s)", path.name, path.duration());
        self.playback = Some(Playback { path, time: 0.0 });
    }

    pub fn play_file(&mut self, file: &Path) {
        match CameraPath::load(file) {
            Ok(path) => self.play(path),
            Err(error) => self.status = format!("{error:#}"),
        }
    }

    /// Leaves the camera where the playback stopped.
    pub fn stop_playback(&mut self) {
        if let Some(playback) = self.playback.take() {
            let pose = playback.path.sample(playback.time);
            let z_scale = self.z_scale();
            self.controller.set_pose(pose, &self.ground, z_scale);
        }
    }

    pub fn toggle_recording(&mut self) {
        match self.recorder.take() {
            Some(recorder) => {
                let path = recorder.finish();
                self.status = format!("Recorded {:.1} s; save it to keep it", path.duration());
                self.recording = Some(path);
            }
            None => {
                self.recorder = Some(Recorder::new(&self.path_name, self.controller.pose()));
                self.status = "Recording the camera".to_owned();
            }
        }
    }

    pub fn save_recording(&mut self) {
        let Some(mut path) = self.recording.take() else { return };
        path.name = sanitized(&self.path_name);
        let file = Path::new(PATH_DIR).join(format!("{}.json", path.name));
        self.status = match path.save(&file) {
            Ok(()) => format!("Saved {}", file.display()),
            Err(error) => format!("{error:#}"),
        };
        self.saved_paths = path::list_saved_paths();
    }

    pub fn save_preset(&mut self) {
        let file = Path::new(PRESET_DIR).join(format!("{}.json", sanitized(&self.preset_name)));
        self.status = match self.sides[self.edited_side].params.save(&file) {
            Ok(()) => format!("Saved {}", file.display()),
            Err(error) => format!("{error:#}"),
        };
        self.presets = params::list_presets();
    }

    pub fn load_preset(&mut self, file: &Path) {
        match ShadingParams::load(file) {
            Ok(params) => {
                *self.edited_params() = params;
                self.status = format!("Loaded {}", file.display());
            }
            Err(error) => self.status = format!("{error:#}"),
        }
    }

    /// Advances the camera by `dt` seconds and returns the pose to render.
    pub fn advance(&mut self, input: &CameraInput, dt: f32) -> Pose {
        let z_scale = self.z_scale();
        let pose = match &mut self.playback {
            Some(playback) => {
                playback.time += dt;
                let pose = playback.path.sample(playback.time);
                if playback.time >= playback.path.duration() {
                    self.stop_playback();
                }
                pose
            }
            None => {
                if input.steers() {
                    (self.place, self.at_home) = (None, false);
                }
                self.controller.update(input, dt, &self.ground, z_scale);
                self.controller.pose()
            }
        };
        if let Some(recorder) = &mut self.recorder {
            recorder.advance(pose, dt);
        }
        pose
    }

    /// Advances both light rigs and returns the lights for the frame.
    pub fn lights(&mut self, pose: &Pose, dt: f32) -> [Vec3; 2] {
        let sides = &mut self.sides;
        [0, 1].map(|side| {
            let SideState { params, rig } = &mut sides[side];
            rig.update(pose, params, dt)
        })
    }
}

/// A file-system-safe version of a user-supplied name.
fn sanitized(name: &str) -> String {
    let cleaned: String =
        name.trim().chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    if cleaned.is_empty() {
        "untitled".to_owned()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitized_names_are_safe_file_stems() {
        assert_eq!(sanitized("  valley run #2 "), "valley_run__2");
        assert_eq!(sanitized("../../etc"), "______etc");
        assert_eq!(sanitized(""), "untitled");
    }
}
