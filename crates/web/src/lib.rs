//! The relief viewer in a browser.
//!
//! This is the renderer of `relief-viewer` behind a small JavaScript API:
//! a page hands over a canvas and the address of a packed terrain, forwards
//! pointer and keyboard input, and calls [`Viewer::frame`] once per display
//! frame. Everything else about the page, its controls included, is the
//! page's own business.
//!
//! It needs WebGPU. The page finds out by trying: [`Viewer::create`] rejects
//! with a message that says what is missing.

#![cfg(target_arch = "wasm32")]

mod feed;
mod trail_feed;

use std::sync::Arc;

use anyhow::{Context, Result};
use glam::Vec2;
use relief_render::camera::{CameraInput, Ground, OrbitView, GROUND_LEVEL};
use relief_render::explore::{opening_view_within, Explorer};
use relief_render::gpu::Gpu;
use relief_render::labels::{Labels, View as LabelView};
use relief_render::params::{Condition, ShadingParams};
use relief_render::renderer::trails::{TrailFeed, TrailStyle};
use relief_render::renderer::{FrameInput, Renderer, SideInput, DEFAULT_PIXELS_PER_QUAD};
use relief_terrain::pack::{self, NoTiles, TileIndex};
use relief_terrain::{lv95, Meta};
use wasm_bindgen::prelude::*;

use crate::feed::{FeedMonitor, FetchFeed};
use crate::trail_feed::FetchTrailFeed;

/// Tiles unpacked and taken in per frame at most. Unpacking one takes a
/// couple of milliseconds on the page's only thread.
const TILES_PER_FRAME: usize = 3;
/// Tiles along a side of each streamed level that are held around the
/// camera. Every tile is a download and every window is GPU memory, so it is
/// less than the eight of the desktop: what the canvas can show at once
/// still fits. A canvas up to this many CSS pixels on its longer side, a
/// phone or a tablet held upright, gets the small windows and skips the
/// finest level: its cells are smaller than such a screen's pixels at any
/// distance the places use, so leaving them out halves the downloads and
/// the GPU memory without changing the picture.
const REACH_TILES: [usize; 2] = [4, 6];
const SMALL_CANVAS_PIXELS: u32 = 1024;
/// The finest pyramid level a small canvas loads: 20 m cells.
const PHONE_FINEST_LEVEL: usize = 1;
/// Frames drawn after the picture has stopped changing, so that the last
/// change is certainly on screen.
const FRAMES_AFTER_SETTLING: u32 = 3;
/// Longest step of simulated time, so a stalled tab does not fling the camera.
const MAX_FRAME_SECONDS: f32 = 0.1;
/// The pitch of the tilted view the tilt control switches to, in radians.
const TILTED_PITCH: f32 = 0.7;
/// Views this close to straight down count as a map.
const MAP_PITCH: f32 = 1.45;

fn to_js(error: anyhow::Error) -> JsError {
    JsError::new(&format!("{error:#}"))
}

/// Sends log messages to the browser's console.
struct ConsoleLog;

impl log::Log for ConsoleLog {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let message = JsValue::from_str(&format!("[relief] {}", record.args()));
        match record.level() {
            log::Level::Error => web_sys::console::error_1(&message),
            log::Level::Warn => web_sys::console::warn_1(&message),
            _ => web_sys::console::log_1(&message),
        }
    }

    fn flush(&self) {}
}

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
    // Only the first of several viewers on a page gets to install it.
    if log::set_logger(&ConsoleLog).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
}

/// Pointer and keyboard input gathered between frames.
#[derive(Default)]
struct Input {
    movement: Vec2,
    turn: Vec2,
    fast: bool,
    primary_drag: Vec2,
    secondary_drag: Vec2,
    scroll: f32,
    pinch: f32,
    cursor: Option<Vec2>,
    dragging: bool,
}

impl Input {
    /// The camera's input for one frame; resets the accumulated motion.
    fn take(&mut self, viewport: Vec2) -> CameraInput {
        CameraInput {
            movement: self.movement.extend(0.0),
            turn: self.turn,
            primary_drag: std::mem::take(&mut self.primary_drag),
            secondary_drag: std::mem::take(&mut self.secondary_drag),
            scroll: std::mem::take(&mut self.scroll),
            pinch: std::mem::take(&mut self.pinch),
            cursor: self.cursor,
            dragging: self.dragging,
            fast: self.fast,
            viewport,
        }
    }
}

/// A terrain viewer drawing into a canvas.
#[wasm_bindgen]
pub struct Viewer {
    gpu: Gpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    explorer: Explorer,
    monitor: FeedMonitor,
    meta: Meta,
    input: Input,
    /// The place the camera was last sent to, until it is steered away.
    place: Option<&'static str>,
    /// How much of the canvas the page's controls cover at the top, right,
    /// bottom and left, in device pixels.
    insets: [f32; 4],
    /// Whether the camera is still on the view of the whole terrain it was
    /// last sent to. That view then follows the canvas when it is resized.
    at_home: bool,
    last_time: Option<f64>,
    settled_frames: u32,
    /// Device pixels per CSS pixel, which sets the size of the names.
    pixel_ratio: f32,
    labels: Labels,
    /// Whether the terrain came with hiking trails, and whether to draw them.
    has_trails: bool,
    trails_on: bool,
}

#[wasm_bindgen]
impl Viewer {
    /// Opens a viewer on `canvas`, whose width and height are its size in
    /// device pixels; `layout_side` is its longer side in CSS pixels, which
    /// tells a phone from a desktop. `meta` and `base` are the contents of the packed
    /// terrain's `meta.json` and `base.bin`, and `data_url` is where its
    /// tiles are served from. `trails_index` is the contents of
    /// `trails/index.bin` where the terrain has hiking trails.
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
        data_url: String,
        meta: String,
        base: Vec<u8>,
        layout_side: u32,
        trails_index: Option<Vec<u8>>,
    ) -> Result<Viewer, JsError> {
        Self::open(canvas, &data_url, &meta, &base, layout_side, trails_index.as_deref()).await.map_err(to_js)
    }

    async fn open(
        canvas: web_sys::HtmlCanvasElement,
        data_url: &str,
        meta: &str,
        base: &[u8],
        layout_side: u32,
        trails_index: Option<&[u8]>,
    ) -> Result<Self> {
        let size = [canvas.width().max(1), canvas.height().max(1)];
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .context("this browser does not offer WebGPU on a canvas")?;
        let gpu = Gpu::open(&instance, Some(&surface)).await.context("WebGPU is not available")?;

        let capabilities = surface.get_capabilities(&gpu.adapter);
        // The composite pass encodes for the display itself.
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|format| !format.is_srgb())
            .or(capabilities.formats.first().copied())
            .context("the canvas supports no texture format")?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size[0],
            height: size[1],
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&gpu.device, &config);

        let meta: Meta = serde_json::from_str(meta).context("parsing the terrain's description")?;
        let mut terrain = pack::unpack(meta, base, Arc::new(NoTiles)).context("unpacking the terrain")?;
        let small = layout_side <= SMALL_CANVAS_PIXELS;
        if let Some(packed) = terrain.meta.packed.as_mut().filter(|_| small) {
            packed.finest_level = packed.finest_level.max(PHONE_FINEST_LEVEL);
        }
        // The camera stands on tiles a level coarser than on the desktop:
        // they reach twice as far, and they are on their way anyway.
        let ground_level = terrain.finest_level().max(GROUND_LEVEL + 1);
        let feed = FetchFeed::new(&terrain, data_url, ground_level)?;
        let monitor = feed.monitor();
        let ground = Ground::with_tiles(&terrain, ground_level, feed.ground());

        let reach = REACH_TILES[(!small) as usize];
        // Trails come with the terrain, if the page fetched their index.
        let trails = match (trails_index, terrain.meta.trails.is_some()) {
            (Some(bytes), true) => {
                let index = TileIndex::from_bytes(bytes).context("reading the trail index")?;
                let feed: Box<dyn TrailFeed> = Box::new(FetchTrailFeed::new(data_url, terrain.meta.cell_size));
                Some((index, feed))
            }
            _ => None,
        };
        let has_trails = trails.is_some();
        let renderer = Renderer::with_feed(&gpu, &terrain, Box::new(feed), reach, trails, format, size);
        let explorer = Explorer::new(&terrain, ground, size[0] as f32 / size[1] as f32);
        let labels = Labels::builtin(terrain.meta.origin_lv95, terrain.meta.data_region());
        Ok(Self {
            gpu,
            surface,
            config,
            renderer,
            explorer,
            monitor,
            meta: terrain.meta.clone(),
            input: Input::default(),
            place: None,
            insets: [0.0; 4],
            at_home: false,
            last_time: None,
            settled_frames: 0,
            pixel_ratio: 1.0,
            labels,
            has_trails,
            trails_on: true,
        })
    }

    /// Call when the page's device pixels per CSS pixel change: the names
    /// keep their size on the page.
    pub fn set_pixel_ratio(&mut self, ratio: f32) {
        self.pixel_ratio = ratio.max(0.25);
    }

    /// Whether the terrain came with hiking trails.
    pub fn has_trails(&self) -> bool {
        self.has_trails
    }

    /// Shows or hides the hiking trails.
    pub fn set_trails(&mut self, on: bool) {
        if self.trails_on != on {
            self.trails_on = on;
            self.explorer.touch();
            self.settled_frames = 0;
        }
    }

    /// The names placed over the last frame, as JSON: see `labels::Placed`.
    pub fn labels(&self) -> String {
        serde_json::to_string(self.labels.placed()).unwrap_or_default()
    }

    /// Call when the canvas has been given a new size in device pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == (self.config.width, self.config.height) {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
        self.renderer.resize(&self.gpu.device, [width, height]);
        self.explorer.touch();
        if self.at_home {
            self.go_home(false);
        }
    }

    /// Advances to `time`, in milliseconds as an animation frame gives it,
    /// and draws if anything has changed. Returns whether it drew.
    pub fn frame(&mut self, time: f64) -> Result<bool, JsError> {
        let elapsed = self.last_time.replace(time).map_or(1.0 / 60.0, |last| ((time - last) / 1000.0) as f32);
        let dt = elapsed.clamp(1e-3, MAX_FRAME_SECONDS);

        let viewport = Vec2::new(self.config.width as f32, self.config.height as f32);
        let input = self.input.take(viewport);
        if input.steers() {
            self.place = None;
            self.at_home = false;
        }
        let frame = self.explorer.frame(&input, dt);
        self.labels.update(
            &LabelView {
                pose: frame.pose,
                fov_y: self.explorer.controller.fov_y,
                viewport,
                pixel_ratio: self.pixel_ratio,
                z_scale: self.explorer.z_scale(),
                ground: &self.explorer.ground,
                insets: self.insets,
            },
            dt,
        );
        if frame.settled && !self.monitor.has_arrivals() && !self.labels.animating() {
            self.settled_frames += 1;
            if self.settled_frames > FRAMES_AFTER_SETTLING {
                return Ok(false);
            }
        } else {
            self.settled_frames = 0;
        }

        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            // The canvas changed under us; reconfigure and draw next frame.
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.gpu.device, &self.config);
                self.settled_frames = 0;
                return Ok(false);
            }
            Err(error) => return Err(to_js(anyhow::Error::new(error).context("acquiring the canvas"))),
        };
        let view = output.texture.create_view(&Default::default());
        let frame_input = FrameInput {
            pose: frame.pose,
            fov_y: self.explorer.controller.fov_y,
            ground_height: frame.ground_height,
            detail_for: frame.detail_for,
            sides: [
                SideInput { params: self.explorer.params(0), light: frame.lights[0] },
                SideInput { params: self.explorer.params(1), light: frame.lights[1] },
            ],
            split: self.explorer.compare(),
            pixels_per_quad: DEFAULT_PIXELS_PER_QUAD,
            tile_budget: Some(TILES_PER_FRAME),
            trails: (self.has_trails && self.trails_on)
                .then(|| TrailStyle { pixel_ratio: self.pixel_ratio, ..TrailStyle::default() }),
        };
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.renderer.render(&self.gpu, &mut encoder, &view, &frame_input).map_err(to_js)?;
        self.gpu.queue.submit([encoder.finish()]);
        output.present();

        if self.renderer.take_streamed() {
            self.settled_frames = 0;
        }
        Ok(true)
    }

    // --- Input -------------------------------------------------------------

    /// Where the pointer is, in device pixels from the canvas's top-left
    /// corner. Zooming holds the ground there in place.
    pub fn pointer_at(&mut self, x: f32, y: f32) {
        self.input.cursor = Some(Vec2::new(x, y));
    }

    pub fn pointer_left(&mut self) {
        self.input.cursor = None;
    }

    /// A button or a finger went down or came up. Coasting starts when the
    /// last one lifts.
    pub fn set_dragging(&mut self, dragging: bool) {
        self.input.dragging = dragging;
    }

    /// Pointer motion in device pixels that moves the ground, or with
    /// `rotate` turns and tilts the view.
    pub fn drag(&mut self, dx: f32, dy: f32, rotate: bool) {
        let drag = if rotate { &mut self.input.secondary_drag } else { &mut self.input.primary_drag };
        *drag += Vec2::new(dx, dy);
    }

    /// Wheel steps; positive zooms in.
    pub fn wheel(&mut self, steps: f32) {
        self.input.scroll += steps;
    }

    /// A pinch by `factor` since the last call; above 1 zooms in.
    pub fn pinch(&mut self, factor: f32) {
        if factor > 0.0 {
            self.input.pinch += factor.ln();
        }
    }

    /// The keyboard's intent, each in -1..1: move right and forward, turn
    /// clockwise and tilt down.
    pub fn set_keys(&mut self, right: f32, forward: f32, turn: f32, tilt: f32, fast: bool) {
        self.input.movement = Vec2::new(right, forward);
        self.input.turn = Vec2::new(turn, tilt);
        self.input.fast = fast;
    }

    // --- Going places ------------------------------------------------------

    /// The places to fly to, as JSON: `[{ "id": ..., "name": ... }]`.
    pub fn places(&self) -> String {
        let places: Vec<_> = self
            .explorer
            .places()
            .iter()
            .map(|place| serde_json::json!({ "id": place.id, "name": place.name }))
            .collect();
        serde_json::Value::from(places).to_string()
    }

    /// Flies to a place by its id. False if there is no such place.
    pub fn fly_to(&mut self, id: &str) -> bool {
        let found = self.explorer.fly_to_place(id);
        if found {
            self.at_home = false;
            self.place = self.explorer.places().iter().find(|place| place.id == id).map(|place| place.id);
        }
        found
    }

    /// Tells the viewer how much of the canvas the page's controls cover at
    /// each edge, in device pixels. The view of the whole terrain keeps
    /// clear of them.
    pub fn set_insets(&mut self, top: f32, right: f32, bottom: f32, left: f32) {
        self.insets = [top, right, bottom, left];
    }

    /// Goes back to the view of the whole terrain: in a flight, or at once.
    pub fn go_home(&mut self, fly: bool) {
        let viewport = Vec2::new(self.config.width as f32, self.config.height as f32);
        let home = opening_view_within(&self.meta, viewport, self.insets);
        self.place = None;
        self.at_home = true;
        if fly {
            self.explorer.fly_to(home);
        } else {
            self.explorer.set_view(home);
        }
    }

    /// Zooms by a factor, eased; above 1 zooms in.
    pub fn zoom_by(&mut self, factor: f32) {
        self.at_home = false;
        self.explorer.zoom_by(factor);
    }

    /// Turns the view so that north is up.
    pub fn face_north(&mut self) {
        let view = self.explorer.controller.view();
        self.at_home = false;
        self.explorer.fly_to(OrbitView { heading: 0.0, ..view });
    }

    /// Switches between looking straight down and a tilted view.
    pub fn toggle_tilt(&mut self) {
        let view = self.explorer.controller.view();
        self.at_home = false;
        let pitch = if view.pitch > MAP_PITCH { TILTED_PITCH } else { std::f32::consts::FRAC_PI_2 };
        self.explorer.fly_to(OrbitView { pitch, ..view });
    }

    /// Starts or stops a round of the places. Steering stops it too.
    pub fn set_tour(&mut self, on: bool) {
        self.place = None;
        self.at_home &= !on;
        self.explorer.set_tour(on);
    }

    /// Whether a view left alone starts to turn slowly.
    pub fn set_attract(&mut self, on: bool) {
        self.explorer.attract = on;
    }

    // --- Looks -------------------------------------------------------------

    /// Shows the second look on the right from this share of the width, or
    /// only the first with `undefined`.
    pub fn set_compare(&mut self, split: Option<f32>) {
        self.explorer.set_compare(split.map(|split| split.clamp(0.0, 1.0)));
    }

    /// Sets a look (0 or 1): `relief` is the method as the viewer shows it,
    /// and `plan`, `realistic`, `draped`, `world-fixed` and `headlight` are
    /// the conditions of the evaluation.
    pub fn set_look(&mut self, look: usize, name: &str) -> Result<(), JsError> {
        let params = match name {
            "relief" => ShadingParams::showcase(),
            "plan" => Condition::Full.params(),
            "realistic" => Condition::Realistic.params(),
            "draped" => Condition::Draped.params(),
            "world-fixed" => Condition::WorldFixed.params(),
            "headlight" => Condition::Headlight.params(),
            _ => return Err(JsError::new(&format!("there is no look called '{name}'"))),
        };
        *self.look_mut(look)? = params;
        Ok(())
    }

    /// Changes some of a look's parameters, given as a JSON object with the
    /// names the presets use, for instance `{"color": false}`.
    pub fn patch_look(&mut self, look: usize, patch: &str) -> Result<(), JsError> {
        let patched = patched(self.look_mut(look)?, patch).map_err(to_js)?;
        *self.look_mut(look)? = patched;
        Ok(())
    }

    fn look_mut(&mut self, look: usize) -> Result<&mut ShadingParams, JsError> {
        if look > 1 {
            return Err(JsError::new("a look is 0 or 1"));
        }
        Ok(self.explorer.params_mut(look))
    }

    // --- What the page shows about the view --------------------------------

    /// The state of the view as JSON, for a compass, a scale bar and such.
    pub fn state(&self) -> String {
        let view = self.explorer.controller.view();
        let [east, north] = [0, 1].map(|axis| self.meta.origin_lv95[axis] + view.pivot[axis] as f64);
        let (longitude, latitude) = lv95::to_wgs84(east, north);
        serde_json::json!({
            "heading": view.heading,
            "pitch": view.pitch,
            "distance": view.distance,
            "metresPerPixel": self.explorer.controller.metres_per_pixel(self.config.height as f32),
            "longitude": longitude,
            "latitude": latitude,
            "elevation": self.explorer.ground.height(view.pivot),
            "place": self.explorer.tour_place().map(|place| place.id).or(self.place),
            "flying": self.explorer.controller.in_flight(),
            "touring": self.explorer.touring(),
            "loading": self.renderer.stats.missing_tiles,
        })
        .to_string()
    }

    /// Megabytes of tiles fetched so far.
    pub fn fetched_megabytes(&self) -> f64 {
        self.monitor.fetched_bytes() as f64 / 1_048_576.0
    }

    /// The graphics adapter in use, for diagnostics.
    pub fn adapter(&self) -> String {
        self.gpu.adapter_name()
    }
}

/// `params` with the members of the JSON object `patch` replaced.
fn patched(params: &ShadingParams, patch: &str) -> Result<ShadingParams> {
    let mut value = serde_json::to_value(params)?;
    let patch: serde_json::Value = serde_json::from_str(patch).context("the patch is not JSON")?;
    let (Some(members), Some(changes)) = (value.as_object_mut(), patch.as_object()) else {
        anyhow::bail!("the patch is not a JSON object");
    };
    for (name, change) in changes {
        anyhow::ensure!(members.contains_key(name), "a look has no parameter called '{name}'");
        members.insert(name.clone(), change.clone());
    }
    serde_json::from_value(value).context("the patch does not fit the parameters")
}
