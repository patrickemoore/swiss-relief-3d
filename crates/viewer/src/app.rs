//! The interactive viewer: window, input and the frame loop.

use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use glam::{Vec2, Vec3};
use relief_terrain::Terrain;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::capture::{self, OffscreenTarget};
use crate::session::Session;
use crate::ui::{self, Performance};
use relief_render::camera::{CameraInput, CameraMode};
use relief_render::gpu::Gpu;
use relief_render::params::Condition;
use relief_render::renderer::timers::Pass;
use relief_render::renderer::trails::TrailStyle;
use relief_render::renderer::{FrameInput, Renderer, SideInput};

/// Directory frame dumps are written to.
const CAPTURE_DIR: &str = "captures";
/// Frames over which the frame-time statistics are taken.
const FRAME_HISTORY: usize = 240;
/// Frames longer than this are reported in the debug log, to find hitches.
const SLOW_FRAME_SECONDS: f32 = 0.03;
/// Longest simulated step, so a stall does not fling the camera.
const MAX_FRAME_SECONDS: f32 = 0.1;
/// How often a minimized window looks for events.
const MINIMIZED_POLL: Duration = Duration::from_millis(50);
/// Scroll distance in pixels that counts as one wheel step.
const PIXELS_PER_SCROLL_STEP: f32 = 60.0;
/// Terrain tiles taken in per frame at most. They are read and prepared on
/// other threads, so each costs the frame only a copy to the GPU.
const TILES_PER_FRAME: usize = 8;

/// Opens the window and runs until it is closed.
pub fn run(terrain: Terrain) -> Result<()> {
    let event_loop = EventLoop::new().context("creating the event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { terrain: Some(terrain), viewer: None, error: None };
    event_loop.run_app(&mut app)?;
    app.error.map_or(Ok(()), Err)
}

struct App {
    /// Held until the window exists, then handed to the renderer.
    terrain: Option<Terrain>,
    viewer: Option<Viewer>,
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Some(terrain) = self.terrain.take() else { return };
        match Viewer::new(event_loop, &terrain) {
            Ok(viewer) => self.viewer = Some(viewer),
            Err(error) => {
                self.error = Some(error);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window: WindowId, event: WindowEvent) {
        let Some(viewer) = &mut self.viewer else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Err(error) = viewer.frame() {
                    self.error = Some(error);
                    event_loop.exit();
                }
            }
            event => viewer.handle_event(&event),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(viewer) = &self.viewer {
            viewer.window.request_redraw();
        }
    }
}

/// Mouse and keyboard state gathered between frames.
#[derive(Default)]
struct Input {
    keys: HashSet<KeyCode>,
    primary_down: bool,
    secondary_down: bool,
    cursor: Option<PhysicalPosition<f64>>,
    primary_drag: Vec2,
    secondary_drag: Vec2,
    scroll: f32,
}

impl Input {
    fn axis(&self, negative: KeyCode, positive: KeyCode) -> f32 {
        self.keys.contains(&positive) as i32 as f32 - self.keys.contains(&negative) as i32 as f32
    }

    /// The camera's input for one frame; resets the accumulated motion.
    fn take(&mut self, viewport: Vec2) -> CameraInput {
        CameraInput {
            movement: Vec3::new(
                self.axis(KeyCode::KeyA, KeyCode::KeyD),
                self.axis(KeyCode::KeyS, KeyCode::KeyW),
                self.axis(KeyCode::KeyQ, KeyCode::KeyE),
            ),
            turn: Vec2::new(
                self.axis(KeyCode::ArrowLeft, KeyCode::ArrowRight),
                self.axis(KeyCode::ArrowUp, KeyCode::ArrowDown),
            ),
            primary_drag: std::mem::take(&mut self.primary_drag),
            secondary_drag: std::mem::take(&mut self.secondary_drag),
            scroll: std::mem::take(&mut self.scroll),
            pinch: 0.0,
            cursor: self.cursor.map(|cursor| Vec2::new(cursor.x as f32, cursor.y as f32)),
            dragging: self.primary_down || self.secondary_down,
            fast: self.keys.contains(&KeyCode::ShiftLeft) || self.keys.contains(&KeyCode::ShiftRight),
            viewport,
        }
    }
}

struct Viewer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    renderer: Renderer,
    session: Session,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    input: Input,
    last_frame: Instant,
    frame_seconds: VecDeque<f32>,
}

impl Viewer {
    fn new(event_loop: &ActiveEventLoop, terrain: &Terrain) -> Result<Self> {
        let attributes = Window::default_attributes()
            .with_title(format!("Swiss relief shading: {}", terrain.meta.name))
            .with_inner_size(LogicalSize::new(1600.0, 900.0));
        let window = Arc::new(event_loop.create_window(attributes).context("creating the window")?);

        let instance = Gpu::instance();
        let surface = instance.create_surface(window.clone()).context("creating the window surface")?;
        let gpu = Gpu::new(&instance, Some(&surface))?;

        let capabilities = surface.get_capabilities(&gpu.adapter);
        // The interface blends in display space and wants a target without
        // hardware sRGB encoding; the composite pass encodes for either kind.
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|format| !format.is_srgb())
            .or(capabilities.formats.first().copied())
            .context("the surface supports no texture format")?;
        let PhysicalSize { width, height } = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&gpu.device, &config);

        let renderer = Renderer::new(&gpu, terrain, format, [config.width, config.height]);
        let mut session = Session::new(terrain)?;
        session.set_viewport(Vec2::new(config.width as f32, config.height as f32));

        let egui_ctx = egui::Context::default();
        ui::style(&egui_ctx);
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(&gpu.device, format, None, 1, false);

        Ok(Self {
            window,
            surface,
            config,
            gpu,
            renderer,
            session,
            egui_ctx,
            egui_state,
            egui_renderer,
            input: Input::default(),
            last_frame: Instant::now(),
            frame_seconds: VecDeque::with_capacity(FRAME_HISTORY),
        })
    }

    fn handle_event(&mut self, event: &WindowEvent) {
        // The interface sees every event first and claims those over it.
        let consumed = self.egui_state.on_window_event(&self.window, event).consumed;
        match event {
            WindowEvent::Resized(size) => self.resize(*size),
            WindowEvent::MouseInput { state, button, .. } => {
                // A press over the panel belongs to the panel; releases
                // always count, so a drag can never get stuck.
                let down = *state == ElementState::Pressed && !consumed;
                match button {
                    MouseButton::Left => self.input.primary_down = down,
                    MouseButton::Right | MouseButton::Middle => self.input.secondary_down = down,
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(last) = self.input.cursor.replace(*position) {
                    let delta = Vec2::new((position.x - last.x) as f32, (position.y - last.y) as f32);
                    if self.input.primary_down {
                        self.input.primary_drag += delta;
                    }
                    if self.input.secondary_down {
                        self.input.secondary_drag += delta;
                    }
                }
            }
            WindowEvent::CursorLeft { .. } => self.input.cursor = None,
            WindowEvent::MouseWheel { delta, .. } if !consumed => {
                self.input.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, lines) => *lines,
                    MouseScrollDelta::PixelDelta(pixels) => pixels.y as f32 / PIXELS_PER_SCROLL_STEP,
                };
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(key), state, repeat, .. },
                ..
            } => {
                if *state == ElementState::Released {
                    self.input.keys.remove(key);
                } else if !self.egui_ctx.wants_keyboard_input() {
                    self.input.keys.insert(*key);
                    if !repeat {
                        self.shortcut(*key);
                    }
                }
            }
            WindowEvent::Focused(false) => {
                self.input.keys.clear();
                self.input.primary_down = false;
                self.input.secondary_down = false;
            }
            _ => {}
        }
    }

    fn shortcut(&mut self, key: KeyCode) {
        let session = &mut self.session;
        let conditions = [KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3, KeyCode::Digit4, KeyCode::Digit5];
        match key {
            KeyCode::F1 => session.show_panel = !session.show_panel,
            KeyCode::F12 => session.capture_requested = true,
            KeyCode::KeyC => session.split_enabled = !session.split_enabled,
            KeyCode::KeyL => session.show_labels = !session.show_labels,
            KeyCode::KeyT => session.show_trails = !session.show_trails,
            KeyCode::Escape => session.stop_playback(),
            KeyCode::Tab => {
                let mode = match session.controller.mode {
                    CameraMode::Orbit => CameraMode::Fly,
                    CameraMode::Fly => CameraMode::Orbit,
                };
                session.set_camera_mode(mode);
            }
            key => {
                if let Some(index) = conditions.iter().position(|&digit| digit == key) {
                    session.apply_condition(Condition::ALL[index]);
                }
            }
        }
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.gpu.device, &self.config);
        self.renderer.resize(&self.gpu.device, [size.width, size.height]);
        self.session.set_viewport(Vec2::new(size.width as f32, size.height as f32));
    }

    fn frame(&mut self) -> Result<()> {
        // A minimized window has no surface to draw on, and waiting for one
        // would hold up its events. Idle instead.
        if self.window.is_minimized() == Some(true) {
            thread::sleep(MINIMIZED_POLL);
            self.last_frame = Instant::now();
            return Ok(());
        }

        let now = Instant::now();
        let elapsed = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        if self.frame_seconds.len() == FRAME_HISTORY {
            self.frame_seconds.pop_front();
        }
        self.frame_seconds.push_back(elapsed);
        if elapsed > SLOW_FRAME_SECONDS {
            log::debug!(
                "slow frame: {:.0} ms, {} tiles to stream",
                1000.0 * elapsed,
                self.renderer.stats.missing_tiles
            );
        }
        let dt = elapsed.min(MAX_FRAME_SECONDS);

        self.gpu.device.poll(wgpu::Maintain::Poll);
        self.renderer.timers.collect();

        // Interface first, so this frame already renders what it changed.
        let performance = self.performance();
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let session = &mut self.session;
        let output = self.egui_ctx.run(raw_input, |ctx| ui::draw(ctx, session, &performance));
        self.egui_state.handle_platform_output(&self.window, output.platform_output);

        let present_mode =
            if self.session.vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync };
        if present_mode != self.config.present_mode {
            self.config.present_mode = present_mode;
            self.surface.configure(&self.gpu.device, &self.config);
        }

        let camera_input = self.input.take(Vec2::new(self.config.width as f32, self.config.height as f32));
        let pose = self.session.advance(&camera_input, dt);
        if std::mem::take(&mut self.session.view_jumped) {
            self.renderer.forget_history();
        }
        let lights = self.session.lights(&pose, dt);
        let [left, right] = &self.session.sides;
        let frame_input = FrameInput {
            pose,
            fov_y: self.session.controller.fov_y,
            ground_height: self.session.ground.height(pose.position.truncate()),
            detail_for: self.session.detail_for(),
            sides: [
                SideInput { params: &left.params, light: lights[0] },
                SideInput { params: &right.params, light: lights[1] },
            ],
            split: self.session.split(),
            pixels_per_quad: self.session.pixels_per_quad,
            tile_budget: Some(TILES_PER_FRAME),
            trails: self
                .session
                .show_trails
                .then(|| TrailStyle { pixel_ratio: self.window.scale_factor() as f32, ..TrailStyle::default() }),
        };

        let capture = std::mem::take(&mut self.session.capture_requested).then(|| {
            save_frame(
                &self.gpu,
                &mut self.renderer,
                self.config.format,
                [self.config.width, self.config.height],
                &frame_input,
            )
        });

        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            // The surface changed under us; reconfigure and draw next frame.
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.gpu.device, &self.config);
                return Ok(());
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(()),
            Err(error) => return Err(error).context("acquiring the next frame"),
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.renderer.render(&self.gpu, &mut encoder, &view, &frame_input)?;

        let paint_jobs = self.egui_ctx.tessellate(output.shapes, output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: output.pixels_per_point,
        };
        for (id, delta) in &output.textures_delta.set {
            self.egui_renderer.update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
        }
        self.egui_renderer.update_buffers(&self.gpu.device, &self.gpu.queue, &mut encoder, &paint_jobs, &screen);
        self.renderer.timers.mark(Pass::Ui);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("interface"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: self.renderer.timers.render_span(Pass::Ui, true, true),
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut pass, &paint_jobs, &screen);
        }
        for id in &output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.renderer.timers.resolve(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
        self.renderer.timers.after_submit();
        frame.present();

        if let Some(result) = capture {
            self.session.status = match result {
                Ok(path) => format!("Saved {}", path.display()),
                Err(error) => format!("{error:#}"),
            };
        }
        Ok(())
    }

    fn performance(&self) -> Performance {
        let frames = self.frame_seconds.len().max(1) as f32;
        let timers = &self.renderer.timers;
        Performance {
            adapter: self.gpu.adapter_name(),
            resolution: [self.config.width, self.config.height],
            mean_frame_ms: 1000.0 * self.frame_seconds.iter().sum::<f32>() / frames,
            max_frame_ms: 1000.0 * self.frame_seconds.iter().copied().fold(0.0, f32::max),
            pass_ms: timers.supported().then(|| Pass::ALL.map(|pass| timers.smoothed_ms(pass))),
            stats: self.renderer.stats,
        }
    }
}

/// Renders the frame once more into an offscreen target and writes it to
/// the first unused `captures/relief_NNNN.png`, without the interface.
fn save_frame(
    gpu: &Gpu,
    renderer: &mut Renderer,
    format: wgpu::TextureFormat,
    size: [u32; 2],
    frame_input: &FrameInput,
) -> Result<PathBuf> {
    let target = OffscreenTarget::new(&gpu.device, format, size);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    renderer.render(gpu, &mut encoder, &target.view, frame_input)?;
    gpu.queue.submit([encoder.finish()]);

    let path = (1..)
        .map(|index| PathBuf::from(CAPTURE_DIR).join(format!("relief_{index:04}.png")))
        .find(|path| !path.exists())
        .expect("an unbounded range always yields an unused name");
    capture::save_png(&path, size, &capture::read_rgba(gpu, &target.texture)?)?;
    Ok(path)
}
