//! The per-frame stage: everything that depends on the camera.
//!
//! Each frame the local light pass runs first, then the terrain is drawn
//! with the shading model, then the post pass adds screen-space depth cues
//! (plan section 3). The renderer draws into any target view, so the window,
//! frame dumps and the headless evaluation share one code path.
//!
//! A split screen shows two parameter sets side by side. Each side owns its
//! parameters, its light and its light field; the camera and mesh are shared.
//!
//! The terrain may be far larger than what a frame needs. Its finest levels
//! are streamed around the camera by the [`clipmap`]; everything else is
//! uploaded once.

pub mod cdlod;
pub mod clipmap;
mod local_light;
mod post;
mod terrain;
mod textures;
pub mod timers;
mod util;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use relief_terrain::terrain::TILED_LEVELS;
use relief_terrain::{Meta, Terrain};

use self::cdlod::{Frustum, LodView, Quadtree, Selection, MORPH_START, PATCH_QUADS};
use self::clipmap::{Clipmap, TileFeed};
use self::local_light::{
    FieldPlacement, LocalLightField, LocalLightPipelines, LocalLightUniform, TimedSpan, FIELD_SIZE,
};
use self::post::{CompositeSide, PostFrame, PostPass};
use self::terrain::{ShaderConfig, ShadingUniform, TerrainPass, COLOR_FORMAT, DEPTH_FORMAT, METRICS_FORMAT};
use self::textures::WholeTextures;
use self::timers::{GpuTimers, Pass};
use crate::camera::Pose;
use crate::gpu::Gpu;
use crate::params::{ShadingMode, ShadingParams};

/// Default size of a mesh quad on screen, in pixels.
pub const DEFAULT_PIXELS_PER_QUAD: f32 = 5.0;
/// Log depth written where no terrain is drawn: the sky is 2000 km away,
/// behind a whole country seen from above.
const SKY_LOG_DEPTH: f64 = 14.5;
/// How far below the lowest terrain the border skirts reach, in metres.
const SKIRT_DEPTH_M: f32 = 400.0;

/// The finest pyramid level a camera `clearance` metres above the ground
/// can show. A pixel on the nearest ground covers some number of cells, and
/// the shading samples the level `bias` above the one whose cells are that
/// size, blended with the next. Half a level is spared, so that a finer
/// level is there before it is needed.
pub fn finest_visible_level(clearance: f32, pixel_angle: f32, cell: f32, bias: f32) -> usize {
    let cells_per_pixel = pixel_angle * clearance.max(1.0) / cell;
    (cells_per_pixel.log2() + bias - 0.5).floor().max(0.0) as usize
}

/// How far above the scale of a pixel the finest level lies that a look
/// samples.
fn generalization_bias(params: &ShadingParams) -> f32 {
    if params.mode == ShadingMode::Swiss && params.generalization {
        params.generalization_bias.max(0.0)
    } else {
        0.0
    }
}

/// Mirrors `Frame` in frame.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    cam_forward: [f32; 4],
    viewport: [f32; 4],
    terrain: [f32; 4],
    elevation: [f32; 4],
    lod: [f32; 4],
    light_field: [f32; 4],
    loaded: [[f32; 4]; TILED_LEVELS],
}

/// The offscreen targets of one frame.
pub struct Targets {
    pub size: [u32; 2],
    /// Linear colour; alpha marks terrain.
    pub color: wgpu::TextureView,
    /// Natural log of the distance to the camera.
    pub log_depth: wgpu::TextureView,
    pub log_depth_texture: wgpu::Texture,
    /// Grey value, steep-slope flag, flat-ground grey and terrain mask.
    pub metrics_texture: wgpu::Texture,
    metrics: wgpu::TextureView,
    depth: wgpu::TextureView,
    /// Half-resolution targets of the separable log-depth blur.
    pub blur: [wgpu::TextureView; 2],
}

impl Targets {
    fn new(device: &wgpu::Device, size: [u32; 2], log_depth_format: wgpu::TextureFormat) -> Self {
        let size = size.map(|side| side.max(1));
        let texture = |label: &str, format, [width, height]: [u32; 2], usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | usage,
                view_formats: &[],
            })
        };
        let view = |texture: &wgpu::Texture| texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampled = wgpu::TextureUsages::TEXTURE_BINDING;
        let copied = wgpu::TextureUsages::COPY_SRC;
        let half = size.map(|side| side.div_ceil(2));

        let log_depth_texture = texture("log depth", log_depth_format, size, sampled | copied);
        let metrics_texture = texture("metrics", METRICS_FORMAT, size, copied);
        Self {
            size,
            color: view(&texture("scene colour", COLOR_FORMAT, size, sampled)),
            log_depth: view(&log_depth_texture),
            log_depth_texture,
            metrics: view(&metrics_texture),
            metrics_texture,
            depth: view(&texture("depth", DEPTH_FORMAT, size, wgpu::TextureUsages::empty())),
            blur: [0, 1].map(|_| view(&texture("blurred log depth", log_depth_format, half, sampled))),
        }
    }

    pub fn half_size(&self) -> [u32; 2] {
        self.size.map(|side| side.div_ceil(2))
    }
}

/// GPU resources owned by one side of the split screen.
struct Side {
    shading: wgpu::Buffer,
    /// Group 1 of the terrain pass.
    group: wgpu::BindGroup,
    field: LocalLightField,
}

/// What one side of the split screen renders with.
pub struct SideInput<'a> {
    pub params: &'a ShadingParams,
    /// Unit vector toward the side's lagged base light, from its light rig.
    pub light: Vec3,
}

pub struct FrameInput<'a> {
    pub pose: Pose,
    pub fov_y: f32,
    /// True terrain height below the camera, the reference for mesh detail.
    pub ground_height: f32,
    /// A pose to stream terrain detail for instead of the camera's, with
    /// the terrain height below it. A flight streams for where it is going.
    pub detail_for: Option<(Pose, f32)>,
    pub sides: [SideInput<'a>; 2],
    /// Fraction of the width at which side 1 begins; `None` shows only
    /// side 0.
    pub split: Option<f32>,
    /// Target size of a mesh quad on screen, in pixels.
    pub pixels_per_quad: f32,
    /// Most terrain tiles to take in this frame, of those the loader has
    /// ready; only the levels the camera is close enough to show are then
    /// loaded at all. `None` waits for every tile of every level before
    /// drawing, which keeps offline renders exact and independent of what
    /// was drawn before.
    pub tile_budget: Option<usize>,
}

/// Numbers about the last frame, for the interface.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub patches: usize,
    pub triangles: usize,
    /// Terrain tiles around the camera that are still to be loaded.
    pub missing_tiles: usize,
}

pub struct Renderer {
    meta: Meta,
    config: ShaderConfig,
    frame: wgpu::Buffer,
    frame_group: wgpu::BindGroup,
    terrain: TerrainPass,
    clipmap: Clipmap,
    quadtree: Quadtree,
    selection: Selection,
    local_light: LocalLightPipelines,
    /// Where the local light fields were placed last frame.
    field_origin: Option<Vec2>,
    sides: [Side; 2],
    post: PostPass,
    targets: Targets,
    pub timers: GpuTimers,
    pub stats: FrameStats,
}

impl Renderer {
    /// Uploads the terrain's whole levels and builds every pass.
    /// `output_format` is the format of the views that
    /// [`render`](Self::render) will draw into. The terrain's tiles are read
    /// from its own source on worker threads.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(gpu: &Gpu, terrain: &Terrain, output_format: wgpu::TextureFormat, size: [u32; 2]) -> Self {
        let feed = Box::new(clipmap::ThreadedFeed::new(terrain));
        Self::with_feed(gpu, terrain, feed, clipmap::FULL_REACH, output_format, size)
    }

    /// [`new`](Self::new) with the terrain's tiles coming from `feed`, and
    /// with `reach` tiles along a side of each streamed level loaded around
    /// the camera: [`clipmap::FULL_REACH`] or, where tiles are costly, less.
    pub fn with_feed(
        gpu: &Gpu,
        terrain: &Terrain,
        feed: Box<dyn TileFeed>,
        reach: usize,
        output_format: wgpu::TextureFormat,
        size: [u32; 2],
    ) -> Self {
        let device = &gpu.device;
        let textures = WholeTextures::upload(device, &gpu.queue, terrain);
        let clipmap = Clipmap::new(device, terrain, feed, reach);
        let sampler = |label: &str, address_mode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: address_mode,
                address_mode_v: address_mode,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })
        };
        let clamp_sampler = sampler("terrain sampler", wgpu::AddressMode::ClampToEdge);
        // The streamed windows wrap around their textures.
        let repeat_sampler = sampler("streamed terrain sampler", wgpu::AddressMode::Repeat);

        let config =
            ShaderConfig { window_texels: clipmap::window_texels(reach), float_filtering: gpu.float_filtering };
        let pass = TerrainPass::new(device, &config);
        let frame = util::uniform_buffer::<FrameUniform>(device, "frame uniform");
        let frame_group = util::bind_group(
            device,
            "terrain frame",
            &pass.frame_layout,
            &[
                frame.as_entire_binding(),
                wgpu::BindingResource::Sampler(&clamp_sampler),
                wgpu::BindingResource::Sampler(&repeat_sampler),
                util::texture(&textures.orientation),
                util::texture(&textures.lut),
                util::texture(&textures.water),
                util::texture(&textures.edge_distance),
                util::texture(&clipmap.height_view),
                util::texture(&clipmap.gradient_view),
                util::texture(&clipmap.water_view),
                util::texture(&textures.height),
                util::texture(&textures.gradient),
            ],
        );

        let local_light = LocalLightPipelines::new(device);
        let sides = [0, 1].map(|_| {
            let shading = util::uniform_buffer::<ShadingUniform>(device, "shading uniform");
            let field = LocalLightField::new(device, &local_light, &textures.orientation, &clamp_sampler);
            let group = util::bind_group(
                device,
                "terrain side",
                &pass.side_layout,
                &[shading.as_entire_binding(), util::texture(&field.delta)],
            );
            Side { shading, group, field }
        });

        let targets = Targets::new(device, size, config.log_depth_format());
        let post = PostPass::new(device, &config, output_format, &frame, &targets);
        Self {
            meta: terrain.meta.clone(),
            config,
            frame,
            frame_group,
            terrain: pass,
            clipmap,
            quadtree: Quadtree::new(&terrain.leaf_ranges, terrain.meta.cell_size),
            selection: Selection::default(),
            local_light,
            field_origin: None,
            sides,
            post,
            targets,
            timers: GpuTimers::new(device, &gpu.queue, gpu.timestamps),
            stats: FrameStats::default(),
        }
    }

    pub fn targets(&self) -> &Targets {
        &self.targets
    }

    /// Whether newly streamed terrain has changed the picture since this
    /// was last asked, even if nothing else has.
    pub fn take_streamed(&mut self) -> bool {
        self.clipmap.take_changed()
    }

    /// Call when the camera jumps to an unrelated view, so the light fields
    /// start afresh instead of carrying hysteresis over from the old view.
    pub fn forget_history(&mut self) {
        self.sides.iter_mut().for_each(|side| side.field.forget_history());
    }

    /// Resizes the offscreen targets to match the output.
    pub fn resize(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        if size.map(|side| side.max(1)) != self.targets.size {
            self.targets = Targets::new(device, size, self.config.log_depth_format());
            self.post.rebind(device, &self.frame, &self.targets);
        }
    }

    /// Encodes one frame into `output`, which must have the size of the
    /// targets and the format given to [`new`](Self::new).
    ///
    /// Fails only if a terrain tile cannot be read.
    pub fn render(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        input: &FrameInput,
    ) -> Result<()> {
        let [width, height] = self.targets.size;
        let camera_xy = input.pose.position.truncate();
        let basis = input.pose.basis();
        let pixel_angle = input.fov_y / height as f32;
        let sides_shown = if input.split.is_some() { 2 } else { 1 };
        let exaggerations = [0, 1].map(|side| input.sides[side.min(sides_shown - 1)].params.geometry_exaggeration);
        let z_scale = exaggerations[0].max(exaggerations[1]);

        let (detail_pose, detail_ground) = input.detail_for.unwrap_or((input.pose, input.ground_height));
        let finest = match input.tile_budget {
            Some(_) => {
                let clearance = detail_pose.position.z - detail_ground * z_scale;
                let bias =
                    (0..sides_shown).map(|side| generalization_bias(input.sides[side].params)).fold(f32::MAX, f32::min);
                finest_visible_level(clearance, pixel_angle, self.meta.cell_size, bias)
            }
            None => 0,
        };
        let ahead = detail_pose.basis().forward.truncate();
        self.clipmap.update(&gpu.queue, detail_pose.position.truncate(), ahead, finest, input.tile_budget)?;
        // The local light fields follow the camera together.
        let placement = FieldPlacement::around(camera_xy, self.field_origin);
        self.field_origin = Some(placement.origin);
        let split_column = input.split.map(|fraction| (fraction.clamp(0.0, 1.0) * width as f32).round());

        // Mesh level of detail, shared by both sides.
        let aspect = width as f32 / height as f32;
        let view_proj = input.pose.view_projection(input.fov_y, aspect);
        let lod_view = LodView {
            camera: input.pose.position,
            reference_z: input.ground_height * z_scale,
            range0: self.quadtree.range0(input.pixels_per_quad, pixel_angle),
            frustum: Frustum::from_view_projection(&view_proj),
            z_scale,
        };
        self.quadtree.select(&lod_view, &mut self.selection);
        let draw_list = self.terrain.upload(&gpu.queue, &self.selection);
        self.stats = FrameStats {
            patches: self.selection.patches.len(),
            triangles: self.selection.patches.len() * (PATCH_QUADS * PATCH_QUADS * 2) as usize,
            missing_tiles: self.clipmap.missing_tiles(),
        };

        let tan_half_fov = (0.5 * input.fov_y).tan();
        let stats = &self.meta.elevation;
        let frame = FrameUniform {
            view_proj: view_proj.to_cols_array_2d(),
            cam_pos: input.pose.position.extend(pixel_angle).into(),
            cam_right: basis.right.extend(tan_half_fov * aspect).into(),
            cam_up: basis.up.extend(tan_half_fov).into(),
            cam_forward: basis.forward.extend(0.0).into(),
            viewport: [width as f32, height as f32, 1.0 / width as f32, 1.0 / height as f32],
            terrain: [self.meta.extent(), self.meta.size as f32, self.meta.cell_size, (self.meta.levels - 1) as f32],
            elevation: [stats.min, stats.max, stats.p5, stats.p95],
            lod: [lod_view.range0, MORPH_START, lod_view.reference_z, PATCH_QUADS as f32],
            light_field: [placement.origin.x, placement.origin.y, FieldPlacement::SIDE_M, FIELD_SIZE as f32],
            loaded: self.clipmap.loaded_regions(),
        };
        gpu.queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&frame));

        // Per-side parameters and light fields.
        for (index, (side, side_input)) in self.sides.iter_mut().zip(&input.sides).take(sides_shown).enumerate() {
            let params = side_input.params;
            let skirt_base = stats.min * params.geometry_exaggeration - SKIRT_DEPTH_M;
            let shading = ShadingUniform::new(params, side_input.light, skirt_base);
            gpu.queue.write_buffer(&side.shading, 0, bytemuck::bytes_of(&shading));

            let uniform = LocalLightUniform::new(
                params,
                side_input.light,
                &placement,
                self.meta.extent(),
                self.meta.orientation_size,
            );
            let span = TimedSpan { opens: index == 0, closes: index == sides_shown - 1 };
            side.field.update(&gpu.queue, encoder, &self.local_light, &uniform, &mut self.timers, span);
        }

        self.timers.mark(Pass::Terrain);
        {
            let clear = |value: f64| wgpu::Color { r: value, g: 0.0, b: 0.0, a: 0.0 };
            let attachment = |view, color| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(color), store: wgpu::StoreOp::Store },
                })
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain"),
                color_attachments: &[
                    attachment(&self.targets.color, clear(0.0)),
                    attachment(&self.targets.log_depth, clear(SKY_LOG_DEPTH)),
                    attachment(&self.targets.metrics, clear(0.0)),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth,
                    // Reverse Z: the far plane is at depth 0.
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: self.timers.render_span(Pass::Terrain, true, true),
                occlusion_query_set: None,
            });
            self.terrain.bind(&mut pass);
            pass.set_bind_group(0, &self.frame_group, &[]);

            let split = split_column.map_or(width, |column| column as u32);
            let columns = [0..split, split..width];
            for (side, columns) in self.sides.iter().zip(columns).take(sides_shown) {
                if columns.is_empty() {
                    continue;
                }
                pass.set_scissor_rect(columns.start, 0, columns.len() as u32, height);
                pass.set_bind_group(1, &side.group, &[]);
                self.terrain.draw(&mut pass, &draw_list);
            }
        }

        let shown = |side: usize| input.sides[side.min(sides_shown - 1)].params;
        let blur_radius = (0..sides_shown)
            .map(shown)
            .filter(|params| params.depth_edges)
            .map(|params| params.depth_edge_radius)
            .reduce(f32::max);
        let post_frame =
            PostFrame { sides: [0, 1].map(|side| CompositeSide::new(shown(side))), split_column, blur_radius };
        self.post.render(&gpu.queue, encoder, output, &self.targets, &post_frame, &mut self.timers);
        Ok(())
    }
}
