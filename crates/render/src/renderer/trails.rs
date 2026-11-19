//! Hiking trails drawn over the terrain as lines of constant width on the
//! screen, draped on whatever surface the terrain pass is drawing.
//!
//! Each tile of trails becomes a buffer of segments; a segment is drawn as
//! an instanced quad that the vertex shader expands in screen space into a
//! capsule, with the two ends draped on the height pyramid. Lines are
//! depth-tested against the terrain, so what a ridge hides stays hidden.
//!
//! Tiles follow the terrain's windows: for every streamed level the tiles
//! the terrain targets are requested from the feed, and a tile is drawn
//! only where no finer tile covers the same ground.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver, Sender};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use relief_terrain::pack::TileIndex;
use relief_terrain::terrain::TILED_LEVELS;
use relief_terrain::trails::{Polyline, TrailClass};
#[cfg(not(target_arch = "wasm32"))]
use relief_terrain::trails::{TrailSource, Trails};
use wgpu::util::DeviceExt;

use super::clipmap::TileRect;
use super::terrain::{linear_rgba, ShaderConfig, TerrainPass, COLOR_FORMAT, DEPTH_FORMAT, METRICS_FORMAT};
use super::util;

/// Trail tiles taken in per frame, at most.
const TILES_PER_FRAME: usize = 6;

/// One tile of one level.
pub type TrailKey = (usize, [i32; 2]);

/// Where trail tiles come from: the renderer asks, and takes what is ready.
pub trait TrailFeed {
    fn request(&mut self, level: usize, tile: [i32; 2]);
    fn ready(&mut self) -> Option<(TrailKey, Result<Vec<Polyline>>)>;
    /// Blocks until a requested tile is ready, where that is possible.
    fn wait(&mut self) -> Option<(TrailKey, Result<Vec<Polyline>>)> {
        self.ready()
    }
}

/// Trails read on worker threads from a dataset on disk.
#[cfg(not(target_arch = "wasm32"))]
pub struct ThreadedTrailFeed {
    source: Arc<dyn TrailSource>,
    finished: Sender<(TrailKey, Result<Vec<Polyline>>)>,
    arrived: Receiver<(TrailKey, Result<Vec<Polyline>>)>,
}

#[cfg(not(target_arch = "wasm32"))]
impl ThreadedTrailFeed {
    pub fn new(trails: &Trails) -> Self {
        let (finished, arrived) = mpsc::channel();
        Self { source: trails.source.clone(), finished, arrived }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl TrailFeed for ThreadedTrailFeed {
    fn request(&mut self, level: usize, tile: [i32; 2]) {
        let (source, finished) = (self.source.clone(), self.finished.clone());
        rayon::spawn(move || {
            let lines = source.tile(level, tile[0] as usize, tile[1] as usize).map(Option::unwrap_or_default);
            let _ = finished.send(((level, tile), lines));
        });
    }

    fn ready(&mut self) -> Option<(TrailKey, Result<Vec<Polyline>>)> {
        self.arrived.try_recv().ok()
    }

    fn wait(&mut self) -> Option<(TrailKey, Result<Vec<Polyline>>)> {
        self.arrived.recv().ok()
    }
}

/// How trails are drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrailStyle {
    /// Width of the coloured core in CSS pixels, and of the casing on each
    /// side of it.
    pub width: f32,
    pub casing: f32,
    /// Device pixels per CSS pixel.
    pub pixel_ratio: f32,
}

impl Default for TrailStyle {
    fn default() -> Self {
        Self { width: 1.5, casing: 0.9, pixel_ratio: 1.0 }
    }
}

/// Signpost colours: yellow, white-red-white, white-blue-white.
const HIKING_COLOR: [f32; 3] = [0.94, 0.72, 0.05];
const MOUNTAIN_COLOR: [f32; 3] = [0.82, 0.12, 0.12];
const ALPINE_COLOR: [f32; 3] = [0.14, 0.40, 0.80];
const CASING_COLOR: [f32; 3] = [0.99, 0.98, 0.95];

/// Ground metres per CSS pixel at which a class begins to fade, and at
/// which it is gone: the dense network of yellow trails only once the view
/// is close; mountain and alpine routes from farther out.
const HIKING_FADE: [f32; 2] = [10.0, 18.0];
const UPLAND_FADE: [f32; 2] = [45.0, 80.0];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TrailUniform {
    /// Core width, casing width, device pixels per CSS pixel, unused.
    style: [f32; 4],
    /// Hiking fade start and end, upland fade start and end.
    fade: [f32; 4],
    /// Casing, then the three classes, in linear RGB.
    colors: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Segment {
    a: [f32; 2],
    b: [f32; 2],
    class: u32,
    level: u32,
}

struct GpuTile {
    buffer: wgpu::Buffer,
    segments: u32,
}

/// The tiles of one level: `None` while a tile is on its way.
type LevelTiles = HashMap<[i32; 2], Option<GpuTile>>;

pub struct TrailLayer {
    pipeline: wgpu::RenderPipeline,
    /// The style for the casing pass and for the core pass.
    uniforms: [wgpu::Buffer; 2],
    groups: [wgpu::BindGroup; 2],
    index: TileIndex,
    feed: Box<dyn TrailFeed>,
    levels: [LevelTiles; TILED_LEVELS],
    changed: bool,
}

impl TrailLayer {
    pub fn new(
        device: &wgpu::Device,
        config: &ShaderConfig,
        terrain_pass: &TerrainPass,
        index: TileIndex,
        feed: Box<dyn TrailFeed>,
    ) -> Self {
        let layout = util::bind_group_layout(
            device,
            "trails",
            wgpu::ShaderStages::VERTEX_FRAGMENT,
            &[wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            }],
        );
        let uniforms = [0, 1].map(|_| util::uniform_buffer::<TrailUniform>(device, "trail style"));
        let groups =
            [0, 1].map(|pass| util::bind_group(device, "trails", &layout, &[uniforms[pass].as_entire_binding()]));

        // The shader shares the terrain's sampling of the pyramid, so it is
        // built from the same sources, with its own entry points.
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trails"),
            source: wgpu::ShaderSource::Wgsl(
                [
                    config.constants().as_str(),
                    include_str!("../shaders/frame.wgsl"),
                    include_str!("../shaders/terrain.wgsl"),
                    if config.float_filtering {
                        include_str!("../shaders/heights_filtered.wgsl")
                    } else {
                        include_str!("../shaders/heights_loaded.wgsl")
                    },
                    include_str!("../shaders/trails.wgsl"),
                ]
                .concat()
                .into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trails"),
            bind_group_layouts: &[&terrain_pass.frame_layout, &terrain_pass.side_layout, &layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trails"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_trail"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Segment>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Uint32, 3 => Uint32],
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                // Lines neither hide the terrain nor each other.
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_trail"),
                // Only the colour target is written; the log depth and the
                // metrics stay the terrain's.
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::COLOR,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: config.log_depth_format(),
                        blend: None,
                        write_mask: wgpu::ColorWrites::empty(),
                    }),
                    Some(wgpu::ColorTargetState {
                        format: METRICS_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::empty(),
                    }),
                ],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });
        Self {
            pipeline,
            uniforms,
            groups,
            index,
            feed,
            levels: std::array::from_fn(|_| HashMap::new()),
            changed: false,
        }
    }

    fn exists(&self, level: usize, tile: [i32; 2]) -> bool {
        tile[0] >= 0 && tile[1] >= 0 && self.index.has(level, tile[0] as usize, tile[1] as usize)
    }

    /// Follows the terrain's windows: `targets` are the tiles each level
    /// wants. Tiles outside are let go, tiles missing are asked for, and
    /// those that have arrived are uploaded, a few per frame.
    /// With `wait`, nothing is left pending: an offline render waits for every
    /// tile, as it does for the terrain's.
    pub fn update(&mut self, device: &wgpu::Device, targets: &[TileRect; TILED_LEVELS], wait: bool) {
        for (level, &target) in targets.iter().enumerate() {
            let before = self.levels[level].len();
            self.levels[level].retain(|tile, _| target.contains(*tile));
            self.changed |= self.levels[level].len() != before;
            for tile in target.tiles() {
                if self.exists(level, tile) && !self.levels[level].contains_key(&tile) {
                    self.levels[level].insert(tile, None);
                    self.feed.request(level, tile);
                }
            }
        }
        let mut taken = 0;
        loop {
            if !wait && taken >= TILES_PER_FRAME {
                break;
            }
            if wait && self.missing_tiles() == 0 {
                break;
            }
            let next = if wait { self.feed.wait() } else { self.feed.ready() };
            let Some(((level, tile), lines)) = next else { break };
            taken += 1;
            // Not wanted any more: the window has moved on.
            if !matches!(self.levels[level].get(&tile), Some(None)) {
                continue;
            }
            let lines = match lines {
                Ok(lines) => lines,
                Err(error) => {
                    log::warn!("trails level {level} tile {tile:?}: {error:#}");
                    Vec::new()
                }
            };
            let segments: Vec<Segment> = lines
                .iter()
                .flat_map(|line| {
                    line.points.windows(2).map(move |pair| Segment {
                        a: pair[0],
                        b: pair[1],
                        class: line.class as u32,
                        level: level as u32,
                    })
                })
                .collect();
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("trail segments"),
                contents: bytemuck::cast_slice(&segments),
                usage: wgpu::BufferUsages::VERTEX,
            });
            self.levels[level].insert(tile, Some(GpuTile { buffer, segments: segments.len() as u32 }));
            self.changed = true;
        }
    }

    /// Whether what is drawn has changed since this was last asked.
    pub fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }

    pub fn set_style(&self, queue: &wgpu::Queue, style: &TrailStyle) {
        for (pass, buffer) in self.uniforms.iter().enumerate() {
            let uniform = TrailUniform {
                style: [style.width, style.casing, style.pixel_ratio.max(0.25), pass as f32],
                fade: [HIKING_FADE[0], HIKING_FADE[1], UPLAND_FADE[0], UPLAND_FADE[1]],
                colors: [
                    linear_rgba(CASING_COLOR, 1.0),
                    linear_rgba(HIKING_COLOR, 1.0),
                    linear_rgba(MOUNTAIN_COLOR, 1.0),
                    linear_rgba(ALPINE_COLOR, 1.0),
                ],
            };
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&uniform));
        }
    }

    /// Whether the finer level has the whole of a tile's ground, so that the
    /// tile itself need not be drawn.
    fn covered_below(&self, level: usize, tile: [i32; 2]) -> bool {
        if level == 0 {
            return false;
        }
        let below = level - 1;
        [[0, 0], [1, 0], [0, 1], [1, 1]].into_iter().all(|[dx, dy]| {
            let child = [2 * tile[0] + dx, 2 * tile[1] + dy];
            !self.exists(below, child) || matches!(self.levels[below].get(&child), Some(Some(_)))
        })
    }

    /// Draws every tile that no finer tile replaces. Groups 0 and 1 are the
    /// terrain's and are bound already.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        // Every casing, then every core, so joints between segments stay clean.
        for group in &self.groups {
            pass.set_bind_group(2, group, &[]);
            for level in 0..TILED_LEVELS {
                for (&tile, loaded) in &self.levels[level] {
                    let Some(gpu) = loaded else { continue };
                    if gpu.segments == 0 || self.covered_below(level, tile) {
                        continue;
                    }
                    pass.set_vertex_buffer(0, gpu.buffer.slice(..));
                    pass.draw(0..6, 0..gpu.segments);
                }
            }
        }
    }

    /// Tiles waiting to arrive.
    pub fn missing_tiles(&self) -> usize {
        self.levels.iter().map(|tiles| tiles.values().filter(|tile| tile.is_none()).count()).sum()
    }
}

/// The class a segment carries, for the interface.
pub fn class_name(class: TrailClass) -> &'static str {
    match class {
        TrailClass::Hiking => "hiking trail",
        TrailClass::Mountain => "mountain trail",
        TrailClass::Alpine => "alpine route",
    }
}
