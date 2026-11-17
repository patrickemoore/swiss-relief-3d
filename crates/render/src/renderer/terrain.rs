//! The terrain pass: instanced CDLOD patches drawn with the shading model.
//!
//! It writes three targets: linear colour, log depth for the depth-edge
//! pass, and a small metrics target the evaluation reads back.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use relief_terrain::oklab::srgb_to_linear;
use relief_terrain::terrain::TILED_LEVELS;
use wgpu::util::DeviceExt;

use super::cdlod::{PatchInstance, PatchMesh, Selection};
use super::util::{self, PLAIN_TEXTURE_ARRAY, SAMPLER, TEXTURE, TEXTURE_ARRAY, UNIFORM};
use crate::light_rig::sun_direction;
use crate::params::{ShadingMode, ShadingParams};

pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const METRICS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Most patches a frame can draw, including skirt instances.
const MAX_INSTANCES: usize = 16_384;
/// sRGB colour and opacity of lakes.
const WATER_COLOR: [f32; 4] = [0.60, 0.72, 0.80, 0.9];

// Feature bits, mirrored in terrain.wgsl.
const FLAG_LOCAL_LIGHT: u32 = 1;
const FLAG_GENERALIZATION: u32 = 2;
const FLAG_ELEVATION_CONTRAST: u32 = 4;
const FLAG_FOG: u32 = 8;
const FLAG_COLOR: u32 = 16;
const FLAG_WATER: u32 = 32;
const FLAG_SHADOWS: u32 = 64;

/// What the shaders are built for.
#[derive(Clone, Copy, Debug)]
pub struct ShaderConfig {
    /// Texels along one side of a streamed window.
    pub window_texels: u32,
    /// Whether the GPU filters 32-bit float textures. Where it does not,
    /// the shaders filter the heights themselves.
    pub float_filtering: bool,
}

impl ShaderConfig {
    /// Constants the shaders share with the rest of the program, as WGSL.
    pub fn constants(&self) -> String {
        format!("const TILED_LEVELS: i32 = {TILED_LEVELS};\nconst WINDOW_TEXELS: f32 = {}.0;\n", self.window_texels)
    }

    /// Format of the log-depth target and its blurred copies. They are
    /// sampled with filtering, so without it for 32 bits they make do
    /// with 16.
    pub fn log_depth_format(&self) -> wgpu::TextureFormat {
        if self.float_filtering {
            wgpu::TextureFormat::R32Float
        } else {
            wgpu::TextureFormat::R16Float
        }
    }
}

/// Converts an sRGB colour to linear RGB with the given alpha.
pub fn linear_rgba([r, g, b]: [f32; 3], alpha: f32) -> [f32; 4] {
    [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), alpha]
}

/// Mirrors `Shading` in terrain.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ShadingUniform {
    light: [f32; 4],
    sun: [f32; 4],
    detail_weights: [f32; 4],
    haze_color: [f32; 4],
    water_color: [f32; 4],
    landform_weight: f32,
    detail_gain: f32,
    shade_exaggeration: f32,
    geometry_exaggeration: f32,
    bias: f32,
    landform_level: f32,
    flat_brightness: f32,
    contrast: f32,
    valley_tone: f32,
    low_contrast: f32,
    contrast_h_lo: f32,
    contrast_h_hi: f32,
    fog_density: f32,
    fog_scale_height: f32,
    fog_max: f32,
    edge_fade: f32,
    delta_max: f32,
    skirt_base: f32,
    pad: [f32; 2],
    flags: u32,
    mode: u32,
    debug_view: u32,
    footprint: u32,
    accentuated: u32,
    pad_end: [u32; 3],
}

impl ShadingUniform {
    /// `light` is the unit vector toward the lagged camera-relative light and
    /// `skirt_base` the world height of the skirts' lower edge.
    pub fn new(params: &ShadingParams, light: Vec3, skirt_base: f32) -> Self {
        let flags = [
            (params.local_light, FLAG_LOCAL_LIGHT),
            (params.generalization, FLAG_GENERALIZATION),
            (params.elevation_contrast, FLAG_ELEVATION_CONTRAST),
            (params.fog, FLAG_FOG),
            (params.color, FLAG_COLOR),
            (params.water, FLAG_WATER),
            (params.shadows, FLAG_SHADOWS),
        ];
        Self {
            light: light.extend(0.0).into(),
            sun: sun_direction(params).extend(params.sun_ambient).into(),
            detail_weights: params.detail_weights(),
            haze_color: linear_rgba(params.haze_color, 1.0),
            water_color: linear_rgba([WATER_COLOR[0], WATER_COLOR[1], WATER_COLOR[2]], WATER_COLOR[3]),
            landform_weight: params.landform_weight,
            detail_gain: params.detail_gain,
            shade_exaggeration: params.shading_exaggeration,
            geometry_exaggeration: params.geometry_exaggeration,
            bias: params.generalization_bias,
            landform_level: params.landform_level,
            flat_brightness: params.flat_brightness,
            contrast: params.contrast,
            valley_tone: params.valley_tone,
            low_contrast: params.low_contrast,
            contrast_h_lo: params.contrast_elevation_lo,
            contrast_h_hi: params.contrast_elevation_hi,
            fog_density: params.fog_density_per_km / 1000.0,
            fog_scale_height: params.fog_scale_height_m.max(1.0),
            fog_max: params.fog_max,
            edge_fade: params.edge_fade_m,
            delta_max: params.local_max_rotation_deg.to_radians(),
            skirt_base,
            pad: [0.0; 2],
            flags: flags.iter().filter(|(on, _)| *on).fold(0, |bits, (_, flag)| bits | flag),
            mode: match params.mode {
                ShadingMode::Swiss => 0,
                ShadingMode::Draped => 1,
                ShadingMode::Realistic => 2,
                ShadingMode::Hillshade => 3,
            },
            debug_view: params.debug_view as u32,
            footprint: params.analytic_footprint as u32,
            accentuated: params.accentuation as u32,
            pad_end: [0; 3],
        }
    }
}

pub struct TerrainPass {
    pipeline: wgpu::RenderPipeline,
    /// Layout of group 0: frame constants and the terrain's textures.
    pub frame_layout: wgpu::BindGroupLayout,
    /// Layout of group 1: one side's parameters and light field.
    pub side_layout: wgpu::BindGroupLayout,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    instances: wgpu::Buffer,
    mesh: PatchMesh,
}

/// Where one frame's instances sit in the instance buffer.
pub struct DrawList {
    patches: u32,
    /// Instance counts of the four skirt strips, stored after the patches.
    skirts: [u32; 4],
}

impl TerrainPass {
    pub fn new(device: &wgpu::Device, config: &ShaderConfig) -> Self {
        let heights = if config.float_filtering { TEXTURE_ARRAY } else { PLAIN_TEXTURE_ARRAY };
        let layout = |label: &str, types: &[wgpu::BindingType]| {
            util::bind_group_layout(device, label, wgpu::ShaderStages::VERTEX_FRAGMENT, types)
        };
        let frame_layout = layout(
            "terrain frame",
            &[
                UNIFORM,
                // One sampler clamps, the other wraps for the streamed windows.
                SAMPLER,
                SAMPLER,
                // Orientation, colour table, whole lake mask, edge distance.
                TEXTURE,
                TEXTURE,
                TEXTURE,
                TEXTURE,
                // Streamed heights, gradients and lake mask.
                heights,
                TEXTURE_ARRAY,
                TEXTURE_ARRAY,
                // Whole heights and gradients.
                heights,
                TEXTURE_ARRAY,
            ],
        );
        let side_layout = layout("terrain side", &[UNIFORM, TEXTURE]);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain"),
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
                ]
                .concat()
                .into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain"),
            bind_group_layouts: &[&frame_layout, &side_layout],
            push_constant_ranges: &[],
        });
        let target = |format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 3]>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<PatchInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x4],
                    },
                ],
                compilation_options: Default::default(),
            },
            // Skirts are seen from either side, and terrain is never seen
            // from below, so nothing is culled.
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                // Reverse Z: nearer fragments have greater depth.
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[target(COLOR_FORMAT), target(config.log_depth_format()), target(METRICS_FORMAT)],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        let mesh = PatchMesh::new();
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patch vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patch indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("patch instances"),
            size: (MAX_INSTANCES * std::mem::size_of::<PatchInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, frame_layout, side_layout, vertices, indices, instances, mesh }
    }

    /// Uploads a frame's selection: patches first, then each skirt strip.
    pub fn upload(&self, queue: &wgpu::Queue, selection: &Selection) -> DrawList {
        let mut all: Vec<PatchInstance> = selection.patches.clone();
        all.extend(selection.skirts.iter().flatten());
        if all.len() > MAX_INSTANCES {
            log::warn!("{} terrain patches selected, drawing only the first {MAX_INSTANCES}", all.len());
            all.truncate(MAX_INSTANCES);
        }
        queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&all));

        let patches = selection.patches.len().min(MAX_INSTANCES) as u32;
        let mut remaining = all.len() as u32 - patches;
        let skirts = [0, 1, 2, 3].map(|edge| {
            let count = (selection.skirts[edge].len() as u32).min(remaining);
            remaining -= count;
            count
        });
        DrawList { patches, skirts }
    }

    /// Binds the pipeline and geometry. Group 0 and group 1 are bound by the
    /// caller, which then issues [`draw`](Self::draw) per side.
    pub fn bind(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_vertex_buffer(1, self.instances.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, list: &DrawList) {
        pass.draw_indexed(self.mesh.patch_indices.clone(), 0, 0..list.patches);
        let mut first = list.patches;
        for (indices, &count) in self.mesh.skirt_indices.iter().zip(&list.skirts) {
            if count > 0 {
                pass.draw_indexed(indices.clone(), 0, first..first + count);
            }
            first += count;
        }
    }
}
