//! The local light pass (plan section 6.2): a compute pass that fills and
//! blurs a coarse field of per-landform light rotations every frame.
//!
//! It has to run every frame because the base light's azimuth depends on the
//! camera. The field covers a square around the camera rather than the whole
//! terrain, which may be a country. The shaders hold the reasoning; this
//! module owns their resources.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

use super::timers::{GpuTimers, Pass};
use super::util::{self, SAMPLER, TEXTURE, UNIFORM};
use crate::params::ShadingParams;

/// Side length of the rotation field in cells.
pub const FIELD_SIZE: u32 = 1024;
/// Cell size of the field in metres. Rotations are blurred over a landform,
/// several cells wide, so nothing finer is needed.
pub const FIELD_CELL_M: f32 = 160.0;
/// The field moves in steps of this many cells, so that every cell keeps
/// covering the same ground and can find its own history.
const FIELD_STEP_CELLS: f32 = 16.0;
const FIELD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const WORKGROUP_SIZE: u32 = 8;
/// Upper bound on the blur's standard deviation in cells, which bounds the
/// cost of the pass whatever the slider says.
const MAX_BLUR_SIGMA: f32 = 24.0;

/// Where the field sits in a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldPlacement {
    /// South-west corner in metres.
    pub origin: Vec2,
    /// Cells the field has moved by since the previous frame.
    pub moved: [i32; 2],
}

impl FieldPlacement {
    /// Side length of the field in metres.
    pub const SIDE_M: f32 = FIELD_SIZE as f32 * FIELD_CELL_M;

    /// Centres the field on the camera. `previous` is where it was placed
    /// last frame, if it was.
    pub fn around(camera: Vec2, previous: Option<Vec2>) -> Self {
        let step = FIELD_STEP_CELLS * FIELD_CELL_M;
        let origin = ((camera - 0.5 * Self::SIDE_M) / step).round() * step;
        let moved = previous.map_or(Vec2::ZERO, |previous| (origin - previous) / FIELD_CELL_M);
        Self { origin, moved: [moved.x.round() as i32, moved.y.round() as i32] }
    }
}

/// Mirrors `LocalLight` in the local light shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LocalLightUniform {
    base_azimuth: f32,
    delta_max: f32,
    theta_ok: f32,
    hysteresis: f32,
    sign_softness: f32,
    blur_sigma: f32,
    orientation_level: f32,
    pad: f32,
    field: [f32; 4],
    moved: [i32; 2],
    pad1: [i32; 2],
}

impl LocalLightUniform {
    /// `extent` is the terrain's side length in metres and
    /// `orientation_size` the side length of the orientation texture.
    pub fn new(
        params: &ShadingParams,
        light: Vec3,
        placement: &FieldPlacement,
        extent: f32,
        orientation_size: usize,
    ) -> Self {
        let orientation_texel = extent / orientation_size as f32;
        Self {
            base_azimuth: light.y.atan2(light.x),
            delta_max: params.local_max_rotation_deg.to_radians(),
            theta_ok: params.local_ok_angle_deg.to_radians(),
            hysteresis: params.local_hysteresis_deg.to_radians(),
            sign_softness: params.local_sign_softness_deg.to_radians().max(1e-4),
            blur_sigma: (params.local_blur_m / FIELD_CELL_M).min(MAX_BLUR_SIGMA),
            // The mip of the orientation texture whose texels match the cells.
            orientation_level: (FIELD_CELL_M / orientation_texel).log2().max(0.0),
            pad: 0.0,
            field: [placement.origin.x, placement.origin.y, FieldPlacement::SIDE_M, extent],
            moved: placement.moved,
            pad1: [0; 2],
        }
    }
}

/// Pipelines shared by both sides of a split screen.
pub struct LocalLightPipelines {
    rotate: wgpu::ComputePipeline,
    blur_x: wgpu::ComputePipeline,
    blur_y: wgpu::ComputePipeline,
    rotate_layout: wgpu::BindGroupLayout,
    blur_layout: wgpu::BindGroupLayout,
}

impl LocalLightPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let storage = wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: FIELD_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        };
        let layout = |label: &str, types: &[wgpu::BindingType]| {
            util::bind_group_layout(device, label, wgpu::ShaderStages::COMPUTE, types)
        };
        let rotate_layout = layout("local light rotate", &[UNIFORM, TEXTURE, SAMPLER, TEXTURE, storage]);
        let blur_layout = layout("local light blur", &[UNIFORM, TEXTURE, storage]);

        let pipeline = |label: &str, source: &str, entry_point: &str, layout: &wgpu::BindGroupLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let rotate_source = include_str!("../shaders/local_light_rotate.wgsl");
        let blur_source = include_str!("../shaders/local_light_blur.wgsl");
        Self {
            rotate: pipeline("local light rotate", rotate_source, "rotate", &rotate_layout),
            blur_x: pipeline("local light blur x", blur_source, "blur_x", &blur_layout),
            blur_y: pipeline("local light blur y", blur_source, "blur_y", &blur_layout),
            rotate_layout,
            blur_layout,
        }
    }
}

/// Whether a field is the first or the last to be updated in a frame. The
/// timed local light span opens with the first and closes with the last.
#[derive(Clone, Copy, Debug)]
pub struct TimedSpan {
    pub opens: bool,
    pub closes: bool,
}

/// One rotation field with its intermediate textures.
pub struct LocalLightField {
    uniform: wgpu::Buffer,
    /// `rotate[i]` reads raw texture `1 - i` as last frame's result and
    /// writes raw texture `i`; the two swap roles every frame so each cell
    /// can see which side it rotated toward before.
    rotate: [wgpu::BindGroup; 2],
    /// `blur_x[i]` blurs raw texture `i` into the scratch texture.
    blur_x: [wgpu::BindGroup; 2],
    blur_y: wgpu::BindGroup,
    /// The finished field: red holds the rotation in radians.
    pub delta: wgpu::TextureView,
    current: usize,
    forget_history: bool,
}

impl LocalLightField {
    pub fn new(
        device: &wgpu::Device,
        pipelines: &LocalLightPipelines,
        orientation: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> Self {
        let field = |label: &str| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: FIELD_SIZE, height: FIELD_SIZE, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: FIELD_FORMAT,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let raw = [field("local light raw 0"), field("local light raw 1")];
        let scratch = field("local light scratch");
        let delta = field("local light rotation");

        let uniform = util::uniform_buffer::<LocalLightUniform>(device, "local light uniform");
        let rotate = [0, 1].map(|i| {
            util::bind_group(
                device,
                "local light rotate",
                &pipelines.rotate_layout,
                &[
                    uniform.as_entire_binding(),
                    util::texture(orientation),
                    wgpu::BindingResource::Sampler(sampler),
                    util::texture(&raw[1 - i]),
                    util::texture(&raw[i]),
                ],
            )
        });
        let blur = |label: &str, source: &wgpu::TextureView, target: &wgpu::TextureView| {
            let resources = [uniform.as_entire_binding(), util::texture(source), util::texture(target)];
            util::bind_group(device, label, &pipelines.blur_layout, &resources)
        };
        let blur_x = [0, 1].map(|i| blur("local light blur x", &raw[i], &scratch));
        let blur_y = blur("local light blur y", &scratch, &delta);

        Self { uniform, rotate, blur_x, blur_y, delta, current: 0, forget_history: false }
    }

    /// Makes the next update ignore which way each cell rotated before.
    ///
    /// Hysteresis makes the field depend on how the camera got to a view.
    /// After a jump to an unrelated view that history is meaningless, and
    /// reproducible measurements must not depend on it.
    pub fn forget_history(&mut self) {
        self.forget_history = true;
    }

    /// Recomputes the field for this frame's light.
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &LocalLightPipelines,
        uniform: &LocalLightUniform,
        timers: &mut GpuTimers,
        span: TimedSpan,
    ) {
        // Without hysteresis the shader never consults last frame's result.
        let hysteresis = if std::mem::take(&mut self.forget_history) { 0.0 } else { uniform.hysteresis };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&LocalLightUniform { hysteresis, ..*uniform }));
        timers.mark(Pass::LocalLight);
        self.current = 1 - self.current;

        let steps = [
            ("local light rotate", &pipelines.rotate, &self.rotate[self.current], span.opens, false),
            ("local light blur x", &pipelines.blur_x, &self.blur_x[self.current], false, false),
            ("local light blur y", &pipelines.blur_y, &self.blur_y, false, span.closes),
        ];
        let groups = FIELD_SIZE.div_ceil(WORKGROUP_SIZE);
        for (label, pipeline, bind_group, first, last) in steps {
            // Each step is its own pass: a texture cannot be read and
            // written within one.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: timers.compute_span(Pass::LocalLight, first, last),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups, groups, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_field_follows_the_camera_in_whole_steps() {
        let first = FieldPlacement::around(Vec2::new(100_000.0, 50_000.0), None);
        let half = 0.5 * FieldPlacement::SIDE_M;
        assert!((first.origin - Vec2::new(100_000.0 - half, 50_000.0 - half)).abs().max_element() <= 1280.0);
        assert_eq!(first.moved, [0, 0]);

        // A short move leaves the field where it is.
        let near = FieldPlacement::around(Vec2::new(100_400.0, 50_000.0), Some(first.origin));
        assert_eq!((near.origin, near.moved), (first.origin, [0, 0]));

        // A longer one moves it by a whole number of steps of 16 cells.
        let far = FieldPlacement::around(Vec2::new(105_300.0, 45_300.0), Some(first.origin));
        assert_eq!(far.moved, [32, -32]);
        assert_eq!(far.origin - first.origin, Vec2::new(5120.0, -5120.0));
    }
}
