//! The post pass (plan section 7.3): depth-edge enhancement on log depth,
//! the sky, and encoding for the output.

use bytemuck::{Pod, Zeroable};

use super::terrain::{linear_rgba, ShaderConfig};
use super::timers::{GpuTimers, Pass};
use super::util::{self, SAMPLER, TEXTURE, UNIFORM};
use super::Targets;
use crate::params::ShadingParams;

/// Most taps on each side of the blur's centre.
const MAX_BLUR_RADIUS: f32 = 32.0;
/// Darkening at silhouette lines when they are enabled.
const SILHOUETTE_STRENGTH: f32 = 0.35;

/// Mirrors `Blur` in post.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurUniform {
    tap_step: [f32; 2],
    sigma: f32,
    radius: f32,
}

/// Mirrors `Side` in post.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CompositeSide {
    haze_color: [f32; 4],
    zenith_color: [f32; 4],
    edges: [f32; 4],
}

impl CompositeSide {
    pub fn new(params: &ShadingParams) -> Self {
        let halo = if params.depth_edges { params.depth_edge_strength } else { 0.0 };
        let silhouette = if params.silhouettes { SILHOUETTE_STRENGTH } else { 0.0 };
        Self {
            haze_color: linear_rgba(params.haze_color, 1.0),
            zenith_color: linear_rgba(params.zenith_color, 1.0),
            edges: [halo, silhouette, 0.0, 0.0],
        }
    }
}

/// Mirrors `Composite` in post.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CompositeUniform {
    sides: [CompositeSide; 2],
    /// Pixel column where side 1 begins, whether the output is an sRGB
    /// format, and whether to draw the divider.
    options: [f32; 4],
}

/// What the post pass draws for one frame.
pub struct PostFrame {
    pub sides: [CompositeSide; 2],
    /// Pixel column where side 1 begins, if the screen is split.
    pub split_column: Option<f32>,
    /// Standard deviation of the depth-edge blur as a fraction of the image
    /// height; `None` skips the blur because no side uses depth edges.
    pub blur_radius: Option<f32>,
}

/// The resources that do not depend on the render targets.
struct Bindings {
    blur_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    /// Horizontal and vertical blur settings.
    blur_uniforms: [wgpu::Buffer; 2],
    composite_uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

/// Bind groups that reference the render targets; rebuilt on resize.
struct TargetGroups {
    blur: [wgpu::BindGroup; 2],
    composite: wgpu::BindGroup,
}

impl Bindings {
    fn bind_targets(&self, device: &wgpu::Device, frame: &wgpu::Buffer, targets: &Targets) -> TargetGroups {
        let sampler = wgpu::BindingResource::Sampler(&self.sampler);
        // The horizontal blur reads full-resolution log depth, the vertical
        // blur reads the horizontal result.
        let blur_sources = [&targets.log_depth, &targets.blur[0]];
        let blur = [0, 1].map(|i| {
            let resources =
                [self.blur_uniforms[i].as_entire_binding(), util::texture(blur_sources[i]), sampler.clone()];
            util::bind_group(device, "depth blur", &self.blur_layout, &resources)
        });
        let composite = util::bind_group(
            device,
            "composite",
            &self.composite_layout,
            &[
                frame.as_entire_binding(),
                self.composite_uniform.as_entire_binding(),
                util::texture(&targets.color),
                util::texture(&targets.log_depth),
                util::texture(&targets.blur[1]),
                sampler,
            ],
        );
        TargetGroups { blur, composite }
    }
}

pub struct PostPass {
    blur_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    bindings: Bindings,
    groups: TargetGroups,
    output_is_srgb: bool,
}

impl PostPass {
    pub fn new(
        device: &wgpu::Device,
        config: &ShaderConfig,
        output_format: wgpu::TextureFormat,
        frame: &wgpu::Buffer,
        targets: &Targets,
    ) -> Self {
        let layout = |label: &str, types: &[wgpu::BindingType]| {
            util::bind_group_layout(device, label, wgpu::ShaderStages::FRAGMENT, types)
        };
        let blur_layout = layout("depth blur", &[UNIFORM, TEXTURE, SAMPLER]);
        let composite_layout = layout("composite", &[UNIFORM, UNIFORM, TEXTURE, TEXTURE, TEXTURE, SAMPLER]);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post"),
            source: wgpu::ShaderSource::Wgsl(
                [&config.constants(), include_str!("../shaders/frame.wgsl"), include_str!("../shaders/post.wgsl")]
                    .concat()
                    .into(),
            ),
        });
        let pipeline = |label: &str, layout: &wgpu::BindGroupLayout, entry_point: &str, format| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry_point),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            })
        };
        let blur_pipeline = pipeline("depth blur", &blur_layout, "fs_blur", config.log_depth_format());
        let composite_pipeline = pipeline("composite", &composite_layout, "fs_composite", output_format);

        let bindings = Bindings {
            blur_layout,
            composite_layout,
            blur_uniforms: [0, 1].map(|_| util::uniform_buffer::<BlurUniform>(device, "depth blur uniform")),
            composite_uniform: util::uniform_buffer::<CompositeUniform>(device, "composite uniform"),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("post sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        };
        let groups = bindings.bind_targets(device, frame, targets);
        Self { blur_pipeline, composite_pipeline, bindings, groups, output_is_srgb: output_format.is_srgb() }
    }

    /// Points the pass at new render targets after a resize.
    pub fn rebind(&mut self, device: &wgpu::Device, frame: &wgpu::Buffer, targets: &Targets) {
        self.groups = self.bindings.bind_targets(device, frame, targets);
    }

    /// Composites the scene into `output`.
    pub fn render(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        targets: &Targets,
        frame: &PostFrame,
        timers: &mut GpuTimers,
    ) {
        if let Some(radius) = frame.blur_radius {
            let [half_width, half_height] = targets.half_size();
            let sigma = (radius * half_height as f32).max(0.5);
            let taps = (2.5 * sigma).ceil().min(MAX_BLUR_RADIUS);
            let steps = [[1.0 / half_width as f32, 0.0], [0.0, 1.0 / half_height as f32]];
            for (buffer, tap_step) in self.bindings.blur_uniforms.iter().zip(steps) {
                queue.write_buffer(buffer, 0, bytemuck::bytes_of(&BlurUniform { tap_step, sigma, radius: taps }));
            }
            // Horizontal into the first half-resolution target, then vertical
            // into the second.
            timers.mark(Pass::DepthEdges);
            let passes = [
                (&targets.blur[0], &self.groups.blur[0], timers.render_span(Pass::DepthEdges, true, false)),
                (&targets.blur[1], &self.groups.blur[1], timers.render_span(Pass::DepthEdges, false, true)),
            ];
            for (view, group, timestamp_writes) in passes {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("depth blur"),
                    color_attachments: &[Some(color_attachment(view))],
                    depth_stencil_attachment: None,
                    timestamp_writes,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.blur_pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        let uniform = CompositeUniform {
            sides: frame.sides,
            options: [
                frame.split_column.unwrap_or(f32::MAX),
                self.output_is_srgb as u32 as f32,
                frame.split_column.is_some() as u32 as f32,
                0.0,
            ],
        };
        queue.write_buffer(&self.bindings.composite_uniform, 0, bytemuck::bytes_of(&uniform));
        timers.mark(Pass::Composite);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite"),
            color_attachments: &[Some(color_attachment(output))],
            depth_stencil_attachment: None,
            timestamp_writes: timers.render_span(Pass::Composite, true, true),
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &self.groups.composite, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn color_attachment(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
    }
}
