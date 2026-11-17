//! Rendering without a window, for snapshots and the evaluation.

use anyhow::Result;
use relief_terrain::Terrain;

use crate::camera::{Ground, Pose};
use crate::capture::{self, OffscreenTarget};
use crate::gpu::Gpu;
use crate::light_rig::LightRig;
use crate::params::ShadingParams;
use crate::renderer::{FrameInput, Renderer, SideInput};

pub use crate::camera::DEFAULT_FOV_Y;
pub use crate::renderer::DEFAULT_PIXELS_PER_QUAD;

const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The parameters of one side of the screen with the light rig that follows
/// the camera for it.
pub type SideRef<'a> = (&'a ShadingParams, &'a mut LightRig);

/// A renderer with an offscreen output and a fixed-size frame.
pub struct Headless {
    pub gpu: Gpu,
    pub renderer: Renderer,
    pub ground: Ground,
    output: OffscreenTarget,
    pub size: [u32; 2],
}

impl Headless {
    pub fn new(terrain: &Terrain, size: [u32; 2]) -> Result<Self> {
        let gpu = Gpu::new(&Gpu::instance(), None)?;
        let renderer = Renderer::new(&gpu, terrain, OUTPUT_FORMAT, size);
        let output = OffscreenTarget::new(&gpu.device, OUTPUT_FORMAT, size);
        Ok(Self { gpu, renderer, ground: Ground::new(terrain), output, size })
    }

    /// Renders one frame, advancing each light rig by `dt` seconds.
    ///
    /// `compare` adds a second parameter set on the right, beginning at the
    /// given fraction of the width. Every terrain tile the view needs is
    /// loaded first, so the frame does not depend on what was drawn before.
    pub fn render(
        &mut self,
        pose: Pose,
        main: SideRef<'_>,
        compare: Option<(SideRef<'_>, f32)>,
        dt: f32,
    ) -> Result<()> {
        fn advance<'a>((params, rig): SideRef<'a>, pose: &Pose, dt: f32) -> SideInput<'a> {
            SideInput { params, light: rig.update(pose, params, dt) }
        }
        let left = advance(main, &pose, dt);
        let (right, split) = match compare {
            Some((compared, split)) => (advance(compared, &pose, dt), Some(split)),
            None => (SideInput { params: left.params, light: left.light }, None),
        };
        let input = FrameInput {
            pose,
            fov_y: DEFAULT_FOV_Y,
            ground_height: self.ground.height(pose.position.truncate()),
            sides: [left, right],
            split,
            pixels_per_quad: DEFAULT_PIXELS_PER_QUAD,
            detail_for: None,
            tile_budget: None,
        };
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.renderer.render(&self.gpu, &mut encoder, &self.output.view, &input)?;
        self.renderer.timers.resolve(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
        self.renderer.timers.after_submit();
        Ok(())
    }

    /// Waits for the GPU to finish all submitted frames.
    pub fn finish(&mut self) {
        self.gpu.device.poll(wgpu::Maintain::Wait);
        self.renderer.timers.collect();
    }

    /// The texture frames are rendered into.
    pub fn output_texture(&self) -> &wgpu::Texture {
        &self.output.texture
    }

    /// The last rendered frame as RGBA8.
    pub fn read_output(&self) -> Result<Vec<u8>> {
        capture::read_rgba(&self.gpu, &self.output.texture)
    }
}
