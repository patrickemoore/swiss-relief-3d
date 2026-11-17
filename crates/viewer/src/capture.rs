//! Reading rendered frames back from the GPU, for frame dumps and metrics.

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result};

use relief_render::gpu::Gpu;

/// A texture the renderer can draw into and the CPU can read back.
pub struct OffscreenTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl OffscreenTarget {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, [width, height]: [u32; 2]) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen output"),
            size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }
}

/// A reusable staging buffer for reading one texture back from the GPU.
pub struct Readback {
    buffer: wgpu::Buffer,
    extent: wgpu::Extent3d,
    row_bytes: u32,
    /// Rows in a copy buffer are padded to the copy alignment.
    padded_row_bytes: u32,
}

impl Readback {
    /// A buffer sized for `texture`, or any texture of its size and format.
    pub fn new(gpu: &Gpu, texture: &wgpu::Texture) -> Result<Self> {
        let texel_bytes = texture.format().block_copy_size(None).context("texture format cannot be read back")?;
        let row_bytes = texture.width() * texel_bytes;
        let padded_row_bytes = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row_bytes * texture.height()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let extent = wgpu::Extent3d { width: texture.width(), height: texture.height(), depth_or_array_layers: 1 };
        Ok(Self { buffer, extent, row_bytes, padded_row_bytes })
    }

    /// Reads several textures with one submission, each into tightly packed
    /// rows of bytes. Blocks until the GPU has finished the frame.
    pub fn read_all(gpu: &Gpu, jobs: &[(&Readback, &wgpu::Texture)]) -> Result<Vec<Vec<u8>>> {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for (readback, texture) in jobs {
            let layout = wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(readback.padded_row_bytes),
                rows_per_image: Some(readback.extent.height),
            };
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo { buffer: &readback.buffer, layout },
                readback.extent,
            );
        }
        gpu.queue.submit([encoder.finish()]);

        let (sender, receiver) = std::sync::mpsc::channel();
        for (readback, _) in jobs {
            let sender = sender.clone();
            readback.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                // The receiver only disappears if this function has already failed.
                let _ = sender.send(result);
            });
        }
        gpu.device.poll(wgpu::Maintain::Wait);

        jobs.iter()
            .map(|(readback, _)| {
                receiver.recv().context("the GPU dropped the readback")?.context("mapping the readback buffer")?;
                let row_bytes = readback.row_bytes as usize;
                let mut pixels = Vec::with_capacity(row_bytes * readback.extent.height as usize);
                for row in readback.buffer.slice(..).get_mapped_range().chunks_exact(readback.padded_row_bytes as usize)
                {
                    pixels.extend_from_slice(&row[..row_bytes]);
                }
                readback.buffer.unmap();
                Ok(pixels)
            })
            .collect()
    }
}

/// Copies a texture to tightly packed rows of bytes. Blocks until the GPU
/// has finished the frame.
pub fn read_texture(gpu: &Gpu, texture: &wgpu::Texture) -> Result<Vec<u8>> {
    let readback = Readback::new(gpu, texture)?;
    let mut textures = Readback::read_all(gpu, &[(&readback, texture)])?;
    Ok(textures.remove(0))
}

/// Reads a rendered output as RGBA8, whatever channel order its format has.
pub fn read_rgba(gpu: &Gpu, texture: &wgpu::Texture) -> Result<Vec<u8>> {
    let mut pixels = read_texture(gpu, texture)?;
    match texture.format() {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {}
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            pixels.as_chunks_mut::<4>().0.iter_mut().for_each(|pixel| pixel.swap(0, 2));
        }
        other => anyhow::bail!("cannot convert {other:?} to RGBA8"),
    }
    Ok(pixels)
}

/// Writes RGBA8 pixels as an sRGB PNG, creating parent directories.
pub fn save_png(path: &Path, [width, height]: [u32; 2], rgba: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    encoder.write_header()?.write_image_data(rgba)?;
    Ok(())
}

/// Reinterprets bytes read from an `R32Float` texture as floats.
pub fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|&bytes| f32::from_le_bytes(bytes)).collect()
}
