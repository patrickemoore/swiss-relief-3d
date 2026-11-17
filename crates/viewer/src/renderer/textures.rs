//! Uploads the parts of a terrain that are held whole: the coarse pyramid
//! levels and the fields that cover the entire terrain square.
//!
//! The pyramids become mip chains built by our own filter rather than the
//! GPU's box filter, so trilinear sampling interpolates between
//! generalization levels. The finer levels are streamed; see `clipmap`.

use half::f16;
use relief_terrain::pyramid::gradient;
use relief_terrain::terrain::TILED_LEVELS;
use relief_terrain::water::halve_mask;
use relief_terrain::{Grid, Terrain};

pub struct WholeTextures {
    /// R32Float heights in metres: a mip chain per pyramid, plain in array
    /// layer 0 and accentuated in layer 1.
    pub height: wgpu::TextureView,
    /// RG16Float slopes `(dh/dx, dh/dy)`, laid out like the heights.
    pub gradient: wgpu::TextureView,
    /// R8Unorm lake coverage.
    pub water: wgpu::TextureView,
    /// RG16Float doubled-angle landform orientation.
    pub orientation: wgpu::TextureView,
    /// R16Float metres to the nearest place without data.
    pub edge_distance: wgpu::TextureView,
    /// sRGB colour table: x is brightness, y is elevation.
    pub lut: wgpu::TextureView,
}

/// One mip level: its side length in texels and its bytes.
type Level = (usize, Vec<u8>);

impl WholeTextures {
    pub fn upload(device: &wgpu::Device, queue: &wgpu::Queue, terrain: &Terrain) -> Self {
        let first_cell = terrain.cell(TILED_LEVELS);
        let heights = |pyramid: &[Grid<f32>]| -> Vec<Level> {
            pyramid.iter().map(|level| (level.width(), bytemuck::cast_slice(level.data()).to_vec())).collect()
        };
        let gradients = |pyramid: &[Grid<f32>]| -> Vec<Level> {
            pyramid
                .iter()
                .enumerate()
                .map(|(index, level)| {
                    let slopes = gradient(level, first_cell * (1u64 << index) as f32);
                    (level.width(), half_floats(slopes.data().iter().flatten().copied()))
                })
                .collect()
        };

        let orientation = mip_chain(terrain.orientation.clone(), |fine| {
            Grid::from_fn(fine.width() / 2, fine.height() / 2, |x, y| {
                let texels =
                    [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dy): (usize, usize)| fine.get(2 * x + dx, 2 * y + dy));
                [0, 1].map(|channel| texels.iter().map(|texel| texel[channel]).sum::<f32>() * 0.25)
            })
        });
        let water = mip_chain(terrain.coarse_water.clone(), halve_mask);
        let edge = &terrain.edge_distance;

        let upload = |label: &str, format, layers: &[Vec<Level>]| upload(device, queue, label, format, layers);
        Self {
            height: upload(
                "whole heights",
                wgpu::TextureFormat::R32Float,
                &[heights(&terrain.coarse[0]), heights(&terrain.coarse[1])],
            ),
            gradient: upload(
                "whole gradients",
                wgpu::TextureFormat::Rg16Float,
                &[gradients(&terrain.coarse[0]), gradients(&terrain.coarse[1])],
            ),
            water: upload(
                "whole lake mask",
                wgpu::TextureFormat::R8Unorm,
                &[water.iter().map(|level| (level.width(), level.data().to_vec())).collect()],
            ),
            orientation: upload(
                "landform orientation",
                wgpu::TextureFormat::Rg16Float,
                &[orientation
                    .iter()
                    .map(|level| (level.width(), half_floats(level.data().iter().flatten().copied())))
                    .collect()],
            ),
            edge_distance: upload(
                "distance to the data's edge",
                wgpu::TextureFormat::R16Float,
                &[vec![(edge.width(), half_floats(edge.data().iter().copied()))]],
            ),
            lut: upload_lut(device, queue, terrain),
        }
    }
}

/// Creates a square texture with one array layer per entry of `layers`, each
/// a mip chain. More than one layer makes it an array texture.
fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    format: wgpu::TextureFormat,
    layers: &[Vec<Level>],
) -> wgpu::TextureView {
    let side = layers[0][0].0 as u32;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: side, height: side, depth_or_array_layers: layers.len() as u32 },
        mip_level_count: layers[0].len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let texel_bytes = format.block_copy_size(None).expect("terrain textures use uncompressed colour formats");
    for (layer, levels) in layers.iter().enumerate() {
        for (mip_level, (side, bytes)) in levels.iter().enumerate() {
            let side = *side as u32;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip_level as u32,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(side * texel_bytes),
                    rows_per_image: Some(side),
                },
                wgpu::Extent3d { width: side, height: side, depth_or_array_layers: 1 },
            );
        }
    }
    let dimension = if layers.len() > 1 { wgpu::TextureViewDimension::D2Array } else { wgpu::TextureViewDimension::D2 };
    texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(dimension), ..Default::default() })
}

fn upload_lut(device: &wgpu::Device, queue: &wgpu::Queue, terrain: &Terrain) -> wgpu::TextureView {
    let lut = &terrain.lut;
    let size = wgpu::Extent3d { width: lut.width as u32, height: lut.height as u32, depth_or_array_layers: 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("colour table"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &lut.rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.width * 4),
            rows_per_image: Some(size.height),
        },
        size,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Half floats as little-endian bytes.
fn half_floats(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(|value| f16::from_f32(value).to_bits().to_le_bytes()).collect()
}

/// A mip chain down to a single texel, each level made from the last by
/// `halve`.
fn mip_chain<T: Copy + Send + Sync>(base: Grid<T>, halve: impl Fn(&Grid<T>) -> Grid<T>) -> Vec<Grid<T>> {
    let mut levels = vec![base];
    while levels.last().is_some_and(|level| level.width() > 1) {
        levels.push(halve(levels.last().expect("the chain has a base level")));
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_chain_ends_at_one_texel() {
        let levels = mip_chain(Grid::from_vec(4, 4, vec![255u8; 16]), halve_mask);
        assert_eq!(levels.iter().map(Grid::width).collect::<Vec<_>>(), [4, 2, 1]);
        assert_eq!(levels[2].data(), &[255]);
    }

    #[test]
    fn half_floats_are_little_endian() {
        assert_eq!(half_floats([1.0, -2.0].into_iter()), [0x00, 0x3c, 0x00, 0xc0]);
    }
}
