//! Separable filters shared by the analysis stages.

use rayon::prelude::*;

use crate::grid::Grid;

/// Normalized Gaussian weights for offsets `-radius..=radius`.
pub fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil().max(1.0) as isize;
    let weights: Vec<f32> = (-radius..=radius).map(|i| (-0.5 * (i as f32 / sigma).powi(2)).exp()).collect();
    let sum: f32 = weights.iter().sum();
    weights.into_iter().map(|w| w / sum).collect()
}

/// Gaussian blur with clamped edges. `sigma` is in cells.
pub fn gaussian_blur(src: &Grid<f32>, sigma: f32) -> Grid<f32> {
    if sigma <= 0.0 {
        return src.clone();
    }
    let kernel = gaussian_kernel(sigma);
    let horizontal = convolve(src, &kernel, Axis::X);
    convolve(&horizontal, &kernel, Axis::Y)
}

#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
}

fn convolve(src: &Grid<f32>, kernel: &[f32], axis: Axis) -> Grid<f32> {
    let radius = (kernel.len() / 2) as isize;
    let width = src.width();
    let mut out = Grid::new(width, src.height(), 0.0f32);
    out.data_mut().par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, cell) in row.iter_mut().enumerate() {
            *cell = kernel
                .iter()
                .enumerate()
                .map(|(k, weight)| {
                    let offset = k as isize - radius;
                    let (sx, sy) = match axis {
                        Axis::X => (x as isize + offset, y as isize),
                        Axis::Y => (x as isize, y as isize + offset),
                    };
                    weight * src.get_clamped(sx, sy)
                })
                .sum();
        }
    });
    out
}

/// Hermite interpolation between two edges, as in GLSL.
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_preserves_constants_and_mass() {
        let flat = Grid::new(16, 16, 3.0f32);
        assert!(gaussian_blur(&flat, 2.0).data().iter().all(|v| (v - 3.0).abs() < 1e-5));

        let mut spike = Grid::new(33, 33, 0.0f32);
        spike.set(16, 16, 1.0);
        let total: f32 = gaussian_blur(&spike, 2.0).data().iter().sum();
        assert!((total - 1.0).abs() < 1e-4);
    }
}
