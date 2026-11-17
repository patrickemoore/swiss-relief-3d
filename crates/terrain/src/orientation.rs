//! The landform orientation field (plan section 4.4).
//!
//! A structure tensor at two scales: gradients come from a pre-smoothed
//! pyramid level, then their outer products are averaged over about a
//! kilometre. Gradients from full-resolution data would let gullies on a
//! slope outweigh the slope itself and flip the orientation by 90 degrees.
//!
//! The dominant eigenvector is the fall-line axis `psi`, the direction across
//! ridges and valleys. It is stored in doubled-angle form
//! `(c cos 2psi, c sin 2psi)` with coherence `c`, which can be blurred and
//! interpolated freely because an axis has no front or back.

use serde::{Deserialize, Serialize};

use crate::filter::gaussian_blur;
use crate::grid::Grid;
use crate::pyramid::gradient;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrientationParams {
    /// Pyramid level the gradients are taken from (4 is 160 m at a 10 m base).
    pub source_level: usize,
    /// Standard deviation of the tensor averaging, in metres.
    pub integration_sigma_m: f32,
    /// Side length of the stored field, in texels.
    pub output_size: usize,
    /// Slope below which terrain counts as unoriented, so plains get no
    /// coherence from numerical noise.
    pub slope_floor: f32,
}

impl Default for OrientationParams {
    fn default() -> Self {
        Self { source_level: 4, integration_sigma_m: 500.0, output_size: 1024, slope_floor: 0.03 }
    }
}

/// Computes the doubled-angle orientation field of a square terrain from its
/// height pyramid.
pub fn orientation_field(heights: &[Grid<f32>], base_cell: f32, params: &OrientationParams) -> Grid<[f32; 2]> {
    let level = params.source_level.min(heights.len() - 1);
    let cell = base_cell * (1u64 << level) as f32;
    let texel = heights[level].width() as f32 * cell / params.output_size as f32;
    orientation_of_level(&heights[level], cell, params, params.output_size, texel)
}

/// Computes the orientation field from one pre-smoothed height level with
/// cells of `cell` metres.
///
/// The field has `size` x `size` texels of `texel` metres and shares the
/// level's south-west corner. It may reach beyond the level, where it is
/// zero: a terrain that fills only part of its square has no landforms
/// elsewhere.
pub fn orientation_of_level(
    heights: &Grid<f32>,
    cell: f32,
    params: &OrientationParams,
    size: usize,
    texel: f32,
) -> Grid<[f32; 2]> {
    let gradients = gradient(heights, cell);
    let (width, height) = (gradients.width(), gradients.height());

    let component = |f: fn([f32; 2]) -> f32| {
        let products = Grid::from_fn(width, height, |x, y| f(gradients.get(x, y)));
        gaussian_blur(&products, params.integration_sigma_m / cell)
    };
    let xx = component(|[gx, _]| gx * gx);
    let xy = component(|[gx, gy]| gx * gy);
    let yy = component(|[_, gy]| gy * gy);

    let floor = params.slope_floor * params.slope_floor;
    let (extent_x, extent_y) = (width as f32 * cell, height as f32 * cell);
    Grid::from_fn(size, size, |x, y| {
        let u = (x as f32 + 0.5) * texel / extent_x;
        let v = (y as f32 + 0.5) * texel / extent_y;
        if u > 1.0 || v > 1.0 {
            return [0.0, 0.0];
        }
        let (jxx, jxy, jyy) = (xx.sample(u, v), xy.sample(u, v), yy.sample(u, v));
        // (jxx - jyy, 2 jxy) has length (l1 - l2) and angle 2 psi; dividing by
        // the trace (l1 + l2) scales it by the coherence.
        let trace = jxx + jyy + floor;
        [(jxx - jyy) / trace, 2.0 * jxy / trace]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyramid::build_heights;

    fn field_for(height: impl Fn(f32, f32) -> f32 + Sync) -> [f32; 2] {
        let dem = Grid::from_fn(512, 512, |x, y| height(x as f32 * 10.0, y as f32 * 10.0));
        let heights = build_heights(dem, 10.0, None);
        let params = OrientationParams { output_size: 64, ..Default::default() };
        orientation_field(&heights, 10.0, &params).get(32, 32)
    }

    #[test]
    fn north_south_ridge_falls_east_west() {
        // Slopes fall along x, so psi = 0 and the doubled angle is (c, 0).
        let [c2, s2] = field_for(|x, _| 2000.0 - 0.5 * (x - 2555.0).abs());
        assert!(c2 > 0.9 && s2.abs() < 0.05, "got ({c2}, {s2})");
    }

    #[test]
    fn diagonal_ridge_has_diagonal_fall_line() {
        // A ridge along the south-west/north-east diagonal falls toward
        // north-west and south-east: psi = 135 degrees, doubled angle 270.
        let [c2, s2] = field_for(|x, y| 2000.0 - 0.4 * (x - y).abs());
        assert!(s2 < -0.9 && c2.abs() < 0.05, "got ({c2}, {s2})");
    }

    #[test]
    fn plains_have_no_coherence() {
        let [c2, s2] = field_for(|_, _| 500.0);
        assert!(c2.abs() < 1e-3 && s2.abs() < 1e-3);
    }
}
