//! The generalized height pyramid and its gradients (plan sections 4.3, 4.4).
//!
//! Level `k` holds the terrain generalized at a scale of `cell * 2^k` metres.
//! The levels become the mip chain of one texture, so hardware trilinear
//! filtering interpolates between generalization scales for free.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::filter::smoothstep;
use crate::grid::Grid;

/// Ridge and valley accentuation, a simplified Terrain Sculptor.
///
/// Gaussian smoothing lowers crests and raises valley floors. Wherever a
/// smoothed level is strongly curved, its cells are pulled back toward the
/// finer level, which keeps major ridges sharp and valleys incised.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Accentuation {
    /// How far curved cells move back toward the finer level, 0 to 1.
    pub strength: f32,
    /// Slope change across one cell at which accentuation starts.
    pub curvature_lo: f32,
    /// Slope change across one cell at which accentuation is complete.
    pub curvature_hi: f32,
}

impl Default for Accentuation {
    fn default() -> Self {
        Self { strength: 0.9, curvature_lo: 0.05, curvature_hi: 0.35 }
    }
}

/// Coarse rows that [`downsample`] makes in one piece.
const BAND_ROWS: usize = 64;

/// Halves a level with the even kernel `[1 3 3 1] / 8` in both directions.
///
/// The kernel has to be even: mip sampling assumes each coarse texel is
/// centred between two fine texels. An odd kernel centred on a fine texel
/// would shift every level by half a fine texel, and by level 7 landform
/// shading would sit hundreds of metres away from the geometry.
pub fn downsample(fine: &Grid<f32>) -> Grid<f32> {
    let (fine_w, fine_h) = (fine.width(), fine.height());
    let (coarse_w, coarse_h) = ((fine_w / 2).max(1), (fine_h / 2).max(1));

    // The kernel is separable: rows are halved across, then along. That is
    // done a band of rows at a time, because the rows halved across would
    // otherwise take half the memory of the fine level, which a country
    // cannot spare.
    let mut coarse = Grid::new(coarse_w, coarse_h, 0.0f32);
    coarse.data_mut().par_chunks_mut(coarse_w * BAND_ROWS).enumerate().for_each(|(band, rows)| {
        // The fine rows from one below the band to one above it.
        let first = 2 * (band * BAND_ROWS) as isize - 1;
        let fine_rows = 2 * (rows.len() / coarse_w) + 2;
        let mut across = Vec::with_capacity(fine_rows * coarse_w);
        for row in 0..fine_rows {
            let y = (first + row as isize).clamp(0, fine_h as isize - 1);
            across.extend((0..coarse_w).map(|x| even_kernel(|i| fine.get_clamped(i, y), 2 * x as isize)));
        }
        for (y, row) in rows.chunks_mut(coarse_w).enumerate() {
            for (x, cell) in row.iter_mut().enumerate() {
                *cell = even_kernel(|j| across[j as usize * coarse_w + x], 2 * y as isize + 1);
            }
        }
    });
    coarse
}

/// Weighted average of fine cells `first - 1 ..= first + 2`.
#[inline]
fn even_kernel(sample: impl Fn(isize) -> f32, first: isize) -> f32 {
    (sample(first - 1) + 3.0 * (sample(first) + sample(first + 1)) + sample(first + 2)) * 0.125
}

/// Mean of the 2 x 2 fine block under coarse cell `(x, y)`.
#[inline]
fn block_mean(fine: &Grid<f32>, x: usize, y: usize) -> f32 {
    let (x, y) = (2 * x as isize, 2 * y as isize);
    0.25 * (fine.get_clamped(x, y)
        + fine.get_clamped(x + 1, y)
        + fine.get_clamped(x, y + 1)
        + fine.get_clamped(x + 1, y + 1))
}

/// Blends strongly curved cells of `smoothed` back toward `fine`.
fn accentuate(smoothed: &Grid<f32>, fine: &Grid<f32>, cell: f32, params: &Accentuation) -> Grid<f32> {
    Grid::from_fn(smoothed.width(), smoothed.height(), |x, y| {
        let (xi, yi) = (x as isize, y as isize);
        let centre = smoothed.get(x, y);
        let neighbours = smoothed.get_clamped(xi - 1, yi)
            + smoothed.get_clamped(xi + 1, yi)
            + smoothed.get_clamped(xi, yi - 1)
            + smoothed.get_clamped(xi, yi + 1);
        // Laplacian times the cell size: the change of slope across one cell.
        let curvature = (neighbours - 4.0 * centre) / cell;
        let mask = params.strength * smoothstep(params.curvature_lo, params.curvature_hi, curvature.abs());
        centre + mask * (block_mean(fine, x, y) - centre)
    })
}

/// Derives the next-coarser level from `fine`. `cell` is the cell size of the
/// level being made.
pub fn next_level(fine: &Grid<f32>, cell: f32, accentuation: Option<&Accentuation>) -> Grid<f32> {
    let smoothed = downsample(fine);
    match accentuation {
        Some(params) => accentuate(&smoothed, fine, cell, params),
        None => smoothed,
    }
}

/// Builds every level from `base` down to a single cell.
///
/// With `accentuation`, each smoothed level is sharpened before the next one
/// is derived from it, so preserved crest heights carry through to the
/// coarsest levels.
pub fn build_heights(base: Grid<f32>, cell: f32, accentuation: Option<&Accentuation>) -> Vec<Grid<f32>> {
    let mut levels = vec![base];
    while levels.last().is_some_and(|level| level.width() > 1 || level.height() > 1) {
        let fine = levels.last().expect("pyramid always has a base level");
        let level_cell = cell * (1u64 << levels.len()) as f32;
        levels.push(next_level(fine, level_cell, accentuation));
    }
    levels
}

/// Slope components `(dh/dx, dh/dy)` by central differences.
///
/// `dh/dy` is the derivative toward **north**. The renderer builds normals as
/// `normalize(-z * gx, -z * gy, 1)`; storing gradients instead of normals
/// keeps trilinear filtering meaningful, because gradients average linearly.
pub fn gradient(heights: &Grid<f32>, cell: f32) -> Grid<[f32; 2]> {
    let (width, height) = (heights.width(), heights.height());
    let derivative = |low: usize, high: usize, sample: &dyn Fn(usize) -> f32| {
        if high == low {
            0.0
        } else {
            (sample(high) - sample(low)) / ((high - low) as f32 * cell)
        }
    };
    Grid::from_fn(width, height, |x, y| {
        let (west, east) = (x.saturating_sub(1), (x + 1).min(width - 1));
        let (south, north) = (y.saturating_sub(1), (y + 1).min(height - 1));
        [derivative(west, east, &|i| heights.get(i, y)), derivative(south, north, &|j| heights.get(x, j))]
    })
}

/// Gradients for every level, each at its own cell size.
pub fn build_gradients(heights: &[Grid<f32>], cell: f32) -> Vec<Grid<[f32; 2]>> {
    heights.iter().enumerate().map(|(level, grid)| gradient(grid, cell * (1u64 << level) as f32)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsample_keeps_a_plane_aligned() {
        // A linear ramp must stay the same ramp at the coarse level: the
        // coarse cell centre lies between fine cells 2i and 2i + 1.
        let fine = Grid::from_fn(16, 16, |x, _| x as f32 + 0.5);
        let coarse = downsample(&fine);
        for x in 1..7 {
            assert!((coarse.get(x, 4) - (2.0 * x as f32 + 1.0)).abs() < 1e-5);
        }
    }

    #[test]
    fn pyramid_ends_at_one_cell() {
        let levels = build_heights(Grid::new(64, 64, 100.0), 10.0, None);
        assert_eq!(levels.len(), 7);
        assert_eq!(levels.last().unwrap().data(), &[100.0]);
    }

    #[test]
    fn gradient_points_uphill_east_and_north() {
        let heights = Grid::from_fn(8, 8, |x, y| 2.0 * x as f32 * 10.0 + 3.0 * y as f32 * 10.0);
        let [gx, gy] = gradient(&heights, 10.0).get(4, 4);
        assert!((gx - 2.0).abs() < 1e-5 && (gy - 3.0).abs() < 1e-5);
    }

    #[test]
    fn accentuation_keeps_crests_higher() {
        let ridge = Grid::from_fn(64, 64, |x, _| 1000.0 - 6.0 * (x as f32 - 31.5).abs());
        let plain = build_heights(ridge.clone(), 10.0, None);
        let sharp = build_heights(ridge, 10.0, Some(&Accentuation::default()));
        let crest = |levels: &[Grid<f32>]| levels[3].row(4).iter().copied().fold(f32::MIN, f32::max);
        assert!(crest(&sharp) > crest(&plain) + 1.0);
    }
}
