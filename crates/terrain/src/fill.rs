//! Filling the cells an elevation model has no data for.
//!
//! A country's elevation model ends at its border, but generalizing it does
//! not: every coarser pyramid level averages across that border. Cells
//! without data are therefore filled with a smooth continuation of the
//! terrain next to them, so coarse levels near the border are not dragged
//! toward an arbitrary value.

use rayon::prelude::*;

use crate::grid::Grid;

/// Replaces every NaN cell by a smooth extrapolation of the valid cells.
///
/// This is a push-pull fill: the grid is averaged down, ignoring holes,
/// until none remain, and each hole is then read back from the next-coarser
/// level. Holes next to data continue it; holes far from any data settle on
/// the mean of a wide surrounding.
pub fn fill_holes(grid: &mut Grid<f32>) {
    if !grid.data().par_iter().any(|value| value.is_nan()) {
        return;
    }
    let (width, height) = (grid.width(), grid.height());
    if width == 1 && height == 1 {
        // Nothing valid anywhere.
        grid.set(0, 0, 0.0);
        return;
    }

    let mut coarse = Grid::from_fn(width.div_ceil(2), height.div_ceil(2), |x, y| {
        let (mut sum, mut count) = (0.0f32, 0u32);
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (fx, fy) = (2 * x + dx, 2 * y + dy);
            if fx < width && fy < height && !grid.get(fx, fy).is_nan() {
                sum += grid.get(fx, fy);
                count += 1;
            }
        }
        if count > 0 {
            sum / count as f32
        } else {
            f32::NAN
        }
    });
    fill_holes(&mut coarse);

    let (coarse_width, coarse_height) = (coarse.width() as f32, coarse.height() as f32);
    grid.data_mut().par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, value) in row.iter_mut().enumerate().filter(|(_, value)| value.is_nan()) {
            // The fine cell's centre in the coarse grid's unit square.
            *value = coarse.sample((x as f32 + 0.5) / (2.0 * coarse_width), (y as f32 + 0.5) / (2.0 * coarse_height));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_cells_are_kept_and_holes_continue_them() {
        // A plateau at 1000 m in the west half, no data in the east half.
        let mut grid = Grid::from_fn(64, 64, |x, _| if x < 32 { 1000.0 } else { f32::NAN });
        fill_holes(&mut grid);
        assert!(grid.data().iter().all(|value| (value - 1000.0).abs() < 1e-3));
    }

    #[test]
    fn fill_blends_between_separate_areas() {
        let mut grid = Grid::from_fn(64, 8, |x, _| match x {
            0..8 => 0.0,
            56..64 => 800.0,
            _ => f32::NAN,
        });
        fill_holes(&mut grid);
        assert_eq!(grid.get(3, 3), 0.0);
        assert_eq!(grid.get(60, 3), 800.0);
        let (near_west, middle, near_east) = (grid.get(12, 4), grid.get(32, 4), grid.get(52, 4));
        assert!(near_west < middle && middle < near_east, "{near_west} {middle} {near_east}");
        assert!(grid.data().iter().all(|value| (0.0..=800.0).contains(value)));
    }

    #[test]
    fn a_grid_without_data_becomes_flat() {
        let mut grid = Grid::new(5, 3, f32::NAN);
        fill_holes(&mut grid);
        assert!(grid.data().iter().all(|&value| value == 0.0));
    }
}
