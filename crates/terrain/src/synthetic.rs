//! Synthetic terrain for debugging the shading.

use anyhow::Result;

use crate::dataset::{BuildOptions, Dataset};
use crate::grid::Grid;
use crate::water::LakeParams;

/// A single cone on a plain.
///
/// Under the Swiss convention its north-west flank must come out brightest,
/// which catches gradient sign errors at once.
pub fn cone(size: usize, cell: f32) -> Grid<f32> {
    let extent = size as f32 * cell;
    let (base, peak, radius) = (500.0, 2500.0, 0.35 * extent);
    Grid::from_fn(size, size, |x, y| {
        let dx = (x as f32 + 0.5) * cell - 0.5 * extent;
        let dy = (y as f32 + 0.5) * cell - 0.5 * extent;
        let distance = (dx * dx + dy * dy).sqrt();
        base + (peak - base) * (1.0 - distance / radius).max(0.0)
    })
}

/// The cone as a complete dataset of `size` x `size` cells of 10 m.
pub fn cone_dataset(size: usize) -> Result<Dataset> {
    const CELL_M: f32 = 10.0;
    // The plain around the cone is perfectly flat, but it is not a lake.
    let options =
        BuildOptions { lakes: LakeParams { min_area_m2: f32::INFINITY, ..Default::default() }, ..Default::default() };
    Dataset::build("cone", "synthetic", cone(size, CELL_M), CELL_M, [0.0, 0.0], &options)
}
