//! Lake detection (plan section 7.2).
//!
//! Lakes appear in a bare-earth elevation model as perfectly flat areas, so a
//! slope-zero, minimum-area mask finds them without any extra data.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::grid::Grid;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LakeParams {
    /// Smallest flat area that counts as a lake, in square metres.
    pub min_area_m2: f32,
    /// Largest height difference between neighbouring cells of a lake.
    pub tolerance_m: f32,
}

impl Default for LakeParams {
    fn default() -> Self {
        Self { min_area_m2: 40_000.0, tolerance_m: 0.02 }
    }
}

const NEIGHBOURS: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// Mask value of a lake cell.
pub const LAKE: u8 = 255;
// States of a cell while flat areas are being measured.
const SLOPED: u8 = 0;
const FLAT: u8 = 1;
const MEASURED: u8 = 2;

/// Returns a mask that is [`LAKE`] on lakes and 0 elsewhere.
///
/// Cells without data are NaN. Look for lakes before filling them in: the
/// fill can be flat too, but it is not a lake.
pub fn lake_mask(heights: &Grid<f32>, cell: f32, params: &LakeParams) -> Grid<u8> {
    let (width, height) = (heights.width(), heights.height());

    // One grid holds each cell's state, then becomes the mask. A terrain the
    // size of a country leaves no room for a grid per step.
    let mut state = Grid::from_fn(width, height, |x, y| {
        let (xi, yi) = (x as isize, y as isize);
        // A NaN fails every comparison, so a cell without data is never
        // flat, and neither are the cells next to it.
        let level = NEIGHBOURS
            .iter()
            .all(|(dx, dy)| (heights.get(x, y) - heights.get_clamped(xi + dx, yi + dy)).abs() <= params.tolerance_m);
        if level {
            FLAT
        } else {
            SLOPED
        }
    });

    let min_cells = (params.min_area_m2 / (cell * cell)).ceil() as usize;
    let mut component = Vec::new();
    let mut queue = VecDeque::new();
    for start_y in 0..height {
        for start_x in 0..width {
            if state.get(start_x, start_y) != FLAT {
                continue;
            }
            component.clear();
            state.set(start_x, start_y, MEASURED);
            queue.push_back((start_x, start_y));
            while let Some((x, y)) = queue.pop_front() {
                component.push((x, y));
                for (dx, dy) in NEIGHBOURS {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    if state.get(nx, ny) == FLAT {
                        state.set(nx, ny, MEASURED);
                        queue.push_back((nx, ny));
                    }
                }
            }
            if component.len() >= min_cells {
                for &(x, y) in &component {
                    state.set(x, y, LAKE);
                }
            }
        }
    }
    grow_to_shoreline(&state)
}

/// Extends lakes by one cell: shoreline cells blend water and land when the
/// source data is resampled, so the flatness test rejects them.
fn grow_to_shoreline(state: &Grid<u8>) -> Grid<u8> {
    Grid::from_fn(state.width(), state.height(), |x, y| {
        let touches_lake =
            NEIGHBOURS.iter().any(|(dx, dy)| state.get_clamped(x as isize + dx, y as isize + dy) == LAKE);
        if state.get(x, y) == LAKE || touches_lake {
            LAKE
        } else {
            0
        }
    })
}

/// Halves a mask by averaging 2 x 2 blocks, so that lakes thin out into
/// partial coverage at coarser levels.
pub fn halve_mask(fine: &Grid<u8>) -> Grid<u8> {
    Grid::from_fn((fine.width() / 2).max(1), (fine.height() / 2).max(1), |x, y| {
        let (x, y) = (2 * x as isize, 2 * y as isize);
        let sum: u32 =
            [(0, 0), (1, 0), (0, 1), (1, 1)].iter().map(|(dx, dy)| fine.get_clamped(x + dx, y + dy) as u32).sum();
        ((sum + 2) / 4) as u8
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 30 x 30 cell lake and a 3 x 3 cell puddle on a tilted plane.
    fn terrain() -> Grid<f32> {
        Grid::from_fn(100, 100, |x, y| {
            let in_lake = (20..50).contains(&x) && (20..50).contains(&y);
            let in_puddle = (70..73).contains(&x) && (70..73).contains(&y);
            if in_lake || in_puddle {
                500.0
            } else {
                600.0 + x as f32 + 0.5 * y as f32
            }
        })
    }

    #[test]
    fn finds_a_lake_and_ignores_slopes_and_puddles() {
        let mask = lake_mask(&terrain(), 10.0, &LakeParams::default());
        assert_eq!(mask.get(35, 35), LAKE);
        assert_eq!(mask.get(71, 71), 0);
        assert_eq!(mask.get(90, 10), 0);
    }

    #[test]
    fn a_lake_ends_where_the_data_does() {
        // The eastern third of the lake has no data.
        let whole = terrain();
        let heights = Grid::from_fn(100, 100, |x, y| if x < 40 { whole.get(x, y) } else { f32::NAN });
        let mask = lake_mask(&heights, 10.0, &LakeParams::default());
        assert_eq!(mask.get(25, 35), LAKE);
        assert_eq!(mask.get(45, 35), 0);
    }

    #[test]
    fn halving_averages_coverage() {
        let mask = Grid::from_vec(2, 2, vec![0, LAKE, LAKE, LAKE]);
        assert_eq!(halve_mask(&mask).data(), &[191]);
    }
}
