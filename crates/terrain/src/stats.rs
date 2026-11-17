//! Elevation statistics of a region (plan section 4.4).

use serde::{Deserialize, Serialize};

use crate::grid::Grid;

/// Range and percentiles used to normalize elevation for aerial perspective
/// and colour.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ElevationStats {
    pub min: f32,
    pub max: f32,
    pub p5: f32,
    pub p95: f32,
}

impl ElevationStats {
    pub fn compute(heights: &Grid<f32>) -> Self {
        Self::from_samples(heights.data().to_vec()).expect("a raster has at least one cell")
    }

    /// Statistics of a set of heights, or `None` if it is empty. A regular
    /// subsample of a large terrain gives the same percentiles.
    pub fn from_samples(mut samples: Vec<f32>) -> Option<Self> {
        samples.sort_unstable_by(f32::total_cmp);
        let last = samples.len().checked_sub(1)?;
        let percentile = |p: f32| samples[(last as f32 * p).round() as usize];
        Some(Self { min: samples[0], max: samples[last], p5: percentile(0.05), p95: percentile(0.95) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_of_a_ramp() {
        let ramp = Grid::from_fn(101, 1, |x, _| x as f32);
        let stats = ElevationStats::compute(&ramp);
        assert_eq!((stats.min, stats.p5, stats.p95, stats.max), (0.0, 5.0, 95.0, 100.0));
    }

    #[test]
    fn no_samples_give_no_statistics() {
        assert_eq!(ElevationStats::from_samples(vec![]), None);
    }
}
