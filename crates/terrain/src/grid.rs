//! A minimal dense 2D raster.

use rayon::prelude::*;

/// A row-major raster. Row 0 is the southern edge, so `y` grows north.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid<T> {
    width: usize,
    height: usize,
    data: Vec<T>,
}

impl<T: Copy + Send + Sync> Grid<T> {
    pub fn new(width: usize, height: usize, fill: T) -> Self {
        Self { width, height, data: vec![fill; width * height] }
    }

    pub fn from_vec(width: usize, height: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), width * height, "raster data does not match its dimensions");
        Self { width, height, data }
    }

    /// Builds a raster by evaluating `f(x, y)` for every cell, in parallel.
    pub fn from_fn(width: usize, height: usize, f: impl Fn(usize, usize) -> T + Sync) -> Self {
        let f = &f;
        let data = (0..height).into_par_iter().flat_map_iter(|y| (0..width).map(move |x| f(x, y))).collect();
        Self { width, height, data }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn row(&self, y: usize) -> &[T] {
        &self.data[y * self.width..(y + 1) * self.width]
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> T {
        self.data[y * self.width + x]
    }

    /// Reads a cell, clamping out-of-range coordinates to the nearest edge.
    #[inline]
    pub fn get_clamped(&self, x: isize, y: isize) -> T {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: T) {
        self.data[y * self.width + x] = value;
    }
}

impl Grid<f32> {
    /// Bilinear lookup with `u`, `v` in `[0, 1]` across the raster.
    ///
    /// Cell centres sit at half-integer positions and edges clamp, which is
    /// exactly how the GPU samples the same data as a texture.
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let x = u * self.width as f32 - 0.5;
        let y = v * self.height as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as isize, y0 as isize);
        let south = self.get_clamped(x0, y0) * (1.0 - fx) + self.get_clamped(x0 + 1, y0) * fx;
        let north = self.get_clamped(x0, y0 + 1) * (1.0 - fx) + self.get_clamped(x0 + 1, y0 + 1) * fx;
        south * (1.0 - fy) + north * fy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_hits_cell_centres_and_clamps() {
        let grid = Grid::from_vec(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(grid.sample(0.25, 0.25), 0.0);
        assert_eq!(grid.sample(0.75, 0.75), 3.0);
        assert_eq!(grid.sample(0.5, 0.5), 1.5);
        assert_eq!(grid.sample(-1.0, 2.0), 2.0);
    }

    #[test]
    fn from_fn_is_row_major() {
        let grid = Grid::from_fn(3, 2, |x, y| x + 10 * y);
        assert_eq!(grid.data(), &[0, 1, 2, 10, 11, 12]);
    }
}
