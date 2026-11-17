//! Image similarity for the top-down fidelity check (plan section 10, H1).
//!
//! Compares a top-down render with a reference relief, for example
//! swisstopo's hand-painted one registered to the same area.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};

pub struct Similarity {
    /// Mean structural similarity over 8 x 8 windows, 1 for identical images.
    pub ssim: f64,
    /// Pearson correlation of the grey values.
    pub correlation: f64,
}

/// A greyscale image with values in `0..=255`.
struct Grey {
    width: usize,
    height: usize,
    values: Vec<f64>,
}

pub fn compare(a: &Path, b: &Path) -> Result<Similarity> {
    let (a, b) = (load_grey(a)?, load_grey(b)?);
    ensure!(
        a.width == b.width && a.height == b.height,
        "images differ in size: {} x {} and {} x {}",
        a.width,
        a.height,
        b.width,
        b.height
    );
    Ok(Similarity { ssim: mean_ssim(&a, &b), correlation: correlation(&a.values, &b.values) })
}

fn load_grey(path: &Path) -> Result<Grey> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = png::Decoder::new(BufReader::new(file)).read_info()?;
    let mut buffer = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buffer)?;
    ensure!(info.bit_depth == png::BitDepth::Eight, "{} is not an 8-bit PNG", path.display());

    let pixels = &buffer[..info.buffer_size()];
    let luma = |r: u8, g: u8, b: u8| 0.2126 * r as f64 + 0.7152 * g as f64 + 0.0722 * b as f64;
    let values = match info.color_type {
        png::ColorType::Grayscale => pixels.iter().map(|&grey| grey as f64).collect(),
        png::ColorType::GrayscaleAlpha => pixels.as_chunks::<2>().0.iter().map(|&[grey, _]| grey as f64).collect(),
        png::ColorType::Rgb => pixels.as_chunks::<3>().0.iter().map(|&[r, g, b]| luma(r, g, b)).collect(),
        png::ColorType::Rgba => pixels.as_chunks::<4>().0.iter().map(|&[r, g, b, _]| luma(r, g, b)).collect(),
        png::ColorType::Indexed => bail!("{} is palette-based; convert it to RGB first", path.display()),
    };
    Ok(Grey { width: info.width as usize, height: info.height as usize, values })
}

fn mean_ssim(a: &Grey, b: &Grey) -> f64 {
    const WINDOW: usize = 8;
    const STRIDE: usize = 4;
    const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0);

    let (mut total, mut windows) = (0.0, 0usize);
    for y in (0..a.height.saturating_sub(WINDOW - 1)).step_by(STRIDE) {
        for x in (0..a.width.saturating_sub(WINDOW - 1)).step_by(STRIDE) {
            let (mut sum_a, mut sum_b, mut sum_aa, mut sum_bb, mut sum_ab) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for row in y..y + WINDOW {
                for i in row * a.width + x..row * a.width + x + WINDOW {
                    let (va, vb) = (a.values[i], b.values[i]);
                    sum_a += va;
                    sum_b += vb;
                    sum_aa += va * va;
                    sum_bb += vb * vb;
                    sum_ab += va * vb;
                }
            }
            let n = (WINDOW * WINDOW) as f64;
            let (mean_a, mean_b) = (sum_a / n, sum_b / n);
            let (var_a, var_b) = (sum_aa / n - mean_a * mean_a, sum_bb / n - mean_b * mean_b);
            let covariance = sum_ab / n - mean_a * mean_b;
            total += (2.0 * mean_a * mean_b + C1) * (2.0 * covariance + C2)
                / ((mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2));
            windows += 1;
        }
    }
    total / windows.max(1) as f64
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (mean_a, mean_b) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut covariance, mut var_a, mut var_b) = (0.0, 0.0, 0.0);
    for (va, vb) in a.iter().zip(b) {
        covariance += (va - mean_a) * (vb - mean_b);
        var_a += (va - mean_a).powi(2);
        var_b += (vb - mean_b).powi(2);
    }
    covariance / (var_a * var_b).sqrt().max(f64::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_image(invert: bool) -> Grey {
        let values = (0..32 * 32)
            .map(|i| {
                let v = ((i % 32) * 8) as f64;
                if invert {
                    255.0 - v
                } else {
                    v
                }
            })
            .collect();
        Grey { width: 32, height: 32, values }
    }

    #[test]
    fn identical_images_score_one() {
        let image = gradient_image(false);
        assert!((mean_ssim(&image, &image) - 1.0).abs() < 1e-9);
        assert!((correlation(&image.values, &image.values) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn inverted_images_are_anticorrelated() {
        let (a, b) = (gradient_image(false), gradient_image(true));
        assert!((correlation(&a.values, &b.values) + 1.0).abs() < 1e-9);
        assert!(mean_ssim(&a, &b) < 0.5);
    }
}
