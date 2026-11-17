//! The Swiss LV95 grid (EPSG:2056).

/// Converts LV95 easting and northing to WGS84 `(longitude, latitude)` in
/// degrees, using swisstopo's approximate formulas (accurate to about 1 m).
pub fn to_wgs84(east: f64, north: f64) -> (f64, f64) {
    let y = (east - 2_600_000.0) / 1_000_000.0;
    let x = (north - 1_200_000.0) / 1_000_000.0;
    let longitude = 2.677_909_4 + 4.728_982 * y + 0.791_484 * y * x + 0.130_6 * y * x * x - 0.043_6 * y.powi(3);
    let latitude = 16.902_389_2 + 3.238_272 * x
        - 0.270_978 * y * y
        - 0.002_528 * x * x
        - 0.044_7 * y * y * x
        - 0.014_0 * x.powi(3);
    (longitude * 100.0 / 36.0, latitude * 100.0 / 36.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bern_reference_point() {
        // The projection centre: the old observatory in Bern.
        let (longitude, latitude) = to_wgs84(2_600_000.0, 1_200_000.0);
        assert!((longitude - 7.438_63).abs() < 1e-4);
        assert!((latitude - 46.951_08).abs() < 1e-4);
    }
}
