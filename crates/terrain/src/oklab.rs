//! sRGB and OKLab conversions.
//!
//! The colour table is built in OKLab because it interpolates without the
//! muddy midtones of sRGB blending (Björn Ottosson, 2020).

pub type Rgb = [f32; 3];
pub type Lab = [f32; 3];

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

pub fn linear_to_oklab([r, g, b]: Rgb) -> Lab {
    let l = (0.412_221_47 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

pub fn oklab_to_linear([lightness, a, b]: Lab) -> Rgb {
    let l = (lightness + 0.396_337_78 * a + 0.215_803_76 * b).powi(3);
    let m = (lightness - 0.105_561_346 * a - 0.063_854_17 * b).powi(3);
    let s = (lightness - 0.089_484_18 * a - 1.291_485_5 * b).powi(3);
    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_93 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_4 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

/// OKLab from an 8-bit sRGB triple such as `0xb4c4ae`.
pub fn oklab_from_hex(hex: u32) -> Lab {
    let channel = |shift: u32| srgb_to_linear(((hex >> shift) & 0xff) as f32 / 255.0);
    linear_to_oklab([channel(16), channel(8), channel(0)])
}

/// 8-bit sRGB from OKLab, clipped to the gamut.
pub fn oklab_to_srgb8(lab: Lab) -> [u8; 3] {
    oklab_to_linear(lab).map(|c| (linear_to_srgb(c.clamp(0.0, 1.0)) * 255.0).round() as u8)
}

pub fn mix(a: Lab, b: Lab, t: f32) -> Lab {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_oklab() {
        for hex in [0x000000, 0xffffff, 0xb4c4ae, 0x4a5878, 0xff8000] {
            let [r, g, b] = oklab_to_srgb8(oklab_from_hex(hex));
            let back = (r as u32) << 16 | (g as u32) << 8 | b as u32;
            assert_eq!(back, hex);
        }
    }

    #[test]
    fn white_has_unit_lightness_and_no_chroma() {
        let [l, a, b] = linear_to_oklab([1.0, 1.0, 1.0]);
        assert!((l - 1.0).abs() < 1e-3 && a.abs() < 1e-3 && b.abs() < 1e-3);
    }
}
