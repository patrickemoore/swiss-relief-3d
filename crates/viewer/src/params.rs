//! Every tunable of the shading model, with the plan's defaults.
//!
//! One [`ShadingParams`] fully describes how a frame is shaded. The evaluation
//! conditions (A to E) and the ablations are just named edits of it, and a
//! preset is the same struct saved as JSON.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Directory user presets are saved to.
pub const PRESET_DIR: &str = "presets";

/// How terrain brightness is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShadingMode {
    /// The full camera-relative Swiss model.
    Swiss,
    /// Swiss relief at one fixed map scale, as if baked top-down and draped
    /// over the terrain as a texture.
    Draped,
    /// Physically motivated baseline: diffuse sun, cast shadows.
    Realistic,
    /// Standard single-light GIS hillshade.
    Hillshade,
}

impl ShadingMode {
    pub const ALL: [Self; 4] = [Self::Swiss, Self::Draped, Self::Realistic, Self::Hillshade];

    pub fn label(self) -> &'static str {
        match self {
            Self::Swiss => "Swiss, camera-relative",
            Self::Draped => "Swiss, draped 2D relief",
            Self::Realistic => "Realistic sun and shadows",
            Self::Hillshade => "GIS hillshade",
        }
    }
}

/// Which frame the base light is defined in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightMode {
    /// Relative to the camera, trailing it by the lag time constant.
    CameraRelative,
    /// Fixed in the world: the map convention, seen north-up from above.
    WorldFixed,
}

/// Diagnostic renderings of the model's intermediate quantities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum DebugView {
    None = 0,
    GeneralizationLevel = 1,
    LightRotation = 2,
    Orientation = 3,
    Detail0 = 4,
    Detail1 = 5,
    Detail2 = 6,
    Detail3 = 7,
    Landform = 8,
    Transmittance = 9,
    GreyRelief = 10,
    MeshLod = 11,
    Water = 12,
    StreamedLevel = 13,
}

impl DebugView {
    pub const ALL: [Self; 14] = [
        Self::None,
        Self::GeneralizationLevel,
        Self::LightRotation,
        Self::Orientation,
        Self::Detail0,
        Self::Detail1,
        Self::Detail2,
        Self::Detail3,
        Self::Landform,
        Self::Transmittance,
        Self::GreyRelief,
        Self::MeshLod,
        Self::Water,
        Self::StreamedLevel,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "Off",
            Self::GeneralizationLevel => "Generalization level l0",
            Self::LightRotation => "Local light rotation",
            Self::Orientation => "Landform orientation",
            Self::Detail0 => "Detail term 0 (finest)",
            Self::Detail1 => "Detail term 1",
            Self::Detail2 => "Detail term 2",
            Self::Detail3 => "Detail term 3",
            Self::Landform => "Landform term",
            Self::Transmittance => "Haze transmittance",
            Self::GreyRelief => "Grey relief",
            Self::MeshLod => "Mesh level of detail",
            Self::Water => "Lake mask",
            Self::StreamedLevel => "Streamed terrain level",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShadingParams {
    pub mode: ShadingMode,

    // 6.1 Camera-relative base light.
    pub light_mode: LightMode,
    /// How far left of screen-up the light sits (theta_s).
    pub light_screen_angle_deg: f32,
    /// Light angle above the image plane (epsilon).
    pub light_elevation_deg: f32,
    /// Time constant of the light's lag behind the camera (tau); 0 is a
    /// rigid headlight.
    pub light_lag_s: f32,

    // 6.2 Local light adjustment.
    pub local_light: bool,
    pub local_max_rotation_deg: f32,
    pub local_ok_angle_deg: f32,
    /// Standard deviation of the rotation blur, about one landform wide.
    pub local_blur_m: f32,
    pub local_hysteresis_deg: f32,
    /// Softness of the majority sign, the epsilon in `tanh(blur / epsilon)`.
    pub local_sign_softness_deg: f32,

    // 6.3 Multi-scale shading and view-dependent generalization.
    pub generalization: bool,
    pub generalization_bias: f32,
    pub landform_level: f32,
    pub landform_weight: f32,
    /// Weight of the finest detail term (k0).
    pub detail_weight: f32,
    /// Each coarser detail term is weighted by this factor again.
    pub detail_falloff: f32,
    /// Detail exaggeration (a).
    pub detail_gain: f32,
    pub shading_exaggeration: f32,
    pub geometry_exaggeration: f32,
    pub flat_brightness: f32,
    pub contrast: f32,
    /// Derive the pixel footprint from distance and slope instead of
    /// screen-space derivatives, which jump at silhouettes.
    pub analytic_footprint: bool,
    /// Use the ridge-accentuated pyramids (4.3).
    pub accentuation: bool,

    // 7.1 Aerial perspective.
    pub elevation_contrast: bool,
    pub valley_tone: f32,
    pub low_contrast: f32,
    pub contrast_elevation_lo: f32,
    pub contrast_elevation_hi: f32,
    pub fog: bool,
    /// Haze density at the lowland level, per kilometre (beta_0).
    pub fog_density_per_km: f32,
    pub fog_scale_height_m: f32,
    /// Cap on `1 - T`, so the horizon keeps some form.
    pub fog_max: f32,

    // 7.2 Colour.
    pub color: bool,
    pub water: bool,

    // 7.3 Depth cues.
    pub depth_edges: bool,
    pub depth_edge_strength: f32,
    /// Blur radius of the depth unsharp mask, as a fraction of image height.
    pub depth_edge_radius: f32,
    pub silhouettes: bool,

    // Realistic baseline.
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub sun_ambient: f32,
    pub shadows: bool,

    /// sRGB colours of the haze (also the horizon) and the zenith.
    pub haze_color: [f32; 3],
    pub zenith_color: [f32; 3],
    /// Width over which the terrain dissolves at the edge of its data, in
    /// metres.
    pub edge_fade_m: f32,

    pub debug_view: DebugView,
}

impl Default for ShadingParams {
    fn default() -> Self {
        Self {
            mode: ShadingMode::Swiss,
            light_mode: LightMode::CameraRelative,
            light_screen_angle_deg: 45.0,
            light_elevation_deg: 45.0,
            light_lag_s: 1.0,
            local_light: true,
            local_max_rotation_deg: 35.0,
            local_ok_angle_deg: 45.0,
            local_blur_m: 640.0,
            local_hysteresis_deg: 10.0,
            local_sign_softness_deg: 4.0,
            generalization: true,
            generalization_bias: 1.5,
            landform_level: 7.0,
            landform_weight: 0.6,
            detail_weight: 0.25,
            detail_falloff: 0.6,
            detail_gain: 3.0,
            shading_exaggeration: 1.5,
            geometry_exaggeration: 1.2,
            flat_brightness: 0.75,
            contrast: 1.0,
            analytic_footprint: true,
            accentuation: true,
            elevation_contrast: true,
            valley_tone: 0.65,
            // 0.4 on its own; raised because height fog reduces lowland
            // contrast a second time.
            low_contrast: 0.7,
            contrast_elevation_lo: 0.0,
            contrast_elevation_hi: 0.7,
            fog: true,
            fog_density_per_km: 0.08,
            fog_scale_height_m: 1200.0,
            fog_max: 0.6,
            color: true,
            water: true,
            depth_edges: true,
            depth_edge_strength: 0.15,
            depth_edge_radius: 0.012,
            silhouettes: false,
            sun_azimuth_deg: 150.0,
            sun_elevation_deg: 40.0,
            sun_ambient: 0.25,
            shadows: true,
            haze_color: [0.78, 0.83, 0.89],
            zenith_color: [0.55, 0.69, 0.86],
            edge_fade_m: 1500.0,
            debug_view: DebugView::None,
        }
    }
}

impl ShadingParams {
    /// The full method tuned for looks rather than for the evaluation.
    ///
    /// At a low camera pitch the default light sits behind the viewer's
    /// shoulder and lights most visible slopes frontally. A lower light from
    /// farther left models them, and with a darker flat tone there is room
    /// above it for the lit slopes. More contrast and less haze go with it.
    pub fn showcase() -> Self {
        Self {
            light_screen_angle_deg: 60.0,
            light_elevation_deg: 30.0,
            flat_brightness: 0.68,
            contrast: 1.3,
            low_contrast: 0.85,
            fog_density_per_km: 0.06,
            fog_max: 0.5,
            ..Self::default()
        }
    }

    /// Detail weights `k0..k3`, finest first.
    pub fn detail_weights(&self) -> [f32; 4] {
        std::array::from_fn(|i| self.detail_weight * self.detail_falloff.powi(i as i32))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, serde_json::to_string_pretty(self)?).with_context(|| format!("writing {}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}

/// Saved presets in [`PRESET_DIR`], sorted by name.
pub fn list_presets() -> Vec<PathBuf> {
    let mut presets: Vec<PathBuf> = fs::read_dir(PRESET_DIR)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "json"))
        .collect();
    presets.sort();
    presets
}

/// The evaluation conditions of plan section 10.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Condition {
    Realistic,
    Draped,
    WorldFixed,
    Full,
    Headlight,
}

impl Condition {
    pub const ALL: [Self; 5] = [Self::Realistic, Self::Draped, Self::WorldFixed, Self::Full, Self::Headlight];

    pub fn letter(self) -> char {
        match self {
            Self::Realistic => 'A',
            Self::Draped => 'B',
            Self::WorldFixed => 'C',
            Self::Full => 'D',
            Self::Headlight => 'E',
        }
    }

    pub fn from_letter(letter: char) -> Option<Self> {
        Self::ALL.into_iter().find(|condition| condition.letter() == letter.to_ascii_uppercase())
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Realistic => "A  Realistic",
            Self::Draped => "B  Draped 2D relief",
            Self::WorldFixed => "C  Ours, world-fixed light",
            Self::Full => "D  Ours, full",
            Self::Headlight => "E  Ours, headlight",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Realistic => {
                "Fixed sun, diffuse shading, cast shadows and physical haze: what a game engine would do."
            }
            Self::Draped => "Swiss relief at a fixed map scale with a north-west light, draped like a texture.",
            Self::WorldFixed => "The full model with a world-fixed north-west light (lag of infinity).",
            Self::Full => "The proposed method: camera-relative light that trails the camera by one second.",
            Self::Headlight => "The full model with a rigid camera light (lag of zero).",
        }
    }

    pub fn params(self) -> ShadingParams {
        let base = ShadingParams::default();
        match self {
            Self::Full => base,
            Self::Headlight => ShadingParams { light_lag_s: 0.0, ..base },
            Self::WorldFixed => ShadingParams { light_mode: LightMode::WorldFixed, ..base },
            Self::Draped => ShadingParams {
                mode: ShadingMode::Draped,
                light_mode: LightMode::WorldFixed,
                // A draped map carries only the cartographic elevation term.
                low_contrast: 0.4,
                fog: false,
                depth_edges: false,
                ..base
            },
            Self::Realistic => ShadingParams {
                mode: ShadingMode::Realistic,
                light_mode: LightMode::WorldFixed,
                local_light: false,
                elevation_contrast: false,
                fog_max: 1.0,
                depth_edges: false,
                // One exaggeration for geometry and shading alike.
                shading_exaggeration: base.geometry_exaggeration,
                ..base
            },
        }
    }
}

/// One component of the full method switched off (plan section 10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ablation {
    LocalLight,
    Generalization,
    Accentuation,
    ElevationContrast,
    Fog,
    DepthEdges,
}

impl Ablation {
    pub const ALL: [Self; 6] = [
        Self::LocalLight,
        Self::Generalization,
        Self::Accentuation,
        Self::ElevationContrast,
        Self::Fog,
        Self::DepthEdges,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::LocalLight => "no_local_light",
            Self::Generalization => "no_generalization",
            Self::Accentuation => "no_accentuation",
            Self::ElevationContrast => "no_elevation_contrast",
            Self::Fog => "no_fog",
            Self::DepthEdges => "no_depth_edges",
        }
    }

    /// The full method with this component removed.
    pub fn params(self) -> ShadingParams {
        let mut params = Condition::Full.params();
        match self {
            Self::LocalLight => params.local_light = false,
            Self::Generalization => params.generalization = false,
            Self::Accentuation => params.accentuation = false,
            Self::ElevationContrast => params.elevation_contrast = false,
            Self::Fog => params.fog = false,
            Self::DepthEdges => params.depth_edges = false,
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_detail_weights_follow_the_plan() {
        let weights = ShadingParams::default().detail_weights();
        for (weight, expected) in weights.iter().zip([0.25, 0.15, 0.10, 0.05]) {
            assert!((weight - expected).abs() < 0.015, "{weights:?}");
        }
    }

    #[test]
    fn presets_round_trip_and_tolerate_missing_fields() {
        let params = Condition::Draped.params();
        let json = serde_json::to_string(&params).unwrap();
        assert_eq!(serde_json::from_str::<ShadingParams>(&json).unwrap(), params);

        let partial: ShadingParams = serde_json::from_str(r#"{ "light_lag_s": 2.5 }"#).unwrap();
        assert_eq!(partial, ShadingParams { light_lag_s: 2.5, ..Default::default() });
    }

    #[test]
    fn condition_letters_round_trip() {
        for condition in Condition::ALL {
            assert_eq!(Condition::from_letter(condition.letter()), Some(condition));
        }
        assert_eq!(Condition::from_letter('d'), Some(Condition::Full));
        assert_eq!(Condition::from_letter('x'), None);
    }
}
