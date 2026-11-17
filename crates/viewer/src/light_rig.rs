//! The camera-relative base light and its lagged rotation (plan section 6.1).
//!
//! The Swiss convention of light from the upper left of the sheet is a light
//! defined relative to the viewer. Looking straight down with north up it is
//! the cartographic north-west light; tilting the camera carries the light
//! from "far upper left" toward "upper left behind the viewer", so visible
//! slopes stay lit.

use std::f32::consts::FRAC_PI_2;

use glam::Vec3;

use crate::camera::{wrap_angle, Basis, Pose};
use crate::params::{LightMode, ShadingParams};

/// Largest angle the light may trail the camera by. Without this clamp a
/// steady orbit settles at a lag of angular speed times tau, and a fast orbit
/// would swing the light toward the bottom of the screen and invert the
/// relief.
const MAX_LAG: f32 = std::f32::consts::FRAC_PI_4;

/// A light direction that follows the camera's heading and pitch with a lag.
///
/// Heading and pitch are smoothed separately, each along the shorter way
/// round, rather than slerping orientations: a slerp between two roll-free
/// orientations introduces roll, and a half turn makes its direction
/// ambiguous.
#[derive(Clone, Copy, Debug, Default)]
pub struct LightRig {
    heading: f32,
    pitch: f32,
    initialized: bool,
}

impl LightRig {
    /// Advances the rig by `dt` seconds and returns the unit vector toward
    /// the light, in world space.
    pub fn update(&mut self, camera: &Pose, params: &ShadingParams, dt: f32) -> Vec3 {
        // A light below the horizon would shade from underneath when the
        // camera looks up, so the rig never follows the camera past level.
        let camera_pitch = camera.pitch.clamp(0.0, FRAC_PI_2);

        match params.light_mode {
            // The map convention: the frame of a north-up, top-down camera.
            LightMode::WorldFixed => {
                self.heading = 0.0;
                self.pitch = FRAC_PI_2;
            }
            LightMode::CameraRelative => {
                if !self.initialized || params.light_lag_s <= 0.0 {
                    self.heading = camera.heading;
                    self.pitch = camera_pitch;
                } else {
                    let blend = 1.0 - (-dt / params.light_lag_s).exp();
                    self.heading += wrap_angle(camera.heading - self.heading) * blend;
                    self.pitch += (camera_pitch - self.pitch) * blend;
                }
                let heading_lag = wrap_angle(self.heading - camera.heading).clamp(-MAX_LAG, MAX_LAG);
                self.heading = wrap_angle(camera.heading + heading_lag);
                self.pitch = camera_pitch + (self.pitch - camera_pitch).clamp(-MAX_LAG, MAX_LAG);
            }
        }
        self.initialized = true;

        light_direction(
            &Basis::new(self.heading, self.pitch),
            params.light_screen_angle_deg.to_radians(),
            params.light_elevation_deg.to_radians(),
        )
    }

    /// Forgets the lag, so the next update snaps to the camera. Used when the
    /// camera jumps to a new viewpoint.
    pub fn reset(&mut self) {
        self.initialized = false;
    }
}

/// `L = cos(e) (-sin(t) r + cos(t) u) + sin(e) b` with `b` pointing back
/// toward the viewer: `t` left of screen-up, `e` above the image plane.
fn light_direction(basis: &Basis, screen_angle: f32, elevation: f32) -> Vec3 {
    let in_plane = -screen_angle.sin() * basis.right + screen_angle.cos() * basis.up;
    elevation.cos() * in_plane - elevation.sin() * basis.forward
}

/// The fixed sun of the realistic baseline. Azimuth is a compass angle.
pub fn sun_direction(params: &ShadingParams) -> Vec3 {
    let (azimuth, elevation) = (params.sun_azimuth_deg.to_radians(), params.sun_elevation_deg.to_radians());
    Vec3::new(elevation.cos() * azimuth.sin(), elevation.cos() * azimuth.cos(), elevation.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(heading_deg: f32, pitch_deg: f32) -> Pose {
        Pose { position: Vec3::ZERO, heading: heading_deg.to_radians(), pitch: pitch_deg.to_radians() }
    }

    fn elevation_deg(light: Vec3) -> f32 {
        light.z.asin().to_degrees()
    }

    #[test]
    fn top_down_north_up_is_the_swiss_convention() {
        // Light from the north-west, 45 degrees above the horizon.
        let light = LightRig::default().update(&pose(0.0, 90.0), &ShadingParams::default(), 0.016);
        assert!((light - Vec3::new(-0.5, 0.5, 0.5f32.sqrt())).length() < 1e-5);
        assert!((elevation_deg(light) - 45.0).abs() < 1e-3);
    }

    #[test]
    fn world_fixed_light_ignores_the_camera() {
        let params = ShadingParams { light_mode: LightMode::WorldFixed, ..Default::default() };
        let mut rig = LightRig::default();
        let reference = rig.update(&pose(0.0, 90.0), &params, 0.016);
        assert!((rig.update(&pose(137.0, 20.0), &params, 0.016) - reference).length() < 1e-6);
    }

    #[test]
    fn light_elevation_follows_the_pitch() {
        // 45 degrees looking down, about 30 at the horizon, up to 60 between.
        let params = ShadingParams { light_lag_s: 0.0, ..Default::default() };
        let elevation = |pitch| elevation_deg(LightRig::default().update(&pose(0.0, pitch), &params, 0.016));
        assert!((elevation(0.0) - 30.0).abs() < 0.1);
        assert!((elevation(45.0) - 58.6).abs() < 0.5);
        assert!((elevation(90.0) - 45.0).abs() < 0.1);
    }

    #[test]
    fn lag_is_clamped_and_settles() {
        let params = ShadingParams { light_lag_s: 1.0, ..Default::default() };
        let mut rig = LightRig::default();
        rig.update(&pose(0.0, 45.0), &params, 0.016);

        // A sudden half turn: the light may trail by at most 45 degrees.
        rig.update(&pose(180.0, 45.0), &params, 0.016);
        let lag = wrap_angle(rig.heading - 180f32.to_radians()).abs();
        assert!((lag - MAX_LAG).abs() < 1e-3, "lag {lag}");

        // Once the camera rests, the light settles into its ideal position.
        for _ in 0..600 {
            rig.update(&pose(180.0, 45.0), &params, 0.016);
        }
        assert!(wrap_angle(rig.heading - 180f32.to_radians()).abs() < 1e-3);
    }

    #[test]
    fn rig_takes_the_short_way_round() {
        let params = ShadingParams { light_lag_s: 1.0, ..Default::default() };
        let mut rig = LightRig::default();
        rig.update(&pose(170.0, 45.0), &params, 0.016);
        rig.update(&pose(-170.0, 45.0), &params, 0.1);
        // 170 -> 190 degrees, not back through zero.
        let heading = rig.heading.to_degrees().rem_euclid(360.0);
        assert!(heading > 170.0 && heading < 190.0, "heading {heading}");
    }
}
