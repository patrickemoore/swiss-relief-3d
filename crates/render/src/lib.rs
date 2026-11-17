//! Real-time 3D terrain with camera-relative Swiss-style relief shading.
//!
//! This is the reusable part: the renderer and what steers it (the camera,
//! the light that follows it, the shading parameters, the places and the
//! map labels). It has no window, no interface and no opinion about where
//! it runs: `relief-viewer` puts a desktop window around it and `relief-web`
//! a JavaScript API in the browser.

pub mod camera;
pub mod explore;
pub mod gpu;
pub mod labels;
pub mod light_rig;
pub mod params;
pub mod path;
pub mod renderer;
