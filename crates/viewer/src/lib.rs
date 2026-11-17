//! Real-time 3D terrain with camera-relative Swiss-style relief shading.
//!
//! The library is the renderer and what steers it: the camera, the light
//! that follows it, the shading parameters. It builds for the desktop and,
//! as WebAssembly, for the browser, where the `relief-web` crate wraps it.
//!
//! The desktop application sits on top: the window and its panel, frame
//! dumps and the offline evaluation. Those modules exist only natively.

pub mod camera;
pub mod explore;
pub mod gpu;
pub mod labels;
pub mod light_rig;
pub mod params;
pub mod path;
pub mod renderer;

#[cfg(not(target_arch = "wasm32"))]
pub mod app;
#[cfg(not(target_arch = "wasm32"))]
pub mod capture;
#[cfg(not(target_arch = "wasm32"))]
pub mod evaluate;
#[cfg(not(target_arch = "wasm32"))]
pub mod headless;
#[cfg(not(target_arch = "wasm32"))]
pub mod session;
#[cfg(not(target_arch = "wasm32"))]
pub mod ui;
