//! Camera-independent terrain analysis for Swiss-style relief shading.
//!
//! Everything in this crate runs once, offline. It turns a digital elevation
//! model into what the renderer samples every frame:
//!
//! * a generalized height pyramid, plain and ridge-accentuated ([`pyramid`]),
//! * a landform orientation field ([`orientation`]),
//! * a lake mask ([`water`]) and a colour lookup table ([`lut`]).
//!
//! A small terrain is analyzed and kept whole ([`dataset`]); a large one is
//! cut into tiles ([`tiled`]), and either can be packed for the web
//! ([`pack`]). The renderer sees all of them as a [`Terrain`].
//!
//! Conventions shared by the whole project: rasters are stored with row 0 at
//! the **southern** edge, so `x` grows east and `y` grows north, and heights
//! are metres. Positions are local metres from the south-west corner; the
//! LV95 coordinate of that corner lives in [`dataset::Meta`].

pub mod dataset;
pub mod fill;
pub mod filter;
pub mod grid;
pub mod lut;
pub mod lv95;
pub mod oklab;
pub mod orientation;
pub mod pack;
pub mod pyramid;
pub mod stats;
pub mod synthetic;
pub mod terrain;
pub mod tiled;
pub mod water;

pub use dataset::{BuildOptions, Dataset, Meta};
pub use grid::Grid;
pub use lut::Lut;
pub use stats::ElevationStats;
pub use terrain::{Terrain, Tile, TileSource};
