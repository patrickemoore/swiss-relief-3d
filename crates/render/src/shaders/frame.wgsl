// Per-frame camera and terrain constants, shared by every pass.
// Mirrors `FrameUniform` in renderer/mod.rs. `TILED_LEVELS` is prepended by
// the renderer, from the constant the terrain library defines.
struct Frame {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,     // xyz: position in local metres; w: angle of one pixel, radians
    cam_right: vec4<f32>,   // xyz: right; w: tan(horizontal fov / 2)
    cam_up: vec4<f32>,      // xyz: up; w: tan(vertical fov / 2)
    cam_forward: vec4<f32>, // xyz: forward
    viewport: vec4<f32>,    // width, height, 1 / width, 1 / height
    terrain: vec4<f32>,     // extent (m), texels per side, cell size (m), coarsest pyramid level
    elevation: vec4<f32>,   // minimum, maximum, 5th percentile, 95th percentile (m)
    lod: vec4<f32>,         // range of mesh level 0 (m), morph start fraction, reference height (m), quads per patch
    light_field: vec4<f32>, // xy: south-west corner of the local light field (m); z: its side (m); w: its cells per side
    // For each streamed pyramid level, the part of it that is loaded:
    // west, south, east, north in metres.
    loaded: array<vec4<f32>, TILED_LEVELS>,
};
