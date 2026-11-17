
// ---------------------------------------------------------------------------
// Heights, sampled with the filtering of 32-bit floats that the GPU offers.

fn streamed_height(xy: vec2<f32>, level: i32) -> f32 {
    return textureSampleLevel(streamed_height_tex, repeat_sampler, streamed_uv(xy, level), streamed_layer(level), 0.0).r;
}

fn whole_height(uv: vec2<f32>, pyramid: i32, mip: f32) -> f32 {
    return textureSampleLevel(whole_height_tex, clamp_sampler, uv, pyramid, mip).r;
}
