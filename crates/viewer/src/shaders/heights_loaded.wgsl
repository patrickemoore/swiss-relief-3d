
// ---------------------------------------------------------------------------
// Heights, filtered here: for GPUs that cannot filter 32-bit floats, which
// is most phones. Four texels are loaded and blended, wrapping around a
// streamed window and stopping at the edge of a whole level, exactly as the
// samplers of the other variant do.

fn blend4(sw: f32, se: f32, nw: f32, ne: f32, f: vec2<f32>) -> f32 {
    return mix(mix(sw, se, f.x), mix(nw, ne, f.x), f.y);
}

fn streamed_height(xy: vec2<f32>, level: i32) -> f32 {
    let side = i32(WINDOW_TEXELS);
    let at = streamed_uv(xy, level) * WINDOW_TEXELS - 0.5;
    let base = floor(at);
    let lo = ((vec2<i32>(base) % side) + side) % side;
    let hi = (lo + 1) % side;
    let layer = streamed_layer(level);
    return blend4(
        textureLoad(streamed_height_tex, vec2(lo.x, lo.y), layer, 0).r,
        textureLoad(streamed_height_tex, vec2(hi.x, lo.y), layer, 0).r,
        textureLoad(streamed_height_tex, vec2(lo.x, hi.y), layer, 0).r,
        textureLoad(streamed_height_tex, vec2(hi.x, hi.y), layer, 0).r,
        at - base,
    );
}

fn whole_height_at(uv: vec2<f32>, pyramid: i32, mip: i32) -> f32 {
    let size = vec2<i32>(textureDimensions(whole_height_tex, mip));
    let at = uv * vec2<f32>(size) - 0.5;
    let base = floor(at);
    let lo = clamp(vec2<i32>(base), vec2(0), size - 1);
    let hi = clamp(vec2<i32>(base) + 1, vec2(0), size - 1);
    return blend4(
        textureLoad(whole_height_tex, vec2(lo.x, lo.y), pyramid, mip).r,
        textureLoad(whole_height_tex, vec2(hi.x, lo.y), pyramid, mip).r,
        textureLoad(whole_height_tex, vec2(lo.x, hi.y), pyramid, mip).r,
        textureLoad(whole_height_tex, vec2(hi.x, hi.y), pyramid, mip).r,
        at - base,
    );
}

fn whole_height(uv: vec2<f32>, pyramid: i32, mip: f32) -> f32 {
    let last = i32(textureNumLevels(whole_height_tex)) - 1;
    let lower = clamp(i32(floor(mip)), 0, last);
    let upper = min(lower + 1, last);
    return mix(whole_height_at(uv, pyramid, lower), whole_height_at(uv, pyramid, upper), clamp(mip - f32(lower), 0.0, 1.0));
}
