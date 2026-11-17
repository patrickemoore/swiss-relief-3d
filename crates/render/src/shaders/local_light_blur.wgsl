// Local light adjustment, steps 2 and 3 of 3 (plan section 6.2).
//
// No continuous rule can keep every orientation away from the bad one, so
// somewhere the rotation must switch from one way to the other. A plain blur
// of the signed rotation would average +delta_max and -delta_max to zero on
// exactly the landforms the rule exists for. Sign and magnitude are therefore
// blurred separately: the magnitude with a Gaussian about one landform wide,
// while the sign is taken from the blurred signed rotation, so a whole
// landform rotates one way by majority.

struct LocalLight {
    base_azimuth: f32,
    delta_max: f32,
    theta_ok: f32,
    hysteresis: f32,
    sign_softness: f32,   // epsilon in tanh(blur / epsilon)
    blur_sigma: f32,      // cells
    orientation_level: f32,
    pad: f32,
    field: vec4<f32>,
    moved: vec2<i32>,
    pad1: vec2<i32>,
};

@group(0) @binding(0) var<uniform> params: LocalLight;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

// Gaussian-weighted mean of the source's (signed rotation, magnitude).
fn blur(cell: vec2<i32>, direction: vec2<i32>) -> vec2<f32> {
    let last = vec2<i32>(textureDimensions(source)) - 1;
    let sigma = max(params.blur_sigma, 0.01);
    let radius = i32(ceil(2.5 * sigma));
    var total = vec2(0.0);
    var weights = 0.0;
    for (var i = -radius; i <= radius; i++) {
        let offset = f32(i) / sigma;
        let weight = exp(-0.5 * offset * offset);
        total += weight * textureLoad(source, clamp(cell + i * direction, vec2(0), last), 0).rg;
        weights += weight;
    }
    return total / weights;
}

@compute @workgroup_size(8, 8)
fn blur_x(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= textureDimensions(output))) {
        return;
    }
    textureStore(output, id.xy, vec4(blur(vec2<i32>(id.xy), vec2(1, 0)), 0.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn blur_y(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= textureDimensions(output))) {
        return;
    }
    let blurred = blur(vec2<i32>(id.xy), vec2(0, 1));
    let delta = tanh(blurred.x / params.sign_softness) * blurred.y;
    textureStore(output, id.xy, vec4(delta, blurred.x, blurred.y, 1.0));
}
