// The post pass: depth-edge enhancement, sky and output encoding
// (plan section 7.3). It only adds screen-space depth cues; no shading
// happens here. `frame.wgsl` is prepended to this file.

struct FullscreenOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One triangle that covers the viewport.
@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> FullscreenOut {
    let corner = vec2(f32((index << 1u) & 2u), f32(index & 2u));
    var out: FullscreenOut;
    out.position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2(corner.x, 1.0 - corner.y);
    return out;
}

// ---------------------------------------------------------------------------
// Separable Gaussian blur of log depth, at half resolution.

struct Blur {
    tap_step: vec2<f32>, // uv offset between taps
    sigma: f32,      // in taps
    radius: f32,     // in taps
};

@group(0) @binding(0) var<uniform> blur: Blur;
@group(0) @binding(1) var blur_source: texture_2d<f32>;
@group(0) @binding(2) var blur_sampler: sampler;

@fragment
fn fs_blur(in: FullscreenOut) -> @location(0) vec4<f32> {
    let radius = i32(blur.radius);
    var total = 0.0;
    var weights = 0.0;
    for (var i = -radius; i <= radius; i++) {
        let offset = f32(i) / blur.sigma;
        let weight = exp(-0.5 * offset * offset);
        total += weight * textureSampleLevel(blur_source, blur_sampler, in.uv + f32(i) * blur.tap_step, 0.0).r;
        weights += weight;
    }
    return vec4(total / weights, 0.0, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Composite.

// The settings of one side of the split screen.
struct Side {
    haze_color: vec4<f32>,   // linear rgb, also the sky's horizon colour
    zenith_color: vec4<f32>, // linear rgb
    edges: vec4<f32>,        // x: halo strength; y: silhouette strength
};

// Mirrors `CompositeUniform` in renderer/post.rs.
struct Composite {
    sides: array<Side, 2>,
    // x: pixel column where side 1 begins; y: 1 if the target is sRGB-encoded
    // by the hardware; z: 1 to draw the divider.
    options: vec4<f32>,
};

// A log-depth gap of this size gives the full halo strength. About 28 %
// farther than the neighbourhood.
const HALO_REFERENCE_GAP: f32 = 0.25;
const HALO_LIMIT: f32 = 0.5;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> composite: Composite;
@group(0) @binding(2) var color_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_2d<f32>;
@group(0) @binding(4) var blurred_tex: texture_2d<f32>;
@group(0) @binding(5) var linear_sampler: sampler;

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let low = c * 12.92;
    let high = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return select(high, low, c <= vec3(0.0031308));
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + 0.055) / 1.055, vec3(2.4));
    return select(high, low, c <= vec3(0.04045));
}

// A vertical gradient whose horizon equals the haze colour, so distant
// terrain fades into the sky behind it.
fn sky(side: Side, uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
    let ray = normalize(
        frame.cam_forward.xyz
        + ndc.x * frame.cam_right.w * frame.cam_right.xyz
        + ndc.y * frame.cam_up.w * frame.cam_up.xyz,
    );
    return mix(side.haze_color.rgb, side.zenith_color.rgb, 1.0 - exp(-3.0 * max(ray.z, 0.0)));
}

@fragment
fn fs_composite(in: FullscreenOut) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    let side = composite.sides[u32(in.position.x >= composite.options.x)];
    let scene = textureLoad(color_tex, pixel, 0);

    var color = sky(side, in.uv);
    if (scene.a > 0.0) {
        // Unsharp masking of the depth buffer (Luft et al. 2006), on log
        // depth so the same relative gap gives the same halo at any
        // distance. Pixels that lie behind their neighbourhood darken: the
        // far terrain just behind a ridge gets a soft halo that separates
        // the ridge from it.
        let depth = textureLoad(depth_tex, pixel, 0).r;
        let blurred = textureSampleLevel(blurred_tex, linear_sampler, in.uv, 0.0).r;
        var darkening = side.edges.x * max(depth - blurred, 0.0) / HALO_REFERENCE_GAP;

        // Optional thin silhouette lines on the far side of large jumps.
        if (side.edges.y > 0.0) {
            let last = vec2<i32>(textureDimensions(depth_tex)) - 1;
            var jump = 0.0;
            for (var i = 0; i < 4; i++) {
                let offset = vec2(i32(i == 0) - i32(i == 1), i32(i == 2) - i32(i == 3));
                jump = max(jump, depth - textureLoad(depth_tex, clamp(pixel + offset, vec2(0), last), 0).r);
            }
            darkening += side.edges.y * smoothstep(0.03, 0.12, jump);
        }
        // The alpha is what is left of the terrain where it dissolves at the
        // edge of its data.
        color = mix(color, scene.rgb * (1.0 - min(darkening, HALO_LIMIT)), scene.a);
    }

    if (composite.options.z > 0.5 && abs(in.position.x - composite.options.x) < 1.0) {
        color = vec3(0.02);
    }

    // Dither in display space to keep the sky and haze gradients free of
    // banding, then hand the value back in the encoding the target expects.
    let noise = fract(52.9829189 * fract(dot(in.position.xy, vec2(0.06711056, 0.00583715))));
    let encoded = clamp(linear_to_srgb(color) + (noise - 0.5) / 255.0, vec3(0.0), vec3(1.0));
    if (composite.options.y > 0.5) {
        return vec4(srgb_to_linear(encoded), 1.0);
    }
    return vec4(encoded, 1.0);
}
