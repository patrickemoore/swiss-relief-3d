
// ---------------------------------------------------------------------------
// Hiking trails: segments expanded into screen-space capsules, draped on the
// terrain the shader above samples. Appended after terrain.wgsl, whose
// bindings and functions it shares.

struct TrailStyle {
    style: vec4<f32>,  // core width, casing width (CSS px), device pixels per CSS px, 0 for the casing pass or 1 for the core
    fade: vec4<f32>,   // hiking fade start and end, upland fade start and end (m per CSS px)
    colors: array<vec4<f32>, 4>, // casing, hiking, mountain, alpine
};

@group(2) @binding(0) var<uniform> trail: TrailStyle;

// A line sits a little above the surface and is pulled toward the eye, so
// that the mesh drawn at a coarser level than the heights sampled here does
// not swallow it, while a ridge well in front of it still does.
const TRAIL_LIFT_M: f32 = 0.8;
const TRAIL_LIFT_FRACTION: f32 = 0.004;

struct TrailIn {
    @builtin(vertex_index) vertex: u32,
    @location(0) a: vec2<f32>,
    @location(1) b: vec2<f32>,
    @location(2) kind: u32,
    @location(3) level: u32,
};

struct TrailOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) ends: vec4<f32>, // both ends in device pixels, y down
    @location(1) @interpolate(flat) kind: u32,
    @location(2) world: vec3<f32>,
    @location(3) @interpolate(flat) scale: f32, // ground metres per CSS pixel at the segment
};

fn draped(xy: vec2<f32>) -> vec3<f32> {
    // The mesh is drawn at a level that grows with distance; sampling the
    // same level puts the line on the surface that is actually there.
    let finest = finest_level(xy);
    let camera_distance = length(vec3(xy - frame.cam_pos.xy, frame.cam_pos.z - frame.lod.z));
    let mesh_level = clamp(log2(max(camera_distance / frame.lod.x, 1.0)), 0.0, frame.terrain.w);
    let height = terrain_height(xy, mesh_level, finest) * shading.geometry_exaggeration;
    let p = vec3(xy, height);
    let to_eye = frame.cam_pos.xyz - p;
    let d = length(to_eye);
    return p + to_eye / max(d, 1e-3) * (TRAIL_LIFT_M + TRAIL_LIFT_FRACTION * d);
}

fn to_pixels(clip: vec4<f32>) -> vec2<f32> {
    return (clip.xy / clip.w * vec2(0.5, -0.5) + 0.5) * frame.viewport.xy;
}

@vertex
fn vs_trail(in: TrailIn) -> TrailOut {
    var out: TrailOut;
    let wa = draped(in.a);
    let wb = draped(in.b);
    let ca = frame.view_proj * vec4(wa, 1.0);
    let cb = frame.view_proj * vec4(wb, 1.0);
    // Behind the camera: the quad collapses off screen.
    if (ca.w <= 0.05 || cb.w <= 0.05) {
        out.clip = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }
    let pa = to_pixels(ca);
    let pb = to_pixels(cb);
    var dir = pb - pa;
    let len = length(dir);
    if (len > 1e-3) {
        dir = dir / len;
    } else {
        dir = vec2(1.0, 0.0);
    }
    let nrm = vec2(-dir.y, dir.x);
    let ratio = trail.style.z;
    // Core, casing on both sides, and a pixel for the anti-aliasing.
    let half = 0.5 * trail.style.x * ratio + trail.style.y * ratio + 1.0;

    // Two triangles: a-, b-, b+, a-, b+, a+.
    let at_b = in.vertex == 1u || in.vertex == 2u || in.vertex == 4u;
    let side = select(-1.0, 1.0, in.vertex == 2u || in.vertex == 4u || in.vertex == 5u);
    let along = select(-1.0, 1.0, at_b);
    let p = select(pa, pb, at_b) + dir * half * along + nrm * half * side;
    let c = select(ca, cb, at_b);
    let ndc = (p * frame.viewport.zw - 0.5) * vec2(2.0, -2.0);
    out.clip = vec4(ndc * c.w, c.z, c.w);
    out.ends = vec4(pa, pb);
    out.kind = in.kind;
    out.world = select(wa, wb, at_b);
    out.scale = frame.cam_pos.w * distance(frame.cam_pos.xyz, 0.5 * (wa + wb)) * ratio;
    return out;
}

fn distance_to_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    return distance(p, a + ab * t);
}

@fragment
fn fs_trail(in: TrailOut) -> @location(0) vec4<f32> {
    let ratio = trail.style.z;
    let d = distance_to_segment(in.clip.xy, in.ends.xy, in.ends.zw);
    let core = 0.5 * trail.style.x * ratio;
    let edge = core + trail.style.y * ratio;
    // Casings of every line go down first, cores after, so a joint between
    // two segments is not a casing painted over a core.
    var coverage: f32;
    var color: vec3<f32>;
    if (trail.style.w < 0.5) {
        coverage = clamp(edge + 0.5 - d, 0.0, 1.0);
        color = trail.colors[0].rgb;
    } else {
        coverage = clamp(core + 0.5 - d, 0.0, 1.0);
        color = trail.colors[min(in.kind, 3u)].rgb;
    }

    // Each kind comes in as the view closes in; everything thins with the
    // haze, like the ground it lies on.
    var fade: f32;
    if (in.kind == 1u) {
        fade = 1.0 - smoothstep(trail.fade.x, trail.fade.y, in.scale);
    } else {
        fade = 1.0 - smoothstep(trail.fade.z, trail.fade.w, in.scale);
    }
    var haze = 1.0;
    if (enabled(FLAG_FOG)) {
        haze = fog_transmittance(in.world);
    }
    let alpha = coverage * fade * mix(0.35, 1.0, haze);
    if (alpha <= 0.002) {
        discard;
    }
    return vec4(color, alpha);
}
