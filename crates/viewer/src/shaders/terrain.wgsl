// The terrain pass: CDLOD vertex morphing and the whole shading model.
//
// This is the only place where shading happens (plan sections 6, 7.1, 7.2),
// so every component can be switched on and off here for comparisons.
// `frame.wgsl` is prepended to this file, and before it the constants
// `TILED_LEVELS` and `WINDOW_TEXELS`.

// Mirrors `ShadingUniform` in renderer/terrain.rs.
struct Shading {
    light: vec4<f32>,          // xyz: lagged camera-relative light, world space
    sun: vec4<f32>,            // xyz: fixed sun of the realistic baseline; w: ambient
    detail_weights: vec4<f32>, // k0..k3, finest first
    haze_color: vec4<f32>,     // linear rgb
    water_color: vec4<f32>,    // linear rgb; a: opacity
    landform_weight: f32,      // kL
    detail_gain: f32,          // a
    shade_exaggeration: f32,
    geometry_exaggeration: f32,
    bias: f32,                 // generalization bias b
    landform_level: f32,       // lL
    flat_brightness: f32,      // B_flat
    contrast: f32,             // gamma
    valley_tone: f32,          // B_mid
    low_contrast: f32,         // C_low
    contrast_h_lo: f32,
    contrast_h_hi: f32,
    fog_density: f32,          // beta_0, per metre at the lowland level
    fog_scale_height: f32,     // H, metres
    fog_max: f32,              // cap on 1 - T
    edge_fade: f32,            // metres
    delta_max: f32,            // largest local light rotation, for the debug view
    skirt_base: f32,           // world height of the skirt's lower edge
    pad0: f32,
    pad1: f32,
    flags: u32,
    mode: u32,
    debug_view: u32,
    footprint: u32,
    accentuated: u32,          // 1 to use the ridge-accentuated pyramid
    pad2: u32,
    pad3: u32,
    pad4: u32,
};

const FLAG_LOCAL_LIGHT: u32 = 1u;
const FLAG_GENERALIZATION: u32 = 2u;
const FLAG_ELEVATION_CONTRAST: u32 = 4u;
const FLAG_FOG: u32 = 8u;
const FLAG_COLOR: u32 = 16u;
const FLAG_WATER: u32 = 32u;
const FLAG_SHADOWS: u32 = 64u;

const MODE_SWISS: u32 = 0u;
const MODE_DRAPED: u32 = 1u;
const MODE_REALISTIC: u32 = 2u;
const MODE_HILLSHADE: u32 = 3u;

const FOOTPRINT_ANALYTIC: u32 = 1u;

// tan(15 degrees): slopes steeper than this count as relief in the metrics.
const SLOPE_THRESHOLD: f32 = 0.268;
// Texels over which a streamed level fades into the next-coarser one toward
// the edge of what is loaded.
const WINDOW_BLEND_TEXELS: f32 = 48.0;
// Cells over which the local light rotation fades out toward the edge of
// the field it is computed in.
const LIGHT_FIELD_BLEND_CELLS: f32 = 16.0;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var clamp_sampler: sampler;
@group(0) @binding(2) var repeat_sampler: sampler;
@group(0) @binding(3) var orientation_tex: texture_2d<f32>;
@group(0) @binding(4) var lut_tex: texture_2d<f32>;
@group(0) @binding(5) var whole_water_tex: texture_2d<f32>;
@group(0) @binding(6) var edge_tex: texture_2d<f32>;
// The streamed levels: one array layer per level and pyramid.
@group(0) @binding(7) var streamed_height_tex: texture_2d_array<f32>;
@group(0) @binding(8) var streamed_gradient_tex: texture_2d_array<f32>;
@group(0) @binding(9) var streamed_water_tex: texture_2d_array<f32>;
// The levels held whole, as mip chains; one array layer per pyramid.
@group(0) @binding(10) var whole_height_tex: texture_2d_array<f32>;
@group(0) @binding(11) var whole_gradient_tex: texture_2d_array<f32>;

@group(1) @binding(0) var<uniform> shading: Shading;
@group(1) @binding(1) var delta_tex: texture_2d<f32>;

// ---------------------------------------------------------------------------
// Sampling the pyramids.
//
// Levels below TILED_LEVELS are streamed: each keeps a window of tiles around
// the camera, addressed with wrap-around. The functions below sample any
// continuous level, like a trilinear texture lookup, and fall back to coarser
// levels where a streamed one is not loaded.

// The finest level loaded at a point, as a continuous value: each level
// fades into the next-coarser one before its window ends, so the transition
// is never a seam.
fn finest_level(xy: vec2<f32>) -> f32 {
    var level = f32(TILED_LEVELS);
    // A level is only usable where every coarser one is.
    var usable = 1.0;
    // The windows wrap around, so the outermost texel of the terrain would
    // filter against the far side of its window. The whole levels do not.
    let extent = frame.terrain.x;
    let border = min(min(xy.x, extent - xy.x), min(xy.y, extent - xy.y));
    for (var k = TILED_LEVELS - 1; k >= 0; k--) {
        let loaded = frame.loaded[k];
        let texel = frame.terrain.z * exp2(f32(k));
        let inside = min(min(xy.x - loaded.x, loaded.z - xy.x), min(xy.y - loaded.y, loaded.w - xy.y)) / texel;
        let faded = clamp((inside - 2.0) / WINDOW_BLEND_TEXELS, 0.0, 1.0);
        usable = min(usable, min(faded, clamp(border / texel - 1.0, 0.0, 1.0)));
        level = mix(level, f32(k), usable);
    }
    return level;
}

// The array layer of a streamed level. Level 0 is shared by both pyramids,
// then come the plain levels, then the accentuated ones.
fn streamed_layer(level: i32) -> i32 {
    if (shading.accentuated == 1u && level > 0) {
        return level + TILED_LEVELS - 1;
    }
    return level;
}

fn streamed_uv(xy: vec2<f32>, level: i32) -> vec2<f32> {
    return xy / (frame.terrain.z * exp2(f32(level)) * WINDOW_TEXELS);
}

// A continuous level split into the streamed level below it and the blend
// toward the level above. `whole` is the mip level in the whole textures, and
// is negative while the level is streamed.
struct LevelBlend {
    lower: i32,
    blend: f32,
    whole: f32,
};

fn level_blend(level: f32, finest: f32) -> LevelBlend {
    let l = max(level, finest);
    let lower = floor(l);
    return LevelBlend(i32(lower), l - lower, l - f32(TILED_LEVELS));
}

fn terrain_gradient(xy: vec2<f32>, level: f32, finest: f32) -> vec2<f32> {
    let at = level_blend(level, finest);
    let uv = xy / frame.terrain.x;
    let pyramid = i32(shading.accentuated);
    if (at.whole >= 0.0) {
        return textureSampleLevel(whole_gradient_tex, clamp_sampler, uv, pyramid, at.whole).rg;
    }
    let fine = textureSampleLevel(streamed_gradient_tex, repeat_sampler, streamed_uv(xy, at.lower), streamed_layer(at.lower), 0.0).rg;
    var coarse: vec2<f32>;
    if (at.lower + 1 < TILED_LEVELS) {
        coarse = textureSampleLevel(streamed_gradient_tex, repeat_sampler, streamed_uv(xy, at.lower + 1), streamed_layer(at.lower + 1), 0.0).rg;
    } else {
        coarse = textureSampleLevel(whole_gradient_tex, clamp_sampler, uv, pyramid, 0.0).rg;
    }
    return mix(fine, coarse, at.blend);
}

fn terrain_height(xy: vec2<f32>, level: f32, finest: f32) -> f32 {
    let at = level_blend(level, finest);
    let uv = xy / frame.terrain.x;
    let pyramid = i32(shading.accentuated);
    // `streamed_height` and `whole_height` come in two variants, of which
    // the renderer appends the one the GPU can run.
    if (at.whole >= 0.0) {
        return whole_height(uv, pyramid, at.whole);
    }
    let fine = streamed_height(xy, at.lower);
    var coarse: f32;
    if (at.lower + 1 < TILED_LEVELS) {
        coarse = streamed_height(xy, at.lower + 1);
    } else {
        coarse = whole_height(uv, pyramid, 0.0);
    }
    return mix(fine, coarse, at.blend);
}

// Lake coverage, 0 to 1.
fn terrain_water(xy: vec2<f32>, level: f32, finest: f32) -> f32 {
    let at = level_blend(level, finest);
    let uv = xy / frame.terrain.x;
    if (at.whole >= 0.0) {
        return textureSampleLevel(whole_water_tex, clamp_sampler, uv, at.whole).r;
    }
    let fine = textureSampleLevel(streamed_water_tex, repeat_sampler, streamed_uv(xy, at.lower), at.lower, 0.0).r;
    var coarse: f32;
    if (at.lower + 1 < TILED_LEVELS) {
        coarse = textureSampleLevel(streamed_water_tex, repeat_sampler, streamed_uv(xy, at.lower + 1), at.lower + 1, 0.0).r;
    } else {
        coarse = textureSampleLevel(whole_water_tex, clamp_sampler, uv, 0.0).r;
    }
    return mix(fine, coarse, at.blend);
}

// Normal of the terrain generalized at pyramid level `level`. Gradients are
// stored rather than normals because they average linearly under filtering.
fn normal_at(xy: vec2<f32>, level: f32, finest: f32, exaggeration: f32) -> vec3<f32> {
    return normalize(vec3(-exaggeration * terrain_gradient(xy, level, finest), 1.0));
}

// ---------------------------------------------------------------------------

struct VertexIn {
    @location(0) grid: vec3<f32>,  // xy: position in the patch, in quads; z: 1 on a skirt's lower edge
    @location(1) instance: vec4<f32>, // xy: south-west corner (m); z: side length (m); w: mesh level
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) mesh_level: f32,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let spacing = in.instance.z / frame.lod.w;
    let level = in.instance.w;
    var xy = in.instance.xy + in.grid.xy * spacing;

    // CDLOD morph. In the outer part of this level's range, odd vertices
    // slide onto their even neighbours, which form the next-coarser grid.
    // The distance metric ignores the vertex's own height so that it matches
    // the CPU's level selection exactly.
    let range = frame.lod.x * exp2(level);
    let camera_distance = length(vec3(xy - frame.cam_pos.xy, frame.cam_pos.z - frame.lod.z));
    let morph = clamp((camera_distance / range - frame.lod.y) / (1.0 - frame.lod.y), 0.0, 1.0);
    xy -= fract(in.grid.xy * 0.5) * 2.0 * spacing * morph;

    // Sample the pyramid at the continuous level. A fully morphed vertex
    // reads exactly what the coarser neighbour reads, so no cracks open, and
    // distant silhouettes are generalized consistently with the shading.
    let height = terrain_height(xy, level + morph, finest_level(xy));
    let z = mix(height * shading.geometry_exaggeration, shading.skirt_base, in.grid.z);

    var out: VertexOut;
    out.world = vec3(xy, z);
    out.clip = frame.view_proj * vec4(out.world, 1.0);
    out.mesh_level = level + morph;
    return out;
}

struct FragmentOut {
    @location(0) color: vec4<f32>,     // linear rgb; a = 1 marks terrain
    @location(1) log_depth: vec4<f32>, // r: natural log of the distance to the camera
    @location(2) metrics: vec4<f32>,   // r: grey value; g: steep-slope flag; b: flat-ground grey; a: terrain
};

fn enabled(flag: u32) -> bool {
    return (shading.flags & flag) != 0u;
}

fn rotate_z(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3(c * v.x - s * v.y, s * v.x + c * v.y, v.z);
}

fn color_table(brightness: f32, elevation: f32) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(lut_tex));
    let uv = (vec2(brightness, elevation) * (size - 1.0) + 0.5) / size;
    return textureSampleLevel(lut_tex, clamp_sampler, uv, 0.0).rgb;
}

// A grey value is a display brightness, so it goes through the display curve.
fn grey(value: f32) -> vec3<f32> {
    return vec3(pow(clamp(value, 0.0, 1.0), 2.2));
}

// `base + offset`, compressed smoothly toward black and white instead of
// clamped. Flat ground is light, so it leaves little room above it; with a
// hard clamp every slope that faces the light would end up the same white.
fn soft_clip(base: f32, offset: f32) -> f32 {
    let room = max(select(base, 1.0 - base, offset > 0.0), 1e-3);
    return base + room * tanh(offset / room);
}

// Section 6.2: the rotation of the light about the vertical axis for the
// landform at a point, from the field computed around the camera.
fn local_rotation(xy: vec2<f32>) -> f32 {
    let uv = (xy - frame.light_field.xy) / frame.light_field.z;
    let cells_inside = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)) * frame.light_field.w;
    let rotation = textureSampleLevel(delta_tex, clamp_sampler, uv, 0.0).r;
    return rotation * clamp(cells_inside / LIGHT_FIELD_BLEND_CELLS, 0.0, 1.0);
}

struct Relief {
    brightness: f32,
    // Detail terms 0..3 (finest first), then the landform term.
    terms: array<f32, 5>,
};

// Section 6.3. Five generalization levels spread evenly from the pixel scale
// `l0` to the landform scale. The landform term uses the plain light, so
// distant mountains still read as mountains. Each finer scale is lit with the
// light projected onto the tangent plane of the next-coarser surface, the
// exaggerated-shading trick: the light is grazing relative to the larger
// form and reveals detail on any slope.
fn swiss_relief(xy: vec2<f32>, l0: f32, finest: f32, light: vec3<f32>) -> Relief {
    let landform = max(shading.landform_level, l0);
    var normals: array<vec3<f32>, 5>;
    for (var i = 0; i < 5; i++) {
        normals[i] = normal_at(xy, l0 + 0.25 * f32(i) * (landform - l0), finest, shading.shade_exaggeration);
    }

    var relief: Relief;
    relief.terms[4] = shading.landform_weight * dot(normals[4], light);
    var response = relief.terms[4];
    for (var i = 0; i < 4; i++) {
        let coarser = normals[i + 1];
        let projected = light - dot(light, coarser) * coarser;
        // Soft normalization. Where the coarser surface faces the light
        // head-on, the projection shrinks to zero and its direction is
        // undefined; normalizing fully would spin the light around such
        // points and draw pinwheels. Dividing by max(|p|, 0.3) fades the
        // detail out there instead.
        let grazing = projected / max(length(projected), 0.3);
        relief.terms[i] = shading.detail_weights[i] * clamp(shading.detail_gain * dot(normals[i], grazing), -1.0, 1.0);
        response += relief.terms[i];
    }

    // Tone mapping relative to flat ground, whose response is the light's
    // vertical component: tilting the camera then neither brightens nor
    // darkens the whole image.
    let flat_response = shading.landform_weight * light.z;
    relief.brightness = soft_clip(shading.flat_brightness, shading.contrast * (response - flat_response));
    return relief;
}

// Section 7.1, distance term. Exponential height fog with density
// beta(h) = beta_0 exp(-h / H) has a closed-form optical depth along a ray.
fn fog_transmittance(world: vec3<f32>) -> f32 {
    // Heights above the lowland level, with the geometry exaggeration undone.
    let base = frame.elevation.z;
    let h_camera = frame.cam_pos.z / shading.geometry_exaggeration - base;
    let h_point = world.z / shading.geometry_exaggeration - base;
    let ray_length = distance(frame.cam_pos.xyz, world);

    // (1 - exp(-x)) / x is evaluated by its series for nearly level rays.
    let x = clamp((h_point - h_camera) / shading.fog_scale_height, -80.0, 80.0);
    var slab = 1.0 - 0.5 * x;
    if (abs(x) > 1e-3) {
        slab = (1.0 - exp(-x)) / x;
    }
    let optical_depth = shading.fog_density * exp(-h_camera / shading.fog_scale_height) * ray_length * slab;
    return max(exp(-optical_depth), 1.0 - shading.fog_max);
}

// Cast shadows for the realistic baseline: marches the height pyramid toward
// the sun with growing steps and a widening cone, which gives soft edges.
fn sun_visibility(world: vec3<f32>) -> f32 {
    let ceiling = frame.elevation.y * shading.geometry_exaggeration;
    var visibility = 1.0;
    var t = 30.0;
    for (var i = 0; i < 48; i++) {
        let point = world + shading.sun.xyz * t;
        let uv = point.xy / frame.terrain.x;
        if (point.z > ceiling || any(uv < vec2(0.0)) || any(uv > vec2(1.0))) {
            break;
        }
        let level = clamp(log2(0.04 * t / frame.terrain.z), 0.0, 6.0);
        let ground = terrain_height(point.xy, level, finest_level(point.xy)) * shading.geometry_exaggeration;
        visibility = min(visibility, smoothstep(-0.02 * t, 0.05 * t, point.z - ground));
        t *= 1.16;
    }
    return visibility;
}

fn ramp(t: f32) -> vec3<f32> {
    let stops = array<vec3<f32>, 5>(
        vec3(0.05, 0.10, 0.45),
        vec3(0.00, 0.55, 0.75),
        vec3(0.20, 0.75, 0.25),
        vec3(0.95, 0.85, 0.10),
        vec3(0.80, 0.10, 0.05),
    );
    let x = clamp(t, 0.0, 1.0) * 4.0;
    let i = min(u32(x), 3u);
    var palette = stops;
    return mix(palette[i], palette[i + 1u], x - f32(i));
}

// Blue for negative, white for zero, red for positive.
fn diverging(t: f32) -> vec3<f32> {
    let x = clamp(t, -1.0, 1.0);
    let cold = vec3(0.05, 0.25, 0.85);
    let warm = vec3(0.85, 0.12, 0.05);
    return mix(vec3(1.0), select(cold, warm, x > 0.0), abs(x));
}

fn hue(angle: f32) -> vec3<f32> {
    let k = vec3(0.0, 2.0943951, 4.1887902);
    return 0.5 + 0.5 * cos(angle - k);
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    let extent = frame.terrain.x;
    let cell = frame.terrain.z;
    let coarsest = frame.terrain.w;
    let xy = in.world.xy;
    let uv = xy / extent;
    // Screen-space derivatives must be taken before any branching.
    let derivative_footprint = frame.terrain.y * max(length(dpdx(uv)), length(dpdy(uv)));

    let to_camera = frame.cam_pos.xyz - in.world;
    let distance_to_camera = length(to_camera);
    let view = to_camera / distance_to_camera;

    // The terrain dissolves toward the edge of its data, whether that is a
    // border inside the terrain square or the square's own edge. Where
    // nothing is left of it, whatever lies behind shows instead. Skirts are
    // shaded like the surface they hang from, so they dissolve with it.
    let to_border = textureSampleLevel(edge_tex, clamp_sampler, uv, 0.0).r;
    let to_square_edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)) * extent;
    let presence = smoothstep(0.0, max(shading.edge_fade, 1.0), min(to_border, to_square_edge));
    if (presence <= 0.0) {
        discard;
    }

    var out: FragmentOut;
    out.log_depth = vec4(log(distance_to_camera), 0.0, 0.0, 1.0);
    let finest = finest_level(xy);

    // Section 6.3: the finest scale shown follows the pixel's footprint on
    // the terrain, in texels of pyramid level 0.
    var footprint = derivative_footprint;
    if (shading.footprint == FOOTPRINT_ANALYTIC) {
        // Derivatives jump at silhouettes, where a 2x2 pixel block straddles
        // a near ridge and terrain kilometres behind it. The analytic
        // footprint is pixel angle x distance / (n.v), using the geometric
        // normal at the landform level.
        let facing = dot(normal_at(xy, shading.landform_level, finest, shading.geometry_exaggeration), view);
        footprint = frame.cam_pos.w * distance_to_camera / max(facing, 0.2) / cell;
    }
    let pixel_level = clamp(log2(footprint), 0.0, coarsest);
    var l0 = clamp(log2(footprint) + shading.bias, 0.0, coarsest);
    if (shading.mode == MODE_DRAPED) {
        // A draped map is generalized once, for its own scale; farther away
        // it is merely minified.
        l0 = clamp(max(shading.bias, log2(footprint)), 0.0, coarsest);
    } else if (!enabled(FLAG_GENERALIZATION)) {
        l0 = 0.0;
    }

    let height = terrain_height(xy, l0, finest);
    let elevation = clamp((height - frame.elevation.z) / (frame.elevation.w - frame.elevation.z), 0.0, 1.0);

    // Section 6.2: the per-pixel light is the base light rotated about the
    // vertical axis by the local adjustment.
    let delta = local_rotation(xy);
    var light = shading.light.xyz;
    if (enabled(FLAG_LOCAL_LIGHT)) {
        light = rotate_z(light, delta);
    }

    let relief = swiss_relief(xy, l0, finest, light);
    var brightness = relief.brightness;
    var flat_brightness = shading.flat_brightness;
    if (shading.mode == MODE_HILLSHADE) {
        brightness = max(dot(normal_at(xy, pixel_level, finest, shading.shade_exaggeration), light), 0.0);
        flat_brightness = light.z;
    }

    // Section 7.1, elevation term: full contrast on the highest terrain,
    // compressed toward a medium tone in the lowlands.
    if (enabled(FLAG_ELEVATION_CONTRAST)) {
        let contrast = mix(shading.low_contrast, 1.0, smoothstep(shading.contrast_h_lo, shading.contrast_h_hi, elevation));
        brightness = mix(shading.valley_tone, brightness, contrast);
        flat_brightness = mix(shading.valley_tone, flat_brightness, contrast);
    }

    // Section 7.2: colour from brightness and elevation.
    var color = grey(brightness);
    if (enabled(FLAG_COLOR)) {
        color = color_table(brightness, elevation);
    }

    if (shading.mode == MODE_REALISTIC) {
        let normal = normal_at(xy, pixel_level, finest, shading.geometry_exaggeration);
        var direct = max(dot(normal, shading.sun.xyz), 0.0);
        if (enabled(FLAG_SHADOWS)) {
            direct *= sun_visibility(in.world);
        }
        // Exposed so that sunlit flat ground shows the table's flat colour.
        let ambient = shading.sun.w;
        let exposure = (ambient + (1.0 - ambient) * direct) / (ambient + (1.0 - ambient) * shading.sun.z);
        flat_brightness = shading.flat_brightness;
        brightness = clamp(flat_brightness * exposure, 0.0, 1.0);
        color = grey(flat_brightness) * exposure;
        if (enabled(FLAG_COLOR)) {
            color = color_table(flat_brightness, elevation) * exposure;
        }
    }

    if (enabled(FLAG_WATER)) {
        color = mix(color, shading.water_color.rgb, terrain_water(xy, pixel_level, finest) * shading.water_color.a);
    }

    var transmittance = 1.0;
    if (enabled(FLAG_FOG)) {
        transmittance = fog_transmittance(in.world);
    }
    color = mix(shading.haze_color.rgb, color, transmittance);

    switch shading.debug_view {
        case 1u: {
            color = ramp(l0 / 9.0);
        }
        case 2u: {
            color = diverging(delta / max(shading.delta_max, 1e-3));
        }
        case 3u: {
            // Hue is the fall-line axis, saturation the coherence.
            let orientation = textureSampleLevel(orientation_tex, clamp_sampler, uv, 0.0).rg;
            let coherence = clamp(length(orientation), 0.0, 1.0);
            color = mix(vec3(0.5), hue(atan2(orientation.y, orientation.x)), coherence);
        }
        case 4u, 5u, 6u, 7u: {
            var terms = relief.terms;
            let weight = max(shading.detail_weights[shading.debug_view - 4u], 1e-3);
            color = grey(0.5 + 0.5 * terms[shading.debug_view - 4u] / weight);
        }
        case 8u: {
            color = grey(0.5 + (relief.terms[4] / max(shading.landform_weight, 1e-3) - light.z));
        }
        case 9u: {
            color = grey(transmittance);
        }
        case 10u: {
            color = grey(relief.brightness);
        }
        case 11u: {
            color = ramp(in.mesh_level / 10.0) * (0.75 + 0.25 * fract(in.mesh_level));
        }
        case 12u: {
            color = mix(vec3(0.9), vec3(0.1, 0.3, 0.8), terrain_water(xy, 0.0, finest));
        }
        case 13u: {
            // Which streamed level is loaded here; red is none of them.
            color = ramp(finest / f32(TILED_LEVELS)) * (0.8 + 0.2 * brightness);
        }
        default: {}
    }

    let slope = length(terrain_gradient(xy, pixel_level, finest));
    // The post pass lets the sky show through what the alpha leaves out.
    out.color = vec4(color, presence);
    out.metrics = vec4(brightness, select(0.0, 1.0, slope > SLOPE_THRESHOLD), flat_brightness, 1.0);
    return out;
}
