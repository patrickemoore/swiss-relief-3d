// Local light adjustment, step 1 of 3 (plan section 6.2).
//
// For every cell of a coarse grid around the camera, computes how far the
// base light should rotate so that it no longer runs parallel to the
// landform. A ridge parallel to the light has equally lit flanks and
// disappears.

// Mirrors `LocalLightUniform` in renderer/local_light.rs.
struct LocalLight {
    base_azimuth: f32,    // lambda_0: direction toward the light, radians counter-clockwise from east
    delta_max: f32,       // largest rotation
    theta_ok: f32,        // angle to the fall line that needs no correction
    hysteresis: f32,      // how far past 90 degrees a cell keeps its side
    sign_softness: f32,
    blur_sigma: f32,      // cells
    orientation_level: f32,
    pad: f32,
    field: vec4<f32>,     // xy: south-west corner of the field (m); z: its side (m); w: the terrain's side (m)
    moved: vec2<i32>,     // cells the field has moved by since last frame
    pad1: vec2<i32>,
};

const HALF_PI: f32 = 1.5707963;

@group(0) @binding(0) var<uniform> params: LocalLight;
@group(0) @binding(1) var orientation_tex: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var previous: texture_2d<f32>;
@group(0) @binding(4) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn rotate(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if (any(id.xy >= size)) {
        return;
    }
    let xy = params.field.xy + (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size) * params.field.z;
    let uv = xy / params.field.w;

    // Doubled-angle orientation (c cos 2 psi, c sin 2 psi): psi is the
    // fall-line axis, c says how clearly the landform is oriented.
    let orientation = textureSampleLevel(orientation_tex, linear_sampler, uv, params.orientation_level).rg;
    let coherence = min(length(orientation), 1.0);
    if (coherence < 1e-4 || any(uv < vec2(0.0)) || any(uv > vec2(1.0))) {
        textureStore(output, id.xy, vec4(0.0));
        return;
    }

    // theta: angle from the fall-line axis to the light, folded to +-90
    // degrees because an axis has no front or back. The two flanks contrast
    // in proportion to |cos theta|.
    let doubled = 2.0 * params.base_azimuth - atan2(orientation.y, orientation.x);
    var theta = 0.5 * atan2(sin(doubled), cos(doubled));

    // Hysteresis. Near |theta| = 90 degrees the folded angle flips sign, and
    // with it the direction of rotation. A cell keeps last frame's side until
    // theta is well past 90 degrees; otherwise a slow camera turn could swing
    // a landform's light by twice delta_max within a few degrees. The field
    // follows the camera, so last frame's cell may sit elsewhere in the grid.
    let before = vec2<i32>(id.xy) + params.moved;
    var previous_side = 0.0;
    if (all(before >= vec2(0)) && all(before < vec2<i32>(size))) {
        previous_side = textureLoad(previous, before, 0).b;
    }
    if (previous_side * theta < 0.0 && abs(theta) > HALF_PI - params.hysteresis) {
        theta = previous_side * (2.0 * HALF_PI - abs(theta));
    }

    // Rotate toward the fall line only as far as needed to bring |theta|
    // down to theta_ok, never by more than delta_max, and only where the
    // landform is clearly oriented.
    let side = sign(theta);
    let magnitude = coherence * min(params.delta_max, max(0.0, abs(theta) - params.theta_ok));
    textureStore(output, id.xy, vec4(-side * magnitude, magnitude, side, 1.0));
}
