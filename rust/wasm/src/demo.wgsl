// Original compact Mandelbrot renderer. Reference implementation:
// https://github.com/thaapasa/webgpu-fractal/blob/main/src/renderer/shaders/mandelbrot.wgsl
struct Frame {
    screen: vec4<f32>, // width, height, seconds, reserved
    view: vec4<f32>, // center x/y, scale, color phase — all controlled by Rust
    parameters: vec4<f32>, // iterations, color density, speed, precision mode
    center: vec4<u32>, // Q8.56 center x low/high, center y low/high
}
@group(0) @binding(0) var<uniform> frame: Frame;

@vertex
fn vertex_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
    );
    return vec4<f32>(positions[index], 0.0, 1.0);
}

// BEGIN_FIXED_POINT
// Signed Q8.56 values stored as low/high u32 words. All arithmetic is integer.
fn q_neg(a: vec2<u32>) -> vec2<u32> {
    let lo = ~a.x + 1u;
    return vec2<u32>(lo, ~a.y + select(0u, 1u, lo == 0u));
}
fn q_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
fn q_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return q_add(a, q_neg(b)); }
fn q_abs(a: vec2<u32>) -> vec2<u32> {
    return select(a, q_neg(a), (a.y & 0x80000000u) != 0u);
}
fn wide_mul(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 65535u;
    let b0 = b & 65535u;
    let a1 = a >> 16u;
    let b1 = b >> 16u;
    let p0 = a0 * b0;
    let p1 = a0 * b1;
    let p2 = a1 * b0;
    let middle = (p0 >> 16u) + (p1 & 65535u) + (p2 & 65535u);
    return vec2<u32>((p0 & 65535u) | (middle << 16u), a1 * b1 + (p1 >> 16u) + (p2 >> 16u) + (middle >> 16u));
}
fn wide_add(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let x = a.x + b.x;
    let y0 = a.y + b.y;
    let y = y0 + select(0u, 1u, x < a.x);
    let z0 = a.z + b.z;
    let z = z0 + select(0u, 1u, y0 < a.y || y < y0);
    let w = a.w + b.w + select(0u, 1u, z0 < a.z || z < z0);
    return vec4<u32>(x, y, z, w);
}
fn q_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let aa = q_abs(a);
    let bb = q_abs(b);
    let p00 = wide_mul(aa.x, bb.x);
    let p01 = wide_mul(aa.x, bb.y);
    let p10 = wide_mul(aa.y, bb.x);
    let p11 = wide_mul(aa.y, bb.y);
    var p = vec4<u32>(p00, 0u, 0u);
    p = wide_add(p, vec4<u32>(0u, p01, 0u));
    p = wide_add(p, vec4<u32>(0u, p10, 0u));
    p = wide_add(p, vec4<u32>(0u, 0u, p11));
    let result = vec2<u32>((p.y >> 24u) | (p.z << 8u), (p.z >> 24u) | (p.w << 8u));
    return select(result, q_neg(result), ((a.y ^ b.y) & 0x80000000u) != 0u);
}
fn q_from_float(value: f32) -> vec2<u32> {
    let bits = bitcast<u32>(abs(value));
    let exponent = i32((bits >> 23u) & 255u);
    if (exponent == 0) { return vec2<u32>(0u); }
    let mantissa = (bits & 0x7fffffu) | 0x800000u;
    // value = mantissa * 2^(exponent - 127 - 23); multiply by 2^56.
    let shift = exponent - 94;
    var result = vec2<u32>(0u);
    if (shift >= 32 && shift < 64) {
        result.y = mantissa << u32(shift - 32);
    } else if (shift > 0 && shift < 32) {
        result = vec2<u32>(mantissa << u32(shift), mantissa >> u32(32 - shift));
    } else if (shift == 0) {
        result.x = mantissa;
    } else if (shift < 0 && shift > -32) {
        result.x = mantissa >> u32(-shift);
    }
    return select(result, q_neg(result), value < 0.0);
}
fn q_to_float(value: vec2<u32>) -> f32 {
    let a = q_abs(value);
    let magnitude = f32(a.y) * (1.0 / 16777216.0) + f32(a.x) * (1.0 / 72057594037927936.0);
    return select(magnitude, -magnitude, (value.y & 0x80000000u) != 0u);
}
// END_FIXED_POINT

fn precise_orbit(offset: vec2<f32>, limit: u32) -> vec2<f32> {
    let cx = q_add(frame.center.xy, q_from_float(offset.x));
    let cy = q_add(frame.center.zw, q_from_float(offset.y));
    var x = vec2<u32>(0u);
    var y = vec2<u32>(0u);
    for (var i = 0u; i <= 1024u; i++) {
        let xx = q_mul(x, x);
        let yy = q_mul(y, y);
        let norm = q_add(xx, yy);
        // Escape at |z| >= 2, keeping all fixed-point intermediates in range.
        if (norm.y >= 0x04000000u) { return vec2<f32>(f32(i), q_to_float(norm)); }
        if (i >= limit) { break; }
        let xy = q_mul(x, y);
        x = q_add(q_sub(xx, yy), cx);
        y = q_add(q_add(xy, xy), cy);
    }
    return vec2<f32>(-1.0, 0.0);
}

fn orbit(c: vec2<f32>, limit: u32) -> vec2<f32> {
    let cx = c.x - 0.25;
    let q = cx * cx + c.y * c.y;
    if (q * (q + cx) <= 0.25 * c.y * c.y || (c.x + 1.0) * (c.x + 1.0) + c.y * c.y <= 0.0625) {
        return vec2<f32>(-1.0, 0.0);
    }
    var z = vec2<f32>(0.0);
    for (var i = 0u; i <= 1024u; i++) {
        let norm = dot(z, z);
        if (norm >= 4.0) { return vec2<f32>(f32(i), norm); }
        if (i >= limit) { break; }
        z = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
    }
    return vec2<f32>(-1.0, 0.0);
}

@fragment
fn fragment_main(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (2.0 * pixel.xy - frame.screen.xy) / frame.screen.y;
    let offset = vec2<f32>(uv.x, -uv.y) * frame.view.z;
    let c = frame.view.xy + offset;
    if (frame.view.z > 1.0e10 || any(abs(c) > vec2<f32>(1.0e10))) {
        return vec4<f32>(0.04, 0.055, 0.09, 1.0);
    }
    let limit = u32(frame.parameters.x);
    var result: vec2<f32>;
    if (frame.parameters.w > 0.5) {
        result = precise_orbit(offset, limit);
    } else {
        result = orbit(c, limit);
    }
    if (result.x < 0.0) { return vec4<f32>(0.015, 0.022, 0.04, 1.0); }
    let smooth_count = result.x + 1.0 - log2(0.5 * log2(result.y));
    let t = frame.parameters.y * smooth_count + frame.view.w;
    let palette = 0.5 + 0.5 * cos(6.2831853 * (vec3<f32>(t) + vec3<f32>(0.0, 0.16, 0.32)));
    let brightness = 0.18 + 0.82 * (1.0 - exp(-0.08 * max(smooth_count, 0.0)));
    return vec4<f32>(pow(palette * brightness, vec3<f32>(0.8)), 1.0);
}
