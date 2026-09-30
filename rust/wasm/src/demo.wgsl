// Original compact Mandelbrot renderer. Reference implementation:
// https://github.com/thaapasa/webgpu-fractal/blob/main/src/renderer/shaders/mandelbrot.wgsl
struct Frame {
    screen: vec4<f32>, // width, height, seconds, reserved
    view: vec4<f32>, // center x/y, scale, color phase — all controlled by Rust
}
@group(0) @binding(0) var<uniform> frame: Frame;

@vertex
fn vertex_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
    );
    return vec4<f32>(positions[index], 0.0, 1.0);
}

@fragment
fn fragment_main(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (2.0 * pixel.xy - frame.screen.xy) / frame.screen.y;
    let c = frame.view.xy + vec2<f32>(uv.x, -uv.y) * frame.view.z;
    let interior = vec4<f32>(0.015, 0.022, 0.04, 1.0);
    // Skip the main cardioid and period-two bulb before the escape-time loop.
    let cx = c.x - 0.25;
    let q = cx * cx + c.y * c.y;
    if (q * (q + cx) <= 0.25 * c.y * c.y || (c.x + 1.0) * (c.x + 1.0) + c.y * c.y <= 0.0625) {
        return interior;
    }
    var z = vec2<f32>(0.0);
    var iterations = 0.0;
    var escaped = false;
    for (var i = 0; i < 384; i++) {
        z = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iterations += 1.0;
        if (dot(z, z) > 256.0) { escaped = true; break; }
    }
    if (!escaped) { return interior; }
    let smooth_count = iterations + 1.0 - log2(0.5 * log2(dot(z, z)));
    let t = 0.025 * smooth_count + frame.view.w;
    let palette = 0.5 + 0.5 * cos(6.2831853 * (vec3<f32>(t) + vec3<f32>(0.0, 0.16, 0.32)));
    let brightness = 0.18 + 0.82 * (1.0 - exp(-0.08 * max(smooth_count, 0.0)));
    return vec4<f32>(pow(palette * brightness, vec3<f32>(0.8)), 1.0);
}
