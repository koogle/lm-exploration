// Rust owns the WGSL source, camera state, and animated palette.
const SHADER: &str = include_str!("demo.wgsl");
#[repr(C)]
struct Frame {
    screen: [f32; 4],
    view: [f32; 4],
    parameters: [f32; 4],
    center: [u32; 4],
}
static mut FRAME: Frame = Frame {
    screen: [0.0; 4],
    view: [0.0; 4],
    parameters: [0.0; 4],
    center: [0; 4],
};
static mut SETTINGS: [f32; 4] = [0.0; 4];

struct State {
    x: f64,
    y: f64,
    scale: f64,
    phase: f32,
    previous_seconds: f32,
    paused: bool,
    iterations: u32,
    density: f32,
    speed: f32,
}

static mut STATE: State = State {
    x: -0.6,
    y: 0.0,
    scale: 1.2,
    phase: 0.0,
    previous_seconds: 0.0,
    paused: false,
    iterations: 384,
    density: 0.025,
    speed: 0.035,
};

// Exports are called serially by the browser's main thread; no shared WASM memory.
fn with_state<T>(update: impl FnOnce(&mut State) -> T) -> T {
    unsafe { update(&mut *core::ptr::addr_of_mut!(STATE)) }
}

#[unsafe(no_mangle)]
pub extern "C" fn shader_ptr() -> *const u8 {
    SHADER.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn shader_len() -> usize {
    SHADER.len()
}

#[unsafe(no_mangle)]
pub extern "C" fn uniform_size() -> usize {
    core::mem::size_of::<Frame>()
}

#[unsafe(no_mangle)]
pub extern "C" fn pan(horizontal: f64, vertical: f64) {
    drag(horizontal * 0.25, vertical * 0.25);
}

#[unsafe(no_mangle)]
pub extern "C" fn zoom(factor: f64) {
    zoom_at(factor, 0.0, 0.0);
}

// Coordinates use the shader's height-normalized view space, with positive y up.
#[unsafe(no_mangle)]
pub extern "C" fn drag(horizontal: f64, vertical: f64) {
    if !horizontal.is_finite() || !vertical.is_finite() {
        return;
    }
    with_state(|state| {
        let x = state.x + horizontal * state.scale;
        let y = state.y + vertical * state.scale;
        if x.is_finite() && y.is_finite() {
            state.x = x;
            state.y = y;
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn zoom_at(factor: f64, horizontal: f64, vertical: f64) {
    if !factor.is_finite() || factor <= 0.0 {
        return;
    }
    if !horizontal.is_finite() || !vertical.is_finite() {
        return;
    }
    with_state(|state| {
        let next_scale = state.scale * factor;
        let x = state.x + horizontal * (state.scale - next_scale);
        let y = state.y + vertical * (state.scale - next_scale);
        // Reject numeric overflow/underflow, without imposing a zoom range.
        if next_scale.is_finite() && next_scale > 0.0 && x.is_finite() && y.is_finite() {
            state.x = x;
            state.y = y;
            state.scale = next_scale;
        }
    });
}

// UI snapshot: logarithmic zoom level, iteration limit, palette density, speed.
#[unsafe(no_mangle)]
pub extern "C" fn read_settings() -> *const f32 {
    let values = with_state(|s| {
        [
            (1.2_f64.log2() - s.scale.log2()) as f32,
            s.iterations as f32,
            s.density,
            s.speed,
        ]
    });
    unsafe {
        SETTINGS = values;
    }
    core::ptr::addr_of!(SETTINGS).cast::<f32>()
}

#[unsafe(no_mangle)]
pub extern "C" fn set_zoom_level(value: f64) {
    if !value.is_finite() {
        return;
    }
    let scale = 1.2_f64 * 2.0_f64.powf(-value);
    if scale.is_finite() && scale > 0.0 {
        with_state(|s| s.scale = scale);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn set_iterations(value: f32) {
    if !value.is_finite() {
        return;
    }
    with_state(|s| s.iterations = value.clamp(64.0, 1024.0).round() as u32);
}

#[unsafe(no_mangle)]
pub extern "C" fn set_color_density(value: f32) {
    if !value.is_finite() {
        return;
    }
    with_state(|s| s.density = value.clamp(0.005, 0.08));
}

#[unsafe(no_mangle)]
pub extern "C" fn set_color_speed(value: f32) {
    if !value.is_finite() {
        return;
    }
    with_state(|s| s.speed = value.clamp(0.0, 0.15));
}

#[unsafe(no_mangle)]
pub extern "C" fn explore_detail() {
    with_state(|state| {
        state.x = -0.743643887037151;
        state.y = 0.13182590420533;
        state.scale = 0.003;
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn toggle_pause() -> u32 {
    with_state(|state| {
        state.paused = !state.paused;
        u32::from(state.paused)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn reset_view() {
    with_state(|state| {
        state.x = -0.6;
        state.y = 0.0;
        state.scale = 1.2;
        state.phase = 0.0;
        state.paused = false;
        state.iterations = 384;
        state.density = 0.025;
        state.speed = 0.035;
    });
}

fn fixed_words(value: f64) -> [u32; 2] {
    // Signed Q8.56: enough integer range for escaped Mandelbrot iterates.
    let bits = ((value * 72_057_594_037_927_936.0).round() as i64) as u64;
    [bits as u32, (bits >> 32) as u32]
}

// Matches WGSL Frame: vec4 blocks at offsets 0/16/32/48, total 64 bytes.
#[unsafe(no_mangle)]
pub extern "C" fn update_frame(width: f32, height: f32, seconds: f32) -> *const f32 {
    let values = with_state(|state| {
        let delta = (seconds - state.previous_seconds).clamp(0.0, 0.1);
        state.previous_seconds = seconds;
        if !state.paused {
            state.phase = (state.phase + delta * state.speed) % 1.0;
        }
        let precise = state.scale < 0.0001 && state.x.abs() < 4.0 && state.y.abs() < 4.0;
        let x = if precise {
            fixed_words(state.x)
        } else {
            [0; 2]
        };
        let y = if precise {
            fixed_words(state.y)
        } else {
            [0; 2]
        };
        Frame {
            screen: [width, height, seconds, 0.0],
            view: [
                state.x as f32,
                state.y as f32,
                state.scale as f32,
                state.phase,
            ],
            parameters: [
                state.iterations as f32,
                state.density,
                state.speed,
                f32::from(precise),
            ],
            center: [x[0], x[1], y[0], y[1]],
        }
    });
    unsafe {
        FRAME = values;
    }
    core::ptr::addr_of!(FRAME).cast::<f32>()
}
