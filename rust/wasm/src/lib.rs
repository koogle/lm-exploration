// Rust owns the WGSL source, camera state, and animated palette.
const SHADER: &str = include_str!("demo.wgsl");
static mut FRAME: [f32; 8] = [0.0; 8];

struct State {
    x: f32,
    y: f32,
    scale: f32,
    phase: f32,
    previous_seconds: f32,
    paused: bool,
}

static mut STATE: State = State {
    x: -0.6,
    y: 0.0,
    scale: 1.2,
    phase: 0.0,
    previous_seconds: 0.0,
    paused: false,
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
    core::mem::size_of::<[f32; 8]>()
}

#[unsafe(no_mangle)]
pub extern "C" fn pan(horizontal: f32, vertical: f32) {
    if !horizontal.is_finite() || !vertical.is_finite() {
        return;
    }
    with_state(|state| {
        state.x = (state.x + horizontal * state.scale * 0.25).clamp(-3.0, 3.0);
        state.y = (state.y + vertical * state.scale * 0.25).clamp(-3.0, 3.0);
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn zoom(factor: f32) {
    if !factor.is_finite() || factor <= 0.0 {
        return;
    }
    // Limit zoom to what f32 shader coordinates can meaningfully resolve.
    with_state(|state| state.scale = (state.scale * factor).clamp(0.0001, 3.0));
}

#[unsafe(no_mangle)]
pub extern "C" fn explore_detail() {
    with_state(|state| {
        state.x = -0.7435;
        state.y = 0.1314;
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
    });
}

// Matches WGSL Frame: two vec4<f32> members, offsets 0 and 16, total 32 bytes.
#[unsafe(no_mangle)]
pub extern "C" fn update_frame(width: f32, height: f32, seconds: f32) -> *const f32 {
    let values = with_state(|state| {
        let delta = (seconds - state.previous_seconds).clamp(0.0, 0.1);
        state.previous_seconds = seconds;
        if !state.paused {
            state.phase = (state.phase + delta * 0.035) % 1.0;
        }
        [
            width,
            height,
            seconds,
            0.0,
            state.x,
            state.y,
            state.scale,
            state.phase,
        ]
    });
    unsafe {
        FRAME = values;
    }
    core::ptr::addr_of!(FRAME).cast::<f32>()
}
