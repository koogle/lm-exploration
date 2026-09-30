# Rust WebAssembly library

From the repository root, run `make build` to compile this crate for
`wasm32-unknown-unknown` and copy the binary to `web/inference.wasm`.

`demo.wgsl` is embedded in the binary and exposed through `shader_ptr` and
`shader_len`. The browser decodes the source and compiles it with WebGPU.

Pointer dragging calls `drag` with movement in shader view coordinates; trackpad
and wheel gestures call `zoom_at`, preserving the fractal point under the cursor.
Button events call `pan`, `zoom`, `explore_detail`, `toggle_pause`, and
`reset_view`. Rust owns the camera and palette state. Each animation frame calls
`update_frame`, which returns a pointer to a uniform block in WASM memory.
`uniform_size` reports its byte size. The block contains three `vec4<f32>` values and one `vec4<u32>`
matching WGSL `Frame`: screen dimensions/time at offset 0, camera center/scale
and palette phase at offset 16, and iteration limit/color density/speed at offset
32. The last block, at offset 48, holds the camera center's Q8.56 low/high words.
JavaScript copies all 64 bytes to the GPU unchanged. The fourth parameter enables
the integer arithmetic path for deep zooms. Rust retains camera state as f64;
WGSL adds local pixel offsets to the encoded center using integer arithmetic.

Sliders call `set_zoom_level`, `set_iterations`, `set_color_density`, and
`set_color_speed`. `read_settings` returns four floats for synchronizing the UI
with Rust: zoom level, iteration limit, color density, and color speed. Reset
restores all parameter defaults.

Exports run serially on the browser's main thread. There is no generated JS glue,
shared memory, external crate dependency, or frontend build system.
