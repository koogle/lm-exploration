# Rust WebAssembly library

From the repository root, run `make build` to compile this crate for
`wasm32-unknown-unknown` and copy the binary to `web/inference.wasm`.

`demo.wgsl` is embedded in the binary and exposed through `shader_ptr` and
`shader_len`. The browser decodes the source and compiles it with WebGPU.

Button events call `pan`, `zoom`, `explore_detail`, `toggle_pause`, and
`reset_view`. Rust owns the camera and palette state. Each animation frame calls
`update_frame`, which returns a pointer to a uniform block in WASM memory.
`uniform_size` reports its byte size. The block contains two `vec4<f32>` values
matching WGSL `Frame`: screen dimensions/time at offset 0, camera center/scale
and palette phase at offset 16. JavaScript copies it to the GPU unchanged.

Exports run serially on the browser's main thread. There is no generated JS glue,
shared memory, external crate dependency, or frontend build system.
