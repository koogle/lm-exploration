# References

## Links

- [Facet rendering with wgpu](https://www.mattkeeter.com/blog/2026-08-23-wgpu-facet/)
- [WGSL offset computer](https://webgpufundamentals.org/webgpu/lessons/resources/wgsl-offset-computer.html)

## WebGPU data layout checks with Facet

Source: [Testing WebGPU data layouts with Facet](https://www.mattkeeter.com/blog/2026-08-23-wgpu-facet/)

### Key takeaways

- **Rust and WGSL can lay out the same-looking struct differently.** In the post's example, WGSL `mat3x3f` occupies 48 bytes because its three columns each have 16-byte stride, while Rust `[[f32; 3]; 3]` is tightly packed into 36 bytes. That mismatch shifts later fields and can make the shader read incorrect values.
- **Check layouts from the shader's parsed representation.** The post parses WGSL with `naga` and compares WGSL member offsets and sizes against Rust's actual layout, instead of assuming that matching field declarations imply matching bytes.
- **Use reflection to avoid fragile hand-maintained checks.** A generic checker combines Naga's WGSL type information with Facet's runtime reflection of Rust structs. It checks total size, each shared field's offset and size, and that both sides contain the expected fields.
- **Handle runtime-sized trailing arrays explicitly.** The Rust struct omits a trailing WGSL runtime-sized array; compare the Rust struct size with the WGSL array's starting offset rather than the WGSL struct's reported total size.
- **Run these checks as tests.** The example derives `Facet` only under `cfg(test)` and invokes the generic checker for each shader/config pair. This catches layout regressions without adding reflection overhead to normal builds.
- **A layout test complements shader validation.** It can detect host/shader data mismatches before they become confusing GPU behavior; it does not replace validating the shader or checking the meaning of the data.

### Relevance to this project

When adding Rust-side configuration or uniform structs for WebGPU shaders, verify their byte offsets and field sizes against the WGSL declarations. Pay particular attention to matrices and aligned vectors. A Naga-based test can use the project's existing shader tooling if available; Facet is one option for discovering Rust field layouts without writing per-struct offset lists.

## Fractal shader reference

- [WebGPU Fractal Explorer — Mandelbrot WGSL shader](https://github.com/thaapasa/webgpu-fractal/blob/main/src/renderer/shaders/mandelbrot.wgsl)

This reference demonstrates GPU escape-time fractals, view uniforms, and smooth
coloring. Our compact shader is an original implementation of `z = z² + c`, with
an escape-time loop and a cosine palette. It uses Rust-owned camera and palette
state rather than importing the reference application's frontend or dependencies.
