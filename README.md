# Web Inference

An experiment in running language models entirely in the browser. A minimal
website loads a Rust library compiled to WebAssembly and renders a Mandelbrot
fractal through WebGPU; model inference is still planned.

## Run locally

Requirements: Rust/Cargo, Make, and Python 3.

```sh
rustup target add wasm32-unknown-unknown
make serve
```

Open http://127.0.0.1:5173/. This compiles the Rust library and serves the `web/`
folder directly. Use a browser with WebGPU enabled. The browser loads the WASM
file with plain JavaScript. HTML, CSS, and JavaScript need no bundler.

The buttons call Rust exports to pan, zoom, explore a detail preset, pause the
palette animation, or reset. Rust owns this state, embeds the WGSL shader, and
writes a 32-byte uniform block each frame. JavaScript copies those bytes to a
WebGPU buffer and submits a fullscreen triangle; the fragment shader calculates
the fractal on the GPU. The demo caps zoom and iterations for single-precision
rendering; it is not an arbitrary-precision deep-zoom renderer.

After editing Rust or WGSL, run `make build` and refresh the browser. Refresh after editing
HTML, CSS, or JavaScript as well.

## Build for hosting

```sh
make build
```

Serve the contents of `web/` on a static host, including the generated
`inference.wasm` file. Configure `.wasm` responses as `application/wasm`.

The [GPT-2 checkpoint](models/gpt2/README.md) is included through Git LFS. See
[the inference sketch](docs/browser-inference.md) and
[references](references/README.md) for the planned implementation.
