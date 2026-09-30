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

Drag the canvas to pan. Two-finger trackpad scrolling, pinching, and mouse-wheel
scrolling zoom around the pointer. The buttons also call Rust exports to pan,
zoom, explore a detail preset, pause the
palette animation, or reset. Sliders adjust logarithmic zoom, iteration detail,
color bands, and color animation speed; Rust validates their ranges. Rust owns this state, embeds the WGSL shader, and
writes a 64-byte uniform block each frame. JavaScript copies those bytes to a
WebGPU buffer and submits a fullscreen triangle; the fragment shader calculates
the fractal on the GPU. The zoom controls have no preset bounds, and the slider expands as you explore.
Rust keeps the camera in 64-bit floating point. At deep zooms the GPU uses
signed Q8.56 integer arithmetic, preserving tiny coordinate differences instead
of rounding every pixel to a single-precision center. The normal view uses the
faster float shader path. Iterations remain bounded to keep rendering responsive.
Precision is still finite: arbitrary-precision coordinates and perturbation-based
reference orbits are needed for indefinitely deep exploration.

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
