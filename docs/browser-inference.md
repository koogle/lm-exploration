# From checkpoint to browser inference

This is an implementation sketch, not a runnable inference engine. A minimal
browser page and Rust-controlled WebGPU fractal demo exist; the checkpoint loader, GPU
kernels, and tokenizer remain to be implemented.

## Deployment

```text
Build machine                             Browser
Rust source -- compile --> .wasm --------> Rust orchestration on CPU
                                           ^
JavaScript -- fetch + instantiate --------|
WGSL source ----------------------------> WebGPU shader compilation
GGUF checkpoint -- static download -----> metadata + weights + tokenizer
                                         |
                                         v
                                   GPU storage buffers
```

The current demo uses `make build` to compile a Rust `cdylib` with Cargo and uses browser JavaScript's
`WebAssembly.instantiateStreaming` to load and call its raw exports. This avoids
generated JavaScript bindings for the small demo ABI. For a full inference engine,
use `wgpu` with its WebGPU backend; `wasm-bindgen` can be added if richer Rust/JS
types are useful. A static HTTPS host serves HTML, JS, WASM, and the checkpoint;
localhost works for development. There is no inference service.
The deployment must materialize the Git LFS binary, not serve its pointer file.
The model is a separate download, not embedded in the WASM executable.

Rust runs as WebAssembly on the CPU. It requests a WebGPU adapter/device, creates
buffers and compute pipelines, and submits work. The browser compiles WGSL shaders
for the GPU. Rust does not itself become a GPU shader.

## Load once, generate repeatedly

1. Fetch the checkpoint. Validate its header, architecture, tensor names, shapes,
   types, offsets, and lengths against the expected model before uploading.
2. Read the tokenizer vocabulary and merges. Implement GPT-2 byte-level BPE,
   including its pre-tokenization; splitting text on spaces is insufficient.
3. Decode F32, Q4_0, and Q6_K tensors. For the first correctness implementation,
   expand one tensor at a time to F32, upload it, and release the CPU temporary.
   Retain the separate output tensor: its quantization differs from the embedding.
4. Check device limits, split matrices exceeding storage-binding limits, create
   buffers, and compile custom compute pipelines. Keep weights on the GPU across
   token-generation steps. Allocate activation scratch space separately.
5. Encode the prompt, run a causal forward pass, and obtain the final position's
   50,257 logits (one score per possible next token).
6. Read those logits back asynchronously and select the highest score for the
   initial deterministic demo. Append the token and run again, stopping at EOS,
   the requested generation limit, or the 1,024-token context limit.

Initially recomputing the growing sequence is simpler to verify. Later, a KV cache
stores previous attention keys and values so each decode step processes only the
new token. Neither approach reloads the model between tokens.

## Rust orchestration sketch

The following is pseudocode. These helper functions are planned responsibilities,
not existing library APIs or implemented functions.

```rust
async fn load(url: &str) -> Result<Session> {
    let bytes = fetch_bytes(url).await?;
    let model = parse_and_validate_gguf(&bytes)?;
    let gpu = request_webgpu_device().await?;
    let weights = decode_and_upload_tensors(&gpu, &model)?;
    let kernels = compile_wgsl_pipelines(&gpu)?;
    Session::new(gpu, weights, kernels, model.tokenizer())
}

async fn next_token(session: &mut Session, token_ids: &[u32]) -> Result<u32> {
    let commands = session.encode_gpt2_forward(token_ids)?;
    session.queue.submit([commands]);
    let logits = session.read_last_position_logits().await?;
    Ok(argmax(&logits))
}
```

A real dispatch inside `encode_gpt2_forward` follows this pattern (surrounding
buffer creation, bind-group layout, pipeline creation, and errors omitted):

```rust
{
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(&matrix_vector_pipeline);
    pass.set_bind_group(0, &weights_input_output_bind_group, &[]);
    pass.dispatch_workgroups(output_rows.div_ceil(64), 1, 1);
}
```

A bind group connects named shader inputs to actual GPU buffers. An encoder records
commands; submitting its finished command buffer schedules execution. GPU readback
uses a staging buffer and asynchronous mapping, rather than a blocking wait in WASM.

## What a shader computes

A projection transforms a hidden-state vector with `y = W x`. This deliberately
simple WGSL kernel gives one invocation one output row. It expects already-decoded,
row-major F32 weights, excludes bias, and illustrates single-token computation.
It cannot consume the GGUF file directly.

```wgsl
struct Shape { rows: u32, cols: u32, }
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> shape: Shape;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= shape.rows) { return; }
    var sum = 0.0;
    for (var col = 0u; col < shape.cols; col++) {
        sum += weights[row * shape.cols + col] * input[col];
    }
    output[row] = sum;
}
```

For 768 output rows, Rust dispatches 12 workgroups of 64 invocations. The GPU can
compute many rows concurrently. A 768-by-768 projection performs 589,824 scalar
multiplications. Tiling and cooperative reductions come after correctness.

This is one operation, not a complete transformer. GPT-2 requires token and position
embeddings, then 12 blocks of layer normalization, Q/K/V projections, causal
multi-head attention with scaled scores and softmax, attention output projection,
residual additions, and an MLP with GPT-2's GELU activation. Final normalization and
the output projection produce logits. Biases and tensor layouts must match the
checkpoint. Prefill needs sequence-aware kernels; the single-vector example above
is only one building block.

The first executable milestone should compare one GPU projection against a CPU
reference using actual checkpoint weights. Then validate a block, full-model logits,
and greedy token generation against a reference using this same quantized file.

## References

- [wgpu in the browser](https://wgpu.rs/doc/wgpu/documentation/platforms/web/index.html)
- [wgpu compute passes](https://docs.rs/wgpu/latest/wgpu/struct.ComputePass.html)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGML quantization types](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h)
- [GPT-2 reference forward pass](https://github.com/openai/gpt-2/blob/master/src/model.py)
