# Local Mac Inference

A learning-first Rust harness for local inference on Apple Silicon, using Apple's Core ML Objective-C API directly. The Rust/Objective-C boundary is owned by this project; no Rust Core ML binding crate or pre-converted model is used.

## First spike: exercise the native boundary

Before converting Qwen, generate a tiny affine Core ML model (`y = 3x + 1`) and call it from Rust through a small Objective-C bridge. This isolates model loading, feature construction, prediction, error handling, and Rust FFI from LLM conversion/state complexity.

The bridge calls `MLModel`, `MLMultiArray`, `MLDictionaryFeatureProvider`, and `MLFeatureProvider` directly. Rust owns the CLI and C ABI wrapper; Objective-C owns Core ML objects and translates errors into C strings. No crates are used.

## Run on a Mac

Requires Apple Silicon, macOS 15+, Xcode Command Line Tools, Rust, Python 3, PyTorch, and `coremltools`.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch coremltools
python scripts/make_smoke_model.py
cargo run -- artifacts/affine-smoke.mlpackage 2
```

Expected output: `input=2 output=7`.

Core ML is macOS-only. The current development host is Linux, so native model conversion/prediction and Objective-C compilation must be verified on the target Mac. `cargo test` checks the portable Rust portion here; it does not claim Core ML ran.

## Next step after the smoke test

Try a small *original* Qwen checkpoint (Qwen3.5-0.8B) and implement its conversion rather than downloading a pre-converted Core ML package. Keep model-specific conversion and tokenizer/template handling separate from the native Core ML session. The 0.8B checkpoint is a smaller weight payload, but shares Qwen3.5's hybrid recurrent + attention state complexity, so first validate the generic Core ML call boundary with the affine spike.

No model binaries belong in Git. Save a manifest with the checkpoint revision, converter version, tensor/state schema, precision, and validation results.
