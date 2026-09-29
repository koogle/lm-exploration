# Web Inference

An experiment in running language models entirely in the browser.

The first goal is to run GPT-2 small and generate tokens, using Rust compiled to
WebAssembly for orchestration and custom WebGPU compute shaders for inference.
Everything runs locally in the browser; no inference server is required.

Start with correct end-to-end generation, then optimize performance.
The long-term goal is to demonstrate end-to-end GPT-2 training in the browser.

This repository is a fresh start. Implementation has not begun.
