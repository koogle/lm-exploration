# Hybrid model family (`model_hybrid`)

`hybrid/` implements the Kimi Delta Attention (KDA) + full-attention hybrid decoder used
for **Malama** (0.93B released / 960M with MTP; see the
[Malama v0.0.1 card](../publication/cards/malama-v0.0.1/README.md) and
[release notes](../plans/MALAMA_V0_0_1.md)), as well as the earlier 720M token and
byte-level [H-Net](../plans/HNET_720M.md) models.

## Architecture

[`model.py`](model.py) defines `HybridConfig` and `HybridLM`:

- **Backbone:** Pre-norm blocks with fp32-statistics `RMSNorm`, bias-free `SwiGLU` MLPs,
  and tied input/output embeddings. Residual output projections (`o_proj`, SwiGLU `down`)
  scale init by `0.02 / sqrt(2 * n_layer)`.
- **3:1 hybrid layout:** `layer_pattern` defaults to `("kda", "kda", "kda", "full")`,
  placing full attention at every fourth layer (e.g., 18 KDA + 6 full-attention layers in
  the 24-layer Malama 960M model).
- **Full-attention layers:** Grouped-query attention (12 query / 4 KV heads, head dim 128),
  QK-norm, RoPE (base 1,000,000), a learned per-head **attention sink** (`sigmoid(lse - sink)`
  so heads can attend to nothing), and a per-channel **sigmoid output gate** before
  `o_proj` (bias init `3.0`). Uses compiled `FlexAttention` on CUDA and explicit softmax
  on CPU.
- **KDA layers ([`kda.py`](kda.py)):** Linear-attention recurrence following
  [Kimi Linear](https://arxiv.org/abs/2510.26692) (`chunk_kda` via `fla-core` on CUDA,
  explicit fp32/fp64 recurrence on CPU): causal depthwise conv (size 4) + SiLU on q/k/v,
  L2-normalized q/k, sigmoid beta, per-channel decay, per-head RMSNorm, and low-rank
  output gate. No positional encoding. See [KDA, step by step](../references/kda.md).
- **Document masking ([`documents.py`](documents.py)):** Packed documents end in EOS;
  every forward pass derives per-token document IDs and `cu_seqlens` from `eos_id`,
  isolating attention, RoPE positions, KDA convolutions, and recurrent states across
  document boundaries.
- **Multi-token prediction (`mtp=True`):** Optional one-block DeepSeek-V3/GLM-style head
  predicting `x_(t+2)` through the shared output layer (weight decays 0.3 → 0.1; optional
  `mtp_loss_fraction < 1` subsampling). Dropped at inference/release (`mtp=False`).
- **Looping (`n_loops > 1`):** Optional shared recurrent middle block with gated updates
  following Elastic Latent Recurrence; see [LOOPED_LM.md](../plans/LOOPED_LM.md).

## Custom kernels and fast decode

- **Fused chunked output loss ([`fused_loss.py`](fused_loss.py), [`kernels/softmax_gradient.cu`](kernels/softmax_gradient.cu)):**
  Computes the vocabulary projection, cross-entropy, z-loss (`1e-4`), and in-place logit
  gradients in chunks without materializing `[tokens, vocab]` logits (+15% throughput on
  H100).
- **CUDA-graph decoding ([`generate.py`](generate.py), [`kernels/decode.cu`](kernels/decode.cu)):**
  Single-pass prefill loads full-attention KV caches and KDA conv/recurrent states into a
  cached `Decoder`, then replays captured CUDA graphs using custom KDA-step and
  sink-attention kernels (64 × 512-token prompts on H100: ~4.1 ms/step, 15.5K tok/s in
  bf16 for 720M). Cached decoders are reused across prompt widths and read weights in
  place for RL rollouts.
- **H100 KDA backward kernels ([`kernels/kda_wy.py`](kernels/kda_wy.py), [`kernels/kda_intra.py`](kernels/kda_intra.py)):**
  Registered via FLA's backend dispatch for Hopper (`sm_90`, head dim 128, chunk 64); see
  [KDA_REAL_DATA_MFU.md](../plans/KDA_REAL_DATA_MFU.md) for measurements and throughput
  comparisons.

## Optimizer and tokenizer

Parameters declare their optimizer role via [`trainer/roles.py`](../trainer/roles.py) and
are grouped by `trainer.optimizers.RoleOptimizer`:

| Role | Optimizer | Parameters |
| --- | --- | --- |
| `muon` | Muon (LR 0.02, mom 0.95, WD 0.1) | 2D hidden matrices (q/k/v/o, gates, KDA `f_proj`/`g_proj`, SwiGLU, MTP `fuse`) |
| `adamw_decay` | AdamW (LR 3e-4, betas 0.9/0.95, WD 0.1) | Tied embedding/output table, KDA `b_proj` |
| `adamw_no_decay` | AdamW (LR 3e-4, betas 0.9/0.95, WD 0) | Norm scales, sinks, gate biases, KDA `A_log`/`dt_bias`, conv kernels |

[`tokenizer/glm.py`](../tokenizer/glm.py) pins `zai-org/GLM-4.7-Flash`'s `tokenizer.json`
(154,856 tokens padded to 154,880; EOS `<|endoftext|>` = 154820; preserves GLM chat,
`<think>`, and tool tokens).

## Training, data, and variants

[`train.py`](train.py) adapts the hybrid family to the shared [`trainer/loop.py`](../trainer/loop.py)
for pretraining (`PreparedMixture`), chat/reasoning SFT (`ChatMixture` with replay), and
long-context mid-training. [`post_train.py`](post_train.py) connects it to the RL and
on-policy distillation loops ([`rl/README.md`](../rl/README.md)). [`modal_data.py`](modal_data.py)
prepares uint32 packed shards on Modal CPUs with license/language/path filters and
13-gram benchmark contamination screening.

Run workflows from `training/` (see [`workflows/README.md`](../workflows/README.md)):

```sh
# Prepare data or run a training workflow on Modal
uv run modal run --detach -m workflows.modal_train::main --workflow workflows/kda-960m-foundation.json --run-id <id>
uv run modal run -m workflows.modal_train::stop --run-id <id>

# Throughput benchmark (supports --fp8-only, --compare-kda-wy, --compare-intra, --hardware-kernel)
uv run modal run -m model_hybrid.modal_tools::bench --config hybrid/configs/kda-960m-100b.json --gpus 1

# Publish an exported checkpoint (add --full for raw training checkpoint with optimizer/cursor)
uv run modal run -m model_hybrid.modal_tools::publish --checkpoint <run>-sNN/step-NNNNNNNNN.pt --repo <owner/name>
```

**Byte-level H-Net ([`hnet.py`](hnet.py), [`byte_data.py`](byte_data.py)):** Wraps the
hybrid trunk in one dynamic-chunking stage operating directly on UTF-8 bytes (0–255 + EOS
256) with cosine boundary routing, chunked EMA dechunking, and ratio loss. Selected via
`"architecture": "hnet"` (`jfrick/hnet-720m`; see [HNET_720M.md](../plans/HNET_720M.md)).

## Benchmarks and playground

- **lm-evaluation-harness ([`benchmarks.py`](benchmarks.py), [`modal_benchmarks.py`](modal_benchmarks.py)):**
  Scores exports and pinned reference models on standard QA/code tasks or `--suite perplexity`
  (WikiText-2 + fresh arXiv JSONL from [`analysis/evaluation/fresh_text.py`](../analysis/evaluation/fresh_text.py)).
  LiveCodeBench/HumanEval/MBPP thinking-mode evaluation lives in
  [`analysis/evaluation/`](../analysis/evaluation/README.md).
  ```sh
  uv run --all-extras modal run --detach -m model_hybrid.modal_benchmarks --exports <run/tokens-N>
  uv run --all-extras modal run --detach -m model_hybrid.modal_benchmarks --suite perplexity \
    --text benchmarks/texts/arxiv-2026-11-16.jsonl --exports <run/tokens-N>
  ```
- **Playground ([`playground/`](playground/)):** Gradio app serving one or more exports
  with Chat, Compare, and Text tabs using `generate.py`. Deploy to a HF Docker Space with
  `playground/publish.py` or run on a Modal H100 (auto-stopping after `--hours`, max 24):
  ```sh
  uv run --all-extras modal run --detach -m model_hybrid.playground.modal_serve \
    --models hybrid/playground/malama-final.json --hours 4
  ```

## Tests

From `training/`:

```sh
# CPU unit tests
uv run python -m unittest tests.test_hybrid_backbone tests.test_hybrid_documents \
  tests.test_hybrid_sink_gate tests.test_hybrid_kda tests.test_hybrid_mtp \
  tests.test_hybrid_optim tests.test_hybrid_tokenizer tests.test_packed_shards \
  tests.test_hybrid_720m tests.test_hybrid_fused_loss tests.test_hybrid_hnet \
  tests.test_chat tests.test_hybrid_long_context tests.test_hybrid_looped -v

# Modal H100 CUDA checks (FlexAttention, FLA KDA, decode, H-Net, fp8_loss)
uv run modal run -m model_hybrid.modal_checks::main [--only decode|hnet|fp8_loss]
```
