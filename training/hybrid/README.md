# Hybrid model family

`hybrid/` (installed as `model_hybrid`) holds the decoder for the
[~720M hybrid pretraining plan](../plans/HYBRID_720M.md). It is separate from
`10m/` and `100m/`, so the 124M Simple LLM runs use unchanged code.

## Backbone

[`model.py`](model.py) defines `HybridConfig` and `HybridLM`:

- Pre-norm blocks with `RMSNorm` (statistics in fp32) and a `SwiGLU`
  feed-forward layer, both the shared classes from [`10m/model.py`](../10m/model.py);
  the hybrid family builds `SwiGLU` without biases at a configurable hidden width.
- Tied input/output embeddings; no bias parameters except the attention output
  gate's (below).
- Full-attention layers use grouped-query attention (`n_head` query heads,
  `n_kv_head` key/value heads, head dimension 128), RMSNorm on each query and key
  head (QK-norm), and RoPE with a configurable base (default 1,000,000).
- Initialization: Normal(0, 0.02) for embeddings and projections; residual
  output projections (attention `o_proj`, SwiGLU `down`) use
  `0.02 / sqrt(2 * n_layer)`. Norm scales start at 1.
- `forward(tokens)` returns logits. `forward(tokens, labels)` returns summed
  cross-entropy `ce`, summed squared log-normalizer `z`, `loss = ce + 1e-4 * z` and
  the valid-target `count`. Logits are computed in chunks of `loss_chunk_size`
  rows and recomputed in backward, so the full `[tokens, vocab]` logit tensor is
  never stored. Labels of `-100` are ignored.

## Document masking

Loaders pack documents back to back, each ending in EOS.
[`documents.py`](documents.py) derives per-row document ids from those EOS
boundaries (`document_ids`), converts them to flattened `cu_seqlens` for varlen
kernels, and `DocumentBatches` wraps any token mixture so `next_batch` returns
`(inputs, labels, document_ids)`. Pass the ids as `forward(tokens, labels, documents)`.

Attention is document-causal: a token sees earlier tokens of its own document
only, and RoPE positions restart at each document. Without document ids each row
is treated as one document.

## Attention kernel

On CUDA, full attention runs our own kernel,
[`kernels/document_attention.cu`](kernels/document_attention.cu), wrapped as
`torch.library` custom ops in [`document_attention.py`](document_attention.py)
so `torch.compile` treats forward and backward as single nodes. It takes bf16 or
fp32 q/k/v (GQA, head dimension 64 or 128) and int32 document starts (row `i`
attends to keys `start_i..i`) and returns the output and the fp32 log-sum-exp.

The kernel is **bit-exact** with [`exact_attention.py`](exact_attention.py), a
PyTorch definition of every float operation and its order: dot products sum over
the head dimension in order, softmax sums and value accumulations sum over keys in
order, gradient accumulations over query heads and then query rows in order. All
operations are correctly rounded float32 multiplies, adds and divides (no fused
multiply-add; built with `-fmad=false`), and `exp_fixed`/`log_fixed` are short
polynomials, so the definition can be ported to other hardware (for example Metal)
with the same bits. `exp_fixed` has relative error below `1.2e-7 * max(1, |x|)`.
Each block owns disjoint outputs, so there are no atomics and repeated runs give
identical bits.

The forward makes two passes over the allowed key tiles of each 64-row query tile
(row maxima, then exponentials, sums and the value accumulation); the backward
recomputes probabilities from the saved maxima and sums, with one kernel for dQ
and one per key/value head for dK and dV. Tiles outside a row's document are
skipped. The arithmetic uses CUDA cores in float32, not tensor cores, whose
internal accumulation order cannot be reproduced exactly.

Measured on one H100 (torch 2.14.1+cu130, bf16, B=8, 12 query / 4 KV heads,
T=4096, head dimension 128): forward 49.8 ms and forward+backward 161 ms with one
document per row; 13.3 ms and 44.1 ms with four documents per row. PyTorch's
causal SDPA (tensor cores, one document) takes 0.69 ms and 3.1 ms. Exactness
costs this factor; a faster kernel must keep the same reference bits.

On CPU, `model.reference_attention` (an explicit masked softmax, fp32 or fp64) is
the model's attention.

## Attention sink and output gate

Each full-attention layer has a learned sink: one logit per head, initialized to
0, acting as an extra key with a zero value. Softmax over the scores plus the sink
equals the plain attention output times `sigmoid(lse - sink)`, where `lse` is the
log-sum-exp of the scores, so the layer applies that factor to the kernel's (or
the reference's) output; the kernel's backward takes the log-sum-exp gradient. A head can then attend to "nothing" without spending
probability on a real token.

After the sink, an elementwise sigmoid gate multiplies the merged heads before
`o_proj`: `sigmoid(gate_proj(x) + gate_bias)`, with `x` the layer's normalized
input. `gate_bias` (per channel) starts at `gate_bias_init = 3.0`, so the gate
starts about 95% open. With the gate fully open and the sink at -inf the layer
reduces exactly to plain document-causal attention.

## Kimi Delta Attention and the hybrid layout

[`kda.py`](kda.py) implements KDA layers following flash-linear-attention's
`KimiDeltaAttention` ([Kimi Linear](https://arxiv.org/abs/2510.26692)), without
biases:

- q/k/v projections to `n_kda_head` heads of width `head_dim`, each followed by a
  causal depthwise convolution (kernel `kda_conv_size = 4`) and SiLU.
- L2-normalized q and k (q further scaled by `1/sqrt(head_dim)`), sigmoid beta
  from `b_proj`, and a per-channel decay `-exp(A_log) * softplus(f_proj(x) + dt_bias)`
  with a low-rank `f_proj`.
- Per-head output RMSNorm, a sigmoid gate from a low-rank `g_proj`, and `o_proj`
  (a residual output projection for initialization).
- No positional encoding; the recurrent state is fp32.

Document boundaries reset both the convolutions (they never read a previous
document's tokens) and the recurrent state. On CUDA the recurrence is fla's
`chunk_kda`, run over the flattened batch with `cu_seqlens` from the document ids;
gate, beta and L2 normalization run inside the kernel. On CPU, `kda_reference`
is an explicit per-token recurrence with fp32 (or fp64) state.

`torch.compile` traces through the fla call. On one H100, a small compiled hybrid
model (4 layers, width 512, 4 x 1024 tokens, bf16) took 18.7–19.1 ms per training
step that way and 19.0–19.5 ms with fla excluded by `torch.compiler.disable`
(three alternating rounds after warming up both), so the graph stays unbroken.

`HybridConfig.layer_pattern` lists `"kda"` and `"full"` layers and repeats to
`n_layer`; the default `("kda", "kda", "kda", "full")` puts full attention at
layers 3, 7, 11 and 15 of 16. Full-attention layers keep RoPE, the sink and the
output gate.

## Multi-token prediction

`MultiTokenPrediction` is one DeepSeek-V3/GLM-style module (`mtp=True` by
default). At position `t` it concatenates RMSNorm of the trunk's last residual
state `h_t` with RMSNorm of the embedding of `x_(t+1)`, projects back to the
model width, runs one hybrid block (`mtp_layer_kind`, full attention by
default, sharing the trunk's document mask and RoPE) and an output RMSNorm, and
predicts `x_(t+2)` through the shared tied output layer. Its loss uses the same
chunked cross-entropy and z-loss.

`documents.mtp_labels(inputs, labels, eos_id)` builds the `t+2` targets:
`labels[t + 1]`, ignored when `x_t` or `x_(t+1)` is EOS (the target would be in
the next document) and at the last position. `DocumentBatches.next_batch`
returns a `Batch(inputs, labels, documents, mtp_labels)`.
`forward(tokens, labels, documents, mtp_labels)` adds `mtp_loss`, `mtp_ce`,
`mtp_z` and `mtp_count`. The trainer combines the means as
`loss / count + mtp_weight(config, progress) * mtp_loss / mtp_count`, where the
weight decays linearly from 0.3 to 0.1 over training progress.

Inference does not run the module: `forward(tokens)` returns trunk logits, and
a model built with `mtp=False` loads the trunk weights (all keys outside `mtp.`)
and produces identical logits.

## Optimizer grouping

[`optim.py`](optim.py) assigns every parameter exactly one role:

| Role | Optimizer | Parameters |
| --- | --- | --- |
| `muon` | Muon, LR 0.02, momentum 0.95, weight decay 0.1 | 2D hidden matrices in blocks and MTP: q/k/v/o, attention `gate_proj`, KDA `f_proj`/`g_proj`, SwiGLU, MTP `fuse` |
| `adamw_decay` | AdamW, LR 3e-4, betas (0.9, 0.95), weight decay 0.1 | tied embedding/output table, KDA beta projection `b_proj` |
| `adamw_no_decay` | AdamW, LR 3e-4, betas (0.9, 0.95), no weight decay | norm scales, attention sinks, gate biases, KDA `A_log`, `dt_bias`, convolution kernels |

Muon is `trainer.optimizers.Muon` (five-step Newton–Schulz with the original
aspect-ratio scaling); its weight decay is decoupled, `p *= 1 - lr * 0.1`.
The learning rates are the 124M comparison's
([MUON_COMPARISON.md](../distributed/MUON_COMPARISON.md)), starting values for
the 720M pilot rather than tuned at this scale. `HybridOptimizer` keeps Muon
groups at `lr_scale = muon_lr / adamw_lr`, so `trainer.optimizers.set_learning_rate`
applies one schedule factor to both. Unclassified parameters raise.

## Tokenizer and data recipe

[`tokenizer.py`](tokenizer.py) pins zai-org/GLM-4.7-Flash's `tokenizer.json`
(revision `7dd20894`, SHA-256 `19e77364…a82d`). On 2026-10-25 it was
byte-identical to zai-org/GLM-5.3-Flash's (revision `46a050be`). It has 154,856
entries; the model pads its embedding to 154,880. Documents end with
`<|endoftext|>` (154820); GLM's chat (`<|system|>`, `<|user|>`, `<|assistant|>`,
`<|observation|>`), reasoning (`<think>`) and tool (`<tool_call>`, `<arg_key>`, …)
tokens keep their GLM ids, which `load()` verifies.

[`recipes/hybrid-40b-v1.json`](recipes/hybrid-40b-v1.json) targets 40B tokens,
each source read once (`max_passes: 1`):

| Source | Share | Pinned input | Shards |
| --- | ---: | --- | ---: |
| ClimbMix | 70% | `karpathy/climbmix-400b-shuffle` | 600 |
| Permissive code | 20% | `codeparrot/github-code-clean`, licenses MIT, Apache-2.0, BSD-2/3, ISC, CC0, Unlicense | 80 |
| Math | 10% | `HuggingFaceTB/finemath` `finemath-4plus` | 38 |

The last shard of each source is held out for evaluation. Preparation drops any
document sharing a 13-token n-gram with the test sets of GSM8K, MATH, HumanEval,
MBPP, MMLU, ARC and HellaSwag (validation), all pinned by revision.

[`data.py`](data.py) turns each upstream parquet file into a shard of uint32 GLM
ids (documents shuffled within the shard, each ending in EOS) plus per-document
token and byte counts. `PreparedMixture` reads each source's shards in a seeded
order and packs 4,097-token windows (consecutive windows share one token), each
from the source furthest below its weight; it is resumable and waits for shards
that are still being prepared, so training can start while preparation runs.

[`modal_data.py`](modal_data.py) prepares shards on Modal CPUs (8 cores, one call
per shard, up to 48 in parallel), held-out shards first and then in the order
training reads them, onto the `lm-exploration-hybrid-data` volume:

```sh
uv run modal run --detach -m model_hybrid.modal_data::main --run-id <id>
```

The launch fixes a deadline 24 hours ahead; an independent supervisor cancels
every recorded call then, and each call has `retries=0` and a timeout bounded by
the deadline. Finished shards are skipped, so another launch continues. The
coordinator writes `capacity.json` (train/held-out tokens, documents and drops
per source) and the screen writes `screen.json`.

Measured on the held-out shards (2026-10-25): ClimbMix 52.6M tokens per shard
(103 of 84,992 documents screened out), code 139M (43,776 of 126,925 files outside
the license allowlist, 740 screened out), FineMath-4+ 125M (8,281 of 104,680
screened out). The screen holds 3,225,682 unique 13-grams. At those sizes the
training shards hold about 31B ClimbMix, 11B code and 4.6B math tokens, above
the 28B/8B/4B shares.

## 720M configuration and training

[`configs/hybrid-720m-40b.json`](configs/hybrid-720m-40b.json): width 1536, 16
layers as 4 x [KDA, KDA, KDA, full] (full attention at 3, 7, 11, 15), 12 query /
4 KV heads, 12 KDA heads, head dimension 128, SwiGLU 4,224, GLM vocabulary
154,880, context 4,096, one MTP module. The trunk used for inference has
707,129,536 parameters; with the MTP module 739,972,556 (`test_hybrid_720m`
asserts both). Training: 256 x 4,096 = 1,048,576 tokens per step, warmup over
500M tokens, then constant, then linear decay to 0 over the last 15%; Muon/AdamW
as above with weight decay 0.1, gradient clipping 1.0, bf16 autocast with fp32
parameters, `torch.compile`, z-loss 1e-4, MTP weight 0.3 to 0.1.

Full attention trains with FlexAttention (`attention_kernel: "flex"`, tensor
cores, document block mask, log-sum-exp gradient for the sink). On an H100 it
agrees with our bit-exact kernel to bf16 rounding (output and gradients within
0.016–0.063, log-sum-exp within 2e-6); `"exact"` selects the exact kernel.
[`fused_loss.py`](fused_loss.py) computes the output layer, cross-entropy, z-loss
and their gradients in one chunked pass, without storing or recomputing the
[tokens, 154,880] logits, for the next-token and MTP heads.

Custom kernels in this family: the bit-exact attention kernel
(`kernels/document_attention.cu`, the correctness reference and the
`"exact"` training option; fp32 CUDA cores, 161 ms vs 3.1 ms for tensor-core
attention per layer at 8 x 4,096 tokens, so not the default), the fused
output-layer loss (+15% training throughput on one H100) and, inside it, a Triton
kernel ([`softmax_kernel.py`](softmax_kernel.py)) that turns each bf16 logit row
into its cross-entropy/z-loss gradient in place in two passes, instead of several
fp32 copies of the [rows, 154,880] block. FlexAttention and fla supply the
tensor-core attention and KDA kernels.

[`train.py`](train.py) is the torchrun trainer: rank 0 reads `PreparedMixture`
and broadcasts each batch and every stop decision (stop time, data exhausted or
late, stop request); every rank derives document ids and MTP targets, and DDP
averages gradients. Each loss is normalized by its global target count. Rank 0
writes `metrics.jsonl` (loss, z, MTP loss, learning rate, gradient norm,
tokens/s, MFU, rows per source), every `eval_every_steps` the held-out loss,
bits per byte and MTP loss per source with per-layer gate openness and sink
weight plus sink, gate-bias and KDA-decay parameters, and every
`sample_every_steps` greedy continuations of fixed prompts. Checkpoints (model,
optimizer, data cursor, RNG, progress; the last two kept) are written every
`checkpoint_seconds` and at stop; a run directory resumes from its latest one,
also with a different GPU count (the global batch is unchanged). With `hf_repo`
set, bf16 trunk weights, config and metrics go to that private HF model repo
every `hf_every_tokens` and at the end.

[`modal_train.py`](modal_train.py) runs it on Modal:

```sh
uv run modal run -m model_hybrid.modal_train::bench --gpus 1      # 30-minute call
uv run modal run --detach -m model_hybrid.modal_train::launch --run-id <id> [--gpus 8]
uv run modal run -m model_hybrid.modal_train::stop --run-id <id>  # checkpoint and exit
```

Each launch fixes a deadline 24 hours ahead (less a margin), starts an
independent supervisor that cancels the recorded segment then, and starts one
segment (`retries=0`, timeout bounded by the deadline) that stops itself 15
minutes before its timeout with a checkpoint. Launching again with the same run
ID and config resumes. MFU counts 6 x (matmul weights, including both
output-layer uses) plus attention FLOPs over the mean attended keys, against
989 TFLOPS per H100.

## Tests

From `training/`:

```sh
uv run python -m unittest tests.test_hybrid_backbone tests.test_hybrid_documents \
  tests.test_hybrid_exact_attention tests.test_hybrid_sink_gate tests.test_hybrid_kda \
  tests.test_hybrid_mtp tests.test_hybrid_optim \
  tests.test_hybrid_tokenizer tests.test_hybrid_data tests.test_hybrid_720m \
  tests.test_hybrid_fused_loss -v
```

On one H100 (30-minute timeout, no retries, refuses work 24 hours after launch):

```sh
uv run modal run -m model_hybrid.modal_checks
```

The Modal image uses the CUDA 13.0 devel base and prebuilds the kernel.
[`check_hybrid_cuda.py`](../tests/check_hybrid_cuda.py) requires the kernel's
outputs, statistics and gradients to equal `exact_attention` bit for bit (fp32 and
bf16, GQA, packed documents, lengths that are not tile multiples, head dimensions
64 and 128) and to repeat bit for bit, reports the distance to fp64 softmax
attention, runs a compiled bf16 model with `fullgraph=True`, and times the kernel.

`test_hybrid_exact_attention` checks `exp_fixed` and `log_fixed` accuracy, and
the reference's output, log-sum-exp and gradients against fp64 autograd.

The same check compares fla's `chunk_kda` with `cu_seqlens` against
`kda_reference` (output and all input gradients, within TF32 tolerance) and times
bf16 training steps of a small compiled hybrid model with the fla call outside the
compiled graph (`torch.compiler.disable`) and inside it, after warming up both.
The CUDA path needs the `hybrid` extra (`fla-core`); the Modal image installs it.

`test_hybrid_720m` checks the configuration, parameter count, schedule, FLOP
accounting, attention-kernel choice and gate/sink statistics;
`test_hybrid_fused_loss` checks the fused loss and its gradients against autograd.
`test_hybrid_data` prepares shards from small parquet files (license filter,
empty documents, contamination drops, EOS layout, byte counts) and checks the
mixture's weights, contiguity across shards, resume and the held-out windows;
`test_hybrid_tokenizer` checks the pinned constants and the recipe, and the GLM
file itself when downloaded. `test_hybrid_optim` checks that every parameter lands in exactly one group, the
role of each parameter kind, the learning rates and decay per group, and a step
plus state round trip. `test_hybrid_mtp` checks target alignment and boundary masking, the loss
against an explicit computation, that the module never sees its own target,
the weight schedule and inference without MTP. `test_hybrid_kda` compares the KDA recurrence with an independent matrix form
`S_t = (I - b k k^T) diag(a) S_(t-1) + b k v^T`, the short convolution with
`F.conv1d`, checks state and convolution resets at document boundaries, the
layout, document isolation and causality through mixed KDA/full stacks, and that
every parameter receives a gradient. `test_hybrid_sink_gate` compares the sink with an explicit softmax that has an
extra zero-value key, checks that an open gate and a -inf sink reproduce plain
attention exactly, and that the gate starts mostly open and receives gradients.
`test_hybrid_documents` checks document ids, positions and `cu_seqlens`, and that
document B's logits are identical whether it follows a different document A, or
none. The backbone tests cover output shapes, tied weights and absent biases, initialization scales,
fp32 RMSNorm statistics, causality (changing future tokens leaves earlier logits
bit-identical), attention against an explicit softmax, GQA against MHA with
duplicated key/value heads, the RoPE base and the chunked loss against
`F.cross_entropy` plus an explicit z-loss.
