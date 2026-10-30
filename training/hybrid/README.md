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

### Grouped-query attention

With the default 12 query heads and 4 key/value heads, each key/value head
serves a group of 3 consecutive query heads: query head `h` reads key/value
head `h // (n_head // n_kv_head)`. Every query head still has its own attention
pattern; the groups share only the keys and values they attend over, so
`k_proj`/`v_proj` and the KV cache are a third the size of multi-head attention.

```mermaid
flowchart TB
    subgraph queries["12 query heads (q_proj)"]
        direction LR
        q0["q0"]
        q1["q1"]
        q2["q2"]
        q3["q3"]
        q4["q4"]
        q5["q5"]
        q6["q6"]
        q7["q7"]
        q8["q8"]
        q9["q9"]
        q10["q10"]
        q11["q11"]
    end
    subgraph keys["4 key/value heads (k_proj, v_proj, KV cache)"]
        direction LR
        kv0["k0, v0"]
        kv1["k1, v1"]
        kv2["k2, v2"]
        kv3["k3, v3"]
    end
    q0 --> kv0
    q1 --> kv0
    q2 --> kv0
    q3 --> kv1
    q4 --> kv1
    q5 --> kv1
    q6 --> kv2
    q7 --> kv2
    q8 --> kv2
    q9 --> kv3
    q10 --> kv3
    q11 --> kv3
```

Multi-head attention is the case `n_kv_head == n_head` (one query per group);
multi-query attention is `n_kv_head == 1` (all queries in one group).

## Document masking

Loaders pack documents back to back, each ending in EOS. Every forward pass is
document-masked: the model derives per-row document ids from the EOS tokens in
its input (`HybridConfig.eos_id`, [`documents.py`](documents.py)
`document_ids`), so a token sees earlier tokens of its own document only and RoPE
positions restart after each EOS. Training, evaluation and sampling all call
`forward(tokens[, labels])` with no separate mask. `cu_seqlens` converts the
ids to flattened document offsets for varlen kernels.

## Attention

On CUDA, full attention is FlexAttention (`torch.nn.attention.flex_attention`,
compiled once, tensor cores) with a block mask built from the document ids: query
`q` may see key `k` when `k <= q` and both are in the same document. Blocks outside
every document are skipped. It returns the output and the fp32 log-sum-exp, whose
gradient flows back (the attention sink uses it). On CPU, `model.reference_attention`
(an explicit masked softmax, fp32 or fp64) is the model's attention.

## Attention sink and output gate

Each full-attention layer has a learned sink: one logit per head, initialized to
0, acting as an extra key with a zero value. Softmax over the scores plus the sink
equals the plain attention output times `sigmoid(lse - sink)`, where `lse` is the
log-sum-exp of the scores, so the layer applies that factor to FlexAttention's (or
the reference's) output; gradients flow through the log-sum-exp. A head can then
attend to "nothing" without spending probability on a real token.

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
[KDA, step by step](../references/kda.md) walks through the recurrence on a toy head.

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
the next document) and at the last position. `forward(tokens, labels, mtp_labels)` adds `mtp_loss`, `mtp_ce`,
`mtp_z` and `mtp_count`. The trainer combines the means as
`loss / count + mtp_weight(config, progress) * mtp_loss / mtp_count`, where the
weight decays linearly from 0.3 to 0.1 over training progress. With
`mtp_loss_fraction < 1` (default 1.0), training passes only a uniformly random
fixed-size fraction of positions through the output layer for the MTP loss (the
module itself still runs over every position) and divides by the expected count;
evaluation uses every position. At 0.5 the 720M step on one H100 runs at 57.3K
tokens/s instead of 54.8K, since the two output-layer projections are about half
of the model's FLOPs.

Inference does not run the module: `forward(tokens)` returns trunk logits, and
a model built with `mtp=False` loads the trunk weights (all keys outside `mtp.`)
and produces identical logits.

## Optimizer grouping

Each module declares the optimizer role of the parameters it creates, next to its
`__init__`: `optimizer()` in [`model.py`](model.py) and [`kda.py`](kda.py)
yields `(parameter, role)` pairs (helper `declare` in
[`trainer/roles.py`](../trainer/roles.py)). `trainer.optimizers.RoleOptimizer`
collects them by parameter identity:

| Role | Optimizer | Parameters |
| --- | --- | --- |
| `muon` | Muon, LR 0.02, momentum 0.95, weight decay 0.1 | 2D hidden matrices in blocks and MTP: q/k/v/o, attention `gate_proj`, KDA `f_proj`/`g_proj`, SwiGLU, MTP `fuse` |
| `adamw_decay` | AdamW, LR 3e-4, betas (0.9, 0.95), weight decay 0.1 | tied embedding/output table, KDA beta projection `b_proj` |
| `adamw_no_decay` | AdamW, LR 3e-4, betas (0.9, 0.95), no weight decay | norm scales, attention sinks, gate biases, KDA `A_log`, `dt_bias`, convolution kernels |

Muon is `trainer.optimizers.Muon` (five-step Newton–Schulz with the original
aspect-ratio scaling); its weight decay is decoupled, `p *= 1 - lr * 0.1`.
The learning rates are the 124M comparison's
([MUON_COMPARISON.md](../distributed/MUON_COMPARISON.md)), starting values for
the 720M pilot rather than tuned at this scale. Every group keeps its base rate
and `trainer.optimizers.set_learning_rate` applies one schedule factor to all of
them. A parameter that no module declares, one declared twice, or a non-matrix
declared for Muon raises.

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

[`data/packed_shards.py`](../data/packed_shards.py) turns each upstream parquet file into a shard of uint32 GLM
ids (documents shuffled within the shard, each ending in EOS) plus per-document
token and byte counts. `PreparedMixture` reads each source's shards in a seeded
order and packs 4,097-token windows (consecutive windows share one token), each
from the source furthest below its weight (`data.mixture.deficit_source`, the rule
`TokenMixture` uses too); it is resumable and waits for shards
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

Full attention trains with FlexAttention (above).
[`fused_loss.py`](fused_loss.py) computes the output layer, cross-entropy, z-loss
and their gradients in one chunked pass, without storing or recomputing the
[tokens, 154,880] logits, for the next-token and MTP heads.

Custom kernels in this family: the fused output-layer loss (+15% training
throughput on one H100) and, inside it, a CUDA kernel
([`kernels/softmax_gradient.cu`](kernels/softmax_gradient.cu)) that turns each bf16 logit row
into its cross-entropy/z-loss gradient in place in two passes, instead of several
fp32 copies of the [rows, 154,880] block. FlexAttention and fla supply the
tensor-core attention and KDA kernels.

[`train.py`](train.py) is the hybrid family for the common loop
([`trainer/loop.py`](../trainer/loop.py), see [trainer/README.md](../trainer/README.md)),
run as `torchrun -m model_hybrid.train`: rank 0 reads the recipe's mixture
(`PreparedMixture` or `ChatMixture`, with replay) and scatters each batch; every
rank derives MTP targets (the model derives its own document ids from `eos_id`,
which must be the GLM EOS), and DDP averages gradients. Each loss is normalized by
its global target count. Rank 0 writes `metrics.jsonl` (loss, z, MTP loss,
learning-rate factor, gradient norm, tokens/s, MFU, rows per source, peak memory per
rank), and at start, every `eval_every_steps` and at the end `validation.jsonl`:
held-out loss, bits per byte and MTP loss per source with per-layer gate openness
and sink weight, sink, gate-bias and KDA-decay parameters, and (every
`sample_every_steps` and at the end) greedy continuations of fixed prompts.
Checkpoints (`step-<step>.pt`: model, optimizer, data cursor, every rank's RNG;
the last `keep_checkpoints` kept) are written every `checkpoint_seconds` and at
stop, and a run directory resumes from its `latest.json` with the same config,
GPU count and source files. `report.json` closes each session. With `hf_repo`
set, bf16 trunk weights and config (under `export/`) and the logs go to that
private HF model repo every `hf_every_tokens` and at the end.

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

## Chat SFT

[`chat.py`](chat.py) maps OpenAI chat messages (`system`/`user`/`assistant`/`tool`,
assistant `tool_calls`, a `tools` list of function schemas) onto GLM-4.7-Flash's
own chat template, pinned by revision and SHA-256 like the tokenizer. Assistant
turns render without reasoning (`<|assistant|></think>`); calls are GLM's
`<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>` and
results `<|observation|><tool_response>…</tool_response>`. `encode` returns token
ids and a loss mask that covers each assistant turn after its prompt plus the token
that ends it (the next role token, or the closing EOS); `parse_assistant` turns
generated text back into an OpenAI assistant message.

[`recipes/chat-sft-v1.json`](recipes/chat-sft-v1.json) takes the non-reasoning
(`no_think`) files of `HuggingFaceTB/smoltalk2` SFT, pinned by revision: general
chat and system prompts, rewriting and summarization, instruction following,
tables, science, OpenThoughts3 answers and two tool-calling sets (xLAM traces,
Hermes function calling). [`sft_data.py`](sft_data.py) converts each row to OpenAI
messages (Hermes `<tool_call>` JSON or Python dicts become `tool_calls`, multiple
`<tool_response>` blocks become separate `tool` turns, instructions around a
`<tools>` block join the system prompt, turns after the last assistant reply are
cut), drops rows with reasoning or unparsable calls, conversations longer than
the context and those sharing a 13-gram with the pretraining benchmark screen,
and holds out 0.5% by hash. [`modal_sft_data.py`](modal_sft_data.py) prepares
each file in its own call (`retries=0`, refuses work 24 hours after launch):

```sh
uv run --all-extras modal run -m model_hybrid.modal_sft_data
```

`ChatMixture` packs whole conversations, in one seeded order across files, into
4,097-token windows padded with EOS; labels are `-100` outside the loss mask, and
the document mask keeps conversations apart. One pass is one epoch. The recipe's
`replay` takes a quarter of every batch's rows from the pretraining mixture
(`hybrid-40b-v1`, read from the start of its order, every token a target), so
fine-tuning keeps the pretraining objective; the run also evaluates the
pretraining held-out sources with the same document counts as the 40B run, which
measures what SFT costs there. `train.py`
selects it for recipes with `"format": "chat-sft"`, evaluates loss and bits per
byte on each file's held-out conversations, and samples greedy replies to fixed
chat prompts (one with a tool) up to the end of the turn.
[`configs/hybrid-720m-chat-sft.json`](configs/hybrid-720m-chat-sft.json) starts
from the 40B export (`init_from`, trunk only, `mtp=False`) with a tenth of the
pretraining learning rates, a 50M-token warmup and linear decay to 0 over one
epoch of the chat data (2,913 steps of 96 chat and 32 replay rows, from packing
simulated with the run's seed), and launches like pretraining with
`--config hybrid/configs/hybrid-720m-chat-sft.json`.

## Playground

[`generate.py`](generate.py) decodes incrementally in plain PyTorch (CPU or GPU):
a KV cache for the full-attention layers, convolution history and fp32 state for
KDA, one token per row per pass through the weights. Rows follow the model's
document rule (after EOS the next token starts a new document), so
`generate_batch` left-pads prompts of different lengths with EOS and each row
decodes exactly as it would alone. [`playground/`](playground) is a
Gradio app for a private Hugging Face Space: text continuation, and for chat
exports a chat tab with OpenAI tools whose replies are parsed back into OpenAI
messages. `playground/publish.py` uploads the app and the inference modules; the
Space downloads the export from the private model repo with its `HF_TOKEN` secret.

```sh
uv run --all-extras python -m model_hybrid.playground.publish \
  --space jfrick/hybrid-720m-playground --export tokens-039998980096 [--chat]
```

## Benchmarks

[`benchmarks.py`](benchmarks.py) wraps an export as an lm-evaluation-harness
model. Multiple-choice and perplexity requests are scored with the model's
document-masked forward pass on CUDA (each request is `[EOS] + context +
continuation`, grouped into five fixed shapes so the compiled attention reuses its
graphs); generation is greedy `generate_batch` cut at the task's stop strings.
[`modal_benchmarks.py`](modal_benchmarks.py) runs the same suite on one H100 per
model for the export and pinned reference models through the harness's Hugging
Face backend (SmolLM2-360M, Qwen3-0.6B, Qwen3.5-0.8B, LFM2.5-1.2B, OLMo-2-1B,
Gemma 3 270M and 1B, Gemma 4 E2B; `--models` picks a subset): HellaSwag, ARC-Easy/Challenge, PIQA, WinoGrande, OpenBookQA, SciQ and
LAMBADA zero-shot, MMLU 5-shot, GSM8K 5-shot, HumanEval zero-shot and MBPP 3-shot
(`retries=0`, 6-hour timeout, refuses work 24 hours after launch; each call writes its
results to `lm-exploration-hybrid-runs/benchmarks/<launch>/`):

```sh
uv run --all-extras modal run --detach -m model_hybrid.modal_benchmarks [--exports tokens-039998980096] [--models M,...]
```

[`modal_livecodebench.py`](modal_livecodebench.py) scores LiveCodeBench v6
(`test6.jsonl`, 175 problems, pinned revision) for the base, SFT and RL exports, the
same reference models and the dense Qwen3 (1.7–14B) and Qwen3.5 (2–9B) base models: one greedy completion per problem (up to 2,048 tokens,
stopping once a code block closes), graded inside the call's container by
[`livecodebench.py`](livecodebench.py) on every public and private test. Base models
get LiveCodeBench's prompt after one worked example of the same test mode; chat
exports get the question as a system and user turn. Results go to
`lm-exploration-hybrid-runs/benchmarks/lcb-<launch>/`:

```sh
uv run --all-extras modal run --detach -m model_hybrid.modal_livecodebench [--models M,...] [--limit N]
```

## PR-to-task data

[`pr_tasks.py`](pr_tasks.py) turns collected commits into repository tasks: the
repository at the parent commit, an issue, the commit's own test changes as the
hidden check and the commit as the reference patch. [`modal_pr_tasks.py`](modal_pr_tasks.py)
runs three stages on the `lm-exploration-hybrid-data` volume under `<name>/`:

- `select` keeps single-language commits (Python, JS/TS, Go) that change 1–3 source
  and 1–3 test files within `pr_tasks.LIMITS`, capped per repository. It runs no
  repository code.
- `verify` gives each task two disposable sandboxes: one installs dependencies with
  network, and its filesystem snapshot runs the test files without network with source
  files reverted to the parent and at the commit ([`pr_task_runner.py`](pr_task_runner.py)).
  Verified tasks have at least one fail-to-pass test and no regression.
- `issues` has GLM-5.3 (through OpenRouter) draft each task's issue from the commit
  message and the test changes only, up to four drafts until one has no leaks (quoted
  added source lines, new internal names the tests never use, mentions of tests or the
  commit). Launch it with `OPENROUTER_API_KEY` set; the key is passed to that run only.

```sh
uv run modal run -m model_hybrid.modal_pr_tasks::select [--count 1000] [--name pr-tasks-pilot]
uv run modal run --detach -m model_hybrid.modal_pr_tasks::verify [--limit N] [--workers 48]
uv run modal run --detach -m model_hybrid.modal_pr_tasks::issues [--limit N]
```

## Tests

From `training/`:

```sh
uv run python -m unittest tests.test_hybrid_backbone tests.test_hybrid_documents \
  tests.test_hybrid_sink_gate tests.test_hybrid_kda \
  tests.test_hybrid_mtp tests.test_hybrid_optim \
  tests.test_hybrid_tokenizer tests.test_packed_shards tests.test_hybrid_720m \
  tests.test_hybrid_livecodebench tests.test_hybrid_pr_tasks \
  tests.test_hybrid_fused_loss tests.test_hybrid_chat tests.test_hybrid_generate -v
```

On one H100 (30-minute timeout, no retries, refuses work 24 hours after launch):

```sh
uv run modal run -m model_hybrid.modal_checks
```

The Modal image uses the CUDA 13.0 devel base with torch 2.14.1.
[`check_hybrid_cuda.py`](../tests/check_hybrid_cuda.py) compares FlexAttention's
output, log-sum-exp and gradients (including through the log-sum-exp) with fp64
softmax attention (fp32 and bf16, GQA, packed documents, a one-token document,
lengths that are not block multiples, head dimensions 64 and 128), checks document
isolation in a compiled bf16 model, and times FlexAttention against causal SDPA.

The same check compares fla's `chunk_kda` with `cu_seqlens` against
`kda_reference` (output and all input gradients, within TF32 tolerance) and times
bf16 training steps of a small compiled hybrid model with the fla call outside the
compiled graph (`torch.compiler.disable`) and inside it, after warming up both.
The CUDA path needs the `hybrid` extra (`fla-core`); the Modal image installs it.

`test_hybrid_720m` checks the configuration, parameter count, schedule, FLOP
accounting and gate/sink statistics;
`test_hybrid_fused_loss` checks the fused loss and its gradients against autograd.
`test_packed_shards` prepares shards from small parquet files (license filter,
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
