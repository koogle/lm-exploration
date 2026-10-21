# Probability distillation into the existing 124.4M model

This experiment starts from the completed code-heavy 124,389,632-parameter
checkpoint, preserving the custom frozen 32K tokenizer and architecture. The
first teacher is **Qwen/Qwen3-4B-Base**, revision
`906bfd4b4dc7f14ee4320094d8b41684abff8539`. The 5B-token goal is a scale-up target;
the launcher currently implements bounded qualification sessions, not a 5B run.

## Teacher choice (verified 2026-10-21)

| Candidate | Total weights | Approx. BF16 weights only | Practical starting GPU |
| --- | ---: | ---: | --- |
| [Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) | 4.02B | 8.0 GB | L40S; selected pretrained base |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4.66B | 9.3 GB | L40S; newer post-trained alternative |
| [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 30.53B | 61.1 GB | H100/H200; code-specialized, post-trained |
| [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | 31.22B | 62.4 GB | H100/H200; post-trained |
| [DeepSeek-Coder-V2-Lite-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base) | 15.71B | 31.4 GB | L40S/H100; older code base |
| [Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct) | 1,026B | 2,053 GB | Large multi-GPU deployment, unsuitable first teacher |

These are storage estimates, not measured serving memory or throughput. MoE
active parameters do not determine weight residency. Allow additional KV cache,
activations and runtime workspace; quantization changes both memory and targets.
Qwen3 base is from April 2025, not the newest release. It is selected for the
base objective, simple architecture and modest size, not a claim of best quality.
The Qwen candidates use Apache 2.0; GLM uses MIT; check the DeepSeek/Kimi model
cards for their own license terms.

## What is actually distilled

Direct token KL is invalid across the two vocabularies. On existing code text:

1. Reconstruct raw bytes from student ByteLevel BPE pieces, tokenize with Qwen,
   and find positions with identical inclusive text prefixes. No lossy Unicode
   decoding of individual tokens. Skip windows that cut UTF-8 characters or
   whose text changes under the teacher's Unicode normalization.
2. Compute the frozen teacher's full next-token distribution with Transformers
   on the same GPU. Sum probabilities into **256 first-byte buckets plus one
   special/unused-token bucket**. No top-k truncation or discarded tail mass.
3. Train the student with `0.5 * code_token_CE + 0.5 * KL(teacher_byte || student_byte)`
   at temperature 1. KL uses only shared boundaries; CE uses every target in
   accepted windows. Both components are means over their respective positions.

This is an experimental, lossy first-byte marginal objective, **not full token
KL**, sequence distillation, or proof of capability transfer. It cannot teach
all alternatives sharing the same first byte. Student and teacher use their
native segmentation; the student additionally receives its native BOS. The
current context is 256 student tokens, a reduction from the parent's 8K training
context. Weights remain compatible with longer contexts, but this pilot does
not validate long-context quality. No chat template or generated conversation
is needed for the base teacher. An instruct teacher would need a separately
specified prompt/assistant-mask contract.

The teacher probabilities are generated online and consumed immediately, avoiding
a huge 5B-token cache. Even 257 BF16 values per token would be about 2.57 TB
before metadata. Standard vLLM top-k logprobs are insufficient for this exact
marginal. The teacher uses Transformers for now; a vLLM worker would need full
logprobs or a GPU-side marginalization hook before claiming objective parity.

## Data and lineage

- Parent: `lm-exploration-training/simple-llm-code60-unique-10b-20261020-r01-s04/step-000153342.pt`.
  Its report records ~20.879B lifetime tokens; this is continued training, not
  fresh initialization. Model weights load strictly; AdamW starts fresh at 1e-5.
- Frozen tokenizer hash: `add5b2c7542e4efded329735825ecdcb94df1432617c3d3e4b7508aa459432f9`.
- Snapshot: `scholarly-code/readiness-20261020-v2/snapshots/code60-onepass-10b` on
  `lm-exploration-mixture-data`. The pilot reads only its multilingual code
  database, not the patch database or non-code mixture.
- Preserve train/validation splits, conflict filtering and held-out parent/hash
  exclusions. One pass only. Verify tokenizer token counts, snapshot SHA and
  expected database byte size. Size checks are not full database content hashes.
  This snapshot has documented pending audits; the pilot is not a new corpus audit.
- Report CE on 16 fixed validation windows, before and after training. This is
  diagnostic, not a coding benchmark or a representative capability estimate.
- Outputs use the dedicated `lm-exploration-distillation` volume. Parent checkpoints
  and corpus are never modified. Checkpoints include weights, optimizer, RNG,
  exact data cursor, token counts, teacher pin, configuration and source hashes.
  Resume goes to a new run directory and rejects settings/source changes.

## Run from repository root

Install `modal[api-proxy-support]==1.6.2` and `truststore`. In a managed environment,
run the CLI through Python after `truststore.inject_into_ssl()` (see existing
training instructions). Local tests additionally use torch, transformers 4.57.1,
tokenizers 0.22.2 and numpy. GPU dependencies are pinned in the Modal image.

```sh
python -m unittest discover -s distillation/tests -v
ruff check distillation
modal run --detach -m distillation.modal_app \
  --run-id UNIQUE_RUN_ID --target-tokens 262144 --max-seconds 1200

# Total token target includes prior exposures; use identical source for resume.
modal run --detach -m distillation.modal_app \
  --run-id NEW_RUN_ID --resume OLD_RUN_ID/step-000001000.pt \
  --target-tokens 1000000 --max-seconds 1200
```

One L40S, no retries, 30-minute hard timeout with a 20-minute cooperative limit
including teacher/model loading. The last 10 minutes reserve checkpoint/error
handling time. Checkpoint every 500 updates and at normal exit. No automatic
continuation. Source exhaustion stops; budgets never wrap into extra epochs.
Unicode/normalization skips advance the cursor and are counted separately.
Token budgets count student loss-bearing exposures, not teacher tokens or
aligned positions. A final window that exceeds the remaining budget is not
consumed. Worker failures preserve committed artifacts; uncheckpointed updates
can be lost. Avoid concurrent writers using the same run ID.

## vLLM student and harness alignment

`export.py` converts the native architecture to standard `LlamaForCausalLM`:
Q/K coordinates are permuted from adjacent-pair RoPE to split-half RoPE; QKV
weights/biases are split; SwiGLU, projection biases, RMS epsilon, tied embeddings
and tokenizer IDs are preserved. Export requires numerical logit parity first.
Local tests also compare cached decoding to native full-sequence decoding.

```sh
modal run -m distillation.modal_vllm \
  --checkpoint RUN_ID/step-000001000.pt
# In a vLLM environment with the resulting export mounted locally:
vllm serve /path/to/step-000001000-hf --dtype float32 --max-model-len 256
```

The bounded L4 qualification checks greedy tokens and log probabilities on five
code prompts using vLLM 0.11.0. Its Llama backend supports both attention and MLP
biases. The [L4 GPU check passed](results/vllm-parity-20261021.json): all five greedy
tokens matched, with maximum log-probability error 1.16e-5.
For evaluation use raw completion prompts, exactly one BOS, identical prompt
IDs, temperature, stop IDs and context truncation. Do not silently apply a chat
template. A serving endpoint alone does not establish harness parity; expanded
coding evaluation and BF16 parity remain follow-up qualification.

## Measured first pilot

The [first report](results/pilot-20261021.md) records 261,907 trained tokens at
1,711 tokens/sec on one L40S. Held-out CE worsened from 1.2724 to 1.3782; this
recipe has not earned a 5B launch. The actual checkpoint and negative result are
preserved.

## Before scaling to 5B

Measure sustained accepted student tokens/sec, aligned boundary coverage, peak
memory, CE and byte KL. Compare with a matched CE-only control and fixed coding
tasks before attributing improvements to distillation. Estimate GPU hours as
`5e9 / measured_tokens_per_second / 3600`; add loading, evaluation, checkpoints,
CPU, volume and failed-run overhead using actual Modal prices. This online,
unbatched implementation prioritizes correctness; batching teacher windows and
student accumulation will likely be necessary for economical scale. Confirm
selected code-only capacity after exclusions. The whole snapshot's ~9.7B code
counter is not a promise that this single database supplies 5B eligible tokens.

## Larger held-out checkpoint comparison

```sh
modal run --detach -m distillation.modal_evaluate --run-id UNIQUE_EVALUATION_ID
```

This evaluates the parent and the three saved pilot checkpoints on an identical
sample: up to 512 validation documents, at most eight 256-token windows per
document (up to 1,048,576 targets). Documents follow deterministic content-hash
order. Reports retain the exact window/token hash and per-document loss sums
for paired comparisons. This is a larger evaluation sample, not longer training;
it does not create new training checkpoints. Same-tokenizer CE comparisons are
required; a shared-Qwen-tokenizer run needs its own fixed evaluation encoding.

For the shared-tokenizer follow-up, proposed training milestones are 1M, 10M and
100M student tokens, then 1B/5B if the learning curves support scaling. Compare
against a CE-only control starting from the same resized embedding table and
backbone. Keep data order, token exposure, optimizer and evaluation sample fixed.
Report code loss, teacher KL and coding-task correctness. The tokenizer change
requires a fresh baseline; old-tokenizer token CE cannot be compared directly
with Qwen-tokenizer token CE. These milestones are a proposed follow-up, not
runs already launched by the evaluation command above.

## Authorized 50M shared-tokenizer run

The follow-up uses `modal_shared.py` / `shared_train.py`, not the earlier byte
pilot. User authorization is **50,000,000 loss-bearing Qwen tokens**; early
validation regression is explicitly tolerated.

```sh
modal run --detach -m distillation.modal_shared \
  --run-id qwen4b-shared-50m-20261021-01
```

One H100, 512-token document-local context, four windows per microbatch and four
accumulated microbatches (up to 8,192 targets/update). Both models receive the
same IDs, with Qwen end-of-text as document prefix and EOS. Right padding receives
no loss. Teacher forcing on existing code continues; no synthetic conversations
are generated. The objective is equal-weight code CE and **full-vocabulary
KL(teacher || student)** at temperature 1, normalized by the actual number of
unmasked targets across the complete optimizer batch.

The backbone retains the original trained weights. Qwen's 151,936 output rows
increase the student to **215,910,656 parameters**, keeping tied embeddings.
Byte-identical vocabulary entries are copied from the old table; other text
entries use the mean of their old-token decomposition (byte fallback for partial
UTF-8 pieces). Special/unused rows retain seeded initialization, with end-of-text
copied from the old EOS. This is an initialization heuristic, not equivalent to
having pretrained with Qwen's tokenizer. Exact transfer counts are recorded.

Fresh AdamW uses backbone LR 1e-5 and embedding LR 1e-4, both linearly warmed over
1M tokens, then constant. Weight decay 0.1, betas 0.9/0.95, gradient clipping at 1.
A matched CE-only control has not been launched. It would be needed to attribute
quality differences specifically to teacher supervision.

Validation uses up to 128 documents / four windows per document in the shared
tokenizer, initially, at 1M tokens, every 5M tokens and at exit. The sample's token
fingerprint is recorded. Checkpoints also save every 500 updates for durability.
Milestone boundaries trim the final window exactly and resume its unused suffix;
no overshoot or dropped suffix is allowed. Snapshots include optimizer, RNG,
loader cursor, tokenizer identity, source hashes and initialization provenance.
A new-directory resume must use identical settings and archived source:

```sh
modal run --detach -m distillation.modal_shared --run-id FRESH_RUN_ID \
  --resume OLD_RUN_ID/step-NNNNNNNNN.pt
```

There is no validation-based early stopping. Stop conditions are 50M tokens,
source exhaustion, nonfinite loss/gradients, a termination signal or the operational
time limit (82,000 seconds cooperative; 24 hours hard). No automatic retries or
unbounded job chaining. The 16,384-token GPU smoke uses the same architecture,
objective and batching before launching the full request.
