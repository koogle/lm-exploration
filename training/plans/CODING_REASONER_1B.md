# Sub-1B coding reasoner: preparation and qualification

Updated 2026-11-04. The user's target is a model just below **one billion total
parameters**, trained from scratch through pretraining, mid-training, reasoning
SFT and RL, within approximately $10,000 of remaining Modal credits. The small
model below is a pipeline and mixture proxy, not the final target.

Current decision (November 5): the user selected KDA and 50% general text,
40% GitHub code and 10% CC-Math after the completed 200M-token study, and authorized
pretraining. [The target-scale launch plan](KDA_960M_PRETRAINING.md) records
throughput qualification, corpus preparation and the bounded first launch.

## Decision and first authorized run

Target **950–980M total deployed parameters**, with a strict ceiling below 1B.
The 959.8M training-parameter hybrid below is the current candidate, not a frozen
architecture: its deployed trunk is 927.6M after removing auxiliary MTP.
Adjust final dimensions toward the desired deployed size after tokenizer choice.
Settle tokenizer/teacher compatibility and target-scale throughput
before allocating the main pretraining budget.

The reusable Modal workflow check is a 127.6M **pretraining pipeline control**, not the main
model or a reasoning capability claim. Workflow:
[`sub1b-control-smoke.json`](../workflows/sub1b-control-smoke.json).
It uses 8,192 tokens (four optimizer steps), 512-token windows, one H100, a two-step qualification
checkpoint and one resumed session. Each GPU call has a 1,800-second timeout,
900 seconds of training time and zero retries; at most two GPU sessions are
allowed. The existing independent remote supervisor tracks calls and enforces
one deadline less than 24 hours from launch, including queue time. Expected GPU
spend is small; the maximum allocated GPU execution is one H100-hour, plus CPU
control costs. Modal billing is the authority; the $1,000 pilot envelope is not
permission for an unbounded job.

CC-Code remains awaiting author approval. Preserve its intended mixture slot.
The first run instead explicitly tests an independent **70% ClimbMix / 20%
permissive GitHub / 10% Nemotron-CC-Math-3+ control**, from a bounded 2,048-row
prefix per source. CC-Math data access is verified. This control introduces the
new math source without claiming to qualify the six-source treatment. The 3+
release is used as provided; a later 4+ ablation must actually filter its quality
metadata. SFT-Math-v4 and Math-v2 belong to the reasoning stages, not this control.

The local control preparation passed: 6,144 sampled rows, 6,079,184 training
and 593,995 validation tokens after filtering, and 1,463,830 conservative usable
mixture tokens. The first 100 windows followed exactly 70/20/10 and cursor resume
was exact. These are prefix-sample pipeline measurements, not corpus-quality evidence.
The active prepared recipe is
[`coding-control-ccmath-pilot.json`](../hybrid/recipes/coding-control-ccmath-pilot.json).
Upload its `prepared/` contents to `/coding-control-ccmath-pilot` on
`lm-exploration-hybrid-data` before launching the workflow.

The first run must establish finite train/eval metrics, correct source usage,
checkpoint continuity and successful export. A loss reduction on this tiny run
is not sufficient evidence of mixture quality. Published run outputs belong in
HF `jfrick/lm-exploration-results`, under `training/sub1b-pilot/`.

### Reproduce the control pilot

Run from `training/` with corpus dependencies and the Modal CLI installed:

```sh
python -m training_data.audit_coding_mix \
  --plan data/coding-control-ccmath-sample-v1.json \
  --out artifacts/coding-control-ccmath-pilot \
  --rows-per-source 2048 --windows 100
modal volume put lm-exploration-hybrid-data \
  artifacts/coding-control-ccmath-pilot/prepared /coding-control-ccmath-pilot
modal run --detach -m workflows.modal_train::main \
  --workflow workflows/sub1b-control-smoke.json --run-id sub1b-control-20261104-01
```

The audit refuses an existing output directory. Compare the generated recipe with
the checked-in active recipe before uploading; a changed source sample requires a
new recipe identity and run ID. In the managed cloud environment, use the existing
`python data/eu/modal_cli.py` wrapper in place of `modal` to retain CA trust, with
`modal[api-proxy-support]` installed. Do not bypass TLS verification or the proxy.

Stop gracefully with `modal run -m workflows.modal_train::stop --run-id <run>`.
The data and checkpoint volumes persist after an app exits. Preparation evidence:
[HF control audit](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/control-ccmath-20261104/preparation).

### Completed original checks

The original November 4 pilot trained 131,072 tokens in 64 steps, but the
workflow reported failure because final evaluation included optional generated
samples while resumed evaluation did not. Non-sample validation matched exactly.
[PR #145](https://github.com/koogle/lm-exploration/pull/145) fixed the comparison.
The November 5 rerun completed all 64 steps and passed the checkpoint/resume
workflow. Its [reports](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/control-ccmath-20261105/results)
are archived on HF. Following those checks, the reusable configuration was reduced
to four steps and one validation document per source. No additional GPU run was
launched for that config-only reduction; startup and sampling can still dominate
these tiny checks. The pilot establishes plumbing, not coding ability.

## Architecture candidate

[`hybrid-960m-model.json`](../hybrid/configs/hybrid-960m-model.json) defines a
candidate in our implemented hybrid family: 24 layers, width 1536, SwiGLU width
4096, 12 query/KDA heads, four KV heads, head dimension 128, and three KDA layers
per full-attention layer. It retains RMSNorm, QK normalization, gated full
attention, tied embeddings, document state resets, RoPE and an auxiliary MTP head.
Initial context is 4096; extend to 8K–32K for reasoning after throughput and
retention checks. This file is model dimensions, not a runnable large-job config.

Counted by constructing the actual implementation on PyTorch's meta device:

| Component | Parameters |
| --- | ---: |
| All training parameters, including MTP and tied embedding once | 959,843,116 |
| Auxiliary MTP module | 32,253,196 |
| Inference trunk without MTP | 927,589,920 |
| Tied embedding/output table | 237,895,680 |

The existing GLM tokenizer makes reuse and testing easy, and already has dedicated
`<think>`/`</think>` IDs. Its large vocabulary is a real cost: nearly a quarter of
parameters are embeddings. Before the final freeze, compare its bytes/token and
throughput against a smaller vocabulary on the intended corpus. A tokenizer change
requires fresh preparation and changes teacher-logit distillation compatibility.

This is a modern, implemented candidate, **not a measured state-of-the-art
architecture**. Qwen3.5 uses a related 3:1 Gated DeltaNet/full-attention pattern;
KDA and Gated DeltaNet are not identical. MobileLLM-R1.5 demonstrates strong small
reasoning with a conventional attention decoder. A matched-compute full-attention
control should qualify the hybrid choice before the main run. Avoid combining
unmeasured architecture changes with the first mixture experiment.

Sources: [Qwen architecture documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md),
[MobileLLM-R1.5](https://huggingface.co/facebook/MobileLLM-R1.5-950M),
[our KDA implementation](../hybrid/kda.py).

## Benchmark targets and corrections to the frontier reference

These are author-reported results, not measurements by us or proof of an exhaustive
current leaderboard. Refresh and rerun selected controls before claiming a record.

| Reference | Stored parameter count | LiveCodeBench v6 |
| --- | ---: | ---: |
| MobileLLM-R1.5-950M | 949,685,760 | 29.1% |
| Qwen3-0.6B, as evaluated in the MobileLLM card | approximately 596M | 14.9% |
| K2 Horizon “0.9B” | **1,078,285,824** | 37.4% |

The first two scores are average pass@1 over 16 samples. K2's card recommends
high reasoning effort and at least 32K output tokens; its exact benchmark protocol
must be reproduced. The HF safetensors metadata shows that K2 exceeds our strict
total-parameter ceiling, despite its name. It remains a useful slightly-larger
control. The November 4 [reference refresh](../references/open-model-frontier-2026-11-01.md)
adds these models and corrects K2's size. Its mixed-version chart is not a controlled leaderboard.

Sources: [R1.5 evaluation](https://huggingface.co/facebook/MobileLLM-R1.5-950M#evaluation),
[K2 model card](https://huggingface.co/IFM/K2-Horizon-0.9B), HF model metadata checked
2026-11-04. Also correct the token-budget interpretation: MobileLLM's approximately
2T unique tokens are resampled to roughly 4.2T training tokens, not just 2T total
training exposure ([ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c62fb98c25c2318bf3aca24d3b0fc85e-Abstract-Conference.html)).

Primary outcome: held-out coding pass@1 at a fixed inference budget, with 2K, 8K
and 32K output-budget curves after context qualification. Compare reasoning-on and
direct-answer modes on identical tasks; report truncation, loops, malformed outputs,
reasoning length, wall time and answer correctness. Separate development tasks
from the untouched final set and report task-level uncertainty. Sparse
Terminal-Bench solves are a secondary goal, with repeated trials and distinct
solved tasks reported separately.

## Original mixture proposal

The [KDA/data study](KDA_DATA_STUDY.md) owns the revised foundation weights and
comparison sequence. The table below preserves the original audit hypothesis.

The preparation manifest is
[`coding-mix-sample-v1.json`](../data/coding-mix-sample-v1.json). Its source commits
are pinned. It is an **audit manifest**, not a full-capacity training recipe.

| Source | Token weight | Purpose |
| --- | ---: | --- |
| ClimbMix | 30% | General language and broad knowledge |
| Permissive GitHub code | 20% | Implementations and real code structure |
| Nemotron Code Concepts | 20% | Programming concepts and problem/solution pairs |
| Nemotron-CC-Code (access pending) | 15% | Coding web documents and explanations |
| FineMath-4+ | 10% | Mathematical reasoning foundations |
| Nemotron STEM-SFT | 5% | Worked technical explanations |

The user chose to keep Nemotron-CC-Code pending, retaining its 15% allocation.
The user subsequently accepted its terms. An authenticated recheck still returned
**403 GatedRepoError**, explicitly stating that access is awaiting review from the
repository authors. The current manifest
marks its access pending, and the audit refuses to start until access is verified
and that marker is updated. Do not substitute another source or redistribute its
weight. The earlier Scientific Coding sample remains a historical preparation
diagnostic; it does not qualify the intended mixture.

ClimbMix is research-only CC-BY-NC; GitHub rows retain the existing permissive
license filter. Code Concepts and STEM-SFT are CC-BY-4.0; Nemotron-CC-Code requires
its NVIDIA data access terms. The earlier Scientific Coding diagnostic has
CC-BY-SA/GFDL terms in its dataset card. This remains the project's research use
case. Preserve record-level license/provenance and review distribution obligations
before publishing weights. Source access is not a substitute for that review.

Nemotron motivates instructional data during pretraining, but these weights are
our hypothesis. In particular, do not interpret AST-valid Code Concepts examples
as execution-verified solutions. The source's concept selection used HumanEval;
measure independent task transfer rather than optimizing a HumanEval score.

Sources: [Nano report, Figure 3](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf#page=9),
[Code Concepts experiment](https://huggingface.co/blog/nvidia/synthetic-code-concepts),
[Specialized-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1),
[Specialized-v1.1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1.1).

## Preparation smoke test

Run from `training/`, with the corpus dependencies installed:

```sh
python -m training_data.audit_coding_mix \
  --plan data/coding-mix-sample-v1.json \
  --out artifacts/coding-mix-cc-code-pilot \
  --rows-per-source 512 --windows 100 \
  --model-config hybrid/configs/hybrid-128m-coding-smoke.json
```

Run this after source access is verified and its pending marker is updated.
The command refuses an existing output directory. It reads a bounded prefix of
each pinned source, records upstream row positions and normalized content hashes,
removes exact normalized duplicates across sources, and assigns 10% to validation
by content hash. It then uses the production `prepare_shard` and `PreparedMixture`
code with the real pinned GLM tokenizer. Remote Parquet column chunks can be much
larger than the requested row sample; this is a row bound, not a network byte cap.

Checks: existing benchmark screen plus all six pinned LiveCodeBench JSONL releases
(question/starter code); valid token IDs; document EOS boundaries; byte/token
accounting; shifted labels; mixture weights within one window; exact cursor resume;
and preservation of reasoning special tokens. The optional model check runs two
CPU SGD updates on a fixed short real-data batch, including MTP; finite gradients
and decreasing minibatch loss are required. This does not qualify CUDA kernels,
the production optimizer or training quality. No benchmark solution/test payload
is used as a training example. JSONL support is in the shared screen builder.
An additional 16-row OpenThoughts3 probe checks the existing thinking-preserving
mid-training renderer, answer retention and the number of complete examples that
fit 4K, 8K and 32K. This source is a separate reasoning-stage probe, not another
weighted source in the pretraining mix.

Outputs are ignored local artifacts: raw sampled Parquet with provenance,
`screen.npy`, prepared shards, `recipe.json`, and structured `audit.json`. Failed
runs retain their failure status. These are prefix samples with an exact-duplicate
split, **not** a global near-duplicate audit or a benchmark-contamination guarantee.
Before scaling, stratify samples across shards, split by repository/problem family,
run cross-source near-deduplication, inspect source quality/length/language, and
measure post-filter capacity with our tokenizer. Avoid repeatedly tuning to LCB v6;
reserve a final dated evaluation window.

### Measured preparation check, 2026-11-04

Local artifacts: `training/artifacts/coding-mix-preparation-20261104/`.
`plan.json` preserves the exact six-source manifest used by the initial audit;
`reasoning_probe.json` records the separately executed probe now included in the
current manifest. This earlier diagnostic used Scientific Coding in the 15% slot,
before the user's source preference arrived. Its results apply to that substitution
only; the intended CC-Code mixture remains pending. It is not a published experiment.

- Read 512 rows from each of six sources (3,072 total). One GitHub row failed the
  license filter. The benchmark screen rejected another 187 rows; 2,884 remained.
- Prepared 3,912,142 training tokens and 422,684 validation tokens. Conservative
  single-pass mixture capacity is 723,699 tokens, enough for the 8,192-token
  plumbing configuration. Total stored tokens exceed usable mixture capacity
  because source lengths differ; the limiting source must not silently repeat.
- Across 100 windows, actual source counts were exactly 30/20/20/15/10/5.
  Token IDs, document endings, byte accounting, label shifting and exact cursor
  resume passed. The 127,631,028-parameter model's two CPU SGD steps had finite
  gradients and fixed-minibatch objective 13.2046 → 12.7672 (2×16 tokens).
- Code Concepts had 115/512 screen rejections. These are conservative 13-token
  overlap hits, **not confirmed contamination**; common code fragments can trigger
  them. Review hit attribution before preparing the full source. All 460 sampled
  training-partition Code Concepts rows parsed as Python before screening; that
  does not verify their solutions.
- The separate 16-row OpenThoughts3 probe retained thinking and final answers.
  Lengths were 10,707–16,788 tokens: 0/16 fit 4K or 8K, 16/16 fit 32K, and 3/16
  matched the benchmark screen. These are a prefix sample, not population rates.
  None of the six foundation sources' inspected training rows contained explicit
  `<think>` markers; reasoning-format supervision therefore needs its own stage.

The historical substitute's data-path check passes. The intended CC-Code mixture,
a larger representative corpus and the GPU training path still need qualification.
No Modal GPU job was launched for these checks.

## Small model, then the main run

[`hybrid-128m-coding-smoke.json`](../hybrid/configs/hybrid-128m-coding-smoke.json)
uses the same tokenizer, 3:1 mixer pattern and MTP objective: 12 layers, width 512,
127,631,028 total training parameters. Its 8,192-token target is only a plumbing
check. The launcher must verify the generated recipe has at least this single-pass capacity before scheduling it.
It points to the frozen control recipe staged on the hybrid data volume.
The sample is prepared locally; the workflow runs the production GPU trainer. GPU jobs
must use the existing independent deadline supervisor and a short execution cap.

After preparation passes, run a bounded GPU smoke to check finite loss/gradients,
training improvement on a fixed minibatch, held-out evaluation, checkpoint resume,
and matching CPU/CUDA behavior. A tiny successful run does not establish coding
ability or mixture quality.

The next authorized comparison is the four-arm, approximately 128M study in
[KDA_DATA_STUDY.md](KDA_DATA_STUDY.md#authorized-fast-comparison--november-5):
KDA versus all-full attention, each on 70/20/10 and 40/50/10
ClimbMix/GitHub/CC-Math, at 200M tokens per arm. It holds tokenizer, held-out
corpus, context, optimizer and training budget fixed. The broader revised
foundation mixture remains conditional on source access and separate qualification.
Small-model results guide the next experiment; they do not establish target-scale
coding or reasoning capability.

## Evidence from failed agent-trajectory training

The merged [public-trajectory and scale-500k findings](HYBRID_V2.md#public-trajectory-mixes-on-eight-h100s-2026-11-02)
constrain this plan. These experiments used **trajectory SFT**, not policy-gradient
RL, even when their source trajectories came from agent/RL environments.

- Qwen3-0.6B completed approximately 494k trajectories, finishing at loss 0.344,
  yet resolved **0/179** held-out PR tasks. It repeated about 75% of calls and
  submitted only two episodes. The prepared mix contained 7.41B total tokens,
  including 2.57B assistant tokens.
- Source repetition was much lower: mini-coder 10.4%, Open-SWE-Traces 2.5%, our
  GLM trajectories 0.7%. Filtering repetitive teacher traces alone does not address
  the student's observed failure to recover during long episodes.
- The roughly 49k mixed arm resolved 3/179 with 100 turns and **0/89** on
  Terminal-Bench 2.1; 80 terminal trials timed out. Low SFT loss, patches and longer
  episodes did not establish reliable task completion.
- The same-mix 1.7B control stopped before a full pass; its 64k-trajectory evaluation
  resolved one task. Separate mini-coder-1.7b resolved 11/179 in its own format with
  40 turns. Size, mixture and format effects are not isolated. The 720M hybrid run
  stopped before its first checkpoint and is not a completed negative result.

**Decision:** retain the near-1B target for single-problem coding and explicit
reasoning training. More agent trajectories, lower SFT loss, or moving from 0.6B to
1B cannot be assumed to establish agentic capability. These findings do not directly
measure LiveCodeBench or rule out gains from verifiable code reasoning.

Allocate no large agent-trajectory SFT run from this plan's budget until a bounded
pilot shows independent resolved-task improvement and substantially better completion
and repetition behavior. Keep task split, format, sampling, turn limit and output
budget fixed for comparisons. Long-episode agent work remains secondary to coding
pass@1. If revisited, test corrections on student-generated failures or verifiable
outcome RL in a small ablation; these are hypotheses, not demonstrated fixes.

Reuse the merged shared-loop reference adapter and chat-SFT layout for controls.
Track conversation and assistant tokens separately from packed-window and padding
tokens when comparing throughput or allocating budget. The current 128M pretraining
plumbing run keeps its launched commit and recipe; it does not consume agent
trajectories and cannot answer this capability question.

## Stage gates and token budgets

| Stage | Planned scope | Required evidence before advancing |
| --- | --- | --- |
| GPU plumbing | Current 128M / 8K-token control | Finite metrics, two-step checkpoint/resume continuity, export and source audit |
| Mixture qualification | 128M / 0.5–1B tokens per selected arm, within pilot budget | Representative deduplicated corpus; held-out BPB and executable-code improvement; comparable cooldowns |
| Architecture/tokenizer qualification | Short target-scale throughput run; matched-compute attention control where affordable | Exact parameter count below 1B, measured dollars/token, teacher compatibility, stable optimization |
| Main pretraining | Approximately 100–130B tokens only if measured costs fit $5,000 | Frozen source capacity, contamination/split review, stable loss, regular cooled checkpoints and independent coding evaluations |
| Mid-training | Code-heavy quality/length curriculum, 4K → 8K → 32K if validated | Complete reasoning traces; long-context retention; no unacceptable general-data regression |
| Reasoning SFT | Verified code majority plus math; thinking and direct-answer modes | Explicit thought/answer/stop supervision; correct masks; improved held-out coding at fixed output budgets |
| Distillation and RL | Compare from same SFT checkpoint; short-budget tasks before long rollouts | Teacher cost measured; compatible logits or text distillation; verifiable execution rewards; no reward-only gains without independent transfer |

Do not schedule all stages automatically. Token budgets beyond the first control
remain conditional on measured cost and data capacity. Reasoning SFT is required;
RL is a planned measured experiment, not a substitute for insufficient pretraining.
For SFT, start with tool-free Nemotron-SFT-Math-v4 and compare shorter Math-v2
traces. Split by seed problem across versions and preserve licenses/attribution.
Choose code traces with executable checks and benchmark decontamination. Reserve
an untouched dated LCB window and report benchmark version, pass@1, output budget,
completion rate, token cost and uncertainty. Milestones of 5–10%, 15–20% and near
29% LCB-v6 are progress/stretch targets, not forecasts; Terminal-Bench is secondary.

See [how open models train reasoning](../references/reasoning-training-2026-11-04.md).

## Reasoning is trained behavior

Pretraining should build code and explanation foundations. Mid-training introduces
complete problem → reasoning → solution examples with a controlled difficulty and
length curriculum. Reasoning SFT then explicitly supervises the assistant's
`<think>...</think>` span, final answer and stop token. Retain a direct-answer
subset with explicit prompting so both modes can be compared. Preserve complete
traces; do not silently cut off the answer to fit context. Tokenizer support alone
does not establish reasoning ability, and traces are not guaranteed faithful.

Compare execution-reward RL with teacher-guided/on-policy distillation after SFT;
pay for teacher generation/scoring explicitly. Reward executable correctness and
measure independent task transfer. Do not reward the mere presence or length of
thinking tags. The current changes validate token preservation; they do not add a
new reasoning SFT trainer or claim reasoning capability. In particular,
`chat_sft.convert` currently rejects thinking traces and `chat.encode` uses the
non-thinking template. The reasoning SFT stage must implement and test an explicit
thinking mode and assistant loss masks before launch; using the current chat-SFT
recipe would discard the behavior we intend to teach.

## Budget envelope

Reserve $1,000 for preparation, small-model tests and evaluation; $5,000 for main
pretraining; $1,500 for mid-training/reasoning SFT; $1,500 for RL/distillation; and
$1,000 contingency. This totals $10,000 and is a ceiling, not an instruction to
launch all stages. Use measured throughput and source capacity to set final token
targets. A linear extrapolation from the old run suggests roughly 130B pretraining
tokens at 960M for $5,000, but vocabulary/output projection, sequence lengths and
teacher costs make a new throughput pilot necessary. Frontier quality at this
budget is a research goal, not an expected result.
