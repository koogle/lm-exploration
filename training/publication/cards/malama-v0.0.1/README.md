---
license: cc-by-nc-4.0
language:
- en
tags:
- code
- reasoning
- linear-attention
- kimi-delta-attention
- hybrid
- reinforcement-learning
---

# Malama v0.0.1

Malama v0.0.1 is an experimental 0.93B-parameter language model that reasons in a `<think>` span before writing Python. Its base model was
pretrained from scratch on **100B tokens** (602 H100-hours, about 5.8e20 FLOPs), followed
by roughly 12B tokens of context extension, code mid-training and reasoning SFT, and 650
steps of reinforcement learning against unit tests.

On LiveCodeBench v6 with thinking it scores **19.1 pass@1**, ahead of Qwen3-0.6B (15.2)
and behind MobileLLM-R1.5-950M (29.0) and Qwen3-1.7B (31.0), alongside **50.1** on
HumanEval and **41.6** on MBPP.

**What this model is—and is not.** For a ~112B-token total budget, those scores are okay,
especially compared to sub-2B models trained on trillions of tokens. To be clear, **Malama v0.0.1 is by far not a great model.** Its base model sits at the weak end
of its size class on raw text perplexity—likely because it is undertrained at 100B
pretraining tokens compared to multi-trillion-token baselines. It is narrow (Python
programming problems only), and medium and hard competitive-programming problems remain
mostly out of reach.

While the weights were trained from scratch, the coding and reasoning improvements lean heavily on
stronger teacher models and synthetic datasets — traces we generated ourselves
(Qwen3-30B-A3B and DeepSeek V4 Flash) and public teacher-generated datasets (DeepSeek-R1
traces and synthetic code corpora) — combined with unit-test RL.

This card documents the released checkpoint, other branches, and the exact data and teacher models used. We show what did and did not work along the way.

![LiveCodeBench v6 by model size](charts/livecodebench-by-size.png)

## Model

| | |
| --- | --- |
| Architecture | Decoder-only hybrid: Kimi Delta Attention (KDA) linear-attention layers with full attention every fourth layer |
| Layers | 24 (pattern `kda, kda, kda, full`: 18 KDA and 6 full-attention layers) |
| Width | 1,536 model dimension, 4,096 SwiGLU MLP hidden |
| Heads | 12 KDA heads; 12 query and 4 key/value heads (GQA) in full-attention layers; head dimension 128 |
| Full-attention extras | QK-norm, RoPE (base 1,000,000), learned per-head attention sink, per-channel sigmoid output gate |
| Context | 32,768 tokens (extended from 4,096 after pretraining) |
| Vocabulary | 154,880 (GLM tokenizer and chat template, `zai-org/GLM-4.7-Flash`) |
| Parameters | 959.8M during pretraining (with a one-block multi-token-prediction head); about 928M in the released weights |
| Optimizer | Muon for matrix parameters, AdamW for embeddings/norms/biases—used through pretraining, SFT and RL |
| Weights | bfloat16 safetensors, `tokens-000288341556/` |

## How it was trained

The released checkpoint is the end of this six-stage chain. Each stage starts from the
weights of the stage before it.

| Stage | Tokens or steps | What it adds |
| --- | --- | --- |
| 1. Pretraining ([Malama Base](https://huggingface.co/jfrick/malama-base)) | 100B tokens at 4K context | Web text (50%), code (40%) and math (10%), trained on 8 × H100 in 75.3 hours |
| 2. Context extension | 2B tokens at 32K | Long documents: repository-grouped code and books (50% replay) |
| 3. Code mid-training | 5B tokens at 32K | Synthetic Python Q&A, snippets, unit-tested answers and GitHub code (45% replay) |
| 4. Reasoning SFT v5 | 4.89B tokens at 32K | Chat format with `<think>` traces on contest problems (10% replay) |
| 5. Contest RL | 250 steps | Verifiable unit-test RL on contest problems (stops repetition loops, raises LCB) |
| 6. Mixed RL | 400 steps | Verifiable unit-test RL on contest and "write a function" problems together |

RL uses **Dr. GRPO** (the group-mean baseline of GRPO without dividing by the group
standard deviation) with **Muon**, 64 problems × 8 sampled answers per step, up to 8,192
answer tokens, on 8 H100s. An answer scores `1` when its program passes every test in a
sandbox, `0` when it fails, and `−0.5` when it does not finish.

## What we tried and what we learned

Building a small reasoning model on a modest token budget turned up several clear lessons
(and a few dead ends):

- **A 3:1 KDA linear-attention hybrid trains cleanly through 32K context and RL.** Only a
  quarter of the layers use full attention (with QK-norm, learned attention sinks and
  output gates); the other 18 layers are Kimi Delta Attention recurrences reset at document
  boundaries. Using Muon for matrix parameters worked not only in pretraining and SFT,
  but also directly in Dr. GRPO RL.
- **Most late pretraining gain came from the LR decay phase.** On Malama Base, held-out
  bits per byte barely moved between 50B and 85B tokens at constant learning rate (about
  1% or less), then dropped 13–28% across sources during the final 15B-token decay.
  Upweighting Python in the pretraining code mix (from 4% to ~8%) improved held-out Python
  bpb by 4.5–5.5% in a matched 128M ablation, whereas upweighting starred non-Python
  repositories did not show a measured held-out bpb gain over unstarred code.
- **RL mostly fixed repetition loops rather than teaching new algorithms.** Straight after
  Reasoning SFT v5, **74%** of LiveCodeBench answers were cut off unfinished (after SFT
  v3, 79% were truncated, and 97% of those were stuck in verbatim repetition loops).
  Adding the `−0.5` unfinished-answer penalty in Dr. GRPO cut truncation to **0.5%** in
  v0.0.1 and pushed pass@1 from 12.8 to 19.1, while pass@6 moved much less (23.3 to 26.6).
  In other words, RL mainly made the model reliably finish solutions it could already
  occasionally reach.
- **Contest-only RL collapsed function writing, and mixed RL recovered it.** After 250
  steps of contest-only RL, LiveCodeBench reached 18.9, but HumanEval collapsed to **10.6**
  and MBPP to **2.6**: two-thirds of its HumanEval answers were full `stdin`/`stdout`
  contest scripts when the prompt asked for a Python function. Mixing contest problems
  with function-writing problems (KodCode + MBPP train/val) in stage 6 recovered HumanEval
  to **50.1** and MBPP to **41.6** while keeping LiveCodeBench at **19.1**.
- **Prompt format details in RL matter immediately.** An earlier mixed RL pool omitted
  the test assertions from MBPP prompts; held-out MBPP stayed at 0% and we stopped the
  run after 45 steps. Stating MBPP problems with their assertions—matching the benchmark—
  resolved it.
- **More tokens or more RL steps did not automatically win.**
  - *Code mid-training (5B vs 20B):* A separate 20B-token code continuation from Base (4K
    context) raised plain-completion MBPP (33.2 → 42.4) but dropped HumanEval (50.0 →
    42.7) relative to the 5B 32K continuation. We built the reasoning line on the 5B 32K
    branch.
  - *Over-running RL:* Contest-only RL from SFT v5 peaked near step 250 (18.9) and fell
    to 17.0 by step 500.
  - *Scaling teacher traces (SFT v4 vs v5):* Scaling our own teacher traces ~14× (from
    2,234 to 31,568) actually started lower straight after SFT (12.8 vs 14.0), then
    overtook v4 after RL (18.9 vs 18.1).
  - *Greedy validation can mislead:* During contest RL, greedy validation on 56 held-out
    problems looped heavily (30–80% truncation mid-run) and failed to predict the
    temperature-0.6 LiveCodeBench benchmark, where the later checkpoint was strongest.
- **What did not make it into v0.0.1 (negative results and side branches):**
  - *Looped (latent-recurrent) conversion did not pay off.* On the earlier 720M KDA
    hybrid, we tested retrofitting the middle blocks into a shared recurrent group
    (`4 / 8 / 4` blocks, up to 4 loops, Elastic Latent Recurrence style) over 1B tokens
    against a plain continuation control (1.45× training FLOPs/token). The ungated
    conversion trailed the control at 1 loop (+0.004–0.007 nats) and degraded at 4 loops;
    adding a closed loop-entry gate removed the 1-loop penalty (+0.0002–0.0004 nats),
    but 2–4 loops still improved held-out loss by only ~0.001 nats on code and ~0.004
    nats on thinking traces and books, with the gate barely opening. Looping was not
    adopted for Malama.
  - *Task-arithmetic weight merging failed.* Adding the 20B code mid-training delta onto
    Reasoning SFT v3 (`v3 + λ·(code-20B − Base)`) broke the model completely: held-out
    loss exploded to 3.09–4.24 at `λ = 0.5` and ~13.7 at `λ = 1.0` (worse than uniform
    over the 154,880-token vocabulary, `ln 154,880 = 11.95`), because the two branches
    had diverged too far.
  - *Fill-in-the-middle (FIM) autocomplete* lives on a separate ~15B-token mid-training
    branch from the 84.8B base checkpoint. It lifts HumanEval-Infilling single-line from
    2.9 to 57.1 (multi-line 0.9 → 19.8), though it still trails dedicated code models.
  - *On-policy distillation from GLM-4.7-Flash* (scoring student rollouts with top-20
    teacher log-probabilities) and *medium/hard-only contest RL* (16K answers) were tested
    in pilots and are not part of v0.0.1.

## Results

All scores are pass@1 in percent unless the column says otherwise.

### Chat with thinking

LiveCodeBench v6 (175 problems from January to April 2025), HumanEval (164) and MBPP (500
test problems), each given as a chat turn with thinking: 16 samples per problem at
temperature 0.6 and top-p 0.95, up to 32,768 new tokens. A problem's score is the share of
its samples whose final Python block passes.

| Model | Parameters | LiveCodeBench v6 | LiveCodeBench pass@6 | HumanEval | MBPP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 0.6B | 15.2 | 23.4 | 56.4 | 42.6 |
| Malama before mixed RL (stage 5) | 0.93B | 18.9 | 27.6 | 10.6 | 2.6 |
| **Malama v0.0.1** | 0.93B | **19.1** | **26.6** | **50.1** | **41.6** |
| MobileLLM-R1.5-950M | 0.95B | 29.0 | 39.4 | 72.4 | 43.9 |
| Qwen3-1.7B | 1.7B | 31.0 | 38.5 | 85.9 | 69.0 |

![Chat benchmarks](charts/chat-benchmarks.png)

By difficulty on LiveCodeBench v6, v0.0.1 solves **59.9%** of easy, **11.1%** of medium
and **2.5%** of hard problems at pass@1. The gap to MobileLLM-R1.5-950M is concentrated on
medium and hard problems (25.1% and 4.3%).

### Progress by RL step

Three rounds of reasoning SFT were each followed by contest RL. v4 used about three times
the tokens of v3; v5 used about fourteen times the teacher traces of v4 (31,568 against
2,234). The orange line is the mixed RL stage that ends in v0.0.1.

![LiveCodeBench by RL step](charts/livecodebench-by-rl-step.png)

| Checkpoint | LiveCodeBench pass@1 | pass@6 |
| --- | ---: | ---: |
| SFT v3 | 8.4 | 14.8 |
| SFT v3 + 400 RL steps | 14.1 | 21.3 |
| SFT v4 | 14.0 | 26.6 |
| SFT v4 + 400 RL steps | 18.1 | 25.0 |
| SFT v5 | 12.8 | 23.3 |
| SFT v5 + 250 RL steps | 18.9 | 27.6 |
| SFT v5 + 500 RL steps | 17.0 | 24.5 |
| SFT v5 + 250 RL steps + 400 mixed RL steps (v0.0.1) | 19.1 | 26.6 |

### Base model: perplexity on unseen text

Bits per byte on 1,000 arXiv abstracts submitted on 2026-11-24, after every model here was
trained, and on the WikiText-2 test set. Lower is better.

| Base model | Parameters | arXiv abstracts | WikiText-2 |
| --- | ---: | ---: | ---: |
| Qwen3-4B-Base | 4B | 0.759 | 0.631 |
| Qwen3-1.7B-Base | 1.7B | 0.807 | 0.680 |
| Gemma 4 E2B | 5.1B | 0.838 | 0.632 |
| OLMo-2-1B | 1B | 0.874 | 0.660 |
| Qwen3-0.6B-Base | 0.6B | 0.882 | 0.758 |
| MobileLLM-R1-950M-base | 0.95B | 0.893 | 0.755 |
| Qwen3.5-0.8B-Base | 0.8B | 0.896 | 0.767 |
| LFM2.5-1.2B-Base | 1.2B | 0.923 | 0.717 |
| **Malama Base** | 0.93B | **0.949** | **0.833** |
| SmolLM2-360M | 0.36B | 0.952 | 0.743 |
| Gemma 3 1B | 1B | 0.954 | 0.698 |
| Gemma 3 270M | 0.27B | 1.071 | 0.822 |

![Perplexity by size](charts/perplexity-by-size.png)

Malama Base is at the weak end of its size class on raw text: level with SmolLM2-360M and
Gemma 3 1B, and about 7% behind Qwen3-0.6B-Base. That gap is likely because it is
undertrained at 100B pretraining tokens relative to modern small models trained on
trillions of tokens. The other models have very likely trained on Wikipedia, so the arXiv
column is the fairer comparison.

### Code variants: plain completion and infilling

These are the base and mid-trained variants, without chat formatting: HumanEval zero-shot
and MBPP 3-shot with greedy decoding, and HumanEval-Infilling in each model's own
fill-in-the-middle format.

| Model | Parameters | HumanEval | MBPP | Infilling, single line | Infilling, multi-line |
| --- | ---: | ---: | ---: | ---: | ---: |
| Malama Base | 0.93B | 12.2 | 12.6 | 2.9 | 0.9 |
| Malama code mid-training, 5B | 0.93B | 50.0 | 33.2 | 3.5 | 6.8 |
| Malama code mid-training, 20B | 0.93B | 42.7 | 42.4 | 3.2 | 2.5 |
| Malama FIM mid-training | 0.93B | – | – | 57.1 | 19.8 |
| MobileLLM-R1-950M-base | 0.95B | 48.8 | 38.8 | – | – |
| Qwen2.5-Coder-0.5B | 0.49B | 27.4 | 40.6 | 76.7 | 37.9 |
| DeepSeek-Coder-1.3B | 1.35B | 29.3 | 44.2 | 80.4 | 53.7 |
| Qwen2.5-Coder-1.5B | 1.54B | 40.9 | 58.6 | 84.7 | 37.2 |
| SantaCoder | 1.1B | – | – | 69.2 | 35.6 |

A dash means not measured; MobileLLM-R1 has no fill-in-the-middle tokens.

![Plain completion, HumanEval](charts/completion-humaneval.png)
![Plain completion, MBPP](charts/completion-mbpp.png)
![Infilling, single line](charts/infilling-single-line.png)

## Transparency: teacher models and synthetic data

Because Malama Base only saw 100B pretraining tokens, a large share of Malama v0.0.1's
Python and reasoning ability comes from **post-training on teacher-generated traces and
synthetic code datasets**, plus outcome-based RL. Specifically:

- **Code mid-training (5B)** lifted plain-completion HumanEval from 12.2 to 50.0 using
  synthetic Python Q&A, snippets and unit-tested synthetic answers (`opc-annealing-corpus`
  and `OpenCodeInstruct`) alongside GitHub code and pretraining replay.
- **Reasoning SFT (v3 → v5)** taught the `<think>` reasoning format by distilling verified
  passing traces from our own teacher runs (**Qwen3-30B-A3B** and **DeepSeek V4 Flash**)
  together with public **DeepSeek-R1** code-reasoning datasets.
- **RL problem filtering** also used Qwen3-30B-A3B: a contest problem entered our RL pools
  only if the teacher produced a passing solution for it.

**Teacher traces generated for this project**

| Teacher | How it ran | What it produced | Kept | Used in |
| --- | --- | --- | --- | --- |
| Qwen3-30B-A3B (`Qwen/Qwen3-30B-A3B-FP8`) | vLLM on Modal, thinking mode, 4 samples per problem at temperature 0.6, up to 16,384 tokens | Reasoning and a final program for LiveCodeBench v1 to v5 and LeetCodeDataset problems before 2025 | Only completions whose program passes the tests: the shortest per problem (traces v1), then up to the four shortest (traces v2) | Reasoning SFT v3 and v4 (v1), v5 (v2). Its passing programs also decide which problems enter the RL pools |
| DeepSeek V4 Flash (`deepseek/deepseek-v4-flash`) | OpenRouter | Reasoning and a final program for DeepMind CodeContests problems | Up to the four shortest passing completions per problem | Reasoning SFT v5 |

Traces v1 hold 2,234 traces and traces v2 hold 31,568. They are stored in the private
dataset `jfrick/lcb-teacher-traces`.

**Teacher used only in an experiment**

GLM-4.7-Flash (`zai-org/GLM-4.7-Flash`) scored answers the student had sampled itself
(top-20 next tokens and their log-probabilities at every answer token) for on-policy
distillation. Only learning-rate pilots ran, and no released checkpoint was trained on
them.

**Third-party datasets that are themselves model-generated**

| Dataset | Generated by | Used in |
| --- | --- | --- |
| `nvidia/OpenCodeReasoning-2` | DeepSeek-R1 | Reasoning SFT |
| `nvidia/AceReason-1.1-SFT` | DeepSeek-R1 | Reasoning SFT |
| `open-r1/codeforces-cots` | DeepSeek-R1 | Reasoning SFT |
| `OpenCoder-LLM/opc-annealing-corpus` (question-answer pairs) | Synthetic; see the dataset's card | Code mid-training |
| `nvidia/OpenCodeInstruct` | Synthetic instructions, answers and unit tests; see the dataset's card | Code mid-training |
| `KodCode/KodCode-Light-RL-10K` | Synthetic problems and tests; see the dataset's card | Mixed RL |
| `jfrick/mit-claude-codex-code-20261019` | MIT-licensed repositories whose commits are attributed to AI coding tools | Pretraining (0.2%) |

No model-generated text was used as an evaluation reference: every benchmark score comes
from running the model's program against the benchmark's tests.

## Training data

Training documents sharing a 13-gram with the test sets of GSM8K, HumanEval, MBPP, MMLU,
ARC, HellaSwag, MATH and LiveCodeBench were removed from pretraining and mid-training data.
The reasoning and RL data are screened against LiveCodeBench release v6, the release used
for evaluation.

| Stage | Sources |
| --- | --- |
| Pretraining (100B) | 50% general web text (`karpathy/climbmix-400b-shuffle`); 10% math (`nvidia/Nemotron-CC-Math-v1`); 35.8% permissively licensed GitHub code (`codeparrot/github-code-clean`: 20% non-Python and 4% Python from repositories with at least 10 watchers, 8.3% and 3.5% from other repositories); 4% curated open-source projects (`jfrick/curated-oss-code-20261019`); 0.2% MIT-licensed code attributed to AI coding tools (`jfrick/mit-claude-codex-code-20261019`) |
| 32K context (2B) | 50% pretraining replay; of the rest, 45% GitHub code grouped by repository, 35% GitHub files not used in pretraining, 20% books (`emozilla/pg19`) |
| Code mid-training, 5B | 45% replay; of the rest, 31% synthetic Python question-answer pairs and 27% snippets (`OpenCoder-LLM/opc-annealing-corpus`), 27% instruction answers that pass all of their generated unit tests (`nvidia/OpenCodeInstruct`), 15% repository-level GitHub code |
| Code mid-training, 20B | 35% replay; of the rest, 30% algorithmic files and 30% code-related web pages (`OpenCoder-LLM/opc-annealing-corpus`, `OpenCoder-LLM/opc-fineweb-code-corpus`), 10% question-answer pairs, 10% snippets, 7% OpenCodeInstruct, 13% GitHub code |
| FIM mid-training | The pretraining data, with half of all code documents rearranged as prefix, suffix, middle |
| LCB format SFT | 25% replay; shortest passing Python solutions from `deepmind/code_contests` and reference solutions from `newfacade/LeetCodeDataset` (problems before 2025) |
| Reasoning SFT pilot | 10% replay; `nvidia/OpenCodeReasoning-2` Python traces; answer-only CodeContests and LeetCode programs |
| Reasoning SFT v3, v4, v5 | 10% replay; reasoning traces from `nvidia/OpenCodeReasoning-2`, `nvidia/AceReason-1.1-SFT` (LeetCode rows) and `open-r1/codeforces-cots`, each kept only when its final program passes the tests; our own teacher traces (above); answer-only CodeContests and LeetCode programs |
| Contest RL | Pool `rl-pool-contest-02`: 1,843 easy and medium problems with tests from LiveCodeBench releases v1 to v5 and LeetCodeDataset (before 2025), 56 held out |
| Mixed RL | Pool `rl-pool-contest-functions-02`: the 1,843 contest problems, 879 function problems from `KodCode/KodCode-Light-RL-10K`, and 374 MBPP train and validation problems stated with their test assertions; 167 held out |

MBPP's test split, HumanEval and LiveCodeBench v6 were never trained on. MBPP's train and
validation splits are in the mixed RL pool, so MBPP is a closer-to-training benchmark than
HumanEval for v0.0.1.

## Variants

Every variant is the same architecture. They are separate checkpoints on different
branches, not one model that does everything.

| Variant | Parent | Training | Use | Repository |
| --- | --- | --- | --- | --- |
| Malama Base | none | 100B tokens, 4K context | Continues text | [`jfrick/malama-base`](https://huggingface.co/jfrick/malama-base) |
| 32K context | Base | 2B tokens at 32K, half replay | Continues long text | [`jfrick/malama-context32k`](https://huggingface.co/jfrick/malama-context32k) |
| Code mid-training, 5B | 32K context | 5B tokens at 32K, 45% replay | Completes code; parent of every reasoning model | [`jfrick/malama-code-anneal`](https://huggingface.co/jfrick/malama-code-anneal) |
| Code mid-training, 20B | Base | 20B tokens at 4K, 35% replay | Completes code; best MBPP of the plain models | `jfrick/malama-code-anneal-20b` |
| FIM mid-training | Base at 84.8B tokens | About 15B tokens, half of code documents rearranged for fill-in-the-middle | Autocomplete (infilling) | `jfrick/malama-fim-midtrain` |
| LCB format SFT | Base | 20M tokens of answer-only contest programs | Pilot: answers contest problems without thinking | [`jfrick/malama-lcb-format-sft`](https://huggingface.co/jfrick/malama-lcb-format-sft) |
| Reasoning SFT pilot | 32K context | 537M tokens | First thinking model | [`jfrick/malama-reasoning-sft-pilot`](https://huggingface.co/jfrick/malama-reasoning-sft-pilot) |
| Reasoning SFT v3 | Code 5B | 1.42B tokens | Thinking model | [`jfrick/malama-reasoning-sft`](https://huggingface.co/jfrick/malama-reasoning-sft) |
| Reasoning SFT v4 | Code 5B | 4.30B tokens | Thinking model | `jfrick/malama-reasoning-sft-v4` |
| Reasoning SFT v5 | Code 5B | 4.89B tokens | Thinking model; parent of v0.0.1 | `jfrick/malama-reasoning-sft-v5` |
| Contest RL (from v3, v4, v5) | Reasoning SFT | Up to 650 steps | Thinking model for contest problems | Checkpoints not published |
| **Malama v0.0.1** | SFT v5 + 250 contest RL steps | 400 mixed RL steps | Thinking model for contest and function problems | this repository |

Repositories without a link are private.

## Limitations

- It is trained for Python programming problems. General chat, other languages and
  factual questions are weak.
- Medium and hard contest problems are mostly out of reach (11.1% and 2.5% at pass@1).
- Thinking can run to thousands of tokens for a simple request, and about 0.5% of answers
  never finish.
- The base model is weaker on raw text than other models of its size—likely because it is
  undertrained at 100B pretraining tokens—which likely limits everything built on it.
- Its coding and reasoning capabilities rely heavily on teacher-generated traces and
  synthetic datasets rather than pure pretraining scale.
- LiveCodeBench scores are from one evaluation run of 175 problems; differences of about a
  point between checkpoints are within noise.

## Files

| Path | Contents |
| --- | --- |
| `tokens-000288341556/` | Released weights: `model.safetensors`, `config.json`, `progress.json` |
| `results.json` | Every number in this card, by protocol |
| `charts/` | The charts in this card, drawn from `results.json` by `training/publication/malama_charts.py` |
| `logs/` | `metrics.jsonl`, `result.json` and `manifest.json` of the mixed RL run |

Per-problem completions and scores of every evaluation, the RL pools and the logs of the
earlier RL runs are in the dataset
[`jfrick/lm-exploration-results`](https://huggingface.co/datasets/jfrick/lm-exploration-results)
under `pr-197/`.

## Usage

The weights load with the model code in
[koogle/lm-exploration](https://github.com/koogle/lm-exploration) (`training/hybrid/`), not
with `transformers`. Decoding needs a CUDA GPU. Prompts use the GLM chat template with the
reasoning span opened; sampling at temperature 0.6 and top-p 0.95 matches the evaluation.

The [Malama versions Space](https://huggingface.co/spaces/jfrick/malama-versions) has a
chat with this model and the earlier variants.
