# Malama v0.0.1

Malama v0.0.1 is the first release of the 960M KDA hybrid line: the reasoning checkpoint at
the end of the mixed RL run `kda-960m-contest-functions-rl-20261125-01`. Its model card,
weights, charts and `results.json` are in
[`jfrick/malama-v0.0.1`](https://huggingface.co/jfrick/malama-v0.0.1); per-problem
completions, RL pools and run logs are in the dataset `jfrick/lm-exploration-results` under
`pr-197/`. This page records the lineage, the data of each stage and where other models
produced training data. Earlier findings are in
[CODING_REASONER_1B_FINDINGS.md](CODING_REASONER_1B_FINDINGS.md).

## How it was trained

The released checkpoint is the end of this line. Each stage starts from the weights of the
stage before it.

| Stage | Tokens or steps | What it adds |
| --- | --- | --- |
| 1. Pretraining ([Malama Base](https://huggingface.co/jfrick/malama-base)) | 100B tokens at 4K context | Web text, code and math |
| 2. Context extension | 2B tokens at 32K | Long documents: repositories and books |
| 3. Code mid-training | 5B tokens at 32K | Python exercises, snippets, unit-tested answers |
| 4. Reasoning SFT v5 | 4.89B tokens at 32K | Chat format with thinking, from reasoning traces on contest problems |
| 5. Contest RL | 250 steps | Answers are run against tests; the model is rewarded when its program passes |
| 6. Mixed RL | 400 steps | The same, on contest problems and "write a function" problems together |

RL is Dr. GRPO (the group-mean baseline of GRPO without dividing by the standard deviation)
with Muon, 64 problems and 8 sampled answers each per step, answers of up to 8,192 tokens,
on 8 H100s. An answer scores 1 when its program passes every test in a sandbox, 0 when it
fails and −0.5 when it does not finish.

Workflows, in order: [`kda-960m-foundation`](../workflows/kda-960m-foundation.json),
[`kda-960m-reasoning-pilot`](../workflows/kda-960m-reasoning-pilot.json) (its context
extension stage), [`kda-960m-code-anneal`](../workflows/kda-960m-code-anneal.json),
[`kda-960m-reasoning-sft-v5`](../workflows/kda-960m-reasoning-sft-v5.json),
[`kda-960m-contest-rl-v5`](../workflows/kda-960m-contest-rl-v5.json) (to step 250) and
[`kda-960m-contest-functions-rl`](../workflows/kda-960m-contest-functions-rl.json).

## Results

Chat with thinking, 16 samples per problem at temperature 0.6, pass@1 in percent
(LiveCodeBench v6, HumanEval, MBPP test):

| Model | LiveCodeBench v6 | pass@6 | HumanEval | MBPP |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 15.2 | 23.4 | 56.4 | 42.6 |
| SFT v5 + 250 contest RL steps | 18.9 | 27.6 | 10.6 | 2.6 |
| + 200 mixed RL steps | 18.2 | 26.8 | 48.4 | 41.1 |
| + 250 mixed RL steps | 17.5 | 25.9 | | |
| + 400 mixed RL steps (v0.0.1) | 19.1 | 26.6 | 50.1 | 41.6 |
| MobileLLM-R1.5-950M | 29.0 | 39.4 | 72.4 | 43.9 |
| Qwen3-1.7B | 31.0 | 38.5 | 85.9 | 69.0 |

- Contest-only RL left the model answering function problems with programs that read
  standard input (67% of its HumanEval answers). Adding function problems to the pool
  fixed that without losing LiveCodeBench.
- An earlier mixed pool (`rl-pool-contest-functions-01`) held MBPP problems without their
  assertions; held-out MBPP stayed at 0% and the run was stopped after 45 steps. Pool
  `-02` states each MBPP problem with its assertions, as the benchmark does.
- Held-out validation during the mixed run was noisy (MBPP between 3% and 42%, with up to
  71% of validation answers unfinished at step 250) while the benchmark scores of the same
  checkpoints were stable. The cause was not investigated.
- Contest-only RL from v5 peaked near step 250 (18.9) and fell to 17.0 by step 500.

Base model perplexity (`model_hybrid.modal_benchmarks --suite perplexity`): Malama Base
scores 0.949 bits per byte on 1,000 arXiv abstracts of 2026-11-24, level with SmolLM2-360M
(0.952) and Gemma 3 1B (0.954) and behind Qwen3-0.6B-Base (0.882) and
MobileLLM-R1-950M-base (0.893). On WikiText-2 it scores 0.833, last of twelve models.

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
| **Malama v0.0.1** | SFT v5 + 250 contest RL steps | 400 mixed RL steps | Thinking model for contest and function problems | [`jfrick/malama-v0.0.1`](https://huggingface.co/jfrick/malama-v0.0.1) |

Repositories without a link are private.

Two further experiments are not part of any released checkpoint: contest RL on medium and
hard problems only, and on-policy distillation from GLM-4.7-Flash, of which only
learning-rate pilots ran.

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
| Reasoning SFT v3, v4, v5 | 10% replay; reasoning traces from `nvidia/OpenCodeReasoning-2`, `nvidia/AceReason-1.1-SFT` (LeetCode rows) and `open-r1/codeforces-cots`, each kept only when its final program passes the tests; our own teacher traces (below); answer-only CodeContests and LeetCode programs |
| Contest RL | Pool `rl-pool-contest-02`: 1,843 easy and medium problems with tests from LiveCodeBench releases v1 to v5 and LeetCodeDataset (before 2025), 56 held out |
| Mixed RL | Pool `rl-pool-contest-functions-02`: the 1,843 contest problems, 879 function problems from `KodCode/KodCode-Light-RL-10K`, and 374 MBPP train and validation problems stated with their test assertions; 167 held out |

MBPP's test split, HumanEval and LiveCodeBench v6 were never trained on. MBPP's train and
validation splits are in the mixed RL pool, so MBPP is a closer-to-training benchmark than
HumanEval for v0.0.1.

## Where other models produced training data

**Teacher traces generated for this project**

| Teacher | How it ran | What it produced | Kept | Used in |
| --- | --- | --- | --- | --- |
| Qwen3-30B-A3B (`Qwen/Qwen3-30B-A3B-FP8`) | vLLM on Modal, thinking mode, 4 samples per problem at temperature 0.6, up to 16,384 tokens | Reasoning and a final program for LiveCodeBench v1 to v5 and LeetCodeDataset problems before 2025 | Only completions whose program passes the tests: the shortest per problem (traces v1), then up to the four shortest (traces v2) | Reasoning SFT v3 and v4 (v1), v5 (v2). Its passing programs also decide which problems enter the RL pools |
| DeepSeek V4 Flash (`deepseek/deepseek-v4-flash`) | OpenRouter | Reasoning and a final program for DeepMind CodeContests problems | Up to the four shortest passing completions per problem | Reasoning SFT v5 |

Traces v1 hold 2,234 traces and traces v2 hold 31,568. They are in the private dataset
`jfrick/lcb-teacher-traces`.

**Teacher used only in an experiment**

GLM-4.7-Flash (`zai-org/GLM-4.7-Flash`) scored answers the student had sampled itself: its
top-20 next tokens and their log-probabilities at every answer token, for on-policy
distillation. Only pilots ran, and no released checkpoint was trained on them.

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
